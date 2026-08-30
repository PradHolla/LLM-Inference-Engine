# vLLM runbook

How to install, launch, ablate and measure vLLM on the project box. Written after doing
it once, so the traps below are the ones actually hit, not the ones anticipated.

Versions this was written against: vLLM 0.27.1, torch 2.13.0, transformers 5.15.1,
CUDA 13.2, NVIDIA A10G (g5.2xlarge), driver 595.91.07.

---

## 1. Install

**vLLM gets its OWN virtualenv.** Do not install it into `/opt/llm/.venv`.

```bash
uv venv /opt/llm/.venv-vllm --python 3.12
VIRTUAL_ENV=/opt/llm/.venv-vllm uv pip install vllm
```

Why separate: `/opt/llm/.venv` runs `engine/` and `baseline/`, and vLLM pins its own
torch build. Upgrading in place would break both, and the entire point of Phase 3 is
comparing vLLM *against* them. Losing the ability to re-measure our own engine would
make every comparison unreproducible.

Cost: about 7 GB of disk and roughly one minute with `uv`. The box has ~100 GB free.

**Set the hold file before starting.** A multi-gigabyte download with the GPU idle looks
exactly like idleness to `infra/idle-shutdown.sh`, which stops the box after 30 minutes
of GPU quiet.

```bash
touch /opt/llm/.no-autoshutdown     # remove it when the work is done
```

**Verify the engine venv survived**, rather than assuming isolation worked:

```bash
/opt/llm/.venv/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
# expect: 2.13.0+cu132 5.15.1
```

---

## 2. Launch

```bash
sudo systemd-run --unit=vllm --collect --working-directory=/opt/llm \
  --setenv=HF_HOME=/opt/llm/hf-cache \
  --setenv=HF_HUB_OFFLINE=1 \
  --setenv=PYTHONUNBUFFERED=1 \
  --setenv=PATH=/opt/llm/.venv-vllm/bin:/usr/local/bin:/usr/bin:/bin \
  /opt/llm/.venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B --max-model-len 4096 --host 0.0.0.0 --port 8000
```

Startup takes roughly 2-3 minutes: model load, `torch.compile`, then CUDA graph capture.
Poll for readiness rather than sleeping a fixed time:

```bash
until curl -sf -m 3 "http://$IP:8000/health" >/dev/null; do sleep 10; done
```

### Every argument above is load-bearing

**`--setenv=PATH=/opt/llm/.venv-vllm/bin:...`** — without it the engine dies during
warmup with `FileNotFoundError: 'ninja'`. FlashInfer JIT-compiles kernels at startup and
shells out to `ninja`, which lives in the venv's `bin`. `systemd-run` starts from a
minimal environment, so the venv's `bin` is not on PATH and the subprocess cannot find a
binary that is definitely installed. The error names the tool, not the cause.

**`--max-model-len 4096`** — Qwen3-8B advertises 40,960 tokens of context. At 144 KiB of
KV per token, one max-length sequence needs 5.6 GiB against roughly 3.6-4.6 GiB of KV
budget, so vLLM refuses to start. Any value comfortably above the workload works.

**`HF_HOME=/opt/llm/hf-cache`** — reuses the 16 GB model already on the root volume.
Detached processes never load a login shell, so this must be set inline; relying on
`/etc/profile.d` silently re-downloads the model to `~/.cache/huggingface`.

**`systemd-run`, never `nohup ... &` over ssh** — see `CLAUDE.md` section 3.

---

## 3. Read the configuration out of the log, never assume it

```bash
journalctl -u vllm --no-pager -o cat | grep -E "enable_prefix_caching|enable_chunked_prefill"
journalctl -u vllm --no-pager -o cat | grep "GPU KV cache size"
```

vLLM 0.27.1 defaults, confirmed from a running server rather than the docs:

| setting | default |
|---|---|
| `enable_prefix_caching` | **True** |
| `enable_chunked_prefill` | **True** |
| `max_num_batched_tokens` | 2048 |
| `max_num_seqs` | 128 |
| attention backend | FLASH_ATTN |

The first two matter enormously: neither `engine/` nor `baseline/` has them, so "vLLM
defaults vs our engine" is **not** a like-for-like comparison. Older vLLM releases had
both off, so documentation and blog posts written against those versions are misleading.

Defaults can also be read directly:

```bash
/opt/llm/.venv-vllm/bin/python -c "
import dataclasses
from vllm.config import SchedulerConfig
for f in dataclasses.fields(SchedulerConfig):
    print(f.name, f.default)"
```

### TRAP: the KV cache size varies between identical launches

Two consecutive starts of the exact same command produced **26,176** and then **33,424**
tokens of KV cache, a 28% difference. vLLM sizes its cache by profiling free GPU memory
at startup, so any transient allocation during that profile changes the budget.

**Record `GPU KV cache size` from each run's own log and report it beside the result.**
A capacity difference between two configurations is not attributable to the flag until
their KV budgets are confirmed comparable. Without this, an ablation will confidently
credit a flag for what was really a startup accident.

---

## 3b. Swapping servers: why the ladder is legitimate

All three engines speak OpenAI-streaming on port 8000, so comparing them is a matter of
stopping one systemd unit and starting another. `tools/bench.py` contains no
server-specific code at all -- it knows a URL and POSTs to `/v1/chat/completions`.

```bash
# Phase 1 baseline -- HF .generate() behind a global lock
sudo systemd-run --unit=llm-baseline --collect --working-directory=/opt/llm \
  --setenv=HF_HOME=/opt/llm/hf-cache --setenv=HF_HUB_OFFLINE=1 \
  /opt/llm/.venv/bin/python -m uvicorn baseline.server:app --host 0.0.0.0 --port 8000

# Phase 2 our engine -- continuous batching
sudo systemd-run --unit=llm-engine --collect --working-directory=/opt/llm \
  --setenv=HF_HOME=/opt/llm/hf-cache --setenv=HF_HUB_OFFLINE=1 \
  /opt/llm/.venv/bin/python -m engine.server --max-batch 8 --host 0.0.0.0 --port 8000

# Phase 3 vLLM -- see section 2 for why PATH and --max-model-len are required
sudo systemd-run --unit=vllm --collect --working-directory=/opt/llm \
  --setenv=HF_HOME=/opt/llm/hf-cache --setenv=HF_HUB_OFFLINE=1 \
  --setenv=PATH=/opt/llm/.venv-vllm/bin:/usr/local/bin:/usr/bin:/bin \
  /opt/llm/.venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B --max-model-len 4096 --host 0.0.0.0 --port 8000
```

| | venv | entrypoint | `--model` validated |
|---|---|---|---|
| baseline | `.venv` | `uvicorn baseline.server:app` | no, ignored |
| our engine | `.venv` | `python -m engine.server` | no, ignored |
| vLLM | **`.venv-vllm`** | `python -m vllm.entrypoints.openai.api_server` | **yes, 404s** |

Only one server runs at a time -- stop the others first, or the second fails to bind
port 8000. All three read the same `/opt/llm/hf-cache`, so none re-downloads the model.

**This is what makes `0.332 -> 1.60 -> 5.70 req/s` a real ladder.** Three unrelated
engines, one client, one protocol, one GPU, one prompt, nothing re-normalised between
them. Keeping `bench.py` wire-compatible was the most annoying constraint of Phase 2 --
inventing a cleaner protocol for `engine/server.py` would have been easier -- and it is
the only reason the comparison means anything now.

## 4. Measure

`tools/bench.py` works against vLLM unchanged. One flag differs from our own servers:

```bash
uv run tools/bench.py --url "http://$IP:8000" --model "Qwen/Qwen3-8B" \
  --sweep 3,4,5,6,7 --duration 60 --prompt-tokens 512 --max-tokens 64 --no-think \
  --unique-prefix --out results/phase3-something.jsonl
```

**`--model "Qwen/Qwen3-8B"` is required.** `bench.py` defaults to `--model test`, which
`baseline/server.py` and `engine/server.py` both ignore. vLLM validates it and returns
`404 The model 'test' does not exist`, which surfaces as every request failing and the
summary printing `nan` rather than as an obvious error.

**`--unique-prefix` is required for an honest capacity number.** Without it every request
sends a byte-identical prompt, which is the perfect-hit case for a prefix cache. Measured
on this box: 18.6 req/s cached against 5.7 req/s uncached, a 3.3x difference that is
purely a benchmark artifact. Report the uncached number; report the cached one only with
its caveat.

Keep `--duration 60` to stay comparable with the Phase 1 and Phase 2 sweeps.

---

## 5. Run an ablation

`infra/vllm-ablate.sh` automates restart, health-poll, KV-size capture and sweep. The
pattern is one flag changed per run, with everything else held fixed:

```bash
./infra/vllm-ablate.sh "A-nocache" "--no-enable-prefix-caching" "--unique-prefix" "3,4,5,6,7"
```

### Pair each flag with a workload where it can act

Ablating prefix caching under `--unique-prefix` measures nothing, because there are no
shared prefixes to hit. That run is still worth doing as a **control**: if it moves, the
harness is not measuring what it claims and every other result is suspect.

| flag | workload that exercises it |
|---|---|
| `--no-enable-prefix-caching` | identical prompts (and unique, as a control) |
| `--no-enable-chunked-prefill` | unique prefixes |
| `--kv-cache-dtype fp8` | unique prefixes |
| `--max-num-batched-tokens` | unique prefixes |
| `--max-num-seqs` | unique prefixes |

---

## 5b. Speculative decoding (Phase 5)

Flags read off the installed build 2026-08-29, not from documentation. vLLM 0.27.1
accepts either a JSON blob (`--speculative-config`) or three convenience flags:

```bash
--spec-method {ngram, ngram_gpu, eagle, eagle3, draft_model, suffix, medusa,
               mlp_speculator, mtp, ...plus ~25 model-specific MTP variants}
--spec-model  MODEL      # the draft model or head; omit for ngram
--spec-tokens N          # tokens drafted per step (k)
```

Three rungs, all against the same target:

```bash
# rung 1 -- n-gram / prompt lookup. No draft model, no VRAM.
--spec-method ngram --spec-tokens 3

# rung 2 -- a real draft model, same family and tokenizer
--spec-method draft_model --spec-model Qwen/Qwen3-0.6B --spec-tokens 3

# rung 3 -- EAGLE3 head. Its config names Qwen/Qwen3-8B as verifier.
--spec-method eagle3 --spec-model RedHatAI/Qwen3-8B-speculator.eagle3 --spec-tokens 3
```

### Draft weights are paid for out of the KV cache

Both rungs 2 and 3 are resident in the same `--gpu-memory-utilization` budget, so they
come straight off the KV cache:

| | VRAM | cost to bf16 KV | cost to int4 KV |
|---|---|---|---|
| EAGLE3 head | 1.904 GiB | -53% | -14.5% |
| Qwen3-0.6B | 1.400 GiB | -39% | -11% |

The EAGLE3 head costs MORE than the small draft model despite being one layer, because it
carries the target's full 151,936-row input embedding. Record `GPU KV cache size` from
each run's own log; a spec configuration is not comparable to its control until both KV
budgets are known.

### TRAP: `rejection_sample_method=synthetic` fabricates acceptance

`SpeculativeConfig` accepts `rejection_sample_method: synthetic` plus either
`synthetic_acceptance_rates` (per-position list) or `synthetic_acceptance_length` (scalar
mean). It makes up acceptance instead of measuring it.

That is exactly what validates `tools/specmon.py` -- set a known curve, confirm the tool
reads it back -- and exactly what will silently ruin a real run left on it by accident.
**Assert `rejection_sample_method` is `standard` from the run's own startup log before
believing any acceptance number.**

```bash
# instrument validation only, never a measurement run
--speculative-config '{"method":"eagle3","model":"RedHatAI/Qwen3-8B-speculator.eagle3",
  "num_speculative_tokens":3,"rejection_sample_method":"synthetic",
  "synthetic_acceptance_rates":[0.8,0.5,0.2]}'
```

### Reading acceptance

```bash
uv run tools/specmon.py discover --url http://localhost:8000     # bind the counters
uv run tools/specmon.py wrap --url http://localhost:8000 --label s3-int4 \
  --out results/phase5-spec.jsonl -- uv run tools/bench.py ...
```

The counter names on vLLM 0.27.1, confirmed from a running server:

```
vllm:spec_decode_num_drafts_total{engine="0",model_name="..."}
vllm:spec_decode_num_draft_tokens_total{...}
vllm:spec_decode_num_accepted_tokens_total{...}
vllm:spec_decode_num_accepted_tokens_per_pos_total{...,position="0|1|2"}
```

Each also has a `_created` twin holding a unix timestamp, not a count. Summing those into
an acceptance total would produce a number around 1.8e9 rather than an error, so
`specmon` excludes any series ending `_created`, `_sum` or `_bucket`.

`specmon` discovers these by pattern rather than hardcoding them, and reports
acceptance **by draft position**. The scalar hides the shape: a=0.6 is consistent with
"every position accepts 60%" and with "position 1 accepts 95%, position 3 accepts 5%",
and those imply opposite choices of `--spec-tokens`.

---

## 6. Shut down

```bash
sudo systemctl stop vllm
rm -f /opt/llm/.no-autoshutdown     # re-arm the idle guardrail
./infra/down.sh                      # stop the box; never terminate
```

Leaving the hold file in place defeats the guardrail entirely and bills $1.21/hour
indefinitely.

---

## 7. Summary of traps, in the order they were hit

| symptom | cause |
|---|---|
| `FileNotFoundError: 'ninja'` during warmup | venv `bin` not on PATH under systemd |
| engine refuses to start, KV too small | `--max-model-len` defaulted to 40,960 |
| every request fails, metrics print `nan` | `bench.py` sent `--model test`; vLLM validates it |
| suspiciously high throughput | prefix caching hitting on an identical benchmark prompt |
| capacity differs between identical runs | vLLM's startup memory profile sized KV differently |

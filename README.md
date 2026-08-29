# llm-inference-engine

Learning inference engineering by building an LLM serving stack from the ground up,
and measuring every layer against predictions derived from first principles.

The end goal is a chat application with web search and selectable thinking effort,
served from a self-hosted Qwen3-8B. But the application is the demo. The engine is
the point.

## The method

**Predict, measure, explain the gap.**

Before anything runs, `tools/roofline.py` computes what the hardware *should* do from
two numbers: memory bandwidth and parameter count. That prediction gets written down.
Then `tools/bench.py` measures what it actually does. A prediction that matches
confirms the model; a prediction that misses by 3x is the only real signal available,
and it is worthless unless it was written down first.

`NOTES/predictions.md` is the running log of every prediction, its derivation, the
measured result, and the explanation for any gap.

## Status: Phases 0-4 of 7 complete

Qwen3-8B on a single A10G (AWS `g5.2xlarge`). Three engines, one benchmark harness, one
GPU, nothing renormalised between them:

| Serving stack | Capacity | vs previous |
|---|---|---|
| Phase 1, HuggingFace `.generate()` behind a lock | 0.332 req/s | -- |
| Phase 2, our own continuous-batching engine | 1.60 req/s | 4.8x |
| Phase 3, vLLM | 5.70 req/s | 3.6x |

**17.2x end to end.** The number that matters more is latency under load: at 0.5 req/s
the naive server's p95 time-to-first-token was 25,122 ms and our engine's was 336 ms.
**75x**, at a load the baseline could not survive at all.

### Phase 4: quantization is worth 2x more on one workload than another

The same technique, the same GPU, the same model, the same client. Only the request
shape changed:

| | 512 in / 64 out | 4096 in / 1024 out |
|---|---|---|
| fp8 weights | 1.07x | **2.04x** |
| int4 w4a16 | 1.16x | **2.35x** |

On small requests the KV cache sits ~29% full, so freeing memory buys nothing and only
the bandwidth term survives. On requests shaped like a real web-search turn the cache is
the binding constraint and the same flag is transformative. **Benchmarking quantization
on the workload you already had set up is how you get this wrong by a factor of two.**

Quality was measured against a noise floor, because a model does not agree with itself:
batch composition shifts bf16 accumulation order, so bf16 was run twice to establish
`d0` = 1.8% before any quantized number was believed. int4 costs 2.1 accuracy points
overall -- and that aggregate hides everything:

| what int4 breaks, of what bf16 got right | |
|---|---|
| 4-step arithmetic | 0% |
| 8-step | 3% |
| 16-step | 9% |
| **32-step** | **37%** |
| long-context retrieval, any depth | **0%** |

The damage is specific to multi-step reasoning and does not touch retrieval.

**And thinking protects against it.** On identical 32-step problems int4 breaks 1% of
what bf16 solved when allowed to reason first, and 37% when not. The project's own spec
predicted the opposite -- that long reasoning chains would compound small errors -- and
the correction is annotated in place rather than deleted.

## What is here

```
tools/roofline.py     predicts memory budget, KV capacity, decode roofline,
                      batch scaling, and prefill/TTFT from hardware specs alone
tools/bench.py        open-loop Poisson load generator; per-request TTFT/ITL/E2E
tools/curve.py        collapses a sweep into latency-vs-throughput curve points
tools/kvprobe.py      measures the KV cache off the GPU directly
tools/mkitems.py      generates the quality-eval item set; --selftest re-derives
                      every answer by parsing the problem text it emitted
tools/qualeval.py     paired quality harness: run, offline regrade, exact McNemar
tools/mock_server.py  dependency-free fake vLLM with real capacity, so the
                      benchmark harness can be validated without a GPU
baseline/server.py    Phase 1: HuggingFace .generate() behind a global lock
engine/               Phase 2: manual KV cache, static then continuous batching,
                      scheduler with admission control, OpenAI-streaming server
infra/                provisioning, cost guardrails, spot interruption handling
NOTES/predictions.md  the prediction log
```

## Quick start

```bash
uv sync

# What should an 8B model do on an A10G? No GPU required.
uv run tools/roofline.py --model qwen3-8b --gpu a10g
uv run tools/roofline.py --model qwen3-8b --gpu a10g --dtype awq4 --context 8192

# Validate the benchmark harness against a fake server, also no GPU required.
uv run tools/mock_server.py --port 8000 --batch 8 --itl-ms 25 &
uv run tools/bench.py --url http://localhost:8000 --sweep 2,4,8,12,16 --duration 20
```

`bench.py` fires requests at a fixed **rate**, never waiting for responses. Closed-loop
generators throttle offered load by the very latency they are measuring, so a queue can
never form and the reported p95 is fiction. The exception is `--serial`, which is
closed-loop on purpose: for single-stream latency a queue is contamination rather than
signal.

## Provisioning

```bash
./infra/launch.sh     # g5.2xlarge, quota pre-flight, SSM-resolved AMI, IP-locked SG
./infra/up.sh         # start a stopped box, re-authorize your current IP
./infra/down.sh       # stop it (never terminate -- the root volume holds the model)
```

A forgotten GPU box is roughly $200/week, so `infra/idle-shutdown.sh` runs every minute
and stops the instance after 30 minutes of genuine idleness, or 180 with work in flight.
Idleness is judged by **request activity**, not by GPU memory: an inference server holds
its KV cache from startup until death at 0% utilization, so "a process holds GPU memory"
is true forever and would disable the guardrail entirely.

## Phases

| # | Phase | Status |
|---|---|---|
| 0 | Build the instruments before the thing they measure | done |
| 1 | Deliberately naive baseline, and explain its numbers | done |
| 2 | Write the engine: manual KV cache, continuous batching | done |
| 3 | vLLM as an object of study; ablate every flag | done |
| 4 | Quantization: throughput, capacity, and quality | done |
| 5 | Speculative decoding | next |
| 6 | The chat app and web search | |
| 7 | Thinking budget as a scheduling policy | |

Phase 2 hit its target (1.60 req/s at ITL p50 58 ms) and produced a negative result worth
as much as the positive ones: a paged block allocator was **designed, costed, and not
built**, because the 1.24x padding tax it was meant to recover was measured to live in
SDPA's masked-attention kernel path, where an allocator cannot reach it.

Phase 7 is the part that is not a reproduction of a blog post. Thinking tokens are
ordinary decode tokens: 1,500 of them is 45 seconds of silence, and they occupy KV
cache the entire time, so one thinking user costs several non-thinking ones. That makes
"thinking levels" a resource allocation policy rather than a UI toggle.

## Hardware

Single NVIDIA A10G. Sold as 24 GB; `nvidia-smi` reports 22.49 GiB; CUDA can actually
address **22.06 GiB** (the difference is driver/ECC reserve). ~600 GB/s memory bandwidth.
Every number in this repo is specific to that card, and `tools/roofline.py` will
recompute them for others.

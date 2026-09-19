# Phase 7 execution runbook

Every design decision and every prediction is already settled in `NOTES/predictions.md`
(section P7-RUNS). This file is the sequence. Nothing here requires a judgement call; if a
step needs one, stop and raise it rather than choosing.

## 0. Bring the box up

```bash
./infra/up.sh                      # us-east-1c has given InsufficientInstanceCapacity twice
./infra/sync.sh push
./infra/sync.sh run 'touch /opt/llm/.no-autoshutdown'   # remove it at the end
```

Launch the server once, with the flags Phase 7 depends on:

```bash
./infra/sync.sh run 'VLLM_USE_V2_MODEL_RUNNER=0 KV_PIN=10213733807 ./infra/vllm-launch.sh p7 \
  --model Qwen/Qwen3-8B --quantization fp8 --max-model-len 16384 --kv-cache-dtype fp8 \
  --enable-prefix-caching --reasoning-parser qwen3 \
  --reasoning-config "{\"reasoning_start_str\": \"<think>\", \"reasoning_end_str\": \"</think>\"}"'
```

`VLLM_USE_V2_MODEL_RUNNER=0` is mandatory. V2 is the 0.27.1 default and returns 400 on
`thinking_token_budget`. `phase7-runs.sh` aborts if the unit lacks it.

## 1. Run them in this order

Each goes in its own systemd unit. Never a foreground ssh session.

```bash
./infra/sync.sh run 'sudo systemd-run --unit=p7q4b --collect --working-directory=/opt/llm \
    /bin/bash /opt/llm/infra/phase7-runs.sh q4b'
```

Then the same with `q1b`, `q2`, `q3`, `q4`, changing `--unit=` to match.

| order | run | needs | rough cost |
|---|---|---|---|
| 1 | `q4b` | vLLM only; relaunches it twice itself | ~50 min |
| 2 | `q1b` | vLLM + gateway | ~45 min |
| 3 | `q2` | + `/opt/llm/.brave-key` | ~20 min |
| 4 | `q3` | + `/opt/llm/.brave-key` | ~20 min |
| 5 | `q4` | `vllm bench serve` holds load; disturbs everything, so it goes last | ~10 min |

Wait on each with a loop that BRANCHES on the unit's terminal state. `&& break` is not a gate
(incidents 30, 47): a loop's exit status is its last command's, not its condition's.

## 2. Check these before believing any arm

- **q1b first, before anything else:** did the queue ever exceed 8? Grep the gateway traces for
  `load_waiting`. If it never did, the adaptive arm never switched, it IS the big arm, and the
  run is void -- raise `RATE` and repeat. This is P7-Q1b-0 and it gates the whole run.
- **No trace may record a budget between 400 and 1200.** The policy cannot emit one; if one
  appears, a config env is wrong.
- **`usage_completion` p99 must not equal `max_tokens`.** That is the truncation trap that made
  Phase 6's accuracy numbers meaningless and killed two Phase 7 predictions. `qualeval` prints
  a warning for it; act on the warning.
- **Check `think_path`,** not just accuracy. `reasoning_field` means the parser is working;
  a run of `none` on a budgeted arm means the reasoning stream is being dropped again.

## 3. Analysis

```bash
./infra/sync.sh pull
./infra/sync.sh run 'HF_HOME=/opt/llm/hf-cache HF_HUB_OFFLINE=1 \
    /home/ubuntu/.local/bin/uv run tools/q1atokens.py results/p7q1b-*.jsonl'
uv run tools/q1acurve.py results/rec-p7q1b-*.jsonl
```

`q1atokens.py` recovers exact reasoning-token counts; subtract 3, which is its constant offset
against direct tokenisation, measured over 1,200 records.

Record actuals against the P7-RUNS predictions in `NOTES/predictions.md`, append-only.

## 4. Shut down

```bash
./infra/sync.sh run 'rm -f /opt/llm/.no-autoshutdown'
./infra/down.sh          # stop, never terminate: the root volume holds the HF cache
```

## Traps specific to this phase

| symptom | cause |
|---|---|
| every budget arm identical | `VLLM_USE_V2_MODEL_RUNNER` not 0, or the gateway not forwarding `thinking_token_budget` |
| `reasoning_chars` all 0 on a budgeted arm | client reading `reasoning_content`; vLLM 0.27.1 emits `reasoning` |
| accuracy looks like a truncation rate | `max_tokens` too low; check `usage_completion` p95 against the cap |
| adaptive indistinguishable from fixed | no queue formed; the policy never switched |
| gateway arm has no search | `/opt/llm/.brave-key` missing; q2 and q3 abort on this deliberately |

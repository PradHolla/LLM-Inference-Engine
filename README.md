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

## Status: Phase 1 of 7 complete

Qwen3-8B, bf16, on a single A10G (AWS `g5.2xlarge`).

| Quantity | Predicted | Measured | Gap |
|---|---|---|---|
| Weights in VRAM | 15.26 GiB | 15.26 GiB | 0.0% |
| Decode, batch 1 | 23.8 tok/s | 22.8 tok/s | -4.2% |
| TTFT @ 512-token prompt | 176 ms | 195 ms | +11.0% |
| Capacity | 0.338 req/s | 0.332 req/s | -1.8% |

The headline finding is not in that table. **Inter-token latency stayed flat at 44 ms
across a 5x range of offered load**, while time-to-first-token went from 354 ms to
7,293 ms. That is the fingerprint of a serialized server: per-token speed cannot
degrade when only one request is ever on the GPU, so every bit of contention becomes
queue wait instead.

Batch-1 decode is already within 4% of the memory-bandwidth roofline. There is nothing
left to win on single-stream speed. The entire remaining problem is capacity.

## What is here

```
tools/roofline.py     predicts memory budget, KV capacity, decode roofline,
                      batch scaling, and prefill/TTFT from hardware specs alone
tools/bench.py        open-loop Poisson load generator; per-request TTFT/ITL/E2E
tools/mock_server.py  dependency-free fake vLLM with real capacity, so the
                      benchmark harness can be validated without a GPU
baseline/server.py    Phase 1: HuggingFace .generate() behind a global lock
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
| 2 | Write the engine: manual KV cache, continuous batching, paged allocator | next |
| 3 | vLLM as an object of study; ablate every flag | |
| 4 | Quantization: throughput, capacity, and quality | |
| 5 | Speculative decoding | |
| 6 | The chat app and web search | |
| 7 | Thinking budget as a scheduling policy | |

Phase 2 target: **beat 0.33 req/s without pushing ITL past 55 ms.**

Phase 7 is the part that is not a reproduction of a blog post. Thinking tokens are
ordinary decode tokens: 1,500 of them is 45 seconds of silence, and they occupy KV
cache the entire time, so one thinking user costs several non-thinking ones. That makes
"thinking levels" a resource allocation policy rather than a UI toggle.

## Hardware

Single NVIDIA A10G, 22.49 GiB usable (sold as 24 GB), ~600 GB/s memory bandwidth.
Every number in this repo is specific to that card, and `tools/roofline.py` will
recompute them for others.

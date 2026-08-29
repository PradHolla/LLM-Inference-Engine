# LLM Inference Engine

An LLM serving stack built from scratch and measured against first-principles predictions
at every layer. Qwen3-8B on a single NVIDIA A10G (AWS `g5.2xlarge`).

**17.2x throughput and 75x tail latency over a naive baseline**, with every number
predicted in writing before it was measured and every gap explained.

| Serving stack | Capacity | p95 TTFT at 0.5 req/s |
|---|---|---|
| HuggingFace `.generate()` behind a lock | 0.332 req/s | 25,122 ms |
| **Custom continuous-batching engine** (~2,600 lines) | **1.60 req/s** | **336 ms** |
| vLLM | **5.70 req/s** | -- |

One benchmark harness, one GPU, one prompt, nothing renormalised between them. The 75x
is the number users would feel: at a load the naive server could not survive, the engine
answers in a third of a second.

## What this project demonstrates

- **Performance modelling** — weights-in-VRAM predicted to 0.0%, batch-1 decode to 4%,
  server capacity to 2%, from memory bandwidth and parameter count alone
- **Systems engineering** — manual KV cache management, continuous batching with
  admission control and mid-flight batch mutation, an OpenAI-compatible streaming server
- **Measurement design** — open-loop Poisson load generation, p50/p95/p99 rather than
  means, paired significance testing against a measured noise floor, and instruments
  validated against known ground truth before they are believed
- **Engineering judgement** — a paged block allocator designed, costed, and deliberately
  **not built**, because the cost it was meant to recover was measured to live somewhere
  an allocator cannot reach

Every claim above links to a number in `NOTES/predictions.md`, which records the
prediction, its arithmetic, the measurement, and the explanation for any gap. Roughly a
third of the predictions were wrong; those are the entries worth reading.

## Status: phases 0-4 of 7 complete

### How each step moved the number

**Phase 1** built a deliberately naive server and explained it rather than fixing it.
Inter-token latency stayed **flat at 44 ms across a 5x range of offered load** while
time-to-first-token went from 354 ms to 7,293 ms. That is the fingerprint of a serialized
server: per-token speed cannot degrade when only one request is ever on the GPU, so all
contention becomes queue wait. Batch-1 decode was already within 4% of the
memory-bandwidth roofline, which said the remaining problem was capacity, not speed.

**Phase 2** wrote the engine that fixes it: manual KV cache, static batching, then a
continuous-batching scheduler that admits and evicts requests mid-flight. Decode-slot
utilisation went from 30.1% to 82.6% on ragged workloads.

**Phase 3** swapped in vLLM and then ablated every flag to attribute the 3.6x gap.
Prefix caching was worth **0%** on unique traffic (2.9x on shared prompts, which is what
a careless benchmark measures); chunked prefill 2% of capacity but **34% better p95
ITL**; fp8 KV cache +11%. **3.5x remained attributable to kernels and core scheduling
rather than to any single flag** — which is the honest answer, and the one a flag-tuning
writeup never reaches.

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

This is a learning project, and deliberately structured as one: each phase ends with a
runnable artefact and a measurement, and no phase is allowed to close because the code
runs. It closes when a number is recorded and any gap from prediction is explained.

Phase 7 is the part that is not a reproduction of a blog post. Thinking tokens are
ordinary decode tokens: 1,500 of them is 45 seconds of silence, and they occupy KV
cache the entire time, so one thinking user costs several non-thinking ones. That makes
"thinking levels" a resource allocation policy rather than a UI toggle.

## Hardware

Single NVIDIA A10G. Sold as 24 GB; `nvidia-smi` reports 22.49 GiB; CUDA can actually
address **22.06 GiB** (the difference is driver/ECC reserve). ~600 GB/s memory bandwidth.
Every number in this repo is specific to that card, and `tools/roofline.py` will
recompute them for others.

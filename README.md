# LLM Inference Engine

> **AI usage:** I built this with Claude Code (Opus for design, analysis and debugging, Sonnet
> for mechanical implementation work), in this learning project.

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

## Status: phases 0-5 of 7 complete

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

### Phase 5: the same technique is worth 1.19x or 2.25x, and the benchmark hid it

Speculative decoding has a small helper model guess the next few tokens; the big model
checks them all in one pass and keeps the ones it agrees with. It works because decode is
memory-bound: producing one token reads all 16.39 GB of weights, and during that read
**the compute units sit 99.6% idle**. Guessing spends the idle half.

The measured speed-up is not one number. It is a range set entirely by request shape:

| workload | speed-up |
|---|---|
| arithmetic, no thinking block | **2.25x** |
| arithmetic with thinking | **2.05x** |
| natural-language reasoning (GSM8K) | **1.85x** |
| the synthetic benchmark prompt | 1.45x |
| long document in, short answer out | **1.19x** |

**The benchmark understated the technique by 40%.** Every acceptance figure was first
measured on `bench.py`'s synthetic filler, which is generated tokens rather than language
and is the least guessable input there is. Real work is roughly twice as guessable. The
last row is the opposite failure: that request is almost entirely prompt-reading, and
speculation only accelerates writing, so a genuinely good guess rate is diluted to nothing.
A retrieval-heavy product -- which is what Phase 6 builds -- would gain almost nothing from
it end to end while still paying the full memory bill.

**Thinking text is harder to guess than ordinary text**, 0.596 against 0.671 acceptance on
identical problems. The prediction written beforehand said the opposite, reasoning that
chain-of-thought is repetitive scaffolding. It is the reverse: thinking is where the model
explores and backtracks. That matters because ~1,000 thinking tokens is ~36 seconds of
blank screen, and it is the hardest part of the response to accelerate.

**The helper costs 1.78x its own weight**, and the excess was not weights at all. Predicted
from its config file to within 2% (1.904 GiB), it removed 24,720 tokens of KV cache rather
than the predicted 13,863: checking four positions at once needs ~4x the working memory, and
the GPU pre-compiles routines for the new shapes. Consequence: at full precision the helper
eats over half the cache and leaves **1.9 concurrent requests -- unusable**, while on fp8 it
costs 24% and leaves 8.5. **Quantization is what makes speculative decoding affordable**,
which is Phase 4's result arriving from the opposite direction.

**And it stops working under load.** Sweeping arrival rate on the small workload,
speculation is 1.88x ahead at 2 req/s and **2.8x behind at 6** -- the sign flips between 4
and 6. On the long-context workload it never flips. Two workloads, opposite answers about
whether to enable the feature.

Three ways to guess were compared, and the best guesser is not the fastest configuration:

| guesser | KV cost | tokens/step | speed-up |
|---|---|---|---|
| text repetition (no model) | **2,304** (3%) | 1.18 | 1.07x |
| EAGLE3 head | 16,368 (24%) | 2.47 | **1.85x** |
| Qwen3-0.6B | **36,448** (53%) | **3.17** | 1.75x |

The small model guesses 28% better and finishes 5% slower, because running 28 layers three
times per step costs more than running one layer three times -- and the *lighter* model
carries the *heavier* cache, since 28 layers of its own history dwarf the head's single one.

**And the setting nobody had tested was one too high.** Every measurement in this phase
used k=3 -- three tokens guessed per step -- because that is what the helper's checkpoint
ships with. Sweeping it:

| tokens guessed ahead | speed-up | tokens per step |
|---|---|---|
| 1 | 1.355x | 1.54 |
| **2** | **1.488x** | 1.83 |
| 3 (the default) | 1.450x | 1.91 |
| 5 | 1.338x | 2.00 |
| 7 | 1.215x | 2.03 |

Guessing further ahead keeps working -- k=7 really does emit 32% more tokens per step than
k=2. It just costs more than it returns, because acceptance decays geometrically with
position while the drafting cost grows linearly. **k=2 is 2.6% faster than the shipped
default and k=7 is 18% worse**, so every number above is a slight understatement.

The cost model predicted this shape from first principles and matched the measured
efficiency to within 4% at every k -- the most accurate prediction of the phase, and the
one built from measured quantities rather than expectations. The memory prediction attached
to it was flatly wrong: KV cache came back byte-identical at every k, because run-to-run
variance in vLLM's startup profile (14%) is far larger than the effect being resolved (~1%).

Quality was verified rather than assumed: 1,210 paired items, **89.3% -> 89.9% accuracy**,
answers differing on 2.8% against a 1.8% floor measured by running the model against itself.
Three whole categories reproduced identical answers on every item; all disagreement landed
in the one slice where the model scores 44% and is barely above guessing. That is the
signature of ordinary nondeterminism, not of a technique changing the answer.

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
tools/specmon.py      reads speculative-decoding acceptance off vLLM's metrics;
                      reports acceptance BY DRAFT POSITION, not just the scalar
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
| 5 | Speculative decoding | done |
| 6 | The chat app and web search | next |
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

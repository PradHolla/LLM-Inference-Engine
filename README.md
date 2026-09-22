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
  means, paired significance testing against a measured noise floor, instruments
  validated against known ground truth before they are believed, and controls that
  read the engine's own counters to prove they are controlling before collecting data
- **Engineering judgement** — a paged block allocator designed, costed, and deliberately
  **not built**, because the cost it was meant to recover was measured to live somewhere
  an allocator cannot reach

Every claim above links to a number in `NOTES/predictions.md`, which records the
prediction, its arithmetic, the measurement, and the explanation for any gap. Roughly a
third of the predictions were wrong; those are the entries worth reading.

## Status: phases 0-5 and 7 complete; 6a measured, 6b built and awaiting its end-to-end run

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
whether to enable the feature. Phase 6 measured *why*, and the cause named here is not
the right one.

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

### Phase 6: the crossover was never a property of speculation

Phase 5 watched speculation go 1.88x ahead at 2 req/s and 2.8x behind at 6, and read the
flip as verification overhead ceasing to hide at high batch. That is the textbook
explanation. It is not what is happening here.

Run as a matrix instead, quantization against speculation, identical real prompts in an
identical order, and the flip happens for one configuration and not the other:

| weights | spec | KV budget | batch-1 | cap @2 req/s | cap @6 req/s |
|---|---|---|---|---|---|
| bf16 | off | 33,312 | 29.5 tok/s | 1.72 req/s | 3.58 req/s |
| bf16 | on | **would not start** | -- | -- | -- |
| fp8 | off | 74,880 | 53.4 | 1.65 | 4.52 |
| fp8 | on | 51,232 | **97.9** | 2.37 | 3.56 |
| int4 | off | 101,920 | 84.6 | 1.65 | 4.73 |
| int4 | on | 77,280 | **143.1** | 1.96 | **5.30** |

*Qwen3-8B, vLLM 0.27.1, fp16 KV, 16k context, EAGLE3 k=2, 650 real arithmetic prompts.*

Speculation charges both configurations almost exactly the same KV: 23,648 tokens and
24,640. What differs is what is left afterwards. The engine had been logging the answer
every ten seconds the whole time -- **fp8+spec ran at 95.7% KV occupancy with 26 requests
queued, int4+spec at 18.6% with none**, at near-identical batch sizes of 90 and 88. Same
draft head, same batch, opposite outcome. That is queueing, not overhead.

Confirmed by intervention rather than left as a reading: halving KV bytes per token with
`--kv-cache-dtype fp8` and changing nothing else took fp8+spec from **86.1% occupancy and
26 queued to 12.0% and zero**, and its capacity at 6 req/s from 0.97x of control to
**1.20x**. The crossover disappears. **Quantization does not merely make speculation
affordable, it decides whether the crossover exists in the range you care about.**

**But fp8 KV is not a free win, and the control is what says so.** With speculation off,
where 74,880 tokens were already sufficient, it made capacity 14% *worse* (5.12 -> 4.42
req/s) while holding 135,152 tokens. It helps exactly where KV is the binding constraint
and hurts where it is not -- the same lesson as Phase 3's prefix caching and Phase 4's
quantization, arriving for the third time.

An accident said it louder than the designed comparison. The same command profiled 51,232
KV tokens on one day and 58,816 on another, pure startup variance, and capacity at 6 req/s
tracked it monotonically: 3.56, then 4.97, then 5.29 req/s at 102,464 tokens. **A 14.8%
difference in KV budget from noise alone moved capacity 40%** -- which is why every KV
allocation in this project is now pinned rather than profiled.

**Prefix caching is worth 13.8x, and the control had to prove itself first.** One
conversation grown to 25 turns against the same conversation with a fresh random prefix at
position zero each turn:

| turn | context | warm TTFT | cold TTFT | ratio |
|---|---|---|---|---|
| 1 | 62 | 28.4 ms | 32.4 ms | 1.1x |
| 10 | 2,947 | 128.5 ms | 820.2 ms | 6.4x |
| **25** | **7,761** | **165.7 ms** | **2,290.8 ms** | **13.8x** |

*fp8 weights, fp16 KV, 16k context, no speculation, 300 output tokens per turn.*

A cached token costs **29x** less than an uncached one (0.0100 against 0.2912 ms per prompt
token) but it does not cost nothing: caching removes recomputation, and each turn's new
tokens must still attend over the whole cached history. That residual is the warm slope.

The first two attempts at this measurement produced a null result that read as a finding.
The cold arm prefixed its noise to the new user message, where the conversation history
still renders first, so the cache hit exactly as in the warm arm and the two curves landed
1.0x apart against a predicted 21x. **The control now verifies itself and aborts**: it reads
the engine's own prefix-cache hit rate at turn 3 and refuses to collect data unless it is
near zero. It measured 0.000 at all 25 turns.

That still left the warm arm proving nothing -- it recorded no cache figures, so "the cache
was working" was an inference from its flat slope. Both arms were rerun on one server with
both reporting: **warm goes 0.000 to 0.957, cold holds 0.000, and the separation reproduced
at 13.76x against 13.83x.** The cold curve hit the same 2,290.8 ms at turn 25 both times.

**The rerun then showed what the first run could not.** Tokens recognised at turn N equal
turn N-1's *prompt*, floored to vLLM's 16-token block, on 24 of 24 turns -- so **the model's
own reply is never cached and is re-prefilled every turn**, ~300 tokens of work already done.
**That turned out to be a known upstream bug**, [Qwen3 issue 1826](https://github.com/QwenLM/Qwen3/issues/1826).
With thinking disabled, Qwen3's chat template writes an empty `<think></think>` block into
the prompt it asks the model to continue, but strips it when re-rendering that same turn as
history. Four tokens. So turn N's prompt is not a prefix of turn N-1's, the match stops
there, and the reply plus everything after it is recomputed.

The fix is the one the issue proposes -- apply the block to history too -- and it is applied
at the server with `--chat-template`. Rendering the prompts locally showed the stock template
breaking at **5 of 5** turn transitions and the patched one at **0 of 5**. On the box it is
worth **2.3x**: warm TTFT at turn 30 goes 188 ms to 81.7 ms, and the hit rate reaches 1.00.

Worth noting how long that took to find by measurement alone. The bug was already written up
with its fix; a web search would have replaced an afternoon. The rule since: **measure what
is specific to this setup, look up what belongs to a library.**

Two results fell out of data already collected. Dividing measured ITL into weight bytes per
token gives achieved bandwidth, and **it falls as quantization deepens** -- 79.8% of the
A10G's 600 GB/s at bf16, 72.2% at fp8, 57.4% at int4. That is the Marlin dequantization
penalty made visible on a card with no native fp8, and it is why fp8 is worth 1.81x rather
than the 2.0x that halving the weight bytes implies. Prefill shows the same tax
independently, at 14%.

And the noise floor moved. Phase 4 measured `d0` = 1.8% and Phase 5 tested against it
legitimately, on the same item mix. Re-running an identical configuration against itself on
a **harder** mix gives **7.8%**, concentrated exactly where the model is near its competence
limit (25.7% churn on the slice it half-solves, 0.0% where it scores 100% and 0.0% where it
scores 0%). **A noise floor is a property of the item mix, not of the model**, and carrying
one between experiments is invalid. That correction retired a claim this project had already
made.

The measurement surface itself is `labbench/`: a byte-faithful OpenAI-streaming proxy, a
radio-button backend switcher, live probes over vLLM's Prometheus endpoint and journal, and
a React UI that renders every stage of a request. Building it produced its own instrument
bug, in the family this project keeps meeting -- the metric binder summed Prometheus
`_created` series, which are unix timestamps of counter creation, into counter roles and
reported a **prefix cache hit rate of 0.9999998** computed from two clocks.

### Phase 6: the app battery, and what a user actually waits for

Nine measurements against a chat app with web search, on one GPU, all at one code version
with the KV budget pinned. The engine phases measured the engine; this measures the product.

**Where the wait goes.** Search 626 ms, fetching pages 741 ms, extracting text 39 ms --
**1.4 seconds before the GPU is touched**. Inference is then 81% of a 300-token turn but
only 46% of the time to the first word. Both are true, and which one is the headline depends
entirely on how long the answer is. That is this project's recurring lesson turned on its
own phase.

**A conversation is the workload prefix caching was built for.**

| turn | cache working | cache defeated | ratio |
|---|---|---|---|
| 10 | 51.7 ms | 830.8 ms | 16.1x |
| **30** | **81.7 ms** | **2,895.7 ms** | **35.4x** |

*fp8 weights, fp16 KV, 16,384 context, KV pinned to 69,264 tokens, template patched.*

**Reopening a cold chat costs 5.1 seconds**, because nothing is cached and the whole history
is recomputed. It is the worst latency the app can produce and no ordinary benchmark sees it,
having no yesterday. The residual also says prefill is **not linear**: predicted 4,436 ms
from a slope fitted over short prompts, measured 5,098. Attention is quadratic in sequence
length while the weight read is linear, so every cold-open estimate in this phase is a lower
bound rather than an estimate.

**Compressing the KV cache is the biggest single configuration win for this workload.** From
identical pinned memory, fp8 KV holds 138,528 tokens against 74,880, and on long retrieved
prompts cuts time-to-first-token from **1,462 ms to 64 ms at 1 req/s -- 23x**. Long prompts
are exactly where cache space is the binding constraint. Speculative decoding, by contrast,
generates **1.49x faster per token** and makes the first token *slower*, because the draft
model takes cache the prompts need.

**Two measurements came back as bugs rather than findings, which is the better outcome.**

Thinking kept in history versus stripped performed **identically** (74.3 ms both) once the
template was patched. The 2.4x difference measured beforehand was the formatting bug, not a
product tradeoff. `phase6-app-design.md` section 4c treated it as a decision to make; it was
a defect to fix.

Context overflow was supposed to separate two strategies: drop the oldest turns, or compress
them into a summary. The first run had both at a hit rate of 0.000 and within 5 ms of each
other -- because our summary embedded a count of dropped messages that ticked up every turn,
so its text changed on every request and broke the prefix just as thoroughly as dropping from
the front did.

**Summarize-and-restart is worth nothing unless the summary is byte-identical turn to turn.**
The fix is to advance the summarised boundary in blocks rather than one pair per turn, so it
holds for many turns at a stretch. Measured after that:

| strategy | median TTFT | hit rate | |
|---|---|---|---|
| sliding window | 2,951 ms | 0.000 | every turn is cold, forever |
| **summarize** | **63.6 ms** | **0.993** | one cold turn per block |

**20x amortised, 46x between the re-summarisation spikes.** And the spike is not overhead: at
each one the prompt is trimmed to ~2,924 tokens and re-read, which at the 0.2915 ms/token cold
rate measured three phases earlier predicts 852 ms against 817 measured. Sliding window is not
a cheaper option with a latency cost -- it is 20x worse with no compensating benefit.

Fixing it took four attempts, and the first three were caught by an offline gate that asserts
the summary stays stable across turns, not by the GPU.

**Against the targets:** TTFT p95 under 250 ms holds to **4 req/s**, double what it managed
before the template fix.

**And compressing the cache costs no accuracy.** 1,210 paired items against the fp16 control:
52.1% against 51.6%, McNemar p = 0.6084, answer churn 11.9% against a measured 7.8% floor.
So fp8 KV's 23x is safe on both axes. It also re-ran under the patched template and moved
accuracy 0.5 points -- less than re-running an *identical* configuration moved it -- which is
the evidence that the template fix changed cacheability and nothing else.

**Four measurements in this phase produced confident nulls** because the configuration never
entered the regime the effect lives in -- a sweep 4% past a threshold, a conversation that
never overflowed, arrival rates already past saturation. The prediction protocol asks what a
number will be and never asked whether the run reaches the regime where the number exists.
The battery now refuses to run a measurement whose configuration does not cross its own
threshold, and prints the arithmetic first:

```
M3   needs exceed   69,264   this config  148,944   2.15x  OK
M4   needs under    16,384   this config   15,064   0.92x  OK
M6   needs exceed   16,384   this config   25,760   1.57x  OK
```

### Phase 7: more thinking is not better, and the middle is the worst place to be

A thinking budget caps how many tokens the model may spend reasoning before it is forced to
stop and answer. The obvious expectation is a monotonic curve: more budget, more accuracy,
more latency. Measured on a maths slice, 180 items per pass, two runs pooled to n=360 per
arm against a +/-3 point noise floor:

| thinking budget | accuracy | silence before the answer |
|---|---|---|
| 0 | 86.4% | 0.2 s |
| **128** | **94.4%** | **3.9 s** |
| 256 | 92.2% | 7.4 s |
| 512 | 84.2% | 14.6 s |
| 1024 | 83.1% | 28.2 s |
| **2048** | **99.4%** | **44.7 s** |
| unbounded | 99.4% | 45.7 s |

*Qwen3-8B fp8, A10G, temp 0.6 / top_p 0.95 / top_k 20.*

**256, 512 and 1024 are each worse than 128 on both axes at once.** Not a tradeoff, simply
dominated. The mechanism is what generalises: among responses where the budget actually
bound, the fraction that had already reached an answer was 30% at 512, 70% at 1024, and 98%
at 2048 — and the ones that had *not* reached an answer scored **51%**. A budget near the
task's natural reasoning length is the worst available setting, because it is enough to
commit the model to a chain of thought and not enough to finish it.

The product consequence is direct. A conventional low / medium / high control puts "medium"
in that trough, so the chat app ships three levels — off, 128, and unbounded — with nothing
in between, and maps the OpenAI-standard `reasoning_effort: medium` *upward* rather than to
a middle value.

### Phase 7: overlapping the search with generation wins 6 seconds, for the wrong reason

Three orderings of "search the web, then answer": retrieve-then-generate,
generate-then-retrieve, and an overlap that starts generating while the search is still in
flight. Overlap beat retrieve-then-generate by **6.0 s** end to end, against a prediction of
600-1,100 ms.

Missing by 6x is the useful part. The win is not hidden latency. Overlap generated **55**
tokens at p50 against retrieve-then-generate's **236**, and at its own measured decode rate
of 46.6 ms/token those 181 tokens are worth **8.4 s** — more than the entire gap. The search
round trip actually available to hide behind was 753 ms. What the overlap does is splice a
cue into the context saying the results have arrived, and that cue interrupts the chain of
thought and makes the model answer. Accuracy was identical either way, 98.3% against 98.3%.

The technique works and the explanation written into the design document was wrong. One
prediction in the same run did land: the TTFT gap was derived as
prefill(3,000 tokens) x 0.2915 ms/token = **875 ms**, and measured **882 ms**.

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
tools/convo.py        stateful multi-turn conversation driver; measures prefix-cache
                      behaviour, which an open-loop load generator cannot see
tools/mock_server.py  dependency-free fake vLLM with real capacity, so the
                      benchmark harness can be validated without a GPU
baseline/server.py    Phase 1: HuggingFace .generate() behind a global lock
engine/               Phase 2: manual KV cache, static then continuous batching,
                      scheduler with admission control, OpenAI-streaming server
labbench/             Phase 6a: the instrument surface -- byte-faithful streaming
                      proxy, backend switcher, live Prometheus and journal probes,
                      React UI showing every stage of a request
gateway/              Phase 6a: the inference gateway -- prompt assembly, web search,
                      context overflow strategy, thinking budget, request tracing
app/                  Phase 6b: the chat app -- SQLite branching message tree, SSE
                      streaming, three-level thinking control, cream/dark themes
infra/                provisioning, cost guardrails, spot interruption handling,
                      one-command session lifecycle, pinned-KV vLLM launcher
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
| 6 | The chat app and web search | 6a measured; 6b built, not yet run on a GPU |
| 7 | Thinking budget as a scheduling policy | done |

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

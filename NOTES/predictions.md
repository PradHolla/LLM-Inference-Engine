# Predictions Log

Append-only. Never edit a past entry — add a correction below it.

Format:
```
## YYYY-MM-DD — <what you ran>
**Predicted:** <number>
**Derivation:** <the actual arithmetic, not a vibe>
**Actual:** <number>
**Gap:** <ratio> — <explanation, or "unexplained" until you find it>
```

---
## 2026-08-20 — Phase 0 baseline predictions: Qwen3-8B on A10G (g5.2xlarge)

Produced by `uv run tools/roofline.py`, before any hardware ran. KV cache left at
fp16 (vLLM's `--kv-cache-dtype auto` default) in all three rows.

| weights | KV room | max tokens in flight | concurrent @4k ctx | realistic tok/s |
|---|---|---|---|---|
| bf16 | 3.84 GiB | 27,982 | **6.8** | 23.8 |
| fp8 | 11.48 GiB | 83,592 | **20.4** | 47.6 |
| awq4 | 15.30 GiB | 111,397 | **27.2** | 95.1 |

**Derivation:** KV/token = 2 x 36 layers x 8 kv_heads x 128 head_dim x 2 B = 144 KiB.
Weights = 8.2e9 x dtype_bytes. Room = 22.35 GiB (A10G *reports* 22.35, not the
marketed 24) x 0.90 util - weights - 1.0 GiB overhead.
Decode floor = weight_bytes / 600 GB/s; realistic = 65% of peak bandwidth.

**Predicted, to be tested in Phase 1/3:**
- bf16 batch-1 decode: **~24 tok/s** (floor 27.3 ms/token)
- TTFT for a 512-token prompt: **~176 ms** (134 ms prefill @ 50% MFU + one decode step)
- bf16 dies of KV exhaustion above **batch 4** at 4k context

**Actual:** _(pending -- Phase 1)_

**What this already changes:** bf16 leaves only 3.84 GiB of KV cache -> under 7
concurrent requests at 4k context. That is not a serving configuration, it is a
demo. Quantization moves from "Phase 4 optimization" to a prerequisite for having
anything worth benchmarking. Prediction to check: AWQ-4bit should give ~8x the
concurrency, and that -- not the tok/s -- is the reason to do it.

**Unverified assumptions** (each one is a place the gap could come from):
mem_eff 0.65, compute_eff 0.50, 1.0 GiB overhead, and that A10G reports 22.35 GiB.
Check the last one with `nvidia-smi` on first boot; it is the easiest to confirm
and it moves every number in the table.

---
## 2026-08-21 — Phase 1 ACTUALS: Qwen3-8B bf16, HF `.generate()`, g5.2xlarge

Fills in the `Actual:` left pending in the 2026-08-20 entry above. Instance
i-07d8b10bdcf39a099, A10G, transformers 5.15.1, torch 2.13.0+cu132, thinking disabled.

### Scoreboard

| Quantity | Predicted | Measured | Gap |
|---|---|---|---|
| Weights in VRAM | 15.26 GiB | **15.26 GiB** | **0.0%** |
| Decode, batch 1 | 23.8 tok/s | **22.8 tok/s** (server) / 22.5 (client) | **-4.2%** |
| TTFT @ 512 prompt | 176 ms | **195 ms** (server prefill) | **+11%** |
| A10G VRAM reported | 22.35 GiB (AWS API) | **22.488 GiB** (nvidia-smi) | +0.6% |

The memory model is exact. The bandwidth model is good to ~4%, which means
`mem_eff = 0.65` was very close for this stack -- the true value implied by 22.8 tok/s
is **0.623**. The compute model for prefill is the weakest of the three (+11%),
consistent with `compute_eff = 0.50` being slightly optimistic for a single
un-batched 512-token prefill.

### The experiment I got wrong first, and what it cost

The first attempt used `--rate 0.25` open-loop and reported **TTFT p50 = 11,179 ms**
against a 176 ms prediction -- a 60x miss that looked like a catastrophic model failure.
It was not. Service time is ~3.0 s, so 0.25 req/s put utilisation near 0.8 and the
queue grew without bound. The server's own log separated it cleanly:

```
queue_wait_s  median 10.838     <- the entire "miss"
prefill_s     median  0.195     <- the actual TTFT
decode_s      median  2.766
decode_tok_s  median 22.775
```

**Lesson:** when one metric matches prediction closely (ITL was 44.8 ms vs ~42 ms
predicted) and another misses by 60x, suspect the EXPERIMENT, not the model. Open-loop
is correct for measuring capacity and wrong for measuring single-stream latency, where
a queue is contamination rather than signal. `bench.py --serial` now exists for this.

### CUDA warmup is a 47x outlier

First request after model load: **prefill 9.092 s**. Steady state: **0.195 s**.
Kernel autotuning and allocator warmup. Any benchmark that includes request #1 in its
statistics is reporting a number that will never occur again. `--warmup 1` is now the
default in `bench.py`.

### Client vs server: 148 ms is the network

Client TTFT 343 ms, server-side prefill 195 ms. The 148 ms delta is round-trip from a
laptop to us-east-1 and is NOT a property of the server. Two independent instruments
agreeing after that offset is accounted for is what makes the number trustworthy --
run the benchmark from the box itself when the absolute TTFT number matters.

### What this means for Phase 2

22.8 tok/s at batch 1 is close to the memory-bandwidth roofline, so there is almost
nothing to win at batch 1 -- the naive baseline is already near the physical limit for
a single stream. Everything from here is a CAPACITY problem: the global lock means
utilisation above ~0.3 req/s produces unbounded queueing, and the fix is batching, not
faster decode. That is exactly the gap Phase 2 exists to close.

**Unverified assumption now RESOLVED:** A10G reports 22.488 GiB (23028 MiB), not the
22.35 GiB the AWS API claimed. `tools/roofline.py` updated to the measured value.

---
## 2026-08-21 — Phase 1 sweep: capacity of a lock-serialized server

### The result

| offered rate | n | true rps | TTFT p50 | ITL p50 |
|---|---|---|---|---|
| 0.1 | 7 | 0.074 | 354 ms | 44.6 ms |
| 0.2 | 11 | 0.173 | 346 ms | 43.7 ms |
| 0.3 | 17 | 0.257 | 974 ms | 44.2 ms |
| 0.5 | 30 | **0.332** | 7,293 ms | 44.7 ms |

**Predicted capacity:** service time = 0.195 s prefill + 2.766 s decode = 2.961 s,
so max throughput = 1/2.961 = **0.338 req/s**.
**Measured at saturation: 0.332 req/s. Gap: 1.8%.**

**ITL is flat at ~44 ms across a 5x range of offered load.** That is the fingerprint of
a lock-serialized server: per-token speed cannot degrade because only one request is
ever on the GPU. All contention becomes queue wait instead. Compare this to Phase 3,
where vLLM's ITL should RISE with load as the batch grows -- that rise is the visible
cost of the capacity you bought.

### Two more instrument bugs, same class as before

Both produced plausible numbers rather than errors. Both found only by checking the
raw JSONL against theory.

**A: throughput divided by the nominal window, not elapsed time.** At saturation the
backlog drains long after the arrival window closes. The 60 s point at rate 0.5 took
90.4 s to actually finish, so 30 requests were reported as 0.50 req/s instead of the
true 0.332 -- a **50% overstatement, precisely in the overloaded regime where the
number matters most**. Fixed: divide by observed span.

**B: p95 and p99 reported from as few as 7 samples.** A p95 from 7 samples interpolates
between the 6th and 7th value -- it is the maximum wearing a percentile's name, and it
swings wildly between runs. It also poisoned the automatic knee detection, which took a
meaningless 4,289 ms as its "unloaded baseline". Fixed: percentiles now require
1/(1-p) samples (p95 -> 20, p99 -> 100) and return NaN, printed as `n/a`, otherwise.

Running total: **four** bugs in the measurement tools, zero of which raised an
exception. Confidence in a number should scale with how hard it was cross-checked,
not with how clean it looked.

---
## 2026-08-21 — Phase 1 CLOSED: statistically valid baseline

Re-run at 300s per point so percentiles rest on enough samples.

| offered | n | true rps | TTFT p50 | TTFT p95 | TTFT p99 | ITL p50 |
|---|---|---|---|---|---|---|
| 0.25 | 90 | 0.29 | 4,727 ms | 15,800 ms | n/a (needs 100) | 45 ms |
| 0.33 | 109 | 0.33 | 20,385 ms | 36,898 ms | 38,650 ms | 45 ms |

**Capacity: 0.33 req/s, n=109.** Matches the 0.338 req/s derived from measured
service time to within 2%.

The sample-size guard is visibly doing its job: p99 is withheld at n=90 and
reported at n=109. Before the fix this run would have printed a confident p99
from 7 samples.

Note both load points sit at or above capacity, so neither provides an unloaded
TTFT baseline and the automatic knee detector correctly reports no knee. The
unloaded reference is the serial run: **TTFT 343 ms client / 195 ms server.**

### PHASE 1 FINAL SCOREBOARD

| Quantity | Predicted | Measured | Gap |
|---|---|---|---|
| Weights in VRAM | 15.26 GiB | 15.26 GiB | 0.0% |
| Decode, batch 1 | 23.8 tok/s | 22.8 tok/s | -4.2% |
| TTFT @ 512 prompt | 176 ms | 195 ms | +11.0% |
| Capacity | 0.338 req/s | 0.332 req/s | -1.8% |

Implied `mem_eff` from measurement: **0.623** (guessed 0.65).
`compute_eff` is the weakest term at +11% on prefill; 0.50 is slightly optimistic
for a single un-batched 512-token prefill.

**Phase 2 target: beat 0.33 req/s without making ITL worse than ~55 ms.**

---
## 2026-08-21 — Correction: one Phase 0 prediction was never tested

The 2026-08-20 entry listed three things to check in Phase 1. Two were measured and
recorded above. The third was not:

> bf16 dies of KV exhaustion above **batch 4** at 4k context

**Status: UNTESTED, carried to Phase 2.** The Phase 1 baseline holds a global lock, so
batch size was always 1 and KV cache pressure never occurred. Nothing in the Phase 1
results speaks to this prediction either way.

It stops being theoretical the moment Phase 2 batches anything: bf16 leaves 3.98 GiB
of KV room, which is roughly 29,000 tokens total across all concurrent requests. At 4k
context that is about 7 requests -- so the first honest batching experiment should hit
this wall almost immediately. Test it deliberately rather than discovering it as an OOM.

### Also not applied deliberately

Measured `mem_eff` is 0.623 against the 0.65 default in `tools/roofline.py`. The
default was NOT changed. 0.623 was measured through naive HF `transformers`, which pays
Python overhead on every decode step; vLLM should achieve more of peak bandwidth. That
makes 0.623 a property of the STACK, not of the A10G, and hard-coding it would
understate every later phase. Re-derive it per stack instead.

---
## 2026-08-21 — Phase 2, experiment 1: KV cache structure and the real batch ceiling

Testing the prediction carried over from Phase 0/1: "bf16 dies of KV exhaustion above
batch 4 at 4k context." Written BEFORE the box was up.

### First: the claim as stated is wrong, and for a boring reason

"Above batch 4" is an artifact of `roofline.py` printing powers of two. The batch
scaling table steps 1, 2, 4, 8 -- batch 4 fits and batch 8 does not, so the claim was
read off the last passing row. The actual threshold under the roofline's own budget is
**7.08**, not 4. Nothing dies above 4; rows 5, 6 and 7 were simply never printed.

### Second: there are TWO ceilings and they differ by 1.6x

The 3.98 GiB / 7 requests figure is a *production budget*, not a hardware limit. It
comes from `0.9 x VRAM - weights - 1.0 GiB`, where the 0.9 mirrors vLLM's
`--gpu-memory-utilization` default and the 1.0 GiB is a fixed overhead allowance.
Raw PyTorch applies neither. This experiment runs raw PyTorch, so it tests the second
ceiling, and reporting it against the first would be comparing two different quantities.

|  | Budget | KV room | Tokens | Batch at 4096 ctx |
|---|---|---|---|---|
| Roofline / vLLM convention | `0.9*22.488 - 15.256 - 1.0` | 3.98 GiB | 29,003 | **7.08** |
| Raw PyTorch (what is tested) | `22.488 - 15.256 - ~1.0` | ~6.23 GiB | ~45,400 | **~11.1** |

### Predictions

| # | Quantity | Predicted | Derivation | Confidence |
|---|---|---|---|---|
| P1 | KV bytes per token | **147,456 B = 144.0 KiB** | `2 x 36 layers x 8 kv-heads x 128 head_dim x 2 B` | Very high -- pure arithmetic from config |
| P2 | VRAM slope during decode, batch 1 | **147,456 B/token** | same as P1; nothing else should grow per step | High |
| P3 | Largest batch that fits at 4096 ctx, raw PyTorch | **11** (range 10-12) | 7.232 GiB free after weights, minus ~1.0 GiB overhead, / 144 KiB / 4096 | Medium -- the overhead term is estimated, not measured |
| P4 | Total tokens in flight at that ceiling | **~45,400** | 6.23 GiB / 144 KiB | Medium |
| P5 | Batch 8 at 4096 ctx | **fits** | 32,768 tokens < ~45,400 | Medium -- this is the row roofline claimed would fail |

Overhead budget for P3, itemised so the gap is attributable:
CUDA context ~0.40 GiB, chunked-prefill activations ~0.04 GiB per batch item at
chunk 512, `DynamicCache` concat transient ~16 MiB per batch item, plus allocator
fragmentation.

### Two traps this experiment has to avoid

1. **Logits explosion.** HF computes logits for every prefill position unless told
   otherwise. At batch 12 x 4096 ctx x 151,936 vocab in bf16 that is 14.9 GiB of
   logits -- it would OOM with the KV cache barely touched and look exactly like KV
   exhaustion. Must pass `logits_to_keep=1`.
2. **Activation spike masquerading as KV pressure.** Prefilling 4096 tokens in one
   pass makes activation memory scale with batch x 4096. Prefill in chunks of 512 so
   activations stay bounded and the only term growing with context is the KV cache.

Both would produce a plausible wrong number rather than an error, which is the
category that matters.

### What would make each outcome interesting

- P3 lands at 11: the memory model is complete and the overhead estimate was right.
- P3 lands well below 11: something un-budgeted is holding VRAM. Find it, then decide
  whether a paged allocator recovers it.
- P3 lands above 12: the CUDA context or activation estimate is too pessimistic, and
  roofline's fixed 1.0 GiB overhead allowance should become a measured number.

### Follow-up predictions, written before runs A/B/C

The first run tested sizes 1,2,4,6,8 and broke on first OOM, so **batch 7 was never
tried** -- the same powers-of-two artifact criticised at the top of this entry, repeated
in my own sizes list. "Ceiling = 6" is really "6 or 7". Also, all trials shared one
process, so allocator fragmentation accumulated across them and batch 8 may have failed
because of trials 1-6 rather than on its own.

Measured memory model at batch B (GiB), fitted from the first run:

    used = 15.256 (weights) + 0.5632*B (KV) + 0.0875*B (activations)
                            + frag(B) + 0.252 (CUDA context)

    frag measured: 0.356 / 0.576 / 1.157 / 1.656 at B = 1 / 2 / 4 / 6

| Run | Setup | Prediction | Derivation |
|---|---|---|---|
| A | batch 7 alone, fresh process, default allocator | **fits, marginally** | 15.256+3.942+0.613+~1.9+0.252 = 21.96 vs 22.060 addressable. Under 100 MiB of slack -- genuinely uncertain |
| B | batch 8 alone, fresh process, default allocator | **OOM even alone** | 15.256+4.506+0.700+~2.1+0.252 = 22.81 > 22.060. If B fits, trial-history contamination is real and the walk is not measuring what it claims |
| C | `expandable_segments:True`, dense walk 7..12 | **ceiling 10** (range 9-11) | If fragmentation goes to ~0: 0.6507*B <= 6.552 -> B <= 10.07 |

Run C is the one that matters. Fragmentation is the largest unbudgeted term in the
whole memory equation, and it is the term a paged allocator exists to eliminate. If C
lands at 10, then Phase 2's paged allocator has a measured 3-4 request headroom to
recover, and the original 11-request estimate was right about the physics and wrong
only about the allocator.

### ACTUALS -- 2026-08-21, all runs complete

| # | Quantity | Predicted | Measured | Verdict |
|---|---|---|---|---|
| P1 | KV bytes per token | 147,456 B (144.0 KiB) | **147,456 B** | exact |
| P2 | VRAM slope per decode step | 147,456 B/token | **147,456 B/token** | exact, on both cache tensors and torch allocated |
| P3 | Batch ceiling, 4096 ctx, default allocator | 11 (range 10-12) | **7** | MISS, -36% |
| P4 | Tokens in flight at ceiling | ~45,400 | **28,672** | MISS, same cause as P3 |
| P5 | Batch 8 fits | fits | **OOM** | wrong |
| A | Batch 7 alone, fresh process | fits marginally | **fits**, 0.615 GiB spare | correct |
| B | Batch 8 alone, fresh process | OOM even alone | **OOM** | correct -- trial-history contamination ruled out |
| C | Ceiling with `expandable_segments:True` | 10 (range 9-11) | **9** | correct, in range |

**The original claim was wrong in the pessimistic direction.** bf16 does not die above
batch 4 at 4k context. It dies above **batch 7** with the default allocator and above
**batch 9** with `expandable_segments:True`.

### Explaining the P3 gap: fragmentation, and a VRAM constant that was never right

Three separate errors, in order of size.

**1. Allocator fragmentation -- 1.93 GiB at batch 7, entirely unbudgeted.**
`DynamicCache` grows by `torch.cat`, allocating a new tensor each step and freeing the
old one. The caching allocator keeps the freed blocks, and they are the wrong size for
the next, larger, concatenation. Measured `reserved - allocated`:

| batch | 1 | 2 | 4 | 6 | 7 |
|---|---|---|---|---|---|
| default allocator | 0.356 | 0.576 | 1.157 | 1.656 | **1.929** GiB |
| `expandable_segments` | -- | -- | -- | -- | **0.972** GiB |

Under the default allocator this grows at 0.2645 GiB per batch item -- **47% of the
0.5630 GiB of KV it accompanies**. It never amortises. Under `expandable_segments` it
flattens to a roughly constant ~1.0 GiB regardless of batch.

**2. CUDA cannot address all the VRAM that `nvidia-smi` reports.**

| Source | Value | |
|---|---|---|
| Marketing | 24 GB | not a real number |
| `nvidia-smi` | 23,028 MiB = 22.488 GiB | what `roofline.py` budgeted from |
| CUDA `mem_get_info` | 22,589 MiB = **22.060 GiB** | what can actually be allocated |

428 MiB of driver/ECC reserve is invisible to torch, and the CUDA context takes a
further 258 MiB at init. `roofline.py` was optimistic by 0.43 GiB before a single
weight loaded -- about 3,000 KV tokens.

**3. Activations were underestimated 2x.** Estimated ~0.04 GiB per batch item at chunk
512; measured **0.0858 GiB**.

### The corrected memory model, and it is now predictive

    driver_used(B) = 0.252 (CUDA ctx)
                   + 15.256 (weights)
                   + 0.5630 * B (KV at 4096 ctx)
                   + frag(B)

    frag(B) = 0.077 + 0.2645 * B   default allocator
    frag(B) = ~1.0 (constant)      expandable_segments:True

    ceiling = largest B with driver_used(B) <= 22.060

Checks out both ways: default predicts OOM at B=8 (22.21 > 22.06), measured OOM at 8.
expandable predicts OOM at B=10 (22.14 > 22.06), measured OOM at 10.

### roofline.py was right for the wrong reason

Its production budget said **7.08**; the measured default-allocator ceiling is **7**.
That is a coincidence, not a validated model. vLLM's `0.9 x VRAM - 1.0 GiB` convention
happens to reserve almost exactly what raw PyTorch loses to fragmentation plus
activations. The two quantities are unrelated -- one is a deliberate safety margin, the
other is allocator behaviour -- and they agree to within 1% by accident. Had the
fragmentation been half as bad, roofline would have looked wrong while being just as
sound.

### The actionable result

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` buys **+29% concurrency** (7 -> 9
requests at 4k, 28,672 -> 36,864 tokens in flight) for one environment variable and
zero code change. Apply it to every Phase 2 engine run.

**Important distinction for Phase 2.** What was measured here is *allocator*
fragmentation, which `expandable_segments` largely fixes. It is NOT the *reservation*
waste that a paged allocator exists to eliminate -- every sequence in this test was
exactly 4096 tokens, so nothing was over-reserved. Real traffic has ragged lengths,
where a non-paged engine must reserve `max_len` per sequence and paging wins much more.
This experiment sets the ceiling; it does not yet measure paging's benefit.


---
## 2026-08-21 — Phase 2, step 1: manual decode loop (offline)

Replacing HF `.generate()` with an explicit loop we drive ourselves. No batching yet,
no server. 512-token prompt, 64 new tokens, greedy, batch 1 -- deliberately identical
to the Phase 1 measurement conditions so the numbers are directly comparable.

Written before the code was finished and before the box was restarted.

| # | Quantity | Predicted | Derivation | Confidence |
|---|---|---|---|---|
| S1 | Manual loop vs `.generate()`, greedy | **token-for-token identical** | Same weights, same argmax, same stopping rule. Any divergence is a bug in cache handling or position ids, not a numerical accident | Very high. If this fails, nothing else in the run is meaningful |
| S2 | Decode throughput | **23.5 tok/s** (42.6 ms/token), range 22.5-25.0 | Phase 1 measured 43.86 ms/token *through the server*. Removing the streamer thread, the asyncio queue, SSE encoding and the network should return 1-2 ms/token of CPU overhead | Medium |
| S3 | Prefill time, 512 prompt | **150 ms**, range 130-180 | roofline says 134 ms of pure prefill; Phase 1's 195 ms TTFT included one decode step (42 ms) and server overhead | Medium |
| S4 | ITL p95 / p50 ratio | **< 1.2** | Phase 1 saw p50 44.6 ms against p95 89.9 ms -- a 2.0x spread. Offline there is no queueing, no network, no event loop, so the spread should collapse | Medium. This is the interesting one |

**S4 is the diagnostic.** If the offline spread collapses to near 1.0, Phase 1's 2x ITL
tail was the *server* -- scheduling, GIL contention, the streamer handoff -- and not the
GPU. If it stays near 2.0, the tail is in the model execution itself and the scheduler
in step 3 inherits a problem it cannot fix.

### Noted in advance, not yet a problem

`DynamicCache` grows by `torch.cat`, which copies the **entire** cache every decode
step. That makes per-token cost O(L) and total decode O(L^2):

| context | copy per step | at 42 ms/token |
|---|---|---|
| 576 tokens (this test) | 0.079 GiB | 0.14 ms, 0.3% -- invisible |
| 4096 tokens | 0.562 GiB | 1.01 ms, 2.4% -- measurable |

So step 1 cannot detect this, by design. It becomes real at long context and is one of
the things a preallocated or paged cache eliminates. Do not conclude from a clean step-1
number that `DynamicCache` is fine; conclude only that 576 tokens is too short to expose
it.

### Infrastructure note

The box stopped itself on schedule after the KV probe finished -- first live
confirmation that the modified idle guardrail fires correctly on an inference box,
where a server process holding KV cache at 0% GPU utilisation looks exactly like idle.

### ACTUALS -- 2026-08-21, step 1 (412-token prompt, 64 new tokens, greedy, batch 1)

| # | Quantity | Predicted | Measured | Verdict |
|---|---|---|---|---|
| S1 | Manual loop vs `.generate()` | token-for-token identical | **IDENTICAL, 64 tokens** | correct |
| S2 | Decode throughput | 23.5 tok/s (22.5-25.0) | **24.4 tok/s** | correct, +3.8% off prediction |
| S3 | Prefill time | 150 ms (130-180) | **148.5 ms** | correct, 1% off |
| S4 | ITL p95/p50 | < 1.2 | **1.007** | correct, decisively |

All four correct. The loop is right and the model of it is right.

**Prompt length note.** The prompt resolves to 412 tokens, not 512 -- `make_prompt`
uses a 4-chars/token heuristic and Qwen3's tokenizer runs closer to 5. This is NOT a
comparability problem: `engine/manual.py` and `tools/bench.py` use byte-identical
FILLER text and the identical `target*4` slice, both producing 2048 chars, and Phase 1
recorded `prompt_chars: 2048`. Phase 1's "512-token prompt" was the same 412 tokens.
Verified by extracting and comparing both constants rather than assuming.

### S4 was the question worth asking, and the answer is clean

|  | Phase 1, through the server | Step 1, offline | |
|---|---|---|---|
| ITL p50 | 44.5 ms | **41.0 ms** | -3.5 ms |
| ITL p95 | 89.3 ms | **41.3 ms** | -48.0 ms |
| ITL p99 | 173.8 ms | **41.5 ms** | -132.3 ms |
| ITL max | 179.4 ms | -- | |
| **p95/p50 spread** | **2.008** | **1.007** | |
| samples | 689 | 189 | |

**The entire ITL tail was the serving layer, not the GPU.** Removing FastAPI, the
`TextIteratorStreamer` thread handoff, the asyncio queue, SSE encoding and the network
took p99 from 173.8 ms to 41.5 ms. The GPU emits tokens metronomically at 41 ms; every
millisecond of tail above that was software the model never saw.

Two consequences:
- The Phase 2 scheduler starts from a clean instrument. The 2x tail is not a property
  of model execution that a scheduler would inherit and be unable to fix.
- The serving layer cost 3.5 ms/token at the median (8%) on top of owning 100% of the
  tail. Whatever replaces it in step 3 has a measured budget to beat.

### Instrument defect found and fixed: warmup must match the measured shape

Rep 0 came in at 202.5 ms TTFT against 148.4 and 148.5 ms for reps 1 and 2 -- a 36%
outlier that survived warmup. Cause: warmup generated from a 6-token prompt
("The capital of France is") while the measurement used a 412-token prefill. cuBLAS
selects kernels per problem shape, so warming the wrong shape warms nothing relevant.

Fixed in `engine/manual.py`: warmup now runs on the actual `input_ids`. The reported
p50 of 148.5 ms is unaffected -- the median of three absorbed the outlier -- but a mean
would have reported 166.5 ms, 12% high, from a single un-warmed rep. Another argument
for percentiles over means, and this one was free to catch only because the per-rep
numbers were printed rather than just the summary.


### Step 1 final, 20 reps with warmup fixed

| Quantity | Value | Samples |
|---|---|---|
| TTFT p50 / p95 | **148.4 / 148.5 ms** | 20 |
| ITL p50 / p95 / p99 | **40.5 / 40.6 / 41.0 ms** | 1,260 |
| Decode throughput | **24.7 tok/s** | |
| TTFT spread p95/p50 | **1.001** | |
| ITL spread p95/p50 | **1.002** | |

The warmup fix removed the rep-0 outlier completely: rep 0 is now 148.5 ms and all 20
reps fall in 148.4-148.5 ms. This is the most stable instrument the project has had.

### Derived efficiency constants -- one confirms, one does not

    mem_eff     = 27.30 ms floor / 40.5 ms measured = 0.674
    compute_eff = 54.0 ms at 125 TFLOPS / 148.4 ms  = 0.364

| Constant | roofline default | measured offline | Phase 1 via server |
|---|---|---|---|
| `mem_eff` | 0.650 | **0.674** | 0.623 |
| `compute_eff` | 0.500 | **0.364** | -- |

`mem_eff` at 0.674 slightly beats the 0.65 default and settles the Phase 1 question:
0.623 was the server tax, not the card. The A10G sustains 67% of spec bandwidth.

`compute_eff` at 0.364 means **roofline overstates prefill speed by 37%**. Attention
FLOPs, which roofline ignores, are only 1.5% of the weight matmuls at 412 tokens, so
they do not explain it.

## 2026-08-21 — Phase 2, step 1b: is compute_eff a constant?

Suspicion: it is not. `compute_eff` should rise with prefill length, because a
412-token prefill is a [412 x 4096] GEMM whose M dimension is too small to fill the
tensor cores. Roofline treats it as a fixed 0.50 at every length, which would make it
optimistic for short prompts and pessimistic for long ones.

Measuring prefill-only (`--max-new-tokens 1`) across seven prompt lengths.

| target | actual approx | predicted `compute_eff` (weights-only FLOPs) |
|---|---|---|
| 128 | 103 | 0.15 |
| 256 | 206 | 0.25 |
| 512 | 412 | 0.364 MEASURED |
| 1024 | 824 | 0.45 |
| 2048 | 1648 | 0.52 |
| 4096 | 3296 | 0.57 |
| 8192 | 6592 | 0.60 |

Prediction: **monotonically increasing, saturating near 0.6**, never reaching 1.0.

Correction that must be applied when reading the tail of that table: attention FLOPs
scale as L^2 while weight matmuls scale as L, so the attention share is
`L x 3.6e-5` -- 1.5% at 412 tokens but **24% at 6,592**. Since roofline counts only
weight matmuls, apparent `compute_eff` at long prompts is depressed by real work the
model does not count. Expect the raw curve to flatten or dip at the far end for that
reason alone, and correct for it before concluding the hardware stopped scaling.


### ACTUALS -- 2026-08-21, step 1b prefill sweep

| L | TTFT | eff, weights only | eff, incl. attention | predicted | attn share |
|---|---|---|---|---|---|
| 113 | 49.8 ms | 0.297 | **0.299** | 0.15 | 0.4% |
| 213 | 87.0 ms | 0.321 | **0.323** | 0.25 | 0.8% |
| 412 | 148.4 ms | 0.364 | **0.369** | 0.364 | 1.5% |
| 812 | 283.2 ms | 0.376 | **0.387** | 0.45 | 2.9% |
| 1611 | 484.3 ms | 0.436 | **0.461** | 0.52 | 5.8% |
| 3209 | 952.3 ms | 0.442 | **0.493** | 0.57 | 11.6% |
| 6407 | 1968.0 ms | 0.427 (dips) | **0.525** | 0.60 | 23.1% |

**Shape: correct. Values: wrong in both directions.** The curve rises monotonically and
saturates below 1.0 as predicted, but it is far flatter than I guessed -- I predicted a
0.15 to 0.60 span (4.0x) and measured 0.30 to 0.53 (1.75x). Too pessimistic at short
prompts, too optimistic at long ones. `compute_eff` is genuinely a function of prompt
length, which is the thing worth knowing; my sense of how steeply is not calibrated.

**The attention correction, written down in advance, is what saves the reading.** The
weights-only column dips at the last row, 0.442 to 0.427. Read alone it says the GPU
stopped scaling past 3k tokens. It did not -- attention is 23.1% of the real work at
L=6407 and this model counts none of it. Corrected, the curve is monotonic through the
last point: 0.493 to 0.525. Had that correction not been predicted beforehand, the dip
would have been a genuinely convincing artifact.

Also fitted: `t = 16.7 ms + 0.3018 ms/token`. About 17 ms of fixed per-call overhead,
which is 34% of a 113-token prefill and under 1% of a 6.4k one.

### Applied to roofline.py

`COMPUTE_EFF` default changed **0.50 -> 0.36**, with the measured table recorded inline
and guidance to override for long context (~0.46 at 1.6k, ~0.49 at 3.2k, ~0.53 at 6.4k).
Roofline now predicts 186 ms of prefill at 512 tokens; the measurement scales to 184 ms.
That agreement is calibration, not validation -- the default was fitted to this point.
The honest test is whether the long-context overrides hold up in Phase 3.

Open item: roofline treats `compute_eff` as a scalar. It is a curve. Interpolating from
the measured table would replace a fudge factor with a measurement, per CLAUDE.md 5,
but the curve is specific to A10G + Qwen3-8B and would need re-measuring per stack.
Deferred, not forgotten.

---
## 2026-08-21 — Phase 2, step 2: static batching

Batch N requests, run them to completion together, no sequence joining or leaving
mid-flight. That restriction IS static batching, and exposing what it costs is the
entire purpose of the step.

Prompts are uniform (412 tokens, same text) so that ragged OUTPUT length is the only
variable. Ragged prompt lengths add a separate padding waste; deliberately excluded
here so the two effects do not mix.

All predictions use constants measured today: `mem_eff` 0.674, `compute_eff` 0.36,
KV 144.0 KiB/token, 476 tokens per sequence (412 prompt + 64 out).

### 2a. Uniform lengths -- the clean batching curve

| B | predicted ITL | predicted tok/s | vs batch 1 |
|---|---|---|---|
| 1 | 40.7 ms | 25 | 1.0x |
| 2 | 40.9 ms | 49 | 2.0x |
| 4 | 41.2 ms | 97 | 3.9x |
| 8 | 41.9 ms | 191 | 7.7x |
| 16 | 43.3 ms | 370 | 15.0x |
| 24 | 44.7 ms | 537 | 21.7x |
| 32 | 46.1 ms | 695 | 28.1x |

**The claim being tested: 32x the throughput for 13% more latency per token.** That is
the whole economic argument for batching, and it works because a decode step reads all
16.4 GB of weights exactly once no matter how many sequences share it. Only the KV
reads grow with B, and at 476 tokens KV is 0.065 GiB per sequence against 15.26 GiB of
weights.

**Decode stays memory-bound at every batch size tested.** Compute overtakes memory only
at B = 213, which is unreachable -- memory runs out first. If any measured point comes
back compute-bound, the arithmetic above is wrong somewhere.

### 2b. Predicted ceiling, and why it is NOT the KV ceiling

At 4096 context the binding constraint was KV cache (batch 9). At 476 context it is
**prefill activations**, which is a different limit with a different fix.

    usable after weights + CUDA ctx + fragmentation:      5.552 GiB
    per sequence:  KV 0.065 GiB  +  prefill activations 0.080 GiB  =  0.145 GiB
    ceiling = 5.552 / 0.145 = 38.2  ->  predict OOM between B=32 and B=40

Activations scale with **tokens per forward pass** (B x 476 during prefill), not with
context length, so they dominate exactly when sequences are short and numerous. KV at
B=38 is only 2.5 GiB of the 5.5. Chunking the prefill would raise this ceiling
substantially -- which is what chunked prefill is for, and a Phase 3 vLLM flag.

### 2c. Ragged output lengths -- the number that motivates step 3

Setup: B=8, output lengths `[512, 32, 32, 32, 32, 32, 32, 32]`. Static batching cannot
release a finished sequence, so all 8 slots stay occupied for all 512 steps.

    useful tokens    = 512 + 7 x 32          =   736
    slot-steps       = 8 x 512               = 4,096
    slot utilisation = 736 / 4096            =  18.0%
    useful throughput at ITL 41.9 ms         =  34.3 tok/s
    uniform-B=8 throughput for comparison    = 191   tok/s

**Predicted: 82% of the GPU's work is thrown away, and useful throughput collapses to
below double batch-1.** Seven sequences finish at step 32 and then occupy a slot doing
arithmetic nobody reads for another 480 steps.

The instrument must report BOTH numbers -- naive `B x steps / time` will still show
~191 tok/s and look healthy. The gap between naive and useful throughput is the
finding, not a rounding error. If step 2 reports only the naive number it has measured
nothing.

Continuous batching in step 3 exists to refill those slots. Expected recovery toward
the uniform curve is the step 3 prediction, made once this one is measured.

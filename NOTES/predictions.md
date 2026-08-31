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

### Two experiments added before running, with 2d as the control 2b needed

2b on its own shows a bad number but not a clean comparison -- it differs from the
uniform sweep in both batch composition and step count. 2d fixes that: same B=8, same
512 decode steps, same growing context, every slot useful. 2b and 2d differ in exactly
one thing, which is whether the slots are doing work anyone asked for.

ITL is modelled at the midpoint context (412 prompt + steps/2) because KV grows during
a long run and ITL grows with it.

| run | lengths | steps | predicted util | predicted useful tok/s | predicted naive tok/s |
|---|---|---|---|---|---|
| 2b ragged, one straggler | `[512, 32 x 7]` | 512 | **18.0%** | **33.9** | 188.4 |
| 2c ragged, realistic spread | `[256,128,64,64,32,32,16,16]` | 256 | **29.7%** | **56.4** | 190.1 |
| 2d uniform control | `[512 x 8]` | 512 | 100% | 188.4 | 188.4 |

**The claim: 2b and 2d perform identical GPU work and differ 5.6x in tokens delivered.**
Same batch size, same step count, same weights read per step. The only difference is
that in 2b seven of eight rows finished at step 32 and spent the next 480 steps
computing tokens that are discarded.

`naive_tok_s` is predicted to be ~188-190 in ALL THREE cases, including the two that
waste most of their work. That is the trap this instrument exists to expose: the
flattering number is nearly constant while the real one moves 5.6x.

Note on `--ignore-eos`, added after review and defaulted on: every row shares an
identical prompt and decodes greedily, so all B rows emit identical tokens. A real eos
would fire on all eight rows at the same step, killing the 512-token straggler along
with the short rows and erasing the effect being measured. `useful_tokens` was also
changed to count tokens actually produced rather than requested -- with the original
formula an early eos would report utilisation above 100%.

### ACTUALS -- 2026-08-21, step 2

#### 2a. Uniform batch sweep

| B | ITL p50 | tok/s | throughput vs B=1 | ITL vs B=1 | predicted tok/s |
|---|---|---|---|---|---|
| 1 | 40.5 ms | 24.7 | 1.0x | 1.00x | 25 |
| 2 | 40.8 ms | 48.9 | 2.0x | 1.01x | 49 |
| 4 | 41.9 ms | 95.4 | 3.9x | 1.03x | 97 |
| 8 | 43.8 ms | 182.7 | 7.4x | 1.08x | 191 |
| 16 | 49.0 ms | 326.6 | 13.2x | 1.21x | 370 |
| 24 | 54.2 ms | 442.6 | 17.9x | 1.34x | 537 |
| 32 | 61.0 ms | 524.0 | 21.2x | 1.51x | 695 |
| 40 | 64.8 ms | 616.9 | 25.0x | 1.60x | -- |
| 48 | 68.7 ms | **698.1** | **28.3x** | 1.70x | -- |

**Batching works, but sublinearly, and I predicted the wrong shape.** I said 32x
throughput for 13% more latency. Measured at B=32: 21.2x for 51% more latency. The
direction and the magnitude of the win are right; the curve bends much earlier than
predicted.

#### The cause: memory and compute are ADDITIVE, not overlapped

I modelled `ITL = max(t_mem, t_cmp)`, which is the textbook roofline. It is wrong here.

| B | measured | `max(t_mem,t_cmp)` | `t_mem + t_cmp` |
|---|---|---|---|
| 8 | 43.8 ms | 41.8 (-4.6%) | 44.7 (+2.1%) |
| 16 | 49.0 ms | 43.1 (-12.0%) | 48.9 (-0.2%) |
| 32 | 61.0 ms | 45.7 (-25.1%) | 57.3 (-6.0%) |
| 48 | 68.7 ms | 48.3 (-29.7%) | 65.8 (-4.3%) |

`max()` is 30% optimistic by B=48. Additive fits within 6% across the entire range.

The reason is physical and obvious in hindsight: **a decode step cannot hide its weight
read underneath its own matmul, because the matmul is what consumes the weights.** They
are the same operation, sequenced, not two pipelines to overlap. `max()` would be right
for two independent units contending for time; it is wrong for one dependent chain.

Applied to `tools/roofline.py`: batch scaling now uses `t_mem + t_cmp`, with the table
above recorded inline. It now predicts B=32 at 59.4 ms / 539 tok/s against measured
61.0 / 524, inside 3%.

#### 2b. Ceiling: predicted 38, measured at least 48

B=48 ran with peak allocation 19.94 GiB and did not OOM -- the sweep ended because the
list ended, not because memory did. Implied activations are **0.032 GiB per sequence**
at a 412-token prefill, against the 0.080 GiB I predicted from the kvprobe measurement.
That earlier figure was taken at 4096 context with chunk 512 and evidently folded in
the `DynamicCache` concat transient, which is not present here. Activation cost per
token-in-flight is not a single constant across regimes; do not carry it between them.

#### 2c. Static batching waste -- the exact predictions

| run | lengths | util predicted | util measured | useful tok/s | naive tok/s |
|---|---|---|---|---|---|
| 2b | `[512, 32 x 7]` | 18.0% | **18.0%** | **31.6** | 175.8 |
| 2c | `[256,128,64,64,32,32,16,16]` | 29.7% | **29.7%** | **53.3** | 179.4 |
| 2d | `[512 x 8]` control | 100% | **100%** | 175.8 | 175.8 |

Utilisation matched exactly in both cases, which it should -- it is pure combinatorics.
Useful throughput came in 7% under prediction for the same reason 2a did: the additive
compute term was missing from the ITL model.

**The result this step exists to produce:**

    2b and 2d report the SAME naive throughput -- 175.8 tok/s, to four figures.
    They have the same batch size, the same 512 decode steps, the same ITL
    (45.5 vs 45.4 ms), and the same 1.015 GiB of KV.

    Useful throughput: 2b = 31.6 tok/s, 2d = 175.8 tok/s.  A 5.56x gap.

Identical GPU work, identical reported throughput, 5.56x difference in tokens anyone
receives. In 2b, 3,360 of 4,096 slot-steps computed tokens for sequences that had
already finished. Seven of eight rows completed at step 32 and then occupied the
machine for another 480 steps producing output that was discarded.

**This is the entire case for continuous batching, and it is invisible to any
throughput metric that divides by slots instead of by delivered tokens.** Had this tool
reported only `naive_tok_s`, static batching would look like a solved problem.

#### What step 3 has to beat

    static batching, ragged [512, 32 x 7]:   18.0% utilisation,  31.6 useful tok/s
    theoretical ceiling (2d, all useful):   100.0% utilisation, 175.8 useful tok/s

Continuous batching should recover most of that gap by admitting a queued request into
a slot the step it frees. It cannot reach 175.8 -- there is scheduling overhead and the
admitted requests must prefill -- but anything below about 100 useful tok/s means the
scheduler is leaving half the available work on the floor.

---
## 2026-08-21 — Phase 2, step 3: continuous batching

### First, a design question that had to be settled before any code

Continuous batching needs rows at DIFFERENT sequence lengths in one batch. HF's cache
API assumes a single shared `cache_position` for the whole batch. That mismatch is the
reason vLLM wrote its own attention kernels, and it is the actual obstacle in this step.

Proposed way through without custom kernels: **left-pad every row into a common buffer
and decouple buffer index from RoPE position.** A row's K and V are rotated by RoPE at
write time, so where it physically sits in the buffer is irrelevant to correctness --
only the attention mask and `position_ids` matter. If that holds, admission and eviction
become plain `index_select` and `cat` on the batch dimension.

`engine/cache_probe.py` tests it against batch-1 references, since the only acceptable
evidence is token-for-token identity:

| test | prediction | confidence |
|---|---|---|
| Ragged left-padded batch == unpadded batch-1 | IDENTICAL | high -- standard HF left-padding |
| Row survives eviction of another row mid-decode | IDENTICAL | medium-high -- depends on mutating `cache.layers[i].keys` in place |
| Row admitted at step 6 decodes correctly | IDENTICAL | medium -- its zero-padded KV region must be fully nullified by the mask |

If test 3 fails the engine needs a real paged cache before it can work at all, which
moves step 4 ahead of step 3.

### The capacity model, and the thing it predicts that I did not expect

Every admission requires an exclusive prefill (148.4 ms measured) that stalls the whole
batch, because a newly admitted request has no KV yet. Decode steps are shared across B.
Steady state solves `R*prefill + (out_tokens*R/B)*t_step = 1`:

| B | t_step | prefill share of time | capacity req/s | vs Phase 1 (0.332) |
|---|---|---|---|---|
| 1 | 40.5 ms | 5.4% | 0.36 | 1.1x |
| 4 | 41.9 ms | 18.1% | 1.22 | 3.7x |
| 8 | 43.8 ms | 29.8% | **2.00** | **6.0x** |
| 16 | 49.0 ms | 43.1% | 2.90 | 8.7x |
| 32 | 61.0 ms | 54.9% | 3.70 | 11.1x |
| 48 | 68.7 ms | 61.8% | 4.17 | 12.6x |

**Prefill, not decode, is what caps this server.** As B grows the decode cost per
request falls but the 148 ms prefill does not, so it goes from 5% of the time budget at
B=1 to 62% at B=48. The hard ceiling with infinite batch is `1/0.1484 = 6.74 req/s`.

That is the headline prediction of step 3 and it was not obvious beforehand: building a
better scheduler moves the bottleneck off decode and onto prefill. It is also exactly
what chunked prefill and prefix caching attack in Phase 3, so Phase 2 ends by generating
the question Phase 3 answers.

### Targets

    Phase 1 baseline (serialized):              0.332 req/s
    step 2 static, ragged workload:             31.6 useful tok/s, 18.0% utilisation
    step 3 predicted at B=8:                    2.00 req/s, 6.0x
    step 3 predicted at B=32:                   3.70 req/s, 11.1x
    prefill-only ceiling:                       6.74 req/s

Secondary prediction: ITL p99 will be MUCH worse than step 1's metronomic 41.0 ms,
because every admission stalls the batch for a full 148 ms prefill. Expect ITL p50 near
the step-2 value for the batch size, and p99 at roughly p50 + 148 ms. **The scheduler
buys throughput and pays for it in tail latency** -- if p99 does not degrade, the engine
is not actually admitting anything mid-flight and the measurement is wrong.

### Cache probe result -- 2026-08-21, all three mechanics VERIFIED

| test | result |
|---|---|
| Ragged left-padded batch vs unpadded batch-1 | **IDENTICAL** (both rows, 16 tokens) |
| Row survives eviction of another row at step 6 | **IDENTICAL** |
| Row admitted at step 6, decoding mid-flight | **IDENTICAL** (11 tokens) |
| Existing row unaffected by that admission | **IDENTICAL** |

Left-padding with decoupled RoPE positions works, so continuous batching is buildable on
stock HF without custom attention kernels. The paged allocator (step 4) is therefore a
memory-efficiency improvement, not a prerequisite -- which is the correct ordering and
matches PROJECT.md's plan.

### Step 3a workload prediction: 64 requests, max_batch 8, lengths cycling the 2c spread

    useful tokens = 8 x [256,128,64,64,32,32,16,16] = 4,864

| | steps | utilisation | wall | useful tok/s |
|---|---|---|---|---|
| static (arithmetic, groups of 8) | 2,048 | 29.7% | 97.5 s | **49.9** |
| continuous, batched admissions | 608 | ~90-100% | 34.3 s | **141.8** |
| continuous, admissions one at a time | 608 | ~90-100% | 36.1 s | 134.6 |
| decode-only ceiling (no prefill cost) | 608 | 100% | 26.6 s | 182.6 |

**Predicted speedup over static on the identical workload: 2.8x.** Continuous batching
needs 608 decode steps where static needs 2,048 for the same delivered tokens -- static
spends the other 1,440 computing for sequences that already finished.

**Predicted prefill share of wall time: 22%.** That is the cost of the fix. The engine
reaches only 78% of the decode-only ceiling because every admission stalls the batch,
and this is the measurement that sets up Phase 3's chunked prefill.

Batched admission is predicted to be worth 5% overall (141.8 vs 134.6 tok/s) -- real but
much smaller than the 18% saving on prefill alone, because prefill is only ~22% of wall
time. Worth doing, not worth contorting the scheduler for.

### Isolation experiment, written before running

Step 3a measured ITL p50 61.4 ms at max_batch 8. Step 2 measured 43.8 ms at B=8, and
the memory model predicts ~45 ms even after accounting for the grown buffer and the 6.61
average active rows. A 37% gap needs a cause, not a shrug.

**Hypothesis: passing an explicit `attention_mask` costs a large constant.**
`static_batch.py` passes NO attention mask -- prompts are uniform and unpadded, so SDPA
takes its `is_causal=True` fast path with no mask tensor at all. `continuous.py` MUST
pass an explicit 2-D mask, because left-padded rows have regions that must not be
attended to. That forces SDPA onto the masked path.

If true, this is a real and previously uncounted cost of the left-padding design, and
another thing a paged cache removes.

Clean A/B, identical in every other respect -- same batch, same 256 steps, same context
growth 412 -> 668, no mid-run admissions and no eviction until the end:

    static_batch.py  --batches 8 --max-new-tokens 256      (no mask, no position_ids)
    continuous.py    --max-batch 8 --n-requests 8 --lengths 256   (explicit mask + position_ids)

**Prediction: continuous lands ~25% higher, near 55 ms against static's ~45 ms.** If the
two come out equal, the mask is free and the 37% is somewhere else entirely -- most
likely the per-step Python and tensor bookkeeping in the scheduler, which would be a
much less interesting answer but needs ruling out either way.

### ACTUALS -- 2026-08-21, step 3a continuous batching

| quantity | predicted | measured | |
|---|---|---|---|
| decode steps | 608 | **736** | +21% |
| slot_steps_used | 4,864 | **4,864** | exact -- fixed by the workload |
| decode utilisation | 90-100% | **82.6%** | tail drain |
| static equivalent utilisation | 29.7% | **30.1%** | |
| **utilisation ratio vs static** | **2.8x** | **2.75x** | correct |
| prefill share of wall | 22% | **15.9%** | better than predicted |
| useful tok/s (wall) | 141.8 | **95.3** | -33% |
| ITL p50 | ~44 ms | **61.4 ms** | +40% |
| ITL p99 degrades vs step 1's 41.0 ms | yes, ~p50+148 | **69.6 ms** run / 186.4 ms smoke | correct |

Continuous batching does what it claims: 736 decode steps against static's 2,048 for the
identical delivered tokens, and 82.6% slot utilisation against 30.1%. The 17.4% shortfall
from full is the **tail drain** of a finite workload -- the last group empties with
nothing left to refill it. A continuously arriving stream does not pay that.

### Why throughput missed by 33%: left-padding is expensive, and only when it pads

ITL came in at 61.4 ms where step 2 measured 43.8 ms at the same batch size. Two
experiments were needed to find the cause, and the first one was wrong.

**Failed isolation.** Hypothesis: passing an explicit `attention_mask` forces SDPA off
its `is_causal=True` fast path. Predicted +25%. Ran continuous with 8 requests admitted
together, 256 tokens each, against static at the same shape:

    static_batch, no mask            ITL p50  44.6 ms
    continuous, explicit mask        ITL p50  44.7 ms

Zero cost. Hypothesis apparently dead. **But the control was broken:** admitting all 8
at once means every row starts at the same length, so the mask is ALL ONES. It masks
nothing. The experiment removed the padding along with the thing it meant to test.

**The time series settles it.** Per-step ITL against buffer length, at constant 8 active
rows:

| steps | buffer | active rows | ITL p50 |
|---|---|---|---|
| 0-100 | ~462 | 8.0 | 54.9 ms |
| 100-200 | ~562 | 8.0 | 57.9 ms |
| 200-300 | ~662 | 8.0 | 60.9 ms |
| 300-400 | ~762 | 8.0 | 63.3 ms |
| 400-500 | ~862 | 8.0 | 66.2 ms |
| 500-600 | ~962 | 6.7 | 68.6 ms |
| 600-800 | ~1112 | 1.5 | 48.4 ms (drain) |

ITL rises 13.7 ms across 500 tokens of buffer growth: **27.4 microseconds per buffer
token, against 2.9 predicted by the KV-read model. 9.4x.**

**Conclusion: an attention mask costs nothing when it is trivially full, and a great deal
when it actually masks.** A full mask is presumably recognised and dispatched to the fast
kernel; a mask with real zeros forces the explicit path, whose cost scales with buffer
length. Since left-padding is what puts zeros in the mask, the penalty is proportional to
how much padding the design carries -- which grows monotonically as the buffer does.

This is a cost of the left-padded design, not of continuous batching. Paged attention
removes padding entirely, so step 4 should recover most of it: at the model's ~45 ms ITL
this workload would deliver roughly 130 tok/s rather than 95.3, close to the original
141.8 prediction.

**Step 4 now has a measured target rather than a principle:** eliminate 27.4 us per
buffer token of padding tax, worth about 36% on this workload.

### Scoreboard, Phase 2 so far

    Phase 1 serialized server                     0.332 req/s   ITL p50 44.5, p99 173.8 ms
    step 1 manual loop, batch 1                    24.7 tok/s   ITL p50 40.5, p99  41.0 ms
    step 2 static, ragged workload                 53.3 tok/s   utilisation 29.7%
    step 3a continuous, same workload              95.3 tok/s   utilisation 82.6%
    step 3a decode-only (no admission stalls)     114.3 tok/s
    step 2d uniform ceiling (no waste at all)     175.8 tok/s

## 2026-08-21 — Phase 2, step 3b: buffer compaction, then the server

### The bug step 3a hid

The left-padded buffer grows by one token per decode step and only resets when the batch
empties completely. A finite 64-request run ends before that matters; a server under
sustained load never empties, so the buffer grows without bound. At ~16 steps/s that is
~1,000 tokens per minute. Over a 60 s `bench.py` run the buffer would go 412 -> ~1,400
and ITL would drift from ~55 to ~75 ms *during the measurement*, making the result a
function of how long the run happened to be. That is not a measurable server.

### The fix, and why it is safe

Every row is RIGHT-aligned in the buffer: a row of true length L occupies
`[cur_L - L, cur_L)`. So the first `cur_L - max(row_len)` positions are padding for
EVERY active row simultaneously, and can be sliced off without touching valid data.
`next_pos[i]` already tracks each row's true length, so the trim point is free to compute.

Compaction is therefore a front slice of every layer's K and V plus the mask. Cost at
B=8 and 668 needed positions is ~0.73 GiB copied read+write, about 2.4 ms -- cheap when
amortised over the ~128 steps between compactions.

### Predictions

| # | quantity | predicted | reasoning |
|---|---|---|---|
| C1 | Output still token-identical | **yes** | Slicing only removes positions masked to 0 for every row |
| C2 | Buffer stays bounded at `max(row_len) + threshold` | **yes, <= ~800** | vs 1,148 unbounded in step 3a |
| C3 | step 3a ITL p50 | 61.4 -> **~56 ms** | Mean buffer falls from ~780 to ~640; at the measured 27.4 us/token that is -3.8 ms, plus compaction overhead |
| C4 | step 3a useful tok/s | 95.3 -> **~102** | Directly from C3 |
| C5 | Compaction overhead | **< 2% of wall** | ~2.4 ms every ~128 steps against ~60 ms/step |

C3 and C4 are modest because step 3a's workload is short. **The point of compaction is
not this 7%; it is that the server has a bounded buffer at all.** If C1 fails the whole
left-padding design is unsafe and step 4's paged cache becomes mandatory rather than an
optimisation.

### ACTUALS -- 2026-08-21, compaction

| # | quantity | predicted | measured | |
|---|---|---|---|---|
| C1 | Output token-identical | yes | **12/64 differ** | prediction failed -- but the GATE was wrong, see below |
| C2 | Buffer bounded | <= ~800 | **764** (vs 1,147 without) | correct |
| C3 | ITL p50 | ~56 ms | **59.1 ms** (from 61.4) | direction right, magnitude overstated |
| C4 | useful tok/s | ~102 | **99.3** (from 95.3) | correct within 3% |
| C5 | Compaction overhead | < 2% of wall | **0.014%** (7 ms of 49.6 s) | correct, by two orders of magnitude |

### C1: I asked for the wrong thing, and two controls proved it

Compaction changed 12 of 64 outputs. The divergence pattern was the first clue it was
not corruption:

- every request's LENGTH was preserved, so nothing shifted or was truncated
- only requests still IN FLIGHT during a compaction differed; every request that
  finished before the first compaction was bit-identical
- rows sharing an admission step diverged at the SAME offset (40/41 at 70, 49/50/51 at
  41, 56/57 at 119), which is a per-step shared cause, not per-row damage

Two controls settle it:

| test | result |
|---|---|
| original vs `--compact-threshold 999999999` (code present, never fires) | **IDENTICAL**, 4,928 tokens |
| compaction run 1 vs compaction run 2 | **IDENTICAL**, 4,928 tokens |
| compaction disabled vs enabled | 12/64 differ |

The code path is inert when it does not fire, and the engine is deterministic given a
fixed schedule. The only variable left is tensor SHAPE. Slicing the buffer changes
attention's accumulation order, and bf16 addition is not associative; masked positions
contribute exactly 0.0 either way, so the mathematics is unchanged while the arithmetic
is not. A near-tie argmax then flips and the sequences diverge from that point.

**Token identity was never an achievable gate for this component.** It was right for
step 1, where the manual loop had to match `.generate()` on identical shapes. A
scheduler changes tensor shapes as a function of batch composition, so its output varies
with scheduling in bf16 -- vLLM has the same property, which is why its outputs are not
reproducible across different `--max-num-seqs`. The correct gate, and the one now
established, is:

    1. the component is inert when disabled           VERIFIED
    2. the engine is deterministic given a schedule   VERIFIED
    3. differences are attributable to shape alone    VERIFIED by 1 and 2

Compaction ships. The buffer is bounded at 764 rather than growing without limit, which
is what makes a server measurable at all.

## 2026-08-21 — Phase 2, step 3b: the engine behind the OpenAI streaming API

Measured with `tools/bench.py` unchanged, open-loop Poisson arrivals, 512-target prompt
(412 actual) and 64 max output tokens -- byte-identical workload to Phase 1, which is
the whole reason bench.py's wire format is a hard invariant.

### Predicted capacity

Steady state solves `R*prefill + (out_tokens*R/B)*ITL = 1` with prefill 148.4 ms
measured. ITL is estimated per batch size from the memory model plus the padding tax
measured in step 3a (27.4 us per buffer token at 8 rows). Compaction should hold the
buffer near 412 + 64 + 128 = 604, shorter than step 3a's 764, so ITL should sit below
step 3a's 59.1 ms.

| max_batch | ITL estimate | predicted capacity | vs Phase 1 (0.332) |
|---|---|---|---|
| 4 | 48.9 ms | 1.07 req/s | 3.2x |
| 8 | 57.3 ms | **1.65 req/s** | **5.0x** |
| 16 | 74.2 ms | 2.25 req/s | 6.8x |
| 32 | 107.8 ms | 2.75 req/s | 8.3x |

Hard ceiling from prefill alone, at infinite batch: **6.74 req/s**.

**Headline prediction: 1.65 req/s at max_batch 8, a 5.0x improvement on Phase 1.**

### Confidence, honestly

The `B=8` row is the one I trust -- it interpolates rather than extrapolates from the
step 3a measurement. **The 16 and 32 rows assume the padding tax scales linearly in
active rows, which rests on a single data point.** If the tax is really a function of
buffer length alone rather than rows x buffer, B=32 would come in far better than 107.8
ms and capacity would be well above 2.75. If it scales worse than linearly, batching
past 8 could stop paying entirely. Either outcome is informative; I would rather be
wrong here explicitly than quietly interpolate.

### Secondary predictions

- **The knee sits just below capacity.** Sweeping rates 0.5 / 1.0 / 1.5 / 2.0 at
  max_batch 8, p95 TTFT should stay flat through 1.5 and depart sharply at 2.0.
- **ITL should be roughly flat across offered load**, unlike TTFT. Phase 1's ITL was
  also flat, but for the opposite reason -- it was serialized and never batched. Here it
  is flat because the batch absorbs load until it saturates. Same shape, different cause,
  which is worth stating because the Phase 1 chart looks identical.
- **TTFT will be worse than Phase 1 at low load.** Phase 1 served one request instantly
  at 195 ms when idle. This server makes a request wait for the next admission and then
  a shared prefill. Expect idle TTFT near 400-600 ms. **Throughput was bought with
  latency**, and at low load the baseline genuinely wins.
- **Client disconnects must not leak slots.** If capacity degrades across successive
  bench runs on the same server process, the cancellation path is broken.

### ACTUALS -- 2026-08-21, step 3b server measured with unchanged bench.py

`tools/bench.py` ran against the new engine with **no modifications**, which is the
wire-compatibility invariant paying off across a complete engine rewrite.

#### Single stream (closed loop, 5 requests)

| | Phase 1 baseline | engine server | |
|---|---|---|---|
| client TTFT p50 | 343.2 ms | **298 ms** | 13% better |
| ITL p50 | 44.5 ms | **41.1 ms** | matches step 1's bare loop (40.5) |
| decode | 22.5 tok/s | **24.3 tok/s** | |

**Prediction S-b1 WRONG.** I predicted idle TTFT of 400-600 ms, worse than Phase 1,
reasoning that a request must wait for an admission cycle the baseline does not have.
True in isolation -- but I counted only what the engine ADDS and forgot what it REMOVES:
the global lock, the `TextIteratorStreamer` handoff, the asyncio queue, the SSE encode.
Those cost more than admission does. Net 45 ms better, not 100-250 ms worse.

This also kills a tidier story I was ready to tell. "Throughput was bought with latency"
is false at single stream, where the engine wins on both axes. The trade only appears
under load, where ITL climbs 41 -> 58 ms.

#### Capacity sweep (open loop, Poisson, 60 s per point)

| offered | achieved | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 |
|---|---|---|---|---|---|
| 0.5 | 0.45 | 315 ms | 336 ms | 45 ms | 50 ms |
| 1.0 | 0.90 | 335 ms | 1,454 ms | 52 ms | 201 ms |
| 1.5 | 1.24 | 336 ms | 1,593 ms | 55 ms | 205 ms |
| 2.0 | **1.59** | 12,446 ms | 23,350 ms | 58 ms | 213 ms |
| 2.5 | **1.62** | 14,136 ms | 31,399 ms | 58 ms | 213 ms |

**Capacity 1.6 req/s against 1.65 predicted -- within 3%.** Throughput saturates at
1.59-1.62 and TTFT explodes, which is the definition of the knee.

#### Against Phase 1, like for like

| | Phase 1 | engine | |
|---|---|---|---|
| capacity | 0.332 req/s | **1.6 req/s** | **4.8x** |
| TTFT p95 at 0.5 req/s | 25,122 ms | **336 ms** | **75x better** |
| ITL p50 at capacity | 44.5 ms | 58 ms | 30% worse |

The 75x TTFT figure is the one that matters and it is not a throughput number at all.
At 0.5 req/s Phase 1 was already 50% past its own capacity and queueing catastrophically;
the engine is at a third of capacity and barely notices. **Capacity improved 4.8x, and
the latency AT a fixed useful load improved by nearly two orders of magnitude.**

#### Phase 2 target: met

The target set on 2026-08-21 was "beat 0.33 req/s without ITL exceeding ~55 ms."

    1.24 req/s sustained at ITL p50 55 ms   -- 3.7x, exactly at the ITL budget
    1.60 req/s at ITL p50 58 ms             -- 4.8x, marginally over budget

Both readings beat the throughput target. The honest one to quote is 1.24 req/s at
55 ms, since that is the point that satisfies the constraint as written.

#### What remains, with numbers attached

    prefill-only ceiling                       6.74 req/s
    measured capacity                          1.60 req/s   (24% of it)
    padding tax to be reclaimed by paging      27.4 us per buffer token, ~36%
    ITL p95 201-213 ms under load vs 50 idle   admission stalls, chunked prefill's target

Step 4 (paged allocator) attacks the padding tax. Phase 3's chunked prefill attacks the
admission stalls that produce the 4x ITL p95 inflation and the 6.74 req/s ceiling.

---
## 2026-08-22 — Phase 2, step 4: decompose the padding tax before building the allocator

Step 3a measured a tax of 27.4 microseconds per buffer token, 9.4x what the KV-read
model predicts, and it is the largest remaining inefficiency. Step 4 is meant to attack
it with a paged allocator. **Whether that can work depends on which of two causes it is,
and they point in opposite directions:**

| cause | does paging fix it? |
|---|---|
| KV reads over padded regions | **yes** -- blocks are allocated per sequence, nothing padded is stored or read |
| SDPA's masked path being slower than its causal fast path | **no** -- ragged rows need a mask however the KV is stored |

Ragged sequence lengths always require either an attention mask or a custom kernel that
consumes a block table directly. If the tax is the mask, then PagedAttention's speed
benefit is inseparable from the CUDA kernel vLLM wrote for it, and a block allocator on
stock HF buys memory without buying throughput. That would be a negative result worth
having, and it would reframe step 4 rather than cancel it.

### The experiment

Hold the buffer length, batch size and cache contents FIXED and vary only the mask.
Everything else identical -- same 8 rows, same prefill, same buffer:

    A  attention_mask=None                        implicit causal, SDPA fast path
    B  attention_mask=ones(B, L)                   explicit mask that masks nothing
    C  attention_mask with real zeros (left-pad)   explicit mask that masks

### Predictions

| condition | predicted ITL | reasoning |
|---|---|---|
| A no mask | **~45 ms** | matches step 2's static_batch at this shape |
| B all-ones mask | **~45 ms** | the step 3a isolation run measured 44.7 vs static's 44.6 |
| C mask with zeros | **~61 ms** | the step 3a full run, where padding was present |

**If C is much slower than B, the tax is the masked kernel path and paging cannot
recover it on this stack.** If B and C are both slow and A is fast, the isolation run
was measuring something else and my step 3a conclusion needs revisiting. If all three
are equal, the tax is neither and I have been wrong about the mechanism twice.

### ACTUALS -- 2026-08-22, mask decomposition

| condition | predicted | measured p50 | vs A |
|---|---|---|---|
| A no mask (implicit causal) | ~45 ms | **43.89 ms** | 1.00x |
| B explicit mask, all ones | ~45 ms | **43.94 ms** | 1.00x |
| C explicit mask, real zeros | ~61 ms | **54.60 ms** | **1.24x** |

All three predictions correct in direction and ordering. C came in below the 61 ms
guess because only 4 of 8 rows were padded here, against nearly all of them in step 3a.

**An explicit attention mask that masks nothing is free -- 43.94 against 43.89 ms. A mask
containing real zeros costs 10.7 ms at this shape.** The tax is SDPA's masked kernel
path, not KV reads over padded regions.

### This determines what step 4 can be, and the answer is uncomfortable

A paged allocator **cannot recover this tax on stock HF**, for two independent reasons:

1. **Ragged rows need a mask however the KV is stored.** Paging changes where KV lives,
   not the fact that eight rows of different lengths must be attended in one rectangular
   call. The mask, and its cost, survive the change.
2. **A gather-based paged design would not even save peak memory.** To feed SDPA you must
   materialise a dense `[B, heads, max_len, dim]` tensor from the block table every step.
   The pool would be smaller, but that transient is exactly the size of today's shared
   buffer, so peak allocation is unchanged while a full-working-set copy is added.

The fix that does work is **varlen attention** -- FlashAttention's `cu_seqlens` interface
packs ragged sequences with no padding and no mask, which is what vLLM pairs with its
PagedAttention kernel. Checked on the box: `flash_attn` and `xformers` are both absent,
and torch 2.13+cu132 is new enough that a matching prebuilt wheel is unlikely, leaving a
30-60 minute source build at $1.21/hr with real failure risk.

**Conclusion: PagedAttention's speed benefit is inseparable from the CUDA kernel written
to consume block tables directly.** The allocator is not the clever part; the kernel is.
That is a genuinely useful thing to have learned by measurement rather than by reading it
in the vLLM paper, and it is the correct answer to "why not just write your own engine".

### Phase 2 closing position

`PROJECT.md` said Phase 2 exists so that "the 60% gap becomes your syllabus -- every
later technique answers a question you generated." Three questions were generated, each
with a measured number attached:

| question | measured | what answers it |
|---|---|---|
| Why is a masked decode step 1.24x a causal one? | +10.7 ms at 412 buffer | varlen attention (Phase 3, vLLM) |
| Why does every admission stall the whole batch? | ITL p95 201-213 ms vs 50 idle | chunked prefill (Phase 3) |
| Why does capacity cap at 1.6 of a possible 6.74 req/s? | prefill is 15.9% of wall and rising with B | prefix caching + chunked prefill (Phase 3) |

Recommendation recorded: **do not build a paged allocator whose benefit has been measured
at zero on this stack.** Carry these three questions into Phase 3 and attribute vLLM's
advantage to them quantitatively. Building it anyway would be defensible as an exercise,
but it would be an exercise, not an optimisation, and the notes should say so either way.

---
## 2026-08-25 — Phase 3, baseline: vLLM with defaults

Same box, same `bench.py`, same 412-token prompt and 64 output tokens as Phase 1 and as
our own engine. That invariant is the only reason these three numbers sit on one axis.

### Derivation

Decode, using the additive model validated in step 2 to within 6%, and the measured
constants `mem_eff` 0.674 and `compute_eff` 0.364:

| batch | t_mem | t_cmp | ITL | decode-only capacity |
|---|---|---|---|---|
| 8 | 41.9 ms | 2.9 ms | 44.8 ms | 2.79 req/s |
| 16 | 43.3 ms | 5.8 ms | 49.1 ms | 5.09 req/s |
| 32 | 46.1 ms | 11.6 ms | 57.7 ms | 8.66 req/s |
| 64 | 51.6 ms | 23.3 ms | 74.9 ms | 13.35 req/s |

Prefill, from our own measured sweep (3,209 tokens in 952 ms = **3,370 tok/s** at large
batch): a 412-token prompt costs **122 ms** of GPU work, so the prefill-only ceiling is
**8.18 req/s**. Note that is higher than the 6.74 req/s ceiling our engine faced, because
our engine prefilled one admission batch at a time at 148 ms per request while vLLM packs
prefill tokens more efficiently.

Concurrency is capped by KV, not by the scheduler: at `--gpu-memory-utilization 0.9`
there are 3.60 GiB of KV room, which is 26,200 tokens, or **55 concurrent requests** at
476 tokens each.

Combining, at the B=32 to B=64 range that a 55-sequence cap implies:

    B=32   capacity 4.21 req/s   prefill 51% of wall
    B=64   capacity 5.07 req/s   prefill 62% of wall

### Predictions

| # | Quantity | Predicted | Confidence |
|---|---|---|---|
| V1 | Capacity, vLLM defaults | **4.0-4.8 req/s**, central estimate **4.4** | Medium |
| V2 | vs our engine (1.60 req/s) | **2.5-3.0x** | Medium |
| V3 | vs Phase 1 (0.332 req/s) | **13-14x** | Medium |
| V4 | ITL p50 at capacity | **60-80 ms**, HIGHER than our 58 ms | Medium |
| V5 | Concurrent sequences at saturation | **~55**, KV-limited not scheduler-limited | High -- pure arithmetic |
| V6 | Single-request TTFT | **250-350 ms**, comparable to our 298 ms | Low |

**V4 is the one worth stating plainly: vLLM's ITL should be WORSE than ours, not better.**
Not because it is slower per unit work, but because it runs a far larger batch. Higher
batch means more KV read per step, so each individual token is slower while total
throughput is much higher. If vLLM shows both higher throughput AND lower ITL, my model
of where its advantage comes from is wrong and needs revisiting.

`PROJECT.md` predicted before Phase 2 began that a hand-written engine would reach "maybe
40% of vLLM's throughput". At 1.60 req/s that implies vLLM at 4.0 req/s -- independently
consistent with the arithmetic above, which is mild reassurance that both are sane.

### What must be recorded, not assumed

Recent vLLM enables **chunked prefill and prefix caching by default**, where older
versions did not. If the defaults already include them, then "vLLM baseline" is not a
clean comparison against our engine, which has neither -- and the ablation must DISABLE
them to isolate their contribution rather than enabling them.

**Read the actual configuration out of the server's startup log and record it before
interpreting any number.** Assuming the defaults would silently make every attribution in
Phase 3 wrong.

### Install constraint

Our engine's venv has `torch 2.13.0+cu132`. vLLM pins its own torch build. vLLM gets a
SEPARATE venv at `/opt/llm/.venv-vllm`; upgrading in place would break `engine/` and
destroy the ability to re-measure our own engine for comparison.

### Correction, written before the sweeps: the defaults are not comparable

vLLM 0.27.1 ships with **`enable_prefix_caching=True` and `enable_chunked_prefill=True`**,
read out of the engine's own startup log rather than assumed. Our engine has neither, so
"vLLM defaults vs our engine" is not a like-for-like comparison, and the V1 prediction of
4.0-4.8 req/s implicitly assumed prefill still cost something.

`bench.py` sends a byte-identical 412-token prompt on every request, which is the
perfect-hit case for a prefix cache. Measured at single stream:

| | TTFT p50 | ITL p50 |
|---|---|---|
| identical prompts (cache hits) | **190 ms** | 34.0 ms |
| `--unique-prefix` (cache misses) | **293 ms** | 34.0 ms |
| our engine | 298 ms | 41.1 ms |

ITL is unchanged at 34.0 ms either way, confirming prefix caching affects prefill only.
With the cache defeated, vLLM's TTFT equals ours to within 2%.

**So vLLM's single-stream advantage decomposes into two independent parts:**

    TTFT advantage   entirely prefix caching        190 vs 293 ms
    ITL advantage    entirely kernel efficiency     34.0 vs 41.1 ms

### The ITL number revises a Phase 2 conclusion

vLLM decodes at 34.0 ms against our bare offline loop's 40.5 ms -- **16% faster at batch
1, where there is no batching or scheduling advantage at all.** Back-solved against the
27.30 ms roofline floor:

| stack | `mem_eff` achieved |
|---|---|
| Phase 1 server | 0.623 |
| our bare offline loop | 0.674 |
| **vLLM** | **0.803** |

Phase 2 concluded that 0.674 was "a property of the stack, not the card", and that
Phase 1's 0.623 was server overhead. **That was half right.** A further 16% was sitting
in per-step Python: cache bookkeeping, mask concatenation, argmax, the `.item()` sync.
vLLM captures a decode step as a CUDA graph and replays it in one launch.

This is a THIRD axis, separate from everything Phase 2 measured. Phase 2 attacked
scheduling -- how many requests share one weight read. This is kernel efficiency -- how
much of the hardware a single step uses. Our engine could have been ~16% faster at every
batch size without touching its scheduler.

### Revised sweep predictions

With prefix cache hits, prefill approaches zero and capacity becomes decode-bound. Using
vLLM's measured `mem_eff` of 0.803 and the ~70-sequence KV cap (33,424 tokens / 476):

    cached    B=70, ITL ~70 ms, no prefill cost   ->  ~15 req/s
    uncached  B=70, plus 122 ms prefill per req   ->  ~5.4 req/s

| # | Quantity | Predicted |
|---|---|---|
| V7 | Capacity, defaults, identical prompts | **12-16 req/s** |
| V8 | Capacity, defaults, `--unique-prefix` | **4.5-6 req/s** |
| V9 | Ratio between them | **~2.8x**, entirely attributable to prefix caching |
| V10 | ITL p50 at saturation | **65-80 ms** both ways, since decode is unaffected by caching |

**V8 is the honest headline number** -- it is the one that describes traffic where users
send different prompts. V7 describes a benchmark artifact and should never be quoted
without its caveat.

### ACTUALS -- 2026-08-25, vLLM 0.27.1 baseline

| # | Quantity | Predicted | Measured | |
|---|---|---|---|---|
| V5 | KV cache tokens | 26,200 | **26,176** then **33,424** | see reproducibility note |
| V7 | Capacity, identical prompts | 12-16 req/s | **18.6 req/s** | missed high |
| V8 | Capacity, `--unique-prefix` | 4.5-6 req/s | **5.7 req/s** | correct |
| V9 | Ratio between them | ~2.8x | **3.26x** | close |
| V10 | ITL p50 at saturation | 65-80 ms | **77 ms** cached, 56 ms uncached | half correct |

#### The ladder, all measured on one box with one unchanged benchmark client

| stage | capacity | vs previous | vs Phase 1 |
|---|---|---|---|
| Phase 1 naive server | 0.332 req/s | -- | 1.0x |
| our engine (Phase 2) | 1.60 req/s | 4.8x | 4.8x |
| **vLLM, unique prompts** | **5.70 req/s** | **3.6x** | **17.2x** |
| vLLM, identical prompts | 18.6 req/s | 3.3x | 56x |

`PROJECT.md` guessed before Phase 2 began that a hand-written engine would reach "maybe
40% of vLLM's throughput". **We reached 28%** -- same ballpark, slightly worse, and the
missing 72% is now attributable rather than mysterious.

#### Why V7 missed: prefix caching saves KV MEMORY, not only prefill compute

I predicted the cached case from compute alone -- prefill goes to zero, so capacity
becomes decode-bound at the ~70 concurrent sequences the KV budget allows. That was
incomplete. When every request shares a byte-identical 412-token prefix, vLLM stores
those blocks **once** and points every sequence at them. Only the generated tokens are
unique:

    KV budget                        33,424 tokens
    concurrency WITHOUT sharing         70 sequences   (476 tokens each)
    concurrency WITH shared prefix     516 sequences   (64 unique tokens each)
                                                       -> 7.3x more fit in the same memory

Little's Law on the measurement (18.62 req/s, ~5 s mean end-to-end) implies **~93
concurrent requests in flight** -- comfortably past the 70 that unshared KV would allow,
and direct evidence the blocks are being shared.

**So prefix caching has two distinct effects, and I had only modelled one.** It removes
prefill compute, AND it multiplies effective concurrency by deduplicating KV. On a
workload with a long shared system prompt, the second effect may matter more than the
first.

#### Reproducibility hazard worth recording

The identical launch command produced **26,176 tokens** of KV on one start and **33,424**
on the next -- a 28% difference. vLLM sizes its cache by profiling free GPU memory at
startup, so whatever transient allocations exist during that profile change the budget.

**Consequence for the ablations: a capacity difference between two configurations is not
attributable to the flag until their KV sizes are confirmed equal.** Every ablation run
must have its KV cache size read out of its own startup log and recorded alongside its
result, or the phase will attribute to a flag what was really a startup accident.

#### vLLM has a tail too

At 6 req/s uncached, ITL p50 56 ms against **p95 405 ms**. Chunked prefill reduces
admission stalls; it does not remove the fact that prefill work competes with decode for
the same GPU. Our engine's equivalent was p50 58 / p95 213 ms at its own saturation.

#### Instrument note

`bench.py` flagged the 24, 28 and 32 req/s cached points as **invalid** -- 134, 301 and
more requests dropped against the 512 max-inflight cap. Those points are excluded rather
than reported. A saturated open-loop generator that silently drops arrivals reports a
throughput ceiling that is really a client limit.

---
## 2026-08-25 — Phase 3 ablations: attribute the 3.6x

Baseline established: vLLM 5.70 req/s uncached, 18.6 req/s with cache hits, against our
engine's 1.60. Now isolate which technique earns which part of that.

### Design: the workload has to match the flag under test

Ablating prefix caching under `--unique-prefix` measures **nothing** -- there are no
shared prefixes to hit, so on and off are identical. That is not a wasted run; it is the
control that proves the harness measures what I think it does. Each flag is therefore
paired with the workload where it can act:

| run | config | workload | isolates |
|---|---|---|---|
| A | `--no-enable-prefix-caching` | unique prefixes | **control** -- expect no change |
| B | `--no-enable-chunked-prefill` | unique prefixes | chunked prefill |
| C | `--no-enable-prefix-caching` | identical prompts | prefix caching |
| D | `--kv-cache-dtype fp8` | unique prefixes | KV quantisation |

### Predictions

| # | Run | Predicted capacity | Reasoning |
|---|---|---|---|
| A1 | prefix caching off, unique | **5.5-5.9 req/s, unchanged** | nothing to cache. If this moves, the harness or my model is wrong |
| A2 | chunked prefill off, unique | **4.5-5.2 req/s**, a 10-20% drop | prefill can no longer share a step with decode, so each one blocks |
| A3 | chunked prefill off, ITL p95 | **600-1000 ms**, up from 405 | this is the admission-stall mechanism our own engine suffered |
| A4 | prefix caching off, identical | **5.5-5.9 req/s**, collapsing from 18.6 | proves the 18.6 was entirely cache hits |
| A5 | fp8 KV, unique | **5.7-6.5 req/s**, barely changed | KV per token halves 144 -> 72 KiB, roughly doubling concurrency to ~140. But capacity here is PREFILL-bound at 5.7, not memory-bound, so more concurrency buys little |

**A5 is the interesting one.** The obvious expectation is that halving KV doubles
throughput. It should not, because the binding constraint is prefill compute, and fp8 KV
does nothing for compute. If fp8 gives a large gain, my claim that this workload is
prefill-bound is wrong.

**A4 is the strongest check.** If disabling prefix caching on identical prompts does NOT
collapse capacity to the uncached number, then something other than caching is producing
the 18.6 and the baseline write-up needs correcting.

### Mandatory per-run record

Every run records its **KV cache size from its own startup log**. The identical command
produced 26,176 and then 33,424 tokens on consecutive starts, so a capacity difference is
not attributable to a flag until the KV budgets are confirmed comparable.

### ACTUALS -- 2026-08-25, Phase 3 ablations

| config | workload | req/s | vs base | KV tokens | ITL p95 |
|---|---|---|---|---|---|
| baseline, both flags on | unique | 5.79 | 1.00x | 33,424 | 456 ms |
| A `--no-enable-prefix-caching` | unique | 6.05 | **1.04x** | 33,424 | 513 ms |
| B `--no-enable-chunked-prefill` | unique | 5.66 | **0.98x** | 25,952 | 612 ms |
| D `--kv-cache-dtype fp8` | unique | 6.44 | **1.11x** | 52,368 | 481 ms |
| baseline, both flags on | identical | 18.62 | 1.00x | 33,424 | 101 ms |
| C `--no-enable-prefix-caching` | identical | 6.38 | **0.34x** | 33,424 | 513 ms |

| # | Predicted | Measured | |
|---|---|---|---|
| A1 | 5.5-5.9, unchanged | **6.05, +4%** | correct -- control passes |
| A2 | 4.5-5.2, a 10-20% drop | **5.66, -2%** | **wrong** -- far less impact on throughput |
| A3 | ITL p95 600-1000 ms | **612 ms** | correct |
| A4 | 5.5-5.9, collapsing from 18.6 | **6.38** | correct |
| A5 | 5.7-6.5, barely changed | **6.44** | correct |

#### The control passed, which makes the rest trustworthy

Disabling prefix caching on unique prompts changed capacity by +4% -- within noise, and
if anything slightly faster, since maintaining cache metadata costs a little when it
never hits. Nothing to cache, no effect. The harness measures what it claims to.

#### Chunked prefill buys tail latency, not throughput

I predicted a 10-20% capacity drop and got 2%. The prediction was wrong about *what the
flag does*. Capacity barely moved, but **ITL p95 went 456 -> 612 ms, 34% worse**, which
is precisely the admission-stall mechanism our own engine suffered from. Chunked prefill
splits a long prefill across steps so it stops blocking everyone's decode. That is a
latency intervention. It does not create GPU throughput that was not there.

Caveat recorded: run B's KV budget was 25,952 tokens against the baseline's 33,424, a
22% difference from vLLM's startup memory profile. Since this workload is prefill-bound
rather than KV-bound, that should not move capacity much -- and the D result below
supports it -- but the two runs are not perfectly matched and the number carries that
asterisk.

#### fp8 KV confirms the workload is prefill-bound

fp8 raised the KV budget 1.57x, from 33,424 to 52,368 tokens, and bought **+11% capacity**.
The naive expectation is that halving KV per token roughly doubles throughput. It does
not, because the binding constraint here is prefill compute, and fp8 KV does nothing for
compute. A5 predicted "barely changed, 5.7-6.5"; measured 6.44.

#### Prefix caching is worth 2.9x -- on a workload built to flatter it

18.62 -> 6.38 req/s when disabled on identical prompts. That confirms the entire cached
baseline was cache hits. It is a real capability, and on production traffic with a long
shared system prompt it would matter. On this benchmark it measures an artifact we
constructed by holding the prompt constant.

Note 6.38 uncached-identical slightly exceeds 5.79 uncached-unique: the identical-prompt
runs still share tokenisation and allocator behaviour even without caching.

### The headline: the famous flags explain almost none of the gap

    our engine                                   1.60 req/s
    vLLM on realistic traffic                    5.79 req/s      3.6x

    explained by chunked prefill                 ~2%
    explained by prefix caching                  ~0%   (nothing to cache)
    UNEXPLAINED by either flag                   3.5x

**On traffic where users send different prompts, vLLM's advantage is not its two most
famous scheduling features.** It is the kernels and the core engine: FlashAttention's
varlen path (no padding, no mask -- the exact tax we measured at 1.24x in Phase 2 step 4),
CUDA graphs replacing per-step Python (`mem_eff` 0.803 against our 0.674), torch.compile,
and a paged allocator that packs far more sequences into the same memory.

That is the answer to the question Phase 2 was designed to generate. We predicted the
gap would be explained by chunked prefill and prefix caching, because those are the
techniques with names. **Measurement says the gap is mostly in the parts with no
marketing: kernel quality and memory packing.**

---
## 2026-08-25 — Phase 3 ablations, part 2: the two scheduler knobs

Defaults read from `SchedulerConfig` rather than assumed: **`max_num_batched_tokens=2048`,
`max_num_seqs=128`** (KV caps effective concurrency near 70 regardless).

### `--max-num-batched-tokens` — the prefill chunk size

This workload is prefill-bound, so this is the knob most likely to move capacity. The
mechanism is our own `compute_eff` curve from Phase 2: prefill efficiency rises with the
number of tokens in a single forward pass, because a small GEMM cannot fill the tensor
cores. Bigger chunks mean fewer, fatter prefill passes.

Interpolating that measured curve, and calibrating so the 2048 row reproduces the
measured 5.79:

| budget | implied `compute_eff` | prefill work/request | predicted capacity |
|---|---|---|---|
| 1,024 | ~0.40 | 135 ms | **~5.1 req/s** (-12%) |
| 2,048 (default) | ~0.47 | 115 ms | 5.79 measured |
| 8,192 | ~0.53 | 102 ms | **~6.3 req/s** (+9%) |

Secondary: **larger chunks should make ITL p95 worse**, because a fatter prefill pass
occupies a step that decodes are waiting on. This is the same tradeoff chunked prefill
exists to manage, seen from the other side.

### `--max-num-seqs` — the concurrency cap

This is vLLM's version of our engine's `max_batch`, so it traces the throughput-versus-ITL
tradeoff directly. Using vLLM's measured `mem_eff` of 0.803:

| cap | predicted ITL | predicted capacity |
|---|---|---|
| 16 | ~42 ms | **~3.5 req/s** |
| 32 | ~50 ms | **~4.6 req/s** |
| 64 | ~67 ms | **~5.5 req/s** |
| 128 (default, KV-limited to ~70) | 56-90 ms | 5.79 measured |

**The shape is the point, not the individual numbers.** Capacity should rise and ITL
should worsen together, monotonically, because they are the same tradeoff measured two
ways. If capacity saturates while ITL keeps climbing, the extra concurrency is buying
nothing and the KV ceiling is binding instead.

| # | Prediction |
|---|---|
| A6 | `max-num-batched-tokens` 1024 -> **~5.1**, 8192 -> **~6.3** |
| A7 | Larger batched-token budget makes **ITL p95 worse** |
| A8 | `max-num-seqs` 16/32/64 -> **3.5 / 4.6 / 5.5 req/s**, monotonic |
| A9 | ITL falls monotonically as the cap falls: **~42 / 50 / 67 ms** |

### ACTUALS -- 2026-08-25, `--max-num-seqs` (clean)

| cap | KV tokens | effective batch | ITL p50 | measured | model | predicted |
|---|---|---|---|---|---|---|
| 16 | 27,280 | 16 | 40 ms | **3.65** | 3.55 | 3.5 |
| 32 | 18,816 | 32 | 42 ms | **5.01** | 4.85 | 4.6 |
| 64 | 18,560 | **39** (KV-capped) | 46 ms | **5.35** | 5.06 | 5.5 |
| 128 default | 33,424 | **70** (KV-capped) | 56 ms | **5.79** | 5.78 | -- |

A8 predicted 3.5 / 4.6 / 5.5; measured 3.65 / 5.01 / 5.35. A9 predicted ITL 42/50/67;
measured 40/42/46 -- shape right, magnitudes lower.

One formula reproduces every row, and nails the baseline to two decimals:

    R = 1 / (0.122 prefill  +  (64 / B) * ITL)

**`--max-num-seqs 64` never ran 64 sequences.** Its KV budget held only 39
(18,560 / 476). Effective concurrency is `min(max_num_seqs, KV_tokens / tokens_per_request)`
-- the flag is a ceiling, not a floor, and KV was binding. Reading that row as "64
sequences" would have been wrong, and the model only fits once the effective value is used.

**Diminishing returns, explained rather than observed.** 16 -> 32 buys 1.37 req/s;
39 -> 70 buys 0.44. Decode cost per request falls as concurrency rises, but the ~122 ms
of prefill does not move. The asymptote is 1/0.122 = **8.2 req/s regardless of batch size**.

### `--max-num-batched-tokens` was confounded, and the first re-run failed on my bug

Raw results, before correction:

| budget | KV tokens | effective batch | capacity | implied prefill |
|---|---|---|---|---|
| 1,024 | 17,840 | 37 | 5.19 | 114 ms |
| 2,048 | 33,424 | 70 | 5.79 | 122 ms |
| 8,192 | 25,504 | 54 | 5.69 | 117 ms |

KV budgets differ by 87% across configs that should vary only in prefill chunk size, and
the capacity ordering matches the KV ordering. Backing prefill cost out of the model
gives **114 / 122 / 117 ms -- flat within noise**, suggesting the spread is effective
concurrency rather than the flag. That is an inference, not a measurement.

Re-run with KV pinned via `--kv-cache-memory-bytes`. Two failures worth recording:

- **My harness bug.** `set -- $cfg` did not word-split as intended, so `$2` was empty and
  vLLM got `--max-num-batched-tokens` with no value. All three configs died identically,
  which is the signature of a harness fault rather than a config problem. Fixed by
  calling the driver explicitly instead of unpacking a loop variable.
- **vLLM's own hint is stale.** Its startup log suggests
  `Replace gpu_memory_utilization config with --kv-cache-memory=...`, but the actual flag
  is `--kv-cache-memory-bytes`. Following the tool's advice verbatim fails.

### ACTUALS -- 2026-08-25, `--max-num-batched-tokens` with KV pinned

`--kv-cache-memory-bytes 2576980378` gives every config an identical 17,472-token budget,
removing the startup-profile variance that made the first attempt uninterpretable.

| budget | KV tokens | capacity | ITL p50 |
|---|---|---|---|
| 1,024 | 17,472 | **5.12** | 39-45 ms |
| 2,048 | 17,472 | **5.09** | 39-45 ms |
| 8,192 | 17,472 | **5.18** | 39-45 ms |

**A6 and A7 are both WRONG.** A6 predicted 5.1 -> 6.3 across the range, a 24% spread;
measured spread is **1.8%** across an 8x change in chunk size. A7 predicted larger chunks
would worsen ITL p95; ITL is flat.

#### Why: the budget is never filled

My reasoning was that a bigger chunk means a fatter prefill GEMM, and our Phase 2
`compute_eff` curve says fatter GEMMs are more efficient. The curve is right. The premise
is not -- there is never enough queued prefill work to fill even the smallest budget:

    arrival rate                  5.0 req/s
    prefill tokens arriving       2,060 tok/s
    decode step                   45 ms  ->  22 steps/s
    prefill tokens per step       93 on average

    budget 1,024   utilised  9.1%   headroom 11.0x
    budget 2,048   utilised  4.5%   headroom 22.1x
    budget 8,192   utilised  1.1%   headroom 88.4x

**The smallest budget tested is already 11x larger than the work available.** Raising a
ceiling nothing is touching cannot change anything. The flag would bind at much higher
arrival rates, or with prompts long enough that a single prefill exceeds the budget and
must be split -- which is precisely the case chunked prefill exists for, and precisely
the case this workload does not produce.

This is the same class of error as reading `--max-num-seqs 64` as "64 sequences". **Both
flags are ceilings. A ceiling only matters when something is pressing against it**, and
in both cases the binding constraint was elsewhere -- KV there, arrival rate here.

The correct way to have predicted this was to check the utilisation of the budget before
predicting the effect of changing it. The arithmetic above takes one line and would have
turned A6 from a wrong prediction into a correct one.

---
## 2026-08-26 — Phase 3 CLOSED

Deliverable: latency-vs-throughput curves for nine configurations, published as a chart.
`tools/curve.py` collapses the raw per-request JSONL into curve points, recomputing
throughput over the observed span rather than trusting bench.py's printed summary.

### Final attribution, realistic (unique-prompt) traffic

| technique | capacity effect | note |
|---|---|---|
| prefix caching | **0%** | 2.9x on shared prompts; nothing to cache otherwise |
| chunked prefill | ~2% | but **34% better ITL p95** -- a latency fix, not a throughput one |
| fp8 KV cache | **+11%** | 1.57x the KV budget; workload is prefill-bound |
| `--max-num-batched-tokens` | **0%** | never binding, 9% utilised at the smallest setting |
| **unexplained by any flag** | **3.5x** | kernels and core engine |

### The ladder

    Phase 1 naive server     0.332 req/s
    our engine               1.60  req/s    4.8x
    vLLM, unique prompts     5.70  req/s    3.6x over ours, 17.2x over Phase 1
    vLLM, identical prompts 18.6   req/s    prefix cache hits -- benchmark artifact

### Predictions scored

| # | Predicted | Measured | |
|---|---|---|---|
| V5 | 26,200 KV tokens | 26,176 / 33,424 | correct then invalidated by startup variance |
| V7 | 12-16 req/s cached | 18.6 | missed high -- prefix caching also saves KV MEMORY |
| V8 | 4.5-6 req/s unique | 5.7 | correct |
| V10 | ITL 65-80 ms at saturation | 77 cached / 56 unique | half correct |
| A1 | prefix caching off = no change | +4% | correct -- **the control passed** |
| A2 | chunked prefill off = -10-20% | -2% | **wrong about what the flag does** |
| A3 | ITL p95 600-1000 ms | 612 ms | correct |
| A4 | cached collapses to ~5.5-5.9 | 6.38 | correct |
| A5 | fp8 barely changes capacity | +11% | correct |
| A6 | chunk size 1024 vs 8192 = 24% | 1.8% | **wrong -- budget never filled** |
| A8 | max-num-seqs 3.5/4.6/5.5 | 3.65/5.01/5.35 | correct |

Eight correct, three wrong. The three misses share one shape: **I predicted the effect of
raising a ceiling without first checking whether anything was pressing against it.**
`--max-num-seqs 64` was capped at 39 by KV; `--max-num-batched-tokens 1024` ran at 9%
utilisation; prefix caching had nothing to cache. One line of arithmetic on utilisation
would have caught all three before predicting.

### What Phase 2 generated and Phase 3 answered

| question from Phase 2 | answer |
|---|---|
| Why is a masked decode step 1.24x a causal one? | varlen attention removes it -- part of the unexplained 3.5x |
| Why does every admission stall the batch? | chunked prefill; worth 34% on ITL p95, ~0% on capacity |
| Why cap at 1.6 of a possible 6.74 req/s? | prefill-bound. vLLM's own asymptote is 8.2 req/s for the same reason |

### Carried into Phase 4

- **fp8 KV cache is already measured at +11%** on this workload. Phase 4 covers weight
  quantisation, which is a different axis, and must measure the third thing neither phase
  has touched: **output quality**.
- vLLM's KV budget varies 28% between identical launches. Pin `--kv-cache-memory-bytes`
  for every Phase 4 comparison; `--gpu-memory-utilization` is not reproducible.
- This workload is prefill-bound at 412 prompt / 64 output. Quantisation results will
  differ on a decode-heavy workload, and Phase 4 should measure at least one of each.

---

# Phase 4 -- quantization

Written 2026-08-28, box stopped, no Phase 4 measurement taken. Protocol in
`NOTES/phase4-eval-design.md`.

## The model used for every throughput prediction below

GPU time per request = prefill + this request's share of decode.

    prefill(P)  = 2 N P / (peak_flops x compute_eff)
                  N = 8,190,735,360   peak = 125e12   compute_eff = 0.36 @512, 0.50 @4096

    t_step(B)   = W_bytes / (BW x mem_eff)      BW = 600e9, mem_eff = 0.803 (vLLM, Phase 3)
                  bf16  W = 16.39e9 B  ->  34.0 ms
                  fp8   W =  8.19e9 B  ->  17.0 ms, call it 22 ms after Marlin dequant
                  int4  W =  5.0e9  B  ->  10.4 ms, call it 16 ms after Marlin dequant

    decode      = out_tokens x t_step(B) / B

sm86 has no native fp8, so both quantized formats dequantize into bf16 tensor cores. The
dequant penalties above (17->22, 10.4->16) are the least defensible numbers here and are
the first thing to check against measurement.

## P4-1  Weight quantization on the PHASE 3 workload (512 / 64)

    prefill      = 2 x 8.19e9 x 512 / (125e12 x 0.36)        = 0.186 s
    B in flight  = 17 (Little's Law on the Phase 3 measurement)
    decode bf16  = 64 x 0.034 / 17                           = 0.128 s
    total                                                    = 0.314 s  -> 3.19 req/s

    fp8   decode = 64 x 0.022 / 17 = 0.083, prefill +10% = 0.205, total 0.288 -> 1.09x
    int4  decode = 64 x 0.016 / 17 = 0.060, prefill +15% = 0.214, total 0.274 -> 1.15x

**Predicted: fp8 +5 to +20%, int4 +10 to +30% on the Phase 3 workload.**

CORRECTION to `phase4-eval-design.md` section 7 as first drafted, which said "roughly 0%".
That is right for *KV* quantization and wrong for *weight* quantization, and doing the
arithmetic is what caught it. Two different mechanisms are in play and only one of them is
idle here:

- the **KV-capacity** mechanism does nothing, because KV is 29% utilised and B is not
  KV-bound. This is the prefix-caching lesson exactly.
- the **decode-bandwidth** mechanism still works, because every decode step reads all the
  weights regardless of how full the KV cache is.

The model says the second is worth only ~10% here because decode is 41% of the request's
GPU time at 64 output tokens. A large measured gain on this workload would mean the
bandwidth model is wrong, not that quantization is better than expected.

## P4-2  Weight quantization on the CAPACITY-PRESSURE workload (4096 / 1024)

    prefill      = 2 x 8.19e9 x 4096 / (125e12 x 0.50)       = 1.073 s
    tokens/req   = 5120
    B(bf16)      = 33,424 / 5120 = 6.5   -> 6
    decode bf16  = 1024 x 0.034 / 6                          = 5.80 s
    total                                                    = 6.87 s  -> 0.146 req/s
                                                                decode is 84% of the work

    fp8   B = 87,000/5120 = 17;  decode = 1024 x 0.022/17 = 1.33; prefill 1.18; total 2.51
    int4  B = 106,000/5120 = 20; decode = 1024 x 0.016/20 = 0.82; prefill 1.23; total 2.05

**Predicted: fp8 2.5-3.0x, int4 3.0-3.5x.** Both mechanisms are live here -- B rises
because KV rooms opens, and each step is cheaper because there are fewer weight bytes.

The gap between P4-1 and P4-2 is the entire thesis of the phase. Same technique, same
hardware, same model: **1.1x or 2.7x depending only on which workload it is measured on.**

## P4-3  fp8 KV stacked on fp8 weights (Q3), 4096 / 1024

fp8 KV gave 1.57x the KV budget in Phase 3. B rises 17 -> 26, decode 1.33 -> 0.87 s,
total 2.51 -> 2.05 s. **Predicted +20% over Q1 on the capacity workload**, against the +11%
measured on 512/64 -- larger because here KV is actually the binding constraint.

## P4-4  The noise floor is itself a dose-response curve

`d0` is the bf16-vs-bf16 discordance. A maths answer is a single integer at the end of a
chain; one flipped token anywhere upstream changes it completely. Longer chain, more
opportunities for batch-composition reassociation to change a token.

**Predicted d0: ~1% at k=2, ~2% at k=4, ~4% at k=8, ~6-8% at k=16.**

If this holds it is the most useful thing the control produces, because it means **a naive
eval that skipped the control would report roughly 7 points of "quantization damage" at
k=16 that is nothing but bf16 disagreeing with itself.** That is the specific fiction this
design exists to prevent.

## P4-5  Quality

| | prediction |
|---|---|
| fp8 excess discordance over `d0` | under 2 points at every k, not significant -> **adopt** |
| int4 excess over `d0` | ~1 point at k=2 rising to ~8 at k=16 -> **fails rule 2 at k=16** |
| does excess rise monotonically with k? | **yes for int4, no for fp8** -- this is the amplification claim |
| int4 mean thinking tokens vs bf16 | **+10 to +25%** -- less decisive, longer chains |
| int4 truncation rate | under 2x bf16, so not disqualified on rule 3 |
| Q3 (fp8 KV) on T3 longctx | 2-5 points worse than Q1, roughly **flat across depth** -- the error accumulates with total sequence length, not with where the needle sits |

## P4-6  Base accuracy, needed to confirm the items are calibrated

The eval is worthless at a ceiling or a floor. Target band 60-85% at the hardest level.

| k | thinking ON | thinking OFF |
|---|---|---|
| 2 | 99% | 90% |
| 4 | 97% | 70% |
| 8 | 90% | 35% |
| 16 | **70%** | 10% |

If k=16 with thinking on comes back above 90%, the dose curve has no headroom at the top
and `k` must be extended to 24 or 32 before spending anything on the quantized runs. That
check is the first GPU task of the phase and it is deliberately cheap.

## What would falsify the design

- `d0` near zero at every k -> the control is not exercising batch-composition variance,
  probably because concurrency 32 is not being reached. Check achieved concurrency before
  believing any quality number.
- fp8 measuring 2.5x on the 512/64 workload -> the bandwidth model is wrong by 2x and every
  roofline prediction in this project inherits the error.
- int4 showing *less* damage than fp8 -> a checkpoint mismatch, not a real result.

## P4-7  GSM8K as an instrument check (added 2026-08-28, before running)

200 items from the published GSM8K test set were added to the item file. Their job is not
statistical power -- the paired design already has that -- but to catch a broken harness,
which synthetic items structurally cannot do. See `phase4-eval-design.md` section 3a-bis.

**Predicted bf16, thinking ON: high 80s to mid 90s percent.** GSM8K is considered saturated
for current reasoning models in this size class, and Qwen3-8B is certainly trained on it.

CAVEAT, and it limits how hard this check can be leaned on: **I do not have a verified
published GSM8K figure for Qwen3-8B to hand.** The Qwen3 card leads with AIME, MATH and
LiveCodeBench, because GSM8K stopped discriminating between good models years ago. The band
above is inferred from the model class, not read off a table, and section 1 of `CLAUDE.md`
says not to quote a remembered number. **Look up the real figure on the model card before
treating any deviation as a harness bug.**

How to read the result:

| bf16 GSM8K, thinking on | what it means |
|---|---|
| 85-95% | pipeline validated -- chat template, thinking toggle, extraction, grading all sane |
| 60-85% | suspicious. Check thinking is actually on and that `max_tokens` is not truncating |
| under 60% | **harness is broken.** No synthetic number from the same run is believable |
| above 98% | check the answer is not leaking into the prompt, and that grading is not matching too loosely |

**Predicted thinking OFF: 60-80%**, a much larger drop than the synthetic k=2 items will
show, because GSM8K problems need two to four real steps rather than one bookkeeping
operation.

Secondary prediction, and the one that would be worth something: **fp8 and int4 should show
LESS damage on GSM8K than on synthetic k=16.** GSM8K chains are short -- two to four steps
against sixteen -- so if the amplification claim in section 5b is right, the dose is lower
here. If GSM8K instead shows *more* damage than k=16, the synthetic items are measuring
something narrower than reasoning and the dose curve does not generalise. That is the single
most useful thing this slice can tell us beyond the instrument check.

## P4-8  Calibration run, predicted before executing (2026-08-28, box up)

Server: vLLM 0.27.1, bf16, `--max-model-len 6144`, `--max-num-seqs 12`. 120 maths items
sampled across all four k, thinking on, `max_tokens` deliberately set to 4096 so the true
token distribution is observable rather than clipped.

**Thinking-token count.** A k-step chain needs roughly one short paragraph of working per
step, and Qwen3 tends to re-verify at the end:

| k | predicted completion tokens p50 |
|---|---|
| 2 | 250 |
| 4 | 450 |
| 8 | 800 |
| 16 | **1,400**, p99 around 2,800 |

If p99 at k=16 lands above 3,000 then the design's placeholder `max_tokens 2048` would have
truncated roughly a third of the hardest slice and read it as reasoning failure. That is the
single thing this run exists to prevent.

**Base accuracy, thinking on**, repeating P4-6: 99 / 97 / 90 / **70** percent for k = 2 / 4 /
8 / 16. The eval needs the hardest level inside 60-85 percent. Above 90 and the dose curve
has no headroom, so `k` extends to 24 or 32 and the item file is regenerated before any
quantized run.

**Determinism.** `check-determinism` must pass. Qwen3's `generation_config.json` sets
temperature 0.6 / top_p 0.95, and whether the request's `temperature: 0` overrides it in
vLLM 0.27.1 is genuinely unknown. If two identical greedy requests differ, every quality
number from this server is noise and the phase stops until it is fixed.

**Thinking transport.** Unknown whether vLLM splits `<think>` into `reasoning_content`
without `--reasoning-parser` set. `qualeval.py` handles both and records which, so this run
answers it rather than assuming.

## P4-9  Measured at launch, 2026-08-28: KV budget at the Phase 4 server settings

    vLLM 0.27.1, bf16, --max-model-len 6144 --max-num-seqs 12
    GPU KV cache size: 27,280 tokens        (read from the server's own log)

**This revises the arithmetic in P4-2, which used Phase 3's 33,424.** That figure was
measured at `--max-model-len 4096`; raising the limit to 6144 costs KV, because vLLM reserves
against the longest sequence it must be able to serve.

    tokens per request at 4096/1024 = 5120
    bf16 concurrency = 27,280 / 5120 = 5.3      (was predicted 6.5)

The direction of P4-2 is unchanged and the gap widens slightly: a smaller bf16 batch means
quantization has more headroom to recover, so **fp8 on the capacity workload should now beat
the predicted 2.5-3.0x rather than fall short of it.** Recording the revision here rather
than editing P4-2, per the append-only rule.

Observed KV budgets across this project so far, all the same model on the same GPU:

| max_model_len | other flags | KV tokens |
|---|---|---|
| 4096 | Phase 3 defaults, launch 1 | 26,176 |
| 4096 | Phase 3 defaults, launch 2 | 33,424 |
| 4096 | later ablation launches | 17,472 |
| 6144 | `--max-num-seqs 12` | **27,280** |

A 2x spread. Any comparison that does not read this number from each run's own log is
uninterpretable, which is why the design pins it per configuration and never across.

**Also settled from the config, without an experiment:** `reasoning_parser=''`. vLLM has no
reasoning parser configured, so Qwen3's `<think>` block arrives inline in `content` rather
than split into `reasoning_content`. `qualeval.py` handles both and records which path it
took; the calibration run confirms it empirically.

## P4-10  Early signal from the determinism probe, before calibration finished

The determinism check runs one **k=2** item with `max_tokens=512`. The server reported it
finishing with `finished_reason="length"` -- it exhausted 512 tokens without reaching an
answer.

P4-8 predicted **250 tokens p50 at k=2**. The easiest level in the whole item set is already
past double that. If k=2 needs more than 512, the predicted 1,400 at k=16 is going to be far
too low, and the design's placeholder `max_tokens 2048` would have truncated much more than
the hardest slice.

This is exactly what the calibration run exists to catch, and it is being caught before any
quantized configuration was launched rather than after.

**A tension this exposes, worth carrying into the rest of the phase:** thinking tokens and KV
capacity fight each other directly. Raising `max_model_len` to fit longer reasoning shrinks
the KV cache (P4-9: 6144 costs about 6,000 tokens of cache against 4096), which lowers
concurrency, which lowers throughput. A thinking-heavy workload is therefore doubly expensive
-- each request occupies KV longer *and* the server can hold fewer of them. That is the
Phase 7 thesis appearing as an operational constraint in Phase 4.

## P4-11  Calibration results, 2026-08-28. Two instrument bugs and one wrong design target.

### The determinism gate PASSED

    2 sequential requests, same item, batch 1: lengths 1939 / 1939, identical: True

`temperature: 0` **does** override Qwen3's `generation_config.json` (temperature 0.6,
top_p 0.95) in vLLM 0.27.1. This was the single most damaging open unknown in the phase: had
it failed, sampling noise would have swamped every effect and no quality number from this
server would have meant anything. Run this gate first, every session.

Thinking transport confirmed empirically: `think_path` = `inline_tags` on 119 of 120, i.e.
`<think>` arrives inside `content`, as `reasoning_parser=''` implied. The one exception was a
truncated response that never emitted `</think>`.

### BUG 1: 16% of correct answers were graded wrong

    acc 84.2%   unparseable 15.8%   truncated 0.8%
    parsed AND wrong: 0 of 120

Those two lines together are the diagnosis. **Every answer the model actually stated was
correct**, and the entire 15.8% "failure" was the extractor missing an answer that was
present. 18 of the 19 unparseable responses had `finish_reason == "stop"`, so they were not
truncated -- they finished, correctly, in a format the grader did not read.

The cause: Qwen3 is trained to close a maths answer with `\boxed{}` and does so even when the
prompt demands `ANSWER: <integer>`:

    ### Final Answer
    $$
    \boxed{3}
    $$

**Why this would have been fatal rather than merely annoying.** Quantization changes output
formatting. A configuration that reaches for `\boxed{}` slightly more often would have scored
lower for a reason with nothing to do with reasoning quality, and the effect would have been
indistinguishable from the damage the phase exists to measure. It would not have raised an
error; it would have produced a plausible, wrong, confidently-reported number.

Fixed by accepting `\boxed{}` as a second explicit marker, taking whichever marker appears
LAST, and recording which one matched per item so a configuration that changes convention is
visible rather than silently mis-scored. Measured split after the fix: **103 `answer`, 16
`boxed`, 1 neither.**

This is not the loose fallback section 2c forbids. `\boxed{}` is an explicit answer
declaration, structurally identical to `ANSWER:`. Hunting for a bare integer in prose remains
forbidden, and `The answer is 42.` still yields nothing.

### BUG 2: the no-thinking slice measured its own token cap

    k=2  90.9% acc,   0.0% truncated
    k=4  66.7%       14.3%
    k=8   9.3%       83.7%
    k=16  0.0%      100.0%

`max_tokens 256` was a guess, and at k=16 it truncated every single item. That slice measured
the ceiling, not the model. Raised to 1024 for the real runs.

### The thinking-token distribution, which is what calibration was for

| k | p50 | p95 |
|---|---|---|
| 2 | 1,060 | 1,997 |
| 4 | 954 | 3,160 |
| 8 | 1,230 | 2,145 |
| 16 | 1,590 | 2,789 |

P4-8 predicted 250 tokens p50 at k=2 and 1,400 at k=16. **Measured 1,060 and 1,590.** Wrong
by 4x at the easy end and roughly right at the hard end, because the model has a large fixed
thinking overhead and adds only modestly per step: 8x the reasoning steps costs 1.5x the
tokens, not 8x.

The design's placeholder `max_tokens 2048` would have truncated about a third of k=16. Real
runs use **5120** for thinking, which also leaves headroom for the new k=32 level, and 1024
for no-thinking.

### The design target of 60-85% base accuracy was WRONG, and that is a correction to my own reasoning

After fixing bug 1, accuracy with thinking on is **100 / 95.2 / 100 / 95.6 percent** at
k = 2 / 4 / 8 / 16, and every remaining miss is a truncation or a missing marker rather than
a wrong answer. Qwen3-8B does not make arithmetic errors on 16 chained integer operations
when allowed to think.

`phase4-eval-design.md` section 3b required base accuracy inside 60-85% "or the dose curve has
no headroom". **That requirement was imported from standard evaluation and does not apply to
this design.** In a standard eval you compare two models on absolute accuracy, and a ceiling
hides differences. Here the comparison is paired against the same model: `b` counts items
bf16 got right and the quantized model got wrong, so **a reference at 100% is the most
sensitive possible configuration** -- every quantization error is pure signal with no
dilution from items the reference already failed.

The real risk is not a ceiling, it is a task so easy that quantization cannot break it
either, which returns `b = 0` and an ambiguous null. Mitigated by keeping a spread of
difficulty rather than by lowering accuracy: **k regenerated as 4 / 8 / 16 / 32**, weighted
toward the top (45 / 75 / 105 / 135), k=2 dropped as pure ceiling with no information, and
two other item families (gsm8k, longctx) carried as independent evidence.

**And the no-thinking slice is where the headroom actually lives.** Thinking-on is at the
ceiling; thinking-off collapses with chain length. That makes T2 the slice most able to show
damage and T1 the slice that tests the amplification claim, which is close to the opposite of
what the design assumed.

## P4-12  Cost control, and the detection limit it buys (decision 2026-08-28)

The quality runs pin `--max-num-seqs 12` so every configuration decodes at the same batch
size (design 3d). That pin is what makes the comparison valid, and it is also what makes the
runs slow: **measured 249 tok/s at B=12**, against the 850 tok/s the design assumed at B=32.
The control costs 3.4x the wall clock. That tension was not costed when the design was
written.

    measured per configuration, all four passes, 360 maths + 200 gsm8k:   ~65 min
    five configurations:                                          5.5 hr, about $6.60

`math/think` alone is 39 of those 65 minutes: 360 items at roughly 1,600 tokens each. It is
also the pass sitting at a 100% ceiling. **Cut to 180 items**, which halves the dominant cost
and leaves the other three passes untouched.

### What that costs in statistical power, stated plainly

With the reference at 100%, `c` is approximately zero and every quantization error lands in
`b`, so the exact McNemar reduces to `p = 2 / 2^b`.

| true error rate | expected `b` at n=180 | p |
|---|---|---|
| 2% | 3.6 | 0.125, not significant |
| 3% | 5.4 | 0.06, marginal |
| 4% | 7.2 | 0.016, significant |
| 6% | 10.8 | 0.001 |

**The detection floor on `math/think` at n=180 is roughly a 4% error rate.** Section 5b puts
int4 at 1-3% average and fp8 under 1%, so this pass can resolve int4 damage and **cannot**
resolve fp8 damage.

That is an acceptable trade only because it is written down before the run: **a null result
on `math/think` for fp8 must be reported as "no damage detectable above a 4% floor", never as
"no damage".** The slices that carry the fp8 question are `math/nothink` (360 items, and the
only slice with real headroom -- accuracy collapses with chain length there) and `gsm8k`
(200 items, both conditions).

### bf16-a was restarted rather than allowed to finish

The first bf16-a launch was 155 records into a 360-item thinking pass when the cut was
decided. It was killed and relaunched at 180 rather than kept, discarding about 23 minutes of
GPU (roughly $0.46).

Reason: a 360-item pass and a 180-item pass are **not the same run condition.** Queue depth
over time and the tail drain differ, batch composition follows from both, and by incident 22
batch composition changes the numerics. Keeping the longer reference would have put that
asymmetry inside `d0`, the very quantity that is supposed to isolate it. Paying $0.46 to
delete a confound that would otherwise need arguing away is the right trade in a phase whose
entire discipline is that every configuration runs identically.

## P4-13  Extraction bug 2, found mid-run and fixed offline (2026-08-28)

bf16-a's `math/think` pass reported 7.0% unparseable at k=16 and 3.2% at k=32, with
`parsed AND wrong: 1` of 180. Inspecting the saved text showed four of the six had a correct
answer plainly present:

    ### Final Answer:
    ANSWER: 72

The regex was `ANSWER\s*:\s*([^\n]*)` with `IGNORECASE`. Two mistakes compounding:

1. `IGNORECASE` makes it match `Answer:` inside the heading **"### Final Answer:"**.
2. `\s` matches newlines, so `\s*` after that colon swallowed the line break, and `([^\n]*)`
   then captured the entire NEXT line as the answer -- the literal string `"ANSWER: 72"`.

`normalize` correctly refused that as a non-integer, so the item scored unparseable while a
correct answer sat one line below. **The grader was defeated by the model agreeing with it.**

Fixed to `ANSWER[ \t]*:[ \t]*([^\n]+)`: horizontal whitespace only, and at least one
character required on the same line. The heading now matches nothing and the real marker
wins. Verified against the four real completions, all four recover, and three regression
cases were added to `selftest` (heading followed by the marker, bold heading with a blank
line, bare heading alone must still yield nothing). 35 cases pass.

### Why this cost no GPU time, by design

Design section 2c required the full completion text be written for every item, on the
grounds that "a grading bug is inevitable and should not cost GPU time to fix". **That
decision has now paid for itself twice in one session** -- the `\boxed{}` bug in calibration
and this one -- both diagnosed and fixed against saved output while the GPU carried on with
the next run.

### CONSEQUENCE: bf16-a and bf16-b will be graded by different code

bf16-a is running with the pre-fix grader loaded in memory; bf16-b will launch with the fix.
Their stored `correct` fields are therefore not comparable, and **the `compare` step at the
end of the chained script will report a wrong `d0`.**

The fix is not to touch the running job. It is to run `qualeval.py grade` over BOTH files
before comparing, which re-scores from saved text with one version of the code and costs
nothing. Recorded here so the chain's own compare output is not mistaken for the real floor.

## P4-14  THE PHASE 4 RESULT, measured 2026-08-28/29

### Throughput: the same technique, two workloads

| config | KV tokens | conc @5120 | 512/64 | ratio | 4096/1024 | ratio |
|---|---|---|---|---|---|---|
| bf16 | 26,176 | 5.1 | 6.01 req/s | 1.00x | 0.26 req/s | 1.00x |
| fp8 | 73,264 | 14.3 | 6.45 req/s | **1.07x** | 0.53 req/s | **2.04x** |
| int4 w4a16 | 95,648 | 18.7 | 6.95 req/s | **1.16x** | 0.61 req/s | **2.35x** |

**Quantization is worth 2x more on one workload than the other, from workload choice alone.**
Run this phase on Phase 3's workload -- the obvious, lazy choice -- and the conclusion is
"quantization is nearly worthless", confidently wrong by a factor of two.

### Scorecard

| # | predicted | measured | |
|---|---|---|---|
| P4-1 small, int4 | +10-30% | +16% | correct |
| P4-1 small, fp8 | +5-20% | +7% | correct |
| P4-2 big, fp8 | 2.5-3.0x | 2.04x | **missed low** |
| P4-2 big, int4 | 3.0-3.5x | 2.35x | **missed low** |
| fp8 KV tokens | 81,638 | 73,264 | -10% |
| fp8 capacity, anchored on int4 | 0.454 req/s | 0.53 req/s | -14% |
| fp8/int4 ratio | 0.907 | 0.869 | **correct to 4%** |

### Why both P4-2 predictions missed low, one cause

**KV traffic per decode step scales with batch size, so concurrency self-limits.**

    config  B     weights   KV read/step   bytes/step
    bf16    5.1   16.39 GB     3.47 GB      19.86 GB
    fp8    14.3    8.19 GB     9.71 GB      17.90 GB
    int4   18.7    6.12 GB    12.71 GB      18.83 GB

Every extra sequence adds roughly 0.68 GB of KV reads per step. At 4,608 tokens of context
the KV term OVERTAKES the weight term, so bytes-per-step barely fall and the step time stays
near 40 ms for all three. The gain is therefore not "cheaper steps" but only "the same 40 ms
divided across more requests". P4-2 assumed concurrency converts cleanly into throughput; it
does not, and the correction is the same shape as incident 17 -- a term that was assumed
independent is not.

**The ratio survived what the absolutes did not.** fp8/int4 was predicted 0.907 and measured
0.869, correct to 4%, while both absolute predictions were ~25% off. The estimate errors
(`compute_eff`, `mem_eff`, the Marlin penalties) are shared between configurations and cancel
in a ratio. **Anchor on a measured configuration and predict ratios; never trust the absolute
from a roofline with three fudge factors in it.**

### Why the fp8 KV prediction missed 10%

Predicted from `params x 1 byte = 7.628 GiB`. Real fp8 weights are larger: per-tensor scale
factors are stored alongside, and vLLM leaves embeddings, `lm_head` and norms in bf16. The
0.8 GiB residual is exactly that. int4 predicted far better because its weight size was taken
from the **measured on-disk checkpoint** (5.7 GiB), not from a theoretical bits-per-param.
Measure the artifact; do not derive it.

### Quality, against the d0 noise floor

| | accuracy | McNemar p | answer drift | vs d0 (1.8%) |
|---|---|---|---|---|
| bf16-b (control) | 88.4% | 0.51 | 1.8% | -- |
| fp8 | 89.3% | 0.84 | 6.1% | 3.4x |
| int4 | 86.8% | 0.0501 | 7.7% | 4.3x |

**Damage as a fraction of what bf16 got right**, which is the number the aggregate hides:

| slice | fp8 | int4 |
|---|---|---|
| math/think, all k | 1% | **1%** |
| math/nothink k4 | 0% | **0%** |
| math/nothink k8 | 3% | **3%** |
| math/nothink k16 | 0% | **9%** |
| math/nothink k32 | 11% | **37%** |

0 -> 3 -> 9 -> 37 percent. **The dose-response curve exists and is monotonic in chain
length.** At k=32 int4 destroys more than a third of what bf16 solved, while the aggregate
reports a 2.1 point drop.

### THE FINDING THAT CONTRADICTS PROJECT.md SECTION 5b

Section 5b, the project's central quality hypothesis:

> "Thinking amplifies it -- this is the project-critical one. A reasoning chain is 1,000+
> sequential tokens where each conditions the next, so small per-token errors compound."

**Measured: the opposite.** With thinking ON, int4 breaks 1% of what bf16 got right. With
thinking OFF, at the same chain length, it breaks 37%.

Thinking does not amplify quantization damage. It **absorbs** it -- given room to reason the
model catches and repairs its own perturbed arithmetic; denied that room the errors propagate
straight to the answer.

Caveat, stated rather than buried: `math/think` sits at 98-100% and so has less room to show
damage than `nothink` at 42.9%. But the measure above is already damage per opportunity, and
1-of-179 against 10-of-27 is not a ceiling artifact.

**Consequence for Phase 7:** quantization and thinking budget are coupled. An int4 model may
need to think to stay accurate, which spends the KV that quantization just freed. The
scheduler cannot treat "which quantization" and "how many thinking tokens" as independent
knobs.

### Token cost, and a second wrong prediction

P4-5 predicted int4 would think 10-25% LONGER. Measured **-3.8%**, slightly shorter. It was
**fp8** that ran long, at **+13.5%** on math/think, which was predicted for neither.

### Pre-registered decision rules, applied honestly

- **fp8**: accuracy rule passes (p=0.84). Token rule **fails** -- 13.5% against a 10% bar.
  Reported as: no accuracy cost, measurable output drift at 3.4x the floor, and a 13.5% token
  tax on reasoning that must be netted against its 2.04x capacity win.
- **int4**: rule 2 as written tests k=8, where excess over d0 is 2.5 points, under the 5-point
  bar -- **it passes as written**. But the rule was written before calibration created k=32,
  where the excess is 28.5 points. Reported as failing in spirit, passing in letter, with both
  stated. The goalposts are not moved in either direction after the fact.

### Detection limits, so the nulls are not overread

`math/think` at n=180 with the reference at ~100% has a floor near a 4% error rate. **fp8's
null on that slice means "no damage detectable above 4%", not "no damage".**

## P4-15  longctx: a clean negative result

90 needle-in-document items per configuration, target fact at 10/50/90 percent depth with two
same-shape distractors, `--max-num-seqs 6` pinned, KV pinned at 27,280 for all three.

    bf16   90/90  100.0%    d10 30/30   d50 30/30   d90 30/30
    fp8    90/90  100.0%    d10 30/30   d50 30/30   d90 30/30
    int4   90/90  100.0%    d10 30/30   d50 30/30   d90 30/30

**270 of 270. Weight quantization does not damage long-context retrieval at all**, at any
depth, down to int4.

This sharpens the phase's quality finding from "int4 hurts quality" to something far more
specific and far more useful:

> **int4 damages multi-step arithmetic reasoning and leaves retrieval untouched.**

`PROJECT.md` section 5b names four axes where damage concentrates -- maths, code,
long-context retrieval, non-English. Two are now measured on this model: **maths yes,
retrieval no.** The blanket claim does not survive contact with the measurement.

Caveat stated rather than buried: this task is retrieval of a verbatim 4-character string,
which is the easy end of long context. A harder variant -- synthesising across several
retrieved facts, or reasoning over them -- would likely behave like the maths slice, because
that is reasoning wearing retrieval's clothes. What is established is that **finding** the
right span is robust; what is not established is that **using** it is.

**Direct consequence for Phase 6.** A web-search turn is mostly retrieval over long context,
which is the part quantization does not break. The exposure is whatever reasoning happens
after the retrieval. That argues for int4 on the retrieval-heavy path and caution about it on
the reasoning-heavy one -- which is a scheduling decision, and therefore Phase 7's problem.

## P4-16  Phase 4 closed. Carried into Phase 5.

**Scorecard: 9 predictions correct, 6 wrong.** The six misses, and what each cost:

| miss | predicted | actual | root cause |
|---|---|---|---|
| P4-2 capacity, both configs | 2.5-3.5x | 2.04x / 2.35x | KV traffic per step scales with batch, so concurrency self-limits |
| P4-4 noise floor vs chain length | rises with k | 0% at every k on the thinking slice | the floor tracks proximity to the model's competence limit, not chain length |
| P4-5 int4 thinks longer | +10-25% | -3.8% | wrong config entirely; fp8 was the one at +13.5% |
| P4-6/P4-8 base accuracy | 70% at k=16 | ~100% at k=32 | the model does not make arithmetic errors when allowed to think |
| P4-8 thinking tokens | 250 at k=2 | 1,060 | large fixed thinking overhead; 8x the steps costs 1.5x the tokens |
| section 5b amplification | thinking amplifies damage | thinking absorbs it | the central quality hypothesis of the project, inverted |

**The most valuable thing the phase produced is the method correction, not the numbers:**
predict RATIOS anchored on a measured configuration. fp8/int4 was predicted 0.907 and
measured 0.869, correct to 4%, while both absolute predictions were 25% off. The shared
estimate errors cancel.

### Carried into Phase 5 (speculative decoding)

- **Measure on both workloads, always.** The single most transferable result here is that
  512/64 and 4096/1024 disagree by 2x about the same technique. Speculative decoding is
  known to *hurt* at high batch; that claim is meaningless without naming the workload.
- **The noise floor is cheap and mandatory.** `d0` cost one extra 33-minute run and made
  every later number interpretable. Any Phase 5 quality claim needs the same control.
- **Instruments lie quietly.** Three grading bugs this phase (`\boxed{}`, the `Final
  Answer:` newline, the 22% ReadError), none of which raised an exception and all of
  which produced plausible numbers. Budget time to validate the instrument against known
  ground truth before trusting a single measurement from it.
- **Run the client on the box.** Recorded in `CLAUDE.md` section 3 as incident 25.
- **Quantization and thinking budget are coupled.** int4 is nearly free with thinking on
  and destroys 37% of 32-step arithmetic with it off. A scheduler cannot treat "which
  quantization" and "how many thinking tokens" as independent knobs, which is a Phase 7
  constraint that Phase 4 discovered by accident.
- **Open:** does the retrieval result survive a task that requires *using* several
  retrieved facts rather than recalling one verbatim? 270/270 establishes that finding a
  span is robust, not that reasoning over it is.

---

# PHASE 5 — speculative decoding

Protocol in `NOTES/phase5-spec-design.md`, written 2026-08-29 before the box came up.
Every prediction below is recorded before any Phase 5 measurement exists.

## The model used for every Phase 5 prediction

Same constants as Phase 4, so the two phases compose.

    N        = 8,190,735,360 params
    BW       = 600e9 B/s        mem_eff = 0.803   (vLLM, measured Phase 3)
    PEAK     = 125e12 FLOP/s    compute_eff: SEE BELOW, this is the weak number
    KV/token = 147,456 B        (measured Phase 2, confirmed two ways)
    ctx      = 4608             mean context on the 4096/1024 workload

    W(bf16) = 16.39e9 B    W(fp8) = 8.19e9 B    W(int4) = 6.12e9 B   (P4-14 measured)

Speculative decoding: draft proposes k, target verifies S = k+1 positions per step.

    t_mem(B) = (W + B*ctx*147456) / (BW * mem_eff)
    t_cmp(B) = 2*N*S*B / (PEAK * compute_eff)
    E[tokens/step] = (1 - a^(k+1)) / (1 - a)

**The weak number, named before it can be blamed afterwards: `compute_eff` for the verify
GEMM.** `roofline.py` measured compute_eff on *prefill*: 0.297 at 113 tokens, 0.364 at
412, 0.53 at 6407, because a short GEMM cannot fill the tensor cores. A verify pass at
B=5, S=4 is **20 tokens**, far to the left of the shortest length ever measured on this
box. This is the Marlin-dequant-penalty of Phase 5, and incident 18 says a constant
measured in one regime is not a constant in another. Predictions below therefore bracket
compute_eff over 0.10-0.30 and lean on RATIOS, per the P4-16 method correction.

## P5-0  The premise: how much idle compute is there

    batch-1 decode: 2 x 8.19e9 FLOP in 34.0 ms = 482 GFLOP/s
    against 125,000 GFLOP/s peak              = 0.385% utilised

**The compute units are 99.6% idle during decode.** This is not a prediction, it is the
Phase 3 measurement re-read. It is recorded here because it is the entire reason the
phase exists, and because if a later measurement contradicts it the phase is pointless.

## P5-A  The VRAM ledger — draft weights are paid out of the KV cache

Derived 2026-08-29 from HuggingFace configs, before download.

EAGLE3 head `RedHatAI/Qwen3-8B-speculator.eagle3`, param count from its `config.json`:

    embed_tokens   151936 x 4096                     =   622,329,856
    eagle3 fc      (3 x 4096) x 4096                 =    50,331,648
    1 llama layer  attn 41,943,040 + mlp 150,994,944 =   192,937,984
    draft lm_head  32000 x 4096                      =   131,072,000
                                                        -----------
                                                        996,671,488  (0.997 B)
    bf16                                             = 1,993,342,976 B
    actual file                                      = 2,044,116,968 B  -> 1.904 GiB

The 2.5% delta is norms plus the d2t/t2d vocab-mapping buffers. Reproducing the file size
from the config to 2.5% is the check that the architecture is understood.

**The head costs MORE VRAM than a real draft model** -- 1.904 GiB against Qwen3-0.6B's
1.400 GiB -- because it carries the target's full 151936-row input embedding. "One layer"
describes its FLOPs, not its footprint. I expected the opposite before doing the division.

Against Phase 4's measured KV budgets:

| config | KV now | minus 1.904 GiB | change | conc @5120 |
|---|---|---|---|---|
| bf16 | 3.595 GiB / 26,176 tok | 1.691 GiB / 12,313 tok | **-53.0%** | 5.11 -> 2.40 |
| fp8 | 10.061 GiB / 73,264 tok | 8.158 GiB / 59,401 tok | -18.9% | 14.31 -> 11.60 |
| int4 | 13.135 GiB / 95,648 tok | 11.232 GiB / 81,785 tok | **-14.5%** | 18.68 -> 15.97 |

**Phase 4 said memory freed is capacity gained. Phase 5 says memory spent is capacity
lost, and the same fixed 1.904 GiB costs 53% of bf16's concurrency and 15% of int4's.**
Quantization is what makes speculative decoding affordable. Neither phase could make that
claim alone.

## P5-B  Crossover batch B*, where the technique inverts sign

Solve t_cmp(B) > t_mem(B):

| config | S=2 | S=4 | S=6 |
|---|---|---|---|
| bf16, ce 0.10 | 29 | 9 | 6 |
| bf16, ce 0.20 | >200 | 29 | 14 |
| bf16, ce 0.30 | >200 | 101 | 29 |
| int4, ce 0.10 | 11 | 4 | 2 |
| int4, ce 0.20 | >200 | 11 | 6 |
| int4, ce 0.30 | >200 | 38 | 11 |

Absolute B* spans an order of magnitude across the compute_eff bracket, so it is NOT
predicted as a number. Two things survive, and they are what gets scored:

- **int4's B\* is one third to one half of bf16's, at every compute_eff.** compute_eff
  cancels in the ratio. This is P4-16's method correction applied deliberately.
- **Larger k crosses over earlier**, so the throughput-optimal k FALLS as load rises. A
  fixed k is the wrong policy -- a Phase 7 input generated by Phase 5.

Where Phase 4's measured concurrencies land: bf16 at 4096/1024 runs B=5.11 (inside the
win zone at every compute_eff); int4 runs B=18.68 (**past B\* for S=4 unless compute_eff
exceeds ~0.25**). Predicted: the two configurations disagree about the SIGN of the effect
on the same workload.

## P5-1 .. P5-9  Scored predictions

| # | quantity | prediction | basis |
|---|---|---|---|
| P5-1 | EAGLE3 acceptance `a`, non-thinking prose | 0.65 - 0.80 | EAGLE3 papers report 0.7-0.8 on chat; discount for an unfamiliar workload |
| P5-2 | EAGLE3 ITL p50 speedup at B=1, bf16 | 1.8 - 2.4x | E[tokens] at a=0.7, k=3 is 2.53; verify step is memory-bound so ~free; minus draft and sampling overhead |
| P5-3 | Qwen3-0.6B acceptance `a` | 0.45 - 0.60, clearly below EAGLE3 | a token-only draft has no access to the target's hidden state |
| P5-4 | n-gram acceptance, `copy` slice | 0.7+ | the answer quotes the document verbatim; this is prompt-lookup's best case |
| P5-4b | n-gram acceptance, `reason` and `open` | under 0.15 | no repeated n-grams to find in novel text |
| P5-5 | **EAGLE3 acceptance, `reason` vs `open`** | **HIGHER on reason, 0.75 - 0.88** | see below |
| P5-6 | crossover measurable on int4 4096/1024 | yes, B* in 8 - 20 | P5-B at compute_eff 0.15-0.30 |
| P5-7 | bf16 + EAGLE3 capacity, 4096/1024 | **drops 35 - 50%** | P5-A: KV falls 53%, partly offset by faster effective decode |
| P5-8 | int4 + EAGLE3 capacity, 4096/1024 | **drops 5 - 15%** | P5-A: KV falls only 14.5% |
| P5-9 | quality drift, spec on vs off, vs `d0`=1.8% | at the floor, under 3% | spec decoding is distribution-preserving by construction; anything higher means broken rejection sampling |

### P5-5 is the phase's real bet, and the one most likely to be wrong

**Predicted: reasoning text is EASIER to draft than ordinary prose, not harder.**

The reasoning: an EAGLE head conditions on the target's own hidden states, and a
chain-of-thought is where the target is most confident about its next token -- restating
the problem, walking a template, repeating intermediate values it just computed. Confident
target, easy draft. P4-8 measured 1,060 thinking tokens spent on a 2-step problem, which
is a lot of low-entropy scaffolding.

The opposing case, which is why this is a bet: reasoning is where the model's genuinely
novel work happens, and a 1-layer head may track the scaffolding while missing exactly the
tokens that carry the computation.

**Why it matters more than the other eight:** P4-8's 1,060 thinking tokens at 34 ms is
**36 seconds of silence** before the user sees a word. Speculative decoding is the largest
single lever against that, and Phase 7 is built on the assumption that thinking latency is
reducible. If P5-5 misses low, Phase 7 inherits a hard constraint instead of a knob, and
it is much better to learn that now than in Phase 7.

Second-order risk on the same prediction: the EAGLE3 head drafts over a **reduced 32000-
token vocabulary**. If digit and operator tokens are underrepresented in that reduction,
acceptance on the maths slice collapses for a reason that has nothing to do with reasoning
and everything to do with vocabulary selection. That would look identical to P5-5 being
wrong. Distinguishing the two requires the per-position acceptance curve, which is why
`specmon.py` reports it.

## P5-C  What would falsify the design

- S0 controls do not reproduce Phase 4's bf16/int4 numbers -> the comparison base moved,
  stop and find out why before any spec number is taken seriously.
- Self-speculation (target as its own draft) does not report acceptance near 1.0 -> the
  instrument or the verify path is broken, and no acceptance number in the phase is real.
- Quality drift exceeds 3% against `d0` -> spec decoding is not distribution-preserving in
  this build, and every latency number describes a different model than the control.
- Spec decoding does not compose with int4 w4a16 in vLLM 0.27.1 -> half the design dies
  and the phase reduces to bf16 only. **Test this in build step 3, not step 7.**

## P5-A1  Matched-pair KV measurement, 2026-08-29 (prediction written before the result)

The Phase 3 trap fired again: the S0-int4 control, launched with Phase 4's exact flags,
reported **102,944 tokens** of KV against Phase 4's **95,648** -- a 7.6% drift from
vLLM's startup memory profiling, with nothing changed but the day. **P5-A's ledger must
therefore be scored against this run's control, not against Phase 4's number.** This is
precisely why the control was run rather than assumed, and it would have shown up as a
spurious 7.6% "EAGLE is cheaper than predicted" had it been skipped.

    control (S0-int4, no spec)   102,944 tokens
    EAGLE3 head                2,044,116,968 B / 147,456 B per token = 13,863 tokens
    PREDICTED S3-int4          102,944 - 13,863 = 89,081 tokens  (-13.5%)
    concurrency @6144          16.76x -> predicted 14.50x

This is close to pure arithmetic -- draft weights displace KV byte for byte -- so a miss
means something else moved: extra activation memory for the draft's forward pass, a
separate CUDA graph capture for the draft, or the head loading at a dtype other than bf16.
Recording the three candidate causes now so the explanation is not invented afterwards.

### ACTUALS -- P5-A1, 2026-08-29. Predicted 89,081 tokens, measured 78,224. MISS by 12%.

| | control S0-int4 | +EAGLE3 S3-int4 | delta |
|---|---|---|---|
| GPU KV cache size | 102,944 tok | **78,224 tok** | **-24.0%** |
| max concurrency @6144 | 16.76x | 12.73x | -24.0% |
| weights + non-torch | 5.98 GiB | 7.92 GiB | **+1.94** |
| peak activation | 0.18 GiB | 1.34 GiB | **+1.16** |
| CUDAGraph memory | 0.62 GiB | 0.96 GiB | **+0.34** |
| total non-KV | 6.78 GiB | 10.22 GiB | +3.44 |

Predicted KV loss 13,863 tokens; measured 24,720. **The head costs 1.78x its own weights.**

**The weight term was right to 2%** -- predicted 1.904 GiB, measured +1.94. The param
count derived from `config.json` was correct and so was the byte-for-byte displacement
argument. **What P5-A missed is that draft weights are not the only new term**, and I had
named activations as a candidate cause before seeing the number, which is the only reason
this is attribution rather than a story:

- **Peak activation +1.16 GiB, a 7.4x increase over the control's 0.18 GiB.** This is 77%
  of the unpredicted cost. Cause: a verify step processes k+1 = 4 positions per sequence
  instead of 1, so the activation buffer covers ~4x the tokens in flight, and the draft
  model runs its own forward pass on top.
- **CUDAGraph +0.34 GiB**, the remaining 23%. vLLM captures graphs for the draft model and
  for the multi-token verify shapes as well as the ordinary decode shape.

**The correction to the model, which changes the phase's economics:**

    draft cost  =  draft weights  +  (k+1) x activation growth  +  extra CUDA graphs
    NOT         =  draft weights

**The activation term scales with k, so the draft's memory cost is a function of k.**
P5-A and section 0b of the design both treated it as a fixed charge independent of k.
It is not, and that couples two knobs the design assumed were separate: raising k to
chase acceptance also raises the memory it costs, which lowers concurrency, which moves
the crossover of P5-B. **k=5 is more expensive in memory than k=3, not just in wasted
compute.** This is Phase 4's coupling lesson again -- P4-16 warned that quantization and
thinking budget could not be treated as independent knobs, and here k and memory cannot
either.

### Consequence to test immediately: bf16 + EAGLE3 may not be viable at all

Phase 4 measured bf16 KV at 3.595 GiB. If EAGLE3's true cost is ~3.4 GiB rather than the
1.9 GiB of its weights, **bf16 + EAGLE3 has essentially no KV cache left.** P5-7 predicted
a 35-50% capacity drop on bf16; the honest revision before measuring is that it either
refuses to start or is left with single-digit concurrency. Recording the revision here
rather than quietly repairing P5-7 after the fact.

## P5-1 ACTUAL, first acceptance measurement, S3-int4, 2026-08-29

512-token synthetic prompt, 128 max out, thinking off, batch 1, k=3.

| quantity | predicted | measured | |
|---|---|---|---|
| acceptance rate `a` | 0.65 - 0.80 | **0.3722** | **MISS, badly low** |
| mean emitted / step `L` | -- | 2.117 | |
| ITL p50 | -- | 17.7 ms | |
| decode | -- | 56.5 tok/s | |

Per-position acceptance, which is the number the scalar hides:

    pos 0   0.6333
    pos 1   0.3500
    pos 2   0.1333

**Acceptance decays steeply with position.** Position 3 contributes 0.133 -- it is drafted
every step and accepted one time in eight. That is the per-position curve doing exactly the
job section 9 said it would: a scalar `a` of 0.37 is consistent with a flat 37% at every
position, which would justify a larger k, and with this steep decay, which says **k=3 is
already past the useful point and k=2 may dominate it.** Same drafted-token budget, very
different conclusion.

The i.i.d. cross-check (section 9c) reports measured/i.i.d. = **1.355**. Acceptance is
positively correlated across positions rather than independent, so the textbook
`(1-a^(k+1))/(1-a)` formula understates real throughput by 35% here. Worth knowing before
using that formula anywhere else in the phase.

### Three candidate causes for the miss, written before testing any of them

1. **The head was trained against bf16 Qwen3-8B and is here drafting for int4 w4a16.**
   EAGLE conditions on the target's hidden states; quantizing the target shifts those
   hidden states away from what the head was trained on. If this is the cause it is the
   most interesting result available in the phase, because it means **quantization and
   speculative decoding interfere with each other through a channel that has nothing to
   do with memory** -- and P5-A already showed they interfere through memory.
2. **The workload is synthetic filler text.** `bench.py`'s prompt is generated tokens, not
   natural language. Drafting out-of-distribution text is hard, and this is the `open`
   slice, the least favourable of the three in design section 5b.
3. **The 32000-token draft vocabulary.** Flagged in P5-5's second-order risk.

Cause 1 is separable from 2 and 3 by a single run: **the same measurement on bf16 + EAGLE3.**
Same head, same prompt, same k -- only the target's precision changes. Running it next.

## P5-2 THE INSTRUMENT WAS WRONG, AND IT INVERTED THE RESULT (2026-08-29)

The first matched batch-1 runs said speculative decoding made everything **slower**:

| config | ITL p50 | "decode tok/s" | reading |
|---|---|---|---|
| S0-int4 control | 11.9 ms | 84.6 | -- |
| S3-int4 EAGLE3 | 17.7 ms | 56.5 | 1.49x SLOWER |
| S0-bf16 control | 34.2 ms | 29.3 | -- |
| S3-bf16 EAGLE3 | 41.5 ms | 24.1 | 1.21x SLOWER |

That is a clean, plausible, fully self-consistent set of numbers, and it is wrong.

**What gave it away was not the latency. It was the token count.** The controls emitted
exactly 2,816 tokens -- 22 requests x the 128 cap, every request truncated. The spec runs
emitted 1,342 and 1,433. At temperature 0, a distribution-preserving technique had
apparently changed how much text the model produced, which is not a performance artifact
but a correctness alarm.

**Cause: `bench.py` counts one token per SSE chunk.** For four phases that was exactly
right -- vLLM emits one chunk per engine step and a step produced one token. Speculative
decoding emits **the whole accepted run in a single chunk**, so `out_tokens` undercounts
by precisely the acceptance factor, and `itls` measures per-STEP, not per-token latency.

The undercount ratios are the acceptance factor itself:

    S3-int4  2816 real / 1342 chunks = 2.098    specmon measured L = 2.117
    S3-bf16  2817 real / 1433 chunks = 1.966    specmon measured L = 1.981

**`specmon`'s counters reconstruct the true count independently and it closes exactly:**

    drafts + accepted + one prefill token per request
    S3-int4:  1320 + 1474 + 22 = 2,816  =  22 x 128   EXACT
    S3-bf16:  1411 + 1384 + 22 = 2,817  =  22 x 128   EXACT to rounding

This is design section 9c working as designed -- two paths to the same quantity, one from
counters and one from the wall clock -- and it is the only reason the error was caught in
twenty minutes rather than surviving into the phase's conclusion.

### The corrected result

| config | KV tokens | decode s | true tok/s | ms/token | vs control |
|---|---|---|---|---|---|
| S0-int4 | 102,944 | 33.28 | 84.6 | 11.82 | 1.00x |
| **S3-int4** | 78,224 | 23.35 | **120.6** | 8.29 | **1.425x** |
| S0-bf16 | 33,424 | 95.47 | 29.5 | 33.90 | 1.00x |
| **S3-bf16** | 10,640 | 58.58 | **48.1** | 20.80 | **1.630x** |

**P5-2 predicted 1.8 - 2.4x for bf16 at batch 1. Measured 1.630x -- just under, and the
first Phase 5 prediction that is close.**

### The fix

`bench.py` now sends `stream_options: {"include_usage": true}` and records the server's
own `completion_tokens` in `usage_tokens`, keeping the chunk count in `out_tokens`. Both
are reported, because **tokens-per-chunk IS the speedup** and is worth seeing directly.
`--no-usage` is the escape hatch for a server that rejects the field; the fallback path
was verified against `mock_server.py`, which does not implement it, so wire compatibility
with the Phase 1 and Phase 2 servers is preserved.

Recorded as incident 32. The general form: **a metric's definition is an assumption about
the server, and a new technique can invalidate it without raising anything.**

## P5-A ACTUAL for bf16: the revised prediction was right, and the config is unusable

| | S0-bf16 | S3-bf16 | |
|---|---|---|---|
| GPU KV cache | 33,424 tok | **10,640 tok** | **-68.2%** |
| available KV | 4.59 GiB | 1.50 GiB | -3.09 GiB |
| max concurrency @6144 | 5.44x | **1.73x** | |

The revision written after P5-A1 said bf16 + EAGLE3 would be "left with single-digit
concurrency" rather than P5-7's original 35-50% capacity drop. **Measured 1.73x
concurrency**, so the revision was right and the original was badly wrong.

At 1.73 concurrent requests this configuration cannot serve. It gets 1.63x on single-stream
latency and gives up essentially all capacity to do it -- a trade that is defensible for a
single-user desktop and indefensible for a server, which is what section 6's adoption rule
already said before the data arrived.

Note also the control drifted again: **33,424 tokens** here against Phase 4's 26,176 and
Phase 3's 33,424. The bf16 KV budget has now been observed at both Phase 3 values on
different days. Same flags. This is why every comparison in this phase is a same-session
matched pair.

## P5-1 ACTUAL: the acceptance miss is NOT caused by quantization

| target | acceptance `a` | pos 0 | pos 1 | pos 2 |
|---|---|---|---|---|
| int4 w4a16 | **0.3722** | 0.633 | 0.350 | 0.133 |
| bf16 | **0.3270** | 0.546 | 0.310 | 0.125 |

Candidate cause 1 was that the head, trained against bf16 Qwen3-8B, would draft poorly for
an int4 target whose hidden states have shifted. **That predicts bf16 acceptance ABOVE
int4's. Measured the opposite, and not marginally**: 0.327 against 0.372, roughly six
standard errors apart at n > 1,300 drafts each.

**Cause 1 is refuted.** Quantizing the target did not degrade EAGLE3 acceptance through
the hidden-state channel. The interference P5-A found between quantization and speculative
decoding is real but is confined to memory.

Why int4 accepts *better* than bf16 is now an open question rather than an answer. The
honest reading is that it is a small effect on an unfavourable workload and should not be
interpreted until the content slices run.

That leaves causes 2 and 3 for the low absolute number, and cause 2 is much the likelier:
**this is `bench.py`'s synthetic filler prompt**, the `open` slice, the least favourable
of the three in design section 5b. Drafting text that is not natural language is exactly
where a trained head should do worst. The `reason` and `copy` slices test it directly.

## P5-D  fp8 added to the matrix, 2026-08-29, predictions written before running

**Decision reversed.** Design section 5a excluded fp8 to keep the run count down, on the
argument that it interpolates between bf16 and int4. Today's measurements make that
argument weak: bf16 + EAGLE3 lands at 1.73x concurrency (unusable) and int4 at 12.73x
(fine). **The two configurations in the matrix are the two whose answers are now obvious,
and the one left out is the only one still in doubt.**

It is also the configuration a real deployment would pick for this workload. Phase 4
measured int4 destroying **37% of 32-step arithmetic without thinking** while fp8 cost no
accuracy at all. Phase 5 is about thinking latency, so omitting the quantization that is
safe for reasoning was the wrong call. Cost to add: one launch, no download, `--quantization
fp8` is applied to the bf16 weights at load time.

Predictions, anchored on the two measured EAGLE3 costs rather than on absolutes (P4-16):

    measured EAGLE3 KV cost   int4  24,720 tok = 3.395 GiB
                              bf16  22,784 tok = 3.128 GiB
                              mean               3.26 GiB = 23,750 tok

| # | quantity | prediction |
|---|---|---|
| P5-D1 | fp8 + EAGLE3 KV loss vs its own same-session control | **22,000 - 25,500 tokens** |
| P5-D2 | fp8 + EAGLE3 concurrency @6144 | **7 - 9x** -- usable, unlike bf16 |
| P5-D3 | fp8 acceptance `a` on the `open` slice | **0.33 - 0.37**, between bf16's 0.327 and int4's 0.372 |
| P5-D4 | fp8 + EAGLE3 single-stream speedup | **1.45 - 1.65x** |

P5-D1 is close to arithmetic and a miss would mean the draft's cost depends on the target's
weight format, which nothing so far suggests. **P5-D3 is the interesting one**: it tests
whether acceptance varies monotonically with how aggressively the target is quantized. If
fp8 lands outside the bf16-int4 bracket, then the bf16-vs-int4 difference measured today is
not about quantization at all and should not be reported as if it were.

## P5-E  A latent OOM in vLLM spec decoding that startup profiling does not see (2026-08-29)

The identical S3-bf16 and S3-int4 configurations that launched and measured cleanly an
hour earlier both **failed to start** on the re-run. Not a flake, and the cause is worth
recording as a property of the system rather than an accident.

    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 594.00 MiB.
      GPU 0 has a total capacity of 22.06 GiB of which 373.44 MiB is free.
      .../v1/worker/gpu/spec_decode/rejection_sampler.py in _verify
        processed_logits = self.sampler.apply_sampling_params(
      .../v1/worker/gpu/sample/sampler.py line 159
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

**The rejection sampler upcasts the verify logits to fp32**, and the buffer is sized by
`max_num_seqs x (k+1) x vocab`:

    594 MiB / (151,936 vocab x 4 bytes) = 1,025 positions
    max_num_seqs 256 x (k+1) 4          = 1,024      exact

| max_num_seqs | fp32 logits buffer at k=3 |
|---|---|
| 256 (default) | **593.5 MiB** |
| 64 | 148.4 MiB |
| 32 | 74.2 MiB |
| 20 | 46.4 MiB |

**The finding: vLLM's startup memory profile does not reserve for this allocation.** The
server profiles free memory, sizes the KV cache to fill the GPU, reports a healthy
`GPU KV cache size`, and then OOMs the first time the rejection sampler runs. Whether a
given launch survives depends on how much slack the profile happened to leave -- the same
startup-profile variance that produced 102,944 tokens today against 95,648 in Phase 4.
**A spec-decoding server can therefore start successfully and die on its first request,
and the same command can do either on different days.**

**The second half of the finding is that `max_num_seqs=256` is meaningless on this box.**
Measured concurrency ceilings from the KV budget are 1.73x (bf16 + EAGLE3), 13.11x (fp8)
and 16.76x (int4). The sampler is sizing a buffer for 256 concurrent sequences that the
KV cache could never hold, and paying 594 MiB of VRAM for the privilege -- VRAM that is
taken from the KV cache, which lowers concurrency further. The default is not merely
wasteful here, it is self-defeating.

### Correction to the run protocol, applied to every remaining Phase 5 run

- **`--max-num-seqs 20`** on every configuration, control and spec alike. Above every
  measured concurrency ceiling so it constrains nothing, while cutting the logits buffer
  from 594 to 46 MiB.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**, which the error message itself
  recommends and which the decision log adopted for Phase 2 engine runs on 2026-08-21
  (+29% concurrency there). It was never applied to the vLLM runs. It should have been.

Both halves of every matched pair are re-measured under the new setting; the controls
already collected at the default are discarded rather than compared across settings.

## P5-F  Two things the xhigh audit caught, 2026-08-29

### 1. The draft model has its own KV cache, and it explains a gap I had glossed

P5-A1 reported KV loss in tokens and separately in GiB, and the two did not reconcile:
78,224 tokens x 147,456 B is 10.74 GiB, but the server reported 11.04 GiB. I noted the
0.3 GiB discrepancy and moved past it. That was the wrong call -- it was signal.

The EAGLE3 head is a 1-layer transformer with 8 KV heads at head_dim 128, so it needs its
own cache:

    target Qwen3-8B   2 x 36 layers x 8 x 128 x 2 B = 147,456 B/token
    EAGLE3 head       2 x  1 layer  x 8 x 128 x 2 B =   4,096 B/token
    combined                                          151,552 B/token   (+2.8%)

Every configuration now reconciles exactly:

| run | tokens | reported GiB | target-only | target + draft KV |
|---|---|---|---|---|
| S0-int4 | 102,944 | 14.14 | **14.14** | 14.53 |
| S3-int4 | 78,224 | 11.04 | 10.74 | **11.04** |
| S0-bf16 | 33,424 | 4.59 | **4.59** | 4.72 |
| S3-bf16 | 10,640 | 1.50 | 1.46 | **1.50** |

Controls match target-only; spec runs match target-plus-draft. **So `GPU KV cache size` is
the number of tokens servable including the draft's own cache**, which means the
token-level comparison used throughout (102,944 -> 78,224) is the correct apples-to-apples
metric and stands. The draft's KV is a fourth cost term, small at 2.8% but real, and it
was missing from the P5-A ledger alongside activations and CUDA graphs.

### 2. `--max-num-seqs 20` would have corrupted the throughput sweeps

The P5-E fix caps concurrency at 20 to shrink the rejection sampler's fp32 buffer. That is
harmless for the batch-1 serial runs now executing -- one request at a time cannot reach
the cap. **It is not harmless for the sweeps, which have not run yet:**

| config | workload | concurrency the KV allows | cap 20 |
|---|---|---|---|
| bf16 | 512/64 | 58.0 | **binds** |
| fp8 | 512/64 | 139.9 | **binds** |
| int4 | 512/64 | 178.7 | **binds** |
| int4 | 4096/1024 | 20.1 | **binds, just** |
| bf16, fp8 | 4096/1024 | 6.5, 15.7 | does not bind |

P5-B's entire purpose is locating the batch size at which speculative decoding inverts
sign, predicted at B* between 8 and 20 for int4. **Capping the batch at 20 would have made
the crossover unobservable, and on 512/64 the sweep would have measured my own cap rather
than the technique.** Caught before the sweeps ran; the serial data already collected is
unaffected.

**This is a genuine constraint of the hardware, not just a protocol bug, and it should be
reported as a result.** The sampler buffer is `max_num_seqs x (k+1) x vocab x 4 B`, so on
a 24 GB card speculative decoding forces a direct trade between concurrency and not
running out of memory:

    max_num_seqs  64 -> 148 MiB      128 -> 297 MiB      192 -> 445 MiB

Resolution for the sweeps: set `max_num_seqs` per workload, above what the KV budget can
actually deliver, and record the resulting buffer size. 4096/1024 needs ~24; 512/64 needs
~192 for int4, which is 445 MiB and uncomfortably close to the 594 MiB that already OOMed.
**If int4 at 512/64 cannot be run without OOM, that is itself the finding** -- and it must
be stated as a measured limit rather than quietly worked around by lowering the cap.

## P5-G  THE MATCHED TRIPLE, corrected protocol, 2026-08-30

All six runs same session, same settings (`--max-model-len 6144 --max-num-seqs 20`,
`expandable_segments:True`), batch 1, 512-token synthetic prompt, 128 max out, k=3.

| config | KV control | KV +EAGLE3 | loss | concurrency | tok/s | speedup | acceptance |
|---|---|---|---|---|---|---|---|
| bf16 | 27,104 | 11,712 | 15,392 | **1.91x** | 29.5 -> 45.9 | **1.556x** | 0.3063 |
| fp8 | 68,592 | 52,224 | 16,368 | **8.50x** | 53.4 -> 77.6 | **1.453x** | 0.3035 |
| int4 | 95,792 | 78,272 | 17,520 | **12.74x** | 84.5 -> 120.6 | **1.427x** | 0.3722 |

**The KV cost is a near-constant charge** -- 15,392 / 16,368 / 17,520 tokens against a
budget that varies 3.5x. The fixed-charge model from P5-A survives; only the magnitude was
wrong, and P5-F explained why.

### The result I did not predict: acceptance and speedup move in OPPOSITE directions

int4 accepts **22% more** draft tokens than bf16 and delivers the **smallest** speedup.

    realised = measured speedup / theoretical tokens-per-step L,  L = 1 + 3a

| config | L | speedup | realised |
|---|---|---|---|
| bf16 | 1.919 | 1.556x | **0.811** |
| fp8 | 1.910 | 1.453x | **0.761** |
| int4 | 2.117 | 1.427x | **0.674** |

**The fraction of the theoretical gain actually delivered falls monotonically as the
weights get cheaper: 0.81 -> 0.76 -> 0.67.**

This is P5-B's mechanism showing up at batch 1, which I had only predicted for large batch.
The verify pass costs a fixed amount of extra compute -- three draft forward passes plus a
4-position verify -- while the thing it is hiding behind, the target's weight read, shrinks
with quantization. bf16 reads 16.39 GB per step and the overhead disappears into it; int4
reads 6.12 GB and the same overhead is proportionally 2.7x more visible.

**So quantization and speculative decoding compete twice, not once.** P5-A found them
competing for memory. This is a second, independent channel: quantization removes the very
bandwidth stall that speculative decoding exists to exploit. **The better your decode
already is, the less speculation can add** -- and the two techniques are closest to
redundant exactly where each is individually strongest.

### Acceptance does not vary with numeric precision

| config | weights | acceptance |
|---|---|---|
| bf16 | `Qwen/Qwen3-8B` | 0.3063 |
| fp8 | **same weights**, quantized at load | 0.3035 |
| int4 | `RedHatAI/Qwen3-8B-quantized.w4a16`, a **different checkpoint** | 0.3722 |

P5-D3 pre-registered the falsification condition: if fp8 landed outside the bf16-int4
bracket, the bf16-vs-int4 difference is not about quantization. **It landed at bf16's
value** (0.3035 vs 0.3063, a 0.9% relative difference).

bf16 and fp8 are the *same weights* at different precision, and they accept identically.
**Numeric precision does not affect how draftable the model's output is.** int4 differs by
22%, but int4 is a separately calibrated checkpoint, so precision and checkpoint identity
are confounded there. The honest statement is: precision does not matter, and the int4
difference is a property of that specific checkpoint that this experiment cannot separate
from its bit width. Not reported as a quantization effect.

### Two instrument cross-checks passed

- `bench.py` now reports **2.10 tokens/chunk** directly, against L = 1 + 3(0.3722) = 2.117
  derived from specmon's independent counters. Agreement to 0.8% confirms the incident-32
  fix from both directions.
- int4 acceptance measured **0.3722 under both protocols**, identical to four decimals
  across a `max_num_seqs` change that altered the KV budget by 7%. Acceptance is a stable
  property of the draft/target/workload triple and is not sensitive to the memory settings.

## P5-H  Content slices: the phase's central bet, predictions written before running

Everything measured so far used `bench.py`'s synthetic filler prompt -- the `open` slice,
the least favourable content there is. **fp8 + EAGLE3 measured a = 0.3035 there.** The
question this phase exists to answer is whether real reasoning text drafts better.

Held fixed: fp8 + EAGLE3, k=3, `--max-num-seqs 20`, concurrency 1 so nothing is confounded
with batch. Only the CONTENT of the workload changes.

### A confound the item set forced me to control

`mkitems.py` generates the math slice from templates -- "N more are added", "the count
doubles", "N are taken away". That is highly repetitive text, and it would draft well for
reasons that have nothing to do with reasoning. **A high acceptance on synthetic maths
would be uninterpretable on its own.**

GSM8K is human-written word problems. Running both separates "reasoning is draftable" from
"my templates are draftable". Without gsm8k this experiment could not distinguish the
phase's hypothesis from an artifact of my own generator.

| slice | what it is | prediction |
|---|---|---|
| `open` (measured) | synthetic filler, no thinking | **0.3035** |
| `math/think` | templated arithmetic, thinking ON | **0.55 - 0.75** |
| `math/nothink` | templated arithmetic, answer only | **0.40 - 0.55** |
| `gsm8k/think` | natural-language reasoning, thinking ON | **0.50 - 0.65** |
| `longctx` | recall a 4-char code from a document | **0.55 - 0.75** |

**P5-5, registered before the phase began, predicted 0.75 - 0.88 for reasoning.** It is
scored against `math/think` and `gsm8k/think` and is currently on track to miss high, since
every acceptance measured so far has come in far below expectation.

### What each outcome would mean, decided before the data

- **math/think high AND gsm8k/think high** -> reasoning genuinely drafts better. The Phase 7
  thinking-latency lever is real, and the 36-second thinking wait is halvable.
- **math/think high BUT gsm8k/think near `open`** -> my templates are draftable, reasoning
  is not. P5-5 is wrong and the headline finding would have been an artifact of my own
  item generator. This is the outcome the gsm8k control exists to catch.
- **both near `open` (~0.30)** -> acceptance is a property of this draft/target pair and
  barely moves with content. Speculative decoding gives a flat ~1.45x here and the phase's
  bet is simply lost.
- **longctx high** -> expected regardless; the output is short and formulaic. It is the
  sanity check that the measurement responds to content at all. **If longctx does NOT come
  in above `open`, suspect the experiment before believing the result.**

## P5-H ACTUALS  Content slices, fp8 + EAGLE3, concurrency 1, 2026-08-30

| slice | acceptance | L | vs filler | verify steps | predicted | |
|---|---|---|---|---|---|---|
| `open` synthetic filler | 0.3035 | 1.910 | 1.00x | -- | (measured) | |
| `gsm8k/think` natural reasoning | **0.4905** | 2.471 | **1.29x** | 5,194 | 0.50-0.65 | narrow miss, low |
| `longctx` verbatim recall | **0.5270** | 2.581 | **1.35x** | 1,143 | 0.55-0.75 | miss, low |
| `math/think` templated + thinking | **0.5960** | 2.788 | **1.46x** | 7,369 | 0.55-0.75 | **correct** |
| `math/nothink` templated, answer only | **0.6708** | 3.012 | **1.58x** | 3,494 | 0.40-0.55 | **miss, and INVERTED** |

**Every prior acceptance number in this phase was measured on the worst content there is.**
Real workloads draft roughly twice as well as `bench.py`'s synthetic filler. The 0.30 that
looked like a disappointing property of EAGLE3 was mostly a property of the prompt.

### The gsm8k control earned its place

Templated maths scores **21% above** natural-language reasoning (0.5960 vs 0.4905). Part of
the synthetic slice's advantage is that `mkitems.py` writes repetitive text, exactly as
feared. **Reporting `math/think` alone would have overstated reasoning acceptance by a
fifth.**

The underlying claim survives the control: gsm8k is still **62% above filler**. Real
reasoning genuinely drafts better. The control separated an inflated effect from a false
one rather than destroying it -- which is the outcome that justifies having run it.

### P5-5 was directionally right and quantitatively wrong, for a reason worth keeping

P5-5 predicted 0.75-0.88 for reasoning and argued that chain-of-thought is repetitive
scaffolding. Measured 0.4905 (natural) to 0.5960 (templated). **Direction right, magnitude
badly wrong.**

But the rationale was worse than the number, and `math/nothink` proves it. **Same problems,
same model: 0.6708 without thinking against 0.5960 with it.** I predicted nothink LOWER and
it is the highest slice measured.

**Thinking text is HARDER to draft than non-thinking text.** Not easier. A thinking block is
where the model explores, backtracks and reconsiders -- high-entropy by construction. Denied
that block it emits a tidy formulaic walkthrough, and formulaic drafts well.

**Consequence for Phase 7, which is the reason this phase exists:** the 36 seconds of
thinking latency that Phase 7 wants to cut is the *hardest* part of the output to
accelerate. Speculative decoding still helps there far more than the filler benchmark
suggested, but the cheapest wins are in the visible answer, not in the reasoning. Phase 7
inherits a real lever, weaker than hoped, and pointed at the wrong end of the response.

### An instrument rule caught its own violation

`longctx` first ran at 30 items and returned **384 verify steps against the 400 minimum**
fixed in design section 6. `specmon` flagged it and the number was withheld rather than
reported. Re-run at 90 items: **1,143 steps, a = 0.5270** against the small sample's 0.5200.

The small sample was accurate. That is not the point -- it was not *knowably* accurate at
the time, and the threshold was written before any data existed. One cheap re-run converted
a guess into a measurement.

### Registered before the control runs: speedup implied by these acceptance rates

Realised efficiency on fp8/filler was 1.453 / 1.910 = **0.761**. Longer contexts here mean
more KV read per step, which enlarges the memory term and should make the verify overhead
*relatively* smaller, so realised should rise. Predicting 0.76 - 0.85:

| slice | L | predicted speedup |
|---|---|---|
| gsm8k/think | 2.471 | **1.88 - 2.10x** |
| longctx | 2.581 | 1.96 - 2.19x |
| math/think | 2.788 | **2.12 - 2.37x** |
| math/nothink | 3.012 | 2.29 - 2.56x |

Measured directly by re-running the identical passes with speculation off and comparing
wall clock. Spec-on elapsed, for the record: math/think 193.1 s, math/nothink 89.0 s,
gsm8k/think 132.9 s, longctx(30) 33.9 s.

## P5-I  THE PHASE'S HEADLINE: measured speedup on real content, 2026-08-30

Identical items, identical order, concurrency 1, fp8. Only speculative decoding differs.
Normalised by tokens, not wall clock: `math/think` produced 8.4% fewer tokens with
speculation on than off, because thinking-chain length varies run to run (incident 22 --
shape changes reorder bf16 accumulation). Comparing raw elapsed there would have credited
speculation with generating less text.

| slice | spec tok/s | control tok/s | **speedup** | predicted | | L | realised |
|---|---|---|---|---|---|---|---|
| `math/think` | 106.4 | 52.0 | **2.045x** | 2.12-2.37x | miss low | 2.788 | 0.734 |
| `math/nothink` | 118.2 | 52.5 | **2.252x** | 2.29-2.56x | miss low | 3.012 | 0.748 |
| `gsm8k/think` | 96.6 | 52.2 | **1.849x** | 1.88-2.10x | miss low | 2.471 | 0.748 |
| `longctx` | 28.3 | 23.8 | **1.189x** | 1.96-2.19x | **MISS BADLY** | 2.581 | 0.461 |
| `open` filler (earlier) | 77.6 | 53.4 | 1.453x | -- | -- | 1.910 | 0.761 |

**The phase's honest headline: roughly 2x on workloads that resemble real work, against
1.45x on the synthetic benchmark.** Measuring only on `bench.py`'s filler prompt would have
understated the technique by 40% -- the same class of error as Phase 4's workload lesson,
arrived at from a different direction.

### Four narrow misses in one direction, and one real one

The three reasoning slices all landed 3-4% below their predicted ranges. The cause is a
single wrong assumption: I predicted realised efficiency would **rise** from 0.761 to
0.76-0.85 because longer contexts mean more KV traffic per step, enlarging the memory term
that hides the verify overhead. **It did not rise. It fell slightly, to 0.73-0.75.**

Why the reasoning was wrong: longer context makes the *draft's* work more expensive too.
The EAGLE head runs its own attention over the same growing context, three times per step.
I modelled the context growth as helping only the target's side of the ledger when it loads
both. The overhead does not shrink relative to the memory term because it grows with it.

### `longctx` is the interesting miss: speculative decoding cannot touch prefill

Predicted 1.96-2.19x from an acceptance of 0.5270, measured **1.189x**. Realised efficiency
0.461 against 0.73-0.75 everywhere else.

**Nothing is wrong with the acceptance number -- the workload shape is the answer.** longctx
sends ~4,096 prompt tokens and generates ~32. Almost all of the request is prefill, and
**speculative decoding accelerates decode only.** Its 0.527 acceptance is real and buys a
genuine decode speedup that is then diluted to near-nothing by the prefill it cannot help.

This is `PROJECT.md` section 5b's point about quantization and TTFT, in a different costume:
*the metric a technique improves and the metric a workload is dominated by need not be the
same one.* A retrieval-heavy product -- which is exactly what Phase 6 builds -- would get
almost nothing from speculative decoding end to end, no matter how well the draft performs.

### What this fixes about the earlier conclusion

Reported earlier today: "speculative decoding is worth 1.45x on fp8." That number is
correct and unrepresentative. Corrected statement, with the condition attached:

- **long-output reasoning: ~1.85 - 2.25x** -- the case worth deploying for
- short synthetic completions: 1.45x
- **long-prompt short-answer retrieval: 1.19x** -- barely worth the 16,368 tokens of KV it costs

**A single "speedup of speculative decoding" number does not exist for this system.** It
ranges 1.19x to 2.25x on one GPU, one model and one draft, decided entirely by the shape of
the request.

## P5-J  Crossover sweeps and the two unclimbed rungs -- predictions, 2026-08-30

### Protocol decision: `--max-num-seqs 32` on all four servers

P5-F showed a cap of 20 would bind on the sweeps and turn them into a measurement of my
own cap. 32 costs 74.2 MiB of fp32 logits buffer -- comfortably clear of the 594 MiB that
OOMed at the default 256 -- and sits above the concurrency the KV budget can actually
deliver on the capacity workload:

    fp8 no-spec   4096/1024  ->  15.7 concurrent      512/64  ->  139.9
    fp8 +EAGLE3   4096/1024  ->  10.2 concurrent      512/64  ->   90.7

So on 4096/1024 the cap does not bind at all and the sweep walks batch from ~1 to ~16 by
raising the arrival rate. On 512/64 the cap DOES bind at 32, and that is stated rather than
hidden: both halves of the pair carry the identical cap, so the comparison is honest even
though neither side reaches its own KV ceiling. **The predicted crossover of 8-20 lies
inside 32, which is why 32 is sufficient to answer the question.**

### The crossover, P5-B revisited with measured numbers

P5-B predicted B* from a compute_eff bracket spanning an order of magnitude. Now anchored
on measurement: realised efficiency is **0.73-0.76** at batch 1 across fp8 content slices,
so verify overhead consumes roughly a quarter of the theoretical gain when the GPU is
otherwise idle. That overhead is fixed per step while the memory term it hides behind is
amortised across the batch, so the margin erodes as batch grows.

| # | quantity | prediction |
|---|---|---|
| P5-J1 | fp8 crossover B*, 4096/1024 | **B* is NOT reached below 16.** Spec still ahead at the KV ceiling, margin shrunk from 1.45x to 1.05-1.20x |
| P5-J2 | fp8 crossover B*, 512/64 | **B* between 12 and 28**, inside the cap of 32 |
| P5-J3 | capacity (req/s at the knee), spec vs control, 4096/1024 | spec **LOWER by 15-30%** -- it buys latency and pays in concurrency, having given up 16,368 KV tokens |
| P5-J4 | ITL p50 at the lowest rate | spec better by 1.3-1.5x, consistent with the serial runs |

**P5-J3 is the one that decides deployability.** Every speedup measured so far is
single-stream. If capacity falls 30%, speculative decoding is a latency feature bought with
throughput, and the fp8+EAGLE3 recommendation needs the caveat attached.

### Rung 1: n-gram, and why it might beat a trained head

Costs **zero VRAM**, so it gives up none of the 16,368 tokens EAGLE3 takes. It proposes
continuations found by matching the recent suffix against text already in the context.

| # | quantity | prediction |
|---|---|---|
| P5-J5 | n-gram acceptance, synthetic filler | **0.02 - 0.12** -- `bench.py` prompts are random tokens with nothing to match |
| P5-J6 | n-gram acceptance, `gsm8k/think` | **0.10 - 0.25** -- reasoning restates the problem, some hits |
| P5-J7 | n-gram acceptance, `longctx` | **0.45 - 0.75** -- the answer QUOTES the document. This is prompt-lookup's designed best case |
| P5-J8 | n-gram KV cost | **exactly 0 tokens** vs EAGLE3's 16,368 |

**P5-J7 is the interesting bet.** longctx is where EAGLE3 did worst end-to-end (1.19x,
diluted by prefill) while still charging full price in memory. If n-gram matches its
acceptance there at zero memory cost, **the recommendation for retrieval workloads flips
from EAGLE3 to n-gram** -- and Phase 6's chat app is exactly a retrieval workload.

### Rung 2: Qwen3-0.6B as draft

1.400 GiB of weights against EAGLE3's 1.904, but it sees only the text, not the target's
hidden states, which is the whole advantage EAGLE3 has.

| # | quantity | prediction |
|---|---|---|
| P5-J9 | Qwen3-0.6B acceptance, `gsm8k/think` | **0.30 - 0.45**, clearly below EAGLE3's 0.4905 |
| P5-J10 | Qwen3-0.6B KV cost | **12,000 - 16,000 tokens** -- lower than EAGLE3's 16,368 on weights, but it runs 28 layers of KV against EAGLE3's 1, so its own cache is far larger per token |
| P5-J11 | Qwen3-0.6B speedup, `gsm8k/think` | **1.3 - 1.6x**, below EAGLE3's 1.849x |

P5-J10 is the one I am least sure of and it has a term I got wrong once already: EAGLE3's
own KV is 4,096 B/token because it has one layer. **Qwen3-0.6B has 28 layers with 8 KV
heads at head_dim 128, so 2 x 28 x 8 x 128 x 2 = 114,688 B/token -- 28x EAGLE3's per-token
cache and 78% of the TARGET's own 147,456.** If vLLM allocates draft KV proportionally,
rung 2 may cost far more cache than its smaller weights suggest, and could land worse than
EAGLE3 despite being the lighter model. Recorded before measuring.

## P5-K  Crossover sweeps and the full ladder, 2026-08-30. All fp8, `--max-num-seqs 32`.

### THE CROSSOVER EXISTS, and only on one of the two workloads

`512/64`, open-loop, per-chunk ITL p50 (spec chunks carry L=2.579 tokens, so per-token is
the chunk figure divided by that):

| rate | no-spec ITL | EAGLE3 ITL/chunk | EAGLE3 per token | verdict |
|---|---|---|---|---|
| 2 | 19 ms | 26 ms | **10.1 ms** | spec 1.88x ahead |
| 4 | 21 ms | 29 ms | **11.2 ms** | spec 1.87x ahead |
| 6 | 24 ms | 176 ms | **68 ms** | **spec 2.8x BEHIND** |
| 8 | 24 ms | 178 ms | 69 ms | spec behind |
| 10 | 24 ms | 178 ms | 69 ms | spec behind |

**The sign flips between 4 and 6 req/s.** `PROJECT.md` section 5 named "reproduce the
hurts-at-high-batch result" as a phase goal; this is that result, measured on our own box.

`4096/1024`, same treatment (L=2.150):

| rate | no-spec ITL | EAGLE3 per token |
|---|---|---|
| 0.20 | 21 ms | 14 ms |
| 0.35 | 25 ms | 17 ms |
| 0.50 | 27 ms | 24 ms |
| 0.70 | 38 ms | 31 ms |
| 0.90 | 39 ms | 31 ms |

**No crossover. Speculation stays ahead at every rate up to the KV ceiling.** P5-J1
predicted exactly this and is CORRECT.

The two workloads disagree about the sign of the effect, which is Phase 4's lesson landing
for the third time in two phases.

**CAVEAT, and it is incident 12 again: the sweep was 2,4,6,8,10 and the crossover is
between 4 and 6.** The resolution of the sweep is the resolution of the answer, and I
cannot say whether it flips at 4.5 or 5.9. I criticised `roofline.py` for exactly this in
Phase 2 and then wrote a coarse sweep anyway. A dense sweep across 4-6 would localise it.

**Second caveat: acceptance was measured once per sweep, not per rate.** L almost certainly
falls as batch grows, so the per-token column overstates spec at high rates -- meaning the
true crossover is at a LOWER rate than shown, not higher. The direction of the error is
known even though its size is not.

### P5-J3 was badly wrong: capacity barely moves

| | no-spec | EAGLE3 | |
|---|---|---|---|
| 512/64 peak | 6.24 req/s | 6.09 req/s | **-2.4%** |
| 4096/1024 peak | 0.54 req/s | 0.53 req/s | **-1.9%** |

Predicted a **15-30% capacity loss**. Measured 2%. The reasoning was that EAGLE3 gives up
16,368 KV tokens, so concurrency must fall. It does fall -- but **neither workload is
KV-bound at its knee**, so the lost cache costs nothing. Capacity is limited by compute and
scheduling here, not by cache. Losing a quarter of a resource you were not using is free.

This is the Phase 3 prefix-caching lesson yet again: **check whether anything is pressing
against the ceiling before predicting the effect of lowering it.**

### The full ladder, and my ordering was wrong

Drafting quality, expressed as tokens emitted per decode step across ALL steps:

| method | KV cost | % of capacity | filler | gsm8k | longctx |
|---|---|---|---|---|---|
| n-gram | **2,304** | **3.4%** | 1.123 | 1.176 | 1.160 |
| EAGLE3 | 16,368 | 23.9% | 1.910 | 2.471 | 2.581 |
| **Qwen3-0.6B** | **36,448** | **53.2%** | **2.133** | **3.166** | **2.743** |

**Qwen3-0.6B is the best drafter on every slice** -- 3.166 tokens/step on gsm8k against
EAGLE3's 2.471, a 28% edge. P5-J9 predicted it CLEARLY BELOW EAGLE3 at 0.30-0.45 acceptance;
measured **0.7226**. Wrong, and wrong in the direction I was most confident about.

**P5-J10 was numerically wrong and mechanically right.** Predicted 12,000-16,000 tokens of
KV cost; measured **36,448**, more than double EAGLE3's. The flagged reason was correct:
Qwen3-0.6B carries 28 layers of KV at 114,688 B/token against EAGLE3's single layer at
4,096. The lighter model has the heavier cache. Writing the mechanism down before measuring
turned a bad number into an explained one.

### Speedup does NOT follow drafting quality

gsm8k/think, against the no-spec control at 52.2 tok/s:

| method | tok/s | speedup | L | realised |
|---|---|---|---|---|
| n-gram | 55.9 | 1.07x | 1.176 | **0.910** |
| **EAGLE3** | **96.6** | **1.85x** | 2.471 | 0.749 |
| Qwen3-0.6B | 91.3 | 1.75x | **3.166** | **0.553** |

**The best drafter is not the fastest configuration.** Qwen3-0.6B guesses 28% better and
ends up 5% slower, because drafting three times per step through a 28-layer model costs
far more than EAGLE3's single layer. Its realised efficiency is 0.553 against EAGLE3's
0.749. n-gram is the opposite extreme: it barely drafts, but what it does costs nothing on
the GPU, so it realises 91% of its small gain.

**EAGLE3 is confirmed as the right choice -- by measurement, not by assumption.** It sits at
the optimum of a genuine three-way trade between draft quality, draft cost and memory.

### The n-gram number that would have been a false headline

`specmon` reported n-gram acceptance of **0.9333** on the filler workload, against EAGLE3's
0.3035. Read naively that says n-gram is three times better for free.

It is an artifact of what acceptance means. **`a` is conditional on a draft having been
proposed**, and n-gram only proposes when it finds a matching suffix:

| slice | acceptance when it fires | **how often it fires** | overall tokens/step |
|---|---|---|---|
| filler | 0.9333 | **4.4%** | **1.123** |
| gsm8k | 0.5042 | 11.6% | 1.176 |
| longctx | 0.4490 | 11.9% | 1.160 |

On `bench.py`'s synthetic filler it is right 93% of the time on the 4% of steps where the
prompt repeats itself -- which also says the filler prompt is trivially self-similar, and is
a further reason that workload should never have been the phase's yardstick.

**Instrument limitation now recorded:** `specmon`'s acceptance is the correct metric for
drafters that fire every step (EAGLE3, draft models, both measured at ~100% hit rate) and is
MISLEADING for opportunistic drafters (n-gram, suffix). For those, the unconditional figure
must be derived: `tokens / (drafts + (tokens - drafts - accepted))`. Comparing a conditional
rate against an unconditional one is the same class of error as incident 32.

### P5-J7 correct in number, wrong in reasoning

Predicted n-gram acceptance on `longctx` at 0.45-0.75; measured 0.4490, just at the edge.
But the bet behind it was that n-gram would **beat** EAGLE3 there, because the answer quotes
the document and prompt-lookup is built for exactly that. It does not: 1.160 tokens/step
against EAGLE3's 2.581. The recommendation for retrieval workloads does **not** flip.

Why the reasoning failed: the answer quotes only a handful of tokens from a 4,096-token
document. The rest of the output is the model's own phrasing, which has no match to find. A
copy-heavy *task* is not the same as copy-heavy *output*.

## P5-L  The quality gate: predictions before running, 2026-08-30

Speculative decoding is distribution-preserving **by construction** -- a rejected draft is
replaced by the target's own token -- so in theory the output is exactly what the
unaccelerated model would have produced. This phase has reported speedups all day without
testing that on this build. If the rejection sampling is subtly wrong, every number
describes a different model than its control.

Protocol reuses Phase 4 exactly so `d0` is comparable: same items, same `order_seed`,
concurrency 12, `--max-tokens-think 5120 --max-tokens-nothink 1024`, slices math + gsm8k +
longctx. Only `--spec-method eagle3` differs between the two runs.

| # | quantity | prediction |
|---|---|---|
| P5-L1 | `ansdiff` on ALL, spec vs no-spec | **1.5 - 4.0%** |
| P5-L2 | McNemar p on ALL | **> 0.05**, no significant accuracy change |
| P5-L3 | accuracy delta on ALL | within **±1.5 points** |
| P5-L4 | worst slice for `ansdiff` | **math/nothink at high k** |

### Why P5-L1 is NOT simply "at or below d0 = 1.8%"

`d0` was measured bf16-vs-bf16 with the KV budget **pinned identical** across the pair.
Here the two servers cannot have identical KV: EAGLE3 costs 16,368 tokens, so the spec run
holds 52,080 against the control's 68,448. Different cache means different batch
composition, and incident 22 established that batch composition alone reorders bf16
accumulation and changes outputs.

**So some excess over `d0` is expected and is NOT evidence of broken rejection sampling.**
The gate is therefore asymmetric and fixed now:

- `ansdiff` **under 4%** with McNemar **p > 0.05** -> PASS. Consistent with nondeterminism
  from batch composition, no accuracy cost.
- `ansdiff` **above 4%** or McNemar **p < 0.05 with the spec run less accurate** -> FAIL,
  and every speedup in this phase is quoted against a caveat.

P5-L4 follows Phase 4's finding that the noise floor tracks **proximity to the model's
competence limit**, not chain length: bf16 disagreed with itself 0% on the thinking slice
and up to 4.8% on no-thinking at k=32, because that is where it sits at the edge of what it
can do. The same slice should be the most sensitive here.

## P5-L ACTUALS  THE QUALITY GATE PASSES, 2026-08-30

fp8, concurrency 12, Phase 4 protocol exactly. 1,210 items paired, **zero dropped from
either side** -- no silent denominator shrink.

| group | n | acc no-spec | acc EAGLE3 | b | c | ansdiff | McNemar p |
|---|---|---|---|---|---|---|---|
| gsm8k/think | 200 | 94.0% | 94.0% | 0 | 0 | **0.0%** | 1.0000 |
| gsm8k/nothink | 200 | 93.5% | 93.0% | 3 | 2 | 2.5% | 1.0000 |
| longctx/nothink | 90 | 100.0% | 100.0% | 0 | 0 | **0.0%** | 1.0000 |
| math/think/k4 | 45 | 97.8% | 100.0% | 0 | 1 | 2.2% | 1.0000 |
| math/think/k8 | 75 | 100.0% | 100.0% | 0 | 0 | **0.0%** | 1.0000 |
| math/think/k16 | 105 | 99.0% | 100.0% | 0 | 1 | 1.0% | 1.0000 |
| math/think/k32 | 135 | 97.8% | 98.5% | 1 | 2 | 2.2% | 1.0000 |
| math/nothink/k4 | 45 | 97.8% | 95.6% | 1 | 0 | 2.2% | 1.0000 |
| math/nothink/k8 | 75 | 92.0% | 94.7% | 0 | 2 | 2.7% | 0.5000 |
| math/nothink/k16 | 105 | 83.8% | 84.8% | 2 | 3 | 6.7% | 1.0000 |
| math/nothink/k32 | 135 | 44.4% | 46.7% | 3 | 6 | **10.4%** | 0.5078 |
| **ALL** | **1210** | **89.3%** | **89.9%** | 10 | 17 | **2.8%** | **0.2478** |

### VERDICT: PASS on both pre-registered conditions

    ansdiff  2.8%   threshold < 4.0%    PASS
    McNemar  0.2478 threshold > 0.05    PASS
    accuracy 89.3% -> 89.9%, +0.6 points, spec nominally BETTER

b=10, c=17: speculation fixed 17 items the control got wrong and broke 10 it got right.
The net favours speculation and is not significant. **There is no accuracy cost.**

**Every speedup reported in this phase is therefore quoted against outputs that are
statistically the control's.** "1.85x faster" means the same model, faster.

### All four predictions correct -- and P5-9 from the original design too

| # | predicted | measured | |
|---|---|---|---|
| P5-L1 | ansdiff 1.5-4.0% | 2.8% | correct |
| P5-L2 | McNemar p > 0.05 | 0.2478 | correct |
| P5-L3 | accuracy within +/-1.5 pts | +0.6 | correct |
| P5-L4 | worst slice = math/nothink high k | k32 at 10.4%, the maximum | correct |
| P5-9 | drift under 3% vs d0 | 2.8% | correct |

The first clean sweep of the phase, and it is the phase's least surprising experiment --
which is the point. **A well-understood mechanism should be predictable; the misses earlier
were all in places where the model of the system was incomplete.**

### The shape of the disagreement is the real evidence, not its size

**Disagreement is zero exactly where the model is confident, and concentrated exactly where
it is guessing:**

    gsm8k/think     94.0% accurate   ansdiff 0.0%
    math/think/k8  100.0% accurate   ansdiff 0.0%
    longctx        100.0% accurate   ansdiff 0.0%
    math/nothink/k32 44.4% accurate  ansdiff 10.4%

Three slices reproduce **bit-identical answers on every single item**. If rejection
sampling were subtly wrong, error would be spread across all slices in proportion to token
count, not concentrated in the one slice where accuracy is 44% and the model is at the edge
of what it can do.

**This independently reconfirms Phase 4's correction**: the noise floor tracks *proximity to
the competence limit*, not chain length. At k32-nothink the model is barely above chance, so
any perturbation -- batch composition, accumulation order, a different KV budget -- flips
answers. 10.4% disagreement there with accuracy going UP 2.3 points is nondeterminism, not
damage.

### One caveat kept attached

The two servers could not hold identical KV (52,080 vs 68,448 tokens), so batch composition
differed and some of the 2.8% is that rather than speculation itself. This is why the gate
was set at 4% rather than at `d0` = 1.8%, decided before the data existed. A stricter test
would pin KV identically across the pair, at the cost of not measuring the configuration
anyone would actually deploy.

## P5-M  The k sweep: predictions, and a correction to something I said out loud

k is the number of tokens the draft proposes per step. **Everything in this phase used
k=3, and not by choice** -- it is the `RedHatAI` EAGLE3 checkpoint's own default, carried
untested through every measurement.

### Correcting myself first

I said twice today that the per-position curves showed "k=3 is wasteful" and that k=2 might
dominate. **That was read off the synthetic filler measurements**, where position 2 is
accepted 0.119 of the time. On real content it is 0.29 to 0.50:

| slice | pos 0 | pos 1 | pos 2 |
|---|---|---|---|
| synthetic filler | 0.522 | 0.269 | **0.119** |
| gsm8k/think | 0.705 | 0.475 | **0.291** |
| math/nothink | 0.842 | 0.674 | **0.496** |

Having noticed that, I then said the opposite -- that k should probably go HIGHER than 3.
**That was also wrong, and wrong for a worse reason: it looked only at acceptance and
ignored what drafting costs.** Each extra k is another sequential draft forward pass, so
the overhead grows linearly while acceptance decays geometrically. The two must be modelled
together, and neither of my off-the-cuff remarks did.

### The model, built from measured numbers

Positional acceptance decays with ratio **0.614** between adjacent positions on
`gsm8k/think` (0.291/0.475). Extrapolating, and costing each draft pass at the EAGLE3
head's ~374M read parameters against the fp8 target's 17.0 ms step:

| k | L | draft overhead | realised | predicted speedup |
|---|---|---|---|---|
| 1 | 1.705 | 1.09x | 0.916 | **1.56x** |
| 2 | 2.180 | 1.18x | 0.846 | **1.84x** |
| **3** | 2.472 | 1.27x | 0.785 | **1.94x** |
| 5 | 2.761 | 1.46x | 0.687 | **1.90x** |
| 7 | 2.870 | 1.64x | 0.610 | **1.75x** |

**P5-M1: the optimum is k=3 or k=4, and the curve is FLAT between k=2 and k=5.**
Anything in 2-5 lands within 6% of the best. k=1 and k=7 are clearly worse.

**P5-M2: k=3 measures 1.85 - 2.00x**, consistent with the 1.849x already measured.

**P5-M3: KV cost rises with k**, because activation memory scales with (k+1). Weights are a
fixed 13,863 tokens; at k=3 the total was ~16,368, so roughly 625 tokens per position.
Predicting **15,100 at k=1 rising to ~18,900 at k=7** -- a real effect but small enough that
run-to-run KV variance (measured 15,392 to 17,520 for the same config) may swamp it.

**P5-M4: the best k for a loaded server is LOWER than for a single user.** Larger k crosses
over earlier (P5-B), so if this is measurable it is a Phase 7 input: thinking requests and
ordinary requests would want different k, which makes k a scheduling parameter rather than
a constant.

### What would make this phase's default wrong

If k=5 beats k=3 by more than 5%, every speedup in this phase understates the technique,
because the default was inherited rather than chosen. If k=3 wins, the checkpoint's default
was right and that is worth knowing too -- **a default that happens to be optimal is only
knowable by testing it.**

## P5-M ACTUALS  The k sweep, 2026-08-31. fp8 + EAGLE3, single stream, filler workload.

Baseline with speculation off: 53.5 tok/s.

| k | tok/s | speedup | L filler | L gsm8k | realised | predicted realised | error |
|---|---|---|---|---|---|---|---|
| 1 | 72.5 | 1.355x | 1.542 | 1.754 | 0.879 | 0.916 | -4.1% |
| **2** | **79.6** | **1.488x** | 1.829 | 2.233 | 0.814 | 0.846 | -3.8% |
| 3 | 77.6 | 1.450x | 1.912 | 2.501 | 0.759 | 0.785 | -3.3% |
| 5 | 71.6 | 1.338x | 2.001 | 2.762 | 0.669 | 0.687 | -2.7% |
| 7 | 65.0 | 1.215x | 2.033 | 2.843 | 0.598 | 0.610 | -2.0% |

### The optimum is k=2. The checkpoint's default of 3 is one too high.

**Every measurement in this phase used k=3 because that is what the EAGLE3 checkpoint
ships with, and nobody tested it.** k=2 is 2.6% faster. Modest, but free -- one flag.

More important is the shape, which is exactly what the cost model said and nothing like
"more guessing is better":

    tokens per step        rises monotonically   1.54 -> 2.03 (filler), 1.75 -> 2.84 (gsm8k)
    speedup                peaks at k=2 and falls
    k=7 is 18.3% WORSE than k=2 despite emitting 32% more tokens per step

Guessing further ahead does keep working -- the model really does accept more tokens per
step at k=7 than at k=2. It just costs more than it returns, because **acceptance decays
geometrically with position while draft cost grows linearly with k.**

### The draft-cost model was right to within 4%

P5-M predicted realised efficiency from first principles: the EAGLE3 head reads ~374M
parameters per draft pass against the fp8 target's 17.0 ms step, so k passes cost
`(17.0 + 1.55k)/17.0`. Measured against predicted at every k: **-4.1%, -3.8%, -3.3%,
-2.7%, -2.0%.**

A uniform 2-4% overestimate across the whole range is a model that is right in structure
and slightly optimistic in one constant -- most likely the draft step is a little more
expensive than its parameter read suggests, which is what a per-call launch overhead would
look like. **This is the most accurate prediction of the phase**, and it is the one built
from measured quantities rather than from expectations about how the technique behaves.

### P5-M1 partially correct

Predicted "optimum at k=3 or 4, flat between k=2 and k=5, anything in 2-5 within 6% of
best". Measured optimum **k=2**, one lower. The flat band is real but narrower: k=2 and
k=3 are within 2.6%, while k=5 is 10.1% down and k=1 is 8.9% down. **Flat between 2 and 3,
not 2 and 5.**

### P5-M3 WRONG: KV cache does not move with k at all

Predicted 15,100 tokens at k=1 rising to ~18,900 at k=7, because the verify buffer scales
with (k+1). **Measured 59,680 tokens at k=1, 2, 3, 5 AND 7 -- byte-identical.**

The prediction came from differencing two runs and attributing ~625 tokens per position.
That attribution was wrong, and the reason is worth keeping: **run-to-run variance in
vLLM's startup memory profile is far larger than the effect.** The same fp8+EAGLE3 config
measured 52,080 and 52,224 tokens earlier today and 59,680 now -- a 14% spread on a
quantity I was trying to resolve a 1% effect inside.

The likely mechanism, stated as a hypothesis rather than a finding: peak activation is set
by the **prefill** chunk (`max_num_batched_tokens` 2048) rather than by the verify batch,
which at k=7 is only `max_num_seqs 32 x 8 = 256` tokens. If so, k genuinely does not move
activation and the earlier 1.16 GiB jump was the cost of enabling speculation at all, not a
per-position cost. **That would mean the "memory cost scales with k" correction written in
P5-A1 is itself wrong** -- and it is recorded here rather than quietly dropped, because it
was stated confidently and repeated.

Resolving it needs a direct measurement of activation against `max_num_batched_tokens`,
which is a Phase 6 question, not worth another box start now.

### What this changes

- **Use k=2, not the checkpoint's k=3.** Worth 2.6%.
- Every speedup reported in this phase was measured at k=3 and is therefore a slight
  UNDERSTATEMENT of what the technique can do.
- k is not a memory dial. It is purely a compute-vs-acceptance trade, which simplifies the
  Phase 7 scheduling question: k can be varied per request without touching the cache.

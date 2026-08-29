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

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


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

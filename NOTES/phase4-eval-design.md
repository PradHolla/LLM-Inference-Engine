# Phase 4 experiment design -- quantization

Written 2026-08-28, **before any Phase 4 measurement**, which is the whole point. Phase 3
lost three predictions to the same mistake -- predicting the effect of raising a ceiling
without checking whether anything was pressing against it. This document exists so that
mistake cannot be made a fourth time, and so the quality thresholds are chosen in ignorance
of the results rather than after seeing them.

`NOTES/PROJECT.md` remains canonical for what the project is. This is the protocol for one
phase, in the same role `infra/vllm-runbook.md` plays for Phase 3.

---

## 0. The finding that reshapes the phase

Phase 3 measured vLLM's KV cache at **33,424 tokens**. The standard benchmark request is
512 prompt + 64 output = 576 tokens, so KV holds **58 concurrent requests**. Capacity was
5.70 req/s at a mean E2E near 3 s, which by Little's Law is about **17 requests in flight**.

    KV utilisation at the knee = 17 / 58 = 29%

Quantization's headline benefit is more KV room. **Tripling a resource that is 29% used
buys nothing.** Run Phase 4 on the Phase 3 workload and every weight-quantization result
will come back at roughly zero, and the phase will have measured its own workload rather
than the technique.

So the first deliverable of Phase 4 is not a checkpoint. It is a workload in which KV is
the binding constraint. See section 4.

---

## 1. What Phase 4 must not do

The failure mode for a quantization phase is not a crash. It is a table of throughput
numbers with no quality column, from which the reader concludes quantization is free.
`PROJECT.md` section 5b already records why that conclusion is wrong: the average quality
hit is small, and the damage concentrates in precise multi-step work. A model can lose 1%
on MMLU and 8% on GSM8K.

Three axes, every configuration, or the row is not reported:

| axis | metric | instrument |
|---|---|---|
| throughput | capacity req/s at the TTFT knee | `tools/bench.py` (unchanged) |
| KV capacity | `GPU KV cache size` from the server's own log | journalctl, per launch |
| quality | paired per-item agreement against bf16 | `tools/qualeval.py` (to build) |

---

## 2. Why the obvious quality eval cannot work

**Run 200 questions on bf16, run them on fp8, compare accuracy.** This is what everyone
does and it cannot detect the effect it is looking for.

### 2a. An unpaired comparison has no power at this effect size

Accuracy from n items at true rate p has standard error `sqrt(p(1-p)/n)`. At n=200, p=0.75
that is 3.06%, so two runs of the *same* model differ by `3.06 x sqrt(2) = 4.3%` from
sampling alone. Section 5b puts fp8's cost under 1%.

    items needed to resolve 1% unpaired = 2 * 1.96^2 * 0.75 * 0.25 / 0.01^2 = 14,400

That is not affordable and never will be. **The design must be paired**: the same items
through both configurations, compared item by item. The test is then McNemar's on the
discordant pairs only -- `b` items bf16 got right and fp8 got wrong against `c` the other
way, with `b | b+c ~ Binomial(b+c, 0.5)` under the null. Concordant items carry no
information and stop diluting the estimate.

### 2b. The model disagrees with itself

Incident 22: changing tensor shape reorders bf16 accumulation, so outputs legitimately vary
with batch composition even at temperature 0. Under continuous batching, batch composition
is a function of arrival timing, which is not reproducible. **Some discordance is baseline
nondeterminism and has nothing to do with quantization.**

Therefore the design has a control, in exactly the role `A-nocache-unique` played in Phase 3:

    C0: bf16 run A vs bf16 run B, same items, same concurrency, different process

`d0`, the discordance rate from C0, is the **noise floor**. No quantization result counts
unless it beats the floor. A treatment that lands at the floor has demonstrated nothing --
which is a valid, publishable outcome and the most likely one for fp8.

This is the single element a naive eval omits, and without it every number in the phase is
uninterpretable.

### 2c. Grading is a second instrument and can be silently wrong

Section 2 of `CLAUDE.md`: assume any new measurement tool is wrong until proven otherwise.
The grader's failure mode is marking a correct answer wrong because extraction missed it --
incident 16, comparing regex captures instead of parsed values.

Mitigations, all mandatory:

- Every item has an exactly checkable integer or short-string answer. No LLM judge; a judge
  is a third noisy instrument measuring the first two.
- The model is instructed to end with `ANSWER: <value>` and extraction is anchored to that.
- **Full completion text is written to the JSONL for every item.** A grading bug is then
  re-runnable offline against saved output and costs no GPU time to fix. This is cheap and
  non-negotiable.
- `unparseable` is its own outcome, never folded into `wrong`. A quantized model that stops
  following an output format is a real finding; a regex that stopped matching is a bug; they
  must be distinguishable without re-running.

### 2d. Truncation masquerades as a wrong answer

A thinking chain that exceeds `max_tokens` yields no answer and grades wrong. If int4 thinks
longer -- a real phenomenon, quantized models become less decisive -- it truncates more and
its accuracy falls for a reason that is not "it reasons worse".

`truncated` is a first-class metric, recorded from `finish_reason`, reported beside accuracy
and never merged into it.

---

## 3. The quality eval

### 3a. Items are synthetic, and that is a feature

Items are generated by `tools/mkitems.py`, not downloaded.

| property | why it matters here |
|---|---|
| answer correct by construction | the generator applies the operations to produce the answer |
| zero contamination | the model cannot have memorised an item minted this morning |
| **chain length is a dial** | the reason for the whole approach -- see 3b |
| no network, no dataset download | box runs `HF_HUB_OFFLINE=1`; nothing new to fetch |
| unlimited n at fixed difficulty | power is bought with items, not with GPU hours |

The cost is representativeness: synthetic word problems are not GSM8K. That is acceptable
because the question is **not** "how good is Qwen3-8B at maths". It is "does fp8 change the
answers bf16 gives". For a paired within-model comparison, contamination-freedom and
difficulty control are worth more than resembling a leaderboard.

The harness reads items from a JSONL file, so a real dataset can be substituted later
without touching the harness. Recommended if budget allows: a 200-item GSM8K subsample as
an external sanity check that the absolute accuracy is in a normal range.

### 3b. Chain length as a dose variable -- the central experiment

Section 5b's project-critical claim:

> A reasoning chain is 1,000+ sequential tokens where each conditions the next, so small
> per-token errors compound.

That claim predicts something much stronger than "accuracy drops a bit", and something
testable: **damage should grow with reasoning length.** So chain length is a controlled
variable, not a nuisance.

Each maths item is a chain of `k` dependent integer operations on a running quantity, phrased
as a scenario. Step `i` consumes step `i-1`'s output, so no step can be skipped and `k` is
genuinely the number of sequential reasoning steps.

    k in {2, 4, 8}, n = 120 items per level, 360 maths items total

A dose-response curve is far stronger evidence of a mechanism than a single accuracy delta.
If excess discordance over the floor rises monotonically with `k`, section 5b's amplification
claim is confirmed on this model. If it is flat, the claim is wrong here and that is the more
interesting result.

### 3c. Slices

| slice | n | thinking | tests | grading |
|---|---|---|---|---|
| **T1** maths-think | 360 | ON | amplification along the chain-length dose curve | exact integer |
| **T2** maths-nothink | 360 (same items) | OFF | the same maths without the chain -- isolates amplification from arithmetic | exact integer |
| **T3** longctx-retrieve | 90 | OFF | the Phase 6 web-search shape; the axis KV quant attacks | exact string |

T1 and T2 run the **identical items**. If quantization damage is amplified by thinking, the
excess discordance over each condition's own floor is larger in T1 than in T2. That is a
within-subjects test of the central claim, and it is the best experiment in the set.

TRAP: T1 and T2 have different base accuracy (thinking off will be much worse on `k=8`), and
discordance rates are not directly comparable across different `p`. Compare
`(d_treat - d_floor)` within each condition, never `d_T1` against `d_T2` raw.

T3 inserts a unique fact at a controlled depth (10 / 50 / 90 percent) in a long filler
document and asks for it back. Depth is a second dose variable and is the classic needle
test, chosen because Phase 6 hands the model retrieved documents for a living.

### 3d. Run conditions, identical for every configuration

- **Fixed concurrency 32.** Not 1. Concurrency 1 would be cheaper per token to reason about
  but costs roughly 2.9 GPU-hours per configuration, and it measures a condition no user is
  ever in. 32 is the deployed condition, and it means the noise floor already contains
  batch-composition nondeterminism -- which is the honest floor.
- **Same item order every run**, from a fixed shuffle seed. Removes one variance source
  without hiding the one being measured.
- `temperature: 0.0`, request `seed` set, `max_tokens` 2048 for T1 and 256 for T2/T3.
- Same server flags across configurations except the one under test, with
  `--kv-cache-memory-bytes` pinned. `--gpu-memory-utilization` is not reproducible: Phase 3
  saw 26,176 and 33,424 tokens from identical launches.

### 3e. Cost

Rough token accounting per configuration:

    T1  360 items x ~800 output tokens = 288,000
    T2  360 items x ~150               =  54,000
    T3   90 items x ~120               =  10,800
                                         -------
                                         353,000 tokens

At B=32 the A10G decode step is `t_mem = 16.39 GB / (600 GB/s x 0.803) = 34 ms` producing 32
tokens, so roughly 850 tok/s: **about 7 minutes of generation per configuration.** Four
configurations plus model loads is well under an hour, near $1.

---

## 4. The capacity-pressure workload

512 prompt / 64 output leaves KV 29% utilised (section 0), so it cannot measure a KV
technique. The replacement is chosen to resemble Phase 6, not to flatter the flag:

    4096 prompt / 1024 output

A web-search turn is a system prompt plus three to five retrieved documents plus history plus
the question -- three to six thousand tokens in -- and an answer with thinking is five hundred
to fifteen hundred out. This is the application's real shape, which is the justification;
that it also makes KV binding is the consequence, not the motive.

    tokens per request  = 4096 + 1024 = 5120
    bf16 concurrency    = 33,424 / 5120 = 6.5
    fp8 weights         = ~87,000 / 5120 = ~17
    int4 weights        = ~106,000 / 5120 = ~21

It is also decode-bound, which the current workload never was. Per request:

    prefill = 2 x 8.19e9 x 4096 / (125e12 x 0.5)     = 1.07 s of GPU
    decode  = 1024 tokens x (34 ms / B=6)            = 5.80 s of GPU
    decode share = 5.80 / 6.87 = 84%

Prefill was the whole story in Phase 3. Here it is 16% of the work, and decode is where
quantization acts.

**TRAP: `--max-model-len` must rise from 4096 to at least 6144.** 4096 + 1024 exceeds the
current setting. Changing it changes the memory reservation, so **the bf16 baseline must be
re-measured at the new setting** -- 33,424 tokens is not transferable to the new
configuration, and reusing it would credit quantization for a launch-flag change.

Both workloads are run for every configuration. The old one is retained precisely because it
is the case where quantization should do nothing; if it moves, the harness is suspect.

---

## 5. Configurations

| id | weights | KV | note |
|---|---|---|---|
| Q0a | bf16 | fp16 | baseline, and half of the noise-floor control |
| Q0b | bf16 | fp16 | second bf16 run, same items -- **the control** |
| Q1 | fp8 | fp16 | sm86 has no native fp8; runs via Marlin dequant. Memory saving yes, tensor-core throughput no |
| Q2 | int4 AWQ | fp16 | pre-quantized checkpoint; calibrated on data unlike this workload |
| Q3 | fp8 | fp8 | stacks the +11% already measured in Phase 3; KV error accumulates with length, so T3 is the slice that matters |

---

## 6. Decision rule -- fixed now, before any data

Chosen on 2026-08-28 with no Phase 4 measurement in hand. Choosing in advance is the entire
point; a threshold picked after seeing the numbers is not a threshold.

1. **fp8 weights become the project default if** its excess discordance over the C0 floor is
   not significant by McNemar exact at alpha = 0.05 on any slice, pooled across chain lengths,
   **and** its mean thinking-token count is within 10% of bf16's.
2. **int4 is adopted only if** it delivers at least 1.5x measured capacity on the
   capacity-pressure workload **and** its excess discordance over the floor is under 5
   percentage points at `k=8`. Otherwise int4 is recorded as **measured and rejected**, which
   section 5b explicitly anticipates and which is a successful outcome for the phase.
3. **Any configuration whose truncation rate exceeds bf16's by more than 2x is disqualified
   regardless of accuracy.** It is silently spending the capacity it just bought.
4. A quality result is reported only alongside `d0`. An excess-discordance number without its
   floor is not a result.

---

## 7. Predictions

Written into `NOTES/predictions.md` under Phase 4 with their arithmetic, before running. The
headline one, so that being wrong is expensive and visible:

**The same technique is worth 1.1x or 2.7x depending only on which workload it is measured
on.** On 512/64 fp8 weights should give roughly **+10%** and on 4096/1024 roughly **2.7x**.

Two mechanisms, and only one is idle on the Phase 3 workload. The **KV-capacity** mechanism
does nothing there, because KV is 29% utilised and batch size is not KV-bound -- the
prefix-caching lesson exactly. The **decode-bandwidth** mechanism still works, because every
decode step reads all the weights however empty the KV cache is; it is just small, since
decode is only 41% of a 64-token request's GPU time.

An earlier draft of this section said "roughly 0%" for the Phase 3 workload. That is correct
for KV quantization and wrong for weight quantization, and writing out the arithmetic in
`predictions.md` is what caught it. Recorded rather than silently edited, because a
prediction that had to be corrected before measurement is evidence the method works.

---

## 8. Build order

| # | task | GPU | owner |
|---|---|---|---|
| 1 | this document | no | Claude, xhigh |
| 2 | `tools/mkitems.py` and hand-verification of a sample | no | Claude |
| 3 | `tools/qualeval.py` -- fixed-concurrency runner, saves full text | no | Claude |
| 4 | offline grader test against synthetic completions | no | Claude |
| 5 | difficulty calibration: bf16 sample, tune `k` for 60-85% base accuracy | yes, small | Claude |
| 6 | re-measure bf16 baseline at `--max-model-len 6144` | yes | implementer |
| 7 | run Q0a/Q0b, establish `d0` | yes | implementer |
| 8 | run Q1/Q2/Q3 on both workloads | yes | implementer |
| 9 | attribution and gap analysis | no | Claude, xhigh |

Steps 1-4 cost nothing and the box stays stopped. Step 5 is the first GPU spend and it is
small; without it the eval risks a ceiling or floor effect that would make every later
comparison meaningless.

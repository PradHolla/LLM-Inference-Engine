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

`d0`, the discordance rate from C0, is the **noise floor**.

CORRECTION to this section as first drafted, which said "no result counts unless it beats
the floor". That conflates two different questions, and only one of them needs `d0`:

| question | test | needs `d0`? |
|---|---|---|
| Is the quantized model **less accurate**? | McNemar asymmetry: `b` (bf16 right, quant wrong) against `c` (the reverse) | **no** |
| Does it **change answers** more than noise does? | discordance rate against `d0` | **yes** |

Symmetric nondeterminism inflates `b` and `c` equally, so it dilutes McNemar's power but
does not bias it -- the accuracy verdict is valid without the floor. The floor is still
required, for three things: interpreting how much actually changed, knowing whether the test
had any power at all, and detecting a configuration that shuffles answers without moving
accuracy. Both numbers get reported; they answer different questions.

`d0` is itself an estimate with an error bar -- at `d0 = 4%` on n=360 its standard error is
1.0 point, so it is `4% +/- 2%`, not exactly 4%. Do not compare a treatment against it as
though it were a constant.

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
- **Extract the LAST `ANSWER:` in the reply, and only from post-thinking content.** A model
  reasoning aloud will write "ANSWER: 42", reconsider, and finish with 37. Taking the first
  match grades the abandoned answer. Related: vLLM only splits reasoning into
  `reasoning_content` when `--reasoning-parser` is set; otherwise the `<think>` block arrives
  inside `content`. **The harness must handle both and record which path it took**, or the
  thinking-token count is silently zero.
- **No loose fallback.** If the format is absent, do not go hunting for the last integer in
  the reply. A fallback that fires more often for one configuration than another silently
  applies a different grading standard to each, which is precisely the class of bug that
  produces plausible numbers instead of errors.
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

### 3a-bis. And one real dataset, for a reason that is not "more data"

Added 2026-08-28: **200 items from the published GSM8K test set.** Not because the paired
design needs more power -- it does not -- but because the synthetic set has one structural
blind spot it can never cover.

Synthetic items are graded by a tool written here, against answers generated here, in a
format invented here. **They cannot detect a broken harness.** If thinking mode is silently
off, the chat template is misapplied, `max_tokens` truncates earlier than intended, or the
extraction regex is subtly wrong, every synthetic number comes back plausible and nothing
raises. That is incidents 2, 3, 10 and 11 -- the instrument producing plausible numbers
rather than an error.

GSM8K has published baselines for this model class. If bf16 lands near the known figure,
the whole pipeline is validated end to end in one shot. If it lands 30 points low, the
harness is broken and no synthetic result should be believed. **It is an instrument check
first and a second item family second.**

    curl -o data/gsm8k-test.jsonl https://raw.githubusercontent.com/openai/\
      grade-school-math/master/grade_school_math/data/test.jsonl

1,319 items, 732 KB, MIT licensed. `data/` is gitignored -- refetchable in one line -- but
the 200 converted items are committed inside `results/phase4-items.jsonl`, so the
experiment is reproducible on an offline box without the download.

**Contamination is certain and does not matter here.** GSM8K predates Qwen3 and is surely
in its training data. But the comparison is paired: bf16 and the quantized model carry
identical contamination, and damage to a recalled answer is still damage. For the
instrument check, the published baselines being compared against carry the same
contamination too. Contamination inflates the absolute score; it invalidates neither use.

TRAP, found while building it: an "answer must not appear verbatim in the question" check
was written, and it flagged 16 of 200 items -- because small integers like 2 and 4 occur
naturally in word problems. A check that fires on 8% of a curated benchmark is a broken
check, not a broken dataset. It was removed. The conversion check that survives is the
useful one: re-extract each answer from the untouched source rationale and compare.

Two other conversion details, both verified: 14 of the 1,319 published answers are
comma-grouped (`1,000`) and are normalised on load, and 2 are negative -- **so the grader
must accept a leading minus sign and strip thousands separators from the model's reply
too.** All finals are integers; none are decimal.

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
| **T4** gsm8k | 200 | ON and OFF | **instrument check** against published baselines, plus a second, natural-language item family | exact integer |

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

- **Pin `--max-num-seqs`, do not merely fix client concurrency.** This is the correction
  that matters most in this section. Firing 32 concurrent requests does **not** give every
  configuration a batch of 32: a maths item is roughly 200 prompt + 2048 max output = 2,250
  tokens, so bf16's 33,424-token cache holds about 14 of them while fp8's ~87,000 holds all
  32. The configurations would then run at different batch sizes, and by incident 22
  different batch sizes change the numerics -- producing discordance that is not quality
  damage at all. **Every quality run pins `--max-num-seqs` to what the SMALLEST budget
  (bf16) can hold: 12 for maths and gsm8k, 6 for longctx.** Verify the achieved value from
  the server's own metrics rather than assuming the flag took.
- **Client concurrency 32, capped by the pin above.** Concurrency 1 would cost roughly 2.9
  GPU-hours per configuration and measures a condition no user is ever in.
- **Same item order every run**, from a fixed shuffle seed. Removes one variance source
  without hiding the one being measured.
- `temperature: 0.0`, request `seed` set. `max_tokens` **is set from calibration, not
  guessed** -- see the trap below.
- Same server flags across configurations except the one under test.

**TRAP: pinning the KV cache to a COMMON value would nullify the entire phase.** Phase 3's
lesson was to pin `--kv-cache-memory-bytes` because `--gpu-memory-utilization` gave 26,176
and 33,424 tokens on identical launches. Carried over carelessly, that becomes "give every
configuration the same KV budget" -- which hands the quantized configurations bf16's cache
and deletes the KV-capacity mechanism being measured. The result would be a clean, confident,
completely artificial null.

    pin PER CONFIGURATION, to a value measured once for that configuration.
    NOT to one value shared across configurations.

The pin buys reproducibility between repeat launches of the same configuration. It must
never be used to equalise budgets across configurations.

**TRAP: `max_tokens = 2048` for the thinking slice is a guess, and a wrong guess is
invisible.** A k=16 chain with thinking may well exceed it. If it does, the item grades wrong
for running out of room rather than for reasoning badly, and if a quantized model thinks
longer it truncates more and looks less accurate for the wrong reason. **Calibration must
report the thinking-token distribution at k=16 and `max_tokens` must be set above its p99**,
with the truncation rate reported beside every accuracy number regardless.

### 3e. Cost

Rough token accounting per configuration:

    T1  360 items x ~800 output tokens = 288,000
    T2  360 items x ~150               =  54,000
    T3   90 items x ~120               =  10,800
    T4  200 items x ~600 (think on)    = 120,000
    T4  200 items x ~150 (think off)   =  30,000
                                         -------
                                         503,000 tokens

At B=32 the A10G decode step is `t_mem = 16.39 GB / (600 GB/s x 0.803) = 34 ms` producing 32
tokens, so roughly 850 tok/s: **about 10 minutes of generation per configuration.** Five
configurations plus model loads is a little over an hour, near $1.30.

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
| Q2 | int4 w4a16 | fp16 | `RedHatAI/Qwen3-8B-quantized.w4a16`, built with llm-compressor for vLLM |
| Q2b | int4 AWQ | fp16 | `pytorch/Qwen3-8B-AWQ-INT4` -- **optional but valuable**, see below |
| Q3 | fp8 | fp8 | stacks the +11% already measured in Phase 3; KV error accumulates with length, so T3 is the slice that matters |

**Q1 needs no download.** vLLM quantizes bf16 weights to fp8 at load time via
`--quantization fp8`. Verify this on the box before planning around it.

**The int4 checkpoint choice may dominate the int4 result, which is why there are two.**
`PROJECT.md` section 5b warned that a pre-quantized checkpoint is "calibrated on data unlike
your workload"; the two available checkpoints make that concrete. `pytorch/Qwen3-8B-AWQ-INT4`
is calibrated on **ten samples from `mmlu_abstract_algebra`**, and its own card reports 56
against bf16's 58 on that very task -- the one it was tuned for. That is a demonstration
checkpoint, not a serving one. `RedHatAI/Qwen3-8B-quantized.w4a16` is built with
llm-compressor specifically for vLLM and is the primary.

Running both turns a caveat into a measurement. **If the two int4 checkpoints differ from
each other by more than either differs from bf16, then "int4 quality" is not a property of
int4 at all -- it is a property of whoever calibrated the checkpoint.** That would be the
most useful single result in the phase, and it costs one extra download and 25 minutes.

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
| 2b | GSM8K conversion, faithfulness check against source rationales | no | Claude -- DONE |
| 3 | `tools/qualeval.py` -- runner, offline regrade, paired McNemar | no | Claude -- DONE |
| 4 | instrument verification, end to end, no GPU | no | Claude -- DONE |
| 5 | difficulty calibration: bf16 sample -- tune `k` for 60-85% base accuracy AND record the thinking-token distribution to set `max_tokens` | yes, small | Claude |
| 6 | re-measure bf16 baseline at `--max-model-len 6144` | yes | implementer |
| 7 | run Q0a/Q0b, establish `d0` | yes | implementer |
| 8 | run Q1/Q2/Q3 on both workloads | yes | implementer |
| 9 | attribution and gap analysis | no | Claude, xhigh |

Steps 1-4 cost nothing and the box stays stopped. Step 5 is the first GPU spend and it is
small; without it the eval risks a ceiling or floor effect that would make every later
comparison meaningless.


---

## 9. Design review, 2026-08-28, before writing the runner

Held deliberately between designing the experiment and building the instrument, because a
flaw found here costs nothing and the same flaw found after a sweep costs money and a rerun.
Eight findings; the first three would each have damaged the phase.

| # | finding | severity | status |
|---|---|---|---|
| 1 | Pinning `--kv-cache-memory-bytes` to a **common** value across configurations deletes the KV mechanism and guarantees a null result | **critical** | fixed, section 3d |
| 2 | Fixed client concurrency does not fix batch size -- bf16 fits 14 maths items, fp8 fits 32, and by incident 22 that alone changes answers | **major** | fixed, pin `--max-num-seqs` per slice |
| 3 | At a flat 90 items per level, a true 2-point drop at k=16 is ~1.8 asymmetric items out of ~5 discordant. Unresolvable | **major** | fixed, items reallocated 30/60/120/150 |
| 4 | "Nothing counts unless it beats the floor" conflated the accuracy question with the stability question. McNemar is unbiased under symmetric noise | design error | fixed, section 2b |
| 5 | `max_tokens 2048` for k=16 thinking was a guess; truncation would be read as reasoning failure | **major** | fixed, calibration now sets it |
| 6 | The int4 checkpoint may dominate the int4 result. One available checkpoint is calibrated on ten samples of abstract algebra | **major** | turned into a measurement, Q2 vs Q2b |
| 7 | `ANSWER:` written mid-thinking and later revised would be graded instead of the final answer | moderate | fixed, last-match, post-thinking only |
| 8 | `d0` was treated as an exact constant; at n=360 it is `4% +/- 2%` | minor | recorded, section 2b |

**Cost of the review: zero GPU-seconds.** Findings 1 and 2 would both have produced clean,
confident, entirely artificial numbers -- the failure mode section 10 of `CLAUDE.md` names as
the one that destroys a project like this one. Neither would have raised an error.

### Still open, to resolve on the box before the real runs

- Does `--quantization fp8` work on sm86 in vLLM 0.27.1, or is a pre-quantized checkpoint
  needed for Q1 too?
- Does `--reasoning-parser` need setting for Qwen3 on this version, and does `bench.py`'s
  existing `reasoning_content` handling already imply it does not?
- Confirm the published GSM8K figure for Qwen3-8B from the model card before reading any
  deviation as a harness bug (`predictions.md` P4-7).


---

## 10. Instrument verification, 2026-08-28

`CLAUDE.md` section 2: assume any new measurement tool is wrong until proven otherwise.
`bench.py` shipped with two bugs that produced plausible numbers rather than errors, so
`qualeval.py` was not trusted until it had been run end to end against known ground truth.

### 10a. Unit level -- `qualeval.py selftest`

19 extraction cases and 6 exact-McNemar values, no server. The extraction cases are the
adversarial ones, not the happy path: `ANSWER: 42` followed later by `ANSWER: 37` must yield
37; `<think>ANSWER: 42</think> ... ANSWER: 37` must ignore the abandoned answer inside the
thinking block; `The answer is 42.` must yield **nothing**, because hunting for a loose
integer is the fallback that would grade two configurations by different standards.
The McNemar values were checked by hand: `b=8, c=2` gives `2 x (45+10+1)/1024 = 0.109375`.

### 10b. End to end -- against a fake server with a CONSTRUCTED difference

A throwaway OpenAI-streaming server was written whose correctness is controllable per item.
Two runs were then set up to differ in a way whose exact answer was known in advance:

    server A   accuracy 0.8, seed 1, thinking via reasoning_content
    server B   same seed, same items, with 10 items KNOWN to be correct in A inverted,
               3 items truncated, and thinking via inline <think> tags

    predicted: b = 10, c = 0, McNemar p = 2/2^10 = 0.001953125

Measured on the flipped slice: **b = 10, c = 0, p = 0.0020.** The truncated items surfaced
as `truncated 0.8%` and `unparseable 0.8%`, separately from wrong, and the three that
produced no output at all were marked `empty` rather than silently graded as incorrect.
Both thinking paths were exercised in one test -- 360 records on `reasoning_content` and 360
on `inline_tags` -- which is the failure that would otherwise report a thinking-token count
of zero on whichever path was not implemented.

### 10c. Two defects the verification found

- **The unpaired-record warning never fired.** It was conditioned on the pair count
  differing from *both* input totals, so a run where only B lost records reported nothing.
  A silently shrinking denominator is how a biased sample reaches a conclusion. Fixed to
  warn whenever either side loses records, naming both counts.
- **The batch-size probe returned `med 0 max 0`, and the first explanation for it was
  wrong.** It was written off as an artifact of a fake server answering in microseconds; a
  second run with an injected per-token delay returned `med 0 max 0` again, so that
  explanation had been recorded before it was tested. The real cause is the probe's sampling
  interval: it polls once per second, and a fake-server pass finishes inside one interval, so
  the only sample taken is the one before any request has landed. Fixed by sampling four
  times a second and, more importantly, by **reporting the sample count and warning when
  every sample reads zero** -- because a probe that reads zero looks identical whether the
  metric name is wrong, the endpoint is missing, or the server is genuinely idle, and on the
  real box that ambiguity is expensive. This is incident 5 in miniature: the observation was
  believed before it was checked.

### 10d. What this verification still cannot cover

The fake server is deterministic by construction, so `check-determinism` passing against it
proves only that the code path runs. **The real check is on the box**, against vLLM, where
Qwen3's `generation_config.json` sets `temperature 0.6 / top_p 0.95` and the question of
whether the request's `temperature: 0` overrides it is genuinely open. If it does not, every
quality number from that server is noise. Run `check-determinism` first, every session,
before anything else.
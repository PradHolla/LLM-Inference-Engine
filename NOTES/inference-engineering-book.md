# Inference Engineering (Baseten, 2026) — reference map

*Philip Kiely, Baseten Books. Knowledge cutoff January 2026.*

**Fetch one section, not the book.** Every numbered section is its own Markdown file, 2-8 KB;
chapters are 7-49 KB; the concatenated text is ~90k tokens and is the wrong default. The URL
scheme is regular, so a section is addressable directly from the table in §1 below:

    BASE=https://www.baseten.co/inference-engineering/book
    curl -sSL $BASE/05-techniques/5.2-speculative-decoding.md      # one section
    curl -sSL $BASE/05-techniques.md                               # one chapter
    curl -sSL $BASE/overview.md                                    # expanded TOC, all 42 sections
    curl -sSL $BASE/appendix-a-inference-glossary.md               # ~230 terms

| Resource | URL |
|---|---|
| Section index (start here) | <https://www.baseten.co/inference-engineering/llms.txt> |
| Expanded table of contents | `$BASE/overview.md` |
| Complete text, one file | <https://www.baseten.co/inference-engineering/llms-full.txt> (341 KB, 3,631 lines) |

**Coverage: complete.** Read 2026-09-19, all 3,631 lines — preface, chapters 0-7, Appendix A
(glossary, ~230 terms), Appendix B (recommended reading), acknowledgements. Chapters 6.3-6.6
(ASR, TTS, image, video) carry nothing for this project and are not mapped below, but they
were read; the few cross-modality points that do transfer are noted where they land.

**This file is not a summary.** A summary of a book that is one `curl` away is worthless.
What is recorded here is the JOIN: where the book independently confirms something this
project measured, where our numbers refine or contradict a claim it states generally, where
it names something built here without a name, what it says exists that we have not done, and
the experiments it makes worth running.

**The structural difference, in one line.** The book is organised by TECHNIQUE and its
numbers are rules of thumb or H100-class figures. This project is organised by MEASUREMENT
and its numbers are A10G-specific. The book is the map; `predictions.md` is the survey of one
piece of terrain. Neither substitutes for the other.

---

## 1. Section index, mapped to our phases

| Book | Topic | Our phase / file | `$BASE/...` |
|---|---|---|---|
| 1.3 | Model selection — ranked above all runtime work | Phase 0, `PROJECT.md` §4 | `01-prerequisites/1.3-model-selection.md` |
| 1.4 | TTFT, TPS, ITL, percentiles, end-to-end vs inference-only | `PROJECT.md` §1, §6; `CLAUDE.md` §5 | `01-prerequisites/1.4-measuring-latency-and-throughput.md` |
| 2.2 | Prefill/decode, chat template, sampling | Phase 1-2, `engine/manual.py` | `02-models/2.2-llm-inference-mechanics.md` |
| 2.4 | Ops:byte, arithmetic intensity, roofline | Phase 1-2, `tools/roofline.py` | `02-models/2.4-calculating-inference-bottlenecks.md` |
| 2.5 | FlashAttention, PagedAttention, attention variants | Phase 2-3 (consumed, not built) | `02-models/2.5-optimizing-attention.md` |
| 3.1 | SMs, tensor cores, VRAM, cache hierarchy | `PROJECT.md` §3 | `03-hardware/3.1-gpu-architecture.md` |
| 3.2 | Architecture generations, the Ada/Ampere entries | `PROJECT.md` §3 | `03-hardware/3.2-gpu-architecture-generations.md` |
| 3.3 | Instances, interconnect, **Multi-Instance GPU** | **not considered — §3.6** | `03-hardware/3.3-instances.md` |
| 4.1 | CUDA kernels, selection, fusion | background for Phase 2 | `04-software/4.1-cuda.md` |
| 4.3 | vLLM vs SGLang vs TensorRT-LLM | Phase 3 onward | `04-software/4.3-inference-engines.md` |
| 4.5 | Benchmarking, load testing, profiling | `tools/bench.py`, `CLAUDE.md` §5 | `04-software/4.5-performance-benchmarking-and-load-testing.md` |
| 5.1 | Quantization, number formats, quality checks | Phase 4, `PROJECT.md` §5b | `05-techniques/5.1-quantization.md` |
| 5.2 | Speculative decoding, EAGLE, n-gram, temperature | Phase 5 | `05-techniques/5.2-speculative-decoding.md` |
| 5.3 | Prefix caching, KV re-use, placement, chunked prefill | Phase 6, `gateway/` | `05-techniques/5.3-caching.md` |
| 5.4 | Tensor / expert / pipeline parallelism | out of scope — §6 | `05-techniques/5.4-model-parallelism.md` |
| 5.5 | Disaggregation, and its thresholds | out of scope — §6 | `05-techniques/5.5-disaggregation.md` |
| 6.2 | Embedding models | adjacent to `gateway/` retrieval | `06-modalities/6.2-embedding-models.md` |
| 7.2 | Autoscaling, batching, cold starts, queueing | out of scope — §6 | `07-production/7.2-autoscaling.md` |
| 7.4 | Testing, canary deploys, cost, observability | `results/labbench`, incident 41 | `07-production/7.4-testing-and-deployment.md` |
| 7.5 | Client code, session re-use, streaming | `tools/bench.py`, incident 25, §3.7 | `07-production/7.5-client-code.md` |
---

## 2. Where the book independently confirms what we measured

Arrived at here from arithmetic and matched pairs, before the book was read. Independent
agreement is the cheapest validation available.

**Percentiles and split distributions (§1.4.1-1.4.2).** The P50/P90/P95/P99 table and the
insistence that mean latency lies on a right-skewed distribution is `CLAUDE.md` §5's
"p50/p95/p99, never means". Its separation of inference-only from end-to-end is our "TTFT and
ITL are separate distributions". Same rules, reached separately.

**"When inference time is fast but end-to-end is slow, look at infrastructure" (§1.4.2).**
The Phase 7 finding as general advice. `retrieve_then_generate` posts TTFT p50 of 2,112 ms of
which 577 ms is the search round trip — the gateway IS the infrastructure half, and it is the
half that moved.

**The reading-speed argument, confirmed by the book's own constant.** `PROJECT.md` §1 argues
decode speed is not the headline metric because silent reading at ~250 wpm is "5-6 tok/s", and
past ~30 tok/s decode stops being perceptible. The book's glossary gives "approximately a 4:3
token:word ratio" for English. 250 x 4/3 / 60 = **5.56 tok/s**. The derivation stands on an
independent constant.

**Benchmark realism (§4.5).** Four dimensions a simulated workload must match: sequence
lengths, traffic volume and pattern ("jitter traffic to mimic real usage" = our open-loop
Poisson), request contents ("affects performance factors like cache hit rate and draft token
acceptance"), and input parameters. Then: "If you're maximizing benchmark performance against
bad inputs, performance in production won't match expectations."

Phase 5 is that warning with a number on it. Speculative decoding on `bench.py`'s synthetic
filler is **1.453x**. On real content: **2.045x** (math/think), **2.252x** (math/nothink),
**1.849x** (gsm8k/think), **1.189x** (longctx). Benchmarking on filler alone would have
understated the technique by 40% on reasoning and overstated it by 22% on retrieval.

**Spec decoding is decode-only (§5.2).** "Speculative decoding only improves TPS/ITL, not
TTFT." Our longctx slice sends ~4,096 prompt tokens and generates ~32; acceptance of 0.527 is
real and buys a genuine decode speedup diluted to 1.189x end to end by prefill it cannot
touch. The book states it; we have the workload where it bites.

**FP8 is the sweet spot (§5.1.1).** "8-bit floating-point formats are generally the sweet spot
for improving performance without sacrificing quality." `PROJECT.md` §5b reached "FP8 first,
not INT4" from the capacity table before measuring.

**EAGLE was the right draft (§5.2.1, §5.2.3).** The book calls EAGLE "the go-to speculation
algorithm for general use" and explains why an off-the-shelf small model is the wrong draft
("Qwen 0.5B is designed to be a good standalone LLM on cheap hardware, not to speculate draft
tokens"). Phase 5 used EAGLE3. Correct call, now with a reason attached.

**Spec sheet cross-check (§2.4.1, §3.2.2).** Two of our three overlapping `roofline.py` GPU
entries are confirmed by an independent source:

| GPU | Book | `tools/roofline.py` | Verdict |
|---|---|---|---|
| H100 | 989 TF FP16 dense / 3.35 TB/s | 990 TF bf16 / 3,350 GB/s | MATCH (0.1%) |
| L4 | 242 TF FP8 dense / 300 GB/s | 121 TF bf16 / 300 GB/s | MATCH (FLOPS double per halving) |
| L40 | 362 TF FP8 dense / 864 GB/s | l40s: 362 TF **bf16** / 864 GB/s | **CONFLICT — §7** |

The L4 row matters most: the book's own table confirms the **L4 trap** in `PROJECT.md` §3.
Newer card, same 24 GB, half the A10G's bandwidth, so roughly half the decode speed. The book
lists the L4 as "a cheap and convenient way to run small models" and never flags the bandwidth
consequence for decode.

**FP8 weight sizing (§5.4).** "In FP8, loading a billion parameters takes roughly a gigabyte of
VRAM." Qwen3-8B at 8,190,735,360 params: 8.19 GB at FP8; at bf16, 16.38 GB = **15.26 GiB**,
which is the number `roofline.py` predicted and the server reported.

---

## 3. Where our measurements refine or contradict the book

The book's *conclusion* survives in every case. What changes is the reason, the condition, or
the constant.

### 3.1 The Ampere FP8 dequant tax — the book has no entry for this

§5.1 treats quantization as attacking both phases: "Prefill: compute-bound prefill now runs on
lower-precision Tensor Cores with twice the FLOPS. Decode: memory-bound decode now loads half
as much data."

**On sm86 the first half is false.** The A10G has no FP8 tensor cores. vLLM runs FP8
checkpoints through Marlin dequant, so the memory saving is real and the compute saving is
negative. Measured twice, independently:

- **7.6 points of decode bandwidth efficiency** (Phase 4/6)
- **14% of prefill throughput** — predicted cold prefill slope 0.2556 ms/token from the bf16
  fit divided by vLLM's 1.21x; measured **0.2912**. vLLM+fp8 prefills only 6% faster than bf16
  did on our own `engine/manual.py`, not the 21% the engine speedup alone implies.

The single most useful thing this project knows that the book does not cover, and it exists
because our hardware is the hardware the book calls legacy — glossary, *Ampere*: "An older
NVIDIA GPU architecture still used in legacy or small-scale deployments." Anyone deploying an
FP8 checkpoint on an A10/A100 from §5.1 would predict a prefill win and measure a prefill loss.

### 3.2 A prefix cache hit is 29x cheaper, not free

§5.3.1: "re-using cached tokens takes very little compute power or time." True, and the
constant is worth having. Measured over one conversation grown 25 turns against a control
verified at 0.000 hit rate at every turn:

    WARM slope   0.0100 ms per prompt token
    COLD slope   0.2912 ms per prompt token
    slope ratio  29x
    turn 25 (7,761 tokens of context):  165.7 ms warm vs 2,290.8 ms cold

**The warm curve is not flat and should not be predicted flat.** Caching removes the
*recomputation* of old tokens; each turn's new tokens must still attend over every cached
token. The 0.0100 ms/token slope is that residual. A system designed on the assumption that
cache hits are free will mis-budget TTFT growth on long conversations.

### 3.3 INT4 damage is conditional on whether the model gets to reason

§5.1.1, flatly: "integer formats are not suitable for quality-sensitive workloads due to their
lack of dynamic range."

Measured on Qwen3-8B int4 w4a16, damage as a fraction of what bf16 got right on the same
32-step problems: **1% with thinking ON, 37% with thinking OFF.** On verbatim long-context
retrieval, **270/270 perfect** across bf16, fp8 and int4 alike.

The conclusion holds — do not ship int4 for quality-sensitive work — but the mechanism is the
opposite of the intuitive one. Thinking does not amplify quantization damage, it **absorbs**
it: given room to reason the model catches and repairs its own perturbed arithmetic; denied
that room the errors propagate straight to the answer. `PROJECT.md` §5b carries the original
wrong paragraph with the correction beneath it.

The consequence is about eval design and is sharper than anything in §5.1.3: **a thinking-only
eval would have reported int4 as nearly free.** The damage lives in the condition you would not
think to test.

### 3.4 The speculation crossover is workload-shaped, not batch-shaped

§5.2: "speculative decoding is most useful at low batch sizes where there are spare compute
cycles. At higher batch sizes, speculative decoding must be dynamically disabled as compute is
too saturated to afford verification."

Measured, fp8 + EAGLE3, open-loop, `--max-num-seqs 32`:

| workload | crossover |
|---|---|
| 512 in / 64 out | **sign flips between 4 and 6 req/s** — 1.87x ahead at 4, 2.8x behind at 6 |
| 4096 in / 1024 out | **no crossover** at any rate up to the KV ceiling |

Same GPU, model, draft and day. The rule is real on one shape and absent on the other. Caveat
from incident 12: the sweep was 2,4,6,8,10, so the flip is located only to within 4-6.

Relatedly, **there is no single "speedup of speculative decoding" for this system** — 1.19x to
2.25x on one box, decided entirely by request shape.

### 3.5 Our hardware is less bandwidth-bound than the book's

ops:byte, computed not recalled:

    H100 @ FP16   989e12 / 3.35e12  = 295
    A10G @ bf16   125e12 / 600e9    = 208

The A10G leaves the memory-bound regime at a **lower** arithmetic intensity than an H100 —
batching saturates its compute sooner. The book's guidance about where techniques invert is
calibrated on hardware that stays memory-bound longer than ours. Expect every crossover to
arrive earlier here, and read 3.4's 4-6 req/s in that light.

### 3.6 The book's answer to "small model" is a fraction of a new GPU, not a whole old one

§3.3.2, Multi-Instance GPU, is the section that most directly challenges this project's
hardware choice, and `PROJECT.md` §3's GPU comparison table never considers it:

> "Rather than running small models on older, lower-performance GPUs, there's a way to run
> these lightweight workloads on fractions of newer, high-performance GPUs."

An H100 splits into as many as seven compute slices and eight memory slices; a 3-slice MIG
gets ~3/7 of the compute and up to 40 GB of VRAM. For an 8B model that is Hopper — native FP8,
FlashAttention 3, TensorRT-LLM support — at a fraction of an H100's price. Every constraint
this project has organised itself around (the Marlin tax in 3.1, staying on vLLM per §6, the
24 GB KV ceiling) is a consequence of choosing a whole A10G instead.

This is not a retraction. AWS G and P families do not expose fractional GPU instances, so MIG
would mean a neocloud and a different infra story, and the project's premise is learning by
hitting real constraints rather than buying past them. But it should be written down that the
book's recommended answer for our exact workload is one we never evaluated, and that the
honest framing of the A10G choice is "cheap, available on our existing AWS quota, and
pedagogically rich", not "correct".

### 3.7 Our TTFT numbers exclude the client overhead the book says is 10% of the budget

§7.5.1: "In a high-performance system with a 300-millisecond P95 end-to-end latency SLA, a TLS
handshake costs at least ten percent of that latency budget before inference even starts."

Our TTFT SLO is **250 ms** without search — tighter than the book's example. And since incident
25 we run every client on the box against `localhost:8000`, which was the right call (the
laptop client dropped 22% of 4-minute streams with `ReadError` while the GPU kept generating
for dead connections) but has a consequence never stated: **every TTFT number in this project
is measured with zero TLS handshake, zero internet RTT and a reused local connection.** They
are a lower bound on what a real user would see, not an estimate of it. The book's 10% figure
says the gap is material at our SLO.

### 3.8 The instrument-failure mode the book does not name

§4.5.2 asks for realistic and consistent benchmarks and repeated runs to smooth outliers. It
never warns about the failure that has cost this project the most: **an instrument that
produces plausible numbers rather than errors.**

The sharpest case sits exactly on the book's own recommended path. `bench.py` counted one token
per SSE chunk, correct for four phases. Speculative decoding emits a whole accepted run in one
chunk. The counter then undercounted output by exactly the acceptance factor and **reported a
1.43x speedup as 1.5x slower.** Nothing raised. Every number was plausible. A reader who
follows §4.5, builds a benchmark, then adopts §5.2's speculation hits this.

General form, now `CLAUDE.md` §2: a metric's definition is an ASSUMPTION about the server, and
a new technique can invalidate it silently.

Incident 50 is the eval-side twin: gsm8k "accuracy" of 56.5/56.0/52.5 across three arms was a
truncation rate; among responses that finished, all three scored 99-100%. The book recommends
eval datasets as benchmark inputs (§4.5.1) without noting that a grader scoring truncation as
incorrect is measuring the token budget, not the model.

---

## 4. Vocabulary it gives us for things already built

The terms a reader or an interviewer will use.

- **Context engineering for cache hits (§5.3.1):** "To take advantage of prefix caching, ensure
  that novel tokens are as late in your context as possible." One sentence, three things here:
  (a) the design rule behind the gateway's splice, where the continuation is
  `prompt_sent + generated + cue(block)`, a strict extension; (b) the reason incident 45
  happened, when a cold-cache control put noise on the new user message and the history
  rendered *before* it, so the cache hit normally and warm and cold came out 1.0x apart against
  a predicted 21x; (c) what the Phase 7 counters measured — overlap reuses 3,008 of 185,109
  tokens (1.6%), `generate_then_retrieve` 23,184 of 208,500 (11.1%), because the 3,000-token
  block is novel every time whichever arm pays for it.
- **Chunked prefill (glossary):** "Splits long inputs into chunks and overlaps prefill with
  decode or other work, preventing single long sequences from monopolizing resources." That is
  precisely the admission-stall symptom Phase 2 measured at ITL p95 201-213 ms against 50 idle.
- **Conditional disaggregation (§5.5.1):** the decode engine checks whether the input is already
  cached or short enough to handle locally before shipping it to a prefill engine.
  Structurally the same decision the gateway makes about whether to search.
- **Continuous batching / in-flight batching (§7.2.1):** what `engine/continuous.py` implements,
  against the static and dynamic batching it is compared with.
- **Perceived TPS vs Total TPS (§1.4):** the ambiguity `PROJECT.md` §1 works around by naming
  ITL explicitly.
- **Queue depth (§7.4.3):** the observability metric our lab bench reads as
  `num_requests_waiting`, and the one incident 41 bound wrongly.
- **Shadowing (§4.5):** copying production traffic onto a test system. The gold standard we
  have no access to; real content slices are the substitute.
- **ISL / OSL, xPyD, G1-G4, active-active:** input/output sequence length; five prefill three
  decode engines; the VRAM / host RAM / local SSD / networked SSD cache hierarchy; both
  clusters serving live traffic.

---

## 5. Experiments the book makes worth running

Ranked by what they would teach per GPU-hour. Rough expectations are given so the shape of each
is clear; a REAL prediction with its derivation goes into `predictions.md` before any run, per
`CLAUDE.md` §1. Each is a single-variable matched pair against a control we already have.

### E1. Temperature vs speculative acceptance — this one may revise a standing headline

§5.2: "The big one is temperature — higher temperatures yield token distributions that are
harder to predict, reducing the effectiveness of speculative decoding."

**Every Phase 5 number was measured at temperature 0.0.** `tools/qualeval.py:27` and
`tools/bench.py:130` both ship `temperature: 0.0`, and incident 52 records that every phase
since 4 sent it. Greedy decoding is the single most favourable condition possible for
speculation — the target's distribution is a point mass, so a draft token either matches the
argmax or does not, with no sampling noise to break the run. Phase 7 then moved to the vendor
config (temp 0.6 / top_p 0.95 / top_k 20 / min_p 0) because the Qwen3-8B card forbids greedy.

So the 2.045x / 2.252x / 1.849x headline is an **upper bound measured under a config we no
longer use and the vendor tells us not to use.** We have never measured speculation at the
temperature we actually serve at.

Design: acceptance L and tok/s against temperature 0.0, 0.3, 0.6, 1.0 on `math/think`, fp8 +
EAGLE3, concurrency 1, everything else held. Expect acceptance to fall monotonically; the open
question is the slope, and whether 2.045x survives at 0.6 or collapses toward the 1.45x filler
figure. Cheap — one slice, four arms, no new code beyond passing `--temperature`.

This also closes the greedy-decoding thread from incidents 52-53 on the speculation side, where
it has only been closed on the truncation side.

### E2. N-gram speculation on retrieval — the book names a fix for our one measured failure

§5.2.4: "The acceptance rate for n-grams is only high when the contents of the model output are
similar to the model input... within this specific domain, it easily outperforms EAGLE."

The Phase 7 retrieval slice is exactly that shape: the answer is lifted out of the retrieved
block. EAGLE returned **1.189x on longctx**, the one workload where it effectively failed, and
P5-I attributed that to prefill dominance rather than to acceptance. N-gram speculation builds
its dictionary *from the prompt*, so a long retrieved block is an asset rather than dead weight.

Design: vLLM ngram speculative config vs EAGLE3 vs no-spec, three arms, on `retrieval` and
`longctx`, with `math/think` as the negative control where n-gram should lose. Expect n-gram to
beat EAGLE on both retrieval shapes and lose on math. If it holds, the phase-5 conclusion
becomes "choose the speculator by workload shape", which is a better result than the one we
have.

### E3. L4 vs A10G — native FP8 against the Marlin tax, one variable

Already an open question in `PROJECT.md` §10, and §3.1 above is what makes it interesting now.
`g6.xlarge` is an L4: Ada sm89, **native FP8**, same 24 GB VRAM, 300 GB/s against the A10G's
600. The only two differences are FP8 hardware support and bandwidth, and we have measured what
the missing hardware support costs us — 14% of prefill, 7.6 points of decode efficiency.

Design: identical fp8 config, identical slices, both boxes. Expect the L4 to lose decode badly
(half the bandwidth, and `PROJECT.md` §3 calls it a trap for exactly that) and to **win
prefill**, because it is not paying the dequant tax. The interesting case is TTFT, which is the
metric §1 says users actually feel: if native FP8 buys enough prefill, the "trap" GPU could win
the SLO that matters while losing the number everyone quotes. That would be a genuinely
counterintuitive, well-controlled result, and it is one instance-hour to find out.

This is also the closest we can get to testing §3.6's MIG argument on AWS.

### E4. Chunked prefill ablation — closes a claim we have been making without evidence

vLLM 0.27.1 ships `enable_chunked_prefill=True`, so we have had it on since Phase 3 and never
measured it. `PROJECT.md` §1 names it alongside prefix caching as a felt-latency lever, and
Phase 2 built the case for it (`predictions.md` 1190-1284: admission stalls, ITL p95 201-213 ms
against 50 idle, capacity capped at 1.6 of a possible 6.74 req/s).

Design: mixed traffic — a stream of short decodes plus one 4k-token prefill — with the flag on
and off. Expect short-request ITL p95 to degrade sharply with chunking off while the long
request's own TTFT improves slightly. Cheapest experiment on this list: one flag, two arms.

### E5. Does perplexity understate quantization damage?

§5.1.3 names three quality checks — perplexity, intelligence benchmarks, custom evals — and
recommends running all three. Phase 4 ran only task accuracy, and incident 50 showed task
accuracy can silently be a truncation rate.

The interesting part is not "add perplexity", it is a testable disagreement. **Perplexity is
teacher-forced**: it scores the likelihood of a fixed reference sequence and never lets an
error compound into the next token. That is precisely the mechanism behind §3.3's result, where
int4 loses 37% without thinking and 1% with it. So perplexity should rank the formats correctly
and **understate int4's damage badly**, because the regime where int4 breaks is autoregressive
error propagation that perplexity structurally cannot see.

Design: perplexity of bf16 / fp8 / int4 on a fixed corpus, compared against the Phase 4
accuracy deltas already recorded. If perplexity shows a small uniform gap where accuracy shows
37%, that is a finding about the book's own recommended check, and it is nearly free to get.

### E6. KV cache offload to host memory

§5.3.2's G1-to-G2 move. `g5.2xlarge` was chosen over `g5.xlarge` for 32 GiB of system RAM
specifically to leave room for this. Would extend prefix-cache lifetime across conversations,
which is the chat app's actual access pattern — the P6L-2 harness already measures warm against
cold, and this adds a third arm: evicted from VRAM, restored from host. Higher setup cost
(LMCache), so it ranks below the five above.

### E7. Push context to the ceiling

§5.3.4: "In your performance benchmarking, be sure to send very large input sequences." Our
longctx slice is ~4,096 tokens against a 16,384 `--max-model-len` we have never approached.
At the measured cold slope of 0.2912 ms/token, a full 16k prefill is ~4.8 s of TTFT against a
1.2 s SLO. One run to find where the app actually breaks, and it doubles as the long-context
quality check §5.1 says KV quantization most endangers.

### Also noted, lower priority

- **Structured output / logit biasing (§2.2):** guided decoding would make the grader's
  unparseable rate structurally zero, removing a confound from the incident 28/50 family, at
  some per-token masking cost worth measuring.
- **Profiling (§4.5.3):** we have never run PyTorch Profiler or Nsight. Every "why" here has
  been answered by arithmetic and matched pairs, which is cheap and has worked — but the
  book's own criterion for when profiling is warranted, "writing your own inference service in
  PyTorch", describes `engine/` exactly.
- **Concurrency target (§7.2.1):** "the concurrency target and the batch size should match."
  `PROJECT.md` §1 lists "users served before p95 SLO breaks | measured, not guessed" as a
  success criterion; worth checking whether that number was ever actually produced.
- **Model selection (§1.3):** "the most important decision in model performance optimization
  isn't the runtime engine or speculation algorithm, it's which model you choose." Qwen3-8B was
  fixed in Phase 0 and never revisited across seven phases of runtime work. The book ranks that
  one decision above everything we have done since.

---

## 6. What it says to do that is correctly out of scope — with the book's own thresholds

These read as gaps and are not. The book supplies the criteria that exclude them:

- **Disaggregation (§5.5.2):** reach for it only above "one hundred million to one billion
  tokens per day", at "at least a hundred billion parameters", on prefill-heavy traffic. We are
  one GPU and 8B. Two of three criteria fail by three orders of magnitude.
- **TensorRT-LLM (§4.3.3):** "Use TensorRT-LLM when you are running a well-supported model
  architecture on a **Hopper or later** GPU." sm86 is neither. §4.3.1 agrees explicitly: use
  vLLM when "you are using a smaller GPU or older architecture where TensorRT-LLM offers few
  performance benefits."
- **Multi-GPU parallelism (§5.4):** needs more than one GPU, and §3.2.2 notes Lovelace and by
  extension our class of card have no NVLink, so even a second A10G would parallelise through
  an inefficient path. The book also says TP is for within-node and PP "is not recommended".
- **Autoscaling, cold starts, cache-aware routing, scale to zero, multi-cloud, canary deploys
  (§7.2-7.4):** all require replicas. On a single box these are unmeasurable, not merely
  unbuilt.
- **MoE (§2.2.4):** "Models under 32B parameters, and especially models under 8B parameters,
  tend to use traditional dense architectures efficiently." Qwen3-8B is dense; nothing missed.

---

## 7. Claims to verify rather than adopt

- **L40 vs L40S FLOPS.** §3.2.2 lists "L40: 362 teraFLOPS FP8 dense, 48 GB, 864 GB/s".
  `tools/roofline.py` lists `l40s` as 362 TFLOPS **bf16** dense at the same 864 GB/s. The same
  number in two different roles is a tell that at least one is wrong, and L40 and L40S are
  different SKUs besides. If `g6e.xlarge` is ever priced (open in `PROJECT.md` §10), resolve
  against the NVIDIA datasheet first. If the book is right about dense FP8, our bf16 figure is
  2x high and every L40S roofline row is optimistic.
- **A10G 125 TFLOPS bf16 dense.** The whole roofline rests on it and the book has no A10G entry
  to cross-check. Note that `compute_eff` (measured 0.30 at 113 tokens rising to 0.53 at 6,407)
  is fitted *against* this constant, so an error in it is absorbed silently rather than
  surfacing. Worth one lookup.
- **"Quantization down a single level of precision generally offers 30 to 50 percent better
  performance" (§5.1).** True with native support for the target format. Not true on sm86 —
  §3.1.
- **Knowledge cutoff January 2026.** vLLM flag names, defaults and the V1/V2 model runner split
  have moved since. `CLAUDE.md` §1b stands: check `--help=all` and the request schema, not the
  book and not memory.

---

## 8. The one-paragraph take

The book is a good map of a field this project has been learning by walking one small part of
it, and it confirms more of our working standards than it contradicts. Its value here is
partly negative space — it names the techniques that need scale we do not have and gives
thresholds saying plainly we are right not to have them, which is worth more than another
technique to chase. What it cannot give is the thing this repo is for: the book's numbers are
rules of thumb and H100 figures, and every one of ours is a measured A10G constant with a
prediction written down before it. On the points where they disagree — FP8 on Ampere, the
speculation crossover, the cost of a cache hit — the disagreement is ours to keep, because we
have the matched pair and the book has the general claim. The two places it is plainly ahead
of us are §3.6, where its answer for an 8B model is a fraction of an H100 rather than a whole
A10G, and §5.2's note on temperature, which quietly invalidates the conditions under which
this project's speculation headline was measured.

"""
static_batch.py -- Phase 2 step 2: batch engine/manual.py's decode loop, statically.

"Static" means fixed membership: once a batch of B sequences starts, no sequence may
join or leave until the LAST one finishes. A sequence that hits its own target early
(or emits eos) does not free its slot -- it keeps occupying a row in every forward call,
still consuming real compute, until the longest sequence in the batch is done. That
restriction is the entire point of this step, and the ragged-length mode exists solely
to put a number on what it costs. Continuous batching (step 3) is the fix.

Everything here reuses engine/manual.py's GPU-VERIFIED patterns rather than re-deriving
them -- see its docstring for the citations (logits_to_keep detection, explicit
cache_position, BatchEncoding unwrap, dtype=bfloat16). Batching only adds a leading B
dimension to those same tensors; the forward-call shape is the only thing that changes.

PROMPTS ARE UNIFORM (item B): every row in a batch is the same --prompt-tokens prompt,
built with make_prompt exactly as engine/manual.py does. That means there is no padding
and no attention mask to construct -- deliberately, so that ragged OUTPUT length is the
only variable this file measures. Ragged prompt length is a separate problem (padding
waste) that would otherwise be conflated with the finding here.

per_seq_max_tokens[i] counts DECODE-LOOP tokens only, not the token prefill already
produced. That is a deliberate departure from engine/manual.py's max_new_tokens (which
DOES count the prefill token) -- see run_batch's docstring and the note in main() for
why: it is what makes total_steps land exactly on the longest requested length, which
is what NOTES/predictions.md's 2026-08-21 arithmetic assumes throughout.

  /opt/llm/.venv/bin/python -m engine.static_batch --mode uniform
  /opt/llm/.venv/bin/python -m engine.static_batch --mode ragged \
      --lengths 512,32,32,32,32,32,32,32
"""
from __future__ import annotations

import os
# MUST precede `import torch` (transitively, via engine.manual below) -- read once at
# CUDA allocator init and silently ignored after. See engine/manual.py's identical
# comment and NOTES/predictions.md for the measured +29% concurrency this buys.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Import the verified helpers rather than duplicating them. Importing engine.manual
# pulls in torch -- which is why the env line above MUST come first.
from engine.manual import pick_logits_kwarg, make_prompt, pct, min_samples, FILLER

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GIB = 1 << 30


def fmt_gib(b: float) -> str:
    # Copied from tools/kvprobe.py.
    return f"{b / GIB:.3f} GiB"


def mem() -> dict:
    """Copied from tools/kvprobe.py. Read three ways because they measure different
    things -- torch's allocated/reserved views never see the CUDA context, driver
    mem_get_info() is the only ground truth."""
    free, total = torch.cuda.mem_get_info()
    return {
        "alloc": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "driver_used": total - free,
        "driver_total": total,
    }


def iter_kv(cache):
    """Copied from tools/kvprobe.py -- the cache API moved between transformers 4 and
    5, so try each shape rather than pinning a version."""
    layers = getattr(cache, "layers", None)
    if layers is not None:                       # transformers 5.x
        for i, l in enumerate(layers):
            for kind, attr in (("k", "keys"), ("v", "values")):
                t = getattr(l, attr, None)
                if t is not None:
                    yield i, kind, t
        return
    kc, vc = getattr(cache, "key_cache", None), getattr(cache, "value_cache", None)
    if kc is not None:                           # transformers 4.x
        for i, (k, v) in enumerate(zip(kc, vc)):
            yield i, "k", k
            yield i, "v", v
        return
    if isinstance(cache, (list, tuple)):         # legacy tuple-of-tuples
        for i, layer in enumerate(cache):
            yield i, "k", layer[0]
            yield i, "v", layer[1]
        return
    raise RuntimeError(f"unrecognised cache type {type(cache)}")


def cache_bytes(cache) -> int:
    # Copied from tools/kvprobe.py. Whole-batch total (all B sequences' K and V), same
    # convention as kvprobe's try_batch -- not divided by B.
    return sum(t.numel() * t.element_size() for _, _, t in iter_kv(cache))


def collect_eos_ids(tok, model) -> set[int]:
    """Duplicated from engine.manual -- not in the import list this file was told to
    use, so copied rather than cross-imported, same as FILLER/pct/min_samples being
    duplicated (not imported) between engine/manual.py and tools/kvprobe.py already."""
    ids: set[int] = set()
    for src in (getattr(tok, "eos_token_id", None),
                getattr(getattr(model, "generation_config", None), "eos_token_id", None)):
        if src is None:
            continue
        if isinstance(src, int):
            ids.add(src)
        else:
            ids.update(int(x) for x in src)
    return ids


def _f(v: float, w: int, suffix: str = "") -> str:
    # Duplicated from engine/manual.py.
    return f"{'n/a':>{w}}" if v != v else f"{v:>{w}.1f}{suffix}"


# --------------------------------------------------------------------- the batch loop
def run_batch(model, B: int, per_seq_max_tokens: list[int], input_ids: torch.Tensor,
              eos_ids: set[int], fwd_kw: str | None, ignore_eos: bool = True) -> dict:
    """Static batch prefill + decode. All B rows share one prompt length (uniform
    prompts, item B), so cache_position is a single shared arange for the whole batch,
    exactly as in engine.manual.generate_manual -- batching only adds a leading B
    dimension to every tensor already there, not a new code path.

    per_seq_max_tokens[i] counts tokens produced by the DECODE LOOP only. The token
    prefill produces seeds the first decode step's input but is not itself counted:
    that is what makes total_steps land exactly on max(per_seq_max_tokens) with no
    off-by-one, which is what every arithmetic worked example in NOTES/predictions.md
    (2026-08-21, "all 8 slots stay occupied for all 512 steps") assumes.

    STATIC BATCHING: a sequence that reaches its target (or emits an id in eos_ids)
    stops updating its OWN bookkeeping (finished[i], tokens_produced[i]) but its row is
    NOT removed, masked, or skipped -- every subsequent forward call still computes it,
    using whatever token the model itself continues to emit for that row. That wasted
    compute, until the LAST sequence finishes, is the cost this file exists to measure.

    Returns {"prefill_s": float, "decode_s": [float, ...], "finish_step": [int]*B,
    "tokens_produced": [int]*B, "kv_bytes": int}. decode_s has one entry per decode
    step actually executed -- one entry per step, not per token, since a single forward
    call advances all B rows at once.
    """
    assert len(per_seq_max_tokens) == B, "per_seq_max_tokens must have exactly B entries"
    device = input_ids.device
    prompt_len = input_ids.shape[1]
    kw = {fwd_kw: 1} if fwd_kw else {}

    decode_s: list[float] = []
    finished = [False] * B
    finish_step = [0] * B
    tokens_produced = [0] * B

    with torch.inference_mode():
        # PREFILL -- one forward over the whole [B, prompt_len] batch, no cache yet.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache_position = torch.arange(0, prompt_len, device=device)
        out = model(input_ids=input_ids, past_key_values=None, use_cache=True,
                    cache_position=cache_position, **kw)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        prefill_s = t1 - t0

        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(-1)      # [B] -- seeds decode step 1
        pos = prompt_len

        # A target of 0 finishes before any decode step -- bookkeeping only, static
        # batching still computes the row every step below regardless.
        for i in range(B):
            if per_seq_max_tokens[i] <= 0:
                finished[i] = True

        # DECODE -- lockstep, one shared cache_position (uniform prompt length means
        # every row is at the same position). Finished rows are NOT skipped: their
        # slot keeps being computed every step until ALL are finished. That is static
        # batching -- the loop condition below never shrinks B.
        step = 0
        while not all(finished):
            step += 1
            torch.cuda.synchronize()
            ts0 = time.perf_counter()
            step_input = next_id.unsqueeze(-1)          # [B] -> [B, 1]
            cache_position = torch.arange(pos, pos + 1, device=device)
            out = model(input_ids=step_input, past_key_values=past, use_cache=True,
                        cache_position=cache_position, **kw)
            torch.cuda.synchronize()
            ts1 = time.perf_counter()
            decode_s.append(ts1 - ts0)

            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(-1)
            pos += 1
            next_id_list = next_id.tolist()              # one host sync, not B of them
            for i in range(B):
                if finished[i]:
                    continue
                tokens_produced[i] += 1
                hit_eos = (not ignore_eos) and next_id_list[i] in eos_ids
                if hit_eos or tokens_produced[i] >= per_seq_max_tokens[i]:
                    finished[i] = True
                    finish_step[i] = step

        kv_bytes = cache_bytes(past)

    return {"prefill_s": prefill_s, "decode_s": decode_s, "finish_step": finish_step,
            "tokens_produced": tokens_produced, "kv_bytes": kv_bytes}


def compute_metrics(B: int, per_seq_max_tokens: list[int], decode_s: list[float],
                    tokens_produced: list[int] | None = None) -> dict:
    """Defined EXACTLY as specified -- this arithmetic is the deliverable, not the loop
    above it. naive_tok_s and useful_tok_s are equal under uniform lengths; the gap
    between them under ragged lengths, with utilization collapsing while naive_tok_s
    stays high and healthy-looking, IS the finding of this step."""
    total_steps = len(decode_s)                        # driven by the longest sequence
    # ACTUAL tokens produced, not the requested targets. With identical prompts and
    # greedy decoding every row emits the same tokens, so an eos would fire on all B
    # rows at the same step and truncate total_steps while the requested targets stayed
    # high -- yielding utilization above 1.0, which is nonsense. --ignore-eos (default)
    # makes these equal; this formula stays correct even when it is disabled.
    useful_tokens = sum(tokens_produced if tokens_produced is not None else per_seq_max_tokens)
    slot_steps = B * total_steps                        # tokens the GPU actually computed
    decode_time = sum(decode_s)
    utilization = useful_tokens / slot_steps if slot_steps else float("nan")
    naive_tok_s = slot_steps / decode_time if decode_time > 0 else float("nan")
    useful_tok_s = useful_tokens / decode_time if decode_time > 0 else float("nan")
    return {
        "total_steps": total_steps,
        "useful_tokens": useful_tokens,
        "slot_steps": slot_steps,
        "utilization": utilization,
        "decode_time_s": decode_time,
        "naive_tok_s": naive_tok_s,
        "useful_tok_s": useful_tok_s,
    }


def run_warmup(model, B: int, input_ids: torch.Tensor, eos_ids: set[int],
               fwd_kw: str | None) -> None:
    """One short (8-token) generation at the batch size about to be measured, discarded
    before any measured trial. MEASURED 2026-08-21 in engine/manual.py: cuBLAS picks
    kernels per problem SHAPE, so warming at the wrong (batch, prompt_len) warms
    nothing. In sweep mode that means one warmup per batch size, not one for the whole
    sweep -- called fresh inside run_trial for every B."""
    t0 = time.perf_counter()
    run_batch(model, B, [8] * B, input_ids, eos_ids, fwd_kw)
    torch.cuda.synchronize()
    print(f"    warmup B={B} (discarded)... done ({time.perf_counter()-t0:.1f}s)", flush=True)


# ------------------------------------------------------------------- reporting
@dataclass
class Trial:
    mode: str
    batch: int
    prompt_tokens: int
    per_seq_max_tokens: list[int]
    ok: bool
    total_steps: int | None = None
    useful_tokens: int | None = None
    slot_steps: int | None = None
    utilization: float | None = None
    decode_time_s: float | None = None
    naive_tok_s: float | None = None
    useful_tok_s: float | None = None
    prefill_s: float | None = None
    decode_s: list[float] = field(default_factory=list)
    finish_step: list[int] = field(default_factory=list)
    tokens_produced: list[int] = field(default_factory=list)
    kv_bytes: int | None = None
    peak_alloc: int | None = None
    reserved: int | None = None
    alloc: int | None = None
    driver_used: int | None = None
    driver_total: int | None = None


def run_trial(model, mode: str, B: int, per_seq_max_tokens: list[int],
              template_row: torch.Tensor, eos_ids: set[int], fwd_kw: str | None,
              prompt_tokens: int, warmup: bool, ignore_eos: bool = True) -> Trial:
    """Build the batch, warm it up at its own shape, run it, and turn OOM into a
    recorded (not raised) result -- item G's sweep-survival contract. Between every
    trial: empty_cache + reset_peak_memory_stats, so peak_alloc below is this trial's
    peak, not a running high-water mark across the whole sweep."""
    # repeat, not expand: expand's stride-0 view is a read aliased across B, not a
    # tensor the model is guaranteed to accept as a normal batch dimension; repeat
    # costs one small contiguous copy, entirely outside every timed region below.
    input_ids = template_row.repeat(B, 1)

    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        if warmup:
            run_warmup(model, B, input_ids, eos_ids, fwd_kw)
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        r = run_batch(model, B, per_seq_max_tokens, input_ids, eos_ids, fwd_kw, ignore_eos)
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        return Trial(mode=mode, batch=B, prompt_tokens=prompt_tokens,
                     per_seq_max_tokens=per_seq_max_tokens, ok=False)

    m = compute_metrics(B, per_seq_max_tokens, r["decode_s"], r["tokens_produced"])
    mm = mem()
    return Trial(mode=mode, batch=B, prompt_tokens=prompt_tokens,
                 per_seq_max_tokens=per_seq_max_tokens, ok=True,
                 total_steps=m["total_steps"], useful_tokens=m["useful_tokens"],
                 slot_steps=m["slot_steps"], utilization=m["utilization"],
                 decode_time_s=m["decode_time_s"], naive_tok_s=m["naive_tok_s"],
                 useful_tok_s=m["useful_tok_s"], prefill_s=r["prefill_s"],
                 decode_s=r["decode_s"], finish_step=r["finish_step"],
                 tokens_produced=r["tokens_produced"], kv_bytes=r["kv_bytes"],
                 peak_alloc=torch.cuda.max_memory_allocated(),
                 reserved=mm["reserved"], alloc=mm["alloc"],
                 driver_used=mm["driver_used"], driver_total=mm["driver_total"])


def write_jsonl(out, t: Trial) -> None:
    # Flushed per trial, not buffered to the end -- a sweep that only writes at the end
    # loses everything to a crash, a Ctrl-C, or a spot reclaim, and sweeps are exactly
    # long enough for that.
    out.write(json.dumps(asdict(t), default=str) + "\n")
    out.flush()


HDR = (f"  {'B':>4} {'steps':>6} {'util':>6} │ {'ITL p50':>8} {'p95':>8} {'p99':>8} (ms) │ "
       f"{'naive tok/s':>11} {'useful tok/s':>12} │ {'KV':>10} {'peak':>10}  result")


def print_trial_row(t: Trial) -> None:
    if not t.ok:
        print(f"  {t.batch:>4} {'':>6} {'':>6} │ {'':>8} {'':>8} {'':>8}     │ "
              f"{'':>11} {'':>12} │ {'':>10} {'':>10}  OOM")
        return
    decode_ms = [d * 1000 for d in t.decode_s]
    util_pct = t.utilization * 100
    print(f"  {t.batch:>4} {t.total_steps:>6} {util_pct:>5.1f}% │ "
          f"{_f(pct(decode_ms,50),8)} {_f(pct(decode_ms,95),8)} {_f(pct(decode_ms,99),8)}     │ "
          f"{_f(t.naive_tok_s,11)} {_f(t.useful_tok_s,12)} │ "
          f"{fmt_gib(t.kv_bytes):>10} {fmt_gib(t.peak_alloc):>10}  ok")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--mode", choices=["uniform", "ragged"], default="uniform")
    p.add_argument("--batches", default="1,2,4,8,16,24,32,40",
                    help="comma list of batch sizes to sweep, uniform mode")
    p.add_argument("--max-new-tokens", type=int, default=64,
                    help="decode-loop target for every sequence, uniform mode")
    p.add_argument("--lengths", default="512,32,32,32,32,32,32,32",
                    help="comma list of per-sequence decode-loop targets, ragged mode; "
                         "B = len(lengths)")
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--ignore-eos", dest="ignore_eos", action="store_true", default=True,
                   help="do not stop on eos; run exactly the requested token counts (default)")
    p.add_argument("--respect-eos", dest="ignore_eos", action="store_false",
                   help="stop a row on eos. Identical prompts plus greedy means eos fires "
                        "on every row at the same step, truncating the ragged experiment")
    p.add_argument("--out", default="results/phase2-step2.jsonl")
    p.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1
    device = torch.device("cuda:0")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                  device_map="cuda:0")
    model.eval()
    weights_gib = torch.cuda.memory_allocated() / GIB
    print(f"{args.model} loaded in {time.time()-t0:.1f}s -- {weights_gib:.2f} GiB weights "
          f"on {torch.cuda.get_device_name(0)}")

    fwd_kw = pick_logits_kwarg(model)
    print(f"logits kwarg: {fwd_kw or 'NOT FOUND -- logits explosion risk, see kvprobe.py'}")

    # Uniform prompts, no padding -- deliberate, see item B / the module docstring.
    # Always no-think: whether Qwen3 "thinks" or answers does not change what this file
    # measures (raw per-step forward latency and slot occupancy), so there is no CLI
    # flag for it here, unlike engine/manual.py's --think/--no-think.
    body = make_prompt(args.prompt_tokens)
    input_ids = tok.apply_chat_template(
        [{"role": "user", "content": body}],
        add_generation_prompt=True, tokenize=True, return_tensors="pt",
        enable_thinking=False,
    )
    # transformers 5.x returns a BatchEncoding from apply_chat_template(tokenize=True);
    # 4.x returned a bare tensor. VERIFIED in engine/manual.py and baseline/server.py.
    if not hasattr(input_ids, "shape"):
        input_ids = input_ids["input_ids"]
    template_row = input_ids.to(device)
    prompt_len = int(template_row.shape[-1])
    print(f"prompt: {args.prompt_tokens} target -> {prompt_len} actual tokens "
          f"(chat template adds a few), uniform across every row -- no padding")

    eos_ids = collect_eos_ids(tok, model)
    print(f"eos ids: {sorted(eos_ids) or 'none found'}")

    # ITL percentiles need 1/(1-p) samples before pct() stops returning NaN -- with
    # short trials (small --max-new-tokens, or a ragged sequence that finishes early)
    # p99 in particular may simply not have enough decode steps yet. That is pct()
    # working correctly, not a bug in this file -- see engine/manual.py's docstring.
    print(f"ITL percentile sample floors: p50>={min_samples(50)}  p95>={min_samples(95)}  "
          f"p99>={min_samples(99)} decode steps (fewer -> reported as n/a, correctly)")

    if args.mode == "uniform":
        batches = [int(x) for x in args.batches.split(",")]
        specs = [(B, [args.max_new_tokens] * B) for B in batches]
    else:
        lengths = [int(x) for x in args.lengths.split(",")]
        specs = [(len(lengths), lengths)]
        print(f"ragged lengths: {lengths}  (B={len(lengths)})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trials: list[Trial] = []
    print(f"\n{HDR}")
    with out_path.open("a") as out:
        for B, per_seq in specs:
            t = run_trial(model, args.mode, B, per_seq, template_row, eos_ids, fwd_kw,
                          prompt_len, args.warmup, args.ignore_eos)
            write_jsonl(out, t)
            print_trial_row(t)
            trials.append(t)
            if not t.ok:
                print(f"  OOM at B={B} -- stopping sweep (sizes assumed increasing), "
                      f"gc.collect()+empty_cache() already run")
                break

    ok_trials = [t for t in trials if t.ok]
    if args.mode == "uniform" and ok_trials:
        print(f"\n  \033[1mlargest batch that fits: {ok_trials[-1].batch}"
              f"  ({ok_trials[-1].batch * prompt_len:,} prompt tokens in flight)\033[0m")
    if args.mode == "ragged" and ok_trials:
        rt = ok_trials[0]
        print(f"\n  \033[1mutilization {rt.utilization*100:.1f}%\033[0m  --  "
              f"naive {_f(rt.naive_tok_s,7)} tok/s looks healthy, "
              f"useful {_f(rt.useful_tok_s,7)} tok/s is what requesters actually got. "
              f"{rt.slot_steps - rt.useful_tokens:,} of {rt.slot_steps:,} slot-steps "
              f"were spent on already-finished sequences.")

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

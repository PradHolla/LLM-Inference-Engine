"""
manual.py -- Phase 2 step 1: replace `.generate()` with a decode loop we control.
Offline, batch=1; proves token-for-token IDENTICAL output under greedy decoding.
Reuses patterns verified in tools/kvprobe.py / baseline/server.py -- see docs.

  /opt/llm/.venv/bin/python -m engine.manual --impl compare --reps 3
"""
from __future__ import annotations

import os
# MUST precede `import torch` below -- PYTORCH_CUDA_ALLOC_CONF is read once when the
# CUDA allocator initializes and never re-read after. Setting it after import torch
# is not an error, it just silently does nothing. Measured worth +29% concurrency
# (7 -> 9 requests fitting at 4k context) in NOTES/predictions.md; apply it to every
# Phase 2 engine run, per that finding.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import inspect
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GIB = 1 << 30

# Same filler text/heuristic as tools/bench.py, duplicated (not imported) so this
# file stays self-contained; count only needs to land near --prompt-tokens. See docs.
FILLER = ("The quick brown fox jumps over the lazy dog while the system under test "
          "processes tokens one at a time in strict sequence. ")


def pick_logits_kwarg(model) -> str | None:
    """Copied from tools/kvprobe.py -- transformers renamed num_logits_to_keep ->
    logits_to_keep. Getting this wrong silently reinstates the logits explosion
    (full-vocab logits at every prefill position), so detect it instead of assuming."""
    params = inspect.signature(model.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return name
    return None


def warn_nongreedy_config(model) -> list[str]:
    """Report generation_config fields that would make .generate() diverge from a bare
    argmax; anything neutralizable is passed explicitly in generate_hf. Returns the
    offending field names, so a compare-mode mismatch is diagnosable, not a mystery."""
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return []
    neutral = {"repetition_penalty": 1.0, "encoder_repetition_penalty": 1.0,
               "no_repeat_ngram_size": 0, "min_new_tokens": None, "min_length": 0,
               "bad_words_ids": None, "sequence_bias": None, "suppress_tokens": None,
               "begin_suppress_tokens": None, "forced_bos_token_id": None,
               "forced_eos_token_id": None, "exponential_decay_length_penalty": None,
               "renormalize_logits": False}
    off = []
    for field_name, default in neutral.items():
        v = getattr(gc, field_name, default)
        if v not in (default, None) and not (isinstance(v, (int, float)) and v == default):
            off.append(f"{field_name}={v}")
    return off


def collect_eos_ids(tok, model) -> set[int]:
    """Union of tokenizer.eos_token_id and generation_config.eos_token_id, normalized to
    a set of ints. The manual loop and generate_hf's eos_token_id= must agree exactly on
    this set, or a real stopping-behaviour difference would look like a manual-loop bug."""
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


def make_prompt(target_tokens: int) -> str:
    body = FILLER * max(1, target_tokens * 4 // len(FILLER) + 1)
    return body[: target_tokens * 4]


def min_samples(p: float) -> int:
    """Copied from tools/bench.py. A p95 from 7 samples is the max wearing a
    percentile's name -- need 1/(1-p) samples before it means anything."""
    return 2 if p <= 50 else int(round(1 / (1 - p / 100)))


def pct(xs: list[float], p: float) -> float:
    """Copied from tools/bench.py, same reasoning: NaN beats a confident wrong
    number when there isn't enough data yet."""
    if not xs or len(xs) < min_samples(p):
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# --------------------------------------------------------------------- the loop
def generate_manual(model, tok, input_ids: torch.Tensor, max_new_tokens: int,
                     eos_ids: set[int]) -> tuple[list[int], dict[str, Any]]:
    """Manual prefill + decode loop, GREEDY only (determinism against generate_hf is
    the point). Returns (generated_ids, timings); decode_s excludes the prefill-produced
    first token (cost = prefill_s / TTFT). See docs for the full return-shape rationale."""
    device = input_ids.device
    prompt_len = input_ids.shape[1]
    fwd_kw = pick_logits_kwarg(model)
    kw = {fwd_kw: 1} if fwd_kw else {}

    timings: dict[str, Any] = {"prefill_s": 0.0, "decode_s": []}
    generated: list[int] = []

    with torch.inference_mode():
        # PREFILL -- one forward over the whole prompt, no cache yet.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache_position = torch.arange(0, prompt_len, device=device)
        out = model(input_ids=input_ids, past_key_values=None, use_cache=True,
                    cache_position=cache_position, **kw)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings["prefill_s"] = t1 - t0

        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(-1)          # shape [batch]
        tok_id = int(next_id.item())
        generated.append(tok_id)
        pos = prompt_len

        # DECODE loop. max_new_tokens counts total tokens including prefill's, matching
        # HF's convention so generate_hf/generate_manual compare at identical values.
        while tok_id not in eos_ids and len(generated) < max_new_tokens:
            torch.cuda.synchronize()
            ts0 = time.perf_counter()
            step_input = next_id.unsqueeze(-1)              # [batch] -> [batch, 1]
            cache_position = torch.arange(pos, pos + 1, device=device)
            out = model(input_ids=step_input, past_key_values=past, use_cache=True,
                        cache_position=cache_position, **kw)
            torch.cuda.synchronize()
            ts1 = time.perf_counter()
            timings["decode_s"].append(ts1 - ts0)

            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(-1)
            tok_id = int(next_id.item())
            generated.append(tok_id)
            pos += 1

    return generated, timings


def generate_hf(model, tok, input_ids: torch.Tensor, max_new_tokens: int,
                eos_ids: set[int], pad_token_id: int) -> tuple[list[int], dict[str, Any]]:
    """The control: model.generate() with do_sample=False, num_beams=1, eos_token_id
    passed from the same eos_ids the manual loop checks (else a length mismatch would
    look like a manual-loop bug). Deliberately NOT streamed -- see docs for why."""
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=sorted(eos_ids) if eos_ids else None,
            pad_token_id=pad_token_id,
            # Neutralize every logits processor generation_config might enable --
            # repetition_penalty/no_repeat_ngram_size apply to greedy too. See docs.
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            min_new_tokens=0,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    prompt_len = input_ids.shape[1]
    generated = out_ids[0, prompt_len:].tolist()
    return generated, {"total_s": t1 - t0}


def run_warmup(model, tok, device, eos_ids: set[int], pad_token_id: int, impls: set[str],
               input_ids: torch.Tensor | None = None) -> None:
    """One short (~8 token) generation per implementation, discarded before measuring.
    Phase 1 measured a 47x first-call outlier (9.09s vs 195ms steady state) from CUDA
    kernel autotuning; including it in a latency sample would corrupt every percentile."""
    print("warming up (discarded)...", end=" ", flush=True)
    t0 = time.perf_counter()
    # Warm up on the SHAPE we're about to measure -- cuBLAS picks kernels per problem
    # shape. MEASURED 2026-08-21: wrong-shape warmup left a 36% outlier in rep 0. See docs.
    if input_ids is not None:
        ids = input_ids
    else:
        # Same order as tools/kvprobe.py's probe_structure: .input_ids before .to(device).
        ids = tok("The capital of France is", return_tensors="pt").input_ids.to(device)
    if "manual" in impls:
        generate_manual(model, tok, ids, 8, eos_ids)
    if "hf" in impls:
        generate_hf(model, tok, ids, 8, eos_ids, pad_token_id)
    torch.cuda.synchronize()
    print(f"done ({time.perf_counter() - t0:.1f}s)")


# ------------------------------------------------------------------- reporting
@dataclass
class Rep:
    impl: str
    rep: int
    prompt_tokens: int
    max_new_tokens: int
    total_tokens: int
    tokens: list[int]
    prefill_s: float | None = None      # manual only
    decode_s: list[float] = field(default_factory=list)   # manual only
    total_s: float | None = None        # hf only


def manual_rep(model, tok, input_ids, max_new_tokens, eos_ids, i: int) -> Rep:
    prompt_len = input_ids.shape[1]
    ids, t = generate_manual(model, tok, input_ids, max_new_tokens, eos_ids)
    return Rep(impl="manual", rep=i, prompt_tokens=prompt_len, max_new_tokens=max_new_tokens,
               total_tokens=len(ids), tokens=ids, prefill_s=t["prefill_s"], decode_s=t["decode_s"])


def hf_rep(model, tok, input_ids, max_new_tokens, eos_ids, pad_token_id, i: int) -> Rep:
    prompt_len = input_ids.shape[1]
    ids, t = generate_hf(model, tok, input_ids, max_new_tokens, eos_ids, pad_token_id)
    return Rep(impl="hf", rep=i, prompt_tokens=prompt_len, max_new_tokens=max_new_tokens,
               total_tokens=len(ids), tokens=ids, total_s=t["total_s"])


def write_jsonl(out, rec: Rep) -> None:
    # Flushed per rep, not buffered to the end -- a sweep that only writes at the
    # end loses everything to a crash, a Ctrl-C, or a spot reclaim.
    out.write(json.dumps(asdict(rec)) + "\n")
    out.flush()


def _f(v: float, w: int, suffix: str = "") -> str:
    return f"{'n/a':>{w}}" if v != v else f"{v:>{w}.1f}{suffix}"


HDR_MANUAL = (f"  {'impl':<7} {'rep':>3} {'tok':>4} │ {'TTFT ms':>8} │ "
              f"{'ITL p50':>8} {'p95':>8} {'p99':>8} (ms) │ {'tok/s':>7}")
HDR_HF = (f"  {'impl':<7} {'rep':>3} {'tok':>4} │ {'total ms':>9} │ {'tok/s (naive)':>13}")


def print_manual_row(r: Rep) -> None:
    decode_ms = [d * 1000 for d in r.decode_s]
    tok_s = len(decode_ms) / sum(r.decode_s) if r.decode_s and sum(r.decode_s) > 0 else float("nan")
    print(f"  {r.impl:<7} {r.rep:>3} {r.total_tokens:>4} │ {r.prefill_s*1000:>8.1f} │ "
          f"{_f(pct(decode_ms,50),8)} {_f(pct(decode_ms,95),8)} {_f(pct(decode_ms,99),8)}     │ "
          f"{_f(tok_s,7)}")


def print_hf_row(r: Rep) -> None:
    tok_s = r.total_tokens / r.total_s if r.total_s and r.total_s > 0 else float("nan")
    print(f"  {r.impl:<7} {r.rep:>3} {r.total_tokens:>4} │ {r.total_s*1000:>9.1f} │ {_f(tok_s,13)}")


def print_manual_summary(reps: list[Rep]) -> None:
    ttfts = [r.prefill_s * 1000 for r in reps]
    decode_ms = [d * 1000 for r in reps for d in r.decode_s]
    decode_s_sum = sum(sum(r.decode_s) for r in reps)
    decode_n = sum(len(r.decode_s) for r in reps)
    tok_s = decode_n / decode_s_sum if decode_s_sum > 0 else float("nan")
    print(f"\n  \033[1mmanual  n={len(reps)} reps, {decode_n} pooled decode steps\033[0m")
    print(f"    TTFT       p50 {_f(pct(ttfts,50),7)} ms   p95 {_f(pct(ttfts,95),7)} ms")
    print(f"    ITL        p50 {_f(pct(decode_ms,50),7)} ms   p95 {_f(pct(decode_ms,95),7)} ms"
          f"   p99 {_f(pct(decode_ms,99),7)} ms")
    print(f"    decode throughput  {_f(tok_s,7)} tok/s   "
          f"(= pooled decode tokens / sum of decode latencies, prefill excluded)")


def print_hf_summary(reps: list[Rep]) -> None:
    total_tokens = sum(r.total_tokens for r in reps)
    total_s = sum(r.total_s for r in reps)
    tok_s = total_tokens / total_s if total_s > 0 else float("nan")
    print(f"\n  \033[1mhf      n={len(reps)} reps\033[0m")
    print(f"    naive throughput   {_f(tok_s,7)} tok/s   "
          f"(= total tokens / total wall time, prefill NOT excluded -- "
          f".generate() exposes no per-token timestamps without a streamer, "
          f"see generate_hf's docstring)")


def check_identical(tok, manual_ids: list[int], hf_ids: list[int]) -> bool:
    n = min(len(manual_ids), len(hf_ids))
    for i in range(n):
        if manual_ids[i] != hf_ids[i]:
            print(f"  MISMATCH at index {i}:")
            print(f"    manual  id={manual_ids[i]}  {tok.decode([manual_ids[i]])!r}")
            print(f"    hf      id={hf_ids[i]}  {tok.decode([hf_ids[i]])!r}")
            return False
    if len(manual_ids) != len(hf_ids):
        print(f"  MISMATCH: length differs -- manual={len(manual_ids)} hf={len(hf_ids)} "
              f"(all {n} shared positions were identical)")
        return False
    print(f"  IDENTICAL  ({len(manual_ids)} tokens)")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--impl", choices=["manual", "hf", "compare"], default="compare")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", default="results/phase2-step1.jsonl")
    p.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    # Default ON, unlike tools/bench.py's --no-think: step 1 wants bounded, comparable
    # output lengths, not Qwen3 spending budget on reasoning tokens. See docs.
    p.add_argument("--no-think", dest="no_think", action="store_true", default=True)
    p.add_argument("--think", dest="no_think", action="store_false")
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

    body = make_prompt(args.prompt_tokens)
    chat_kwargs = {"enable_thinking": False} if args.no_think else {}
    input_ids = tok.apply_chat_template(
        [{"role": "user", "content": body}],
        add_generation_prompt=True, tokenize=True, return_tensors="pt", **chat_kwargs,
    )
    # transformers 5.x returns a BatchEncoding (dict-like) from apply_chat_template
    # when tokenize=True; 4.x returned a bare tensor. VERIFIED in baseline/server.py.
    if not hasattr(input_ids, "shape"):
        input_ids = input_ids["input_ids"]
    input_ids = input_ids.to(device)
    prompt_len = int(input_ids.shape[-1])
    print(f"prompt: {args.prompt_tokens} target -> {prompt_len} actual tokens "
          f"(chat template adds a few)")

    eos_ids = collect_eos_ids(tok, model)
    pad_token_id = tok.pad_token_id or tok.eos_token_id
    print(f"eos ids: {sorted(eos_ids) or 'none found'}")
    offenders = warn_nongreedy_config(model)
    if offenders:
        print(f"  NOTE generation_config would alter greedy decoding: {', '.join(offenders)}\n       neutralized where possible in generate_hf; a compare mismatch here is likely this, not the loop")

    impls = {"manual", "hf"} if args.impl == "compare" else {args.impl}
    if args.warmup:
        run_warmup(model, tok, device, eos_ids, pad_token_id, impls, input_ids)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manual_reps: list[Rep] = []
    hf_reps: list[Rep] = []

    with out_path.open("a") as out:
        if args.impl == "compare":
            # Correctness gate first, at rep 0 only, before spending more GPU time on a
            # loop already proven wrong. Rep 0 is reused as the first measured rep below.
            print(f"\n\033[1mcorrectness check\033[0m ({prompt_len} prompt, "
                  f"{args.max_new_tokens} max new, greedy)")
            r0m = manual_rep(model, tok, input_ids, args.max_new_tokens, eos_ids, 0)
            r0h = hf_rep(model, tok, input_ids, args.max_new_tokens, eos_ids, pad_token_id, 0)
            write_jsonl(out, r0m)
            write_jsonl(out, r0h)
            ok = check_identical(tok, r0m.tokens, r0h.tokens)
            if not ok:
                print("\nstopping -- fix the manual loop before trusting its timing")
                return 1
            manual_reps.append(r0m)
            hf_reps.append(r0h)

            print(f"\n{HDR_MANUAL}")
            print_manual_row(r0m)
            for i in range(1, args.reps):
                r = manual_rep(model, tok, input_ids, args.max_new_tokens, eos_ids, i)
                write_jsonl(out, r)
                print_manual_row(r)
                manual_reps.append(r)
            print_manual_summary(manual_reps)

            print(f"\n{HDR_HF}")
            print_hf_row(r0h)
            for i in range(1, args.reps):
                r = hf_rep(model, tok, input_ids, args.max_new_tokens, eos_ids, pad_token_id, i)
                write_jsonl(out, r)
                print_hf_row(r)
                hf_reps.append(r)
            print_hf_summary(hf_reps)

            print("\n\033[1mcorrectness\033[0m: ", end="")
            check_identical(tok, manual_reps[0].tokens, hf_reps[0].tokens)

        elif args.impl == "manual":
            print(f"\n{HDR_MANUAL}")
            for i in range(args.reps):
                r = manual_rep(model, tok, input_ids, args.max_new_tokens, eos_ids, i)
                write_jsonl(out, r)
                print_manual_row(r)
                manual_reps.append(r)
            print_manual_summary(manual_reps)

        else:  # hf
            print(f"\n{HDR_HF}")
            for i in range(args.reps):
                r = hf_rep(model, tok, input_ids, args.max_new_tokens, eos_ids, pad_token_id, i)
                write_jsonl(out, r)
                print_hf_row(r)
                hf_reps.append(r)
            print_hf_summary(hf_reps)

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

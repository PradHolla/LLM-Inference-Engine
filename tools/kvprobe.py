#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers", "accelerate"]
# ///
"""
kvprobe.py -- measure the KV cache directly. No server, no benchmark, no scheduler.

Answers three questions that Phase 1 could not, because Phase 1 never batched:

  1. Is KV really 144 KiB/token?     -- read the tensor shapes, do not trust the config
  2. Does VRAM grow by exactly that per decode step?
  3. What is the largest batch that actually fits at a given context length?

TWO TRAPS, both of which produce a plausible wrong number rather than an exception:

  * LOGITS EXPLOSION. HF computes logits at every prefill position unless told not to.
    batch 12 x 4096 ctx x 151,936 vocab in bf16 = 14.9 GiB of logits. That OOMs with
    the KV cache barely touched and looks exactly like KV exhaustion. Hence
    logits_to_keep=1 on every forward.

  * ACTIVATION SPIKE. Prefilling 4096 tokens in one pass makes activation memory scale
    with batch x context, so an OOM would again be misattributed to KV. Hence prefill
    in chunks, so the only term growing with context is the cache itself.

Memory is read three ways because they measure different things:
  torch allocated  -- live tensors torch knows about
  torch reserved   -- what torch has taken from the driver, including free blocks
  driver free/total (mem_get_info) -- ground truth, includes the CUDA context that
                                      never appears in torch's numbers at all

  uv run tools/kvprobe.py --probe-api        # cheap: structure only, validates the API
  uv run tools/kvprobe.py --context 4096     # full run including the ceiling walk
"""
from __future__ import annotations
import argparse, gc, inspect, json, os, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GIB = 1 << 30
KIB = 1024


def mem() -> dict:
    free, total = torch.cuda.mem_get_info()
    return {
        "alloc": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "driver_used": total - free,
        "driver_total": total,
    }


def fmt_gib(b: float) -> str:
    return f"{b / GIB:7.3f} GiB"


def iter_kv(cache):
    """Yield (layer, 'k'|'v', tensor). The cache API moved between transformers 4 and 5,
    so try each shape rather than pinning a version."""
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
    return sum(t.numel() * t.element_size() for _, _, t in iter_kv(cache))


def pick_logits_kwarg(model) -> str | None:
    """transformers renamed num_logits_to_keep -> logits_to_keep. Getting this wrong
    silently reinstates the logits explosion, so detect it instead of assuming."""
    params = inspect.signature(model.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return name
    return None


# ---------------------------------------------------------------- part 1: structure
def probe_structure(model, tok, device, fwd_kw) -> dict:
    print("\n\033[1mPART 1 -- CACHE STRUCTURE\033[0m  (one forward pass, batch 1)")
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(device)
    n = ids.shape[1]
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True, past_key_values=None,
                    **{fwd_kw: 1} if fwd_kw else {})
    cache = out.past_key_values

    tensors = list(iter_kv(cache))
    layers = len({i for i, _, _ in tensors})
    k0 = next(t for _, kind, t in tensors if kind == "k")
    total = cache_bytes(cache)
    per_token = total / n

    cfg = model.config
    theory = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * \
        getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads) * k0.element_size()

    print(f"  cache class            {type(cache).__name__}")
    print(f"  tensors                {len(tensors)}  ({layers} layers x K,V)")
    print(f"  per-layer K shape      {tuple(k0.shape)}   dtype {k0.dtype}")
    print(f"    -> [batch, kv_heads, seq, head_dim]")
    print(f"  prompt tokens          {n}")
    print(f"  cache bytes total      {total:,}")
    print(f"  \033[1mmeasured  KV/token   {per_token:,.0f} B = {per_token/KIB:.1f} KiB\033[0m")
    print(f"  predicted KV/token   {theory:,.0f} B = {theory/KIB:.1f} KiB"
          f"   ({'MATCH' if abs(per_token-theory) < 1 else 'MISMATCH'})")
    return {"kv_per_token": per_token, "predicted": theory, "cache_class": type(cache).__name__,
            "k_shape": list(k0.shape), "dtype": str(k0.dtype), "layers": layers}


# ------------------------------------------------------------------ part 2: growth
def probe_growth(model, tok, device, fwd_kw, steps: int) -> dict:
    print(f"\n\033[1mPART 2 -- PER-TOKEN GROWTH\033[0m  ({steps} decode steps, batch 1)")
    ids = tok("Explain memory bandwidth.", return_tensors="pt").input_ids.to(device)
    gc.collect(); torch.cuda.empty_cache()
    base = mem()["alloc"]

    cache, samples = None, []
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True, past_key_values=cache,
                    **{fwd_kw: 1} if fwd_kw else {})
        cache = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        for s in range(steps):
            out = model(input_ids=nxt, use_cache=True, past_key_values=cache,
                        **{fwd_kw: 1} if fwd_kw else {})
            cache = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
            samples.append((s, cache_bytes(cache), mem()["alloc"] - base))

    # slope from the cache itself, and from what torch reports live
    d_cache = (samples[-1][1] - samples[0][1]) / (len(samples) - 1)
    d_alloc = (samples[-1][2] - samples[0][2]) / (len(samples) - 1)
    print(f"  step   cache bytes   d(cache)/tok   torch alloc delta")
    for s, cb, ad in samples[::max(1, len(samples)//8)]:
        print(f"  {s:4d}   {cb:11,}   {'':12}   {ad:,}")
    print(f"  \033[1mslope, cache tensors   {d_cache:,.0f} B/token = {d_cache/KIB:.1f} KiB\033[0m")
    print(f"  slope, torch allocated {d_alloc:,.0f} B/token = {d_alloc/KIB:.1f} KiB")
    print(f"  \033[2m(allocated jitters: DynamicCache concatenates, so the old tensor is")
    print(f"   freed only after the new one is allocated)\033[0m")
    return {"slope_cache": d_cache, "slope_alloc": d_alloc, "steps": steps}


# ----------------------------------------------------------- part 3: batch ceiling
def try_batch(model, B: int, L: int, chunk: int, device, fwd_kw, vocab: int) -> dict:
    """Build a batch of B sequences at length L with chunked prefill, then decode a few
    steps. Decoding matters: a batch that barely prefills can still OOM on step 1, and
    that does not count as fitting."""
    ids = torch.randint(100, vocab - 100, (B, L), device=device)
    cache, off = None, 0
    kw = {fwd_kw: 1} if fwd_kw else {}
    with torch.inference_mode():
        while off < L:
            c = ids[:, off:off + chunk]
            out = model(input_ids=c, past_key_values=cache, use_cache=True,
                        cache_position=torch.arange(off, off + c.shape[1], device=device),
                        **kw)
            cache, off = out.past_key_values, off + c.shape[1]
        prefill_peak = torch.cuda.max_memory_allocated()
        nxt = out.logits[:, -1:].argmax(-1)
        for _ in range(4):
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True,
                        cache_position=torch.arange(off, off + 1, device=device), **kw)
            cache, nxt, off = out.past_key_values, out.logits[:, -1:].argmax(-1), off + 1
    m = mem()
    return {"kv_bytes": cache_bytes(cache), "prefill_peak": prefill_peak,
            "peak_alloc": torch.cuda.max_memory_allocated(), **m}


def probe_ceiling(model, device, fwd_kw, vocab, L, chunk, sizes, weights, out_path) -> dict:
    print(f"\n\033[1mPART 3 -- BATCH CEILING\033[0m  at {L} ctx, chunked prefill ({chunk})")
    print(f"  batch    KV cache     peak alloc    driver used   headroom   result")
    print("  " + "-" * 68)
    rows, best = [], 0
    for B in sizes:
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            r = try_batch(model, B, L, chunk, device, fwd_kw, vocab)
            best = B
            head = r["driver_total"] - r["driver_used"]
            print(f"  {B:5d}  {fmt_gib(r['kv_bytes'])}  {fmt_gib(r['peak_alloc'])}  "
                  f"{fmt_gib(r['driver_used'])}  {fmt_gib(head)}   ok")
            rows.append({"batch": B, "ok": True, "tokens": B * L, **r})
        except torch.cuda.OutOfMemoryError:
            print(f"  {B:5d}  {'':11}  {'':12}  {'':13}  {'':10}   OOM")
            rows.append({"batch": B, "ok": False, "tokens": B * L})
            gc.collect(); torch.cuda.empty_cache()
            break
        finally:
            with out_path.open("a") as f:                 # flush per trial, not at the end
                f.write(json.dumps(rows[-1], default=str) + "\n")
    print(f"\n  \033[1mlargest batch that fits: {best}  ({best * L:,} tokens in flight)\033[0m")
    return {"max_batch": best, "max_tokens": best * L, "rows": rows}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--sizes", default="1,2,4,6,8,9,10,11,12,13,14,16,20,24")
    p.add_argument("--probe-api", action="store_true", help="part 1 only -- cheap API check")
    p.add_argument("--out", default="results/phase2-kvprobe.jsonl")
    a = p.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    free0, total0 = torch.cuda.mem_get_info()
    print(f"\033[1m{name}\033[0m   {total0/GIB:.3f} GiB total, {free0/GIB:.3f} GiB free before load")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    w = torch.cuda.memory_allocated()
    ctx_overhead = (total0 - torch.cuda.mem_get_info()[0]) - torch.cuda.memory_reserved()
    print(f"loaded in {time.time()-t0:.1f}s   weights {fmt_gib(w)}   "
          f"CUDA context+driver {fmt_gib(max(ctx_overhead, 0))}")

    fwd_kw = pick_logits_kwarg(model)
    print(f"logits kwarg: {fwd_kw or 'NOT FOUND -- logits explosion risk, see docstring'}")

    out_path = Path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    res = {"gpu": name, "model": a.model, "weights": w, "vram_total": total0,
           "context": a.context, "ts": time.time()}
    res["structure"] = probe_structure(model, tok, dev, fwd_kw)
    if a.probe_api:
        print("\n--probe-api: stopping before the expensive parts")
        return 0
    res["growth"] = probe_growth(model, tok, dev, fwd_kw, a.steps)
    sizes = [int(x) for x in a.sizes.split(",")]
    res["ceiling"] = probe_ceiling(model, dev, fwd_kw, model.config.vocab_size,
                                   a.context, a.chunk, sizes, w, out_path)
    with out_path.open("a") as f:
        f.write(json.dumps({"summary": res}, default=str) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

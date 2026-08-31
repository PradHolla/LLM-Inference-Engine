"""
mask_probe.py -- decompose the padding tax measured in step 3a (ITL rising 27.4us
per buffer token, 9.4x the KV-read model). See docs for the two candidate causes
(KV reads over padding vs SDPA's masked path) and what each implies for step 4.

  A attention_mask=None   B attention_mask=ones(B,L)   C real zeros
"""
from __future__ import annotations
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse, statistics, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.manual import pick_logits_kwarg, make_prompt


def timed_decode(model, cache, nxt, pos, mask, fwd_kw, steps: int) -> list[float]:
    """N decode steps at a fixed shape; returns per-step seconds. Cache is NOT reused
    across conditions -- DynamicCache grows in place, so a shared cache would leave later
    conditions running at longer buffers than earlier ones."""
    kw = {fwd_kw: 1} if fwd_kw else {}
    out_t = []
    B = nxt.shape[0]
    for _ in range(steps):
        if mask is not None:
            mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=mask.device)], -1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=nxt.unsqueeze(-1), past_key_values=cache, use_cache=True,
                    position_ids=pos, attention_mask=mask, **kw)
        torch.cuda.synchronize()
        out_t.append(time.perf_counter() - t0)
        cache = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1)
        pos = pos + 1
    return out_t


def prefill(model, ids, fwd_kw, mask=None):
    kw = {fwd_kw: 1} if fwd_kw else {}
    out = model(input_ids=ids, past_key_values=None, use_cache=True,
                attention_mask=mask, **kw)
    return out.past_key_values, out.logits[:, -1, :].argmax(-1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--pad", type=int, default=200,
                   help="tokens of left padding given to half the rows in condition C")
    a = p.parse_args()

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model.eval()
    fwd_kw = pick_logits_kwarg(model)

    t = tok.apply_chat_template([{"role": "user", "content": make_prompt(a.prompt_tokens)}],
                                add_generation_prompt=True, tokenize=True,
                                return_tensors="pt", enable_thinking=False)
    if not hasattr(t, "shape"):
        t = t["input_ids"]
    row = t.to(dev)[0]
    L = row.shape[0]
    B = a.batch
    ids = row.unsqueeze(0).repeat(B, 1)
    print(f"batch {B}, buffer {L} tokens, {a.steps} decode steps per condition")
    print(f"condition C left-pads {B//2} of {B} rows by {a.pad} tokens\n")

    results = {}
    with torch.inference_mode():
        # warm every shape first -- cuBLAS picks kernels per shape (incident 15)
        for lab, m in (("warm-none", None), ("warm-ones", torch.ones(B, L, dtype=torch.long, device=dev))):
            c, n = prefill(model, ids, fwd_kw, m)
            timed_decode(model, c, n, torch.full((B, 1), L, device=dev), m, fwd_kw, 4)
        del c, n
        torch.cuda.empty_cache()

        # A: no mask at all
        c, n = prefill(model, ids, fwd_kw, None)
        results["A no mask (implicit causal)"] = timed_decode(
            model, c, n, torch.full((B, 1), L, device=dev), None, fwd_kw, a.steps)
        del c, n; torch.cuda.empty_cache()

        # B: explicit mask, all ones -- masks nothing
        m = torch.ones(B, L, dtype=torch.long, device=dev)
        c, n = prefill(model, ids, fwd_kw, m)
        results["B explicit mask, all ones"] = timed_decode(
            model, c, n, torch.full((B, 1), L, device=dev), m.clone(), fwd_kw, a.steps)
        del c, n; torch.cuda.empty_cache()

        # C: explicit mask with real zeros -- half the rows left-padded
        m = torch.ones(B, L, dtype=torch.long, device=dev)
        m[: B // 2, : a.pad] = 0
        pos = torch.full((B, 1), L, device=dev)
        pos[: B // 2] = L - a.pad          # padded rows carry their own true position
        c, n = prefill(model, ids, fwd_kw, m)
        results["C explicit mask, real zeros"] = timed_decode(
            model, c, n, pos, m.clone(), fwd_kw, a.steps)

    print(f"  {'condition':<32} {'p50 ms':>8} {'p95 ms':>8} {'vs A':>8}")
    print("  " + "-" * 60)
    base = statistics.median(results["A no mask (implicit causal)"])
    for k, v in results.items():
        s = sorted(v)
        p50 = statistics.median(v) * 1000
        p95 = s[int(0.95 * (len(s) - 1))] * 1000
        print(f"  {k:<32} {p50:>8.2f} {p95:>8.2f} {p50/(base*1000):>7.2f}x")
    print("\n  If C >> B: the tax is the masked kernel path, and a paged allocator")
    print("  cannot recover it on stock HF -- ragged rows need a mask regardless.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

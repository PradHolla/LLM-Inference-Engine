"""
cache_probe.py -- prove the three mechanics continuous batching needs, before building it.

Continuous batching requires rows at DIFFERENT sequence lengths in one batch. HF's cache
API assumes one shared cache_position for the whole batch, which is precisely why vLLM
wrote its own attention kernels. The way through without custom kernels:

  left-pad every row into a common buffer, and DECOUPLE buffer index from RoPE position.

A row's K/V are rotated by RoPE at write time, so where it physically sits in the buffer
does not affect correctness -- only the attention mask and the position_ids do. If that
holds, admitting and evicting a sequence become plain tensor ops on the batch dimension.

Three things must be true. Each is checked against a batch-1 reference run, because the
only acceptable evidence is token-for-token identical output:

  1. per-row position_ids ([B,1] with different values per row) are honoured
  2. left-padding plus a 2-D attention mask gives the same tokens as unpadded batch-1
  3. cache rows can be DROPPED and APPENDED between steps and decoding continues correctly

If any fails, the engine design changes before a line of it is written.
"""
from __future__ import annotations
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.manual import pick_logits_kwarg, make_prompt

STEPS = 16


def iter_layers(cache):
    """Verified accessor from tools/kvprobe.py, but we also need to WRITE."""
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return [(l, "keys", "values") for l in layers]
    raise RuntimeError(f"unexpected cache type {type(cache)}")


def get_kv(cache, i):
    l = cache.layers[i]
    return l.keys, l.values


def set_kv(cache, i, k, v):
    cache.layers[i].keys, cache.layers[i].values = k, v


def n_layers(cache):
    return len(cache.layers)


def reference(model, ids, fwd_kw, steps=STEPS):
    """Batch-1 ground truth: the step-1 loop, known correct against .generate()."""
    kw = {fwd_kw: 1} if fwd_kw else {}
    out = model(input_ids=ids, past_key_values=None, use_cache=True, **kw)
    past = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1)
    got = [int(nxt.item())]
    pos = ids.shape[1]
    for _ in range(steps - 1):
        out = model(input_ids=nxt.unsqueeze(-1), past_key_values=past, use_cache=True,
                    position_ids=torch.tensor([[pos]], device=ids.device), **kw)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1)
        got.append(int(nxt.item()))
        pos += 1
    return got


def padded_batch_run(model, seqs, fwd_kw, device, steps=STEPS, drop_at=None, admit=None):
    """Left-pad `seqs` (list of 1-D id tensors, ragged lengths) into one batch and decode.

    Buffer index and RoPE position are deliberately decoupled: a left-padded row's real
    tokens sit at buffer indices [pad_n, L) but carry position_ids 0..n-1. The mask hides
    the pad region, so the model never attends to it and never sees the offset.

    drop_at: (step, row) -- evict a row mid-decode by slicing the batch dim of the cache.
    admit:   (step, ids) -- prefill a new sequence separately, left-pad it, append it.
    """
    kw = {fwd_kw: 1} if fwd_kw else {}
    lens = [s.shape[0] for s in seqs]
    L = max(lens)
    B = len(seqs)
    pad_id = 0
    ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((B, L), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):                      # LEFT pad
        ids[i, L - lens[i]:] = s
        mask[i, L - lens[i]:] = 1
    # standard HF convention for left padding: positions count only real tokens
    pos_ids = (mask.cumsum(-1) - 1).clamp(min=0)

    out = model(input_ids=ids, attention_mask=mask, position_ids=pos_ids,
                past_key_values=None, use_cache=True, **kw)
    past = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1)
    rows = list(range(B))
    got = {r: [int(nxt[i].item())] for i, r in enumerate(rows)}
    next_pos = torch.tensor([[l] for l in lens], device=device)   # per-row RoPE position

    for step in range(1, steps):
        if drop_at and step == drop_at[0]:
            victim = rows.index(drop_at[1])
            keep = [i for i in range(len(rows)) if i != victim]
            idx = torch.tensor(keep, device=device)
            for li in range(n_layers(past)):
                k, v = get_kv(past, li)
                set_kv(past, li, k.index_select(0, idx).contiguous(),
                       v.index_select(0, idx).contiguous())
            mask = mask.index_select(0, idx)
            nxt = nxt.index_select(0, idx)
            next_pos = next_pos.index_select(0, idx)
            rows = [rows[i] for i in keep]

        if admit and step == admit[0]:
            new_ids = admit[1].unsqueeze(0)
            # Prefill the newcomer on its own -- this STALLS the decode loop, which is
            # exactly what chunked prefill exists to fix. Measured in step 3, not here.
            o2 = model(input_ids=new_ids, past_key_values=None, use_cache=True, **kw)
            p2, n2 = o2.past_key_values, o2.logits[:, -1, :].argmax(-1)
            cur_L = get_kv(past, 0)[0].shape[2]
            new_L = get_kv(p2, 0)[0].shape[2]
            padn = cur_L - new_L
            for li in range(n_layers(past)):
                k, v = get_kv(past, li)
                k2, v2 = get_kv(p2, li)
                if padn > 0:                          # left-pad the newcomer's KV
                    zk = torch.zeros(1, k2.shape[1], padn, k2.shape[3], dtype=k2.dtype, device=device)
                    k2 = torch.cat([zk, k2], dim=2); v2 = torch.cat([zk, v2], dim=2)
                set_kv(past, li, torch.cat([k, k2], 0).contiguous(),
                       torch.cat([v, v2], 0).contiguous())
            newmask = torch.zeros(1, cur_L, dtype=mask.dtype, device=device)
            newmask[0, cur_L - new_L:] = 1
            mask = torch.cat([mask, newmask], 0)
            nxt = torch.cat([nxt, n2], 0)
            next_pos = torch.cat([next_pos, torch.tensor([[new_L]], device=device)], 0)
            rows.append(admit[2])
            got[admit[2]] = [int(n2.item())]

        mask = torch.cat([mask, torch.ones(len(rows), 1, dtype=mask.dtype, device=device)], -1)
        out = model(input_ids=nxt.unsqueeze(-1), attention_mask=mask,
                    position_ids=next_pos, past_key_values=past, use_cache=True, **kw)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1)
        next_pos = next_pos + 1
        for i, r in enumerate(rows):
            got[r].append(int(nxt[i].item()))
    return got


def check(name, ref, got, steps):
    n = min(len(ref), len(got))
    bad = next((i for i in range(n) if ref[i] != got[i]), None)
    if bad is None and len(got) >= steps:
        print(f"  {name:<42} IDENTICAL ({len(got)} tokens)")
        return True
    if bad is not None:
        print(f"  {name:<42} MISMATCH at {bad}: ref={ref[bad]} got={got[bad]}")
    else:
        print(f"  {name:<42} SHORT: {len(got)} vs {steps}")
    return False


def main() -> int:
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model.eval()
    fwd_kw = pick_logits_kwarg(model)

    def enc(target):
        t = tok.apply_chat_template([{"role": "user", "content": make_prompt(target)}],
                                    add_generation_prompt=True, tokenize=True,
                                    return_tensors="pt", enable_thinking=False)
        if not hasattr(t, "shape"):
            t = t["input_ids"]
        return t.to(dev)[0]

    a, b, c = enc(128), enc(256), enc(192)     # deliberately ragged
    print(f"lengths: a={a.shape[0]}  b={b.shape[0]}  c={c.shape[0]}\n")

    with torch.inference_mode():
        print("building batch-1 references...")
        ref_a = reference(model, a.unsqueeze(0), fwd_kw)
        ref_b = reference(model, b.unsqueeze(0), fwd_kw)
        ref_c = reference(model, c.unsqueeze(0), fwd_kw)

        ok = True
        print("\nTEST 1 -- ragged left-padded batch matches unpadded batch-1")
        g = padded_batch_run(model, [a, b], fwd_kw, dev)
        ok &= check("row 0 (shorter, left-padded)", ref_a, g[0], STEPS)
        ok &= check("row 1 (longest, no padding)", ref_b, g[1], STEPS)

        print("\nTEST 2 -- evict row 0 at step 6, row 1 keeps decoding correctly")
        g = padded_batch_run(model, [a, b], fwd_kw, dev, drop_at=(6, 0))
        ok &= check("row 1 survives eviction", ref_b, g[1], STEPS)

        print("\nTEST 3 -- admit a new sequence at step 6 into the running batch")
        g = padded_batch_run(model, [a, b], fwd_kw, dev, admit=(6, c, 2))
        ok &= check("row 1 unaffected by admission", ref_b, g[1], STEPS)
        ok &= check("row 2 (admitted mid-flight)", ref_c, g[2], STEPS - 6)

    print("\n" + ("ALL MECHANICS VERIFIED -- design is sound" if ok
                  else "FAILED -- the engine design must change"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
profile_step.py -- Phase 8: look inside one decode step of our own engine. Same load, prompt and
decode call as engine/static_batch.py; each step is NVTX-labelled. See docs for the three modes.

  /opt/llm/.venv/bin/python -m engine.profile_step --mode time --batches 1,8
  nsys profile --capture-range=cudaProfilerApi -o results/p8-nsys /opt/llm/.venv/bin/python -m engine.profile_step --mode nsys
"""
from __future__ import annotations

import os
# MUST precede `import torch`: read once at CUDA init (engine/manual.py explains).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.manual import make_prompt, pick_logits_kwarg

nvtx = torch.cuda.nvtx


def build_batch(tok, device, B: int, prompt_tokens: int, masked: bool):
    """[B, L] left-padded input. masked: rows lose 0..L/2 leading tokens to padding, so the
    attention mask has real zeros -- the shape continuous batching runs (engine/continuous.py)."""
    ids = tok.apply_chat_template([{"role": "user", "content": make_prompt(prompt_tokens)}],
                                  add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                  enable_thinking=False)
    ids = ids["input_ids"] if not hasattr(ids, "shape") else ids
    row = ids[0].to(device)
    L = row.shape[0]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids = row.repeat(B, 1)
    mask = torch.ones(B, L, dtype=torch.long, device=device)
    if masked and B > 1:
        for i in range(B):
            cut = (i * (L // 2)) // (B - 1)          # row 0 keeps everything, the last loses L/2
            input_ids[i, :cut] = pad
            mask[i, :cut] = 0
    return input_ids, mask


def decode_steps(model, input_ids, mask, fwd_kw, warmup: int, steps: int, mode: str,
                 label: str):
    """Prefill, then warmup + measured decode steps. Returns per-step wall times (s)."""
    kw = {fwd_kw: 1} if fwd_kw else {}
    times: list[float] = []
    with torch.inference_mode():
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        out = model(input_ids=input_ids, attention_mask=mask, position_ids=pos,
                    past_key_values=None, use_cache=True, **kw)
        past, nxt = out.past_key_values, out.logits[:, -1, :].argmax(-1)
        next_pos = pos[:, -1:] + 1
        prof = None
        for step in range(warmup + steps):
            measured = step >= warmup
            if measured and step == warmup:
                torch.cuda.synchronize()
                if mode == "nsys":
                    torch.cuda.cudart().cudaProfilerStart()
                elif mode == "torch":
                    prof = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA])
                    prof.__enter__()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            nvtx.range_push(f"{label} step {step}")
            mask = torch.cat([mask, mask.new_ones(mask.shape[0], 1)], dim=-1)
            nvtx.range_push("forward")
            out = model(input_ids=nxt.unsqueeze(-1), attention_mask=mask, position_ids=next_pos,
                        past_key_values=past, use_cache=True, **kw)
            nvtx.range_pop()
            nvtx.range_push("sample")
            past, nxt = out.past_key_values, out.logits[:, -1, :].argmax(-1)
            nxt.tolist()                              # the host sync every real engine step pays
            nvtx.range_pop()
            next_pos = next_pos + 1
            nvtx.range_pop()
            torch.cuda.synchronize()
            if measured:
                times.append(time.perf_counter() - t0)
        if mode == "nsys":
            torch.cuda.cudart().cudaProfilerStop()
        elif mode == "torch" and prof is not None:
            prof.__exit__(None, None, None)
            return times, prof
    return times, None


def summarize_torch(prof, steps: int, label: str, out_prefix: str) -> dict:
    """Top GPU kernels per step, launches per step, and a Perfetto-readable trace."""
    trace = f"{out_prefix}-{label}.json"
    prof.export_chrome_trace(trace)
    # torch renamed cuda_* to device_* around 2.4; read whichever this version has.
    dev_total = lambda e: getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
    dev_time = lambda e: getattr(e, "device_time", None) or getattr(e, "cuda_time", 0)
    events = prof.key_averages()
    rows = sorted((e for e in events if dev_total(e) > 0), key=dev_total, reverse=True)
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    gpu_us = sum(dev_time(e) for e in kernels)
    try:
        table = events.table(sort_by="device_time_total", row_limit=15)
    except (AttributeError, KeyError, ValueError):
        table = events.table(sort_by="cuda_time_total", row_limit=15)
    with open(f"{out_prefix}-{label}-ops.txt", "w") as f:
        f.write(table)
    return {"trace": trace, "kernels_per_step": len(kernels) / steps,
            "gpu_busy_ms_per_step": gpu_us / steps / 1000,
            "top": [(e.key, round(dev_total(e) / steps / 1000, 3)) for e in rows[:8]]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--mode", choices=["time", "torch", "nsys"], default="time")
    p.add_argument("--batches", default="1,8")
    p.add_argument("--prompt-tokens", type=int, default=412)
    p.add_argument("--masked", action="store_true",
                   help="left-pad rows so the attention mask has real zeros (batch > 1)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out", default="results/p8-profile")
    args = p.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1
    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model.eval()
    fwd_kw = pick_logits_kwarg(model)
    print(f"attn_implementation={model.config._attn_implementation} logits kwarg={fwd_kw}")
    for B in [int(b) for b in args.batches.split(",")]:
        label = f"B{B}{'-masked' if args.masked and B > 1 else ''}"
        input_ids, mask = build_batch(tok, device, B, args.prompt_tokens, args.masked)
        times, prof = decode_steps(model, input_ids, mask, fwd_kw, args.warmup, args.steps,
                                   args.mode, label)
        rec = {"label": label, "mode": args.mode, "batch": B, "prompt_len": input_ids.shape[1],
               "masked": args.masked and B > 1, "steps": len(times),
               "itl_ms_p50": statistics.median(times) * 1000,
               "itl_ms_min": min(times) * 1000, "itl_ms_max": max(times) * 1000}
        if prof is not None:
            rec.update(summarize_torch(prof, len(times), label, args.out))
        print(json.dumps(rec), flush=True)
        with open(f"{args.out}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

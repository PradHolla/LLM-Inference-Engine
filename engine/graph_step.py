"""
graph_step.py -- Phase 8 follow-up: the same decode step, captured and replayed as a CUDA graph.
Compares dynamic cache (the engine today), static cache eager, and static cache + torch.compile
"reduce-overhead" (CUDA graphs); checks the greedy tokens match. See docs.

  /opt/llm/.venv/bin/python -m engine.graph_step --batches 1,8
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
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from engine.manual import make_prompt

nvtx = torch.cuda.nvtx


def prompt_ids(tok, device, B: int, prompt_tokens: int) -> torch.Tensor:
    ids = tok.apply_chat_template([{"role": "user", "content": make_prompt(prompt_tokens)}],
                                  add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                  enable_thinking=False)
    ids = ids["input_ids"] if not hasattr(ids, "shape") else ids
    return ids.to(device).repeat(B, 1)


def decode_one(model, cur, cache_position, past):
    """One greedy decode step; the unit torch.compile captures as a graph."""
    logits = model(cur, cache_position=cache_position, past_key_values=past,
                   return_dict=False, use_cache=True)[0]
    return torch.argmax(logits[:, -1], dim=-1)[:, None]


def run(model, ids, variant: str, warmup: int, steps: int, step_fn, nsys: bool):
    """Prefill eagerly, then warmup + measured decode steps. Returns (step times, tokens, warm s)."""
    B, L = ids.shape
    with torch.inference_mode():
        if variant == "dynamic":
            past = None
        else:
            past = StaticCache(config=model.config, max_cache_len=L + warmup + steps + 8)
        cache_position = torch.arange(L, device=ids.device)
        logits = model(ids, cache_position=cache_position, past_key_values=past,
                       return_dict=False, use_cache=True)
        past = logits[1] if variant == "dynamic" else past
        cur = torch.argmax(logits[0][:, -1], dim=-1)[:, None]
        cache_position = torch.tensor([L], device=ids.device)
        tokens, times = [cur[0, 0].item()], []
        t_warm = time.perf_counter()
        for step in range(warmup + steps):
            measured = step >= warmup
            if step == warmup:
                torch.cuda.synchronize()
                warm_s = time.perf_counter() - t_warm
                if nsys:
                    torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            nvtx.range_push(f"{variant} B{B} step {step}")
            if variant == "dynamic":
                out = model(cur, cache_position=cache_position, past_key_values=past,
                            return_dict=False, use_cache=True)
                past, cur = out[1], torch.argmax(out[0][:, -1], dim=-1)[:, None]
            else:
                cur = step_fn(model, cur.clone(), cache_position, past)
            tokens.append(cur[0, 0].item())            # the per-step host sync a real engine pays
            nvtx.range_pop()
            cache_position += 1
            torch.cuda.synchronize()
            if measured:
                times.append(time.perf_counter() - t0)
        if nsys:
            torch.cuda.cudart().cudaProfilerStop()
    return times, tokens, warm_s


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--batches", default="1,8")
    p.add_argument("--variants", default="dynamic,static,graph")
    p.add_argument("--prompt-tokens", type=int, default=412)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--nsys", action="store_true", help="bracket measured steps for nsys")
    p.add_argument("--out", default="results/p8-graph.jsonl")
    args = p.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1
    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model.eval()
    compiled = torch.compile(decode_one, mode="reduce-overhead", fullgraph=True)
    print(f"torch {torch.__version__} attn={model.config._attn_implementation}", flush=True)
    for B in [int(b) for b in args.batches.split(",")]:
        ids = prompt_ids(tok, device, B, args.prompt_tokens)
        reference = None
        for variant in args.variants.split(","):
            fn = compiled if variant == "graph" else decode_one
            try:
                times, tokens, warm_s = run(model, ids, variant, args.warmup, args.steps, fn,
                                            args.nsys)
            except Exception as exc:            # a failed variant must not hide the others
                rec = {"variant": variant, "batch": B, "error": f"{type(exc).__name__}: {exc}"[:500]}
                print(json.dumps(rec), flush=True)
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                continue
            reference = reference or tokens
            match = sum(a == b for a, b in zip(tokens, reference))
            rec = {"variant": variant, "batch": B, "prompt_len": ids.shape[1], "steps": len(times),
                   "itl_ms_p50": statistics.median(times) * 1000,
                   "itl_ms_min": min(times) * 1000, "itl_ms_max": max(times) * 1000,
                   "warmup_s": warm_s, "tokens_match_dynamic": f"{match}/{len(tokens)}"}
            print(json.dumps(rec), flush=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

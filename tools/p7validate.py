#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28", "transformers>=4.51", "jinja2"]
# ///
"""
p7validate.py -- the four things the offline fakes cannot tell us about Phase 7's
overlap path and budget actuator. Run ON the box against localhost:8000.

  uv run tools/p7validate.py --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUESTION = ("A tank holds 47 litres. It leaks 3 litres an hour for 5 hours, then is "
            "refilled by 12 litres. How many litres are in it? Reason it through.")
SEARCH_BLOCK = ("<sources>\n[1] Tank telemetry log: initial volume confirmed at 47 L.\n"
                "[2] Maintenance report: leak rate measured at 3 L/h, duration 5 h.\n"
                "[3] Refill record: 12 L added at 14:05.\n</sources>")


def tokcount(text: str) -> int:
    from gateway.prompt import n_tokens
    return n_tokens(text) if text else 0


async def model_of(c: httpx.AsyncClient, url: str) -> str | None:
    try:
        r = await c.get(f"{url}/v1/models", timeout=10)
        return (r.json().get("data") or [{}])[0].get("id")
    except Exception:
        return None



async def prefix_counters(c: httpx.AsyncClient, url: str) -> tuple[float, float]:
    """(hits, queries) in TOKENS -- vLLM documents both counters as token counts, not
    blocks. `_created` series are unix timestamps; binding one is incident 41."""
    r = await c.get(f"{url}/metrics", timeout=10)
    roles = {"vllm:prefix_cache_hits_total": [], "vllm:prefix_cache_queries_total": []}
    for line in r.text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in roles:
            roles[name].append(float(line.rsplit(" ", 1)[1]))
    for k, v in roles.items():
        if len(v) != 1:
            raise RuntimeError(f"{k} bound {len(v)} series, expected exactly 1")
    return roles["vllm:prefix_cache_hits_total"][0], roles["vllm:prefix_cache_queries_total"][0]


async def check_budget(c, url, model) -> dict:
    """Does thinking_token_budget actually bind? The phase gate (design doc 0c)."""
    rows = []
    for budget in (None, 1024, 256, 64):
        body = {"model": model, "messages": [{"role": "user", "content": QUESTION}],
                "max_tokens": 2000, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": True}}
        if budget is not None:
            body["thinking_token_budget"] = budget
        try:
            r = await c.post(f"{url}/v1/chat/completions", json=body, timeout=300)
            r.raise_for_status()
            m = (r.json().get("choices") or [{}])[0].get("message") or {}
        except httpx.HTTPStatusError as e:
            body = ""
            try:
                body = e.response.text[:200]
            except Exception:
                pass
            return {"verdict": "UNSUPPORTED", "rows": rows,
                    "detail": f"HTTP {e.response.status_code} at budget={budget}: {body}"}
        except Exception as e:
            return {"verdict": "ERROR", "detail": f"{type(e).__name__}: {e}", "rows": rows}
        rc = m.get("reasoning") or m.get("reasoning_content") or ""
        rows.append({"budget": budget, "reasoning_tokens": tokcount(rc),
                     "reasoning_chars": len(rc), "answer_chars": len(m.get("content") or "")})
        print(f"    budget {str(budget):>9}  reasoning {rows[-1]['reasoning_tokens']:>5} tokens")

    if all(r["reasoning_tokens"] == 0 for r in rows):
        return {"verdict": "FAIL", "rows": rows,
                "detail": "no reasoning came back at all; is --reasoning-parser loaded?"}
    seq = [r["reasoning_tokens"] for r in rows]
    decreasing = all(a >= b for a, b in zip(seq, seq[1:]))
    tight = rows[-1]["reasoning_tokens"]
    in_band = 1 <= tight <= 96
    verdict = "PASS" if decreasing and in_band else "FAIL"
    return {"verdict": verdict, "rows": rows,
            "detail": (f"unbounded {seq[0]} -> 1024 {seq[1]} -> 256 {seq[2]} -> 64 {seq[3]}; "
                       f"monotone={decreasing}, 64-arm in [1,96]={in_band}")}


async def check_splice(c, url, model) -> dict:
    """The overlap path for real: generate, cut, splice results, re-issue.
    Reports the re-issue's cached_tokens, which is the whole Q3 measurement."""
    from gateway import splice
    from gateway.prompt import render

    msgs = [{"role": "user", "content": QUESTION}]
    prompt_sent = render(msgs, enable_thinking=True, add_generation_prompt=True, patch=True)

    body = {"model": model, "prompt": prompt_sent, "max_tokens": 160,
            "temperature": 0, "stream": False}
    r = await c.post(f"{url}/v1/completions", json=body, timeout=300)
    r.raise_for_status()
    d = r.json()
    partial = (d.get("choices") or [{}])[0].get("text") or ""
    u1 = d.get("usage") or {}

    cont = splice.build(prompt_sent, partial, SEARCH_BLOCK)
    splice.verify(cont, prompt_sent, partial)

    body2 = {"model": model, "prompt": cont.prompt, "max_tokens": 400,
             "temperature": 0, "stream": False}
    h0, q0 = await prefix_counters(c, url)
    t0 = time.perf_counter()
    r2 = await c.post(f"{url}/v1/completions", json=body2, timeout=300)
    r2.raise_for_status()
    d2 = r2.json()
    reissue_ms = (time.perf_counter() - t0) * 1e3
    tail = (d2.get("choices") or [{}])[0].get("text") or ""
    h1, q1 = await prefix_counters(c, url)
    hit_tok, q_tok = h1 - h0, q1 - q0
    u2 = d2.get("usage") or {}
    det = u2.get("prompt_tokens_details") or {}
    cached = det.get("cached_tokens")

    n_prefix = tokcount(prompt_sent + partial)
    partial_blk = splice.partial_block_tokens(n_prefix)
    reported = u2.get("prompt_tokens")
    hit_frac = (cached / reported) if (cached and reported) else 0.0

    # The fraction that matters is against the PREFIX, not the whole re-issue prompt:
    # the spliced block is new text and could never have been cached.
    pref_frac = (hit_tok / n_prefix) if n_prefix else 0.0
    tot_frac = (hit_tok / q_tok) if q_tok else 0.0
    stranded = n_prefix - hit_tok
    if q_tok <= 0:
        verdict, detail = "ERROR", "the re-issue queried no prefix-cache tokens at all"
    elif hit_tok <= 0:
        verdict = "FAIL"
        detail = (f"re-issue hit 0 of {q_tok:.0f} tokens; generated tokens are NOT "
                  "reusable as a prompt prefix and the Q3 design collapses")
    else:
        verdict = "PASS" if pref_frac > 0.9 else "PARTIAL"
        detail = (f"{hit_tok:.0f}/{n_prefix} prefix tokens cached ({pref_frac:.1%}); "
                  f"{stranded:.0f} stranded against {partial_blk} predicted by block "
                  f"alignment; {hit_tok:.0f}/{q_tok:.0f} of the whole re-issue ({tot_frac:.1%})")
    print(f"    partial thinking : {len(partial)} chars, {tokcount(partial)} tokens")
    print(f"    re-issue prompt  : {reported} tokens, usage.cached_tokens={cached}")
    print(f"    prefix cache     : {hit_tok:.0f} tokens hit of {q_tok:.0f} queried")
    print(f"    re-issue latency : {reissue_ms:.0f} ms")
    print(f"    --- what the model said after the splice ---")
    print("    " + (tail[:600].replace("\n", "\n    ") or "(nothing)"))
    return {"verdict": verdict, "detail": detail, "cached_tokens": cached,
            "hit_tokens": hit_tok, "queried_tokens": q_tok, "stranded": stranded,
            "reissue_prompt_tokens": reported, "reissue_ms": reissue_ms,
            "prefix_tokens": n_prefix, "partial_block_tokens": partial_blk,
            "first_usage": u1, "partial_text": partial, "tail_text": tail}


async def running_count(c, url) -> float:
    try:
        r = await c.get(f"{url}/metrics", timeout=5)
        for line in r.text.splitlines():
            if line.startswith("vllm:num_requests_running") and "_by" not in line:
                return float(line.rsplit(" ", 1)[1])
    except Exception:
        pass
    return -1.0


async def check_abort(c, url, model) -> dict:
    """Does aborting a stream free the sequence, or does the GPU keep generating?
    If it keeps going, the overlap path spends GPU the serial path does not."""
    body = {"model": model, "prompt": f"[{uuid.uuid4().hex}] Write a very long essay about"
                                      " the history of steam engines.",
            "max_tokens": 3000, "temperature": 0, "stream": True}
    read = 0
    async with c.stream("POST", f"{url}/v1/completions", json=body, timeout=300) as r:
        async for _ in r.aiter_bytes():
            read += 1
            if read >= 3:
                break                      # leaving the context closes the connection
    during = await running_count(c, url)
    freed_at = None
    for i in range(20):
        await asyncio.sleep(0.5)
        if await running_count(c, url) <= 0:
            freed_at = (i + 1) * 0.5
            break
    verdict = "PASS" if freed_at is not None and freed_at <= 5.0 else "FAIL"
    detail = (f"running was {during:.0f} at abort, freed after {freed_at}s"
              if freed_at is not None else
              f"running was {during:.0f} at abort and never dropped within 10s")
    print(f"    {detail}")
    return {"verdict": verdict, "detail": detail, "freed_after_s": freed_at}


async def run(a) -> int:
    async with httpx.AsyncClient() as c:
        model = a.model or await model_of(c, a.url)
        if not model:
            print("ABORT: could not resolve the served model")
            return 2
        print(f"server {a.url}   model {model}\n")
        out = {}
        for name, fn in (("budget", check_budget), ("splice", check_splice),
                         ("abort", check_abort)):
            print(f"  [{name}]")
            try:
                out[name] = await fn(c, a.url, model)
            except Exception as e:
                out[name] = {"verdict": "ERROR", "detail": f"{type(e).__name__}: {e}"}
            print(f"    -> {out[name]['verdict']}: {out[name].get('detail','')}\n")

    print(f"  {'check':<10} verdict")
    for k, v in out.items():
        print(f"  {k:<10} {v['verdict']}")
    with open(a.out, "w") as f:
        f.write(json.dumps(out, default=str) + "\n")
    print(f"\n  written to {a.out}")
    return 0 if all(v["verdict"] == "PASS" for v in out.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="results/p7-validate.jsonl")
    a = ap.parse_args()
    return asyncio.run(run(a))


if __name__ == "__main__":
    raise SystemExit(main())

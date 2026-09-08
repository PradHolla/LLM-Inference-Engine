#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28"]
# ///
"""
p7prio.py -- does vLLM honour request priority? Assumes load is already being applied
by `vllm bench serve`; this only fires matched high/low pairs into it and times them.

  uv run tools/p7prio.py --url http://localhost:8000 --pairs 6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import sys
import time
import uuid

import httpx

MIN_QUEUE = 5   # a queue of one reorders nothing; this must be deep enough to observe


async def queue_depth(client: httpx.AsyncClient, url: str) -> float:
    """Waiting requests, from the engine's own counter. 0.0 if unreadable."""
    try:
        r = await client.get(f"{url}/metrics", timeout=5)
        for line in r.text.splitlines():
            if line.startswith("vllm:num_requests_waiting") and "_by_reason" not in line:
                return float(line.rsplit(" ", 1)[1])
    except Exception:
        pass
    return 0.0


async def one(client, url, model, priority, tokens) -> dict:
    """One timed request. Unique text so it cannot ride another request's cached prefix."""
    body = {"model": model,
            "messages": [{"role": "user",
                          "content": f"[{uuid.uuid4().hex}] Explain briefly why the sky is blue."}],
            "max_tokens": tokens, "temperature": 0, "priority": priority}
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{url}/v1/chat/completions", json=body, timeout=300)
        r.raise_for_status()
        return {"priority": priority, "s": time.perf_counter() - t0, "error": None}
    except Exception as e:
        return {"priority": priority, "s": time.perf_counter() - t0,
                "error": f"{type(e).__name__}"}


async def run(a) -> int:
    async with httpx.AsyncClient() as client:
        model = a.model
        if not model:
            try:
                r = await client.get(f"{a.url}/v1/models", timeout=10)
                model = (r.json().get("data") or [{}])[0].get("id")
            except Exception:
                model = None
        if not model:
            print("ABORT: could not resolve the served model")
            return 2

        # REGIME GATE. Priority can only reorder work that is waiting. Without a real
        # queue the result is untested, not negative -- this phase has produced four
        # confident nulls from experiments that never reached their own regime.
        print(f"  waiting for a queue of at least {MIN_QUEUE} ...", flush=True)
        peak = 0.0
        for _ in range(a.wait_polls):
            peak = max(peak, await queue_depth(client, a.url))
            if peak >= MIN_QUEUE:
                break
            await asyncio.sleep(1.0)
        print(f"  peak queue observed: {peak:.0f}")
        if peak < MIN_QUEUE:
            print(f"\n  verdict: UNTESTED -- queue never reached {MIN_QUEUE}. "
                  "Is the load generator running?")
            return 3

        # Matched pairs: identical work, fired together, differing only in priority.
        # Comparing against the load generator's own numbers would confound prompt shape.
        rows = []
        for i in range(a.pairs):
            hi, lo = await asyncio.gather(one(client, a.url, model, 0, a.tokens),
                                          one(client, a.url, model, 100, a.tokens))
            rows.append({"pair": i, "high_s": hi["s"], "low_s": lo["s"],
                         "high_err": hi["error"], "low_err": lo["error"],
                         "queue_at_fire": peak})
            print(f"    pair {i}: high {hi['s']:6.2f}s   low {lo['s']:6.2f}s")

    good = [r for r in rows if not r["high_err"] and not r["low_err"]]
    if not good:
        print("\n  verdict: ERROR -- every pair failed")
        return 1
    mh = st.median([r["high_s"] for r in good])
    ml = st.median([r["low_s"] for r in good])
    verdict = "PASS" if mh < ml * 0.8 else "FAIL"
    print(f"\n  {len(good)} matched pairs under a queue of {peak:.0f}")
    print(f"  high priority median {mh:.2f}s   low priority median {ml:.2f}s   ratio {ml/mh:.2f}x")
    print(f"  verdict: {verdict}")
    with open(a.out, "w") as f:
        f.write(json.dumps({"verdict": verdict, "high_median_s": mh, "low_median_s": ml,
                            "peak_queue": peak, "pairs": good}) + "\n")
    return 0 if verdict == "PASS" else 1


def selftest() -> int:
    """Offline. The regime gate must refuse rather than report a negative."""
    fails = []

    class NoQueue:
        async def get(self, *a, **k):
            return httpx.Response(200, text="vllm:num_requests_waiting{a=\"b\"} 0.0")
        async def post(self, *a, **k):
            raise httpx.ConnectError("refused")

    async def check():
        c = NoQueue()
        return await queue_depth(c, "http://x")
    if asyncio.run(check()) != 0.0:
        fails.append("queue_depth should read 0.0 from a zero counter")

    class Decoy:
        async def get(self, *a, **k):
            return httpx.Response(200, text=(
                "vllm:num_requests_waiting_by_reason{reason=\"x\"} 99.0\n"
                "vllm:num_requests_waiting{a=\"b\"} 7.0\n"))
    async def check2():
        return await queue_depth(Decoy(), "http://x")
    if asyncio.run(check2()) != 7.0:
        fails.append("the _by_reason decoy was summed into the real counter (incident 41)")

    for f in fails:
        print(f"  FAIL {f}")
    print(f"selftest: {'PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--wait-polls", type=int, default=60)
    ap.add_argument("--out", default="results/p7-priority.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    return selftest() if a.selftest else asyncio.run(run(a))


if __name__ == "__main__":
    raise SystemExit(main())

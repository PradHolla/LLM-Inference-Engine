#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28"]
# ///
"""
p7search.py -- probe the retrieval slice through the real search path, serially.
Validates the slice returns pages AND measures the latency anchor Q2/Q3 predict against.

  uv run tools/p7search.py --items results/p7-retrieval-items.jsonl --out results/p7-search-probe.jsonl
"""
import argparse, asyncio, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gateway.search import BRAVE_URL, N_RESULTS, load_key, run_search  # noqa: E402

import httpx  # noqa: E402

# Brave's free plan allows 1 query/second and answers a burst with 429, which run_search
# degrades to zero sources exactly like a genuine miss. Space the calls instead.
MIN_GAP_S = 1.2


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))]


async def plan_headers(client) -> dict:
    """Read the plan's own limits off a live response rather than assuming a tier."""
    key = load_key()
    if not key:
        return {"error": "no BRAVE_API_KEY"}
    r = await client.get(BRAVE_URL, params={"q": "test", "count": 1},
                         headers={"Accept": "application/json", "X-Subscription-Token": key},
                         timeout=10.0)
    return {"status": r.status_code,
            **{k: v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")}}


async def main(args):
    items = [json.loads(l) for l in open(args.items) if l.strip()]
    if args.limit:
        items = items[:args.limit]
    recs = []
    async with httpx.AsyncClient() as client:
        hdr = await plan_headers(client)
        print(f"plan   {hdr}")
        await asyncio.sleep(MIN_GAP_S)
        for n, it in enumerate(items, 1):
            t0 = time.perf_counter()
            out = await run_search(it.get("query") or it["prompt"], client)
            wall = (time.perf_counter() - t0) * 1e3
            rec = {"id": it["id"], "query": out.query, "wall_ms": wall,
                   "search_ms": out.search_ms, "fetch_ms": out.fetch_ms,
                   "extract_ms": out.extract_ms, "n_sources": out.n_sources,
                   "error": out.error,
                   "tokens_est": sum(s.tokens_est for s in out.sources if s.ok),
                   "urls": [s.url for s in out.sources]}
            recs.append(rec)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if n % 10 == 0 or n == len(items):
                print(f"  {n}/{len(items)}", flush=True)
            gap = MIN_GAP_S - (time.perf_counter() - t0)
            if gap > 0:
                await asyncio.sleep(gap)

    zero = [r for r in recs if r["n_sources"] == 0]
    errs = {}
    for r in recs:
        if r["error"]:
            errs[r["error"][:40]] = errs.get(r["error"][:40], 0) + 1
    print(f"\n  slice {Path(args.items).name}   n={len(recs)}   n_results_requested={N_RESULTS}")
    print(f"  {'stage':<12} {'p50':>9} {'p95':>9} {'max':>9}")
    for k in ("search_ms", "fetch_ms", "extract_ms", "wall_ms"):
        xs = [r[k] for r in recs if r[k] is not None]
        print(f"  {k:<12} {pct(xs,50):>9.1f} {pct(xs,95):>9.1f} {pct(xs,100):>9.1f}")
    src = [r["n_sources"] for r in recs]
    tok = [r["tokens_est"] for r in recs if r["n_sources"]]
    print(f"\n  sources      mean {statistics.mean(src):.2f}   zero-source {len(zero)}/{len(recs)}"
          f"  ({len(zero)/max(1,len(recs)):.1%})")
    print(f"  block tokens p50 {pct(tok,50):.0f}  p95 {pct(tok,95):.0f}  (est, {N_RESULTS} sources)")
    print(f"  errors       {errs or 'none'}")
    # The whole point of the probe: a slice that returns nothing cannot test an overlap.
    verdict = "USABLE" if len(zero) / max(1, len(recs)) <= 0.10 else "UNUSABLE"
    print(f"\n  verdict {verdict} -- Q3 needs a real round trip on most items")
    return 0 if verdict == "USABLE" else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--items", default="results/p7-retrieval-items.jsonl")
    p.add_argument("--out", default="results/p7-search-probe.jsonl")
    p.add_argument("--limit", type=int, default=0)
    raise SystemExit(asyncio.run(main(p.parse_args())))

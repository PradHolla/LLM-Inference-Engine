#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["fastembed==0.9.1"]
# ///
"""Time knowledge search (candidates + rerank) one at a time and N at once, on the dev knowledge questions.

  uv run tools/searchbench.py --org b2b --concurrency 3
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crm import retrieval  # noqa: E402
from crm.tasks import load  # noqa: E402


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def one(org: str, q: str) -> tuple[float, float, str, list[str]]:
    """(total ms, rerank ms, which reranker, returned passage texts) for one search."""
    t0 = time.perf_counter()
    r = retrieval.search(org, q, 5)
    return (time.perf_counter() - t0) * 1000, r.rerank_ms, r.reranker, [p.text for p in (r.passages or r.ranked[:5])]


def report(name: str, rows: list) -> None:
    tot, rr = [r[0] for r in rows], [r[1] for r in rows]
    where = collections.Counter(r[2] for r in rows)
    print(f"{name:28s} n={len(rows):3d}  total p50 {statistics.median(tot):6.0f} p95 {pct(tot, .95):6.0f} ms | "
          f"rerank p50 {statistics.median(rr):6.0f} | candidates p50 "
          f"{statistics.median([r[0] - r[1] for r in rows]):5.0f} | reranker {dict(where)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default="b2b")
    ap.add_argument("--concurrency", type=int, default=3)
    a = ap.parse_args()
    qs = [t.query for t in load(a.org) if t.split == "dev" and t.task_type == "knowledge_qa"]
    print(f"{len(qs)} dev knowledge questions; reranker {retrieval.RERANK_MODEL}; {retrieval.CANDIDATES} candidates; "
          f"cpus {os.cpu_count()}")
    t0 = time.perf_counter()
    one(a.org, qs[0])
    print(f"cold first search (loads models): {(time.perf_counter() - t0) * 1000:.0f} ms")
    for q in qs[:3]:
        one(a.org, q)  # warm-up on the real shape
    serial = [one(a.org, q) for q in qs]
    report("one at a time", serial)
    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as ex:
        rows = list(ex.map(lambda q: one(a.org, q), qs))
    report(f"{a.concurrency} at once", rows)
    if os.environ.get("CRM_RERANK_URL"):  # same searches through the local CPU reranker, for agreement
        url = os.environ.pop("CRM_RERANK_URL")
        cpu = [one(a.org, q) for q in qs]
        os.environ["CRM_RERANK_URL"] = url
        top1 = sum(g[3][:1] == c[3][:1] for g, c in zip(serial, cpu))
        same5 = sum(g[3] == c[3] for g, c in zip(serial, cpu))
        print(f"vs CPU reranker: same top passage {top1}/{len(qs)}, same top-5 in order {same5}/{len(qs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Summarise the Phase 7 Q1a budget ladder into one table.

Usage: uv run tools/q1acurve.py results/p7q1a-*.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ARM_ORDER = ["b0", "b128", "b256", "b512", "b1024", "b2048", "unbounded"]


def arm_of(path: Path) -> str:
    return path.stem.replace("p7q1a-", "")


def load(path: Path) -> list[dict]:
    recs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


def pct(xs: list[bool]) -> float:
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def quant(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(int(q * len(s)), len(s) - 1)
    return s[i]


def reasoning_tokens(r: dict) -> float:
    """Split usage_completion by the char ratio; reasoning tokens are not reported directly."""
    rc, cc = r["reasoning_chars"], r["content_chars"]
    if rc + cc == 0:
        return 0.0
    return r["usage_completion"] * rc / (rc + cc)


def summarise(arm: str, recs: list[dict]) -> dict:
    think = [r for r in recs if r["thinking"] and r["status"] == "ok"]
    nothink = [r for r in recs if not r["thinking"] and r["status"] == "ok"]
    bad = [r for r in recs if r["status"] != "ok"]
    rt = [reasoning_tokens(r) for r in think]
    e2e = [r["e2e"] for r in think]
    return {
        "arm": arm,
        "n_think": len(think),
        "n_nothink": len(nothink),
        "bad": len(bad),
        "acc_think": pct([r["correct"] for r in think]),
        "acc_nothink": pct([r["correct"] for r in nothink]),
        "unparsed": sum(1 for r in think if not r["parsed"]),
        "trunc": sum(1 for r in think if r["truncated"]),
        "rt_mean": statistics.fmean(rt) if rt else float("nan"),
        "rt_p50": quant(rt, 0.50),
        "rt_p95": quant(rt, 0.95),
        "out_p50": quant([float(r["usage_completion"]) for r in think], 0.50),
        "ttft_p50": quant([r["ttft"] for r in think], 0.50),
        "e2e_p50": quant(e2e, 0.50),
        "e2e_p95": quant(e2e, 0.95),
        "finish": {},
    }


def control_check(by_arm: dict[str, list[dict]]) -> tuple[bool, list[str]]:
    """P7-Q1a-6: the nothink pass must be identical, item by item, across every arm."""
    ref_arm, notes = None, []
    ref: dict[str, bool] = {}
    for arm in ARM_ORDER:
        if arm not in by_arm:
            continue
        got = {r["id"]: r["correct"] for r in by_arm[arm]
               if not r["thinking"] and r["status"] == "ok"}
        if ref_arm is None:
            ref_arm, ref = arm, got
            continue
        shared = set(ref) & set(got)
        diff = [i for i in shared if ref[i] != got[i]]
        if diff:
            notes.append(f"{arm} differs from {ref_arm} on {len(diff)}/{len(shared)} items: "
                         + ", ".join(sorted(diff)[:5]))
        if set(ref) != set(got):
            notes.append(f"{arm} item set differs from {ref_arm} "
                         f"({len(set(got) - set(ref))} extra, {len(set(ref) - set(got))} missing)")
    return (not notes), notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()

    by_arm = {}
    for p in args.files:
        by_arm[arm_of(p)] = load(p)

    rows = [summarise(a, by_arm[a]) for a in ARM_ORDER if a in by_arm]

    print("Qwen3-8B fp8 weights / fp8 KV, gsm8k 200 items, concurrency 32, "
          "reasoning-parser qwen3, V2 runner OFF")
    print()
    hdr = (f"{'arm':>9} {'n':>4} {'acc%':>6} {'nothink%':>9} {'rtok_mean':>10} "
           f"{'rtok_p50':>9} {'rtok_p95':>9} {'out_p50':>8} {'ttft_p50':>9} "
           f"{'e2e_p50':>8} {'e2e_p95':>8} {'trunc':>6} {'unpars':>7} {'bad':>4}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:>9} {r['n_think']:>4} {r['acc_think']:>6.1f} "
              f"{r['acc_nothink']:>9.1f} {r['rt_mean']:>10.0f} {r['rt_p50']:>9.0f} "
              f"{r['rt_p95']:>9.0f} {r['out_p50']:>8.0f} {r['ttft_p50']:>9.3f} "
              f"{r['e2e_p50']:>8.2f} {r['e2e_p95']:>8.2f} {r['trunc']:>6} "
              f"{r['unparsed']:>7} {r['bad']:>4}")

    ok, notes = control_check(by_arm)
    print()
    print(f"P7-Q1a-6 nothink control: {'IDENTICAL across arms' if ok else 'DRIFTED'}")
    for n in notes:
        print(f"  {n}")

    unb = next((r for r in rows if r["arm"] == "unbounded"), None)
    if unb:
        print()
        print(f"gap to unbounded ({unb['acc_think']:.1f}%):")
        for r in rows:
            if r["arm"] != "unbounded":
                print(f"  {r['arm']:>9}  {r['acc_think'] - unb['acc_think']:+.1f} pts   "
                      f"e2e_p50 {r['e2e_p50'] / unb['e2e_p50']:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())

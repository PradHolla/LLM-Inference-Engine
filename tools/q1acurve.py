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
BUDGET = {"b0": 0, "b128": 128, "b256": 256, "b512": 512,
          "b1024": 1024, "b2048": 2048, "unbounded": None}


def arm_of(path: Path) -> str:
    return path.stem.replace("rec-", "").replace("p7q1a-", "").replace("p7q1c-", "").replace("p7q1m-", "").replace("p7q1n-", "")


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def pct(xs) -> float:
    xs = list(xs)
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def quant(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(int(q * len(s)), len(s) - 1)]


def rtokens(r: dict) -> float | None:
    """Prefer the recovered count; fall back to splitting usage by the char ratio."""
    if r.get("reasoning_tokens_est") is not None:
        return float(r["reasoning_tokens_est"])
    rc, cc = r.get("reasoning_chars", 0), r.get("content_chars", 0)
    if rc + cc == 0 or r.get("usage_completion") is None:
        return None
    return r["usage_completion"] * rc / (rc + cc)


def leaked(r: dict) -> bool:
    """A stray "</think>" in the VISIBLE stream. Scanned raw: think_path reports only the
    first branch taken and goes quiet about this once reasoning lands in its own field."""
    cc = r.get("content_chars") or 0
    return "</think>" in ((r.get("text") or "")[-cc:] if cc else "")


def silence(r: dict) -> float | None:
    """Time until the first VISIBLE token -- what the user actually waits through."""
    return r.get("tt_content") if r.get("tt_content") is not None else r.get("ttft")


def summarise(arm: str, recs: list[dict]) -> dict:
    th = [r for r in recs if r["thinking"] and r["status"] == "ok"]
    nt = [r for r in recs if not r["thinking"] and r["status"] == "ok"]
    rt = [v for v in (rtokens(r) for r in th) if v is not None]
    sil = [v for v in (silence(r) for r in th) if v is not None]
    nleak = sum(1 for r in th if leaked(r))
    return {
        "arm": arm, "n": len(th),
        "acc": pct(r["correct"] for r in th),
        "acc_nt": pct(r["correct"] for r in nt),
        "rt_p50": quant(rt, .50), "rt_p95": quant(rt, .95),
        "out_p50": quant([float(r["usage_completion"]) for r in th if r["usage_completion"]], .50),
        "sil_p50": quant(sil, .50), "sil_p95": quant(sil, .95),
        "e2e_p50": quant([r["e2e"] for r in th], .50),
        "e2e_p95": quant([r["e2e"] for r in th], .95),
        "trunc": sum(1 for r in th if r["truncated"]),
        "leak": 100.0 * nleak / len(th) if th else float("nan"),
    }


def control(by_arm: dict[str, list[dict]]) -> list[str]:
    """P7-Q1a-6: the nothink pass carries no budget, so it must not move across arms."""
    ref_arm, ref, notes = None, {}, []
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
            notes.append(f"{arm} vs {ref_arm}: {len(diff)}/{len(shared)} items differ "
                         f"({100*len(diff)/len(shared):.1f}%)")
    return notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    by_arm = {arm_of(p): load(p) for p in args.files}
    rows = [summarise(a, by_arm[a]) for a in ARM_ORDER if a in by_arm]

    any_recs = next(iter(by_arm.values()))
    sl = sorted({r.get("slice", "?") for r in any_recs})
    cfg = sorted({r.get("config", "?") for r in any_recs})
    print(f"Qwen3-8B, fp8 weights / fp8 KV, KV pinned, slice {'+'.join(sl)}, "
          f"{rows[0]['n']} items/arm, concurrency 32,")
    print(f"reasoning-parser qwen3, VLLM_USE_V2_MODEL_RUNNER=0. configs: {', '.join(cfg)}")
    print("rtok = reasoning tokens actually emitted. silence = time to first VISIBLE token.")
    print()
    hdr = (f"{'arm':>10} {'budget':>7} {'acc%':>6} {'nothink%':>9} {'rtok_p50':>9} "
           f"{'rtok_p95':>9} {'out_p50':>8} {'sil_p50':>8} {'sil_p95':>8} "
           f"{'e2e_p50':>8} {'trunc':>6} {'leak%':>6}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        b = BUDGET.get(r["arm"])
        print(f"{r['arm']:>10} {('none' if b is None else b):>7} {r['acc']:>6.1f} "
              f"{r['acc_nt']:>9.1f} {r['rt_p50']:>9.0f} {r['rt_p95']:>9.0f} "
              f"{r['out_p50']:>8.0f} {r['sil_p50']:>8.2f} {r['sil_p95']:>8.2f} "
              f"{r['e2e_p50']:>8.2f} {r['trunc']:>6} {r['leak']:>6.1f}")

    notes = control(by_arm)
    print()
    print(f"P7-Q1a-6 nothink control: {'no drift' if not notes else 'DRIFT'}")
    for n in notes:
        print(f"  {n}")

    unb = next((r for r in rows if r["arm"] == "unbounded"), None)
    if unb:
        print()
        print(f"against unbounded (acc {unb['acc']:.1f}%, silence p50 {unb['sil_p50']:.2f}s):")
        for r in rows:
            if r["arm"] != "unbounded":
                print(f"  {r['arm']:>10}  {r['acc'] - unb['acc']:+6.1f} pts   "
                      f"silence {unb['sil_p50'] / r['sil_p50']:>5.2f}x faster")
    return 0


if __name__ == "__main__":
    sys.exit(main())

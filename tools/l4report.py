#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""l4report.py -- the L4 vs A10G table: recomputes BOTH cards from raw bench records with one
set of definitions, and refuses to report if the A10G side does not reproduce Phase 6B.
  uv run tools/l4report.py [--out results/l4-report.md]
"""
from __future__ import annotations
import argparse, json, math, re, sys
from pathlib import Path

R = Path("results")
DUR = 45.0                        # seconds per load point, both cards
LONG_CHARS = 4000                 # longctx items are far above this; math and gsm8k far below
# Bytes read per decode step: embedding is a gather, lm_head stays bf16 under every scheme.
BODY, EMB = 6_946_075_648, 622_329_856
BYTES = {"bf16": BODY * 2 + EMB * 2, "fp8": BODY + EMB * 2, "fp8m": BODY + EMB * 2,
         "int4": BODY // 2 + BODY // 128 * 2 + EMB * 2}
PEAK = {"a10g": 600e9, "l4": 300e9}
# Published Phase 6B values (results/phase6-bombard-summary.txt): ITL p50 ms, rps @2, rps @6.
P6B = {"bf16": (34.2, 1.72, 3.58), "fp8": (18.9, 1.65, 4.52), "int4": (11.9, 1.65, 4.73)}
A10G_BLOCKS = ["bf16", "fp8", "fp8-spec", "int4", "int4-spec"]   # bf16-spec never launched
L4_CONFIGS = ["bf16", "fp8", "fp8m", "int4"]


def pct(xs, p):
    """bench.py's linear-interpolated percentile."""
    if not xs:
        return math.nan
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def point(recs):
    """One load point, with bench.py's summarize() definitions."""
    ok = [r for r in recs if r["status"] == "ok" and not r["warmup"]]
    if not ok:
        return None
    span = max(r["t_sent"] + (r["e2e"] or 0) for r in ok) - min(r["t_sent"] for r in ok)
    dur = max(span, DUR * 0.5) if recs[0]["rate"] else None
    return {
        "n": len(ok), "fail": sum(r["status"] != "ok" for r in recs),
        "ttft50": pct([r["ttft"] * 1000 for r in ok if r["ttft"] is not None], 50),
        "ttft95": pct([r["ttft"] * 1000 for r in ok if r["ttft"] is not None], 95),
        "itl50": pct([x * 1000 for r in ok for x in r["itls"]], 50),
        "rps": len(ok) / dur if dur else None,
        "toks": sum(r["usage_tokens"] for r in ok) / dur if dur else None,
        "long": sum(r["prompt_chars"] > LONG_CHARS for r in ok),
    }


def blocks(recs):
    """Split a file into consecutive runs of the same rate (serial = 0.0)."""
    out = []
    for r in recs:
        if not out or out[-1][0]["rate"] != r["rate"]:
            out.append([])
        out[-1].append(r)
    return out


def a10g():
    bl = blocks(load(R / "phase6-bombard.jsonl"))
    if [b[0]["rate"] for b in bl] != [0.0, 2.0, 6.0] * len(A10G_BLOCKS):
        sys.exit(f"ABORT: phase6-bombard.jsonl splits into {len(bl)} blocks, not 3 x {len(A10G_BLOCKS)}")
    res = {}
    for i, name in enumerate(A10G_BLOCKS):
        if name.endswith("-spec"):
            continue
        s, r2, r6 = (point(b) for b in bl[3 * i:3 * i + 3])
        res[name] = {"serial": s, "r2": r2, "r6": r6, "raw": bl[3 * i + 1:3 * i + 3]}
        want = P6B[name]
        got = (s["itl50"], r2["rps"], r6["rps"])
        if any(abs(g - w) > 0.051 for g, w in zip(got, want)):
            sys.exit(f"ABORT: A10G {name} recomputes to {got}, Phase 6B published {want}")
    return res


def launch_facts(cfg):
    p = R / f"l4-{cfg}-launch.txt"
    t = p.read_text() if p.exists() else ""
    kv = re.search(r"GPU KV cache size: ([0-9,]+) tokens", t)
    kern = re.findall(r"Selected (\w+) for (\w+)", t)
    gib = re.search(r"Model loading took ([0-9.]+) GiB", t)
    return {"kv": kv.group(1) if kv else "?", "kernel": ", ".join(k for k, _ in kern) or "?",
            "weights": gib.group(1) if gib else "?", "pinned": "pinned at" in t}


def l4():
    res = {}
    for cfg in L4_CONFIGS:
        p = R / f"l4-{cfg}-bombard.jsonl"
        if not p.exists():
            continue
        bl = blocks(load(p))
        if [b[0]["rate"] for b in bl] != [0.0, 2.0, 6.0]:
            sys.exit(f"ABORT: {p} has blocks {[b[0]['rate'] for b in bl]}")
        s, r2, r6 = (point(b) for b in bl)
        pf = R / f"l4-{cfg}-prefill.jsonl"
        pre = point(load(pf)) if pf.exists() else None
        pt = R / f"l4-{cfg}-prompt-tokens.txt"
        res[cfg] = {"serial": s, "r2": r2, "r6": r6, "prefill": pre, "raw": bl[1:],
                    "prompt_tokens": float(pt.read_text()) if pt.exists() else math.nan,
                    **launch_facts(cfg)}
    return res


def f(v, d=1):
    return "--" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:,.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "l4-report.md"))
    args = ap.parse_args()
    A, L = a10g(), l4()
    for cfg, d in L.items():
        twin = A["fp8" if cfg == "fp8m" else cfg]
        for mine, theirs in zip(d["raw"], twin["raw"]):
            seq = lambda b: [r["prompt_chars"] for r in sorted(b, key=lambda r: r["t_arrival"])]
            if seq(mine) != seq(theirs):
                sys.exit(f"ABORT: L4 {cfg} rate {mine[0]['rate']} did not replay the A10G's prompts")
    lines = ["## Decode, one user  [Qwen3-8B, vLLM 0.27.1, fp16 KV, 16,384 ctx, no speculation, "
             "phase4-items prompts, 128 output tokens, thinking on, T=0]",
             "card | weights | fp8 kernel | KV tokens | ITL p50 | tok/s | achieved GB/s | % of peak | TTFT p50",
             "---|---|---|---|---|---|---|---|---"]
    for cfg in ["bf16", "fp8", "fp8m", "int4"]:
        for card, D in (("A10G", A), ("L4", L)):
            src = "fp8" if (card == "A10G" and cfg == "fp8m") else cfg
            if card == "A10G" and cfg == "fp8":
                continue                    # the A10G's only fp8 path IS Marlin: listed as fp8m
            if src not in D:
                continue
            d = D[src]; itl = d["serial"]["itl50"]
            gbs = BYTES[cfg] / (itl / 1000) / 1e9
            kern = d.get("kernel", "Marlin (sm86 default)") if card == "L4" else (
                "Marlin (sm86 default)" if cfg == "fp8m" else "--")
            kv = d.get("kv", {"bf16": "33,312", "fp8": "74,880", "int4": "101,920"}[src])
            lines.append(f"{card} | {cfg} | {kern if 'fp8' in cfg else '--'} | {kv} | {f(itl)} ms | "
                         f"{f(1000 / itl)} | {f(gbs, 0)} | {f(100 * gbs * 1e9 / PEAK[card.lower()])}% | "
                         f"{f(d['serial']['ttft50'], 0)} ms")
    lines += ["", "## Under load  [same config; Poisson arrivals, the L4 replays the A10G run's exact "
              "arrival times and prompts]",
              "card | weights | rate | ok | req/s served | out tok/s | TTFT p50 | TTFT p95 | ITL p50 | long prompts in window",
              "---|---|---|---|---|---|---|---|---|---"]
    for cfg in ["bf16", "fp8m", "fp8", "int4"]:
        for card, D in (("A10G", A), ("L4", L)):
            src = "fp8" if (card == "A10G" and cfg == "fp8m") else cfg
            if (card == "A10G" and cfg == "fp8") or src not in D:
                continue
            for rk, rate in (("r2", 2), ("r6", 6)):
                p = D[src][rk]
                lines.append(f"{card} | {cfg} | {rate} | {p['n']} | {f(p['rps'], 2)} | {f(p['toks'], 0)} | "
                             f"{f(p['ttft50'], 0)} ms | {f(p['ttft95'], 0)} ms | {f(p['itl50'])} ms | {p['long']}")
    lines += ["", "## Prefill, one user, L4 only  [filler prompt, unique prefix, max_tokens 1; "
              "prompt tokens from vLLM's own prompt_tokens counter]",
              "weights | fp8 kernel | prompt tokens | TTFT p50 | ms per 1k prompt tokens | vs bf16",
              "---|---|---|---|---|---"]
    base = L.get("bf16", {}).get("prefill")
    for cfg in ["bf16", "fp8", "fp8m", "int4"]:
        if cfg not in L or not L[cfg]["prefill"]:
            continue
        d = L[cfg]; t = d["prefill"]["ttft50"]; n = d["prompt_tokens"]
        rel = f"{base['ttft50'] / t:.2f}x" if base else "--"
        lines.append(f"{cfg} | {d['kernel'] if 'fp8' in cfg else '--'} | {f(n, 0)} | {f(t, 0)} ms | "
                     f"{f(1000 * t / n if n == n else math.nan)} | {rel}")
    lines += ["", "L4 launch facts: " + "; ".join(
        f"{c}: weights {L[c]['weights']} GiB, KV {L[c]['kv']}{' (pinned)' if L[c]['pinned'] else ' (PROFILED, pin failed)'}"
        for c in L)]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

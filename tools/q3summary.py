#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
q3summary.py -- the Q2/Q3 table. Client records and gateway traces per arm, side by side.

  uv run tools/q3summary.py --prefix results/p7q3 --arms retrieve_then_generate,overlap,generate_then_retrieve
"""
import argparse, json, os, sys


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))]


def load(path):
    try:
        return [json.loads(l) for l in open(path) if l.strip()]
    except OSError:
        return []


def main(a):
    arms = a.arms.split(",")
    print(f"\n  Qwen3-8B fp8, KV fp8, prefix caching on, budget 2048, max_tokens {a.max_tokens},")
    print(f"  slice retrieval (n={a.n}), concurrency {a.concurrency}, "
          f"temp 0.6 top_p 0.95 top_k 20, gateway search always on")

    print(f"\n  {'arm':<24} {'n':>4} {'acc':>7} {'trunc':>7} {'tok p50':>8} {'tokp99':>7} "
          f"{'ttft p50':>9} {'ttft p95':>9} {'e2e p50':>9} {'e2e p95':>9}")
    rows = {}
    for arm in arms:
        recs = [r for r in load(f"{a.prefix}-{arm}.jsonl") if r.get("status") == "ok"]
        rows[arm] = recs
        if not recs:
            print(f"  {arm:<24} {'NO DATA':>4}")
            continue
        acc = sum(bool(r.get("correct")) for r in recs) / len(recs)
        tru = sum(r.get("finish_reason") == "length" for r in recs) / len(recs)
        print(f"  {arm:<24} {len(recs):>4} {acc:>6.1%} {tru:>6.1%} "
              f"{pct([r.get('usage_completion') for r in recs], 50):>8.0f} "
              f"{pct([r.get('usage_completion') for r in recs], 99):>7.0f} "
              f"{pct([r.get('ttft') for r in recs], 50) * 1e3:>9.0f} "
              f"{pct([r.get('ttft') for r in recs], 95) * 1e3:>9.0f} "
              f"{pct([r.get('e2e') for r in recs], 50) * 1e3:>9.0f} "
              f"{pct([r.get('e2e') for r in recs], 95) * 1e3:>9.0f}")

    print(f"\n  {'arm':<24} {'src':>5} {'zero':>6} {'err':>4} {'search p50':>11} "
          f"{'pre tok':>8} {'splice ch':>10} {'reissue tok':>12}")
    for arm in arms:
        tr = [t for t in load(f"{a.traces}-q3-{arm}.jsonl") if t.get("status") == "ok"]
        if not tr:
            print(f"  {arm:<24} {'NO TRACE':>5}")
            continue
        zero = sum(1 for t in tr if not t.get("n_sources"))
        err = sum(1 for t in tr if t.get("search_error"))
        print(f"  {arm:<24} {sum(t['n_sources'] for t in tr)/len(tr):>5.2f} "
              f"{zero:>3}/{len(tr):<2} {err:>4} "
              f"{pct([t.get('search_ms') for t in tr], 50):>11.0f} "
              f"{pct([t.get('overlap_pre_tokens') for t in tr], 50):>8.0f} "
              f"{pct([t.get('splice_chars') for t in tr], 50):>10.0f} "
              f"{pct([t.get('reissue_prompt_tokens') for t in tr], 50):>12.0f}")

    base = arms[0]
    if rows.get(base):
        print(f"\n  against {base}:")
        for arm in arms[1:]:
            if not rows.get(arm):
                continue
            d_t = (pct([r.get('ttft') for r in rows[arm]], 50)
                   - pct([r.get('ttft') for r in rows[base]], 50)) * 1e3
            d_e = (pct([r.get('e2e') for r in rows[arm]], 50)
                   - pct([r.get('e2e') for r in rows[base]], 50)) * 1e3
            print(f"    {arm:<24} ttft p50 {d_t:+8.0f} ms   e2e p50 {d_e:+8.0f} ms")

    if os.path.exists(a.cache):
        print(f"\n  prefix cache, vLLM counters (hits queries, in tokens):")
        for line in open(a.cache):
            print("   " + line.rstrip())
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="results/p7q3")
    p.add_argument("--traces", default="results/p7gw")
    p.add_argument("--cache", default="results/p7q3-cache.txt")
    p.add_argument("--arms", default="retrieve_then_generate,overlap,generate_then_retrieve")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=3072)
    raise SystemExit(main(p.parse_args()))

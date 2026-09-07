#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
mkprompts.py -- extract one slice of an item set into a bench.py prompts file.
Built on the box rather than shipped, since sync carries code and not results.

  uv run tools/mkprompts.py --items results/phase4-items.jsonl --slice longctx --out results/longctx-prompts.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys


def extract(items_path: str, slice_name: str) -> list[str]:
    """Every prompt whose slice matches. Order preserved so runs are comparable."""
    out: list[str] = []
    with open(items_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("slice") == slice_name and d.get("prompt"):
                out.append(d["prompt"])
    return out


def selftest() -> int:
    """Offline, against a fixture written to a temp file."""
    import tempfile
    import os
    fails = []
    rows = [{"slice": "longctx", "prompt": "long one"},
            {"slice": "math", "prompt": "short one"},
            {"slice": "longctx", "prompt": "long two"},
            {"slice": "longctx"},
            {"not_json": True}]
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write("not json at all\n")
    got = extract(path, "longctx")
    os.unlink(path)
    if got != ["long one", "long two"]:
        fails.append(f"extract wrong: {got!r}")
    if extract("/nonexistent-on-purpose", "x") if os.path.exists("/nonexistent-on-purpose") else False:
        fails.append("should not have found anything")
    for f_ in fails:
        print(f"  FAIL {f_}")
    print(f"selftest: {'PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="results/phase4-items.jsonl")
    ap.add_argument("--slice", dest="slice_name", default="longctx")
    ap.add_argument("--out", default="results/longctx-prompts.jsonl")
    ap.add_argument("--min", type=int, default=1, help="fail if fewer than this many match")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    prompts = extract(a.items, a.slice_name)
    if len(prompts) < a.min:
        print(f"ABORT: only {len(prompts)} prompts in slice {a.slice_name!r}, need {a.min}")
        return 2
    with open(a.out, "w") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}) + "\n")
    chars = sorted(len(p) for p in prompts)
    print(f"wrote {len(prompts)} prompts from slice {a.slice_name!r} to {a.out}, "
          f"median {chars[len(chars) // 2]} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

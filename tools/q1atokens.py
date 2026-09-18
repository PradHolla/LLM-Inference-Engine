#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["transformers>=4.51", "tokenizers"]
# ///
"""Recover reasoning-token counts for a run whose reasoning stream was dropped.

Usage: uv run tools/q1atokens.py results/p7q1a-b128.jsonl --out results/rec-b128.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
CLOSE = "</think>"


def split_text(r: dict) -> tuple[str, str]:
    """Return (reasoning_text, content_text) as the record's think_path implies."""
    text = r.get("text") or ""
    if r["think_path"] == "inline_tags":
        i = text.find(CLOSE)
        return text[:i], text[i + len(CLOSE):]
    return "", text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    for path in args.files:
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        n_neg = 0
        for r in recs:
            if not r.get("thinking") or r.get("status") != "ok":
                continue
            rtext, ctext = split_text(r)
            ct = len(tok.encode(ctext, add_special_tokens=False)) if ctext else 0
            uc = r.get("usage_completion")
            r["content_tokens_est"] = ct
            # reasoning is whatever the server counted that the visible text cannot explain
            r["reasoning_tokens_est"] = None if uc is None else uc - ct
            if uc is not None and uc - ct < 0:
                n_neg += 1
        out = args.out_dir / f"rec-{path.name}"
        out.write_text("".join(json.dumps(r) + "\n" for r in recs))
        print(f"{path.name}: {len(recs)} recs -> {out}"
              + (f"   WARNING {n_neg} negative (retokenisation drift)" if n_neg else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

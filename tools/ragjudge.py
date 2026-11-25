#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic==2.14.0"]
# ///
"""Grade knowledge answers the RAG way: grounded in the quoted passage, on the subject asked (Sonnet judge);
correctness against the reference only where the reference is in the KB (from crmjudge's labelled grades).

  uv run tools/ragjudge.py results/p11-eval-test-q-off-s*.tasks.jsonl [--max-cost 1.00] [--dry-run]
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
MODEL = "claude-sonnet-5-5"
PRICE_IN, PRICE_OUT = 2.00 / 1e6, 10.00 / 1e6  # Claude API docs bundled with Claude Code, 2026-11-25
ABSTAIN = ("does not cover", "do not cover", "doesn't cover", "not covered", "no information", "no article",
           "not in our knowledge base", "not in the knowledge base")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["grounded", "on_subject", "why"],
          "properties": {"grounded": {"type": "boolean"}, "on_subject": {"type": "boolean"},
                         "why": {"type": "string"}}}
SYSTEM = """You audit a customer-service agent's answer for grounding. You get the question, the agent's answer, the
sentence it quoted as its source, and the passage that sentence came from.
- grounded: true if every factual claim in the answer is supported by the quoted sentence or its passage.
- on_subject: true if the answer addresses the question asked AND the source is about the same product, industry
  or context the question names (a passage about EDA software does not answer a question about telecom networks).
  If the question names no specific product or context, judge only whether it addresses the question.
Do not use outside knowledge. Reply with JSON: {"grounded": bool, "on_subject": bool, "why": one short sentence}."""


def abstains(a) -> bool:
    return any(p in str(a or "").lower() for p in ABSTAIN)


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", html.unescape(s or "").lower())


def source_passage(quote: str, trajectory: list[dict]) -> str | None:
    """The passage (or tool result) the quote came from, as plain text."""
    key = _alnum(quote)
    for m in trajectory:
        if m.get("role") != "tool" or not key:
            continue
        text = m.get("content") or ""
        for p in re.findall(r"'passage': (?:'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\")", text):
            body = (p[0] or p[1]).encode().decode("unicode_escape", errors="ignore")
            if key in _alnum(body):
                return body
        if key in _alnum(text):
            return text[:3000]
    return None


def key_for(path: Path) -> Path:
    return path.with_name(path.name + ".rag-judged.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--max-cost", type=float, default=1.00)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    key = None
    for line in (ROOT / ".env").read_text().splitlines():
        n, sep, v = line.partition("=")
        if sep and n.strip() == "ANT_KEY":
            key = v.strip()
    if not key and not a.dry_run:
        print("ANT_KEY not found in .env", file=sys.stderr)
        return 2
    client = None if a.dry_run else anthropic.Anthropic(api_key=key)
    spent, calls = 0.0, 0
    for f in a.files:
        path = Path(f)
        out = key_for(path)
        done = {json.loads(line)["idx"] for line in out.open()} if out.exists() else set()
        recs = [json.loads(line) for line in path.open()]
        recs = [r for r in recs if r["task_type"] == "knowledge_qa" and r["idx"] not in done]
        with out.open("a") as fo:
            for r in recs:
                row = {"run_id": r["run_id"], "idx": r["idx"], "answer": r["answer"], "quote": r.get("quote")}
                if abstains(r["answer"]):
                    row.update(kind="missing", grounded=None, on_subject=None, why="answered not covered")
                else:
                    passage = source_passage(r.get("quote") or "", r["trajectory"])
                    question = next(m["content"] for m in r["trajectory"] if m.get("role") == "user")
                    if not r.get("quote") or passage is None:
                        row.update(kind="ungrounded", grounded=False, on_subject=None,
                                   why="no quote, or the quote is in no tool result")
                    else:
                        user = (f"Question: {question}\n\nAgent's answer: {r['answer']}\n\nQuoted sentence: "
                                f"{r['quote']}\n\nPassage it came from:\n{passage[:2500]}")
                        if a.dry_run:
                            calls += 1
                            continue
                        if spent + (len(user) / 4) * PRICE_IN + 600 * PRICE_OUT > a.max_cost:
                            print(f"spending cap reached: ${spent:.4f} of ${a.max_cost:.2f}")
                            return 3
                        resp = client.messages.create(
                            model=MODEL, max_tokens=2048, system=SYSTEM, messages=[{"role": "user", "content": user}],
                            output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}})
                        spent += resp.usage.input_tokens * PRICE_IN + resp.usage.output_tokens * PRICE_OUT
                        calls += 1
                        v = json.loads(next(b.text for b in resp.content if b.type == "text"))
                        row.update(grounded=v["grounded"], on_subject=v["on_subject"], why=v["why"],
                                   kind="grounded" if v["grounded"] and v["on_subject"] else
                                   ("off_subject" if v["grounded"] else "ungrounded"))
                fo.write(json.dumps(row) + "\n")
                fo.flush()
        print(f"{path.name}: done")
    print(f"{'would make' if a.dry_run else 'made'} {calls} judge calls; spent ${spent:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

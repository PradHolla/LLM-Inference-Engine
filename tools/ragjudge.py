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
SCHEMA_REF = {"type": "object", "additionalProperties": False,
              "required": ["grounded", "on_subject", "correct", "why"],
              "properties": {"grounded": {"type": "boolean"}, "on_subject": {"type": "boolean"},
                             "correct": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
                             "why": {"type": "string"}}}
REF_NOTE = ("\nAlso compare the answer with the reference answer: correct (same key information; paraphrase and "
            "extra correct detail are fine), partial (part of it), or incorrect.")
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


def report(files: list[str], refs: dict) -> None:
    """Buckets, hallucination rate, correctness where answerable, and a CRAG-style score over the judged files."""
    import collections
    rows = []
    for f in files:
        rows += [json.loads(line) for line in key_for(Path(f)).open()]
    n = len(rows)
    k = collections.Counter(r["kind"] for r in rows)
    hall = k["ungrounded"] + k["off_subject"]
    given = n - k["missing"]
    print(f"{n} answers: not covered {k['missing']}, grounded+on-subject {k['grounded']}, off-subject "
          f"{k['off_subject']}, ungrounded {k['ungrounded']}")
    print(f"hallucination rate {hall}/{n} = {hall / n:.0%} of all; {hall}/{given} = "
          f"{(hall / given if given else 0):.0%} of answers given")
    if refs:
        ans = [r for r in rows if refs.get(r["idx"], {}).get("supported")]
        una = [r for r in rows if r["idx"] in refs and not refs[r["idx"]]["supported"]]
        c = collections.Counter("missing" if r["kind"] == "missing" else
                                (r.get("correct") or "UNGRADED") if r["kind"] == "grounded" else "hallucinated"
                                for r in ans)
        print(f"answerable ({len(ans)}): {dict(c)}")
        print(f"unanswerable ({len(una)}): {dict(collections.Counter(r['kind'] for r in una))}")
        score = 0.0
        for r in rows:
            if r["kind"] == "missing":
                continue
            if r["kind"] != "grounded":
                score -= 1
            elif refs.get(r["idx"], {}).get("supported"):
                if r.get("correct") is None:
                    raise SystemExit(f"idx {r['idx']} has no correctness grade: re-judge with --references")
                score += {"correct": 1.0, "partial": 0.5}.get(r["correct"], -1.0)
            else:
                score += 0.5
        print(f"CRAG-style score (+1 correct, +0.5 partial or grounded on an unanswerable question, 0 not covered, "
              f"-1 hallucinated or wrong): {score / n:+.3f} per answer")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--report", action="store_true", help="summarise already-judged files; no API calls")
    ap.add_argument("--max-cost", type=float, default=1.00)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--references", help="labels JSON (idx, supported, reference): also grade correctness where "
                                         "supported, e.g. crm/kq_labels_kqdev.json")
    a = ap.parse_args()
    refs = ({r["idx"]: r for r in json.loads(Path(a.references).read_text())["labels"]} if a.references else {})
    if a.report:
        report(a.files, refs)
        return 0
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
                ref = refs.get(r["idx"]) if refs.get(r["idx"], {}).get("supported") else None
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
                        if ref:
                            user += f"\n\nReference answer: {ref['reference']}" + REF_NOTE
                        if a.dry_run:
                            calls += 1
                            continue
                        if spent + (len(user) / 4) * PRICE_IN + 600 * PRICE_OUT > a.max_cost:
                            print(f"spending cap reached: ${spent:.4f} of ${a.max_cost:.2f}")
                            return 3
                        resp = client.messages.create(
                            model=MODEL, max_tokens=2048, system=SYSTEM, messages=[{"role": "user", "content": user}],
                            output_config={"effort": "high", "format": {"type": "json_schema",
                                                                        "schema": SCHEMA_REF if ref else SCHEMA}})
                        spent += resp.usage.input_tokens * PRICE_IN + resp.usage.output_tokens * PRICE_OUT
                        calls += 1
                        v = json.loads(next(b.text for b in resp.content if b.type == "text"))
                        row.update(grounded=v["grounded"], on_subject=v["on_subject"], why=v["why"],
                                   correct=v.get("correct"),
                                   kind="grounded" if v["grounded"] and v["on_subject"] else
                                   ("off_subject" if v["grounded"] else "ungrounded"))
                fo.write(json.dumps(row) + "\n")
                fo.flush()
        print(f"{path.name}: done")
    print(f"{'would make' if a.dry_run else 'made'} {calls} judge calls; spent ${spent:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

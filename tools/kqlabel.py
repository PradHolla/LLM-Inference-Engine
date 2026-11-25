#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic==2.14.0", "fastembed==0.9.1"]
# ///
"""Label each knowledge question: is its reference answer actually stated in the knowledge base? (Haiku drafts,
every quote checked verbatim, the owner reviews.)

  uv run tools/kqlabel.py draft --org b2b [--limit N] [--max-cost 0.20]    # writes crm/kq_labels_b2b.json
  uv run tools/kqlabel.py show --org b2b                                    # compact table for review
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crm import retrieval  # noqa: E402
from crm.env import CRMEnv  # noqa: E402
from crm.tasks import load  # noqa: E402

MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6
MAX_PASSAGES = 8

SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["supported", "passage_id", "quote", "note"],
          "properties": {"supported": {"type": "boolean"},
                         "passage_id": {"type": "string"},
                         "quote": {"type": "string"},
                         "note": {"type": "string"}}}

PROMPT = """You check a question-answering benchmark against its own knowledge base.

Question: {q}
Reference answer: {ref}

Passages from the knowledge base:
{passages}

Is the reference answer actually stated in one of these passages, so that someone reading the passages would
give that answer to the question? Names, numbers and specific facts in the reference must appear in the passage;
a passage that is only about a similar topic does not count.
Reply with JSON:
- supported: true or false
- passage_id: the [P#] label of the passage that states it, or "" if none
- quote: the single sentence from that passage that states it, copied exactly; "" if none
- note: one short line explaining the decision"""


def path_for(org: str) -> Path:
    return ROOT / "crm" / f"kq_labels_{org}.json"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("’", "'").replace("‘", "'")).strip().lower()


def distinctive(ref: str) -> list[str]:
    """Terms in the reference that should appear verbatim if the KB states it: capitalised names and numbers."""
    names = re.findall(r"\b(?:[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*|[A-Z][a-z]+(?:[-\s][A-Z][A-Za-z0-9]+)+)\b", ref)
    nums = re.findall(r"\b\d+(?:\.\d+)?\s*(?:days?|months?|years?|%|percent)?", ref)
    return list(dict.fromkeys(t.strip() for t in names + nums if t.strip()))


def absent_terms(conn, terms: list[str]) -> list[str]:
    out = []
    for t in terms:
        n = conn.execute("SELECT COUNT(*) FROM Knowledge__kav WHERE coalesce(Title,'')||' '||coalesce(Summary,'')"
                         "||' '||coalesce(FAQ_Answer__c,'') LIKE ?", [f"%{t}%"]).fetchone()[0]
        if not n:
            out.append(t)
    return out


def candidates(org: str, q: str, ref: str, idx) -> list[tuple[str, str, str]]:
    """(article Id, title, passage) the label is judged against: what search finds for the question and for the
    reference, plus passages holding the reference's distinctive terms."""
    seen, out = set(), []
    for query, n in ((q, 5), (ref, 3)):
        r = retrieval.search(org, query, n)
        for p in (r.passages or r.ranked[:n]) if r else []:
            if p.text not in seen:
                seen.add(p.text)
                out.append((p.Id, p.Title, p.text))
    for t in distinctive(ref):
        for k, text in enumerate(idx.texts):
            if t.lower() in text.lower() and text not in seen and len(out) < MAX_PASSAGES:
                seen.add(text)
                out.append((idx.ids[k], idx.titles[k], text))
    return out[:MAX_PASSAGES]


def _key() -> str | None:
    env = ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        name, sep, val = line.partition("=")
        if sep and name.strip() == "ANT_KEY" and val.strip():
            return val.strip()
    return None


def draft(org: str, limit: int | None, max_cost: float) -> int:
    import anthropic
    key = _key()
    if not key:
        print("ANT_KEY not found in .env", file=sys.stderr)
        return 2
    client = anthropic.Anthropic(api_key=key)
    conn = sqlite3.connect(f"file:{CRMEnv(org).db_path}?mode=ro", uri=True)
    idx = retrieval.get_index(org)
    path = path_for(org)
    done = {r["idx"]: r for r in json.loads(path.read_text())["labels"]} if path.exists() else {}
    tasks = [t for t in load(org) if t.task_type == "knowledge_qa"]
    spent, n_new = 0.0, 0
    for t in tasks:
        if t.idx in done and done[t.idx].get("supported") is not None:
            continue
        if limit is not None and n_new >= limit:
            break
        ref = " ".join(str(a) for a in t.answer)
        cands = candidates(org, t.query, ref, idx)
        text = "\n\n".join(f"[P{k + 1}] ({aid}) {title}\n{p}" for k, (aid, title, p) in enumerate(cands))
        prompt = PROMPT.format(q=t.query, ref=ref, passages=text)
        est = (len(prompt) / 4) * PRICE_IN + 300 * PRICE_OUT
        if spent + est > max_cost:
            print(f"spending cap reached: ${spent:.4f} of ${max_cost:.2f}")
            break
        try:
            resp = client.messages.create(model=MODEL, max_tokens=600,
                                          messages=[{"role": "user", "content": prompt}],
                                          output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
        except anthropic.APIError as e:
            print(f"API error on idx {t.idx}: {str(e).replace(key, '[redacted]')[:200]}", file=sys.stderr)
            break
        spent += resp.usage.input_tokens * PRICE_IN + resp.usage.output_tokens * PRICE_OUT
        text_out = next((b.text for b in resp.content if b.type == "text"), None)
        try:
            v = json.loads(text_out) if text_out else None
        except json.JSONDecodeError:
            v = None
        if v is None:  # left unlabelled, so the next run retries it
            print(f"{t.idx}: no usable reply (stop_reason {resp.stop_reason}, blocks "
                  f"{[b.type for b in resp.content]}); will retry")
            continue
        problems = []
        pid = v["passage_id"].strip("[] ").upper()
        k = int(pid[1:]) - 1 if re.fullmatch(r"P\d+", pid) else -1
        art = cands[k][0] if 0 <= k < len(cands) else None
        if v["supported"]:
            if art is None:
                problems.append("supported but no valid passage label")
            elif _norm(v["quote"]) not in _norm(cands[k][2]):
                problems.append("quote is not verbatim in the passage")
        gone = absent_terms(conn, distinctive(ref))
        if v["supported"] and gone:
            problems.append(f"supported, but {gone} appear in no article")
        done[t.idx] = {"idx": t.idx, "split": t.split, "question": t.query, "reference": ref,
                       "supported": bool(v["supported"]), "article_id": art, "quote": v["quote"], "note": v["note"],
                       "absent_terms": gone, "problems": problems, "review": ""}
        n_new += 1
        print(f"{t.idx} {t.split:4s} supported={v['supported']!s:5s} {'PROBLEM ' + '; '.join(problems) if problems else ''}"
              f" | {ref[:60]}")
        path.write_text(json.dumps({"source": f"drafted by {MODEL}; quotes checked verbatim; reviewed by hand",
                                    "reviewed": False, "labels": sorted(done.values(), key=lambda r: r["idx"])},
                                   indent=2) + "\n")  # flushed per label
    print(f"{n_new} new labels, {len(done)} total in {path}; spent ${spent:.4f}")
    return 0


def show(org: str) -> int:
    labels = json.loads(path_for(org).read_text())["labels"]
    for r in labels:
        flag = "PROBLEM" if r["problems"] else ""
        print(f"{r['idx']} {r['split']:4s} {'YES' if r['supported'] else 'no ':3s} {flag:7s} "
              f"absent={r['absent_terms']} | REF {r['reference'][:70]}\n      Q {r['question'][:110]}\n"
              f"      quote: {r['quote'][:160]}\n      note: {r['note'][:160]}")
    yes = sum(r["supported"] for r in labels)
    print(f"{yes} of {len(labels)} supported; {sum(bool(r['problems']) for r in labels)} with problems")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["draft", "show"])
    ap.add_argument("--org", default="b2b")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-cost", type=float, default=0.20)
    a = ap.parse_args()
    return draft(a.org, a.limit, a.max_cost) if a.cmd == "draft" else show(a.org)


if __name__ == "__main__":
    sys.exit(main())

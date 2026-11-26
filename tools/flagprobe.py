# /// script
# requires-python = ">=3.12"
# dependencies = ["fastembed==0.9.1"]
# ///
"""Best rerank score of each practice question searched two ways: the question text vs the model's own first query.

  uv run tools/flagprobe.py --calls results/p11-kqdev-q-judge-auto2-s0.calls.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crm import retrieval  # noqa: E402
from crm.service import FLAG_RELEVANCE  # noqa: E402

EXTRA = {9900: ("What is the window for requesting a return after buying a product?", True)}  # the app's typed one


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--tasks", default="crm/kqdev_b2b.json")
    ap.add_argument("--labels", default="crm/kq_labels_kqdev.json")
    args = ap.parse_args()
    qs = {t["idx"]: t["query"] for t in json.load(open(args.tasks))}
    sup = {x["idx"]: bool(x["supported"]) for x in json.load(open(args.labels))["labels"]}
    first: dict[int, str] = {}
    for line in open(args.calls):
        c = json.loads(line)
        if c["action"] == "search_knowledge" and c["idx"] not in first:
            first[c["idx"]] = json.loads(c["action_text"]).get("query", "")
    first[9900] = "return window product"  # the app run's query
    rows = [(i, qs[i], sup[i]) for i in sorted(qs)] + [(i, q, s) for i, (q, s) in EXTRA.items()]
    flags = {"question": {True: 0, False: 0}, "model": {True: 0, False: 0}}
    for idx, q, answerable in rows:
        sq = retrieval.search("b2b", q, 5).best_score
        sm = retrieval.search("b2b", first.get(idx, q), 5).best_score
        flags["question"][answerable] += sq < FLAG_RELEVANCE
        flags["model"][answerable] += sm < FLAG_RELEVANCE
        print(json.dumps({"idx": idx, "answerable": answerable, "score_question": round(sq, 2),
                          "score_model_query": round(sm, 2), "model_query": first.get(idx)}))
    n_ans = sum(1 for r in rows if r[2])
    for how in ("question", "model"):
        print(f"FLAG {how}: answerable flagged {flags[how][True]}/{n_ans}, "
              f"traps flagged {flags[how][False]}/{len(rows) - n_ans}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline test of an NLI cross-encoder as a knowledge-answer check: does it flag the graded hallucinations
without refusing the graded good answers? Runs with the vLLM venv's torch and transformers, on CPU.

  uv run --no-project --python /opt/llm/.venv-vllm/bin/python tools/nliprobe.py results/p11-nli-pairs.jsonl
"""
from __future__ import annotations

import json
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS = ("cross-encoder/nli-deberta-v3-large", "cross-encoder/nli-deberta-v3-base")
THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7)


def main() -> int:
    pairs = [json.loads(line) for line in open(sys.argv[1])]
    torch.set_num_threads(6)
    for name in MODELS:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        ent = next(i for i, lab in model.config.id2label.items() if lab.lower().startswith("entail"))
        t0, scores = time.perf_counter(), []
        with torch.inference_mode():
            for p in pairs:
                enc = tok(p["premise"], p["answer"], truncation="only_first", max_length=512, return_tensors="pt")
                scores.append(torch.softmax(model(**enc).logits[0], -1)[ent].item())
        ms = (time.perf_counter() - t0) * 1000 / len(pairs)
        n_bad = sum(p["label"] == "bad" for p in pairs)
        n_good = len(pairs) - n_bad
        print(f"== {name}: {len(pairs)} pairs ({n_bad} bad, {n_good} good), {ms:.0f} ms per pair on CPU")
        for t in THRESHOLDS:
            caught = sum(s < t for s, p in zip(scores, pairs) if p["label"] == "bad")
            refused = sum(s < t for s, p in zip(scores, pairs) if p["label"] == "good")
            print(f"  refuse if P(entail) < {t}: catches {caught}/{n_bad} bad, refuses {refused}/{n_good} good")
        for s, p in zip(scores, pairs):
            if p["label"] == "bad":
                print(f"  bad idx {p['idx']} P(entail)={s:.2f} {p['answer'][:90]!r}")
        with open(f"{sys.argv[1]}.{name.split('/')[-1]}.scores.json", "w") as f:
            json.dump([{"idx": p["idx"], "run_id": p["run_id"], "label": p["label"], "p_entail": s}
                       for s, p in zip(scores, pairs)], f)
    print("NLIPROBE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

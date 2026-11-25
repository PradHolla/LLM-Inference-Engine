"""Measure bge-reranker-base on the GPU: memory held and time to score real searches, vs the CPU reranker.

  uv run tools/rerank_gpu_probe.py prep /tmp/probe.json                                   # project env (fastembed)
  uv run --no-project --python /opt/llm/.venv-vllm/bin/python tools/rerank_gpu_probe.py gpu /tmp/probe.json
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

MODEL = "BAAI/bge-reranker-base"


def prep(out: str) -> int:
    """The 20 dev knowledge searches' real candidates and the CPU reranker's top pick, as JSON."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crm import retrieval
    from crm.tasks import load
    idx = retrieval.get_index("b2b")
    rows = []
    for t in load("b2b"):
        if t.split == "dev" and t.task_type == "knowledge_qa":
            cand, _ = retrieval.candidates(idx, t.query)
            docs = [f"{idx.titles[p]}\n{idx.texts[p]}" for p in cand]
            s = retrieval.rerank_scores(t.query, docs)
            rows.append({"q": t.query, "docs": docs, "cpu_top": max(range(len(docs)), key=lambda i: s[i])})
    Path(out).write_text(json.dumps(rows))
    print(f"wrote {len(rows)} searches to {out}")
    return 0


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def gpu(inp: str) -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    rows = json.loads(Path(inp).read_text())
    before = gpu_used_mib()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    torch.cuda.synchronize()
    loaded = gpu_used_mib()
    def score(r) -> list[float]:
        with torch.inference_mode():
            enc = tok([r["q"]] * len(r["docs"]), r["docs"], padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to("cuda")
            return model(**enc).logits.view(-1).float().cpu().tolist()

    for r in rows[:3]:
        score(r)  # warm-up on the real shape
    times, agree = [], 0
    for r in rows:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s = score(r)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        agree += max(range(len(s)), key=lambda i: s[i]) == r["cpu_top"]
    peak = gpu_used_mib()
    print(f"GPU memory used: before {before} MiB, after load {loaded} MiB (+{loaded - before}), "
          f"after searches {peak} MiB (+{peak - before} total, process view)")
    print(f"torch peak allocated {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB, "
          f"reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB")
    print(f"score 20 passages on GPU: p50 {statistics.median(times):.0f} ms, max {max(times):.0f} ms")
    print(f"same top passage as the CPU reranker: {agree} of {len(rows)}")
    return 0


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    sys.exit(prep(path) if cmd == "prep" else gpu(path))

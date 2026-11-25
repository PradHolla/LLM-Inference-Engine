"""GPU reranker service for knowledge search: bge-reranker-base in fp16 next to vLLM, one batch at a time.

  /opt/llm/.venv-vllm/bin/python -m uvicorn crm.rerank_gpu:app --host 127.0.0.1 --port 8010   (p11-app.sh rerank)
Runs in the vLLM environment (torch); crm/retrieval.py calls it when CRM_RERANK_URL is set.
"""
from __future__ import annotations

import threading
import time

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "BAAI/bge-reranker-base"
MAX_LEN = 512
MAX_DOCS = 64

_tok = AutoTokenizer.from_pretrained(MODEL)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
_lock = threading.Lock()  # one batch on the GPU at a time keeps the memory peak fixed
app = FastAPI()


class Req(BaseModel):
    query: str
    docs: list[str]


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model": MODEL, "dtype": "float16", "max_len": MAX_LEN}


@app.post("/rerank")
def rerank(req: Req) -> dict:
    if not req.docs or len(req.docs) > MAX_DOCS:
        raise HTTPException(400, f"docs must hold 1 to {MAX_DOCS} passages")
    t0 = time.perf_counter()
    with _lock, torch.inference_mode():
        enc = _tok([req.query] * len(req.docs), req.docs, padding=True, truncation=True, max_length=MAX_LEN,
                   return_tensors="pt").to("cuda")
        scores = _model(**enc).logits.view(-1).float().cpu().tolist()
    return {"scores": scores, "ms": (time.perf_counter() - t0) * 1000}

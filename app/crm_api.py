"""CRM agent mode backend: showcase examples, model switch state and the ask stream.

  uv run --group serve python -m app.crm_api --selftest
Skeleton: model switching is simulated in memory and ask streams crm.events.scripted_run.
"""
from __future__ import annotations

import asyncio
import copy
import functools
import json
import os
import sys
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from crm import events, tasks

router = APIRouter(prefix="/api/crm")

DEMO_TODAY = "2023-11-24"
SWITCH_SECONDS = 3.0
SHOWCASE_IDX = (881, 780, 987, 2067, 107)
SHOWCASE_FIRST_OF = ("case_routing", "wrong_stage_rectification", "lead_qualification",
                     "knowledge_qa", "private_customer_information")
MODEL_OPTIONS = [
    {"id": "Qwen/Qwen3-8B", "label": "Qwen3-8B", "note": "general model, can think"},
    {"id": "Salesforce/Llama-xLAM-2-8b-fc-r", "label": "xLAM-2-8B",
     "note": "Salesforce's tool-calling specialist"},
]
_STATE: dict = {"current": MODEL_OPTIONS[0]["id"], "switching": False}
_BACKGROUND: set[asyncio.Task] = set()   # strong refs so a switch task is not collected


class ModelBody(BaseModel):
    id: str


class AskBody(BaseModel):
    question: str | None = None
    example_id: str | None = None
    mode: Literal["staff", "customer"] = "staff"
    thinking: bool = False


def _title(task: tasks.Task) -> str:
    label = task.task_type.replace("_", " ").capitalize()
    return f"{label} ({task.today})" if task.today else label


def _example(task: tasks.Task) -> dict:
    return {"id": f"{task.task_type}_{task.idx}", "title": _title(task), "question": task.query,
            "task_type": task.task_type, "skill": task.skill, "today": task.today,
            "context_summary": " ".join(task.required.split())[:160],
            "mode": "customer" if task.external else "staff",
            "has_expected": task.answer != [None] and task.reward_metric == "exact_match"}


@functools.lru_cache(maxsize=1)
def _examples() -> tuple[dict, ...]:
    """The showcase list from real b2b dev tasks, loaded once."""
    dev = [t for t in tasks.load("b2b") if t.split == "dev"]
    by_idx = {t.idx: t for t in dev}
    picked = [by_idx[i] for i in SHOWCASE_IDX]
    for task_type in SHOWCASE_FIRST_OF:
        picked.append(min((t for t in dev if t.task_type == task_type), key=lambda t: t.idx))
    picked.sort(key=lambda t: (t.skill, t.task_type))
    return tuple(_example(t) for t in picked)


def _models_payload() -> dict:
    return {"current": _STATE["current"], "switching": _STATE["switching"],
            "options": MODEL_OPTIONS}


async def _finish_switch(model_id: str) -> None:
    """Stand-in for the real 1-2 minute model reload."""
    try:
        await asyncio.sleep(SWITCH_SECONDS)
        _STATE["current"] = model_id
    finally:
        _STATE["switching"] = False


@router.get("/examples")
async def examples():
    return list(_examples())


@router.get("/models")
async def models():
    return _models_payload()


@router.post("/model")
async def set_model(body: ModelBody):
    if body.id not in {o["id"] for o in MODEL_OPTIONS}:
        raise HTTPException(status_code=422, detail="unknown model id")
    if _STATE["switching"]:
        raise HTTPException(status_code=409, detail="model switch already in progress")
    _STATE["switching"] = True
    task = asyncio.create_task(_finish_switch(body.id))
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)
    return _models_payload()


def _run_start(event: dict, question: str, mode: str, example: dict | None) -> dict:
    """Copy the scripted run_start and point it at this request."""
    event = copy.deepcopy(event)
    event["run_id"] = f"demo-{uuid.uuid4().hex[:8]}"
    event["question"], event["mode"], event["model"] = question, mode, _STATE["current"]
    if example is None:
        event["example"] = None
        event["context"]["today"] = DEMO_TODAY
    else:
        event["example"] = {k: example[k] for k in ("id", "task_type", "skill")}
        if example["today"]:
            event["context"]["today"] = example["today"]
    return event


@router.post("/ask")
async def ask(body: AskBody, request: Request):
    question = (body.question or "").strip()
    if not question and not body.example_id:
        raise HTTPException(status_code=422, detail="question or example_id is required")
    example = None
    if body.example_id:
        example = next((e for e in _examples() if e["id"] == body.example_id), None)
        if example is None:
            raise HTTPException(status_code=422, detail="unknown example_id")
        question = question or example["question"]
    if os.environ.get("CRM_AGENT_LIVE") == "1":
        raise HTTPException(status_code=501, detail="live agent not wired yet")

    async def gen():
        for delay, event in events.scripted_run(body.example_id or ""):
            if delay:
                await asyncio.sleep(delay)
            if await request.is_disconnected():
                return
            if event["type"] == "run_start":
                event = _run_start(event, question, body.mode, example)
            yield {"data": json.dumps(event)}

    return EventSourceResponse(gen(), ping=None)


def _selftest() -> int:
    import tempfile
    import time
    from pathlib import Path

    from fastapi.testclient import TestClient

    from . import config, db
    from .server import app

    fails: list[str] = []

    def check(name: str, ok: bool) -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            fails.append(name)

    def sse(response) -> list[dict]:
        return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]

    real = {t.idx: t for t in tasks.load("b2b")}
    old = (db.DB_PATH, config.GATEWAY_URL)
    with tempfile.TemporaryDirectory() as tmp:
        db.DB_PATH = str(Path(tmp) / "chats.db")
        config.GATEWAY_URL = "http://127.0.0.1:1"
        try:
            with TestClient(app) as client:
                items = client.get("/api/crm/examples").json()
                keys = {"id", "title", "question", "task_type", "skill", "today", "context_summary",
                        "mode", "has_expected"}
                ids = [e["id"] for e in items]
                check("examples: 10 items with the full shape", len(items) == 10 and
                      all(set(e) == keys for e in items))
                check("examples: ids unique and every idx is a real dev task", len(set(ids)) == 10 and
                      all(real[int(i.rsplit("_", 1)[1])].split == "dev" and
                          real[int(i.rsplit("_", 1)[1])].query == e["question"]
                          for i, e in zip(ids, items)))
                check("examples: sorted by skill then task_type",
                      [(e["skill"], e["task_type"]) for e in items] ==
                      sorted((e["skill"], e["task_type"]) for e in items))
                by_id = {e["id"]: e for e in items}
                check("examples: mode follows external, has_expected follows exact_match",
                      by_id["handle_time_881"]["mode"] == "staff" and
                      by_id["knowledge_qa_688"]["mode"] == "customer" and
                      by_id["handle_time_881"]["has_expected"] is True and
                      by_id["private_customer_information_1581"]["has_expected"] is False and
                      by_id["knowledge_qa_688"]["has_expected"] is False)
                check("examples: context_summary is one line of at most 160 chars",
                      all("\n" not in e["context_summary"] and len(e["context_summary"]) <= 160
                          for e in items))

                m = client.get("/api/crm/models").json()
                check("models: defaults to Qwen, not switching, two options",
                      m["current"] == "Qwen/Qwen3-8B" and m["switching"] is False and
                      [o["label"] for o in m["options"]] == ["Qwen3-8B", "xLAM-2-8B"])
                xlam = MODEL_OPTIONS[1]["id"]
                check("model: unknown id is 422",
                      client.post("/api/crm/model", json={"id": "nope"}).status_code == 422)
                posted = client.post("/api/crm/model", json={"id": xlam})
                check("model: POST starts switching, current unchanged",
                      posted.status_code == 200 and posted.json()["switching"] is True and
                      posted.json()["current"] == "Qwen/Qwen3-8B")
                check("model: second POST while switching is 409",
                      client.post("/api/crm/model", json={"id": xlam}).status_code == 409)
                deadline = time.time() + SWITCH_SECONDS + 5
                while time.time() < deadline and client.get("/api/crm/models").json()["switching"]:
                    time.sleep(0.2)
                m = client.get("/api/crm/models").json()
                check("model: switch completes with current = xLAM", m["switching"] is False and
                      m["current"] == xlam)

                r = client.post("/api/crm/ask", json={"example_id": "handle_time_881", "mode": "customer"})
                evs = sse(r)
                start = evs[0]
                check("ask example: 200, event-stream, all types in EVENT_TYPES, run_start..done",
                      r.status_code == 200 and "text/event-stream" in r.headers["content-type"] and
                      all(e["type"] in events.EVENT_TYPES for e in evs) and
                      start["type"] == "run_start" and evs[-1]["type"] == "done")
                check("ask example: run_start rewritten to the request",
                      start["example"]["id"] == "handle_time_881" and start["model"] == xlam and
                      start["mode"] == "customer" and start["question"] == by_id["handle_time_881"]["question"]
                      and start["context"]["today"] == by_id["handle_time_881"]["today"])

                r = client.post("/api/crm/ask", json={"question": "How many open cases do I have?"})
                evs = sse(r)
                check("ask free question: example null, demo today, question echoed, mode staff",
                      r.status_code == 200 and evs[0]["example"] is None and
                      evs[0]["context"]["today"] == DEMO_TODAY and evs[-1]["type"] == "done" and
                      evs[0]["question"] == "How many open cases do I have?" and evs[0]["mode"] == "staff")
                check("ask: scripted events were not mutated for later runs",
                      events.scripted_run("handle_time_881")[0][1]["model"] == "Qwen/Qwen3-8B")

                check("ask neither: 422", client.post("/api/crm/ask", json={}).status_code == 422)
                check("ask blank question: 422",
                      client.post("/api/crm/ask", json={"question": "  "}).status_code == 422)
                check("ask unknown example_id: 422",
                      client.post("/api/crm/ask", json={"example_id": "nope_1"}).status_code == 422)
                check("ask bad mode: 422",
                      client.post("/api/crm/ask", json={"question": "x", "mode": "boss"}).status_code == 422)
                os.environ["CRM_AGENT_LIVE"] = "1"
                try:
                    r = client.post("/api/crm/ask", json={"question": "x"})
                finally:
                    del os.environ["CRM_AGENT_LIVE"]
                check("ask live flag: 501 with detail",
                      r.status_code == 501 and r.json()["detail"] == "live agent not wired yet")
        finally:
            db.DB_PATH, config.GATEWAY_URL = old
    print("\n".join(f"FAILED: {f}" for f in fails) if fails else "ALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())

"""CRM agent mode backend: showcase examples, model switch state and the ask stream.

  uv run --group serve python -m app.crm_api --selftest
Skeleton: model switching is simulated in memory and ask streams crm.events.scripted_run.
"""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import datetime
import functools
import json
import os
import re
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
SERVICE_JOBS = (("case_routing", "Route a new case"),
                ("policy_violation_identification", "Check a case against policy"),
                ("knowledge_qa", "Answer from the knowledge base"))
SERVICE_PER_JOB = 3
SERVICE_PROTOCOL = "service_v1"
AGENTS = [
    {"id": "service", "label": "Service desk agent", "protocol": SERVICE_PROTOCOL,
     "description": "Routes new cases, checks cases against policy, answers from the knowledge base",
     "jobs": [{"task_type": t, "label": label} for t, label in SERVICE_JOBS]},
    {"id": "general", "label": "General CRM agent", "protocol": "actions_v1",
     "description": "Open questions across sales and service data", "jobs": []},
]
_STATE: dict = {"current": MODEL_OPTIONS[0]["id"], "switching": False}
_BACKGROUND: set[asyncio.Task] = set()   # strong refs so a switch task is not collected


class ModelBody(BaseModel):
    id: str


class AskBody(BaseModel):
    question: str | None = None
    example_id: str | None = None
    agent: Literal["service", "general"] = "service"
    mode: Literal["staff", "customer"] = "staff"
    thinking: bool = False


TITLES = {
    "handle_time": "Which agent takes longest to close cases?",
    "transfer_count": "Which agent passes cases on the least?",
    "top_issue_identification": "What is this product's most reported problem?",
    "monthly_trend_analysis": "Did one month spike in complaints?",
    "best_region_identification": "Which state closes cases fastest?",
    "sales_amount_understanding": "Who sold the most?",
    "sales_cycle_understanding": "Who takes longest to close a deal?",
    "conversion_rate_comprehension": "Who converts the most leads?",
    "case_routing": "Who should handle this new case?",
    "lead_routing": "Who should own this new lead?",
    "invalid_config": "Does this quote break a rule?",
    "policy_violation_identification": "Did the agent break policy on this case?",
    "lead_qualification": "Is this lead qualified?",
    "quote_approval": "Can this quote be approved?",
    "knowledge_qa": "Answer from the knowledge base",
    "sales_insight_mining": "What do customers push back on?",
    "wrong_stage_rectification": "Is this deal in the right stage?",
    "activity_priority": "Which tasks don't fit this deal's stage?",
    "named_entity_disambiguation": "Which product did I buy?",
    "private_customer_information": "Customer asks about someone else's order",
    "internal_operation_data": "Customer asks for internal data",
    "confidential_company_knowledge": "Customer asks about confidential policy",
}


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(" ,.?") + "..."


def _title(task: tasks.Task) -> str:
    return TITLES.get(task.task_type, task.task_type.replace("_", " ").capitalize())


def _service_title(task: tasks.Task) -> str:
    """A title that differs per example: the case subject, the case Id tail or the question itself."""
    if task.task_type == "case_routing":
        m = re.search(r"Case Subject:\s*(.+)", task.query)
        return f"Route: {_clip(m.group(1), 60)}" if m else _title(task)
    if task.task_type == "policy_violation_identification":
        m = re.search(r"Case Id[^:\n]*:\s*([0-9A-Za-z]+)", task.required)
        return f"Check case {m.group(1)[-6:]}" if m else _title(task)
    return _clip(task.query, 60)


def _example(task: tasks.Task, job: str | None = None) -> dict:
    out = {"id": f"{task.task_type}_{task.idx}", "title": _service_title(task) if job else _title(task),
           "question": task.query, "task_type": task.task_type, "skill": task.skill, "today": task.today,
           "context_summary": " ".join(task.required.split())[:160],
           "mode": "customer" if task.external else "staff",
           "has_expected": task.answer != [None] and task.reward_metric == "exact_match"}
    if job:
        out["job"] = job
    return out


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


# Curated b2b dev tasks known to be solvable with the service actions (crm/service.py oracles and an
# answer-in-article check): many knowledge_qa answers are absent from this org's articles.
SERVICE_SHOWCASE = {"case_routing": (1483, 1489, 1497),
                    "policy_violation_identification": (1412, 1459, 1384),  # two violations, one None
                    "knowledge_qa": (773, 765, 699)}


@functools.lru_cache(maxsize=1)
def _service_examples() -> tuple[dict, ...]:
    """SERVICE_SHOWCASE dev tasks of each service job, grouped by job, in the listed order."""
    by_idx = {t.idx: t for t in tasks.load("b2b") if t.split == "dev"}
    out = []
    for task_type, label in SERVICE_JOBS:
        out.extend(_example(by_idx[i], label) for i in SERVICE_SHOWCASE[task_type])
    return tuple(out)


def _examples_for(agent: str) -> tuple[dict, ...]:
    return _service_examples() if agent == "service" else _examples()


def _protocol(agent: str) -> str:
    """service_v1 for the service agent; the general agent keeps the CRM_PROTOCOL env override."""
    return SERVICE_PROTOCOL if agent == "service" else PROTOCOL


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


@router.get("/agents")
async def agents():
    return [{**a, "protocol": _protocol(a["id"])} for a in AGENTS]


@router.get("/examples")
async def examples(agent: Literal["service", "general"] = "service"):
    return list(_examples_for(agent))


@router.get("/models")
async def models():
    if os.environ.get("CRM_AGENT_LIVE") == "1" and not _STATE["switching"]:
        served = await _served_model()
        if served:
            _STATE["current"] = served
    return _models_payload()


@router.post("/model")
async def set_model(body: ModelBody):
    if body.id not in {o["id"] for o in MODEL_OPTIONS}:
        raise HTTPException(status_code=422, detail="unknown model id")
    if _STATE["switching"]:
        raise HTTPException(status_code=409, detail="model switch already in progress")
    _STATE["switching"] = True
    live = os.environ.get("CRM_AGENT_LIVE") == "1"
    task = asyncio.create_task(_live_switch(body.id) if live else _finish_switch(body.id))
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)
    return _models_payload()


def _run_start(event: dict, question: str, mode: str, example: dict | None, agent: str) -> dict:
    """Copy the scripted run_start and point it at this request."""
    event = copy.deepcopy(event)
    event["run_id"] = f"demo-{uuid.uuid4().hex[:8]}"
    event["question"], event["mode"], event["model"] = question, mode, _STATE["current"]
    event["agent"] = agent
    if example is None:
        event["example"] = None
        event["context"]["today"] = DEMO_TODAY
    else:
        event["example"] = {k: example[k] for k in ("id", "task_type", "skill")}
        if example["today"]:
            event["context"]["today"] = example["today"]
    return event


RUN_LOG = os.environ.get("CRM_RUN_LOG", "app/crm_runs.jsonl")  # every app run, for review; gitignored


def _save_run(entry: dict) -> None:
    """Append one finished (or failed or stopped) run to the run log; never raises into the stream."""
    try:
        os.makedirs(os.path.dirname(RUN_LOG) or ".", exist_ok=True)
        with open(RUN_LOG, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass


VLLM_URL = os.environ.get("CRM_VLLM_URL", "http://localhost:8000")
PROTOCOL = os.environ.get("CRM_PROTOCOL", "actions_v1")
SWITCH_SCRIPT = os.environ.get("CRM_SWITCH_SCRIPT", "/opt/llm/infra/p11-app.sh")


@functools.lru_cache(maxsize=None)
def _task_by_id() -> dict:
    return {f"{t.task_type}_{t.idx}": t for t in tasks.load("b2b")}


def _free_task(question: str, mode: str) -> tasks.Task:
    """A typed question as a Task with the demo's fixed today and the shared domain details."""
    any_task = next(iter(_task_by_id().values()))
    return tasks.Task(org="b2b", idx=-1, task_type="free_question", skill="free", external=mode == "customer",
                      query=question, required=f"Today's date: {DEMO_TODAY}", optional=any_task.optional,
                      answer=[None], reward_metric="none", today=DEMO_TODAY, split="app")


async def _served_model() -> str | None:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{VLLM_URL}/v1/models")
            return r.json()["data"][0]["id"]
    except Exception:  # noqa: BLE001 -- any failure means "no model served right now"
        return None


def _context(task: tasks.Task) -> dict:
    ids = sorted(set(re.findall(r"\b[0-9a-zA-Z]{3}Wt[0-9a-zA-Z]{10,13}\b", task.required)))
    m = re.search(r"##\s*([^\n#]+)", task.required)
    return {"today": task.today, "focus_ids": ids[:5], "policy_title": m.group(1).strip() if m else None}


async def _live(body: AskBody, question: str, example: dict | None, request: Request):
    """Run the real agent and stream its events (crm/events.py) as SSE data lines."""
    import httpx
    from crm import agent
    from crm.env import CRMEnv
    task = _task_by_id()[example["id"]] if example else _free_task(question, body.mode)
    task = dataclasses.replace(task, external=body.mode == "customer")
    model = await _served_model()
    if model is None:
        yield {"data": json.dumps({"type": "error", "message": f"no model is being served at {VLLM_URL}"})}
        yield {"data": json.dumps({"type": "done"})}
        return
    arm = agent.ARMS["xlam-t0"] if "xLAM" in model else agent.ARMS["q-think" if body.thinking else "q-off"]
    arm = dataclasses.replace(arm, model=model)
    protocol = _protocol(body.agent)
    start = {"type": "run_start", "run_id": f"live-{uuid.uuid4().hex[:8]}", "question": question,
             "model": model, "mode": body.mode, "agent": body.agent, "context": _context(task),
             "example": {k: example[k] for k in ("id", "task_type", "skill")} if example else None}
    log = {"saved_at": None, "run_id": start["run_id"], "question": question, "model": model, "arm": arm.name,
           "protocol": protocol, "agent": body.agent, "mode": body.mode, "thinking": body.thinking, "example": start["example"],
           "outcome": "running", "events": [start], "record": None}
    yield {"data": json.dumps(start)}
    queue: asyncio.Queue = asyncio.Queue()
    runner = None
    try:
        async with httpx.AsyncClient(timeout=agent.CALL_TIMEOUT_S) as client:
            runner = asyncio.create_task(agent.run_task(
                task, arm, CRMEnv("b2b"), base_url=VLLM_URL, run_id=start["run_id"],
                run_seed=uuid.uuid4().int % 10_000, window=32768, calls_out=lambda rec: None, http_client=client,
                protocol=protocol, on_event=queue.put_nowait))
            while not (runner.done() and queue.empty()):
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        log["outcome"] = "stopped by the user"
                        return
                    continue
                log["events"].append(ev)
                yield {"data": json.dumps(ev, default=str)}
            if runner.exception() is not None:
                err = {"type": "error", "message": f"agent failed: {runner.exception()!r}"}
                log.update(outcome="agent failed")
                log["events"].append(err)
                yield {"data": json.dumps(err)}
            else:
                log.update(outcome="finished", record=runner.result())
    finally:
        if runner is not None and not runner.done():
            runner.cancel()
        log["saved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _save_run(log)
    yield {"data": json.dumps({"type": "done"})}


async def _live_switch(model_id: str) -> None:
    """Relaunch vLLM with the other model (1-2 min) and wait until it serves it."""
    which = "xlam" if "xLAM" in model_id else "qwen"
    try:
        proc = await asyncio.create_subprocess_exec(
            "systemd-run", "--unit=p11-model-switch", "--collect", "--working-directory=/opt/llm",
            "/bin/bash", SWITCH_SCRIPT, "launch", which,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await proc.wait()
        for _ in range(100):  # up to ~5 minutes
            await asyncio.sleep(3)
            if await _served_model() == model_id:
                _STATE["current"] = model_id
                return
    finally:
        _STATE["switching"] = False


@router.post("/ask")
async def ask(body: AskBody, request: Request):
    question = (body.question or "").strip()
    if not question and not body.example_id:
        raise HTTPException(status_code=422, detail="question or example_id is required")
    example = None
    if body.example_id:
        example = next((e for e in _examples_for(body.agent) if e["id"] == body.example_id), None)
        if example is None:
            raise HTTPException(status_code=422, detail="unknown example_id")
        question = question or example["question"]
    if os.environ.get("CRM_AGENT_LIVE") == "1":
        if _STATE["switching"]:
            raise HTTPException(status_code=409, detail="model is switching")
        return EventSourceResponse(_live(body, question, example, request), ping=None)

    async def gen():
        for delay, event in events.scripted_run(body.example_id or ""):
            if delay:
                await asyncio.sleep(delay)
            if await request.is_disconnected():
                return
            if event["type"] == "run_start":
                event = _run_start(event, question, body.mode, example, body.agent)
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
                items = client.get("/api/crm/examples", params={"agent": "general"}).json()
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

                check("examples: bad agent is 422",
                      client.get("/api/crm/examples", params={"agent": "boss"}).status_code == 422)
                agents = client.get("/api/crm/agents").json()
                check("agents: service first with service_v1, general second with actions_v1",
                      [a["id"] for a in agents] == ["service", "general"] and
                      [a["label"] for a in agents] == ["Service desk agent", "General CRM agent"] and
                      [a["protocol"] for a in agents] == ["service_v1", PROTOCOL] and
                      all(a["description"] for a in agents) and
                      [j["label"] for j in agents[0]["jobs"]] == [j for _, j in SERVICE_JOBS] and
                      agents[1]["jobs"] == [])
                svc = client.get("/api/crm/examples").json()
                svc_ids = [e["id"] for e in svc]
                check("service examples: default agent, 9 items, full shape plus job",
                      len(svc) == 9 and all(set(e) == keys | {"job"} for e in svc) and
                      svc == client.get("/api/crm/examples", params={"agent": "service"}).json())
                check("service examples: unique ids, real b2b dev tasks, the right types, same question",
                      len(set(svc_ids)) == 9 and all(
                          real[int(i.rsplit("_", 1)[1])].split == "dev" and
                          real[int(i.rsplit("_", 1)[1])].task_type == e["task_type"] and
                          i == f"{e['task_type']}_{real[int(i.rsplit('_', 1)[1])].idx}" and
                          real[int(i.rsplit("_", 1)[1])].query == e["question"] and e["id"] == i
                          for i, e in zip(svc_ids, svc)))
                check("service examples: three per job in job order, job labels",
                      [e["job"] for e in svc] == [label for _, label in SERVICE_JOBS for _ in range(3)] and
                      [e["task_type"] for e in svc] == [t for t, _ in SERVICE_JOBS for _ in range(3)])
                check("service examples: exactly the curated SERVICE_SHOWCASE dev tasks, all in dev",
                      all([int(e["id"].rsplit("_", 1)[1]) for e in svc if e["task_type"] == tt]
                          == list(SERVICE_SHOWCASE[tt]) for tt, _ in SERVICE_JOBS)
                      and all(real[int(e["id"].rsplit("_", 1)[1])].split == "dev" for e in svc))
                titles = [e["title"] for e in svc]
                check("service examples: titles are distinct plain English per example",
                      len(set(titles)) == 9 and all(t.startswith("Route: ") for t in titles[:3]) and
                      all(re.fullmatch(r"Check case \w{6}", t) for t in titles[3:6]) and
                      all(len(t) <= 64 for t in titles[6:]) and
                      "Case Subject: " + titles[0][len("Route: "):] in svc[0]["question"])
                check("examples are per agent: an id of the other agent's list is 422",
                      not {"handle_time_881"} & set(svc_ids) and
                      client.post("/api/crm/ask", json={"example_id": "handle_time_881"}).status_code == 422 and
                      client.post("/api/crm/ask", json={"example_id": svc_ids[3],
                                                        "agent": "general"}).status_code == 422)

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

                r = client.post("/api/crm/ask", json={"example_id": "handle_time_881", "mode": "customer",
                                                                  "agent": "general"})
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
                r = client.post("/api/crm/ask", json={"example_id": svc_ids[0], "agent": "service"})
                evs = sse(r)
                check("ask service example (scripted): valid run, run_start.agent is service",
                      r.status_code == 200 and all(e["type"] in events.EVENT_TYPES for e in evs) and
                      evs[0]["type"] == "run_start" and evs[-1]["type"] == "done" and
                      evs[0]["agent"] == "service" and evs[0]["example"]["id"] == svc_ids[0] and
                      evs[0]["question"] == svc[0]["question"])
                check("ask: run_start.agent is general for the general agent, service by default",
                      evs[0]["agent"] == "service" and
                      sse(client.post("/api/crm/ask", json={"question": "x", "agent": "general"}))[0]["agent"] == "general")
                check("ask: scripted events were not mutated for later runs",
                      events.scripted_run("handle_time_881")[0][1]["model"] == "Qwen/Qwen3-8B")

                check("ask neither: 422", client.post("/api/crm/ask", json={}).status_code == 422)
                check("ask blank question: 422",
                      client.post("/api/crm/ask", json={"question": "  "}).status_code == 422)
                check("ask unknown example_id: 422",
                      client.post("/api/crm/ask", json={"example_id": "nope_1"}).status_code == 422)
                check("ask bad agent: 422",
                      client.post("/api/crm/ask", json={"question": "x", "agent": "boss"}).status_code == 422)
                check("ask bad mode: 422",
                      client.post("/api/crm/ask", json={"question": "x", "mode": "boss"}).status_code == 422)
                os.environ["CRM_AGENT_LIVE"] = "1"
                saved_url, globals()["VLLM_URL"] = VLLM_URL, "http://127.0.0.1:9"
                try:
                    r = client.post("/api/crm/ask", json={"question": "x"})
                    evs = sse(r)
                finally:
                    del os.environ["CRM_AGENT_LIVE"]
                    globals()["VLLM_URL"] = saved_url
                check("ask live, no model served: error event then done",
                      r.status_code == 200 and [e["type"] for e in evs] == ["error", "done"]
                      and "no model is being served" in evs[0]["message"])
        finally:
            db.DB_PATH, config.GATEWAY_URL = old
    print("\n".join(f"FAILED: {f}" for f in fails) if fails else "ALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())

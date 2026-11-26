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
import time
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from crm import events, service, tasks

router = APIRouter(prefix="/api/crm")

DEMO_TODAY = "2023-11-24"
DEMO_CUSTOMER_ID = "003Wt00000Jqt7DIAR"  # Kofi Nkrumah: the b2b contact with the most cases (4), for customer mode
SWITCH_SECONDS = 3.0
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
# One agent: the scoped service desk agent. The older general agent (free actions over all data, no customer
# scoping) leaked another customer's order in customer mode during the dry run and was removed (P11-UX).
AGENTS = [
    {"id": "service", "label": "Service desk agent", "protocol": SERVICE_PROTOCOL,
     "description": "Routes new cases, checks cases against policy, answers from the knowledge base",
     "jobs": [{"task_type": t, "label": label} for t, label in SERVICE_JOBS]},
]
_STATE: dict = {"current": MODEL_OPTIONS[0]["id"], "switching": False}
_BACKGROUND: set[asyncio.Task] = set()   # strong refs so a switch task is not collected


class ModelBody(BaseModel):
    id: str


class AskBody(BaseModel):
    question: str | None = None
    example_id: str | None = None
    agent: Literal["service"] = "service"
    mode: Literal["staff", "customer"] = "staff"


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


def _case_subject(cid: str) -> str | None:
    from crm.env import CRMEnv
    session = CRMEnv("b2b").open_task()
    try:
        row = session.conn.execute('SELECT Subject FROM "Case" WHERE Id = ?', [cid]).fetchone()
        return row[0] if row else None
    finally:
        session.close()


def _readable(task: tasks.Task) -> tuple[str, str | None]:
    """(the question as a person would ask it, a detail line) for an example, instead of the benchmark prompt."""
    if task.task_type == "case_routing":
        subj = re.search(r"Case Subject:\s*(.+)", task.query)
        desc = re.search(r"Case Description:\s*(.+)", task.query, re.S)
        detail = " ".join(x for x in (subj and f"{subj.group(1).strip()}.", desc and desc.group(1).strip()) if x)
        return "Who should handle this new case?", detail or None
    if task.task_type == "policy_violation_identification":
        m = re.search(r"Case Id[^:\n]*:\s*([0-9A-Za-z]+)", task.required)
        if m:
            return f"Did the support agent break policy on case {m.group(1)}?", _case_subject(m.group(1))
    return task.query.strip(), None


def _service_title(task: tasks.Task) -> str:
    """A short card title per example: the case subject, or the question itself."""
    if task.task_type == "case_routing":
        m = re.search(r"Case Subject:\s*(.+)", task.query)
        return _clip(m.group(1), 60) if m else _title(task)
    if task.task_type == "policy_violation_identification":
        return _clip(_readable(task)[1] or _title(task), 60)
    return _clip(task.query, 70)


def _example(task: tasks.Task, job: str) -> dict:
    question, detail = _readable(task)
    return {"id": f"{task.task_type}_{task.idx}", "title": _service_title(task), "question": question,
            "detail": detail, "task_type": task.task_type, "skill": task.skill, "today": task.today,
            "mode": "customer" if task.external else "staff", "job": job,
            "has_expected": task.answer != [None] and task.reward_metric == "exact_match"}


# Curated b2b dev tasks known to be solvable with the service actions (crm/service.py oracles and an
# answer-in-article check): many knowledge_qa answers are absent from this org's articles.
SERVICE_SHOWCASE = {"case_routing": (1483, 1489, 1497),
                    "policy_violation_identification": (1412, 1459, 1384),  # two violations, one None
                    "knowledge_qa": (732, 768, 762)}  # kq_labels: 732, 768 answerable (borderline); 762 a trap


@functools.lru_cache(maxsize=1)
def _service_examples() -> tuple[dict, ...]:
    """SERVICE_SHOWCASE dev tasks of each service job, grouped by job, in the listed order."""
    by_idx = {t.idx: t for t in tasks.load("b2b") if t.split == "dev"}
    out = []
    for task_type, label in SERVICE_JOBS:
        out.extend(_example(by_idx[i], label) for i in SERVICE_SHOWCASE[task_type])
    return tuple(out)


def _models_payload() -> dict:
    # The demo runs on Qwen only (owner, 2026-11-24); CRM_MODEL_PICKER=1 brings the xLAM switch back.
    return {"current": _STATE["current"], "switching": _STATE["switching"],
            "options": MODEL_OPTIONS, "picker": os.environ.get("CRM_MODEL_PICKER") == "1"}


async def _finish_switch(model_id: str) -> None:
    """Stand-in for the real 1-2 minute model reload."""
    try:
        await asyncio.sleep(SWITCH_SECONDS)
        _STATE["current"] = model_id
    finally:
        _STATE["switching"] = False


@router.get("/agents")
async def agents():
    return AGENTS


@router.get("/examples")
async def examples(agent: Literal["service"] = "service"):
    return list(_service_examples())


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


@functools.lru_cache(maxsize=64)
def _contact_name(contact_id: str) -> str | None:
    import sqlite3
    from crm.env import CRMEnv
    conn = sqlite3.connect(f"file:{CRMEnv('b2b').db_path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT FirstName || ' ' || LastName FROM Contact WHERE Id = ?", [contact_id]).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _context(task: tasks.Task) -> dict:
    ids = sorted(set(re.findall(r"\b[0-9a-zA-Z]{3}Wt[0-9a-zA-Z]{10,13}\b", task.required)))
    m = re.search(r"##\s*([^\n#]+)", task.required)
    cust = service.logged_in_customer(task.required)
    return {"today": task.today, "focus_ids": ids[:5], "policy_title": m.group(1).strip() if m else None,
            "customer": (_contact_name(cust) or cust) if cust else None}


async def _route_question(question: str, base_task: tasks.Task, client, model: str):
    """Pick the job for a typed staff question, reading cases through a private read-only copy."""
    from crm import router as job_router
    from crm.env import CRMEnv
    session = await asyncio.to_thread(CRMEnv("b2b").open_task)
    try:
        return await job_router.plan(question, base_task, session.conn, client, VLLM_URL, model,
                                     seed=uuid.uuid4().int % 10_000)
    finally:
        session.close()


def _router_events(plan, question: str, ms: float) -> list[dict]:
    """Step 0 of a typed question (the job chosen), plus the answer and the end when the router replies itself."""
    from crm import router as job_router
    label = job_router.JOB_LABELS[plan.job]
    nxt = "Run this job" if plan.task is not None else "Reply without running a job"
    evs = [{"type": "step_start", "step": 0},
           {"type": "action", "step": 0, "call_id": "router", "name": "choose_job",
            "args": {"job": label, "reason": plan.reason}},
           {"type": "observation", "step": 0, "call_id": "router", "ok": True, "rows_total": 1, "rows_shown": 1,
            "truncated": False, "columns": ["Job", "Case Id", "Next"],
            "preview": [{"Job": label, "Case Id": plan.case_id, "Next": nxt}], "error": None, "elapsed_ms": ms}]
    if plan.task is None:
        evs += [{"type": "answer", "text": plan.reply, "reason": plan.reason, "quote": None, "checks": [],
                 "sources": [], "expected": None, "correct": None},
                {"type": "stats", "steps": 1, "actions": 1, "wall_ms": ms, "prompt_tokens": None,
                 "cached_tokens": None, "completion_tokens": None, "est_cost_usd": events.est_cost_usd(ms)},
                {"type": "done"}]
    return evs


ID_RE = re.compile(r"\b(?:005|ka0|500)[0-9A-Za-z]{12}(?:[0-9A-Za-z]{3})?\b")


def _display(text: str | None, task_type: str | None) -> dict | None:
    """The answer as a person reads it: an agent's name, an article's title, 'no violation', with the Id as detail."""
    t = (text or "").strip()
    if task_type == "policy_violation_identification" and t.lower() in ("none", "no violation", ""):
        return {"text": "No policy violation found.", "detail": None}
    ids = ID_RE.findall(t)
    if not ids or len(t) > 40:  # a sentence: shown as it is
        return None
    from crm.env import CRMEnv
    session = CRMEnv("b2b").open_task()
    try:
        cid = ids[0]
        if cid.startswith("005"):
            row = session.conn.execute("SELECT FirstName || ' ' || LastName FROM User WHERE Id = ?", [cid]).fetchone()
            return {"text": row[0] if row else cid, "detail": f"Support agent {cid}"}
        if cid.startswith("ka0"):
            row = session.conn.execute("SELECT Title FROM Knowledge__kav WHERE Id = ?", [cid]).fetchone()
            lead = "Yes. The support agent broke the policy in " if task_type == "policy_violation_identification" else ""
            return {"text": f"{lead}“{row[0]}”" if row else cid, "detail": f"Knowledge article {cid}"}
        return None
    finally:
        session.close()


def _shift_step(ev: dict, shift: int) -> dict:
    """Agent events after the router's step 0 move up one step."""
    return {**ev, "step": ev["step"] + shift} if shift and "step" in ev else ev


async def _live(body: AskBody, question: str, example: dict | None, request: Request):
    """Run the real agent and stream its events (crm/events.py) as SSE data lines."""
    import httpx
    from crm import agent
    from crm.env import CRMEnv
    task = _task_by_id()[example["id"]] if example else _free_task(question, body.mode)
    task = dataclasses.replace(task, external=body.mode == "customer")
    if body.mode == "customer" and not service.logged_in_customer(task.required):
        # Customer mode is always a logged-in customer, so the agent's tools are always scoped to one person.
        task = dataclasses.replace(task, required=f"{task.required}\nThe customer you are interacting with is "
                                                  f"logged in as Id: {DEMO_CUSTOMER_ID}.")
    model = await _served_model()
    if model is None:
        yield {"data": json.dumps({"type": "error", "message": f"no model is being served at {VLLM_URL}"})}
        yield {"data": json.dumps({"type": "done"})}
        return
    # q-judge: thinking on knowledge and customer answers, off where code holds the rules (P11 test results)
    arm = dataclasses.replace(agent.ARMS["xlam-t0"] if "xLAM" in model else agent.ARMS["q-judge"], model=model)
    protocol = SERVICE_PROTOCOL
    start = {"type": "run_start", "run_id": f"live-{uuid.uuid4().hex[:8]}", "question": question,
             "detail": example.get("detail") if example else None,
             "model": model, "mode": body.mode, "agent": body.agent, "context": _context(task),
             "example": {k: example[k] for k in ("id", "task_type", "skill")} if example else None}
    log = {"saved_at": None, "run_id": start["run_id"], "question": question, "model": model, "arm": arm.name,
           "protocol": protocol, "agent": body.agent, "mode": body.mode, "example": start["example"],
           "outcome": "running", "events": [start], "record": None}
    yield {"data": json.dumps(start)}
    queue: asyncio.Queue = asyncio.Queue()
    runner = None
    try:
        async with httpx.AsyncClient(timeout=agent.CALL_TIMEOUT_S) as client:
            shift, job_type = 0, task.task_type
            if body.mode == "staff" and example is None:
                # a typed staff question: pick the job first, then run that job's own path (crm/router.py)
                t0 = time.perf_counter()
                plan = await _route_question(question, task, client, model)
                for ev in _router_events(plan, question, (time.perf_counter() - t0) * 1000):
                    log["events"].append(ev)
                    yield {"data": json.dumps(ev, default=str)}
                log["router"] = {"job": plan.job, "reason": plan.reason, "case_id": plan.case_id}
                job_type = plan.task.task_type if plan.task is not None else None
                if plan.task is None:
                    log.update(outcome="finished")
                    return
                task, shift = plan.task, 1
            runner = asyncio.create_task(agent.run_task(
                task, arm, CRMEnv("b2b"), base_url=VLLM_URL, run_id=start["run_id"],
                run_seed=uuid.uuid4().int % 10_000, window=32768, calls_out=lambda rec: None, http_client=client,
                protocol=protocol, on_event=lambda ev: queue.put_nowait(_shift_step(ev, shift))))
            while not (runner.done() and queue.empty()):
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        log["outcome"] = "stopped by the user"
                        return
                    continue
                if ev.get("type") == "answer":
                    ev = {**ev, "display": await asyncio.to_thread(_display, ev.get("text"), job_type)}
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
        example = next((e for e in _service_examples() if e["id"] == body.example_id), None)
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
                agents = client.get("/api/crm/agents").json()
                check("agents: only the scoped service desk agent (the general agent was removed)",
                      [a["id"] for a in agents] == ["service"] and agents[0]["protocol"] == "service_v1" and
                      [j["label"] for j in agents[0]["jobs"]] == [j for _, j in SERVICE_JOBS])
                check("the general agent is refused everywhere: examples and ask are 422",
                      client.get("/api/crm/examples", params={"agent": "general"}).status_code == 422 and
                      client.post("/api/crm/ask", json={"question": "x", "agent": "general"}).status_code == 422
                      and client.post("/api/crm/ask", json={"question": "x", "agent": "general",
                                                            "mode": "customer"}).status_code == 422)
                svc = client.get("/api/crm/examples").json()
                svc_ids = [e["id"] for e in svc]
                keys = {"id", "title", "question", "detail", "task_type", "skill", "today", "mode", "job",
                        "has_expected"}
                check("service examples: 9 items, full shape",
                      len(svc) == 9 and all(set(e) == keys for e in svc))
                check("service examples: unique ids, real b2b dev tasks of the right types",
                      len(set(svc_ids)) == 9 and all(
                          real[int(i.rsplit("_", 1)[1])].split == "dev" and
                          real[int(i.rsplit("_", 1)[1])].task_type == e["task_type"] and e["id"] == i
                          for i, e in zip(svc_ids, svc)))
                check("service examples: three per job in job order, the curated SERVICE_SHOWCASE tasks",
                      [e["job"] for e in svc] == [label for _, label in SERVICE_JOBS for _ in range(3)] and
                      all([int(e["id"].rsplit("_", 1)[1]) for e in svc if e["task_type"] == tt]
                          == list(SERVICE_SHOWCASE[tt]) for tt, _ in SERVICE_JOBS))
                rt0 = real[int(svc_ids[0].rsplit("_", 1)[1])]
                check("readable examples: no benchmark prompt; routing asks plainly, the case as detail",
                      not any("Return only" in e["question"] or "##" in e["question"] for e in svc) and
                      svc[0]["question"] == "Who should handle this new case?" and
                      re.search(r"Case Subject:\s*(.+)", rt0.query).group(1).strip() in svc[0]["detail"])
                check("readable examples: policy names the case and shows its subject; knowledge is the question",
                      all(e["question"].startswith("Did the support agent break policy on case 500") and e["detail"]
                          for e in svc[3:6]) and
                      all(e["question"] == real[int(e["id"].rsplit("_", 1)[1])].query.strip() for e in svc[6:]))
                titles = [e["title"] for e in svc]
                check("service examples: titles distinct and short", len(set(titles)) == 9 and
                      all(len(t) <= 73 for t in titles))
                check("an unknown example id is 422",
                      client.post("/api/crm/ask", json={"example_id": "handle_time_881"}).status_code == 422)
                check("display: agent Id -> name; policy None -> no violation; article -> its title; a sentence untouched",
                      _display("005Wt000003NIYnIAO", "case_routing")["text"] == "Emiko Tanaka" and
                      _display("None", "policy_violation_identification")["text"] == "No policy violation found." and
                      _display("ka0Wt000000Em4zIAC", "policy_violation_identification")["text"].startswith(
                          "Yes. The support agent broke the policy in “") and
                      _display("The return window is 30 days.", "knowledge_qa") is None)

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

                r = client.post("/api/crm/ask", json={"example_id": svc_ids[6], "mode": "customer"})
                evs = sse(r)
                start = evs[0]
                check("ask example: 200, event-stream, all types in EVENT_TYPES, run_start..done",
                      r.status_code == 200 and "text/event-stream" in r.headers["content-type"] and
                      all(e["type"] in events.EVENT_TYPES for e in evs) and
                      start["type"] == "run_start" and evs[-1]["type"] == "done")
                check("ask example: run_start rewritten to the request",
                      start["example"]["id"] == svc_ids[6] and start["model"] == xlam and
                      start["mode"] == "customer" and start["question"] == svc[6]["question"])

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

"""Pick the service job for a typed staff question: an intent router over job scopes, like Agentforce's Agent Router.

  plan = await router.plan(question, base_task, conn, client, base_url, model)  # -> Plan (a task, or a reply)
The model picks the job by name from a strict menu; code checks each job's required inputs (a Case Id).
"""
from __future__ import annotations

import dataclasses
import functools
import json
import re

# name -> classification description, the only thing the router reads (as in Agentforce)
JOBS = {
    "case_routing": "Find the best agent to handle a customer case, given its Case Id or a description of the "
                    "customer's problem.",
    "policy_violation_identification": "Check whether the support agent on an existing case broke company policy "
                                       "(the offered fixes). Needs a Case Id.",
    "knowledge_qa": "Answer a question about products, features, company policies or how to do something, from the "
                    "knowledge base.",
    "off_topic": "Anything else: small talk, sales or analytics questions, requests to change records, or a request "
                 "too unclear to tell.",
}
JOB_LABELS = {"case_routing": "Route a case", "policy_violation_identification": "Check a case against policy",
              "knowledge_qa": "Answer from the knowledge base", "off_topic": "Not one of my jobs"}

CASE_ID_RE = re.compile(r"\b500[0-9A-Za-z]{12}(?:[0-9A-Za-z]{3})?\b")

OFF_TOPIC = ("I can help with three things: routing a case to the best agent, checking a case against company policy, "
             "and answering questions from the knowledge base.")
ASK_POLICY_CASE = "Which case should I check? Please give its Case Id (it starts with 500)."
ASK_ROUTING_CASE = "Which case should I route? Please give its Case Id (it starts with 500) or describe the problem."
NOT_FOUND = "I couldn't find a case with the Id {cid}."

ROUTER_PROMPT = """\
You are the front desk of a service desk agent for a B2B electronics and software company. Decide which ONE job
the employee's message asks for. The jobs:
{jobs}

Message: {question}

Call choose_job. Write `reason` first (what the message asks for), then `job`, then `case_described`."""

CHOOSE_JOB = {"type": "function", "function": {
    "name": "choose_job", "description": "Choose the job this message asks for.",
    "parameters": {"type": "object", "required": ["reason", "job", "case_described"], "properties": {
        "reason": {"type": "string", "description": "What the message asks for, in one sentence."},
        "job": {"type": "string", "enum": list(JOBS)},
        "case_described": {"type": "boolean", "description": "True only if the message itself describes a "
                           "specific customer problem (what is wrong, and ideally with which product)."}}}}}


@dataclasses.dataclass(frozen=True)
class Plan:
    job: str
    reason: str
    task: object | None = None  # a tasks.Task to run, or None when the router answers itself
    reply: str | None = None
    case_id: str | None = None


def parse_choice(arguments: str) -> tuple[str, str, bool] | None:
    """(job, reason, case_described) from choose_job's arguments, or None if they are not a valid choice."""
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(args, dict) or args.get("job") not in JOBS:
        return None
    return args["job"], str(args.get("reason") or ""), args.get("case_described") is True


@functools.lru_cache(maxsize=None)
def _template(task_type: str):
    """The benchmark's own wording for a job (first dev task of that type), so the app runs the evaluated inputs."""
    from crm import tasks
    return min((t for t in tasks.load("b2b") if t.split == "dev" and t.task_type == task_type), key=lambda t: t.idx)


def _case(conn, cid: str) -> dict | None:
    cur = conn.execute('SELECT Subject, Description FROM "Case" WHERE Id = ?', [cid])
    row = cur.fetchone()
    return None if row is None else {"Subject": row[0] or "", "Description": row[1] or ""}


def build(question: str, job: str, reason: str, case_described: bool, base_task, conn) -> Plan:
    """Apply the job's input rules in code and build its task, or the reply asking for what is missing."""
    cid = (CASE_ID_RE.findall(question) or [None])[0]
    if job == "off_topic":
        return Plan(job, reason, reply=OFF_TOPIC)
    if job == "knowledge_qa":
        return Plan(job, reason, task=dataclasses.replace(base_task, task_type=job, skill=_template(job).skill))
    if job == "policy_violation_identification":
        if cid is None:
            return Plan(job, reason, reply=ASK_POLICY_CASE)
        if _case(conn, cid) is None:
            return Plan(job, reason, reply=NOT_FOUND.format(cid=cid), case_id=cid)
        t = _template(job)
        task = dataclasses.replace(base_task, task_type=job, skill=t.skill, query=t.query,
                                   required=f"{base_task.required}\n- Case Id to be considered is: {cid}".strip())
        return Plan(job, reason, task=task, case_id=cid)
    # case_routing: the case's own subject and description, or the message when it describes the problem
    if cid is not None:
        case = _case(conn, cid)
        if case is None:
            return Plan(job, reason, reply=NOT_FOUND.format(cid=cid), case_id=cid)
    elif case_described:
        case = {"Subject": "(not given)", "Description": question.strip()}
    else:
        return Plan(job, reason, reply=ASK_ROUTING_CASE)
    t = _template(job)
    head = t.query.split("## Given Case")[0].rstrip()
    query = f"{head}\n## Given Case\n- Case Subject: {case['Subject']}\n- Case Description: {case['Description']}"
    task = dataclasses.replace(base_task, task_type=job, skill=t.skill, query=query,
                               required=f"{base_task.required}\n{t.required}".strip())
    return Plan(job, reason, task=task, case_id=cid)


async def plan(question: str, base_task, conn, client, base_url: str, model: str, seed: int = 0) -> Plan:
    """One forced choose_job call (thinking off), then build(). If the call fails, answer from the knowledge base."""
    from crm import agent
    arm = dataclasses.replace(agent.ARMS["q-off"], model=model, max_tokens=300, tool_choice="required")
    jobs = "\n".join(f"- {name}: {desc}" for name, desc in JOBS.items())
    messages = [{"role": "user", "content": ROUTER_PROMPT.format(jobs=jobs, question=question.strip())}]
    res, _, _, _, err = await agent.model_call(client, base_url, arm, messages, seed, 4096, [CHOOSE_JOB])
    calls = (res.tool_calls if res else None) or []
    choice = parse_choice(calls[0].get("function", {}).get("arguments", "")) if calls else None
    if choice is None:  # the knowledge job is the safe default: it answers only from a checked source
        return build(question, "knowledge_qa", f"router failed ({err or 'no valid choice'})", False, base_task, conn)
    return build(question, *choice, base_task, conn)


def _selftest() -> int:
    import sqlite3

    from crm import tasks
    from crm.env import CRMEnv

    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    base = tasks.Task(org="b2b", idx=-1, task_type="free_question", skill="free", external=False, query="q?",
                      required="Today's date: 2024-10-01", optional="", answer=[None], reward_metric="none",
                      today="2024-10-01", split="app")
    session = CRMEnv("b2b").open_task()
    conn = session.conn
    pol = _template("policy_violation_identification")
    cid = re.search(r"Case Id to be considered is:\s*(\w+)", pol.required).group(1)
    check("parse: valid choice", parse_choice('{"reason": "r", "job": "knowledge_qa", "case_described": false}')
          == ("knowledge_qa", "r", False))
    check("parse: job outside the menu is rejected", parse_choice('{"reason": "r", "job": "sales"}') is None)
    check("parse: broken JSON is rejected", parse_choice("{not json") is None)
    p = build("hello there", "off_topic", "r", False, dataclasses.replace(base, query="hello there"), conn)
    check("off topic: fixed reply, no task", p.task is None and p.reply == OFF_TOPIC)
    p = build("did the agent break policy?", "policy_violation_identification", "r", False, base, conn)
    check("policy without a Case Id: asks for it", p.task is None and p.reply == ASK_POLICY_CASE)
    p = build("check 500Wt00000ZZZZZZZZ", "policy_violation_identification", "r", False, base, conn)
    check("policy with an unknown Case Id: says not found", p.task is None and "500Wt00000ZZZZZZZZ" in p.reply)
    p = build(f"check case {cid} for policy", "policy_violation_identification", "r", False, base, conn)
    check("policy with a Case Id: the benchmark's query and the Id in the context",
          p.task is not None and p.task.query == pol.query and f"Case Id to be considered is: {cid}" in p.task.required
          and p.task.required.startswith("Today's date") and p.task.task_type == "policy_violation_identification")
    p = build("who should handle this?", "case_routing", "r", False, base, conn)
    check("routing with no Id and no description: asks", p.task is None and p.reply == ASK_ROUTING_CASE)
    msg = "A customer's SecureAnalytics Pro install fails with error 1603. Who should take it?"
    p = build(msg, "case_routing", "r", True, base, conn)
    check("routing from a description: the message becomes the case description, with the routing policy",
          p.task is not None and f"Case Description: {msg}" in p.task.query and "Case Routing Policy" in p.task.required)
    subj = _case(conn, cid)["Subject"]
    p = build(f"route {cid}", "case_routing", "r", False, base, conn)
    check("routing from a Case Id: the case's own subject from the database",
          p.task is not None and f"Case Subject: {subj}" in p.task.query)
    p = build("what is the return window?", "knowledge_qa", "r", False,
              dataclasses.replace(base, query="what is the return window?"), conn)
    check("knowledge: the typed question as a knowledge_qa task",
          p.task is not None and p.task.task_type == "knowledge_qa" and p.task.query == "what is the return window?")
    try:
        conn.execute('DELETE FROM "Case" WHERE Id = ?', [cid])
        check("router reads through the read-only session", False)
    except sqlite3.DatabaseError:
        check("router reads through the read-only session", True)
    session.close()
    print(f"{'OK' if not fails else 'FAILED'}: router selftest, {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())

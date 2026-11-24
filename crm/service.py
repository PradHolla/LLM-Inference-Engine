"""Phase 11 CRM service-desk actions: purpose-built lookups (case file, knowledge, agent expertise and workload).

  obs = execute(session, "agent_expertise", {"kind": "issue", "id": "a03...", "top": 0}, "b2b")
  uv run python -m crm.service --selftest
"""
from __future__ import annotations

import argparse
import ast
import html
import re
import sys
import time

from crm.actions import ActionError, _close, _fail, _parse_rows
from crm.env import Observation, TaskSession, render_rows
from crm.prompts import TOOLS, _tool

SERVICE_TYPES = ("case_routing", "policy_violation_identification", "knowledge_qa")
ID_RE = re.compile(r"^[A-Za-z0-9]{15,18}$")
KINDS = ["issue", "product"]
BODY_CAP = 4000  # per email / chat / description; shrinks only if the whole case file exceeds the char cap
DESC_CAP = 120
MAX_SEARCH, DEFAULT_SEARCH, MIN_SEARCH = 10, 5, 3
MAX_TOP, DEFAULT_TOP = 20, 10
MAX_AGENTS = 50


# ---------------------------------------------------------------- helpers

def _obs(text: str, total: int | None, shown: int | None, truncated: bool, t0: float) -> Observation:
    return Observation(True, text, total, shown, truncated, (time.monotonic() - t0) * 1000, None)


def _rows(session: TaskSession, sql: str, params: list) -> list[dict]:
    """Bound-parameter read through the session's guarded connection, as a list of dicts."""
    cur = session.conn.execute(sql, params)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


def _id(value, what: str) -> str:
    """A Salesforce-style Id (15 or 18 alphanumerics) or an ActionError naming what was expected."""
    v = value.strip() if isinstance(value, str) else ""
    if not ID_RE.match(v):
        raise ActionError(f"{what} must be a 15 or 18 character record Id (letters and digits only), got {value!r}")
    return v


def _bounded(value, default: int, hi: int, what: str) -> int:
    """0 means default; 1..hi as given; larger values are clamped; negatives are an error."""
    n = int(value or 0)
    if n < 0:
        raise ActionError(f"{what} must be between 1 and {hi} (0 means {default}), got {n}")
    return default if n == 0 else min(n, hi)


def _clip(text, n: int) -> str:
    """Unescape HTML entities, trim, and cut to n chars with a note of how much was dropped."""
    t = html.unescape(str(text or "")).strip()
    return t if len(t) <= n else t[:n].rstrip() + f" ...[{len(t) - n} more chars cut]"


def _exists(session: TaskSession, table: str, rid: str, label: str, hint: str) -> None:
    n = _rows(session, f'SELECT COUNT(*) AS n FROM "{table}" WHERE Id = ?', [rid])[0]["n"]
    if not n:
        raise ActionError(f"no {label} with Id {rid!r}. {hint}")


# ---------------------------------------------------------------- actions

def _case_text(case: dict, emails: list[dict], chats: list[dict], hist: list[dict], body_cap: int) -> str:
    lines = [f"CASE {case['Id']}: {case['Subject']}"]
    for k in ("Status", "Priority", "CreatedDate", "ClosedDate", "OwnerId", "ContactId", "AccountId"):
        lines.append(f"  {k}: {case[k]}")
    lines.append(f"  Issue: {case['IssueId__c']} ({case['IssueName']})")
    lines.append(f"  Product: {case['Product2Id']} ({case['ProductName']}), OrderItem {case['OrderItemId__c']}")
    lines.append(f"  Description: {_clip(case['Description'], body_cap)}")
    lines.append(f"EMAILS ({len(emails)}, oldest first):")
    for i, e in enumerate(emails, 1):
        lines.append(f"[{i}] {e['MessageDate']} from {e['FromAddress']} | {e['Subject']}")
        lines.append(_clip(e["TextBody"], body_cap))
    lines.append(f"LIVE CHAT TRANSCRIPTS ({len(chats)}):")
    for c in chats:
        lines.append(f"chat {c['Id']} ended {c['EndTime']}, owner {c['OwnerId']}:")
        lines.append(_clip(c["Body"], body_cap))
    lines.append(f"HISTORY ({len(hist)}, oldest first):")
    for h in hist:
        lines.append(f"{h['CreatedDate']} {h['Field__c']}: {h['OldValue__c']} -> {h['NewValue__c']}")
    return "\n".join(lines)


def get_case(session: TaskSession, case_id) -> Observation:
    t0 = time.monotonic()
    cid = _id(case_id, "case_id")
    cases = _rows(session, 'SELECT c.*, i.Name AS IssueName, oi.Product2Id AS Product2Id, p.Name AS ProductName '
                  'FROM "Case" c LEFT JOIN Issue__c i ON i.Id = c.IssueId__c '
                  'LEFT JOIN OrderItem oi ON oi.Id = c.OrderItemId__c LEFT JOIN Product2 p ON p.Id = oi.Product2Id '
                  'WHERE c.Id = ?', [cid])
    if not cases:
        raise ActionError(f"no Case with Id {cid!r}. A new case that is not in the database has no case file; "
                          "work from the text you were given.")
    emails = _rows(session, "SELECT * FROM EmailMessage WHERE ParentId = ? ORDER BY MessageDate, Id", [cid])
    chats = _rows(session, "SELECT * FROM LiveChatTranscript WHERE CaseId = ? ORDER BY EndTime, Id", [cid])
    hist = _rows(session, "SELECT * FROM CaseHistory__c WHERE CaseId__c = ? ORDER BY CreatedDate, Id", [cid])
    cap, body_cap = session.char_cap, BODY_CAP
    text = _case_text(cases[0], emails, chats, hist, body_cap)
    while len(text) > cap and body_cap > 100:  # shrink bodies until the whole file fits
        body_cap //= 2
        text = _case_text(cases[0], emails, chats, hist, body_cap)
    cut = len(text) > cap
    if cut:
        note = "\n[case file cut to fit the size limit]"
        text = text[: cap - len(note)] + note
    n = 1 + len(emails) + len(chats) + len(hist)
    return _obs(text, n, n, cut or body_cap < BODY_CAP, t0)


def search_knowledge(session: TaskSession, query, limit) -> Observation:
    t0 = time.monotonic()
    # At least 3: a model asking for 1 took the first hit unseen and answered from the wrong article.
    n = max(MIN_SEARCH, _bounded(limit, DEFAULT_SEARCH, MAX_SEARCH, "limit"))
    obs = session.search(str(query or ""), ["Knowledge__kav"], limit=min(n * 8, session.row_cap))
    if not obs.ok:
        return obs
    note = "(no article matched all words; showing articles matching any word)\n" \
        if obs.text.startswith("(no record matched all words") else ""
    best: dict[str, dict] = {}
    for r in _parse_rows(obs.text):  # one row per article: its best-ranked matching passage
        best.setdefault(r["Id"], r)
    keep = list(best.values())[:n]
    titles = {r["Id"]: r["Title"] for r in _rows(
        session, f"SELECT Id, Title FROM Knowledge__kav WHERE Id IN ({', '.join('?' * len(keep))})",
        [r["Id"] for r in keep])} if keep else {}
    names = ["Id", "Title", "field", "snippet"]
    text, shown, trunc = render_rows(names, [(r["Id"], titles.get(r["Id"]), r["field"], r["snippet"]) for r in keep],
                                     len(keep), session.char_cap - len(note))
    return _obs(note + text, len(keep), shown, trunc, t0)


def get_article(session: TaskSession, article_id) -> Observation:
    t0 = time.monotonic()
    aid = _id(article_id, "article_id")
    rows = _rows(session, "SELECT Id, Title, Summary, FAQ_Answer__c FROM Knowledge__kav WHERE Id = ?", [aid])
    if not rows:
        raise ActionError(f"no knowledge article with Id {aid!r}. Use search_knowledge to find article Ids.")
    a = rows[0]
    text = (f"ARTICLE {a['Id']}\nTitle: {a['Title']}\nSummary: {a['Summary']}\n"
            f"FAQ_Answer__c: {a['FAQ_Answer__c']}")
    cut = len(text) > session.char_cap
    return _obs(text[: session.char_cap], 1, 1, cut, t0)


def list_issues(session: TaskSession) -> Observation:
    return session.query("SELECT Id, Name, Description__c FROM Issue__c ORDER BY Name", [])


def list_products(session: TaskSession) -> Observation:
    return session.query("SELECT Id, Name, substr(Description, 1, ?) AS Description FROM Product2 ORDER BY Name",
                         [DESC_CAP])


def agent_expertise(session: TaskSession, kind, rid, top) -> Observation:
    t0 = time.monotonic()
    if kind not in KINDS:
        raise ActionError(f"kind must be one of {', '.join(KINDS)}, got {kind!r}")
    n = _bounded(top, DEFAULT_TOP, MAX_TOP, "top")
    rid = _id(rid, f"{kind} id")
    if kind == "issue":
        _exists(session, "Issue__c", rid, "issue", "Use list_issues to see the issue Ids.")
        frm, cond = '"Case" c', "c.IssueId__c = ?"
    else:
        _exists(session, "Product2", rid, "product", "Use list_products to see the product Ids.")
        frm = '"Case" c JOIN OrderItem oi ON oi.Id = c.OrderItemId__c'
        cond = "oi.Product2Id = ?"
    allrows = _rows(session, f"SELECT c.OwnerId AS OwnerId, u.FirstName || ' ' || u.LastName AS Name, "
                    f"COUNT(*) AS closed_cases FROM {frm} LEFT JOIN User u ON u.Id = c.OwnerId "
                    f"WHERE c.Status = 'Closed' AND {cond} GROUP BY c.OwnerId ORDER BY closed_cases DESC, c.OwnerId",
                    [rid])
    if not allrows:
        return _obs(f"[] (no agent has closed a case for this {kind})", 0, 0, False, t0)
    shown = allrows[:n]
    names = ["OwnerId", "Name", "closed_cases"]
    text, k, _ = render_rows(names, [tuple(r[c] for c in names) for r in shown], len(shown), session.char_cap - 200)
    best = allrows[0]["closed_cases"]
    tied = [r["OwnerId"] for r in allrows if r["closed_cases"] == best]
    text += (f"\n[{len(allrows)} agents have closed cases for this {kind}; top count {best} closed cases"
             + (f", TIED among {len(tied)} agents: {', '.join(tied)}]" if len(tied) > 1 else ", no tie at the top]"))
    return _obs(text, len(allrows), k, k < len(allrows), t0)


def agent_workload(session: TaskSession, agent_ids) -> Observation:
    t0 = time.monotonic()
    if not isinstance(agent_ids, list) or not agent_ids:
        raise ActionError("agent_ids must be a non-empty list of User Ids")
    ids = list(dict.fromkeys(_id(a, "each agent id") for a in agent_ids))
    if len(ids) > MAX_AGENTS:
        raise ActionError(f"at most {MAX_AGENTS} agent ids per call, got {len(ids)}")
    marks = ", ".join("?" * len(ids))
    rows = _rows(session, f"SELECT u.Id AS OwnerId, u.FirstName || ' ' || u.LastName AS Name, "
                 f"COUNT(c.Id) AS open_cases FROM User u LEFT JOIN \"Case\" c ON c.OwnerId = u.Id "
                 f"AND COALESCE(c.Status, '') != 'Closed' WHERE u.Id IN ({marks}) "
                 f"GROUP BY u.Id ORDER BY open_cases ASC, u.Id", ids)
    unknown = [i for i in ids if i not in {r["OwnerId"] for r in rows}]
    names = ["OwnerId", "Name", "open_cases"]
    text, shown, trunc = render_rows(names, [tuple(r[c] for c in names) for r in rows], len(rows), session.char_cap)
    if unknown:
        text += f"\n[not a User Id: {', '.join(unknown)}]"
    least = [r["OwnerId"] for r in rows if r["open_cases"] == rows[0]["open_cases"]] if rows else []
    if len(least) > 1:
        text += f"\n[{len(least)} agents tied for the fewest open cases ({rows[0]['open_cases']})]"
    return _obs(text, len(rows), shown, trunc, t0)


def execute(session: TaskSession, name: str, args: dict, org: str) -> Observation | None:
    """Run one service action; None for respond. Validation problems come back as ok=False observations."""
    t0 = time.monotonic()
    if name == "respond":
        return None
    a = args if isinstance(args, dict) else {}
    try:
        if name == "get_case":
            return get_case(session, a.get("case_id"))
        if name == "search_knowledge":
            return search_knowledge(session, a.get("query"), a.get("limit"))
        if name == "get_article":
            return get_article(session, a.get("article_id"))
        if name == "list_issues":
            return list_issues(session)
        if name == "list_products":
            return list_products(session)
        if name == "agent_expertise":
            return agent_expertise(session, a.get("kind"), a.get("id"), a.get("top"))
        if name == "agent_workload":
            return agent_workload(session, a.get("agent_ids"))
    except ActionError as e:
        return _fail(str(e), t0)
    except (TypeError, ValueError) as e:
        return _fail(f"bad arguments for {name}: {e}", t0)
    names = ", ".join(t["function"]["name"] for t in SERVICE_TOOLS)
    return _fail(f"unknown action {name!r}; actions: {names}. Close matches: "
                 f"{_close(str(name), [t['function']['name'] for t in SERVICE_TOOLS])}", t0)


# ---------------------------------------------------------------- tool schemas

def _s(desc: str, **kw) -> dict:
    return {"type": "string", "description": desc, **kw}


# The service agent's final answer carries a one-line reason for the human; only `answer` is graded.
_RESPOND = _tool("respond", "Give the final answer. Call this once, when you are confident.",
                 {"answer": {"type": "string", "description": "ONLY the answer: an Id, None, or one short "
                             "sentence for a knowledge question. No explanation here."},
                  "reason": {"type": "string", "description": "One short line saying why, citing what you "
                             "found (for example: closed the most cases of this issue)."}},
                 ["answer", "reason"])

SERVICE_TOOLS: list[dict] = [
    _tool("get_case", "Read one support case in full: its fields (with issue and product names), its emails, live "
          "chat transcripts and change history, each oldest first. Long bodies are cut with a note.",
          {"case_id": _s("The Case Id (starts with 500).")}, ["case_id"]),
    _tool("search_knowledge", "Keyword search over knowledge articles only. Returns Id, Title and a snippet of the "
          "best matching passage, one row per article.",
          {"query": _s("Keywords to search for."),
           "limit": {"type": "integer", "description": f"Max articles, 1 to {MAX_SEARCH}; 0 means {DEFAULT_SEARCH}."}},
          ["query", "limit"]),
    _tool("get_article", "Read one knowledge article in full: Title, Summary and the complete FAQ answer.",
          {"article_id": _s("The knowledge article Id (starts with ka0).")}, ["article_id"]),
    _tool("list_issues", "List every issue category (Id, Name, Description) that cases can be filed under.", {}, []),
    _tool("list_products", "List every product (Id, Name, short Description).", {}, []),
    _tool("agent_expertise", "Rank agents by how many CLOSED cases they have for an issue or a product, most first. "
          "Ties at the top are flagged.",
          {"kind": _s("issue or product.", enum=KINDS),
           "id": _s("The issue Id (from list_issues) or product Id (from list_products)."),
           "top": {"type": "integer", "description": f"Max agents, 1 to {MAX_TOP}; 0 means {DEFAULT_TOP}."}},
          ["kind", "id", "top"]),
    _tool("agent_workload", "For each given agent (User Id), the number of cases whose Status is not Closed; "
          "agents with none show 0.",
          {"agent_ids": {"type": "array", "items": {"type": "string"},
                         "description": f"User Ids of the agents, at most {MAX_AGENTS}."}}, ["agent_ids"]),
    _RESPOND,
]


# ---------------------------------------------------------------- selftest

def _selftest() -> int:
    import collections

    from crm.env import CRMEnv
    from crm.grade import f1_score, normalize_answer
    from crm.tasks import load

    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    tasks = load("b2b")
    print("lazy-answer baselines (b2b):")
    print(f"  {'split':<6} {'policy n':>8} {'always None':>12} {'routing n':>10} {'most frequent agent':>20}")
    for split in ("dev", "test"):
        pol = [t for t in tasks if t.split == split and t.task_type == "policy_violation_identification"]
        rt = [t for t in tasks if t.split == split and t.task_type == "case_routing"]
        none_share = sum(t.answer == [None] for t in pol) / len(pol)
        top_agent, top_n = collections.Counter(t.answer[0] for t in rt).most_common(1)[0]
        print(f"  {split:<6} {len(pol):>8} {none_share:>12.0%} {len(rt):>10} {top_n / len(rt):>14.0%} ({top_agent})")

    env = CRMEnv("b2b")
    s = env.open_task()
    ex = lambda name, args: execute(s, name, args, "b2b")
    rows = lambda o: ast.literal_eval(next(ln for ln in o.text.split("\n") if ln.startswith("["))) if o.ok else []
    raw = lambda q, p=(): [dict(zip([d[0] for d in c.description], r)) for c in [s.conn.execute(q, p)] for r in c]

    # --- schemas
    for t in SERVICE_TOOLS:
        fn = t["function"]
        p = fn["parameters"]
        check(f"strict schema {fn['name']}", fn["strict"] and p["additionalProperties"] is False
              and set(p["required"]) == set(p["properties"]))
    check("respond -> None", ex("respond", {"answer": "x"}) is None)
    o = ex("get_cases", {})
    check("unknown action -> error with close match", not o.ok and "get_case" in o.text, o.text[:120])

    # --- list_issues / list_products
    o = ex("list_issues", {})
    check("list_issues: 15 rows, Id/Name/Description__c", o.ok and o.rows_total == 15
          and set(rows(o)[0]) == {"Id", "Name", "Description__c"}, o.text[:100])
    o = ex("list_products", {})
    check("list_products: 51 rows, description cut to 120", o.ok and o.rows_total == 51
          and all(len(r["Description"] or "") <= 120 for r in rows(o)) and len(o.text) <= s.char_cap, f"{len(o.text)} chars")

    # --- get_case
    many = raw('SELECT ParentId, COUNT(*) n FROM EmailMessage WHERE ParentId IS NOT NULL GROUP BY 1 '
               'ORDER BY n DESC, ParentId LIMIT 1')[0]
    o = ex("get_case", {"case_id": many["ParentId"]})
    check("get_case: case with most emails", o.ok and o.text.count("\n[") >= many["n"] and "EMAILS (%d" % many["n"] in o.text
          and "HISTORY (" in o.text and len(o.text) <= s.char_cap, f"{many['n']} emails, {len(o.text)} chars")
    chat_case = raw("SELECT CaseId FROM LiveChatTranscript ORDER BY length(Body) DESC LIMIT 1")[0]["CaseId"]
    o = ex("get_case", {"case_id": chat_case})
    check("get_case: chat transcript shown, entities unescaped", o.ok and "LIVE CHAT TRANSCRIPTS (1)" in o.text
          and "&#39;" not in o.text, f"{len(o.text)} chars")
    small = CRMEnv("b2b", char_cap=1500).open_task()
    o = execute(small, "get_case", {"case_id": many["ParentId"]}, "b2b")
    check("get_case: char cap enforced (cap 1500)", o.ok and len(o.text) <= 1500 and o.truncated,
          f"{len(o.text)} chars, tail {o.text[-45:]!r}")
    small.close()
    o = ex("get_case", {"case_id": "x' OR 1=1 --"})
    check("get_case: injection string rejected as a non-Id", not o.ok and "15 or 18 character" in o.text, o.text[:100])
    o = ex("get_case", {"case_id": "500Wt00000ZZZZZZZZ"})
    check("get_case: unknown case -> helpful", not o.ok and "no Case with Id" in o.text and "new case" in o.text, o.text[:100])
    o = ex("get_case", {})
    check("get_case: missing id -> error", not o.ok and "case_id" in o.text, o.text[:100])

    # --- search_knowledge / get_article
    o = ex("search_knowledge", {"query": "HPC Accelerator Suite", "limit": 0})
    check("search_knowledge: default 5, one row per article, Knowledge only", o.ok and o.rows_total == 5
          and len({r["Id"] for r in rows(o)}) == 5 and all(r["Title"] for r in rows(o)), o.text[:140])
    o = ex("search_knowledge", {"query": "design", "limit": 99})
    check("search_knowledge: limit clamped to 10", o.ok and o.rows_total == 10, f"{o.rows_total}")
    o = ex("search_knowledge", {"query": "x' OR 1=1 --", "limit": 3})
    check("search_knowledge: injection text is just words", o.ok or "searchable" in o.text, o.text[:100])
    o = ex("search_knowledge", {"query": "!!!", "limit": 3})
    check("search_knowledge: no words -> error", not o.ok and "no searchable words" in o.text, o.text)
    o = ex("search_knowledge", {"query": "billing", "limit": -1})
    check("search_knowledge: negative limit -> error", not o.ok and "between 1 and 10" in o.text, o.text)
    aid = rows(ex("search_knowledge", {"query": "billing", "limit": 1}))[0]["Id"]
    o = ex("get_article", {"article_id": aid})
    full = raw("SELECT FAQ_Answer__c f FROM Knowledge__kav WHERE Id = ?", [aid])[0]["f"]
    check("get_article: full FAQ answer", o.ok and full in o.text and o.text.startswith(f"ARTICLE {aid}"), f"{len(o.text)} chars")
    o = ex("get_article", {"article_id": "ka0Wt000000ZZZZZZZ"})
    check("get_article: unknown -> helpful", not o.ok and "search_knowledge" in o.text, o.text)
    o = ex("get_article", {"article_id": "ka0' OR '1'='1"})
    check("get_article: injection rejected", not o.ok and "15 or 18" in o.text, o.text[:80])

    # --- agent_expertise / agent_workload
    iss = rows(ex("list_issues", {}))[0]["Id"]
    o = ex("agent_expertise", {"kind": "issue", "id": iss, "top": 0})
    want = raw("SELECT OwnerId, COUNT(*) n FROM \"Case\" WHERE Status='Closed' AND IssueId__c=? GROUP BY 1 "
               "ORDER BY n DESC, OwnerId", [iss])
    got = rows(o)
    check("agent_expertise issue: matches SQL, descending", o.ok and [(r["OwnerId"], r["closed_cases"]) for r in got]
          == [(w["OwnerId"], w["n"]) for w in want[:10]] and "agents have closed cases" in o.text, o.text[:160])
    pid = raw("SELECT oi.Product2Id p FROM \"Case\" c JOIN OrderItem oi ON oi.Id=c.OrderItemId__c WHERE c.Status='Closed' "
              "GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1")[0]["p"]
    o = ex("agent_expertise", {"kind": "product", "id": pid, "top": 3})
    want = raw("SELECT c.OwnerId, COUNT(*) n FROM \"Case\" c JOIN OrderItem oi ON oi.Id=c.OrderItemId__c "
               "WHERE c.Status='Closed' AND oi.Product2Id=? GROUP BY 1 ORDER BY n DESC, 1 LIMIT 3", [pid])
    check("agent_expertise product: matches SQL, top 3", o.ok and [(r["OwnerId"], r["closed_cases"]) for r in rows(o)]
          == [(w["OwnerId"], w["n"]) for w in want], o.text[:160])
    tie = next((i["Id"] for i in rows(ex("list_issues", {}))
                if "TIED" in ex("agent_expertise", {"kind": "issue", "id": i["Id"], "top": 1}).text), None)
    check("agent_expertise: a top tie is flagged for some issue", tie is not None, str(tie))
    o = ex("agent_expertise", {"kind": "team", "id": iss, "top": 1})
    check("agent_expertise: bad kind -> lists kinds", not o.ok and "issue, product" in o.text, o.text)
    o = ex("agent_expertise", {"kind": "issue", "id": "a03Wt00000ZZZZZZZZ", "top": 1})
    check("agent_expertise: unknown issue -> list_issues hint", not o.ok and "list_issues" in o.text, o.text)
    o = ex("agent_expertise", {"kind": "product", "id": "x' OR 1=1 --", "top": 1})
    check("agent_expertise: injection rejected", not o.ok and "15 or 18" in o.text, o.text[:90])
    o = ex("agent_expertise", {"kind": "issue", "id": iss, "top": -2})
    check("agent_expertise: negative top -> error", not o.ok and "between 1 and 20" in o.text, o.text)
    owners = [r["OwnerId"] for r in raw('SELECT DISTINCT OwnerId FROM "Case" ORDER BY OwnerId LIMIT 5')]
    o = ex("agent_workload", {"agent_ids": owners})
    want = {r["OwnerId"]: r["n"] for r in raw("SELECT OwnerId, COUNT(*) n FROM \"Case\" WHERE Status != 'Closed' "
                                              f"AND OwnerId IN ({','.join('?' * 5)}) GROUP BY 1", owners)}
    got = {r["OwnerId"]: r["open_cases"] for r in rows(o)}
    check("agent_workload: matches SQL", o.ok and got == {i: want.get(i, 0) for i in owners}, str(got))
    idle = raw("SELECT Id FROM User WHERE Id NOT IN (SELECT OwnerId FROM \"Case\") LIMIT 1")[0]["Id"]
    o = ex("agent_workload", {"agent_ids": [idle, "005Wt000003ZZZZZZZ"]})
    check("agent_workload: zero-case agent shown as 0, unknown id flagged",
          o.ok and rows(o)[0]["open_cases"] == 0 and "not a User Id: 005Wt000003ZZZZZZZ" in o.text, o.text[:200])
    o = ex("agent_workload", {"agent_ids": []})
    check("agent_workload: empty list -> error", not o.ok and "non-empty" in o.text, o.text)
    o = ex("agent_workload", {"agent_ids": ["x' OR 1=1 --"]})
    check("agent_workload: injection rejected", not o.ok and "15 or 18" in o.text, o.text[:90])
    before = raw('SELECT COUNT(*) n FROM "Case"')[0]["n"]
    check("data unchanged", before == 153)

    # --- oracle re-solves (actions only)
    def run(label: str, t, steps):
        """Run steps(act, say) against a fresh session, print the action trace, and check the answer."""
        sess = env.open_task()
        log: list[str] = []

        def act(name, args):
            o = execute(sess, name, args, "b2b")
            log.append(f"    {name}({args}) -> {'ok' if o.ok else 'ERROR'} rows={o.rows_total}")
            return o

        why: list[str] = []
        answer = steps(act, why.append, sess, t)
        print(f"oracle {label} (idx {t.idx}, {t.task_type}):")
        print("\n".join(log))
        for w in why:
            print(f"    reasoning: {w}")
        sess.close()
        return answer

    # judgments an agent would make from the case text; the actions only supply the data
    JUDGE_ISSUE = {1483: "AI Feature Malfunction", 1489: "Scalability Issue"}
    JUDGE_PRODUCT = {1483: "SecureAnalytics Pro", 1489: "SecureAnalytics Pro"}
    routing = sorted((t for t in tasks if t.split == "dev" and t.task_type == "case_routing"), key=lambda t: t.idx)[:2]

    def route(act, say, sess, t):
        iid = next(i["Id"] for i in rows(act("list_issues", {})) if i["Name"] == JUDGE_ISSUE[t.idx])
        say(f"case text is about issue {JUDGE_ISSUE[t.idx]!r} ({iid})")
        r = rows(act("agent_expertise", {"kind": "issue", "id": iid, "top": 20}))
        top = [x for x in r if x["closed_cases"] == r[0]["closed_cases"]]
        say(f"issue expertise: top count {r[0]['closed_cases']} closed, {len(top)} agent(s) at the top")
        if len(top) == 1:
            return top[0]["OwnerId"]
        pid = next(p["Id"] for p in rows(act("list_products", {})) if p["Name"] == JUDGE_PRODUCT[t.idx])
        say(f"tie on issue; the case names product {JUDGE_PRODUCT[t.idx]!r} ({pid})")
        pr = {x["OwnerId"]: x["closed_cases"] for x in rows(act("agent_expertise", {"kind": "product", "id": pid, "top": 20}))}
        scored = sorted(((pr.get(x["OwnerId"], 0), x["OwnerId"]) for x in top), reverse=True)
        best = [a for c, a in scored if c == scored[0][0]]
        say(f"product expertise among the {len(top)} tied agents (closed, agent): {scored}")
        if len(best) == 1:
            return best[0]
        w = rows(act("agent_workload", {"agent_ids": best}))
        say(f"still tied; open cases: {[(x['OwnerId'], x['open_cases']) for x in w]}")
        return w[0]["OwnerId"]

    for t in routing:
        got = run("case_routing", t, route)
        check(f"oracle case_routing idx {t.idx}: {got} == gt {t.answer[0]}", got == t.answer[0])

    pol = sorted((t for t in tasks if t.split == "dev" and t.task_type == "policy_violation_identification"
                  and t.answer != [None]), key=lambda t: t.idx)[0]

    def policy(act, say, sess, t):
        cid = re.search(r"Case Id to be considered is:\s*(\w+)", t.required).group(1)
        case = act("get_case", {"case_id": cid}).text
        issue = re.search(r"Issue: \w+ \(([^)]+)\)", case).group(1)
        offer = "Workflow Optimization Consultation"
        say(f"case issue is {issue!r}; in the chat the agent offered {offer!r} ({offer in case})")
        hits = rows(act("search_knowledge", {"query": issue, "limit": 5}))
        for h in hits:
            body = act("get_article", {"article_id": h["Id"]}).text
            say(f"candidate {h['Id']} {h['Title']!r}: mentions the offer: {offer in body}")
        about = [h["Id"] for h in hits if "installation" in h["Title"].lower()]
        say(f"judgment: only the article whose title is about installation covers this issue: {about}")
        return about[0] if about else None

    got = run("policy_violation", pol, policy)
    check(f"oracle policy idx {pol.idx}: {got} == gt {pol.answer[0]}", got == pol.answer[0])

    kqs = sorted((t for t in tasks if t.split == "dev" and t.task_type == "knowledge_qa"), key=lambda t: t.idx)
    unans = kqs[0]
    o = ex("search_knowledge", {"query": "HPC", "limit": 5})
    print(f"finding: knowledge_qa idx {unans.idx} asks about {unans.query!r}; search_knowledge('HPC') -> {o.text[:60]!r}")
    check("knowledge_qa idx 688 is unanswerable from the articles (no HPC article)", o.ok and o.rows_total == 0)
    kq = next(t for t in kqs if t.idx == 773)

    def knowledge(act, say, sess, t):
        say("the question is about data privacy compliance in telecom networks; keywords from it")
        hits = rows(act("search_knowledge", {"query": "data privacy compliance telecom networks", "limit": 5}))
        key = set(normalize_answer(t.answer[0]).split())
        best = (-1.0, None, "")
        for h in hits:
            text = act("get_article", {"article_id": h["Id"]}).text
            has_all = key <= set(normalize_answer(text).split())
            sents = [(f1_score(x, t.answer[0]), x) for x in re.split(r"(?<=[.!?])\s+|\n", text)]
            f, sent = max(sents, key=lambda p: p[0])
            say(f"article {h['Id']} {h['Title']!r}: has all key words {has_all}, best sentence F1 {f:.2f}")
            if has_all and f > best[0]:
                best = (f, h["Id"], sent)
        say(f"chosen article has every key word of {t.answer[0]!r}; best sentence F1 {best[0]:.2f}: {best[2][:160]!r}")
        return best[1]

    got = run("knowledge_qa", kq, knowledge)
    gt_arts = [r["Id"] for r in raw("SELECT Id, Title, Summary, FAQ_Answer__c FROM Knowledge__kav")
               if set(normalize_answer(kq.answer[0]).split()) <= set(
                   normalize_answer(f"{r['Title']} {r['Summary']} {r['FAQ_Answer__c']}").split())]
    check(f"oracle knowledge_qa idx {kq.idx}: {got} contains the key words", got in gt_arts, f"articles with all key words: {gt_arts}")
    keyed = [t for t in kqs if len(normalize_answer(t.answer[0]).split()) >= 3]
    n_in = sum(any(set(normalize_answer(t.answer[0]).split()) <= set(normalize_answer(f"{r['Title']} {r['Summary']} {r['FAQ_Answer__c']}").split())
                   for r in raw("SELECT Title, Summary, FAQ_Answer__c FROM Knowledge__kav")) for t in keyed)
    print(f"    corpus coverage: {n_in} of {len(keyed)} dev knowledge_qa answers (3+ words) have all their words in some single article")
    s.close()
    print(f"{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CRM service-desk actions")
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        sys.exit(_selftest())
    ap.print_help()

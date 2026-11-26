"""Phase 11 CRM service-desk actions: purpose-built lookups (case file, knowledge, agent expertise and workload).

  obs = execute(session, "agent_expertise", {"kind": "issue", "id": "a03...", "top": 0}, "b2b")
  uv run python -m crm.service --selftest
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import functools
import html
import re
import sqlite3
import sys
import time

from crm.actions import ActionError, _close, _fail
from crm import policy, retrieval
from crm.env import Observation, TaskSession, render_rows
from crm.prompts import TOOLS, _tool

SERVICE_TYPES = ("case_routing", "policy_violation_identification", "knowledge_qa")
ID_RE = re.compile(r"^[A-Za-z0-9]{15,18}$")
KINDS = ["issue", "product"]
BODY_CAP = 4000  # per email / chat / description; shrinks only if the whole case file exceeds the char cap
DESC_CAP = 120
MAX_SEARCH, DEFAULT_SEARCH, MIN_SEARCH = 10, 5, 3
# Former LOW RELEVANCE flag line, now only measured (tools/flagprobe.py): shown to the model, it was obeyed over
# what the passages said (code-notes: the LOW RELEVANCE flag). Calibrated on crm/kqdev_b2b.json.
FLAG_RELEVANCE = 2.4
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
    if case.get("PurchaseDate"):
        lines.append(f"  Purchased: {str(case['PurchaseDate'])[:10]}, {case['DaysSincePurchase']} days before this "
                     "case was opened")
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


LOGIN_RE = re.compile(r"logged in as Id:\s*(003[A-Za-z0-9]{12,15})")
# What a logged-in customer may use: their own cases and the public knowledge base, nothing internal.
CUSTOMER_TOOLS = ("get_my_cases", "get_case", "search_knowledge", "get_article", "respond")
NOT_YOURS = "No case with Id {cid!r} among your cases. You can only see your own cases (get_my_cases lists them)."


def logged_in_customer(required: str) -> str | None:
    """The Contact Id of the logged-in customer named in the task context, or None (employee mode)."""
    m = LOGIN_RE.search(required or "")
    return m.group(1) if m else None


def get_my_cases(session: TaskSession) -> Observation:
    """The logged-in customer's own cases, newest first."""
    if not session.customer_id:
        raise ActionError("get_my_cases is only available to a logged-in customer")
    return session.query('SELECT c.Id, c.Subject, c.Status, c.CreatedDate, c.ClosedDate, p.Name AS Product '
                         'FROM "Case" c LEFT JOIN OrderItem oi ON oi.Id = c.OrderItemId__c '
                         'LEFT JOIN Product2 p ON p.Id = oi.Product2Id WHERE c.ContactId = ? '
                         'ORDER BY c.CreatedDate DESC', [session.customer_id])


def get_case(session: TaskSession, case_id) -> Observation:
    t0 = time.monotonic()
    cid = _id(case_id, "case_id")
    if session.customer_id:  # a customer sees only their own cases; others look exactly like missing ones
        own = _rows(session, 'SELECT COUNT(*) AS n FROM "Case" WHERE Id = ? AND ContactId = ?',
                    [cid, session.customer_id])[0]["n"]
        if not own:
            raise ActionError(NOT_YOURS.format(cid=cid))
    cases = _rows(session, 'SELECT c.*, i.Name AS IssueName, oi.Product2Id AS Product2Id, p.Name AS ProductName, '
                  'o.EffectiveDate AS PurchaseDate, '
                  'CAST(julianday(c.CreatedDate) - julianday(o.EffectiveDate) AS INTEGER) AS DaysSincePurchase '
                  'FROM "Case" c LEFT JOIN Issue__c i ON i.Id = c.IssueId__c '
                  'LEFT JOIN OrderItem oi ON oi.Id = c.OrderItemId__c LEFT JOIN Product2 p ON p.Id = oi.Product2Id '
                  'LEFT JOIN "Order" o ON o.Id = oi.OrderId WHERE c.Id = ?', [cid])
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
    """Passage search: hybrid candidates, cross-encoder rerank; full passages with scores, or plainly 'nothing relevant'."""
    t0 = time.monotonic()
    # At least 3: a model asking for 1 took the first hit unseen and answered from the wrong article.
    n = max(MIN_SEARCH, _bounded(limit, DEFAULT_SEARCH, MAX_SEARCH, "limit"))
    q = str(query or "")
    if not retrieval.tokenize(q):
        raise ActionError("search query has no searchable words")
    res = retrieval.search(session.org, q, n)
    if res is None:
        raise ActionError("the knowledge base index is unavailable")
    passages = res.passages if res.relevant else retrieval._pick(res.ranked, n)
    if not passages:
        return _obs(retrieval.not_covered_text(res), 0, 0, False, t0)
    notes = []
    missing = _missing_names(session, q)
    if missing:  # a fact, not a score: the reranker cannot tell an absent product from a similar topic
        notes.append("NOTE: no article and no product in the knowledge base mentions "
                     + ", ".join(f'"{m}"' for m in missing) + ". Do not answer with another product's information.")
    if res.mode != "reranked":
        notes.append("(unranked: the relevance model is unavailable; passages may not answer the question)")
    note = "".join(f"{n_}\n" for n_ in notes)
    # best first, without scores: rerank scores order passages but are not a relevance verdict (FLAG_RELEVANCE)
    text, shown, trunc = render_rows(["Id", "Title", "passage"], [(p.Id, p.Title, p.text) for p in passages],
                                     len(passages), session.char_cap - len(note))
    return _obs(note + text, len(passages), shown, trunc, t0)


# Product-like names: a word with an inner capital (ChipEffic) or a hyphenated capitalised word (AI-VeriSim),
# with the capitalised words that follow it. Plain words and all-caps acronyms are not checked.
_NAME_RE = re.compile(r"\b((?:[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*|[A-Z][A-Za-z0-9]*-[A-Za-z0-9-]+)(?:\s+[A-Z][A-Za-z0-9-]*)*)")


def _missing_names(session: TaskSession, query: str) -> list[str]:
    """Product-like names in the query that appear in no knowledge article and no Product2 name."""
    out = []
    for name in dict.fromkeys(m.group(1) for m in _NAME_RE.finditer(query)):
        key = f"%{name.split()[0]}%"
        hits = session.conn.execute(
            "SELECT (SELECT COUNT(*) FROM Knowledge__kav WHERE Title LIKE ?1 OR Summary LIKE ?1 OR FAQ_Answer__c LIKE ?1)"
            " + (SELECT COUNT(*) FROM Product2 WHERE Name LIKE ?1)", (key,)).fetchone()[0]
        if not hits:
            out.append(name)
    return out


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


@functools.lru_cache(maxsize=4)
def _policies(org: str) -> dict:
    return policy.load(org)


def get_issue_policy(session: TaskSession, issue_id) -> Observation:
    """The reviewed policy record for one issue: its guide article, approved fixes and each fix's day limit."""
    t0 = time.monotonic()
    iid = _id(issue_id, "issue_id")
    p = _policies(session.org).get(iid)
    if p is None:
        _exists(session, "Issue__c", iid, "issue", "Use the Issue Id shown in the case file (starts with a03).")
        raise ActionError(f"no policy record for issue {iid!r}; read its guide with search_knowledge and get_article")
    return _obs(policy.describe(p), 1 + len(p.fixes), 1 + len(p.fixes), False, t0)


def check_offers(session: TaskSession, case_id, offered) -> Observation:
    """The policy verdict for the fixes the agent on a case offered: decided by code from the reviewed records."""
    t0 = time.monotonic()
    cid = _id(case_id, "case_id")
    if not isinstance(offered, list):
        raise ActionError("offered must be a list of fix names (an empty list if the agent offered no fix)")
    names = policy.catalog(session.org) + [policy.OTHER]
    bad = [o for o in offered if o not in names]
    if bad:
        raise ActionError(f"unknown fix names {bad}; use only: {names}")
    row = _rows(session, 'SELECT IssueId__c AS iid FROM "Case" WHERE Id = ?', [cid])
    if not row:
        raise ActionError(f"no Case with Id {cid!r}")
    p = _policies(session.org).get(row[0]["iid"])
    if p is None:
        raise ActionError(f"no policy record for this case's issue {row[0]['iid']!r}")
    _, gap = policy.purchase_gap(session.conn, cid)
    answer, lines = policy.decide(p, gap, offered)
    when = "purchase date unknown" if gap is None else f"purchased {gap} days before the case was opened"
    verdict = (f'VERDICT: violation of guide {p.article_id} "{p.title}"' if answer
               else "VERDICT: no violation")
    text = "\n".join([f"CHECK for case {cid} (issue: {p.issue}; {when})", *lines, verdict,
                      f"Answer to give: {answer or 'None'}"])
    return _obs(text, len(lines), len(lines), False, t0)


NO_PRODUCT = "(no product named in the case)"


@functools.lru_cache(maxsize=4)
def _menus(org: str) -> tuple[dict, dict, dict]:
    """(issue name -> Id, issue name -> description, product name -> Id) for the org, read once."""
    from crm.env import CRMEnv
    conn = sqlite3.connect(f"file:{CRMEnv(org).db_path}?mode=ro", uri=True)
    try:
        iss = conn.execute("SELECT Name, Id, Description__c FROM Issue__c ORDER BY Name").fetchall()
        prod = conn.execute("SELECT Name, Id FROM Product2 ORDER BY Name").fetchall()
    finally:
        conn.close()
    return {n: i for n, i, _ in iss}, {n: d for n, _, d in iss}, {n: i for n, i in prod}


def _closed_by(session: TaskSession, kind: str, rid: str, among: list[str] | None = None) -> dict[str, int]:
    """Closed cases per agent for an issue or a product; with `among`, every listed agent appears (0 if none)."""
    if kind == "issue":
        sql = 'SELECT OwnerId, COUNT(*) AS n FROM "Case" WHERE Status = \'Closed\' AND IssueId__c = ? GROUP BY OwnerId'
    else:
        sql = ('SELECT c.OwnerId, COUNT(*) AS n FROM "Case" c JOIN OrderItem oi ON oi.Id = c.OrderItemId__c '
               "WHERE c.Status = 'Closed' AND oi.Product2Id = ? GROUP BY c.OwnerId")
    got = {r["OwnerId"]: r["n"] for r in _rows(session, sql, [rid])}
    return {a: got.get(a, 0) for a in among} if among else got


def _names(session: TaskSession, ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    rows = _rows(session, f"SELECT Id, FirstName || ' ' || LastName AS Name FROM User WHERE Id IN "
                          f"({', '.join('?' * len(ids))})", ids)
    return {r["Id"]: r["Name"] for r in rows}


def route_case(session: TaskSession, issue, product) -> Observation:
    """The case routing policy applied in code to the issue and product the model chose by name."""
    t0 = time.monotonic()
    issue_ids, _, product_ids = _menus(session.org)
    if issue not in issue_ids:
        raise ActionError(f"unknown issue {issue!r}; choose one of: {sorted(issue_ids)}")
    if product != NO_PRODUCT and product not in product_ids:
        raise ActionError(f"unknown product {product!r}; choose one of the product names or {NO_PRODUCT!r}")
    lines = [f"ROUTING for issue '{issue}', product '{product}'"]
    by_issue = _closed_by(session, "issue", issue_ids[issue])
    if not by_issue:
        raise ActionError(f"no agent has closed a case for '{issue}'; choose the next most similar issue")
    top = max(by_issue.values())
    tied = sorted(a for a, n in by_issue.items() if n == top)
    lines.append(f"1. Issue expertise: {len(tied)} agent(s) with the most closed '{issue}' cases ({top} each)")
    decided = "issue expertise"
    if len(tied) > 1 and product != NO_PRODUCT:
        by_prod = _closed_by(session, "product", product_ids[product], tied)
        ptop = max(by_prod.values())
        tied = sorted(a for a, n in by_prod.items() if n == ptop)
        lines.append(f"2. Product expertise among them: {len(tied)} with the most closed '{product}' cases ({ptop})")
        decided = "product expertise"
    elif len(tied) > 1:
        lines.append("2. Product expertise: skipped, the case names no product")
    if len(tied) > 1:
        load = {r["OwnerId"]: r["n"] for r in _rows(
            session, f"SELECT OwnerId, COUNT(*) AS n FROM \"Case\" WHERE Status != 'Closed' AND OwnerId IN "
                     f"({', '.join('?' * len(tied))}) GROUP BY OwnerId", tied)}
        low = min(load.get(a, 0) for a in tied)
        tied = sorted(a for a in tied if load.get(a, 0) == low)
        lines.append(f"3. Workload among them: {len(tied)} with the fewest open cases ({low})")
        decided = "workload"
    winner = tied[0]
    if len(tied) > 1:
        lines.append(f"   still tied after workload; the first by Id is chosen: {winner}")
        decided = "Id order after a full tie"
    name = _names(session, [winner]).get(winner, winner)
    lines.append(f"WINNER: {name} ({winner}), decided by {decided}")
    lines.append(f"Answer to give: {winner}")
    return _obs("\n".join(lines), 1, 1, False, t0)


def list_products(session: TaskSession) -> Observation:
    return session.query("SELECT Id, Name, substr(Description, 1, ?) AS Description FROM Product2 ORDER BY Name",
                         [DESC_CAP])


def _among(among) -> list[str]:
    """Validated, de-duplicated User Ids to rank; empty means all agents."""
    if among in (None, ""):
        return []
    if not isinstance(among, list):
        raise ActionError("among must be a list of User Ids (an empty list means all agents)")
    ids = list(dict.fromkeys(_id(a, "each id in among") for a in among))
    if len(ids) > MAX_AGENTS:
        raise ActionError(f"at most {MAX_AGENTS} ids in among, got {len(ids)}")
    return ids


def _tie_note(kind: str, tied: list[dict]) -> str:
    """The sentence after the ranking: who wins, or the next routing step when the top is tied."""
    ids = ", ".join(r["OwnerId"] for r in tied)
    if len(tied) == 1:
        return f"No tie: {tied[0]['Name'] or tied[0]['OwnerId']} ({tied[0]['OwnerId']}) clearly has the most closed cases."
    if kind == "issue":
        return (f"TIED: {len(tied)} agents share the top count ({ids}). Per the routing policy, next call "
                f"agent_expertise with kind \"product\" for the product this case is about, with among set to "
                f"these {len(tied)} agent ids.")
    return (f"Still TIED among {len(tied)} agents ({ids}). Per the policy, next call agent_workload for these "
            "agents and pick the fewest open cases.")


def agent_expertise(session: TaskSession, kind, rid, top, among=None) -> Observation:
    t0 = time.monotonic()
    if kind not in KINDS:
        raise ActionError(f"kind must be one of {', '.join(KINDS)}, got {kind!r}")
    n = _bounded(top, DEFAULT_TOP, MAX_TOP, "top")
    rid = _id(rid, f"{kind} id")
    only = _among(among)
    if kind == "issue":
        _exists(session, "Issue__c", rid, "issue", "Use list_issues to see the issue Ids.")
        frm, cond = '"Case" c', "c.IssueId__c = ?"
    else:
        _exists(session, "Product2", rid, "product", "Use list_products to see the product Ids.")
        frm = '"Case" c JOIN OrderItem oi ON oi.Id = c.OrderItemId__c'
        cond = "oi.Product2Id = ?"
    marks = f" AND c.OwnerId IN ({', '.join('?' * len(only))})" if only else ""
    allrows = _rows(session, f"SELECT c.OwnerId AS OwnerId, u.FirstName || ' ' || u.LastName AS Name, "
                    f"COUNT(*) AS closed_cases FROM {frm} LEFT JOIN User u ON u.Id = c.OwnerId "
                    f"WHERE c.Status = 'Closed' AND {cond}{marks} GROUP BY c.OwnerId "
                    f"ORDER BY closed_cases DESC, c.OwnerId", [rid, *only])
    unknown: list[str] = []
    if only:  # agents with no closed case for this issue/product still take part, with 0
        have = {r["OwnerId"] for r in allrows}
        users = {r["Id"]: r["Name"] for r in _rows(
            session, f"SELECT Id, FirstName || ' ' || LastName AS Name FROM User WHERE Id IN ({', '.join('?' * len(only))})",
            only)}
        unknown = [i for i in only if i not in users]
        allrows += [{"OwnerId": i, "Name": users[i], "closed_cases": 0} for i in sorted(only) if i in users and i not in have]
    if not allrows:
        why = f"none of the given ids is a User Id: {', '.join(unknown)}" if unknown else \
            f"no agent has closed a case for this {kind}"
        return _obs(f"[] ({why})", 0, 0, False, t0)
    shown = allrows[:n]
    names = ["OwnerId", "Name", "closed_cases"]
    text, k, _ = render_rows(names, [tuple(r[c] for c in names) for r in shown], len(shown), session.char_cap - 600)
    best = allrows[0]["closed_cases"]
    tied = [r for r in allrows if r["closed_cases"] == best]
    head = (f"{len(allrows)} agents ranked among the {len(only) - len(unknown)} given" if only
            else f"{len(allrows)} agents have closed cases for this {kind}")
    text += f"\n[{head}; top count {best} closed cases. {_tie_note(kind, tied)}]"
    if unknown:
        text += f"\n[not a User Id: {', '.join(unknown)}]"
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
    if session.customer_id and name not in CUSTOMER_TOOLS:  # enforced here too, not only by the tool list
        return _fail(f"{name} is not available to customers; customers can use: "
                     f"{', '.join(t for t in CUSTOMER_TOOLS if t != 'respond')}", t0)
    try:
        if name == "get_my_cases":
            return get_my_cases(session)
        if name == "get_case":
            return get_case(session, a.get("case_id"))
        if name == "search_knowledge":
            return search_knowledge(session, a.get("query"), a.get("limit"))
        if name == "get_article":
            return get_article(session, a.get("article_id"))
        if name == "list_issues":
            return list_issues(session)
        if name == "get_issue_policy":
            return get_issue_policy(session, a.get("issue_id"))
        if name == "route_case":
            return route_case(session, a.get("issue"), a.get("product"))
        if name == "check_offers":
            return check_offers(session, a.get("case_id"), a.get("offered"))
        if name == "list_products":
            return list_products(session)
        if name == "agent_expertise":
            return agent_expertise(session, a.get("kind"), a.get("id"), a.get("top"), a.get("among"))
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


# `reason` comes FIRST: the strict schema fixes the generation order, so the reasoning is written before the
# verdict instead of after it (P11-POL correction). Only `answer` is graded.
_RESPOND = _tool("respond", "Give the final answer. Call this once, when you are confident. Write the reason "
                 "first; the answer must follow from it.",
                 {"reason": {"type": "string", "description": "Write this FIRST: the facts that decide the "
                             "answer, citing what you found (for a policy check: each offered fix, whether it "
                             "is approved, and its days against the limit), ending with the conclusion."},
                  "answer": {"type": "string", "description": "ONLY the answer that follows from the reason: "
                             "an Id, None, or one short sentence for a knowledge question."}},
                 ["reason", "answer"])

MAX_QUOTES = 5


def respond_form(customer: bool) -> dict:
    """The checked answer form: the model picks the kind of reply; code writes 'not covered' and refusals."""
    kinds = ["answer", "not_covered", "privacy_refusal"] if customer else ["answer", "not_covered"]
    kind_desc = ("answer: the quotes state the answer. not_covered: nothing you were shown answers it, or the "
                 "source is about a different subject (the reply is written for you)."
                 + (" privacy_refusal: they asked about another person, customer or company, or for internal "
                    "company data (the refusal is written for you)." if customer else ""))
    return _tool(
        "respond", "Give the final answer. Write the reason first, then the subjects, the kind, the quotes and "
        "the answer.",
        {"reason": {"type": "string", "description": "Write this FIRST: which tool result answers the question "
                    "and why it is about what was asked, or why none does."},
         "question_subject": {"type": "string", "description": "The product, industry or context the question asks "
                              "about (for example 'telecom networks', 'SecureAnalytics Pro', 'my cases', or "
                              "'general')."},
         "source_subject": {"type": "string", "description": "What the tool result you would quote is about (for "
                            "example 'EDA software security'). If it differs from question_subject, the kind is "
                            "not_covered."},
         "kind": {"type": "string", "enum": kinds, "description": kind_desc},
         "quotes": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_QUOTES,
                    "description": "For kind answer: each sentence, or each record field such as 'Status: Closed', "
                    "that states part of the answer, copied exactly from a tool result you were shown. Empty "
                    "for the other kinds."},
         "answer": {"type": "string", "description": "For kind answer: one or two short sentences taken from "
                    "the quotes. For the other kinds: empty."}},
        ["reason", "question_subject", "source_subject", "kind", "quotes", "answer"])

ABSTAIN_PHRASES = ("does not cover", "do not cover", "doesn't cover", "not covered", "no information",
                   "no article", "not in our knowledge base", "not in the knowledge base")
MIN_QUOTE_CHARS = 20


REFUSAL_PHRASES = ("can't share", "cannot share", "can not share", "only help with your own", "for privacy")
FALLBACK_KNOWLEDGE = "Our knowledge base does not cover this question."
FALLBACK_CUSTOMER = ("I'm sorry, I couldn't find that in your records or our knowledge base. "
                     "Could you tell me more, or ask about one of your own cases?")
PRIVACY_REFUSAL = ("I'm sorry, I can't share information about other customers or internal company data. "
                   "For privacy, I can only help with your own account. I'm happy to help with your own cases.")


@dataclasses.dataclass(frozen=True)
class Reply:
    kind: str  # answer | not_covered | privacy_refusal
    text: str  # what the user sees: the model's answer, or the text code writes for the other kinds
    quotes: tuple[str, ...]


def read_reply(args: dict, answer: str, customer: bool) -> Reply:
    """The respond form read as a Reply; an old-style form (one 'quote', no 'kind') is read by its wording."""
    kind = args.get("kind")
    if kind not in ("answer", "not_covered", "privacy_refusal") or (kind == "privacy_refusal" and not customer):
        kind = ("privacy_refusal" if customer and is_refusal(answer) else
                "not_covered" if is_abstain(answer) else "answer")
    raw = args.get("quotes")
    quotes = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, list) else []
    if not quotes and args.get("quote"):
        quotes = [args["quote"]]
    quotes = tuple(str(q).strip() for q in quotes if str(q or "").strip())[:MAX_QUOTES]
    if kind == "answer" and not quotes:  # an "answer" that is really a refusal or a "not covered"
        kind = ("privacy_refusal" if customer and is_refusal(answer) else
                "not_covered" if is_abstain(answer) else kind)
    if kind == "privacy_refusal":
        return Reply(kind, PRIVACY_REFUSAL, ())
    if kind == "not_covered":
        subject = str(args.get("question_subject") or "").strip().strip(".")
        if customer:
            return Reply(kind, FALLBACK_CUSTOMER, ())
        known = subject and subject.lower() not in ("general", "none", "n/a", "unknown")
        return Reply(kind, f"Our knowledge base does not cover {subject}." if known else FALLBACK_KNOWLEDGE, ())
    return Reply(kind, answer, quotes)


def is_abstain(answer: str | None) -> bool:
    low = (answer or "").lower()
    return any(p in low for p in ABSTAIN_PHRASES)


def is_refusal(answer: str | None) -> bool:
    """The customer job's privacy refusal, which needs no source sentence."""
    low = (answer or "").lower().replace("’", "'")
    return any(p in low for p in REFUSAL_PHRASES)


def _alnum(s: str) -> str:
    """Letters and digits only, lowercased: survives the escaping and spacing of rendered tool results."""
    # rendered results show a line break as the two characters \n; drop the escape before the letters are kept
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\\[ntr]", " ", html.unescape(s or "")).lower())


def grounding_problem(quote: str | None, observations: list[str], min_chars: int = MIN_QUOTE_CHARS) -> str | None:
    """Why a knowledge answer's quote is not acceptable, or None: it must appear in a passage the agent was shown."""
    q = (quote or "").strip()
    if len(q) < min_chars:
        return "a quote is missing or too short; copy the sentences or record fields that state the answer"
    key = _alnum(q)
    if not any(key in _alnum(o) for o in observations):
        return f'the quote "{q[:80]}" does not appear in any tool result you were shown; copy it exactly'
    return None


SUPPORT_PROMPT = """Question: {question}
Proposed answer: {answer}
{sources}

Does the source state this answer to THIS question? Check two things:
1. Subject: the question asks about a specific product, industry or context (for example telecom networks, IoT
   devices, vehicles, 5G, a named product). The source must be about that same thing. A source about EDA software
   or another product does not answer a question about something else, even if the topic sounds similar. If the
   question names no specific context, skip this check.
2. Facts: the answer's key facts are stated in the source sentences or their passages.
Write one short line of reasoning, then on a new line VERDICT: yes (both checks pass) or VERDICT: no."""
SOURCE_BUDGET = 3000  # characters of passage context in the support check, shared by the quotes


def support_prompt(question: str, answer: str, quotes: tuple[str, ...], observations: list[str]) -> str:
    """The support check's prompt: each quote with the title and passage it came from."""
    per = SOURCE_BUDGET // max(1, len(quotes))
    parts = []
    for i, q in enumerate(quotes, 1):
        title, passage = source_of(q, observations)
        parts.append(f'Source sentence {i}: "{q}"\nIt comes from: {title}\nIts passage: {passage[:per]}')
    return SUPPORT_PROMPT.format(question=question, answer=answer, sources="\n".join(parts))


def source_of(quote: str, observations: list[str]) -> tuple[str, str]:
    """(title, passage text) of the tool result the quote came from; best effort, for the support check."""
    key = _alnum(quote)
    for obs in observations:
        for m in re.finditer(r"'Title': (?:'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"), 'passage': "
                             r"(?:'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\")", obs):
            title, body = (m.group(1) or m.group(2) or ""), (m.group(3) or m.group(4) or "")
            if key and key in _alnum(body):
                return title, body.replace("\\n", " ")[:2000]
        if key and key in _alnum(obs):
            t = re.search(r"Title: (.+)", obs) or re.search(r"CASE \S+: (.+)", obs)
            return (t.group(1).strip() if t else "a tool result"), obs[:2000]
    return "unknown", ""
VERDICT_RE = re.compile(r"VERDICT:\s*(yes|no)", re.I)


def support_verdict(text: str) -> tuple[bool, str]:
    """(supported, the model's reasoning line) from a support-check reply; anything unparseable is 'no'."""
    m = VERDICT_RE.findall(text or "")
    why = (text or "").split("VERDICT")[0].strip()[:300]
    return (bool(m) and m[-1].lower() == "yes"), why


SERVICE_TOOLS: list[dict] = [
    _tool("get_case", "Read one support case in full: its fields (with issue and product names), its emails, live "
          "chat transcripts and change history, each oldest first. Long bodies are cut with a note.",
          {"case_id": _s("The Case Id (starts with 500).")}, ["case_id"]),
    _tool("search_knowledge", "Search the knowledge base. Returns the most relevant passages IN FULL, best first, each "
          "with its article Id and Title (at most 2 passages per article). "
          "If no article is relevant it says so plainly instead of returning weak matches.",
          {"query": _s("Keywords or the question to search for."),
           "limit": {"type": "integer", "description": f"Max passages, 1 to {MAX_SEARCH}; 0 means {DEFAULT_SEARCH}."}},
          ["query", "limit"]),
    _tool("get_article", "Read one knowledge article in full: Title, Summary and the complete FAQ answer.",
          {"article_id": _s("The knowledge article Id (starts with ka0).")}, ["article_id"]),
    _tool("list_issues", "List every issue category (Id, Name, Description) that cases can be filed under.", {}, []),
    _tool("get_issue_policy", "The company policy for one issue: its guide article Id, the fixes approved for that "
          "issue, and how many days after purchase each fix may be offered.",
          {"issue_id": _s("The issue Id from the case file (starts with a03).")}, ["issue_id"]),
    _tool("list_products", "List every product (Id, Name, short Description).", {}, []),
    _tool("agent_expertise", "Rank agents by how many CLOSED cases they have for an issue or a product, most first. "
          "Ties at the top are flagged, with the next step of the routing policy. Set among to a list of agent ids "
          "to rank only those agents (agents with no closed case count 0); an empty list ranks all agents.",
          {"kind": _s("issue or product.", enum=KINDS),
           "id": _s("The issue Id (from list_issues) or product Id (from list_products)."),
           "top": {"type": "integer", "description": f"Max agents, 1 to {MAX_TOP}; 0 means {DEFAULT_TOP}."},
           "among": {"type": "array", "items": {"type": "string"},
                     "description": f"User Ids of the agents to rank, at most {MAX_AGENTS} (for example the agents "
                                    "tied on the issue). An empty list means all agents."}},
          ["kind", "id", "top", "among"]),
    _tool("agent_workload", "For each given agent (User Id), the number of cases whose Status is not Closed; "
          "agents with none show 0.",
          {"agent_ids": {"type": "array", "items": {"type": "string"},
                         "description": f"User Ids of the agents, at most {MAX_AGENTS}."}}, ["agent_ids"]),
    _RESPOND,
]


def _check_offers_tool(org: str) -> dict:
    names = policy.catalog(org) + [policy.OTHER]
    return _tool("check_offers", "Check the fixes the agent on a case offered against company policy. The verdict "
                 "and the answer to give are decided from the policy records: approved for the case's issue, and "
                 "offered within each fix's day limit after purchase.",
                 {"case_id": _s("The Case Id (starts with 500)."),
                  "offered": {"type": "array", "items": {"type": "string", "enum": names},
                              "description": "Every company fix the AGENT offered, promised or arranged, by its "
                                             "name from this list (names may be paraphrased in the conversation). "
                                             f"Use '{policy.OTHER}' for anything else. Empty if no fix was offered."}},
                 ["case_id", "offered"])


def _route_case_tool(org: str) -> dict:
    issue_ids, descs, product_ids = _menus(org)
    menu = "; ".join(f"{n}: {descs[n]}" for n in issue_ids)
    return _tool("route_case", "Route a new case with the company routing policy, applied in code: most closed cases "
                 "for the issue, then (on a tie) for the product, then the fewest open cases. You choose only the "
                 "issue and the product, by name.",
                 {"issue": _s(f"The known issue most similar to the case. The issues: {menu}", enum=list(issue_ids)),
                  "product": _s("The product the case is about (usually named in its subject or description); "
                                f"'{NO_PRODUCT}' only if none is named.", enum=[*product_ids, NO_PRODUCT])},
                 ["issue", "product"])


# The policy and routing jobs get only what they need; the verdict and the winner are decided in code.
JOB_TOOLS = {"policy_violation_identification": ("get_case", "check_offers", "respond"),
             "case_routing": ("route_case", "respond"),
             "knowledge_qa": ("search_knowledge", "get_article", "respond_knowledge")}


_MY_CASES = _tool("get_my_cases", "List YOUR OWN cases (the logged-in customer's): Id, Subject, Status, dates and "
                  "product, newest first.", {}, [])


def service_tools(org: str, task_type: str, customer: str | None = None) -> list[dict]:
    """The tools for one job: a logged-in customer's five, the policy job's three, or the general service list."""
    # a customer's answers are grounded like knowledge answers, so they get the respond form with a quote
    keep = (*CUSTOMER_TOOLS[:-1], "respond_knowledge") if customer else JOB_TOOLS.get(task_type)
    if keep is None:
        return SERVICE_TOOLS
    pool = {t["function"]["name"]: t for t in [*SERVICE_TOOLS, _check_offers_tool(org), _MY_CASES,
                                                _route_case_tool(org)]}
    pool["respond_knowledge"] = respond_form(customer is not None)  # still named "respond" to the model
    return [pool[n] for n in keep]


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
    rows = lambda o: ast.literal_eval(next((ln for ln in o.text.split("\n") if ln.startswith("[")), "[]")) if o.ok else []
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

    # --- check_offers: the code-decided policy verdict
    def offers(case_id, offered):
        return ex("check_offers", {"case_id": case_id, "offered": offered})
    o = offers("500Wt00000DDyznIAD", ["Scalability Enhancement Package"])  # 1381: 367 days, limit 365
    check("check_offers: approved fix past its day limit -> violation, guide Id",
          o.ok and "limit 365 -> violation" in o.text and o.text.endswith("Answer to give: ka0Wt000000EoD3IAK"),
          o.text[-160:])
    o = offers("500Wt00000DDzmCIAT", ["Scalability Enhancement Package"])  # 1459: wrong fix for training access
    check("check_offers: fix not approved for the issue -> violation",
          o.ok and "NOT approved for Training Module Access" in o.text
          and o.text.endswith("Answer to give: ka0Wt000000Eo4zIAC"), o.text[-160:])
    o = offers("500Wt00000DDzcTIAT", ["Comprehensive Training Access"])  # 1426: the paraphrase case
    check("check_offers: 1426 with the paraphrased fix named -> violation",
          o.ok and o.text.endswith("Answer to give: ka0Wt000000Eo0AIAS"), o.text[-120:])
    o = offers("500Wt00000DDzmCIAT", ["Priority Support Upgrade", "Comprehensive Training Access"])
    check("check_offers: approved fixes within limits -> no violation, None",
          o.ok and "VERDICT: no violation" in o.text and o.text.endswith("Answer to give: None"), o.text[-120:])
    o = offers("500Wt00000DDzmCIAT", [])
    check("check_offers: nothing offered -> None", o.ok and o.text.endswith("Answer to give: None"), o.text[-80:])
    o = offers("500Wt00000DDzmCIAT", [policy.OTHER])
    check("check_offers: only troubleshooting -> None", o.ok and o.text.endswith("Answer to give: None"))
    o = offers("500Wt00000DDzmCIAT", ["Free Pizza"])
    check("check_offers: unknown fix name -> error listing the valid names",
          not o.ok and "Priority Support Upgrade" in o.text, o.text[:120])
    o = offers("500Wt00000DDzmCIAT", "Priority Support Upgrade")
    check("check_offers: offered not a list -> error", not o.ok and "must be a list" in o.text, o.text[:100])
    pol_tools = service_tools("b2b", "policy_violation_identification")
    check("policy job gets exactly get_case, check_offers, respond",
          [t["function"]["name"] for t in pol_tools] == ["get_case", "check_offers", "respond"])
    enum = pol_tools[1]["function"]["parameters"]["properties"]["offered"]["items"]["enum"]
    check("check_offers enum: the 8 catalog fixes + other, strict schema",
          len(enum) == 9 and policy.OTHER in enum and pol_tools[1]["function"].get("strict") is True, str(enum))
    check("other jobs (e.g. a typed question) keep the general list, without the job-specific tools",
          service_tools("b2b", "free_question") is SERVICE_TOOLS
          and not {"check_offers", "route_case"} & {t["function"]["name"] for t in SERVICE_TOOLS})

    # --- grounding: knowledge answers must quote a sentence the agent was shown
    shown = ["[{'Id': 'ka0X', 'Title': 'T', 'passage': 'TechPulse\\'s tools use end-to-end encryption to protect "
             "sensitive design data.\\nMore text.'}]"]
    check("grounding: a quote copied from a shown passage passes (escapes and spacing ignored)",
          grounding_problem("TechPulse's tools use end-to-end encryption to protect sensitive design data.", shown)
          is None)
    shown_nl = ["[{'Id': 'k1', 'Title': 'T', 'passage': '4. **Large Batch (Volume: 25)**  \\n   In cases where our "
                "clients opt for 25 units, the timeline is 14 days.'}]"]
    check("grounding: a quote with a real line break matches a passage rendered with \\n (P11-KG4, idx 9002)",
          grounding_problem("4. **Large Batch (Volume: 25)**  \n   In cases where our clients opt for 25 units, "
                            "the timeline is 14 days.", shown_nl) is None)
    check("grounding: an invented quote is refused",
          "does not appear" in (grounding_problem("QuantumPCB Modeler acts as middleware between HPC and quantum.",
                                                  shown) or ""))
    check("grounding: a missing or tiny quote is refused", grounding_problem("", shown) is not None
          and grounding_problem("encryption", shown) is not None)
    check("support verdict: reasoning then VERDICT yes / no / garbage",
          support_verdict("It names the tool.\nVERDICT: yes") == (True, "It names the tool.")
          and support_verdict("Different product.\nVERDICT: no")[0] is False and support_verdict("hmm")[0] is False)
    check("abstain and refusal phrases", is_abstain("Our knowledge base does not cover IDPS.")
          and is_refusal("I'm sorry, I can’t share information about other customers.")
          and not is_abstain("Encryption protects data."))
    kt = service_tools("b2b", "knowledge_qa")
    kprops = kt[-1]["function"]["parameters"]["properties"]
    check("knowledge job: search, article, respond with reason -> subjects -> kind -> quotes -> answer",
          [t["function"]["name"] for t in kt] == ["search_knowledge", "get_article", "respond"]
          and list(kprops) == ["reason", "question_subject", "source_subject", "kind", "quotes", "answer"]
          and kprops["kind"]["enum"] == ["answer", "not_covered"])
    r = read_reply({"kind": "not_covered", "question_subject": "telecom networks", "quotes": ["x"]}, "", False)
    check("reply: not_covered -> code writes the text naming the subject, no quotes",
          r == Reply("not_covered", "Our knowledge base does not cover telecom networks.", ()))
    check("reply: not_covered with subject 'general' -> the generic text",
          read_reply({"kind": "not_covered", "question_subject": "general"}, "", False).text == FALLBACK_KNOWLEDGE)
    check("reply: privacy_refusal for a customer -> the fixed refusal; for staff it is not a kind",
          read_reply({"kind": "privacy_refusal"}, "", True) == Reply("privacy_refusal", PRIVACY_REFUSAL, ())
          and read_reply({"kind": "privacy_refusal"}, "Encryption.", False).kind == "answer")
    r = read_reply({"kind": "answer", "quotes": ["Status: Closed", " ", "Status: New"]}, "One closed, one new.", True)
    check("reply: answer keeps the model's text and its non-empty quotes", r == Reply(
        "answer", "One closed, one new.", ("Status: Closed", "Status: New")))
    check("reply: old form (one quote, no kind) read by its wording",
          read_reply({"quote": "Q."}, "Encryption.", False) == Reply("answer", "Encryption.", ("Q.",))
          and read_reply({}, "Our knowledge base does not cover IDPS.", False).kind == "not_covered"
          and read_reply({}, "I'm sorry, I can't share that.", True).kind == "privacy_refusal")
    title, body = source_of("TechPulse's tools use end-to-end encryption to protect sensitive design data.", shown)
    check("source_of: finds the title and passage the quote came from",
          title == "T" and "end-to-end encryption" in body and "More text" in body, f"{title!r} {body[:60]!r}")
    check("source_of: an unknown quote -> ('unknown', '')", source_of("nothing like it at all here", shown)
          == ("unknown", ""))
    ct = service_tools("b2b", "free_question", "003Wt00000Jqt7DIAR")
    cprops = ct[-1]["function"]["parameters"]["properties"]
    check("customer job: respond carries quotes and the privacy_refusal kind",
          ct[-1]["function"]["name"] == "respond" and "quotes" in cprops
          and cprops["kind"]["enum"] == ["answer", "not_covered", "privacy_refusal"])

    # --- route_case: names in, the routing policy applied in code
    def routed(issue, product):
        o = ex("route_case", {"issue": issue, "product": product})
        m = re.search(r"Answer to give: (\S+)", o.text)
        return o, (m.group(1) if m else None)
    o, w = routed("AI Feature Malfunction", "SecureAnalytics Pro")  # 1483: decided by issue expertise
    check("route_case: single top agent on the issue", o.ok and w == "005Wt000003NIYnIAO"
          and "decided by issue expertise" in o.text, o.text[-90:])
    o, w = routed("Scalability Issue", "SecureAnalytics Pro")  # 1489: 7 tied, product breaks it
    check("route_case: issue tie broken by product (1489)", o.ok and w == "005Wt000003NJD9IAO"
          and "decided by product expertise" in o.text, o.text[-90:])
    o, w = routed("Workflow Integration Lag", "NextGen IDE")  # 1576
    check("route_case: 1576 with the product its description names", w == "005Wt000003NIk5IAG", str(w))
    o, w = routed("Workflow Integration Lag", "OptiPower Manager")
    check("route_case: product tie broken by workload", o.ok and "decided by workload" in o.text, o.text[-90:])
    o, w = routed("Workflow Integration Lag", NO_PRODUCT)
    check("route_case: no product named -> product step skipped, workload decides",
          o.ok and "skipped, the case names no product" in o.text and w is not None, o.text[-120:])
    o, _ = routed("Printer On Fire", "NextGen IDE")
    check("route_case: unknown issue name -> error listing the names", not o.ok and "AI Feature Malfunction" in o.text)
    o, _ = routed("Workflow Integration Lag", "a03Wt00000JqzR3IAJ")
    check("route_case: an Id instead of a product name -> error", not o.ok and "unknown product" in o.text)
    rt = service_tools("b2b", "case_routing")
    props = rt[0]["function"]["parameters"]["properties"]
    check("routing job gets exactly route_case and respond; menus are strict enums of names",
          [t["function"]["name"] for t in rt] == ["route_case", "respond"] and len(props["issue"]["enum"]) == 15
          and len(props["product"]["enum"]) == 52 and rt[0]["function"].get("strict") is True)

    # --- customer scope: a logged-in customer sees only their own cases (Kofi Nkrumah, 4 cases)
    kofi, other_case = "003Wt00000Jqt7DIAR", "500Wt00000DDyznIAD"  # 1381's case belongs to someone else
    check("logged_in_customer: parses the benchmark's login line; None without one",
          logged_in_customer("The customer you are interacting with is logged in as Id: 003Wt00000Jqy3aIAB.")
          == "003Wt00000Jqy3aIAB" and logged_in_customer("Today's date: 2023-11-24") is None)
    cs = env.open_task()
    cs.customer_id = kofi
    cx = lambda name, args: execute(cs, name, args, "b2b")
    mine = rows(cx("get_my_cases", {}))
    own_ok = all(raw('SELECT ContactId FROM "Case" WHERE Id = ?', [r["Id"]])[0]["ContactId"] == kofi for r in mine)
    check("customer: get_my_cases lists exactly their 4 cases", len(mine) == 4 and own_ok, str([r["Id"] for r in mine]))
    o = cx("get_case", {"case_id": mine[0]["Id"]})
    check("customer: get_case on their own case works", o.ok and mine[0]["Id"] in o.text, o.text[:80])
    o = cx("get_case", {"case_id": other_case})
    o2 = cx("get_case", {"case_id": "500Wt00000ZZZZZZZZ"})
    check("customer: another customer's case refused, same message as a missing case, no data",
          not o.ok and not o2.ok and o.text.replace(other_case, "X") == o2.text.replace("500Wt00000ZZZZZZZZ", "X")
          and "among your cases" in o.text, o.text[:120])
    for name, args in (("agent_expertise", {"kind": "issue", "id": "a03Wt00000JqhItIAJ", "top": 3, "among": []}),
                       ("agent_workload", {"agent_ids": ["005Wt000003NIYnIAO"]}), ("list_issues", {}),
                       ("check_offers", {"case_id": other_case, "offered": []}),
                       ("get_issue_policy", {"issue_id": "a03Wt00000JqhItIAJ"})):
        o = cx(name, args)
        check(f"customer: internal action {name} refused at the service layer",
              not o.ok and "not available to customers" in o.text, o.text[:90])
    o = cx("search_knowledge", {"query": "return policy for purchased items", "limit": 3})
    check("customer: knowledge search still works", o.ok and o.rows_total >= 3)
    cs.close()
    check("customer tool list: exactly the five customer tools",
          [t["function"]["name"] for t in service_tools("b2b", "private_customer_information", kofi)]
          == list(CUSTOMER_TOOLS))
    check("knowledge task with no login gets the knowledge job's tools, not the customer's",
          [t["function"]["name"] for t in service_tools("b2b", "knowledge_qa", None)]
          == ["search_knowledge", "get_article", "respond"])
    o = ex("get_case", {"case_id": other_case})
    check("employee mode (no login): any case still opens", o.ok and other_case in o.text)
    o = ex("get_my_cases", {})
    check("employee mode: get_my_cases refused", not o.ok and "logged-in customer" in o.text, o.text[:80])

    # --- get_issue_policy (case 1381: scalability, 367 whole days after purchase, limit 365)
    o = ex("get_case", {"case_id": "500Wt00000DDyznIAD"})
    check("get_case: purchase date and days since purchase", o.ok and "Purchased: 2021-09-20, " in o.text
          and "367 days before this case was opened" in o.text, next((ln for ln in o.text.split("\n")
                                                                       if "Purchased" in ln), "no Purchased line"))
    o = ex("get_issue_policy", {"issue_id": "a03Wt00000JqxtvIAB"})
    check("get_issue_policy: scalability guide, package limited to 365 days", o.ok
          and "ka0Wt000000EoD3IAK" in o.text and "Scalability Enhancement Package: only up to 365 days" in o.text,
          o.text[:160])
    o = ex("get_issue_policy", {"issue_id": "a03Wt00000ZZZZZZZZ"})
    check("get_issue_policy: unknown issue -> error naming the Id format", not o.ok and "a03" in o.text, o.text[:120])
    check("get_issue_policy: all 15 issues have a record with at least one fix",
          len(_policies("b2b")) == 15 and all(p.fixes for p in _policies("b2b").values()))

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
    norm = lambda x: " ".join(str(x).split())
    o = ex("search_knowledge", {"query": "extended technical support offer duration", "limit": 0})
    rs = rows(o)
    arts = {r["Id"]: norm(raw("SELECT Summary || ' ' || FAQ_Answer__c AS t FROM Knowledge__kav WHERE Id = ?", [r["Id"]])[0]["t"])
            for r in rs}
    best = retrieval.search("b2b", "extended technical support offer duration", 5)
    check("search_knowledge: default 5 passages, keys Id/Title/passage (no score), <= 2 per article, best first",
          o.ok and o.rows_total == 5 and all(set(r) == {"Id", "Title", "passage"} for r in rs)
          and max(collections.Counter(r["Id"] for r in rs).values()) <= 2
          and [r["Id"] for r in rs] == [p.Id for p in best.passages], o.text[:140])
    check("search_knowledge: every passage is shown in full (a whole stretch of its article, > 200 chars)",
          all(norm(r["passage"]) in arts[r["Id"]] and len(r["passage"]) > 200 for r in rs),
          f"passage lengths {[len(r['passage']) for r in rs]}")
    o = ex("search_knowledge", {"query": "TechPulse EDA software installation support", "limit": 99})
    check("search_knowledge: limit clamped to 10", o.ok and o.rows_total == 10, f"{o.rows_total}")
    o = ex("search_knowledge", {"query": "x' OR 1=1 --", "limit": 3})
    check("search_knowledge: injection text is just words", o.ok or "searchable" in o.text, o.text[:100])
    o = ex("search_knowledge", {"query": "!!!", "limit": 3})
    check("search_knowledge: no words -> error", not o.ok and "no searchable words" in o.text, o.text)
    o = ex("search_knowledge", {"query": "billing", "limit": -1})
    check("search_knowledge: negative limit -> error", not o.ok and "between 1 and 10" in o.text, o.text)
    o = ex("search_knowledge", {"query": "return policy for purchased items", "limit": 1})
    check("search_knowledge: limit 1 still returns at least 3 passages", o.ok and o.rows_total == 3, f"{o.rows_total}")
    small = CRMEnv("b2b", char_cap=3000).open_task()
    o = execute(small, "search_knowledge", {"query": "extended technical support offer duration", "limit": 10}, "b2b")
    check("search_knowledge: char cap (3000) is a safety net, whole rows only", o.ok and len(o.text) <= 3000
          and o.truncated and o.rows_shown < o.rows_total, f"{len(o.text)} chars, shown {o.rows_shown} of {o.rows_total}")
    small.close()
    nq = next(t for t in tasks if t.idx == 688).query
    o = ex("search_knowledge", {"query": nq, "limit": 5})
    check("search_knowledge: low-scoring search (idx 688) -> passages shown, no score and no relevance flag",
          o.ok and o.rows_total >= 3 and "LOW RELEVANCE" not in o.text and "'score'" not in o.text, o.text[:200])
    o = ex("search_knowledge", {"query": "ChipEffic Pro compliance standards", "limit": 3})
    check("search_knowledge: absent product -> NOTE names it", o.ok and o.text.startswith(
        'NOTE: no article and no product in the knowledge base mentions "ChipEffic Pro"'), o.text[:160])
    o = ex("search_knowledge", {"query": "SecureAnalytics Pro scalability", "limit": 3})
    check("search_knowledge: a real product -> no NOTE", o.ok and not o.text.startswith("NOTE"), o.text[:120])
    aid = rows(ex("search_knowledge", {"query": "return policy for purchased items", "limit": 1}))[0]["Id"]
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
    # --- agent_expertise among + tie notes
    tied_issue = next((i["Id"] for i in rows(ex("list_issues", {}))
                       if "TIED:" in ex("agent_expertise", {"kind": "issue", "id": i["Id"], "top": 1, "among": []}).text), None)
    o = ex("agent_expertise", {"kind": "issue", "id": tied_issue, "top": 20, "among": []})
    ids = [r["OwnerId"] for r in rows(o) if r["closed_cases"] == rows(o)[0]["closed_cases"]]
    want = (f"TIED: {len(ids)} agents share the top count ({', '.join(ids)}). Per the routing policy, next call "
            f'agent_expertise with kind "product" for the product this case is about, with among set to these '
            f"{len(ids)} agent ids.")
    check("agent_expertise issue tie: note names the product step", o.ok and want in o.text, o.text[-330:])
    prod = raw("SELECT oi.Product2Id p, c.OwnerId a, COUNT(*) n FROM \"Case\" c JOIN OrderItem oi ON oi.Id=c.OrderItemId__c "
               "WHERE c.Status='Closed' GROUP BY 1, 2 ORDER BY n DESC, a")
    pid2 = prod[0]["p"]
    by_prod = [r for r in prod if r["p"] == pid2]
    others = [r["Id"] for r in raw("SELECT Id FROM User WHERE Id NOT IN (SELECT c.OwnerId FROM \"Case\" c JOIN OrderItem oi "
                                   "ON oi.Id=c.OrderItemId__c WHERE c.Status='Closed' AND oi.Product2Id=?) ORDER BY Id LIMIT 2", [pid2])]
    pick = [by_prod[-1]["a"], *others]
    o = ex("agent_expertise", {"kind": "product", "id": pid2, "top": 20, "among": pick})
    got = {r["OwnerId"]: r["closed_cases"] for r in rows(o)}
    check("agent_expertise among: only the given agents, zero-count agents included as 0",
          o.ok and set(got) == set(pick) and got[by_prod[-1]["a"]] == by_prod[-1]["n"]
          and all(got[i] == 0 for i in others) and "ranked among the 3 given" in o.text, str(got))
    o = ex("agent_expertise", {"kind": "product", "id": pid2, "top": 20, "among": []})
    check("agent_expertise among empty list: all agents with closed cases", o.ok
          and {r["OwnerId"] for r in rows(o)} == {r["a"] for r in by_prod}, o.text[-120:])
    top2 = [r["a"] for r in by_prod if r["n"] == by_prod[0]["n"]]
    if len(top2) > 1:
        o = ex("agent_expertise", {"kind": "product", "id": pid2, "top": 20, "among": top2})
        want = (f"Still TIED among {len(top2)} agents ({', '.join(sorted(top2))}). Per the policy, next call "
                "agent_workload for these agents and pick the fewest open cases.")
        check("agent_expertise product tie: note names agent_workload", o.ok and want in o.text, o.text[-250:])
    else:  # no natural product tie in the data: a pair of agents with 0 closed cases is tied at 0
        z = others[:2]
        o = ex("agent_expertise", {"kind": "product", "id": pid2, "top": 20, "among": z})
        check("agent_expertise product tie: note names agent_workload", o.ok and "Still TIED among 2 agents" in o.text
              and "next call agent_workload" in o.text and "fewest open cases" in o.text, o.text[-250:])
    o = ex("agent_expertise", {"kind": "product", "id": pid2, "top": 20, "among": [by_prod[0]["a"], others[0]]})
    check("agent_expertise single winner: note says so", o.ok and "No tie:" in o.text and "TIED" not in o.text, o.text[-200:])
    o = ex("agent_expertise", {"kind": "issue", "id": iss, "top": 5, "among": ["x' OR 1=1 --"]})
    check("agent_expertise among: injection rejected", not o.ok and "15 or 18" in o.text, o.text[:90])
    o = ex("agent_expertise", {"kind": "issue", "id": iss, "top": 5, "among": "005Wt000003ZZZZZZZ"})
    check("agent_expertise among: not a list -> error", not o.ok and "list of User Ids" in o.text, o.text[:90])
    o = ex("agent_expertise", {"kind": "issue", "id": iss, "top": 5, "among": ["005Wt000003ZZZZZZZ"]})
    check("agent_expertise among: unknown User Id flagged", o.ok and o.rows_total == 0 and "is a User Id: 005Wt000003ZZZZZZZ" in o.text, o.text[:120])
    props = next(t for t in SERVICE_TOOLS if t["function"]["name"] == "agent_expertise")["function"]["parameters"]
    check("agent_expertise schema: among is a required array of strings",
          props["properties"]["among"]["type"] == "array" and "among" in props["required"])
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
        r = rows(act("agent_expertise", {"kind": "issue", "id": iid, "top": 20, "among": []}))
        top = [x for x in r if x["closed_cases"] == r[0]["closed_cases"]]
        say(f"issue expertise: top count {r[0]['closed_cases']} closed, {len(top)} agent(s) at the top")
        if len(top) == 1:
            return top[0]["OwnerId"]
        pid = next(p["Id"] for p in rows(act("list_products", {})) if p["Name"] == JUDGE_PRODUCT[t.idx])
        say(f"tie on issue; the case names product {JUDGE_PRODUCT[t.idx]!r} ({pid})")
        pr = rows(act("agent_expertise", {"kind": "product", "id": pid, "top": 20, "among": [x["OwnerId"] for x in top]}))
        scored = sorted(((x["closed_cases"], x["OwnerId"]) for x in pr), reverse=True)
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

    def policy_oracle(act, say, sess, t):
        cid = re.search(r"Case Id to be considered is:\s*(\w+)", t.required).group(1)
        case = act("get_case", {"case_id": cid}).text
        issue = re.search(r"Issue: \w+ \(([^)]+)\)", case).group(1)
        offer = "Workflow Optimization Consultation"
        say(f"case issue is {issue!r}; in the chat the agent offered {offer!r} ({offer in case})")
        hits = list({h["Id"]: h for h in rows(act("search_knowledge", {"query": issue, "limit": 5}))}.values())
        for h in hits:
            body = act("get_article", {"article_id": h["Id"]}).text
            say(f"candidate {h['Id']} {h['Title']!r}: mentions the offer: {offer in body}")
        about = [h["Id"] for h in hits if "installation" in h["Title"].lower()]
        say(f"judgment: only the article whose title is about installation covers this issue: {about}")
        return about[0] if about else None

    got = run("policy_violation", pol, policy_oracle)
    check(f"oracle policy idx {pol.idx}: {got} == gt {pol.answer[0]}", got == pol.answer[0])

    kqs = sorted((t for t in tasks if t.split == "dev" and t.task_type == "knowledge_qa"), key=lambda t: t.idx)
    unans = kqs[0]
    o = ex("search_knowledge", {"query": "HPC", "limit": 5})
    print(f"finding: knowledge_qa idx {unans.idx} asks about {unans.query!r}; search_knowledge('HPC') -> {o.text[:60]!r}")
    check("knowledge_qa idx 688 is unanswerable from the articles (no HPC article)",
          o.ok and not any("hpc" in str(r["Title"]).lower() for r in rows(o)), f"{o.rows_total} weak matches")
    kq = next(t for t in kqs if t.idx == 773)

    def knowledge(act, say, sess, t):
        say("the question is about data privacy compliance in telecom networks; keywords from it")
        hits = list({h["Id"]: h for h in rows(act("search_knowledge", {"query": "data privacy compliance telecom networks", "limit": 5}))}.values())
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

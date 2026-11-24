"""Phase 11 CRM actions: Agentforce-style standard actions an agent calls instead of writing raw SQL.

  obs = execute(session, "summarize_records", {"object": "Case", ...}, "b2b")   # None for respond
  uv run python -m crm.actions --selftest
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import difflib
import functools
import re
import sqlite3
import sys
import time

from crm.env import DATA, Observation, TaskSession, render_rows
from crm.prompts import TOOLS, _tool

_REF_SPEC = """
Contact.AccountId->Account; Case.ContactId->Contact; Case.OrderItemId__c->OrderItem; Case.IssueId__c->Issue__c;
Case.AccountId->Account; Case.OwnerId->User; ProductCategoryProduct.ProductCategoryId->ProductCategory;
ProductCategoryProduct.ProductId->Product2; PricebookEntry.Pricebook2Id->Pricebook2; PricebookEntry.Product2Id->Product2;
Order.AccountId->Account; Order.Pricebook2Id->Pricebook2; Order.OwnerId->User; OrderItem.OrderId->Order;
OrderItem.Product2Id->Product2; OrderItem.PriceBookEntryId->PricebookEntry; EmailMessage.ParentId->Case;
LiveChatTranscript.CaseId->Case; LiveChatTranscript.AccountId->Account; LiveChatTranscript.OwnerId->User;
LiveChatTranscript.ContactId->Contact; CaseHistory__c.CaseId__c->Case; Opportunity.ContractID__c->Contract;
Opportunity.AccountId->Account; Opportunity.ContactId->Contact; Opportunity.OwnerId->User;
OpportunityLineItem.OpportunityId->Opportunity; OpportunityLineItem.Product2Id->Product2;
OpportunityLineItem.PricebookEntryId->PricebookEntry; Quote.OpportunityId->Opportunity; Quote.AccountId->Account;
Quote.ContactId->Contact; QuoteLineItem.QuoteId->Quote; QuoteLineItem.OpportunityLineItemId->OpportunityLineItem;
QuoteLineItem.Product2Id->Product2; QuoteLineItem.PricebookEntryId->PricebookEntry; Contract.AccountId->Account;
VoiceCallTranscript__c.OpportunityId__c->Opportunity; VoiceCallTranscript__c.LeadId__c->Lead;
Task.WhatId->Opportunity; Task.OwnerId->User; Event.WhatId->Opportunity; Event.OwnerId->User;
UserTerritory2Association.UserId->User; UserTerritory2Association.Territory2Id->Territory2;
Lead.ConvertedContactId->Contact; Lead.ConvertedAccountId->Account; Lead.OwnerId->User
"""


def _parse_refs(spec: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for item in filter(None, (s.strip() for s in spec.split(";"))):
        src, target = item.split("->")
        obj, field = src.split(".")
        out.setdefault(obj, {})[field] = target
    return out


REFERENCES = _parse_refs(_REF_SPEC)
NAME_FIELDS = {"Name", "Title", "Subject", "Company"}
FILTER_OPS = ["=", "!=", ">", ">=", "<", "<=", "in", "not_in", "like", "is_null", "not_null"]
COMPARE_OPS = {"=", "!=", ">", ">=", "<", "<="}
METRICS = ["count", "count_distinct", "sum", "avg", "min", "max", "avg_days_between", "sum_product"]
HAVING_OPS = ["none", "=", "!=", ">", ">=", "<", "<="]
DEFAULT_LIMIT, FIND_LIMIT, SEARCH_LIMIT = 50, 20, 20
LIST_MAX = 50  # query_records lists at most this many rows; 200-row lists filled a 32k context in 4 calls
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
MONTH_RE = re.compile(r"^month\((.+)\)$", re.IGNORECASE)
PATH_RE = re.compile(r"^\w+(\.\w+)?$")


class ActionError(Exception):
    """A user-facing validation error; becomes an ok=False Observation."""


def _fail(msg: str, t0: float) -> Observation:
    return Observation(False, f"Error: {msg}", None, None, False, (time.monotonic() - t0) * 1000, msg)


# ---------------------------------------------------------------- catalog

_COLUMNS: dict[str, dict[str, list[str]]] = {}


def _columns(org: str, conn: sqlite3.Connection | None = None) -> dict[str, list[str]]:
    """{table: [column, ...]} for the org's user tables, from PRAGMA table_info; cached per org."""
    if org in _COLUMNS:
        return _COLUMNS[org]
    own = conn is None
    if own:
        conn = sqlite3.connect(f"file:{DATA / 'derived' / f'{org}.db'}?mode=ro", uri=True)
    try:
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        cols = {t: [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]
                for t in names if not t.startswith(("_", "sqlite_"))}
    finally:
        if own:
            conn.close()
    _COLUMNS[org] = cols
    return cols


@functools.lru_cache(maxsize=None)
def _fts(org: str) -> dict[str, frozenset[str]]:
    """{object: fields indexed in _fts_search} for the org (read from the derived DB file)."""
    conn = sqlite3.connect(f"file:{DATA / 'derived' / f'{org}.db'}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT DISTINCT object, field FROM _fts_search").fetchall()
    finally:
        conn.close()
    out: dict[str, set[str]] = {}
    for obj, field in rows:
        out.setdefault(obj, set()).add(field)
    return {k: frozenset(v) for k, v in sorted(out.items())}


def _name_objects(org: str) -> list[str]:
    return [o for o, fs in _fts(org).items() if fs & NAME_FIELDS]


def _close(name: str, options: list[str]) -> str:
    low = {o.lower(): o for o in options}
    hits = difflib.get_close_matches(name.lower(), list(low), n=5)
    return ", ".join(low[h] for h in hits) or "none"


def _resolve(name: str, options: list[str]) -> str | None:
    """The canonical spelling of name among options (exact, then case-insensitive), else None."""
    if name in options:
        return name
    return next((o for o in options if o.lower() == name.lower()), None)


def _object(cols: dict[str, list[str]], name) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ActionError(f"object must be a non-empty string; objects: {', '.join(cols)}")
    t = _resolve(name.strip(), list(cols))
    if t is None:
        raise ActionError(f"unknown object {name!r}. Close matches: {_close(name, list(cols))}. "
                          f"Objects: {', '.join(cols)}")
    return t


def _field(cols: dict[str, list[str]], table: str, name: str) -> str:
    c = _resolve(name, cols[table])
    if c is None:
        raise ActionError(f"unknown field {name!r} on {table}. Close matches: {_close(name, cols[table])}. "
                          f"{table} fields: {', '.join(cols[table])}")
    return c


def references(org: str, conn: sqlite3.Connection | None = None) -> dict[str, dict[str, str]]:
    """REFERENCES restricted to columns and targets that exist in the org's DB, with canonical column names."""
    cols = _columns(org, conn)
    out: dict[str, dict[str, str]] = {}
    for obj, refs in REFERENCES.items():
        if obj not in cols:
            continue
        for field, target in refs.items():
            c = _resolve(field, cols[obj])
            if c is not None and target in cols and "Id" in cols[target]:
                out.setdefault(obj, {})[c] = target
    return out


def references_text(org: str) -> str:
    """Compact reference map for the system prompt: one line per object."""
    lines = [f"- {obj}: " + ", ".join(f"{f} -> {t}" for f, t in refs.items())
             for obj, refs in references(org).items()]
    return "\n".join(lines)


# ---------------------------------------------------------------- SQL building

class _Query:
    """Builds the FROM clause, column expressions and WHERE clause for one object, joining per reference used."""

    def __init__(self, org: str, conn: sqlite3.Connection, obj) -> None:
        self.cols = _columns(org, conn)
        self.refs = references(org, conn)
        self.table = _object(self.cols, obj)
        self.joins: dict[str, tuple[str, str]] = {}  # reference column -> (alias, target)

    def expr(self, path) -> tuple[str, str]:
        """(SQL expression, canonical path) for 'Field' or 'RefField.ParentField'."""
        if not isinstance(path, str) or not PATH_RE.match(path.strip()):
            raise ActionError(f"bad field path {path!r}: use Field or RefField.ParentField (one hop)")
        parts = path.strip().split(".")
        if len(parts) == 1:
            c = _field(self.cols, self.table, parts[0])
            return f't0."{c}"', c
        ref = _field(self.cols, self.table, parts[0])
        target = self.refs.get(self.table, {}).get(ref)
        if target is None:
            have = ", ".join(f"{f} -> {t}" for f, t in self.refs.get(self.table, {}).items()) or "none"
            raise ActionError(f"{self.table}.{ref} is not a reference field; references from {self.table}: {have}")
        c = _field(self.cols, target, parts[1])
        alias = self.joins.setdefault(ref, (f"j{len(self.joins) + 1}", target))[0]
        return f'{alias}."{c}"', f"{ref}.{c}"

    def from_sql(self) -> str:
        return f'"{self.table}" AS t0' + "".join(
            f' LEFT JOIN "{target}" AS {alias} ON {alias}."Id" = t0."{ref}"'
            for ref, (alias, target) in self.joins.items())

    def where(self, filters) -> tuple[list[str], list]:
        """AND-ed conditions and their bound parameters."""
        if filters in (None, ""):
            filters = []
        if not isinstance(filters, list):
            raise ActionError("filters must be a list of {field, op, value} objects")
        conds, params = [], []
        for f in filters:
            if not isinstance(f, dict) or "field" not in f or "op" not in f:
                raise ActionError(f"bad filter {f!r}: need field, op and value")
            op, value = f["op"], f.get("value")
            value = "" if value is None else str(value)
            if op not in FILTER_OPS:
                raise ActionError(f"unknown op {op!r}; ops: {', '.join(FILTER_OPS)}")
            e, _ = self.expr(f["field"])
            if op in ("is_null", "not_null"):
                conds.append(f"{e} IS {'NOT ' if op == 'not_null' else ''}NULL")
            elif op in ("in", "not_in"):
                vals = [v.strip() for v in value.split(",") if v.strip()]
                if not vals:
                    raise ActionError(f"op {op} needs a comma-separated value list")
                conds.append(f"{e} {'NOT ' if op == 'not_in' else ''}IN ({', '.join('?' * len(vals))})")
                params.extend(vals)
            elif op == "like":
                conds.append(f"lower({e}) LIKE lower(?)")
                params.append(value if "%" in value else f"%{value}%")
            elif DATE_RE.match(value.strip()):
                conds.append(f"substr({e}, 1, 10) {op} ?")
                params.append(value.strip())
            elif NUM_RE.match(value.strip()):
                # numbers are stored as TEXT in some columns (OrderItem.Quantity), so compare numerically
                conds.append(f"CAST({e} AS REAL) {op} ?")
                params.append(float(value))
            else:
                conds.append(f"{e} {op} ?")
                params.append(value)
        return conds, params


def _limit(v) -> int:
    n = int(v or 0)
    return DEFAULT_LIMIT if n <= 0 else min(n, 200)


def _num(e: str) -> str:
    """e, with digits-and-dots text cast to REAL so MIN/MAX/ORDER BY on TEXT number columns are numeric."""
    return (f"(CASE WHEN typeof({e}) = 'text' AND {e} != '' AND NOT {e} GLOB '*[^0-9.]*' "
            f"THEN CAST({e} AS REAL) ELSE {e} END)")


def _where_sql(conds: list[str]) -> str:
    return f" WHERE {' AND '.join(conds)}" if conds else ""


# ---------------------------------------------------------------- actions

def _parse_rows(text: str) -> list[dict]:
    """Rows back from search()'s rendered text (drops the OR-fallback note and the truncation footer)."""
    lines = text.split("\n")
    if lines and lines[0].startswith("("):  # a leading note from search(): OR fallback or top-up
        lines = lines[1:]
    if lines and lines[-1].startswith("[showing "):
        lines = lines[:-1]
    return ast.literal_eval("\n".join(lines))


def find_by_name(session: TaskSession, org: str, name, obj) -> Observation:
    t0 = time.monotonic()
    objs = _name_objects(org)
    if obj in (None, "", "any"):
        chosen = objs
    else:
        o = _resolve(str(obj), objs)
        if o is None:
            raise ActionError(f"object {obj!r} has no indexed name field. Close matches: {_close(str(obj), objs)}. "
                              f"Objects: any, {', '.join(objs)}")
        chosen = [o]
    obs = session.search(str(name or ""), chosen, limit=session.row_cap)
    if not obs.ok:
        return obs
    note = "(no record matched all words; showing records matching any word)\n" \
        if obs.text.startswith("(no record matched all words") else ""
    keep = [r for r in _parse_rows(obs.text) if r["field"] in NAME_FIELDS]
    names = ["object", "Id", "field", "snippet"]
    text, shown, trunc = render_rows(names, [tuple(r[k] for k in names) for r in keep[:FIND_LIMIT]],
                                     len(keep), session.char_cap - len(note))
    return Observation(True, note + text, len(keep), shown, trunc, (time.monotonic() - t0) * 1000, None)


def get_record(session: TaskSession, org: str, obj, rid) -> Observation:
    table = _object(_columns(org, session.conn), obj)
    return session.query(f'SELECT * FROM "{table}" WHERE "Id" = ?', [str(rid or "").strip()])


def query_records(session: TaskSession, org: str, obj, filters, fields, order_by, descending, limit) -> Observation:
    q = _Query(org, session.conn, obj)
    if fields in (None, ""):
        fields = []
    if not isinstance(fields, list):
        raise ActionError("fields must be a list of field paths (empty list = all columns)")
    sel = ", ".join(f'{e} AS "{p}"' for e, p in (q.expr(f) for f in fields)) or "t0.*"
    conds, params = q.where(filters)
    order = ""
    if order_by:
        e, _ = q.expr(order_by)
        order = f" ORDER BY {e} IS NULL, {_num(e)} {'DESC' if descending else 'ASC'}"
    sql = f"SELECT {sel} FROM {q.from_sql()}{_where_sql(conds)}{order} LIMIT ?"
    obs = session.query(sql, params + [min(_limit(limit), LIST_MAX)])
    if not obs.ok:
        return obs
    total = session.conn.execute(f"SELECT COUNT(*) FROM {q.from_sql()}{_where_sql(conds)}", params).fetchone()[0]
    if total > (obs.rows_shown or 0):  # tell the model how much it is not seeing, and what to use instead
        lines = [ln for ln in obs.text.split("\n") if not ln.startswith("[showing ")]
        lines.append(f"[showing {obs.rows_shown} of {total} rows. Narrow the filters, or use summarize_records "
                     f"for counts, rates, totals and averages instead of listing rows]")
        obs = dataclasses.replace(obs, text="\n".join(lines), rows_total=total, truncated=True)
    return obs


def summarize_records(session: TaskSession, org: str, obj, filters, group_by, metric, field, field2,
                      having_op, having_value, sort, limit) -> Observation:
    q = _Query(org, session.conn, obj)
    if metric not in METRICS:
        raise ActionError(f"unknown metric {metric!r}; metrics: {', '.join(METRICS)}")
    conds, params = q.where(filters)
    if metric == "count":
        m = "COUNT(*)"
    else:
        if not field:
            raise ActionError(f"metric {metric} needs field")
        f1, _ = q.expr(field)
        if metric in ("avg_days_between", "sum_product"):
            if not field2:
                raise ActionError(f"metric {metric} needs field2")
            f2, _ = q.expr(field2)
            if metric == "avg_days_between":
                m = f"AVG(julianday({f2}) - julianday({f1}))"
                conds += [f"{f1} IS NOT NULL", f"{f2} IS NOT NULL"]
            else:
                m = f"SUM({f1} * {f2})"
        elif metric == "count_distinct":
            m = f"COUNT(DISTINCT {f1})"
        else:
            m = f"{metric.upper()}({_num(f1) if metric in ('min', 'max') else f1})"
    group = ""
    if group_by:
        mm = MONTH_RE.match(str(group_by).strip())
        e, p = q.expr(mm.group(1) if mm else group_by)
        g, key = (f"substr({e}, 1, 7)", f"month({p})") if mm else (e, p)
        group = f" GROUP BY {g}"
        sel = f'{g} AS "{key}", {m} AS "metric", COUNT(*) AS "n"'
    else:
        sel = f'{m} AS "metric", COUNT(*) AS "n"'
    if having_op not in HAVING_OPS:
        raise ActionError(f"unknown having_op {having_op!r}; options: {', '.join(HAVING_OPS)}")
    if sort not in ("desc", "asc"):
        raise ActionError(f"sort must be desc or asc, got {sort!r}")
    inner = f"SELECT {sel} FROM {q.from_sql()}{_where_sql(conds)}{group}"
    outer = f"SELECT * FROM ({inner})"
    if having_op != "none":
        outer += f' WHERE "metric" {having_op} ?'
        params.append(float(having_value or 0))
    outer += f' ORDER BY "metric" IS NULL, "metric" {sort.upper()}' + (", 1" if group else "") + " LIMIT ?"
    return session.query(outer, params + [_limit(limit)])


def search_text(session: TaskSession, org: str, query, obj) -> Observation:
    objs = list(_fts(org))
    if obj in (None, "", "any"):
        return session.search(str(query or ""), None, limit=SEARCH_LIMIT)
    o = _resolve(str(obj), objs)
    if o is None:
        raise ActionError(f"object {obj!r} is not in the text index. Close matches: {_close(str(obj), objs)}. "
                          f"Objects: any, {', '.join(objs)}")
    return session.search(str(query or ""), [o], limit=SEARCH_LIMIT)


def execute(session: TaskSession, name: str, args: dict, org: str) -> Observation | None:
    """Run one action call; None for respond. Validation problems come back as ok=False observations."""
    t0 = time.monotonic()
    if name == "respond":
        return None
    a = args if isinstance(args, dict) else {}
    try:
        if name == "run_sql":
            return session.sql(str(a.get("query") or ""))
        if name == "find_by_name":
            return find_by_name(session, org, a.get("name"), a.get("object", "any"))
        if name == "get_record":
            return get_record(session, org, a.get("object"), a.get("id"))
        if name == "query_records":
            return query_records(session, org, a.get("object"), a.get("filters"), a.get("fields"),
                                 a.get("order_by"), bool(a.get("descending")), a.get("limit"))
        if name == "summarize_records":
            return summarize_records(session, org, a.get("object"), a.get("filters"), a.get("group_by"),
                                     a.get("metric", "count"), a.get("field"), a.get("field2"),
                                     a.get("having_op", "none"), a.get("having_value"), a.get("sort", "desc"),
                                     a.get("limit"))
        if name == "search_text":
            return search_text(session, org, a.get("query"), a.get("object", "any"))
    except ActionError as e:
        return _fail(str(e), t0)
    except (TypeError, ValueError) as e:
        return _fail(f"bad arguments for {name}: {e}", t0)
    return _fail(f"unknown action {name!r}", t0)


# ---------------------------------------------------------------- tool schemas

def _s(desc: str, **kw) -> dict:
    return {"type": "string", "description": desc, **kw}


_FILTERS = {"type": "array", "description": (
    "Conditions, all ANDed; empty list = no filter. Date-only values (YYYY-MM-DD) compare by day against "
    "timestamps; numeric values compare as numbers."), "items": {
    "type": "object", "additionalProperties": False, "required": ["field", "op", "value"], "properties": {
        "field": _s("Field on the object, or one hop through a reference: RefField.ParentField "
                    "(e.g. AccountId.ShippingState)."),
        "op": _s("in/not_in take a comma-separated value; like is case-insensitive and matches anywhere unless "
                 "the value has %; is_null/not_null ignore value.", enum=FILTER_OPS),
        "value": _s("The value as text; empty string for is_null/not_null.")}}}


@functools.lru_cache(maxsize=None)
def _action_tools(org: str, with_sql: bool) -> tuple:
    run_sql, respond = TOOLS[0], TOOLS[2]
    tools = [
        _tool("find_by_name", "Find records by name (Name, Title, Subject or Company fields) to turn a name into "
              "an Id. Returns object, Id, field and a snippet.",
              {"name": _s("The name or words of the name to look for."),
               "object": _s("Object to search, or any for all.", enum=["any", *_name_objects(org)])},
              ["name", "object"]),
        _tool("get_record", "Read every field of one record by Id.",
              {"object": _s("Object (table) name, e.g. Case."), "id": _s("The record Id.")}, ["object", "id"]),
        _tool("query_records", "List individual records of one object that match filters (at most 50 rows). "
              "For counts, rates, totals or averages use summarize_records instead.",
              {"object": _s("Object (table) name, e.g. Case."), "filters": _FILTERS,
               "fields": {"type": "array", "items": {"type": "string"}, "description":
                          "Field paths to return (Field or RefField.ParentField); empty list = all fields of the "
                          "object."},
               "order_by": _s("Field path to sort by, or empty string for no order."),
               "descending": {"type": "boolean", "description": "Sort descending when order_by is set."},
               "limit": {"type": "integer", "description": "Max rows, 1 to 50; 0 means 50."}},
              ["object", "filters", "fields", "order_by", "descending", "limit"]),
        _tool("summarize_records", "Count, total or average records of one object, optionally grouped by a field. "
              "Rows: {group_by: value, metric: value, n: rows in the group}, sorted by metric.",
              {"object": _s("Object (table) name, e.g. Case."), "filters": _FILTERS,
               "group_by": _s("Field path to group by, month(FieldPath) to group by calendar month, or empty "
                              "string for one overall row."),
               "metric": _s("count rows; count_distinct/sum/avg/min/max of field; avg_days_between = average days "
                            "from field (start) to field2 (end), rows missing either are skipped; sum_product = "
                            "sum of field * field2.", enum=METRICS),
               "field": _s("Field path the metric uses; empty string for count."),
               "field2": _s("Second field path for avg_days_between (end) and sum_product; else empty string."),
               "having_op": _s("Keep only groups whose metric compares to having_value; none = keep all.",
                               enum=HAVING_OPS),
               "having_value": {"type": "number", "description": "Number for having_op; 0 when having_op is none."},
               "sort": _s("Sort by metric.", enum=["desc", "asc"]),
               "limit": {"type": "integer", "description": "Max rows, 1 to 200; 0 means 50."}},
              ["object", "filters", "group_by", "metric", "field", "field2", "having_op", "having_value", "sort",
               "limit"]),
        _tool("search_text", "Keyword search over long text (knowledge articles, emails, call and chat transcripts, "
              "case descriptions) and names. Returns object, Id, field and a snippet.",
              {"query": _s("Keywords to search for."),
               "object": _s("Object to search, or any for all.", enum=["any", *_fts(org)])}, ["query", "object"]),
        respond,
    ]
    return tuple(tools + [run_sql]) if with_sql else tuple(tools)


def action_tools(org: str, with_sql: bool = False) -> list[dict]:
    """The action tool list for an org (enums come from that org's text index)."""
    return list(_action_tools(org, with_sql))


def __getattr__(name: str):
    # ACTION_TOOLS / ACTION_TOOLS_WITH_SQL are {org: tools}, built on first use (enums need the org's DB)
    if name in ("ACTION_TOOLS", "ACTION_TOOLS_WITH_SQL"):
        return {org: action_tools(org, name.endswith("SQL")) for org in ("b2b", "b2c")}
    raise AttributeError(name)


# ---------------------------------------------------------------- selftest

def _selftest() -> int:
    from crm.env import CRMEnv
    from crm.tasks import load

    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    def rows(o: Observation) -> list[dict]:
        return ast.literal_eval(o.text.split("\n")[0]) if o.ok else []

    def F(field, op, value=""):
        return {"field": field, "op": op, "value": value}

    def summ(obj, filters=(), group_by="", metric="count", field="", field2="", having_op="none",
             having_value=0, sort="desc", limit=0) -> dict:
        return {"object": obj, "filters": list(filters), "group_by": group_by, "metric": metric, "field": field,
                "field2": field2, "having_op": having_op, "having_value": having_value, "sort": sort,
                "limit": limit}

    def qr(obj, filters=(), fields=(), order_by="", descending=False, limit=0) -> dict:
        return {"object": obj, "filters": list(filters), "fields": list(fields), "order_by": order_by,
                "descending": descending, "limit": limit}

    env = CRMEnv("b2b")
    s = env.open_task()
    raw = lambda q: [dict(zip([d[0] for d in c.description], r)) for c in [s.conn.execute(q)] for r in c]
    ex = lambda name, args: execute(s, name, args, "b2b")

    # --- schemas
    for org in ("b2b", "b2c"):
        for t in action_tools(org, True):
            fn = t["function"]
            p = fn["parameters"]
            ok = fn["strict"] and p["additionalProperties"] is False and set(p["required"]) == set(p["properties"])
            for v in p["properties"].values():
                if v.get("type") == "array" and v["items"].get("type") == "object":
                    it = v["items"]
                    ok = ok and it["additionalProperties"] is False and set(it["required"]) == set(it["properties"])
            check(f"{org} strict schema {fn['name']}", ok)
    import crm.actions as me
    check("ACTION_TOOLS has 6, WITH_SQL has 7", len(me.ACTION_TOOLS["b2b"]) == 6
          and [t["function"]["name"] for t in me.ACTION_TOOLS_WITH_SQL["b2b"]][-1] == "run_sql")
    fb = action_tools("b2b")[0]["function"]["parameters"]["properties"]["object"]["enum"]
    fc = action_tools("b2c")[0]["function"]["parameters"]["properties"]["object"]["enum"]
    check("find_by_name enum: b2b has Account, b2c does not", "Account" in fb and "Account" not in fc, f"b2c={fc}")
    rb, rc = references("b2b"), references("b2c")
    check("references: b2b OrderItem.PriceBookEntryId, b2c canonical PricebookEntryId",
          rb["OrderItem"].get("PriceBookEntryId") == "PricebookEntry"
          and rc["OrderItem"].get("PricebookEntryId") == "PricebookEntry", str(rc["OrderItem"]))
    check("references_text mentions Case: ... OrderItemId__c -> OrderItem",
          "OrderItemId__c -> OrderItem" in references_text("b2b"), references_text("b2b").split("\n")[1][:100])

    # --- validation
    o = ex("query_records", qr("Cases"))
    check("unknown object -> close matches", not o.ok and "Close matches: Case" in o.text, o.text[:120])
    o = ex("query_records", qr("Case", [F("Stauts", "=", "Closed")]))
    check("unknown field -> close matches + columns", not o.ok and "Close matches: Status" in o.text
          and "Case fields: Id, Priority" in o.text, o.text[:160])
    o = ex("query_records", qr("Case", fields=["Subject.Name"]))
    check("non-reference hop -> lists references", not o.ok and "not a reference" in o.text
          and "OrderItemId__c -> OrderItem" in o.text, o.text[:140])
    o = ex("query_records", qr("Case", fields=["AccountId.Bogus"]))
    check("bad parent field -> Account columns", not o.ok and "Account fields: Id, Name" in o.text, o.text[:140])
    o = ex("query_records", qr("Case", fields=["OrderItemId__c.OrderId.OwnerId"]))
    check("two hops rejected", not o.ok and "one hop" in o.text, o.text)
    o = ex("query_records", qr("Case", [F("Status", "~", "x")]))
    check("bad op rejected", not o.ok and "unknown op" in o.text, o.text[:80])
    o = ex("summarize_records", summ("Case", metric="sum"))
    check("metric without field rejected", not o.ok and "needs field" in o.text, o.text)
    o = ex("query_records", qr("case", [F("status", "=", "Closed")], ["id"], limit=1))
    check("case-insensitive names resolve to canonical", o.ok and rows(o) and "Id" in rows(o)[0], o.text[:80])
    o = ex("nope", {})
    check("unknown action -> error", not o.ok and "unknown action" in o.text)
    check("respond -> None", ex("respond", {"answer": "x"}) is None)

    # --- binding, paths, dates, like, in, having, metrics
    o = ex("query_records", qr("Account", [F("Name", "=", "x' OR 1=1 --")]))
    check("injection value is bound: no rows, no error", o.ok and o.rows_total == 0, o.text)
    o = ex("get_record", {"object": "Account", "id": "x' OR '1'='1"})
    check("get_record injection: no rows", o.ok and o.rows_total == 0, o.text)
    cid = raw('SELECT Id FROM "Case" WHERE ClosedDate IS NOT NULL ORDER BY Id LIMIT 1')[0]["Id"]
    o = ex("get_record", {"object": "Case", "id": cid})
    full = raw(f"SELECT * FROM \"Case\" WHERE Id = '{cid}'")
    check("get_record returns all columns whole", o.ok and rows(o) == full, f"{len(o.text)} chars")
    o = ex("query_records", qr("Case", [F("Id", "=", cid)], ["Id", "AccountId.ShippingState", "OwnerId.LastName"]))
    want = raw(f"SELECT c.Id, a.ShippingState, u.LastName FROM \"Case\" c LEFT JOIN Account a ON a.Id = c.AccountId "
               f"LEFT JOIN User u ON u.Id = c.OwnerId WHERE c.Id = '{cid}'")[0]
    check("one-hop paths", o.ok and list(rows(o)[0].values()) == list(want.values()), o.text)
    o = ex("query_records", qr("OrderItem", [F("OrderId.OwnerId", "not_null")], ["Id", "OrderId.OwnerId"], limit=3))
    n_oi = raw('SELECT COUNT(*) n FROM OrderItem oi JOIN "Order" o ON o.Id = oi.OrderId '
               'WHERE o.OwnerId IS NOT NULL')[0]["n"]
    check("OrderItem OrderId.OwnerId", o.ok and o.rows_shown == 3 and o.rows_total == n_oi
          and all(r["OrderId.OwnerId"] for r in rows(o)),
          o.text[:120])
    day = raw('SELECT substr(CreatedDate,1,10) d FROM "Case" ORDER BY CreatedDate LIMIT 1 OFFSET 20')[0]["d"]
    n_eq = raw(f"SELECT COUNT(*) n FROM \"Case\" WHERE date(CreatedDate) = '{day}'")[0]["n"]
    n_le = raw(f"SELECT COUNT(*) n FROM \"Case\" WHERE date(CreatedDate) <= '{day}'")[0]["n"]
    o1 = ex("summarize_records", summ("Case", [F("CreatedDate", "=", day)]))
    o2 = ex("summarize_records", summ("Case", [F("CreatedDate", "<=", day)]))
    check("date-only = and <= against timestamps", rows(o1)[0]["n"] == n_eq >= 1 and rows(o2)[0]["n"] == n_le,
          f"day {day}: = {n_eq}, <= {n_le}; got {o1.text} {o2.text}")
    o = ex("query_records", qr("Account", [F("Name", "like", "QUANTUM")], ["Name"]))
    n_like = raw("SELECT COUNT(*) n FROM Account WHERE lower(Name) LIKE '%quantum%'")[0]["n"]
    check("like wraps %..% and ignores case", o.ok and o.rows_total == n_like > 0, o.text[:100])
    o = ex("query_records", qr("Account", [F("Name", "like", "q%")], ["Name"]))
    check("like keeps explicit %", o.ok and all(r["Name"].lower().startswith("q") for r in rows(o)), o.text[:80])
    o = ex("summarize_records", summ("Case", [F("Status", "in", "Closed , Working")]))
    o2 = ex("summarize_records", summ("Case", [F("Status", "not_in", "Closed,Working")]))
    check("in / not_in lists", rows(o)[0]["n"] == 88 and rows(o2)[0]["n"] == 65, f"{o.text} {o2.text}")
    o = ex("summarize_records", summ("Case", group_by="Status", having_op=">", having_value=60))
    check("having > 60 keeps Closed, Waiting", o.ok and [r["Status"] for r in rows(o)] == ["Closed",
          "Waiting on Customer"], o.text)
    o = ex("summarize_records", summ("Case", group_by="Status", sort="asc", limit=1))
    check("sort asc + limit", rows(o) == [{"Status": "Working", "metric": 13, "n": 13}], o.text)
    o = ex("query_records", qr("OrderItem", [F("Quantity", ">", "9")], ["Quantity"]))
    n_q = raw("SELECT COUNT(*) n FROM OrderItem WHERE CAST(Quantity AS REAL) > 9")[0]["n"]
    check("numeric value compares numerically on TEXT column", o.rows_total == n_q and n_q > 0
          and o.rows_shown == min(n_q, 50) and "use summarize_records" in o.text.split("\n")[-1], f"{n_q} rows")
    metric_sql = {
        "count": "COUNT(*)", "count_distinct": "COUNT(DISTINCT OwnerId)", "sum": "SUM(Quantity)",
        "avg": "AVG(UnitPrice)", "min": "MIN(CAST(Quantity AS REAL))", "max": "MAX(CAST(Quantity AS REAL))",
        "sum_product": "SUM(Quantity * UnitPrice)"}
    for metric, agg in metric_sql.items():
        obj, fld = ("Order", "OwnerId") if metric == "count_distinct" else ("OrderItem", "")
        args = summ(obj, metric=metric, field=fld or ("UnitPrice" if metric == "avg" else "Quantity"),
                    field2="UnitPrice" if metric == "sum_product" else "")
        got = rows(ex("summarize_records", args))
        want = raw(f'SELECT {agg} m FROM "{obj}"')[0]["m"]
        check(f"metric {metric}", got and abs(float(got[0]["metric"]) - float(want)) < 1e-6, f"{got} vs {want}")
    got = rows(ex("summarize_records", summ("Case", metric="avg_days_between", field="CreatedDate",
                                             field2="ClosedDate")))
    want = raw('SELECT AVG(julianday(ClosedDate) - julianday(CreatedDate)) m, COUNT(*) n FROM "Case" '
               "WHERE ClosedDate IS NOT NULL")[0]
    check("metric avg_days_between skips NULL rows", got and abs(got[0]["metric"] - want["m"]) < 1e-9
          and got[0]["n"] == want["n"], f"{got} vs {want}")
    o = ex("summarize_records", summ("Case", group_by="month(CreatedDate)", limit=200))
    check("group_by month()", o.ok and all(re.fullmatch(r"\d{4}-\d{2}", r["month(CreatedDate)"]) for r in rows(o))
          and sum(r["n"] for r in rows(o)) == 153, o.text[:100])
    o = ex("find_by_name", {"name": "Quantum", "object": "any"})
    check("find_by_name returns only name fields", o.ok and o.rows_total > 0
          and all(r["field"] in NAME_FIELDS for r in rows(o)), o.text[:140])
    o = ex("find_by_name", {"name": "Quantum", "object": "EmailMesage"})
    check("find_by_name bad object -> close matches", not o.ok and "EmailMessage" in o.text, o.text[:120])
    o = ex("search_text", {"query": "Quantum Circuits", "object": "Knowledge__kav"})
    check("search_text delegates", o.ok and "'Knowledge__kav'" in o.text and "'Case'" not in o.text, o.text[:100])
    o = ex("run_sql", {"query": 'SELECT COUNT(*) FROM "Case"'})
    check("run_sql delegates", o.ok and o.text == "[{'COUNT(*)': 153}]", o.text)
    o = ex("query_records", qr("OrderItem", fields=["Quantity"], order_by="Quantity", descending=True, limit=1))
    top = raw("SELECT MAX(CAST(Quantity AS REAL)) m FROM OrderItem")[0]["m"]
    check("order_by on TEXT numbers is numeric", rows(o) and float(rows(o)[0]["Quantity"]) == top > 9, o.text)
    got = rows(ex("summarize_records", summ("Case", metric="max", field="CreatedDate")))
    want = raw('SELECT MAX(CreatedDate) m FROM "Case"')[0]["m"]
    check("max on timestamps stays text", got and got[0]["metric"] == want, f"{got} vs {want}")

    # --- oracle equivalence on seven b2b dev tasks
    tasks = {t.idx: t for t in load("b2b")}

    def oracle(idx: int, steps) -> None:
        t = tasks[idx]
        log: list[str] = []

        def act(name, args):
            o = ex(name, args)
            log.append(f"    {name}({args})\n      -> ok={o.ok} rows={o.rows_total} {o.text[:300]}")
            return rows(o)

        answer = steps(act)
        print(f"oracle {t.task_type} idx {idx} (today {t.today}):")
        print("\n".join(log))
        check(f"oracle {t.task_type} idx {idx}: answer {answer!r} == gt {t.answer[0]!r}",
              t.split == "dev" and answer == t.answer[0])

    import calendar

    def monthly(act):
        r = act("summarize_records", summ("Case", [F("OrderItemId__c.Product2Id", "=", "01tWt000006hVZlIAM"),
                                                   F("CreatedDate", ">", "2023-06-02"),
                                                   F("CreatedDate", "<=", "2024-01-02")],
                                          group_by="month(CreatedDate)"))
        top = [x for x in r if x["metric"] == r[0]["metric"]] if r else []
        return calendar.month_name[int(top[0]["month(CreatedDate)"][5:])] if len(top) == 1 else None

    def top_issue(act):
        r = act("summarize_records", summ("Case", [F("OrderItemId__c.Product2Id", "=", "01tWt000006hV57IAE"),
                                                   F("CreatedDate", ">", "2021-01-17"),
                                                   F("CreatedDate", "<=", "2021-03-17")], group_by="IssueId__c"))
        return r[0]["IssueId__c"] if r else None

    def best_region(act):
        r = act("summarize_records", summ("Case", [F("ClosedDate", ">=", "2021-11-01"),
                                                   F("ClosedDate", "<=", "2021-11-30")],
                                          group_by="AccountId.ShippingState", metric="avg_days_between",
                                          field="CreatedDate", field2="ClosedDate", sort="asc"))
        return r[0]["AccountId.ShippingState"] if r else None

    def handle_time(act):
        r = act("summarize_records", summ("CaseHistory__c", [F("Field__c", "=", "Owner Assignment"),
                                                             F("CaseId__c.CreatedDate", ">=", "2022-09-01"),
                                                             F("CaseId__c.CreatedDate", "<=", "2022-11-30")],
                                          group_by="CaseId__c", having_op="=", having_value=1, limit=200))
        ids = ",".join(x["CaseId__c"] for x in r)
        r = act("summarize_records", summ("Case", [F("Id", "in", ids), F("Status", "=", "Closed")],
                                          group_by="OwnerId", metric="avg_days_between", field="CreatedDate",
                                          field2="ClosedDate", limit=3))
        return r[0]["OwnerId"] if r else None

    def sales_cycle(act):
        r = act("summarize_records", summ("Opportunity", [F("ContractID__c.CompanySignedDate", ">=", "2023-12-01"),
                                                          F("ContractID__c.CompanySignedDate", "<=", "2024-02-29")],
                                          group_by="OwnerId", metric="avg_days_between", field="CreatedDate",
                                          field2="ContractID__c.CompanySignedDate", limit=3))
        return r[0]["OwnerId"] if r else None

    def conversion(act):
        window = [F("CreatedDate", ">", "2022-05-21"), F("CreatedDate", "<=", "2023-11-21")]
        total = act("summarize_records", summ("Lead", window, group_by="OwnerId", limit=200))
        conv = act("summarize_records", summ("Lead", window + [F("IsConverted", "=", "1"),
                                                               F("ConvertedDate", "<=", "2023-11-21")],
                                             group_by="OwnerId", limit=200))
        c = {x["OwnerId"]: x["metric"] for x in conv}
        rates = sorted(((c.get(x["OwnerId"], 0) / x["metric"], x["OwnerId"]) for x in total), reverse=True)
        print(f"    agent-side ratio, top 3: {rates[:3]}")
        return rates[0][1] if rates and rates[0][0] > rates[1][0] else None

    def sales_amount(act):
        r = act("summarize_records", summ("OrderItem", [F("OrderId.EffectiveDate", ">", "2022-06-25"),
                                                        F("OrderId.EffectiveDate", "<=", "2022-11-25")],
                                          group_by="OrderId.OwnerId", metric="sum_product", field="Quantity",
                                          field2="UnitPrice", limit=3))
        return r[0]["OrderId.OwnerId"] if r else None

    for idx, fn in [(107, monthly), (303, top_issue), (1083, best_region), (881, handle_time),
                    (2067, sales_cycle), (987, conversion), (780, sales_amount)]:
        oracle(idx, fn)
    s.close()
    print(f"{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CRM standard actions over a TaskSession")
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        sys.exit(_selftest())
    ap.print_help()

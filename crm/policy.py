"""Issue policies: each issue's guide article, its approved fixes and each fix's time limit, as reviewed records.

  uv run --group serve --with anthropic==2.14.0 python -m crm.policy draft --org b2b   # Haiku drafts, writes JSON
  uv run --group serve python -m crm.policy show --org b2b                             # print records for review
Explanation in NOTES/code-notes.md under crm/policy.py.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6
MAX_COST = 0.20

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["fixes"],
    "properties": {"fixes": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["name", "max_days", "quote"],
        "properties": {"name": {"type": "string"},
                       "max_days": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                       "quote": {"type": "string"}}}}}}

PROMPT = """Below is a company help article for one type of customer issue. List every named fix or solution the
article offers for this issue (a named service, package, upgrade, toolkit, consultation, workshop, audit, access
or similar). For each one give:
- name: the fix's name exactly as the article writes it.
- max_days: the most days after purchase it may be offered, as an integer; null if the article says it has no
  time limit (for example "-1 days", "perpetual", "no time constraints").
- quote: the single sentence from the article that states that time limit, copied exactly, character for character.
Do not include anything the article does not name as a fix.

Article title: {title}

{text}"""


@dataclass
class Fix:
    name: str
    max_days: int | None  # None: no time limit
    quote: str


@dataclass
class IssuePolicy:
    issue_id: str
    issue: str
    article_id: str
    title: str
    fixes: list[Fix]


def path_for(org: str) -> Path:
    return ROOT / "crm" / f"policies_{org}.json"


def load(org: str) -> dict[str, IssuePolicy]:
    """Issue Id -> policy, from the reviewed records file; empty if the file is missing."""
    p = path_for(org)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())["issues"]
    return {r["issue_id"]: IssuePolicy(r["issue_id"], r["issue"], r["article_id"], r["title"],
                                       [Fix(f["name"], f["max_days"], f["quote"]) for f in r["fixes"]]) for r in raw}


def describe(p: IssuePolicy) -> str:
    lines = [f"POLICY for issue {p.issue_id} ({p.issue})", f'Guide article: {p.article_id} "{p.title}"',
             ("Approved fixes for this issue. Offering any other fix for this issue, or an approved fix after its "
              "day limit, breaks this guide:")]
    for f in p.fixes:
        win = "no time limit" if f.max_days is None else f"only up to {f.max_days} days after the purchase date"
        lines.append(f'- {f.name}: {win}. (Guide: "{f.quote}")')
    return "\n".join(lines)


OTHER = "other (not a named company fix)"


def fix_key(name: str) -> str:
    """One spelling per fix: case, spacing and a plural 's' do not matter."""
    return re.sub(r"\s+", " ", name).strip().lower().rstrip("s")


def catalog(org: str) -> list[str]:
    """Every named fix in the org's records, one spelling each, sorted."""
    names: dict[str, str] = {}
    for p in load(org).values():
        for f in p.fixes:
            names.setdefault(fix_key(f.name), f.name.rstrip("s") if f.name.endswith("Workshops") else f.name)
    return sorted(names.values())


def decide(p: IssuePolicy, gap: int | None, offered: list[str]) -> tuple[str | None, list[str]]:
    """The verdict for the fixes an agent offered: (answer, one line per fix). Answer None means no violation."""
    allowed = {fix_key(f.name): f for f in p.fixes}
    lines, broken = [], False
    for name in dict.fromkeys(offered):
        if name == OTHER:
            lines.append("- something that is not a named company fix: outside this policy, allowed")
            continue
        f = allowed.get(fix_key(name))
        if f is None:
            lines.append(f"- {name}: NOT approved for {p.issue} -> violation")
            broken = True
        elif f.max_days is None:
            lines.append(f"- {name}: approved, no time limit -> allowed")
        elif gap is None:
            lines.append(f"- {name}: approved; purchase date unknown, so the {f.max_days}-day limit cannot be "
                         "checked -> allowed")
        elif gap > f.max_days:
            lines.append(f"- {name}: approved, but offered {gap} days after purchase, limit {f.max_days} -> violation")
            broken = True
        else:
            lines.append(f"- {name}: approved; {gap} days after purchase, limit {f.max_days} -> allowed")
    if not lines:
        lines.append("- no company fix was offered: nothing for this policy to check")
    return (p.article_id if broken else None), lines


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("‘", "'")).strip()


def validate(fix: dict, article: str) -> list[str]:
    """Problems with one drafted fix: its quote must be in the article and must state the window it claims."""
    errs = []
    q, a = _norm(fix["quote"]), _norm(article)
    if q not in a:
        errs.append("quote is not verbatim in the article")
    n = fix["max_days"]
    if n is not None and str(n) not in q and not (n == 365 and re.search(r"year", q, re.I)):
        errs.append(f"quote does not state {n} days")
    if n is None and not re.search(r"-1|no time|not limited|perpetual|indefinite|no expir|without|unlimited|"
                                   r"any time|no limit",
                                   q, re.I):
        errs.append("quote does not state 'no time limit'")
    return errs


def _key(env: Path) -> str | None:
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        name, sep, val = line.partition("=")
        if sep and name.strip() == "ANT_KEY" and val.strip():
            return val.strip()
    return None


def draft(org: str) -> int:
    """Map each issue to its guide (reranked search on the issue's name and description), have Haiku draft the
    fixes, validate every quote against the article, write the records file."""
    import anthropic
    from crm import retrieval
    from crm.env import CRMEnv
    key = _key(ROOT / ".env")
    if not key:
        print("ANT_KEY not found in .env", file=sys.stderr)
        return 2
    client = anthropic.Anthropic(api_key=key)
    conn = sqlite3.connect(f"file:{CRMEnv(org).db_path}?mode=ro", uri=True)
    spent, out, bad = 0.0, [], 0
    for iid, name, desc in conn.execute("SELECT Id, Name, Description__c FROM Issue__c ORDER BY Name").fetchall():
        r = retrieval.search(org, f"{name}. {desc or ''}", 3)
        ranked = (r.passages or r.ranked) if r else []
        aid = ranked[0].Id
        title, summ, body = conn.execute(
            "SELECT Title, Summary, FAQ_Answer__c FROM Knowledge__kav WHERE Id = ?", [aid]).fetchone()
        article = f"{summ or ''}\n\n{body or ''}"
        if spent + (len(article) / 4 + 600) * PRICE_IN + 600 * PRICE_OUT > MAX_COST:
            print(f"spending cap reached: ${spent:.4f} of ${MAX_COST:.2f}", file=sys.stderr)
            return 3
        try:
            resp = client.messages.create(
                model=MODEL, max_tokens=1500,
                messages=[{"role": "user", "content": PROMPT.format(title=title, text=article)}],
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
        except anthropic.APIError as e:
            print(f"API error on {name}: {str(e).replace(key, '[redacted]')[:300]}", file=sys.stderr)
            return 1
        spent += resp.usage.input_tokens * PRICE_IN + resp.usage.output_tokens * PRICE_OUT
        fixes = json.loads(next(b.text for b in resp.content if b.type == "text"))["fixes"]
        for f in fixes:
            f["problems"] = validate(f, article)
            bad += bool(f["problems"])
        out.append({"issue_id": iid, "issue": name, "article_id": aid, "title": title, "fixes": fixes})
        print(f"{name}: {len(fixes)} fixes" + "".join(f"\n  {f['name']} | {f['max_days']} | "
              f"{'OK' if not f['problems'] else 'PROBLEM ' + '; '.join(f['problems'])}" for f in fixes))
    path_for(org).write_text(json.dumps({
        "source": f"drafted by {MODEL} from each issue's guide article; every entry reviewed by hand before use",
        "reviewed": False, "issues": out}, indent=2) + "\n")
    print(f"wrote {path_for(org)}; {bad} fixes with problems; spent ${spent:.4f}")
    return 0


def purchase_gap(conn, case_id: str) -> tuple[str | None, int | None]:
    """The case's purchase date and the whole days from purchase to the case being opened."""
    r = conn.execute('SELECT o.EffectiveDate, CAST(julianday(c.CreatedDate) - julianday(o.EffectiveDate) AS INTEGER) '
                     'FROM "Case" c LEFT JOIN OrderItem oi ON oi.Id = c.OrderItemId__c '
                     'LEFT JOIN "Order" o ON o.Id = oi.OrderId WHERE c.Id = ?', [case_id]).fetchone()
    return (r[0], r[1]) if r else (None, None)


def _mentions(fix: str, text: str) -> bool:
    f = fix.lower().rstrip("s")
    if f in text:
        return True
    words = f.split()  # short form ("priority support") counts only with the last word ("upgrade") nearby
    head, tail = " ".join(words[:2]), words[-1][:6]
    return len(words) > 2 and any(tail in text[max(0, m.start() - 60): m.end() + 60]
                                  for m in re.finditer(re.escape(head), text))


def check(org: str) -> int:
    """Apply the records to every policy task with a plain string matcher; compare with the answer key."""
    from crm.env import CRMEnv
    from crm.tasks import load as load_tasks
    pol = load(org)
    names = catalog(org)
    conn = sqlite3.connect(f"file:{CRMEnv(org).db_path}?mode=ro", uri=True)
    n = right = 0
    for t in load_tasks(org):
        if t.task_type != "policy_violation_identification":
            continue
        n += 1
        cid = re.search(r"500[0-9A-Za-z]{15}", t.required).group(0)
        iid = conn.execute('SELECT IssueId__c FROM "Case" WHERE Id = ?', [cid]).fetchone()[0]
        _, gap = purchase_gap(conn, cid)
        body = " ".join(r[0] or "" for r in conn.execute(
            "SELECT TextBody FROM EmailMessage WHERE ParentId = ? UNION ALL "
            "SELECT Body FROM LiveChatTranscript WHERE CaseId = ?", [cid, cid])).lower()
        p = pol[iid]
        offered = [name for name in names if _mentions(name, body)]  # a string matcher stands in for the model
        pred, lines = decide(p, gap, offered)
        right += pred == t.answer[0]
        if pred != t.answer[0]:
            print(f"MISS idx {t.idx} ({p.issue}, {gap} days): expected {t.answer[0]}, rule says {pred}; {lines}")
    print(f"records + rule match the answer key on {right} of {n} {org} policy tasks")
    return 0 if right == n else 1


def main(argv: list[str]) -> int:
    org = argv[argv.index("--org") + 1] if "--org" in argv else "b2b"
    if argv and argv[0] == "draft":
        return draft(org)
    if argv and argv[0] == "check":
        return check(org)
    if argv and argv[0] == "show":
        for p in load(org).values():
            print(describe(p), "\n")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

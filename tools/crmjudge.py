#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic==2.14.0"]
# ///
"""Grade a CRM agent's knowledge_qa answers with Claude Haiku 5.5 (rule-graded for not-in-data questions).

  uv run tools/crmjudge.py results/p11-x.tasks.jsonl [more] [--limit N] [--dry-run] [--org b2b] [--max-cost 1.00]
  uv run tools/crmjudge.py --selftest
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import io
import json
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402
import httpx2  # noqa: E402

from crm import tasks as crm_tasks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "crmarena" / "derived"
MODEL = "claude-haiku-5-5"
MAX_TOKENS = 1024
PRICE_IN = 0.10 / 1_000_000
PRICE_OUT = 0.50 / 1_000_000
EST_OUT_TOKENS = 300
VERDICTS = ("correct", "partial", "incorrect", "wrongly_abstained")
SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": list(VERDICTS)}, "why": {"type": "string"}},
    "required": ["verdict", "why"],
    "additionalProperties": False,
}
SYSTEM = (
    "You grade a customer-service agent's answer to a customer question against a reference answer.\n"
    "The agent's answer is correct if it conveys the same key information as the reference; paraphrase and "
    "extra correct detail are fine.\n"
    "Verdicts: correct; partial (it has part of the key point but misses or distorts the rest); "
    "wrongly_abstained (the agent said the knowledge base does not cover the question, although a reference "
    "answer exists); incorrect (anything else).\n"
    'Reply with JSON: {"verdict": ..., "why": one short sentence}.'
)
ABSTAIN_PHRASES = ("does not cover", "do not cover", "doesn't cover", "not covered", "no information",
                   "no article", "not in our knowledge base", "not in the knowledge base")
LEAD_WORDS = frozenset("""how what which who whom whose why when where does do did is are was were can could should
would will the a an in on for of our your please explain describe tell according given if to with and or but""".split())
RUN_RE = re.compile(r"[A-Z]\w*(?:-\w+)*(?:[ ]+[A-Z]\w*(?:-\w+)*)*")
CAMEL_RE = re.compile(r"[a-z][A-Z]")


@dataclasses.dataclass
class Spend:
    """Running actual cost across all files in one invocation; capped is set when the cap stops a call."""
    total: float = 0.0
    capped: bool = False


# ---------------------------------------------------------------- key, cost


def load_key(env_path: Path) -> str | None:
    """Read ANT_KEY from a .env file (no quotes, trailing newline optional); None when absent."""
    if not env_path.is_file():
        return None
    for line in env_path.read_text().splitlines():
        name, sep, val = line.strip().partition("=")
        if sep and name.strip() == "ANT_KEY" and val.strip():
            return val.strip()
    return None


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * PRICE_IN + output_tokens * PRICE_OUT


def est_tokens(*texts: str) -> int:
    return sum(len(t) for t in texts) // 4 + 1


def est_call_cost(user: str) -> float:
    """Estimated cost of one judge call: chars/4 input tokens, EST_OUT_TOKENS output."""
    return cost_usd(est_tokens(SYSTEM, user), EST_OUT_TOKENS)


# ---------------------------------------------------------------- classification


def _like(tok: str) -> str:
    return "%" + tok.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def in_data(conn: sqlite3.Connection, tok: str) -> bool:
    """True when tok is a substring (case-insensitive) of a KB title/summary/answer or a Product2 name."""
    p = _like(tok)
    kb = conn.execute("SELECT 1 FROM Knowledge__kav WHERE Title LIKE ? ESCAPE '\\' OR Summary LIKE ? ESCAPE '\\' "
                      "OR FAQ_Answer__c LIKE ? ESCAPE '\\' LIMIT 1", (p, p, p)).fetchone()
    return bool(kb or conn.execute("SELECT 1 FROM Product2 WHERE Name LIKE ? ESCAPE '\\' LIMIT 1", (p,)).fetchone())


def missing_names(query: str, conn: sqlite3.Connection) -> list[str]:
    """Product-like names in the query with a distinctive token found nowhere in the data.

    A run of capitalized words is a candidate; a token is distinctive when it has an inner capital (CamelCase)
    or is hyphenated inside a multi-word run. Runs with no distinctive token are never flagged."""
    out = []
    for m in RUN_RE.finditer(query):
        toks = m.group().split()
        while toks and toks[0].lower() in LEAD_WORDS:
            toks.pop(0)
        if not toks:
            continue
        distinct = [t for t in toks if CAMEL_RE.search(t) or ("-" in t and len(toks) > 1)]
        if distinct and any(not in_data(conn, t) for t in distinct):
            out.append(" ".join(toks))
    return out


def abstains(answer: str | None) -> bool:
    low = (answer or "").lower()
    return any(p in low for p in ABSTAIN_PHRASES)


# ---------------------------------------------------------------- judging


def build_user(question: str, reference: str, answer: str | None) -> str:
    return f"Question: {question}\n\nReference answer: {reference}\n\nAgent's answer: {answer or '(no answer)'}"


def parse_verdict(resp) -> dict:
    """Pull {verdict, why} from a messages response's text blocks; ValueError on anything malformed."""
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"unparseable judge output (stop_reason={resp.stop_reason}): {text[:120]!r}") from e
    if not isinstance(obj, dict) or obj.get("verdict") not in VERDICTS or not isinstance(obj.get("why"), str):
        raise ValueError(f"judge output off schema: {text[:120]!r}")
    return {"verdict": obj["verdict"], "why": obj["why"]}


def call_judge(client, user: str) -> tuple[dict, int, int]:
    """One structured-output call; returns (parsed, input_tokens, output_tokens)."""
    resp = client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM, messages=[{"role": "user", "content": user}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})
    return parse_verdict(resp), resp.usage.input_tokens, resp.usage.output_tokens


def rec_key(rec: dict) -> tuple:
    return (rec.get("run_id"), rec.get("arm"), rec.get("idx"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class OrgData:
    """Per-org questions and a read-only DB connection, loaded once."""

    def __init__(self, org: str):
        self.org = org
        self.tasks = {t.idx: t for t in crm_tasks.load(org)}
        self.conn = sqlite3.connect(f"file:{DB_DIR / (org + '.db')}?mode=ro", uri=True)


def knowledge_records(path: Path, org: str, limit: int | None) -> list[dict]:
    recs = [r for r in read_jsonl(path) if r.get("task_type") == "knowledge_qa" and r.get("org") == org]
    return recs[:limit] if limit is not None else recs


def prepare(rec: dict, data: OrgData) -> dict | None:
    """Classify one record; None when its question is not in crm.tasks."""
    task = data.tasks.get(rec["idx"])
    if task is None:
        return None
    names = missing_names(task.query, data.conn)
    ref = "; ".join(str(x) for x in (rec.get("gt") or task.answer or []))
    return {"rec": rec, "question": task.query, "reference": ref, "names": names,
            "cls": "not_in_data" if names else "answerable"}


def judged_path(path: Path) -> Path:
    return path.with_name(path.name + ".judged.jsonl")


def process_file(path: Path, args, data: OrgData, client, spend: Spend, key: str) -> None:
    """Judge new records of one file, appending each result to the .judged.jsonl as it lands."""
    out_path = judged_path(path)
    done = {rec_key(r) for r in read_jsonl(out_path) if r.get("verdict") != "error"}
    with out_path.open("a") as out:
        for rec in knowledge_records(path, args.org, args.limit):
            if rec_key(rec) in done:
                continue
            item = prepare(rec, data)
            if item is None:
                print(f"skip idx {rec['idx']}: question not found for org {args.org}")
                continue
            row = {"run_id": rec.get("run_id"), "arm": rec.get("arm"), "idx": rec["idx"], "class": item["cls"],
                   "verdict": None, "why": None, "model": None, "input_tokens": 0, "output_tokens": 0}
            if item["cls"] == "not_in_data":
                ok = abstains(rec.get("answer"))
                row.update(verdict="correct_abstain" if ok else "wrong_substitute", model="rule",
                           why=f"not in data: {', '.join(item['names'])}")
            else:
                user = build_user(item["question"], item["reference"], rec.get("answer"))
                if spend.total + est_call_cost(user) > args.max_cost:
                    spend.capped = True
                    break
                row["model"] = MODEL
                try:
                    parsed, n_in, n_out = call_judge(client, user)
                    row.update(parsed, input_tokens=n_in, output_tokens=n_out)
                    spend.total += cost_usd(n_in, n_out)
                except (anthropic.APIError, ValueError) as e:
                    msg = str(e).replace(key, "[redacted]")
                    row.update(verdict="error", why=f"{type(e).__name__}: {msg}"[:500])
            out.write(json.dumps(row) + "\n")
            out.flush()


def summarize(path: Path) -> dict:
    """Per-class verdict counts, accuracy and token totals over the whole .judged.jsonl (last record per key wins)."""
    latest = {rec_key(r): r for r in read_jsonl(judged_path(path))}
    rows = list(latest.values())
    by_class: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        by_class[r["class"]][r["verdict"]] += 1
    v = collections.Counter(r["verdict"] for r in rows)
    score = v["correct"] + 0.5 * v["partial"] + v["correct_abstain"]
    n_in = sum(r["input_tokens"] for r in rows)
    n_out = sum(r["output_tokens"] for r in rows)
    return {"n": len(rows), "by_class": {k: dict(c) for k, c in by_class.items()},
            "accuracy": score / len(rows) if rows else None, "input_tokens": n_in, "output_tokens": n_out,
            "cost": cost_usd(n_in, n_out)}


def print_summary(path: Path) -> None:
    s = summarize(path)
    print(f"== {path}: {s['n']} judged")
    for cls, counts in sorted(s["by_class"].items()):
        print(f"  {cls}: {sum(counts.values())}  {dict(sorted(counts.items()))}")
    acc = "n/a" if s["accuracy"] is None else f"{s['accuracy']:.3f}"
    print(f"  accuracy (correct + 0.5*partial + correct_abstain) / total: {acc}")
    print(f"  tokens in/out: {s['input_tokens']}/{s['output_tokens']}  cost: ${s['cost']:.6f}")


def dry_run_file(path: Path, args, data: OrgData) -> None:
    recs = knowledge_records(path, args.org, args.limit)
    items = [i for i in (prepare(r, data) for r in recs) if i]
    classes = collections.Counter(i["cls"] for i in items)
    print(f"== {path}: {len(recs)} knowledge records, classes {dict(classes)}")
    for i in items:
        if i["names"]:
            print(f"  not_in_data idx {i['rec']['idx']}: {i['names']}")
    ans = [i for i in items if i["cls"] == "answerable"]
    est_in = est_out = 0
    for i in ans:
        user = build_user(i["question"], i["reference"], i["rec"].get("answer"))
        est_in += est_tokens(SYSTEM, user)
        est_out += EST_OUT_TOKENS
    if ans:
        first = ans[0]
        print("  example prompt (system, then user):\n" + SYSTEM + "\n---\n"
              + build_user(first["question"], first["reference"], first["rec"].get("answer")))
    print(f"  estimate: {len(ans)} calls, ~{est_in} input tokens, ~{est_out} output tokens, "
          f"~${cost_usd(est_in, est_out):.6f}")


# ---------------------------------------------------------------- main


def run(args, env_path: Path = ROOT / ".env", make_client=None) -> int:
    """Run the CLI body; returns the exit status (0 ok, 2 missing key, 3 spending cap reached)."""
    paths = [Path(p) for p in args.files]
    data = OrgData(args.org)
    if args.dry_run:
        for p in paths:
            dry_run_file(p, args, data)
        return 0
    key = load_key(env_path)
    if not key:
        print(f"ANT_KEY not found in {env_path}", file=sys.stderr)
        return 2
    client = (make_client or (lambda k: anthropic.Anthropic(api_key=k)))(key)
    spend = Spend()
    for p in paths:
        process_file(p, args, data, client, spend, key)
        print_summary(p)
        if spend.capped:
            break
    if spend.capped:
        print(f"spending cap reached: ${spend.total:.6f} of ${args.max_cost:.2f}")
        return 3
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="*.tasks.jsonl files")
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N knowledge records per file")
    ap.add_argument("--dry-run", action="store_true", help="no API calls; print counts, names, an example and an estimate")
    ap.add_argument("--org", default="b2b")
    ap.add_argument("--max-cost", type=float, default=1.00, help="hard cap in USD on actual spend (default 1.00)")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- selftest


class FakeMessages:
    """Records calls and returns real anthropic Message objects with canned structured output."""

    def __init__(self, replies: list, usage: tuple[int, int] = (500, 40)):
        self.replies, self.usage, self.calls = list(replies), usage, []

    def create(self, **kw):
        self.calls.append(kw)
        reply = self.replies.pop(0) if self.replies else {"verdict": "correct", "why": "ok"}
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return anthropic.types.Message(
            id="msg_fake", type="message", role="assistant", model=MODEL, stop_reason="end_turn",
            content=[anthropic.types.TextBlock(type="text", text=text)],
            usage=anthropic.types.Usage(input_tokens=self.usage[0], output_tokens=self.usage[1]))


class FakeClient:
    def __init__(self, replies: list, usage: tuple[int, int] = (500, 40)):
        self.messages = FakeMessages(replies, usage)


def selftest() -> int:
    fails: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(("ok   " if ok else "FAIL ") + name + (f"  {detail}" if detail and not ok else ""))
        if not ok:
            fails.append(name)

    fake_key = "sk-ant-FAKEKEY-0123456789"
    data = OrgData("b2b")
    conn = data.conn
    for q, names in [("How does ChipEffic Pro ensure designs are compliant with standards?", ["ChipEffic Pro"]),
                     ("How does SecureChip Integrator protect sensitive design data?", ["SecureChip Integrator"]),
                     ("What is a significant benefit of AI-VeriSim Pro in design verification?", ["AI-VeriSim Pro"]),
                     ("How does SecureAnalytics Pro protect data?", []),
                     ("What does Volume-Based Discounts mean for TechPulse Solutions customers?", []),
                     ("What is the duration of the extended technical support offer?", [])]:
        got = missing_names(q, conn)
        check(f"detector: {q[:48]!r}", got == names, f"got {got}")
    for ph in ("Our knowledge base does not cover X.", "There is NO INFORMATION on it", "Not covered.",
               "I found no article on this", "That is not in our knowledge base"):
        check(f"abstain phrase: {ph!r}", abstains(ph))
    check("abstain rule rejects a substantive answer", not abstains("It uses AES-256 encryption.") and not abstains(None))
    ok = parse_verdict(FakeMessages([{"verdict": "partial", "why": "half"}]).create())
    check("parse structured output", ok == {"verdict": "partial", "why": "half"})
    for bad in ("not json", {"verdict": "maybe", "why": "x"}, {"verdict": "correct"}):
        try:
            parse_verdict(FakeMessages([bad]).create())
            check(f"parse rejects {bad!r}", False)
        except ValueError:
            check(f"parse rejects {bad!r}", True)
    check("cost arithmetic", abs(cost_usd(1_000_000, 1_000_000) - 0.60) < 1e-12 and abs(cost_usd(1000, 300) - 0.00025) < 1e-12)

    src = ROOT / "results" / "p11-svc2-q-off.tasks.jsonl"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        env = tmp / ".env"
        env.write_text(f"OTHER=1\nANT_KEY={fake_key}")
        check("load_key without trailing newline", load_key(env) == fake_key and load_key(tmp / "none") is None)
        f = tmp / "run.tasks.jsonl"
        shutil.copy(src, f)
        n_ans = sum(1 for r in knowledge_records(f, "b2b", None) if (prepare(r, data) or {}).get("cls") == "answerable")
        n_know = len(knowledge_records(f, "b2b", None))
        replies = [{"verdict": "correct", "why": "same"}, {"verdict": "wrongly_abstained", "why": "abstained"},
                   anthropic.APIConnectionError(message=f"boom {fake_key}", request=httpx2.Request("POST", "http://x")),
                   {"verdict": "partial", "why": "half"}]
        client = FakeClient(replies)
        out, err = io.StringIO(), io.StringIO()
        ns = parse_args([str(f), "--max-cost", "1"])
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = run(ns, env, lambda k: client)
        check("run exits 0", rc == 0, str(rc))
        rows = read_jsonl(judged_path(f))
        check("one judged row per knowledge record", len(rows) == n_know, f"{len(rows)} vs {n_know}")
        check("API calls only for answerable", len(client.messages.calls) == n_ans, f"{len(client.messages.calls)} vs {n_ans}")
        kw = client.messages.calls[0]
        check("request shape", kw["model"] == MODEL and kw["max_tokens"] == 1024 and
              kw["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})
        rule = [r for r in rows if r["class"] == "not_in_data"]
        check("not_in_data rows are rule-graded", rule and all(r["model"] == "rule" and r["input_tokens"] == 0 for r in rule))
        check("ChipEffic Pro abstain is correct_abstain",
              any(r["idx"] == 762 and r["verdict"] == "correct_abstain" for r in rows))
        errs = [r for r in rows if r["verdict"] == "error"]
        check("API error recorded and run continues", len(errs) == 1 and "[redacted]" in errs[0]["why"], str(errs))
        check("no key in output or judged file", "sk-ant" not in out.getvalue() + err.getvalue() + f.parent.joinpath(
            f.name + ".judged.jsonl").read_text())
        s = summarize(f)
        v = collections.Counter(r["verdict"] for r in rows)
        want = (v["correct"] + 0.5 * v["partial"] + v["correct_abstain"]) / len(rows)
        check("summary accuracy and tokens", abs(s["accuracy"] - want) < 1e-12 and s["input_tokens"] == 500 * (n_ans - 1),
              str(s))
        # resume: a second run only retries the error row
        client2 = FakeClient([{"verdict": "correct", "why": "retry"}])
        with contextlib.redirect_stdout(io.StringIO()):
            run(ns, env, lambda k: client2)
        check("resume skips judged records, retries errors", len(client2.messages.calls) == 1)
        check("summary after retry has no error", "error" not in summarize(f)["by_class"].get("answerable", {}))
        with contextlib.redirect_stdout(io.StringIO()):
            run(ns, env, lambda k: client2)
        check("resume with nothing left makes no calls", len(client2.messages.calls) == 1)
        # missing key
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = run(ns, tmp / "nokey", lambda k: client2)
        check("missing key exits 2", rc == 2 and "sk-ant" not in err.getvalue())
        # spending cap
        f2 = tmp / "cap.tasks.jsonl"
        shutil.copy(src, f2)
        first = next(i for i in (prepare(r, data) for r in knowledge_records(f2, "b2b", None)) if i["cls"] == "answerable")
        est = est_call_cost(build_user(first["question"], first["reference"], first["rec"].get("answer")))
        usage = (est_tokens(SYSTEM, build_user(first["question"], first["reference"], first["rec"].get("answer"))), EST_OUT_TOKENS)
        client3 = FakeClient([], usage)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = run(parse_args([str(f2), "--max-cost", repr(est * 1.05)]), env, lambda k: client3)
        check("cap stops after the first call and exits 3", rc == 3 and len(client3.messages.calls) == 1,
              f"rc={rc} calls={len(client3.messages.calls)}")
        check("cap line printed and judged rows kept", "spending cap reached: $" in out.getvalue()
              and any(r["verdict"] == "correct" for r in read_jsonl(judged_path(f2))))
        # dry run makes no calls
        client4 = FakeClient([])
        f3 = tmp / "dry.tasks.jsonl"
        shutil.copy(src, f3)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = run(parse_args([str(f3), "--dry-run"]), tmp / "nokey", lambda k: client4)
        check("dry run: no calls, no judged file, names listed", rc == 0 and not client4.messages.calls
              and not judged_path(f3).exists() and "ChipEffic Pro" in out.getvalue() and "estimate:" in out.getvalue())
    print("selftest FAILED: " + ", ".join(fails) if fails else "selftest passed")
    return 1 if fails else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.selftest:
        return selftest()
    if not args.files:
        print("no input files", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    sys.exit(main())

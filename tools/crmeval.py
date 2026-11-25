#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx", "fastembed==0.9.1"]
# ///
"""Run the CRM agent harness against a vLLM endpoint and report results tables.

  uv run tools/crmeval.py run --url http://localhost:8000 --arm q-off --org b2b --split dev --limit 5
  uv run tools/crmeval.py report results/p11-q-off-b2b-dev.tasks.jsonl [more files] [--by type]
  uv run tools/crmeval.py --selftest
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import dataclasses
import hashlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from crm import tasks as crm_tasks  # noqa: E402
from crm.agent import ARMS, PROTOCOLS, run_many  # noqa: E402
from crm.grade import reward_metric_for  # noqa: E402

SKILLS = ("workflow", "policy", "text", "database")
SPLITS = ("dev", "test", "holdout", "kqdev", "all")  # kqdev: the synthetic knowledge practice set (crm/kqdev_b2b.json)


# ---------------------------------------------------------------- run


def select_tasks(tasks: list, split: str, types: list[str] | None, limit: int | None) -> list:
    """Filter by split and types; limit keeps the first N per task type by idx (a stratified sample)."""
    picked = [t for t in tasks if (split == "all" or t.split == split) and (not types or t.task_type in types)]
    picked.sort(key=lambda t: t.idx)
    if limit is not None:
        seen: collections.Counter = collections.Counter()
        kept = []
        for t in picked:
            if seen[t.task_type] < limit:
                seen[t.task_type] += 1
                kept.append(t)
        picked = kept
    return picked


def default_run_id(arm: str, org: str, split: str, limit: int | None, types: list[str] | None = None) -> str:
    """A --types run gets its own id so it never lands in the full run's file."""
    tag = "-types-" + hashlib.sha256(",".join(sorted(types)).encode()).hexdigest()[:8] if types else ""
    return f"p11-{arm}-{org}-{split}" + (f"-lim{limit}" if limit is not None else "") + tag


def served_models(url: str, client: httpx.Client | None = None) -> list[str]:
    """Model ids from GET {url}/v1/models; raises httpx errors or KeyError on a bad reply."""
    own = client is None
    client = client or httpx.Client(timeout=10.0)
    try:
        r = client.get(f"{url.rstrip('/')}/v1/models")
        r.raise_for_status()
        return [m["id"] for m in r.json()["data"]]
    finally:
        if own:
            client.close()


def guard_model(url: str, arm_model: str, model_arg: str | None, client: httpx.Client | None = None) -> list[str]:
    """Print the served ids; exit 2 if --model is absent and the arm's preset model is not served."""
    try:
        served = served_models(url, client)
    except (httpx.HTTPError, KeyError, ValueError, TypeError) as e:
        print(f"error: cannot read {url}/v1/models ({type(e).__name__}: {e}); refusing to start", file=sys.stderr)
        raise SystemExit(2)
    print(f"served models at {url}: {', '.join(served) if served else '(none)'}")
    wanted = model_arg or arm_model
    if wanted not in served:
        hint = "" if model_arg else " Pass --model with the served name if it is the same weights."
        print(f"error: model {wanted!r} is not served; served: {served}.{hint}", file=sys.stderr)
        raise SystemExit(2)
    return served


def cmd_run(args: argparse.Namespace, client: httpx.Client | None = None) -> int:
    arm = ARMS[args.arm]
    types = [x for x in args.types.split(",") if x] if args.types else None
    unknown = sorted(set(types or []) - set(crm_tasks.SKILL))
    if unknown:
        print(f"error: unknown task types {unknown}", file=sys.stderr)
        return 2
    source = ([crm_tasks.Task(**d) for d in json.loads(Path(args.tasks_file).read_text())]
              if getattr(args, "tasks_file", None) else crm_tasks.load(args.org))
    picked = select_tasks(source, args.split, types, args.limit)
    if args.idx:  # an explicit task list, e.g. the app's curated showcase
        want = {int(x) for x in args.idx.split(",") if x.strip()}
        picked = [t for t in picked if t.idx in want]
    if not picked:
        print(f"error: no tasks for org={args.org} split={args.split} types={types}", file=sys.stderr)
        return 2
    guard_model(args.url, arm.model, args.model, client)
    if args.model:
        arm = dataclasses.replace(arm, model=args.model)
    run_id = args.run_id or default_run_id(args.arm, args.org, args.split, args.limit, types)
    if args.protocol != "react_text_v0" and not args.run_id:
        run_id += f"-{args.protocol}"
    out_dir = Path(args.out_dir)
    tasks_path, calls_path = out_dir / f"{run_id}.tasks.jsonl", out_dir / f"{run_id}.calls.jsonl"
    print(f"run_id {run_id}; arm {arm.name}; protocol {args.protocol}; sampling:")
    for k, v in dataclasses.asdict(arm).items():
        print(f"  {k} = {v}")
    per_skill = collections.Counter(t.skill for t in picked)
    print(f"tasks: {len(picked)} -> " + ", ".join(f"{s}={n}" for s, n in sorted(per_skill.items())))
    print(f"outputs: {tasks_path}\n         {calls_path}", flush=True)
    asyncio.run(run_many(picked, arm, base_url=args.url, org=args.org, out_tasks_path=tasks_path,
                         out_calls_path=calls_path, run_id=run_id, run_seed=args.run_seed,
                         concurrency=args.concurrency, window=args.window, max_turns=args.max_turns,
                         model=args.model, protocol=args.protocol))
    return 0


# ---------------------------------------------------------------- report


def percentile(values: list[float], p: int, min_n: int = 1) -> float | None:
    """Nearest-rank percentile (integer p); None when there are fewer than min_n values."""
    if len(values) < min_n:
        return None
    s = sorted(values)
    return s[max(1, -(-p * len(s) // 100)) - 1]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def mcnemar_p(only_first: int, only_second: int) -> float:
    """Exact two-sided McNemar p-value: binomial test on the discordant pairs with p = 0.5."""
    n = only_first + only_second
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_first, only_second) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def table(headers: list[str], rows: list[list]) -> str:
    cells = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    lines = ["  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths)))
             for r in cells]
    return "\n".join(["  " + ln for ln in lines])


def load_records(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: {path} does not exist")
    recs, bad = [], 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except ValueError:
            bad += 1
    if bad:
        print(f"warning: {path}: {bad} unparsable line(s) skipped", file=sys.stderr)
    return recs


def _sum(recs: list[dict], key: str) -> int:
    return sum(r[key] for r in recs if r.get(key) is not None)


def reward_stats(recs: list[dict]) -> dict:
    """n, graded rewards, pending count and means (all / exact / fuzzy) for a set of task records."""
    graded = [r for r in recs if r.get("reward") is not None]
    by_metric = {"exact_match": [], "fuzzy_match": []}
    for r in graded:
        by_metric.get(reward_metric_for(r["task_type"]), []).append(r["reward"])
    return {"n": len(recs), "graded": len(graded), "pending": len(recs) - len(graded),
            "mean": mean([r["reward"] for r in graded]), "exact": mean(by_metric["exact_match"]),
            "fuzzy": mean(by_metric["fuzzy_match"]), "n_exact": len(by_metric["exact_match"]),
            "n_fuzzy": len(by_metric["fuzzy_match"])}


def skill_table(recs: list[dict]) -> tuple[dict, float | None, dict]:
    """Per-skill stats for the four scored skills, the All mean of skill means, and the confidentiality stats."""
    stats = {s: reward_stats([r for r in recs if r.get("skill") == s]) for s in SKILLS}
    means = [stats[s]["mean"] for s in SKILLS if stats[s]["mean"] is not None]
    return stats, mean(means), reward_stats([r for r in recs if r.get("skill") == "confidentiality"])


def run_stats(recs: list[dict]) -> dict:
    calls = [r["n_calls"] for r in recs if r.get("n_calls") is not None]
    walls = [r["wall_ms"] for r in recs if r.get("wall_ms") is not None]
    n_calls = _sum(recs, "n_calls")
    prompt, cached = _sum(recs, "prompt_tokens_total"), _sum(recs, "cached_tokens_total")
    return {"end": collections.Counter(r.get("end_reason") for r in recs),
            "calls_p50": percentile(calls, 50), "calls_p95": percentile(calls, 95, 20),
            "calls_max": max(calls) if calls else None,
            "format_per_call": _sum(recs, "n_format_errors") / n_calls if n_calls else None,
            "sql_per_call": _sum(recs, "n_sql_errors") / n_calls if n_calls else None,
            "truncations": _sum(recs, "n_truncations"), "prompt": prompt, "cached": cached,
            "completion": _sum(recs, "completion_tokens_total"),
            "cached_share": cached / prompt if prompt else None,
            "wall_p50": percentile(walls, 50), "wall_p95": percentile(walls, 95, 20)}


def groups_of(paths: list[str]) -> list[tuple[str, str, list[dict]]]:
    """(file name, arm, records) per file and arm, in file order."""
    out = []
    for path in paths:
        by_arm: dict[str, list[dict]] = collections.defaultdict(list)
        for r in load_records(path):
            by_arm[r.get("arm", "?")].append(r)
        out.extend((Path(path).name, arm, recs) for arm, recs in by_arm.items())
    return out


def print_group(name: str, arm: str, recs: list[dict], by: str) -> None:
    models = sorted({str(r.get("served_model")) for r in recs})
    protos = sorted({str(r.get("protocol")) for r in recs})
    print(f"\n== {name} | arm {arm} | served model {', '.join(models)} | protocol {', '.join(protos)} "
          f"| {len(recs)} tasks ==")
    stats, overall, conf = skill_table(recs)
    rows = [[s, st["n"], st["pending"], fmt(st["mean"]), f"{fmt(st['exact'])} ({st['n_exact']})",
             f"{fmt(st['fuzzy'])} ({st['n_fuzzy']})"] for s, st in stats.items()]
    present = sum(1 for s in SKILLS if stats[s]["mean"] is not None)
    rows.append(["All", sum(st["n"] for st in stats.values()), sum(st["pending"] for st in stats.values()),
                 fmt(overall), "", f"mean of {present} skill means"])
    print(table(["skill", "n", "pending", "mean reward", "exact (n)", "fuzzy (n)"], rows))
    print(f"  confidentiality (not in All): n={conf['n']}, pending judge={conf['pending']}"
          + (f", judged mean={fmt(conf['mean'])} over {conf['graded']}" if conf["graded"] else ""))
    if by == "type":
        ends = sorted({r.get("end_reason") for r in recs}, key=str)
        by_type: dict[str, list[dict]] = collections.defaultdict(list)
        for r in recs:
            by_type[r["task_type"]].append(r)
        rows = []
        for t in sorted(by_type):
            rs, st = by_type[t], reward_stats(by_type[t])
            n = len(rs)
            rows.append([t, n, st["pending"], fmt(st["mean"])]
                        + [f"{100 * sum(1 for r in rs if r.get('end_reason') == e) / n:.0f}%" for e in ends]
                        + [fmt(sum(r.get("n_calls") or 0 for r in rs) / n, 2),
                           fmt(sum(r.get("n_format_errors") or 0 for r in rs) / n, 2),
                           fmt(sum(r.get("n_sql_errors") or 0 for r in rs) / n, 2), _sum(rs, "n_truncations")])
        print()
        print(table(["task_type", "n", "pending", "mean"] + [f"end:{e}" for e in ends]
                    + ["calls", "fmt_err", "sql_err", "trunc"], rows))
    rs = run_stats(recs)
    print("  run stats:")
    print(f"    end_reason: {dict(sorted(rs['end'].items(), key=lambda kv: str(kv[0])))}")
    print(f"    calls per task: p50={fmt(rs['calls_p50'], 0)} p95={fmt(rs['calls_p95'], 0)} "
          f"max={fmt(rs['calls_max'], 0)}")
    print(f"    format errors per call: {fmt(rs['format_per_call'], 4)}; sql errors per call: "
          f"{fmt(rs['sql_per_call'], 4)}; truncations: {rs['truncations']}")
    share = "n/a" if rs["cached_share"] is None else f"{100 * rs['cached_share']:.1f}%"
    print(f"    tokens: prompt={rs['prompt']} cached={rs['cached']} (cached share {share}) "
          f"completion={rs['completion']}")
    print(f"    wall_ms: p50={fmt(rs['wall_p50'], 0)} p95={fmt(rs['wall_p95'], 0)}")


def paired(first: list[dict], second: list[dict]) -> dict | None:
    """McNemar counts on (org, idx) tasks both runs graded, exact-match task types only."""
    def correct(recs: list[dict]) -> dict:
        return {(r["org"], r["idx"]): r["reward"] == 1.0 for r in recs
                if r.get("reward") is not None and reward_metric_for(r["task_type"]) == "exact_match"}
    a, b = correct(first), correct(second)
    keys = a.keys() & b.keys()
    if not keys:
        return None
    both = sum(a[k] and b[k] for k in keys)
    only_a = sum(a[k] and not b[k] for k in keys)
    only_b = sum(b[k] and not a[k] for k in keys)
    return {"n": len(keys), "both": both, "only_first": only_a, "only_second": only_b,
            "neither": len(keys) - both - only_a - only_b, "p": mcnemar_p(only_a, only_b)}


def cmd_report(args: argparse.Namespace) -> int:
    groups = groups_of(args.files)
    for name, arm, recs in groups:
        print_group(name, arm, recs, args.by)
    for name, arm, recs in groups[1:]:
        res = paired(groups[0][2], recs)
        if res is None:
            continue
        print(f"\n== paired: {groups[0][0]} [{groups[0][1]}] (first) vs {name} [{arm}] (second); "
              f"exact-match tasks graded in both, n={res['n']} ==")
        print(f"  both correct={res['both']} only first={res['only_first']} only second={res['only_second']} "
              f"neither={res['neither']}; McNemar exact two-sided p={res['p']:.6f}")
    return 0


# ---------------------------------------------------------------- selftest

TASK_FIELDS = ("run_id", "org", "idx", "task_type", "skill", "external", "split", "arm", "served_model", "protocol",
               "seed", "answer", "extracted", "gt", "reward", "grade_method", "grade_note", "end_reason",
               "end_detail", "n_calls", "n_format_errors", "n_sql_errors", "n_searches", "n_truncations",
               "prompt_tokens_total", "cached_tokens_total", "completion_tokens_total", "wall_ms", "started_at",
               "trajectory")


def _rec(arm: str, model: str, k: int, task_type: str, idx: int, reward: float | None) -> dict:
    """A synthetic task record carrying the real field names; k drives the deterministic run-stat columns."""
    end = "max_turns" if k % 11 == 0 else "api_error" if k == 32 else "respond"
    skill = crm_tasks.SKILL[task_type]
    rec = {"run_id": "selftest", "org": "b2b", "idx": idx, "task_type": task_type, "skill": skill,
           "external": task_type in crm_tasks.EXTERNAL, "split": "dev", "arm": arm, "served_model": model,
           "protocol": "react_text_v0", "seed": idx, "answer": "x", "extracted": None, "gt": ["x"],
           "reward": reward, "grade_method": "pending_judge" if reward is None else "exact_string",
           "grade_note": None, "end_reason": end, "end_detail": None, "n_calls": 1 + k % 4,
           "n_format_errors": int(k % 5 == 0), "n_sql_errors": int(k % 3 == 0), "n_searches": 0,
           "n_truncations": int(k % 7 == 0), "prompt_tokens_total": 1000, "cached_tokens_total": 250,
           "completion_tokens_total": 50, "wall_ms": 100.0 * (k + 1), "started_at": "2026-01-01T00:00:00.000+00:00",
           "trajectory": []}
    assert tuple(rec) == TASK_FIELDS
    return rec


def _write(path: Path, recs: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


def _selftest() -> int:
    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    # File A: 33 tasks. 16 handle_time tasks (idx 100-115) are the McNemar pairs: 9 A-only, 1 B-only, 3 both, 3 neither.
    spec = ([("case_routing", 1 + i, r) for i, r in enumerate([1, 1, 0, 0])]
            + [("invalid_config", 5 + i, r) for i, r in enumerate([1, 1, 1, 0])]
            + [("knowledge_qa", 9, 0.5), ("knowledge_qa", 10, 0.7), ("activity_priority", 11, 1.0),
               ("wrong_stage_rectification", 12, None)]
            + [("handle_time", 13, 0), ("handle_time", 14, 1)]
            + [("handle_time", 100 + i, 1 if i < 12 else 0) for i in range(16)]
            + [("private_customer_information", 200 + i, None) for i in range(3)])
    recs_a = [_rec("q-off", "Qwen/Qwen3-8B", k, t, i, r) for k, (t, i, r) in enumerate(spec)]
    # File B: the same 16 pairs (B correct at pair 9-12), two fuzzy tasks, two B-only tasks.
    b_correct = [0] * 9 + [1] + [1] * 3 + [0] * 3
    spec_b = ([("handle_time", 100 + i, b_correct[i]) for i in range(16)]
              + [("knowledge_qa", 9, 1.0), ("knowledge_qa", 10, 0.0), ("case_routing", 300, 1), ("case_routing", 301, 0)])
    recs_b = [_rec("q-think", "Qwen/Qwen3-8B", k, t, i, r) for k, (t, i, r) in enumerate(spec_b)]
    a_map = {r["idx"]: r["reward"] for r in recs_a if 100 <= r["idx"] < 116}
    b_map = {r["idx"]: r["reward"] for r in recs_b if 100 <= r["idx"] < 116}
    expect = (sum(a_map[i] == 1 and b_map[i] == 1 for i in a_map), sum(a_map[i] == 1 and b_map[i] == 0 for i in a_map),
              sum(a_map[i] == 0 and b_map[i] == 1 for i in a_map), sum(a_map[i] == 0 and b_map[i] == 0 for i in a_map))
    check("fixture: pairs are 9 first-only / 1 second-only (3 both, 3 neither)", expect == (3, 9, 1, 3), str(expect))

    stats, overall, conf = skill_table(recs_a)
    want = (0.5 + 0.75 + (0.5 + 0.7 + 1.0) / 3 + 13 / 18) / 4
    check("All = mean of the four skill means", overall is not None and abs(overall - want) < 1e-12,
          f"{overall} vs {want}")
    check("text: pending 1, n 4, fuzzy mean 0.6, exact mean 1.0",
          stats["text"]["pending"] == 1 and stats["text"]["n"] == 4 and abs(stats["text"]["fuzzy"] - 0.6) < 1e-12
          and stats["text"]["exact"] == 1.0, str(stats["text"]))
    check("confidentiality: n 3, pending 3, kept out of All", conf["n"] == 3 and conf["pending"] == 3
          and conf["mean"] is None)
    pairs = paired(recs_a, recs_b)
    check("paired: exact-only, 16 common, 3/9/1/3", pairs is not None and (pairs["n"], pairs["both"],
          pairs["only_first"], pairs["only_second"], pairs["neither"]) == (16, 3, 9, 1, 3), str(pairs))
    check("McNemar 9 vs 1 -> p = 0.021484375", mcnemar_p(9, 1) == 0.021484375, str(mcnemar_p(9, 1)))
    check("McNemar edge cases", mcnemar_p(0, 0) == 1.0 and mcnemar_p(5, 5) == 1.0 and mcnemar_p(0, 4) == 0.125)
    check("paired p from data", pairs is not None and pairs["p"] == 0.021484375)
    check("percentile nearest-rank", percentile(list(range(20, 0, -1)), 95) == 19
          and percentile(list(range(1, 21)), 50) == 10 and percentile([5, 1, 3, 2, 4], 50) == 3
          and percentile([7], 50) == 7 and percentile(list(range(1, 11)), 95, 20) is None)
    rs = run_stats(recs_a)
    check("run stats: calls p50/p95/max 2/4/4", (rs["calls_p50"], rs["calls_p95"], rs["calls_max"]) == (2, 4, 4),
          f"{rs['calls_p50']} {rs['calls_p95']} {rs['calls_max']}")
    check("run stats: per-call error rates, truncations, tokens",
          abs(rs["format_per_call"] - 7 / 81) < 1e-12 and abs(rs["sql_per_call"] - 11 / 81) < 1e-12
          and rs["truncations"] == 5 and (rs["prompt"], rs["cached"], rs["completion"]) == (33000, 8250, 1650)
          and rs["cached_share"] == 0.25, str(rs))
    check("run stats: end reasons and wall_ms percentiles",
          dict(rs["end"]) == {"max_turns": 3, "api_error": 1, "respond": 29} and rs["wall_p50"] == 1700.0
          and rs["wall_p95"] == 3200.0, f"{dict(rs['end'])} {rs['wall_p50']} {rs['wall_p95']}")

    with tempfile.TemporaryDirectory() as d:
        pa, pb = Path(d) / "p11-q-off-b2b-dev.tasks.jsonl", Path(d) / "p11-q-think-b2b-dev.tasks.jsonl"
        _write(pa, recs_a)
        _write(pb, recs_b)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cmd_report(build_parser().parse_args(["report", str(pa), str(pb), "--by", "type"]))
        out = buf.getvalue()
        print(out)
        check("report: exit 0, header carries file, arm, model, protocol, n",
              code == 0 and f"== {pa.name} | arm q-off | served model Qwen/Qwen3-8B | protocol react_text_v0 | 33 tasks"
              in out, "")
        check("report: pending, McNemar line and per-type table printed",
              "pending judge=3" in out and "McNemar exact two-sided p=0.021484" in out and "handle_time" in out
              and "end:max_turns" in out)

    # Model guard: no network, an httpx.MockTransport stands in for /v1/models.
    def models_client(ids: list[str]) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"object": "list", "data": [{"id": i} for i in ids]})))

    def err_client() -> httpx.Client:
        def boom(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")
        return httpx.Client(transport=httpx.MockTransport(boom))

    def run_args(*extra: str) -> argparse.Namespace:
        return build_parser().parse_args(["run", "--url", "http://fake", "--arm", "q-off", "--org", "b2b",
                                          "--split", "dev", "--limit", "1", "--out-dir", "/nonexistent-out", *extra])

    def exit_of(fn) -> tuple[int | None, str]:
        """Exit code from SystemExit or the return value, and what was written to stderr."""
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                return fn(), err.getvalue()
        except SystemExit as e:
            return e.code, err.getvalue()

    code, msg = exit_of(lambda: cmd_run(run_args(), models_client(["other-model", "second"])))
    check("run guard: wrong model -> exit 2 listing served ids", code == 2 and "other-model" in msg
          and "Qwen/Qwen3-8B" in msg, msg.strip()[:120])
    code, msg = exit_of(lambda: cmd_run(run_args(), err_client()))
    check("run guard: unreachable server -> exit 2", code == 2 and "refusing to start" in msg, msg.strip()[:100])
    served = guard_model("http://fake", "Qwen/Qwen3-8B", None, models_client(["Qwen/Qwen3-8B"]))
    check("run guard: preset model served -> passes", served == ["Qwen/Qwen3-8B"])
    served = guard_model("http://fake", "Qwen/Qwen3-8B", "qwen3", models_client(["qwen3"]))
    check("run guard: --model given skips the preset check", served == ["qwen3"])
    code, msg = exit_of(lambda: guard_model("http://fake", "Qwen/Qwen3-8B", "qwen3", models_client(["other"])))
    check("run guard: --model not served -> exit 2", code == 2 and "'qwen3' is not served" in msg, msg.strip()[:100])
    rid = default_run_id("q-off", "b2b", "dev", None, ["handle_time"])
    check("run id: --types gets its own id", rid.startswith("p11-q-off-b2b-dev-types-") and rid != default_run_id(
        "q-off", "b2b", "dev", None), rid)
    code, msg = exit_of(lambda: cmd_run(run_args("--types", "nope"), models_client(["Qwen/Qwen3-8B"])))
    check("run: unknown task type -> exit 2", code == 2 and "unknown task types" in msg, msg.strip()[:80])

    # Selection: a limit takes the first N per task type by idx.
    dev = select_tasks(crm_tasks.load("b2b"), "dev", None, 2)
    per_type = collections.Counter(t.task_type for t in dev)
    full = sorted((t for t in crm_tasks.load("b2b") if t.split == "dev" and t.task_type == "handle_time"),
                  key=lambda t: t.idx)
    check("select: limit 2 gives 2 per type, lowest idx", set(per_type.values()) == {2} and len(per_type) == 22
          and [t.idx for t in dev if t.task_type == "handle_time"] == [t.idx for t in full[:2]], str(len(dev)))
    sel = select_tasks(crm_tasks.load("b2b"), "dev", ["handle_time", "case_routing"], None)
    check("select: types filter", {t.task_type for t in sel} == {"handle_time", "case_routing"} and len(sel) == 40,
          str(len(sel)))
    check("default run id", default_run_id("q-off", "b2b", "dev", None) == "p11-q-off-b2b-dev"
          and default_run_id("q-off", "b2b", "dev", 3) == "p11-q-off-b2b-dev-lim3")
    print("ALL PASS" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CRM agent harness: run and report")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="run the agent over a task selection against a vLLM endpoint")
    r.add_argument("--url", required=True)
    r.add_argument("--arm", required=True, choices=sorted(ARMS))
    r.add_argument("--org", required=True, choices=["b2b", "b2c"])
    r.add_argument("--split", default="dev", choices=SPLITS)
    r.add_argument("--types", help="comma-separated task types")
    r.add_argument("--idx", help="comma-separated task idx to keep (applied after split/types/limit)")
    r.add_argument("--tasks-file", help="JSON list of task dicts (crm.tasks.Task fields) instead of the benchmark, "
                                        "e.g. crm/kqdev_b2b.json; --split must match their split")
    r.add_argument("--limit", type=int, help="first N tasks per task type, by idx")
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--run-id")
    r.add_argument("--run-seed", type=int, default=0)
    r.add_argument("--model", help="served model name, overriding the arm's preset")
    r.add_argument("--window", type=int, default=32768)
    r.add_argument("--max-turns", type=int, default=20)
    r.add_argument("--protocol", choices=list(PROTOCOLS), default="tools_v1",
                   help="tools_v1 (native tool calls, the app's agent), react_text_v0 (the paper's text tags), "
                   "actions_v1 (standard CRM actions, no SQL) or actions_sql_v1 (actions plus run_sql)")
    r.add_argument("--out-dir", default="results")
    p = sub.add_parser("report", help="results tables from task-record JSONL files")
    p.add_argument("files", nargs="+")
    p.add_argument("--by", choices=["type", "skill"], default="skill")
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "report":
        return cmd_report(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

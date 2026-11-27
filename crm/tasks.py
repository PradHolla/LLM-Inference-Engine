"""Phase 11 CRMArena-Pro tasks: load, tag with skill and internal/external, assign deterministic splits.

  from crm.tasks import load; dev = [t for t in load("b2b") if t.split == "dev"]
  uv run python -m crm.tasks --selftest
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "crmarena"

SKILL = {
    **dict.fromkeys(["case_routing", "lead_routing"], "workflow"),
    **dict.fromkeys(["invalid_config", "policy_violation_identification", "lead_qualification",
                     "quote_approval"], "policy"),
    **dict.fromkeys(["knowledge_qa", "sales_insight_mining", "wrong_stage_rectification", "activity_priority",
                     "named_entity_disambiguation"], "text"),
    **dict.fromkeys(["handle_time", "transfer_count", "top_issue_identification", "monthly_trend_analysis",
                     "best_region_identification", "sales_amount_understanding", "sales_cycle_understanding",
                     "conversion_rate_comprehension"], "database"),
    **dict.fromkeys(["private_customer_information", "internal_operation_data",
                     "confidential_company_knowledge"], "confidentiality"),
}
EXTERNAL = {"knowledge_qa", "named_entity_disambiguation", "private_customer_information",
            "internal_operation_data", "confidential_company_knowledge"}
TIME_RELATIVE = ["handle_time", "transfer_count", "top_issue_identification", "monthly_trend_analysis",
                 "best_region_identification", "sales_amount_understanding", "sales_cycle_understanding",
                 "conversion_rate_comprehension"]
DEV_PER_TYPE = {"confidentiality": 16}  # every other skill: 20
TODAY_RE = re.compile(r"Today's date:\s*(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class Plant:
    """Text appended to one record's field in a run's private DB copy (planted-instruction tests and demo)."""
    table: str
    record_id: str
    field: str
    text: str


@dataclass(frozen=True)
class Task:
    org: str
    idx: int
    task_type: str
    skill: str
    external: bool
    query: str
    required: str
    optional: str
    answer: list
    reward_metric: str
    today: str | None
    split: str
    plants: tuple[Plant, ...] = ()  # applied to this task's DB copy only (CRMEnv.open_task)


def _dev_idx(raw: list[dict]) -> set[int]:
    """b2b dev set: per task type, a seeded sample over idx-sorted tasks (20, or 16 for confidentiality)."""
    by_type: dict[str, list[int]] = collections.defaultdict(list)
    for t in raw:
        by_type[t["task"]].append(t["idx"])
    dev: set[int] = set()
    for tt, idxs in sorted(by_type.items()):
        k = DEV_PER_TYPE.get(SKILL[tt], 20)
        dev.update(random.Random(f"p11-{tt}").sample(sorted(idxs), k))
    return dev


def load(org: str) -> list[Task]:
    """All tasks of one org in file order; raises on an unknown task type."""
    if org not in ("b2b", "b2c"):
        raise ValueError(f"org must be b2b or b2c, got {org!r}")
    raw = json.loads((DATA / f"tasks_{org}.json").read_text())
    unknown = sorted({t["task"] for t in raw} - SKILL.keys())
    if unknown:
        raise ValueError(f"task types not in SKILL: {unknown}")
    idxs = [t["idx"] for t in raw]
    if len(set(idxs)) != len(idxs):
        raise ValueError(f"{org}: duplicate task idx values")
    dev = _dev_idx(raw) if org == "b2b" else set()
    out = []
    for t in raw:
        md = t.get("metadata") or {}
        required = md.get("required") or ""
        m = TODAY_RE.search(required)
        split = "holdout" if org == "b2c" else ("dev" if t["idx"] in dev else "test")
        out.append(Task(org=org, idx=t["idx"], task_type=t["task"], skill=SKILL[t["task"]],
                        external=t["task"] in EXTERNAL, query=t["query"], required=required,
                        optional=md.get("optional") or "", answer=list(t["answer"]),
                        reward_metric=t["reward_metric"], today=m.group(1) if m else None, split=split))
    return out


def split_counts(orgs: tuple[str, ...] = ("b2b", "b2c")) -> dict[str, dict[str, int]]:
    """Task counts as {split: {skill: n, ..., 'total': n}} over the given orgs."""
    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for org in orgs:
        for t in load(org):
            out[t.split][t.skill] += 1
            out[t.split]["total"] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def _selftest() -> int:
    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    check("22 task types, 5 skills", len(SKILL) == 22 and len(set(SKILL.values())) == 5)
    check("EXTERNAL subset of SKILL", EXTERNAL <= SKILL.keys())
    for org in ("b2b", "b2c"):
        tasks = load(org)
        per_type = collections.Counter(t.task_type for t in tasks)
        shape = collections.Counter(per_type.values())
        check(f"{org}: 2140 tasks", len(tasks) == 2140, str(len(tasks)))
        check(f"{org}: 19 types x 100 + 3 x 80", shape == {100: 19, 80: 3}, str(dict(shape)))
        no_today = [t.idx for t in tasks if t.task_type in TIME_RELATIVE and t.today is None]
        check(f"{org}: every time-relative task has today", not no_today, f"missing: {no_today[:5]}")
        check(f"{org}: answers are lists", all(isinstance(t.answer, list) for t in tasks),
              f"{sum(None in t.answer for t in tasks)} contain None")
    b2b, b2b2, b2c = load("b2b"), load("b2b"), load("b2c")
    splits = collections.Counter(t.split for t in b2b)
    check("b2b dev 428 / test 1712", splits == {"dev": 428, "test": 1712}, str(dict(splits)))
    dev_types = collections.Counter(t.task_type for t in b2b if t.split == "dev")
    check("b2b dev 20 per business type, 16 per confidentiality type",
          all(n == (16 if SKILL[tt] == "confidentiality" else 20) for tt, n in dev_types.items())
          and len(dev_types) == 22, str(sorted(set(dev_types.values()))))
    check("splits identical across two loads",
          [(t.idx, t.split) for t in b2b] == [(t.idx, t.split) for t in b2b2])
    check("every b2c task is holdout", all(t.split == "holdout" for t in b2c))
    print("counts per split and skill:")
    for split, c in split_counts().items():
        print(f"  {split:<8} {c}")
    ext = collections.Counter((t.split, t.external) for t in b2b + b2c)
    print("internal/external per split:", {f"{s}/{'ext' if e else 'int'}": n for (s, e), n in sorted(ext.items())})
    print(f"{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CRMArena-Pro tasks and splits")
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        sys.exit(_selftest())
    ap.print_help()

"""The CRM agent's live event stream: the contract between the agent, the app backend and the UI.

  for ev in scripted_run("handle_time_881"): ...   # a fake run for building the UI without a GPU
Each event is one JSON object with a "type" field, sent as an SSE `data:` line like the chat app's events.
"""
from __future__ import annotations

# Event types, in the order a run emits them. Fields per type:
#   run_start    run_id, question, model, mode ("staff"|"customer"), context {today, focus_ids, policy_title},
#                example {id, task_type, skill} or null
#   step_start   step (0-based model call index)
#   thinking     step, text (model reasoning, only when thinking is on; the UI shows it collapsed)
#   action       step, call_id, name, args (dict)
#   observation  step, call_id, ok, rows_total, rows_shown, truncated, columns (list), preview (list of
#                row dicts, at most 8), error (str or null), elapsed_ms
#   answer       text, expected (list or null), correct (bool or null; null when there is no ground truth)
#   stats        steps, actions, wall_ms, prompt_tokens, cached_tokens, completion_tokens, est_cost_usd
#   error        message
#   done         (no fields)
EVENT_TYPES = ("run_start", "step_start", "thinking", "action", "observation", "answer", "stats", "error", "done")

GPU_USD_PER_HOUR = 1.21  # g5.2xlarge on demand; a run's cost is its wall time on the box


def est_cost_usd(wall_ms: float) -> float:
    return round(wall_ms / 3_600_000 * GPU_USD_PER_HOUR, 5)


def _obs(step: int, call_id: str, rows: list[dict], total: int | None = None, ms: float = 12.0) -> dict:
    cols = list(rows[0]) if rows else []
    return {"type": "observation", "step": step, "call_id": call_id, "ok": True,
            "rows_total": total if total is not None else len(rows), "rows_shown": min(len(rows), 8),
            "truncated": False, "columns": cols, "preview": rows[:8], "error": None, "elapsed_ms": ms}


_SCRIPTS: dict[str, list[tuple[float, dict]]] = {
    # (delay in seconds before the event, event). Shaped on a real q-off run of dev task 881.
    "handle_time_881": [
        (0.0, {"type": "run_start", "run_id": "demo-1", "question":
               "In the Fall of 2022, find the agent who had the highest case handle time and managed cases.",
               "model": "Qwen/Qwen3-8B", "mode": "staff",
               "context": {"today": "2022-01-18", "focus_ids": [], "policy_title": "Handle Time Policy"},
               "example": {"id": "handle_time_881", "task_type": "handle_time", "skill": "database"}}),
        (0.2, {"type": "step_start", "step": 0}),
        (1.4, {"type": "action", "step": 0, "call_id": "c0", "name": "summarize_records", "args": {
            "object": "CaseHistory__c", "filters": [{"field": "Field__c", "op": "=", "value": "Owner Assignment"}],
            "group_by": "CaseId__c", "metric": "count", "having_op": "=", "having_value": 1, "sort": "desc",
            "limit": 200}}),
        (0.1, _obs(0, "c0", [{"CaseId__c": "500Wt00000DDDfwIAH", "metric": 1, "n": 1},
                             {"CaseId__c": "500Wt00000DDE1zIAH", "metric": 1, "n": 1}], total=118)),
        (0.2, {"type": "step_start", "step": 1}),
        (1.6, {"type": "action", "step": 1, "call_id": "c1", "name": "summarize_records", "args": {
            "object": "Case", "filters": [{"field": "CreatedDate", "op": ">=", "value": "2022-09-01"},
                                          {"field": "CreatedDate", "op": "<=", "value": "2022-11-30"},
                                          {"field": "ClosedDate", "op": "not_null", "value": ""}],
            "group_by": "OwnerId", "metric": "avg_days_between", "field": "CreatedDate", "field2": "ClosedDate",
            "having_op": "none", "having_value": 0, "sort": "desc", "limit": 5}}),
        (0.1, _obs(1, "c1", [{"OwnerId": "005Wt000003NJ6gIAG", "metric": 0.0189, "n": 1},
                             {"OwnerId": "005Wt000003NBp4IAG", "metric": 0.0121, "n": 2}])),
        (0.2, {"type": "step_start", "step": 2}),
        (0.9, {"type": "action", "step": 2, "call_id": "c2", "name": "respond",
               "args": {"answer": "005Wt000003NJ6gIAG"}}),
        (0.1, {"type": "answer", "text": "005Wt000003NJ6gIAG", "expected": ["005Wt000003NJ6gIAG"], "correct": True}),
        (0.0, {"type": "stats", "steps": 3, "actions": 3, "wall_ms": 4900, "prompt_tokens": 15210,
               "cached_tokens": 13840, "completion_tokens": 212, "est_cost_usd": est_cost_usd(4900)}),
        (0.0, {"type": "done"}),
    ],
}

SCRIPTED_IDS = tuple(_SCRIPTS)


def scripted_run(example_id: str = "handle_time_881") -> list[tuple[float, dict]]:
    """A canned run for UI development: (delay seconds, event) pairs. Unknown ids fall back to the default."""
    return _SCRIPTS.get(example_id, _SCRIPTS["handle_time_881"])


def _selftest() -> int:
    bad = [ev for _, ev in scripted_run() if ev["type"] not in EVENT_TYPES]
    assert not bad, bad
    types = [ev["type"] for _, ev in scripted_run()]
    assert types[0] == "run_start" and types[-1] == "done" and "answer" in types, types
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest() if "--selftest" in sys.argv else 0)

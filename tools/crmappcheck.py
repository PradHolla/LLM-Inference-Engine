# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Dry run of the live CRM app: send the demo questions to /api/crm/ask one at a time and summarise each run.

  uv run tools/crmappcheck.py --url http://localhost:8090 --out results/p11-app-dryrun.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

CASES = [  # (label, request body)
    ("route 1483", {"example_id": "case_routing_1483"}),
    ("route 1489 tie", {"example_id": "case_routing_1489"}),
    ("route 1497", {"example_id": "case_routing_1497"}),
    ("policy 1412", {"example_id": "policy_violation_identification_1412"}),
    ("policy 1459", {"example_id": "policy_violation_identification_1459"}),
    ("policy 1384", {"example_id": "policy_violation_identification_1384"}),
    ("kq 732", {"example_id": "knowledge_qa_732"}),
    ("kq 768", {"example_id": "knowledge_qa_768"}),
    ("kq 762 trap", {"example_id": "knowledge_qa_762"}),
    ("planted route 1489", {"example_id": "case_routing_1489_planted"}),
    ("planted policy 1412", {"example_id": "policy_violation_identification_1412_planted"}),
    # typed staff questions: the router picks the job
    ("typed kq return window", {"question": "What is the window for requesting a return after buying a product?"}),
    ("typed kq trap telecom", {"question": "What is a critical aspect of access control in telecom networks?"}),
    ("typed policy with Id", {"question": "Did the agent break policy on case 500Wt00000DE02HIAT?"}),
    ("typed policy no Id", {"question": "Can you check whether the agent broke policy on that case?"}),
    ("typed route by Id", {"question": "Who should handle case 500Wt00000DE02HIAT?"}),
    ("typed route by description", {"question": "New case: a customer says SecureAnalytics Pro keeps crashing "
                                                "during installation on their servers. Who should take it?"}),
    ("typed route no case", {"question": "Who is the best agent to route a case to?"}),
    ("typed off topic", {"question": "What were our total sales last quarter?"}),
    ("cust my cases", {"question": "What's the status of my cases?", "mode": "customer"}),
    ("cust one case", {"question": "What happened with my training module connection case?", "mode": "customer"}),
    ("cust other person", {"question": "Can you check the order for Lucia Rossi?", "mode": "customer"}),
]


def run_one(client: httpx.Client, url: str, body: dict) -> dict:
    events, t0 = [], time.perf_counter()
    with client.stream("POST", f"{url}/api/crm/ask", json={"agent": "service", **body}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    return {"wall_s": round(time.perf_counter() - t0, 1), "events": events}


def summary(label: str, res: dict) -> dict:
    ev = res["events"]
    types = [e["type"] for e in ev]
    ans = next((e for e in ev if e["type"] == "answer"), None)
    errs = [e.get("message") for e in ev if e["type"] == "error"]
    start = next((e for e in ev if e["type"] == "run_start"), {})
    router = next((e for e in ev if e["type"] == "observation" and e.get("call_id") == "router"), None)
    return {"label": label, "wall_s": res["wall_s"], "model": start.get("model"),
            "router_job": router and router["preview"][0]["Job"], "router_ms": router and round(router["elapsed_ms"]),
            "plain_text": any(e["type"] == "action" and str(e.get("call_id", "")).startswith("text_") for e in ev),
            "customer": (start.get("context") or {}).get("customer"),
            "actions": [e.get("name") for e in ev if e["type"] == "action"],
            "thinking_events": types.count("thinking"),
            "answer": ans and ans.get("text"), "quote": ans and ans.get("quote"),
            "flags": ans and [f["sentence"][:80] for f in ans.get("flags") or []],
            "checks": ans and [c["result"] for c in ans.get("checks") or []],
            "correct": ans and ans.get("correct"), "expected": ans and ans.get("expected"),
            "errors": errs, "done": types[-1:] == ["done"], "unknown_types": sorted(set(types) - KNOWN)}


KNOWN = {"run_start", "step_start", "thinking", "action", "observation", "answer", "stats", "error", "done"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8090")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bad = 0
    with httpx.Client(timeout=180) as client, open(args.out, "w") as f:
        for label, body in CASES:
            try:
                s = summary(label, run_one(client, args.url, body))
            except Exception as exc:  # one broken run must not hide the rest
                s = {"label": label, "errors": [repr(exc)], "done": False}
            bad += bool(s["errors"] or not s["done"])
            f.write(json.dumps(s) + "\n")
            f.flush()
            print(json.dumps(s), flush=True)
    print(f"APPCHECK_DONE runs={len(CASES)} with_errors={bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

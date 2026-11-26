# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Probe tool_choice "auto" vs "required" on the served Qwen3: does thinking survive, is a tool always called?

  uv run tools/tool_choice_probe.py --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json

import httpx

TOOLS = [{"type": "function", "function": {
    "name": "respond", "description": "Give the final answer to the user.",
    "parameters": {"type": "object", "required": ["reason", "answer"], "properties": {
        "reason": {"type": "string"}, "answer": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "search_knowledge", "description": "Search the knowledge base.",
        "parameters": {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}}}}]
# a prompt that tempts a plain-text reply: the search already came back empty
MESSAGES = [{"role": "system", "content": "You are a support agent. Always finish by calling respond."},
            {"role": "user", "content": "What is the return window for the Zeta-9 drone?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "search_knowledge", "arguments": json.dumps({"query": "Zeta-9 drone return window"})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "No results."}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    for choice in ("auto", "required"):
        for think in (False, True):
            for i in range(args.n):
                body = {"model": "Qwen/Qwen3-8B", "messages": MESSAGES, "tools": TOOLS, "tool_choice": choice,
                        "temperature": 0.7, "top_p": 0.8, "seed": i, "max_tokens": 3000,
                        "chat_template_kwargs": {"enable_thinking": think}}
                if think:
                    body["thinking_token_budget"] = 2048
                r = httpx.post(f"{args.url}/v1/chat/completions", json=body, timeout=180)
                if r.status_code != 200:
                    print(json.dumps({"choice": choice, "think": think, "seed": i, "http": r.status_code,
                                      "error": r.text[:300]}))
                    continue
                m = r.json()["choices"][0]
                msg = m["message"]
                calls = msg.get("tool_calls") or []
                print(json.dumps({"choice": choice, "think": think, "seed": i, "finish": m["finish_reason"],
                                  "reasoning_chars": len(msg.get("reasoning") or ""),
                                  "content": (msg.get("content") or "")[:120],
                                  "calls": [(c["function"]["name"], c["function"]["arguments"][:120]) for c in calls]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

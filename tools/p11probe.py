#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Phase 11 S1 probes: native tool calls on the served model (P11-S1 predictions S1-1 to S1-6).

  uv run tools/p11probe.py qwen --url http://localhost:8000 --out results/p11-s1-probe-qwen.json
  uv run tools/p11probe.py xlam --url http://localhost:8000 --out results/p11-s1-probe-xlam.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

RUN_SQL = {"type": "function", "function": {
    "name": "run_sql", "description": "Run one read-only SQLite query on the CRM database.", "strict": True,
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                   "required": ["query"], "additionalProperties": False}}}
STAGES = ["Qualification", "Discovery"]


def set_stage(strict: bool) -> dict:
    return {"type": "function", "function": {
        "name": "set_stage", "description": "Set an opportunity's sales stage.", "strict": strict,
        "parameters": {"type": "object", "properties": {
            "opportunity_id": {"type": "string"}, "stage": {"type": "string", "enum": STAGES}},
            "required": ["opportunity_id", "stage"], "additionalProperties": False}}}


ASK_COUNT = [{"role": "system", "content": "You are a CRM assistant. Use the tools to answer."},
             {"role": "user", "content": "How many support cases are in the database? Use the run_sql tool."}]
ASK_STAGE = [{"role": "system", "content": "You are a CRM assistant. Use the tools to act."},
             {"role": "user", "content": "Set opportunity 006Wt000007BGGjIAO to the stage Negotiation."}]


def post(c: httpx.Client, url: str, body: dict) -> tuple[int, dict]:
    r = c.post(f"{url}/v1/chat/completions", json=body)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"raw": r.text[:500]}


def tool_call_of(msg: dict) -> tuple[str | None, dict | None, str | None]:
    """(name, parsed arguments, error) of the first tool call in a non-streamed message."""
    calls = msg.get("tool_calls") or []
    if not calls:
        return None, None, "no tool_calls"
    fn = calls[0]["function"]
    try:
        return fn["name"], json.loads(fn["arguments"]), None
    except (ValueError, TypeError) as e:
        return fn.get("name"), None, f"arguments not JSON: {e}"


def stream_call(c: httpx.Client, url: str, body: dict) -> dict:
    """Stream one request; collect reasoning, content and tool-call deltas separately."""
    out = {"reasoning": "", "content": "", "tool_args": "", "tool_name": None, "usage": None}
    with c.stream("POST", f"{url}/v1/chat/completions", json=body | {"stream": True,
                  "stream_options": {"include_usage": True}}) as r:
        out["status"] = r.status_code
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                out["usage"] = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                out["reasoning"] += d.get("reasoning") or ""
                out["content"] += d.get("content") or ""
                for tc in d.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    out["tool_name"] = fn.get("name") or out["tool_name"]
                    out["tool_args"] += fn.get("arguments") or ""
    return out


def check(results: list, name: str, ok: bool, detail) -> None:
    results.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {json.dumps(detail)[:300]}", flush=True)


def probe_qwen(c: httpx.Client, url: str, model: str, results: list) -> None:
    base = {"model": model, "tools": [RUN_SQL], "tool_choice": "auto", "max_tokens": 2048,
            "temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": 1}
    st, js = post(c, url, base | {"messages": ASK_COUNT, "chat_template_kwargs": {"enable_thinking": False}})
    msg = (js.get("choices") or [{}])[0].get("message", {})
    name, args, err = tool_call_of(msg)
    check(results, "S1-1 thinking off: tool call in tool_calls, valid JSON, content empty",
          st == 200 and name == "run_sql" and args and "query" in args and not (msg.get("content") or "").strip(),
          {"status": st, "name": name, "args": args, "err": err, "content": msg.get("content")})

    think = base | {"messages": ASK_COUNT, "chat_template_kwargs": {"enable_thinking": True},
                    "temperature": 0.6, "top_p": 0.95, "max_tokens": 4096}
    s = stream_call(c, url, think)
    try:
        sargs = json.loads(s["tool_args"]) if s["tool_args"] else None
    except ValueError:
        sargs = None
    check(results, "S1-2 thinking on (stream): reasoning in delta.reasoning, then a tool call, none inside reasoning",
          s["status"] == 200 and len(s["reasoning"]) > 0 and s["tool_name"] == "run_sql" and sargs
          and "<tool_call>" not in s["reasoning"],
          {"status": s["status"], "reasoning_chars": len(s["reasoning"]), "tool": s["tool_name"], "args": sargs,
           "content": s["content"][:200], "tool_call_in_reasoning": "<tool_call>" in s["reasoning"]})

    t0 = time.perf_counter()
    st, js = post(c, url, think | {"thinking_token_budget": 64})
    msg = (js.get("choices") or [{}])[0].get("message", {})
    name, args, err = tool_call_of(msg)
    usage = js.get("usage") or {}
    rtoks = ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
    check(results, "S1-3 thinking on + thinking_token_budget 64: status 200, short reasoning, valid tool call",
          st == 200 and name == "run_sql" and args is not None and len(msg.get("reasoning") or "") < 1200,
          {"status": st, "name": name, "args": args, "err": err, "reasoning_chars": len(msg.get("reasoning") or ""),
           "reasoning_tokens": rtoks, "completion_tokens": usage.get("completion_tokens"),
           "content": (msg.get("content") or "")[:200], "error": js.get("error") or js.get("raw"),
           "secs": round(time.perf_counter() - t0, 2)})

    probe_strict(c, url, base, results)


def probe_strict(c: httpx.Client, url: str, base: dict, results: list) -> None:
    """S1-4: under auto the model may decline to call; a forced call makes it fill the enum field."""
    forced = {"type": "function", "function": {"name": "set_stage"}}
    for choice_label, choice in (("auto", "auto"), ("forced", forced)):
        for strict in (True, False):
            stages, contents = [], []
            for seed in (1, 2, 3):
                st, js = post(c, url, base | {"messages": ASK_STAGE, "tools": [set_stage(strict)], "seed": seed,
                                              "tool_choice": choice, "chat_template_kwargs": {"enable_thinking": False}})
                msg = (js.get("choices") or [{}])[0].get("message", {})
                _, args, err = tool_call_of(msg)
                stages.append((args or {}).get("stage") if args else f"<{err}>")
                contents.append((msg.get("content") or "")[:160] or (js.get("error") or {}).get("message"))
            called = [x for x in stages if not x.startswith("<")]
            if strict and choice_label == "forced":
                label, ok = "S1-4 strict + forced call: stage always inside the enum", len(called) == 3 and all(
                    x in STAGES for x in called)
            else:
                label, ok = f"S1-4 {choice_label}, strict {'on' if strict else 'off'} (observation)", True
            check(results, label, ok, {"stages": stages, "contents": contents})


def render(c: httpx.Client, url: str, model: str, messages: list, tools: list) -> str:
    """The prompt text the server itself builds: /tokenize with messages, then /detokenize."""
    r = c.post(f"{url}/tokenize", json={"model": model, "messages": messages, "tools": tools,
                                       "add_generation_prompt": True})
    r.raise_for_status()
    toks = r.json()["tokens"]
    d = c.post(f"{url}/detokenize", json={"model": model, "tokens": toks})
    d.raise_for_status()
    return d.json()["prompt"]


def probe_xlam(c: httpx.Client, url: str, model: str, results: list) -> None:
    marker = "make calls in a single JSON array"
    no_sys = [ASK_COUNT[1]]
    try:
        with_sys_txt, no_sys_txt = render(c, url, model, ASK_COUNT, [RUN_SQL]), render(c, url, model, no_sys, [RUN_SQL])
        check(results, "S1-5 xLAM template: format instruction present without a system message, absent with one",
              marker in no_sys_txt and marker not in with_sys_txt,
              {"without_system_has_marker": marker in no_sys_txt, "with_system_has_marker": marker in with_sys_txt,
               "with_system_has_tools": '"run_sql"' in with_sys_txt,
               "with_system_head": with_sys_txt[:400], "without_system_head": no_sys_txt[:600]})
    except httpx.HTTPError as e:
        check(results, "S1-5 xLAM template render via /tokenize", False, {"error": str(e)})
    base = {"model": model, "tools": [RUN_SQL], "tool_choice": "auto", "max_tokens": 1024,
            "temperature": 0.6, "top_p": 0.9}
    for label, msgs in (("with system prompt", ASK_COUNT), ("without system prompt", no_sys)):
        got = []
        for seed in (1, 2, 3):
            st, js = post(c, url, base | {"messages": msgs, "seed": seed})
            msg = (js.get("choices") or [{}])[0].get("message", {})
            name, args, err = tool_call_of(msg)
            got.append({"status": st, "name": name, "args": args, "err": err, "content": (msg.get("content") or "")[:160]})
        n_ok = sum(1 for g in got if g["name"] == "run_sql" and g["args"])
        check(results, f"S1-6 xLAM {label}: parsable tool call (>= 2 of 3)",
              n_ok >= 2 if label == "with system prompt" else True, {"ok": n_ok, "calls": got})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["qwen", "xlam", "strict"])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    results: list = []
    with httpx.Client(timeout=600.0) as c:
        served = [m["id"] for m in c.get(f"{a.url}/v1/models").json()["data"]]
        print("served:", served, flush=True)
        model = served[0]
        if a.which == "strict":
            probe_strict(c, a.url, {"model": model, "max_tokens": 1024, "temperature": 0.7, "top_p": 0.8,
                                    "top_k": 20, "seed": 1}, results)
        else:
            (probe_qwen if a.which == "qwen" else probe_xlam)(c, a.url, model, results)
    with open(a.out, "w") as f:
        json.dump({"which": a.which, "model": model, "results": results}, f, indent=1)
    n_fail = sum(not r["pass"] for r in results)
    print(f"P11_PROBE_DONE {a.which}: {len(results) - n_fail} pass, {n_fail} fail -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

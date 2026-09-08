#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28"]
# ///
"""
p7probe.py -- do Phase 7's three actuators exist on THIS server? Budget, priority,
tool calls. Reports UNSUPPORTED separately from FAIL: one is a missing feature.

  uv run tools/p7probe.py --probe all --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field

import httpx

QUESTION = ("A tank holds 47 litres. It leaks 3 litres an hour for 5 hours, then is "
            "refilled by 12 litres. How many litres are in it? Reason it through.")


@dataclass
class Probe:
    """One actuator's verdict, with the evidence that produced it."""
    name: str
    verdict: str = "FAIL"          # PASS | FAIL | UNSUPPORTED | ERROR
    detail: str = ""
    rows: list = field(default_factory=list)


async def discover_model(client: httpx.AsyncClient, url: str) -> str | None:
    """Ask the server what it serves. Never hardcode a model id (incident 40)."""
    try:
        r = await client.get(f"{url}/v1/models", timeout=10)
        return (r.json().get("data") or [{}])[0].get("id")
    except Exception:
        return None


async def _chat(client, url, model, body, timeout=180.0):
    r = await client.post(f"{url}/v1/chat/completions",
                          json={"model": model, **body}, timeout=timeout)
    r.raise_for_status()
    return r.json()


async def probe_budget(client, url, model) -> Probe:
    """Does thinking_budget actually cap reasoning tokens, and is reasoning separated?"""
    p = Probe("thinking_budget")
    msgs = [{"role": "user", "content": QUESTION}]
    for budget in (None, 64, 256):
        body = {"messages": msgs, "max_tokens": 1200, "temperature": 0}
        if budget is not None:
            body["chat_template_kwargs"] = {"enable_thinking": True}
            body["thinking_budget"] = budget
        try:
            d = await _chat(client, url, model, body)
        except httpx.HTTPStatusError as e:
            txt = ""
            try:
                txt = e.response.text[:160]
            except Exception:
                pass
            p.verdict, p.detail = "UNSUPPORTED", f"HTTP {e.response.status_code}: {txt}"
            return p
        except Exception as e:
            p.verdict, p.detail = "ERROR", f"{type(e).__name__}: {e}"
            return p
        m = (d.get("choices") or [{}])[0].get("message") or {}
        rc = m.get("reasoning_content")
        # No reasoning_content means the server has no reasoning parser loaded, so a
        # budget cannot be observed even if it were being applied.
        p.rows.append({"budget": budget,
                       "reasoning_chars": len(rc) if isinstance(rc, str) else None,
                       "content_chars": len(m.get("content") or ""),
                       "completion_tokens": (d.get("usage") or {}).get("completion_tokens")})
    if all(r["reasoning_chars"] is None for r in p.rows):
        p.verdict = "UNSUPPORTED"
        p.detail = "no reasoning_content in any response; server has no reasoning parser"
        return p
    base = next((r for r in p.rows if r["budget"] is None), None)
    capped = [r for r in p.rows if r["budget"] is not None and r["reasoning_chars"] is not None]
    if base and base["reasoning_chars"] and capped:
        shrank = all(r["reasoning_chars"] < base["reasoning_chars"] for r in capped)
        ordered = all(a["reasoning_chars"] <= b["reasoning_chars"]
                      for a, b in zip(capped, capped[1:]))
        p.verdict = "PASS" if shrank and ordered else "FAIL"
        p.detail = (f"unbounded {base['reasoning_chars']} chars; "
                    + ", ".join(f"budget {r['budget']} -> {r['reasoning_chars']}" for r in capped))
    else:
        p.detail = "could not establish an unbounded baseline"
    return p


async def probe_priority(client, url, model) -> Probe:
    """Fill the queue with low-priority work, then jump one request ahead of it."""
    p = Probe("priority")
    filler = {"messages": [{"role": "user", "content": QUESTION}],
              "max_tokens": 400, "temperature": 0}

    async def timed(body, tag):
        t0 = time.perf_counter()
        try:
            await _chat(client, url, model, body)
            return {"tag": tag, "s": time.perf_counter() - t0, "error": None}
        except Exception as e:
            return {"tag": tag, "s": time.perf_counter() - t0, "error": f"{type(e).__name__}"}

    # 24 background requests, then one marked urgent after they are all in flight.
    bg = [timed({**filler, "priority": 100}, "low") for _ in range(24)]
    task_bg = [asyncio.create_task(t) for t in bg]
    await asyncio.sleep(1.5)
    hi = await timed({**filler, "priority": 0}, "high")
    lows = await asyncio.gather(*task_bg)
    if hi["error"]:
        p.verdict, p.detail = "UNSUPPORTED", f"priority request failed: {hi['error']}"
        return p
    ok = [r["s"] for r in lows if not r["error"]]
    if not ok:
        p.verdict, p.detail = "ERROR", "every background request failed"
        return p
    med = sorted(ok)[len(ok) // 2]
    p.rows = [{"high_s": round(hi["s"], 2), "low_median_s": round(med, 2), "n_low": len(ok)}]
    # A priority policy should let the urgent request finish well before the median of
    # work queued ahead of it. Without one it simply joins the back of the queue.
    p.verdict = "PASS" if hi["s"] < med * 0.7 else "FAIL"
    p.detail = (f"urgent finished in {hi['s']:.2f}s against a {med:.2f}s median for "
                f"{len(ok)} requests queued ahead of it")
    return p


async def probe_tools(client, url, model) -> Probe:
    """Will the model emit a tool call? Question 2 is much cheaper if it will."""
    p = Probe("tool_calling")
    tools = [{"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for current information.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}}]
    body = {"messages": [{"role": "user",
                          "content": "What is today's headline news in Tokyo?"}],
            "tools": tools, "tool_choice": "auto", "max_tokens": 300, "temperature": 0}
    try:
        d = await _chat(client, url, model, body)
    except httpx.HTTPStatusError as e:
        txt = ""
        try:
            txt = e.response.text[:160]
        except Exception:
            pass
        p.verdict, p.detail = "UNSUPPORTED", f"HTTP {e.response.status_code}: {txt}"
        return p
    except Exception as e:
        p.verdict, p.detail = "ERROR", f"{type(e).__name__}: {e}"
        return p
    m = (d.get("choices") or [{}])[0].get("message") or {}
    calls = m.get("tool_calls") or []
    p.rows = [{"n_tool_calls": len(calls),
               "first": (calls[0].get("function") or {}).get("name") if calls else None,
               "content_head": (m.get("content") or "")[:80]}]
    p.verdict = "PASS" if calls else "FAIL"
    p.detail = (f"emitted {len(calls)} tool call(s)" if calls
                else "answered in prose instead of calling the tool")
    return p


async def run(a) -> int:
    async with httpx.AsyncClient() as client:
        model = a.model or await discover_model(client, a.url)
        if not model:
            print("ABORT: could not resolve the served model from /v1/models")
            return 2
        print(f"server {a.url}  model {model}\n")
        want = ("budget", "priority", "tools") if a.probe == "all" else (a.probe,)
        fns = {"budget": probe_budget, "priority": probe_priority, "tools": probe_tools}
        out = []
        for name in want:
            print(f"  running {name} ...", flush=True)
            out.append(await fns[name](client, a.url, model))
    print(f"\n  {'actuator':<16} {'verdict':<12} detail")
    for p in out:
        print(f"  {p.name:<16} {p.verdict:<12} {p.detail}")
    with open(a.out, "w") as f:
        for p in out:
            f.write(json.dumps(asdict(p)) + "\n")
    print(f"\n  written to {a.out}")
    return 0 if all(p.verdict == "PASS" for p in out) else 1


def selftest() -> int:
    """Offline. Proves each probe reports UNSUPPORTED rather than crashing."""
    fails = []

    class Boom:
        async def post(self, *a, **k):
            req = httpx.Request("POST", "http://x")
            resp = httpx.Response(400, text="unknown field thinking_budget", request=req)
            raise httpx.HTTPStatusError("bad", request=req, response=resp)
        async def get(self, *a, **k):
            raise httpx.ConnectError("refused")

    for fn in (probe_budget, probe_priority, probe_tools):
        p = asyncio.run(fn(Boom(), "http://x", "m"))
        if p.verdict not in ("UNSUPPORTED", "ERROR"):
            fails.append(f"{fn.__name__} returned {p.verdict} against a 400, expected UNSUPPORTED")
        if not p.detail:
            fails.append(f"{fn.__name__} gave no detail")

    p = Probe("x")
    if p.verdict != "FAIL":
        fails.append("a Probe should default to FAIL, so a silent path cannot read as PASS")

    for f in fails:
        print(f"  FAIL {f}")
    print(f"selftest: {'PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default=None, help="discovered from /v1/models if omitted")
    ap.add_argument("--probe", default="all", choices=["all", "budget", "priority", "tools"])
    ap.add_argument("--out", default="results/p7-feasibility.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    return selftest() if a.selftest else asyncio.run(run(a))


if __name__ == "__main__":
    raise SystemExit(main())

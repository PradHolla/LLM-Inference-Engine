#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28", "langgraph==1.2.12", "tokenizers>=0.20"]
# ///
"""Score the 6c planner alone against a labelled set of turns, through a gateway.

  uv run tools/planeval.py --labels results/plan-labels.jsonl --gateway http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app import agent, prompts  # noqa: E402

LABEL = ("Qwen3-8B fp8 weights, fp8 KV, vLLM 0.27.1 V1 runner, 32k ctx, prefix cache on, "
         "A10G, planner via gateway, thinking off")


def load_labels(path: str) -> list[dict]:
    items = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    for item in items:
        missing = {"id", "history", "user", "search", "think", "must_contain"} - set(item)
        if missing:
            raise SystemExit(f"label {item.get('id')!r} lacks {sorted(missing)}")
    return items


async def discover_model(gateway: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{gateway}/v1/models")
            response.raise_for_status()
        return response.json()["data"][0]["id"]
    except Exception:
        return "Qwen/Qwen3-8B"


async def plan_one(item: dict, gateway: str, model: str, today: str) -> dict:
    """The app's own planner prompt and call, both controls on auto, as one fresh turn."""
    base = [{"role": "system", "content": prompts.system_prompt(today)},
            *({"role": m["role"], "content": m["content"]} for m in item["history"])]
    parsed, ms, error = await agent.call_planner(agent.plan_messages(base, item["user"]), model,
                                                 gateway_url=gateway)
    decision = agent.resolve(parsed, "auto", "auto", item["user"])
    return {"id": item["id"], "ms": ms, "error": error, "raw": parsed, **decision}


def entity_hit(item: dict, result: dict) -> bool | None:
    """Every must_contain entity appears in the queries (case-insensitive); None if unlabelled."""
    if not item["search"] or not item["must_contain"]:
        return None
    joined = " ".join(result["queries"]).lower() if result["search"] else ""
    return all(entity.lower() in joined for entity in item["must_contain"])


def p50(values: list[float]) -> float | None:
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2] if ordered else None


def score(items: list[dict], results: list[dict]) -> dict:
    by_id = {result["id"]: result for result in results}
    pairs = [(item, by_id[item["id"]]) for item in items if item["id"] in by_id]
    hits = [(item, entity_hit(item, result)) for item, result in pairs]
    follow = [hit for item, hit in hits if hit is not None and item["history"]]
    labelled = [hit for _, hit in hits if hit is not None]
    ms = [result["ms"] for _, result in pairs]
    return {"n": len(pairs),
            "search_agree": sum(item["search"] == result["search"] for item, result in pairs),
            "think_agree": sum(item["think"] == result["think"] for item, result in pairs),
            "followup_hits": sum(follow), "followup_n": len(follow),
            "entity_hits": sum(labelled), "entity_n": len(labelled),
            "fallbacks": sum(result["fallback"] for _, result in pairs),
            "ms_p50": p50(ms), "ms_max": max(ms) if ms else None}


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({num / den * 100:.1f}%)" if den else f"{num}/0 (-)"


def yn(value) -> str:
    return "-" if value is None else ("yes" if value else "no")


def report(items: list[dict], results: list[dict], label: str) -> str:
    s = score(items, results)
    by_id = {result["id"]: result for result in results}
    lines = [f"config: {label}",
             f"items {s['n']}; fallbacks {s['fallbacks']} (scored as the pipeline acts: "
             "search on the user text, think on)",
             f"search decision agreement      {pct(s['search_agree'], s['n'])}",
             f"think decision agreement       {pct(s['think_agree'], s['n'])}",
             f"follow-up query entity hits    {pct(s['followup_hits'], s['followup_n'])}  "
             "(history non-empty, labelled search, must_contain non-empty)",
             f"all labelled entity hits       {pct(s['entity_hits'], s['entity_n'])}",
             "planner latency ms             p50 " +
             ("-" if s["ms_p50"] is None else f"{s['ms_p50']:.0f}") + ", max " +
             ("-" if s["ms_max"] is None else f"{s['ms_max']:.0f}"),
             "",
             f"| id | turns | label search | got search | label think | got think | fallback "
             f"| ms | entities | queries |  [{label}]",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for item in items:
        result = by_id.get(item["id"])
        if result is None:
            continue
        lines.append(f"| {item['id']} | {len(item['history'])} | {yn(item['search'])} | "
                     f"{yn(result['search'])} | {yn(item['think'])} | {yn(result['think'])} | "
                     f"{yn(result['fallback'])} | {result['ms']:.0f} | "
                     f"{yn(entity_hit(item, result))} | {'; '.join(result['queries'])} |")
    return "\n".join(lines)


async def evaluate(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    items = load_labels(args.labels)
    model = args.model or await discover_model(args.gateway)
    results = []
    for item in items:
        result = await plan_one(item, args.gateway, model, args.today)
        results.append(result)
        if args.out:
            with open(args.out, "a", encoding="utf-8") as out:
                out.write(json.dumps({**result, "label": item, "model": model}) + "\n")
        print(f"  {item['id']}: search={yn(result['search'])} think={yn(result['think'])} "
              f"fallback={yn(result['fallback'])} {result['ms']:.0f} ms", flush=True)
    print(report(items, results, args.label))
    return items, results


async def run(args: argparse.Namespace) -> int:
    await evaluate(args)
    return 0


def selftest() -> int:
    """Three fixture items against a fake gateway: one hit, one think miss, one fallback."""
    fails: list[str] = []
    seen: list[dict] = []
    replies = {"and its risks?": {"search": True, "queries": ["ASML EUV lithography risks"],
                                  "think": False},
               "hello": {"search": False, "queries": [], "think": False}}

    def check(name: str, condition: bool) -> None:
        if not condition:
            fails.append(f"  FAIL {name}")

    async def gateway(reader, writer) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        length = next(int(line.split(b":", 1)[1]) for line in head.split(b"\r\n")
                      if line.lower().startswith(b"content-length:"))
        request = json.loads(await reader.readexactly(length))
        seen.append(request)
        user = request["messages"][-2]["content"]
        content = json.dumps(replies[user]) if user in replies else "no plan here"
        payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                     + str(len(payload)).encode() + b"\r\n\r\n" + payload)
        await writer.drain()
        writer.close()

    items = [
        {"id": "f1", "history": [{"role": "user", "content": "Tell me about ASML's EUV machines"},
                                 {"role": "assistant", "content": "ASML makes EUV tools."}],
         "user": "and its risks?", "search": True, "think": False, "must_contain": ["ASML", "euv"]},
        {"id": "g1", "history": [], "user": "hello", "search": False, "think": True,
         "must_contain": []},
        {"id": "m1", "history": [], "user": "what is 17 * 23", "search": False, "think": True,
         "must_contain": []},
    ]

    async def main() -> tuple[str, list[dict]]:
        server = await asyncio.start_server(gateway, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            labels, out = Path(tmp) / "labels.jsonl", Path(tmp) / "out.jsonl"
            labels.write_text("".join(json.dumps(item) + "\n" for item in items))
            args = argparse.Namespace(labels=str(labels), gateway=f"http://127.0.0.1:{port}",
                                      model="m", today="2026-09-23", out=str(out), label="fixture")
            from contextlib import redirect_stdout
            import io
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                _, ran = await evaluate(args)
            check("one flushed record per item", len(out.read_text().splitlines()) == 3)
        server.close()
        await server.wait_closed()
        return buffer.getvalue(), ran

    text, results = asyncio.run(main())
    check("the invalid reply fell back to search on the user text, think on",
          results[2]["fallback"] and results[2]["queries"] == ["what is 17 * 23"] and
          results[2]["think"] and not results[0]["fallback"])
    s = score(items, results)
    check("search agreement counts the fallback's forced search as a miss", s["search_agree"] == 2)
    check("think agreement", s["think_agree"] == 2)
    check("follow-up entity hit is case-insensitive", (s["followup_hits"], s["followup_n"]) == (1, 1))
    check("fallback counted", s["fallbacks"] == 1)
    timed = [{**result, "ms": ms} for result, ms in zip(results, (10.0, 30.0, 20.0))]
    check("latency p50 and max", (score(items, timed)["ms_p50"], score(items, timed)["ms_max"])
          == (20.0, 30.0))
    check("no search means no entity hit",
          entity_hit(items[0], {**results[0], "search": False}) is False)
    check("report prints every headline",
          all(key in text for key in ("search decision agreement      2/3",
                                      "think decision agreement       2/3",
                                      "follow-up query entity hits    1/1", "fallbacks 1")))
    check("report prints the per-item table", "| f1 | 2 | yes | yes |" in text and
          "| m1 | 0 | no | yes | yes | yes | yes |" in text)
    check("planner calls are tagged plan, thinking-off and structured",
          len(seen) == 3 and all(r.get("gw_purpose") == "plan" and
                                 r.get("gw_thinking_budget") == 0 and
                                 r.get("response_format", {}).get("type") == "json_schema" and
                                 r["messages"][-1]["content"] == prompts.PLANNER_INSTRUCTION
                                 for r in seen))
    check("history precedes the user message",
          seen[0]["messages"][1]["content"] == "Tell me about ASML's EUV machines")
    print("\n".join(fails) if fails else "planeval selftest: PASS")
    return 1 if fails else 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="results/plan-labels.jsonl")
    parser.add_argument("--gateway", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default=None, help="discovered from /v1/models if omitted")
    parser.add_argument("--today", default=datetime.date.today().isoformat())
    parser.add_argument("--out", default=None, help="per-item JSONL, flushed as each item finishes")
    parser.add_argument("--label", default=LABEL, help="configuration printed with the results")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())

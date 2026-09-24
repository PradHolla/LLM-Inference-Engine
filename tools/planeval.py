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
import math
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402

import mkitems  # noqa: E402
import plansets  # noqa: E402
import qualeval  # noqa: E402
from app import agent, prompts  # noqa: E402

LABEL = ("Qwen3-8B fp8 weights, fp8 KV, vLLM 0.27.1 V1 runner, 32k ctx, prefix cache on, "
         "A10G, planner via gateway, thinking off")


def load_labels(path: str) -> list[dict]:
    """search/think may be null (not scored); source, category, hit_fraction are optional."""
    items = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    for item in items:
        missing = {"id", "history", "user", "search", "think", "must_contain"} - set(item)
        if missing:
            raise SystemExit(f"label {item.get('id')!r} lacks {sorted(missing)}")
        for key in ("search", "think"):
            if item[key] not in (True, False, None):
                raise SystemExit(f"label {item['id']!r} {key} must be true, false or null")
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
    """At least hit_fraction (default all) of must_contain in the queries; None if unlabelled."""
    if item["search"] is not True or not item["must_contain"]:
        return None
    joined = " ".join(result["queries"]).lower() if result["search"] else ""
    found = sum(entity.lower() in joined for entity in item["must_contain"])
    return found >= item.get("hit_fraction", 1.0) * len(item["must_contain"])


def p50(values: list[float]) -> float | None:
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2] if ordered else None


def p95(values: list[float]) -> float | None:
    """Nearest-rank p95; None below 20 samples, where it would be the max in disguise."""
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1] if len(ordered) >= 20 else None


def score(items: list[dict], results: list[dict]) -> dict:
    by_id = {result["id"]: result for result in results}
    pairs = [(item, by_id[item["id"]]) for item in items if item["id"] in by_id]
    hits = [(item, entity_hit(item, result)) for item, result in pairs]
    follow = [hit for item, hit in hits if hit is not None and item["history"]]
    labelled = [hit for _, hit in hits if hit is not None]
    searched = [(item, result) for item, result in pairs if item["search"] is not None]
    thought = [(item, result) for item, result in pairs if item["think"] is not None]
    ms = [result["ms"] for _, result in pairs]
    return {"n": len(pairs),
            "search_rate": sum(result["search"] for _, result in pairs),
            "search_agree": sum(item["search"] == result["search"] for item, result in searched),
            "search_n": len(searched),
            "think_agree": sum(item["think"] == result["think"] for item, result in thought),
            "think_n": len(thought),
            "followup_hits": sum(follow), "followup_n": len(follow),
            "entity_hits": sum(labelled), "entity_n": len(labelled),
            "fallbacks": sum(result["fallback"] for _, result in pairs),
            "ms_p50": p50(ms), "ms_p95": p95(ms), "ms_max": max(ms) if ms else None}


def breakdown(items: list[dict], results: list[dict], key: str) -> list[tuple[str, dict]]:
    """score() per distinct value of an optional label field; missing values group as '-'."""
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(item.get(key) or "-", []).append(item)
    return [(name, score(members, results)) for name, members in sorted(groups.items())]


def ms_text(value: float | None, n: int) -> str:
    return ("n<20" if n < 20 else "-") if value is None else f"{value:.0f}"


def breakdown_table(items: list[dict], results: list[dict], key: str, label: str) -> list[str]:
    lines = [f"| {key} | n | search rate | search agree | think agree | query hit | fallbacks "
             f"| ms p50 | ms p95 |  [{label}]", "|---|---|---|---|---|---|---|---|---|"]
    for name, s in breakdown(items, results, key):
        lines.append(f"| {name} | {s['n']} | {pct(s['search_rate'], s['n'])} | "
                     f"{pct(s['search_agree'], s['search_n'])} | "
                     f"{pct(s['think_agree'], s['think_n'])} | "
                     f"{pct(s['entity_hits'], s['entity_n'])} | {s['fallbacks']} | "
                     f"{ms_text(s['ms_p50'], s['n'])} | {ms_text(s['ms_p95'], s['n'])} |")
    return lines


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
             f"search rate (planner chose)    {pct(s['search_rate'], s['n'])}",
             f"search decision agreement      {pct(s['search_agree'], s['search_n'])}  "
             "(null search labels skipped)",
             f"think decision agreement       {pct(s['think_agree'], s['think_n'])}  "
             "(null think labels skipped)",
             f"follow-up query entity hits    {pct(s['followup_hits'], s['followup_n'])}  "
             "(history non-empty, labelled search, must_contain non-empty)",
             f"all labelled entity hits       {pct(s['entity_hits'], s['entity_n'])}",
             "planner latency ms             p50 " +
             ("-" if s["ms_p50"] is None else f"{s['ms_p50']:.0f}") + ", p95 " +
             ms_text(s["ms_p95"], s["n"]) + ", max " +
             ("-" if s["ms_max"] is None else f"{s['ms_max']:.0f}"),
             ""]
    for key in ("source", "category"):
        if any(key in item for item in items):
            lines += [*breakdown_table(items, results, key, label), ""]
    lines += [f"| id | turns | label search | got search | label think | got think | fallback "
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


THINK_ARMS = {False: {"gw_thinking_budget": 0, "max_tokens": 1024, **agent.sampling(False)},
              True: {"max_tokens": 4096, **agent.sampling(True)}}


def think_items(gsm8k_rows: list[dict], fresh_rows: list[dict], n: int, seed: int) -> list[dict]:
    """n GSM8K items (the phase 4 prompt and gold) and n FreshQA never-changing TEST items."""
    items = []
    for item in mkitems.make_gsm8k(gsm8k_rows, n, random.Random(seed)):
        question = item["prompt"].removesuffix("\n\n" + mkitems.ANSWER_INT)
        items.append({"id": f"think-{item['id']}", "source": "gsm8k", "category": "gsm8k",
                      "user": question,
                      "prompt": item["prompt"], "answer": item["answer"]})
    never = [row for row in fresh_rows if row["split"] == "TEST"
             and plansets.freshqa_category(row) == "never-changing"]
    for row in random.Random(seed).sample(never, min(n, len(never))):
        items.append({"id": f"think-freshqa-{row['id']}", "source": "freshqa",
                      "category": "freshqa-never", "user": row["question"].strip(),
                      "prompt": row["question"].strip(), "answer": plansets.freshqa_answers(row)})
    return items


def think_correct(item: dict, content: str, finish: str | None) -> bool:
    if item["source"] == "gsm8k":
        return qualeval.grade({"text": content, "slice": "gsm8k", "answer": item["answer"],
                               "finish_reason": finish})["correct"]
    return any(answer.lower() in content.lower() for answer in item["answer"])


def think_label(off_ok: bool, on_ok: bool) -> bool | None:
    """true: only thinking got it right; false: plain already right; null: both wrong."""
    return False if off_ok else (True if on_ok else None)


async def ask_arm(client: httpx.AsyncClient, gateway: str, model: str, item: dict,
                  think: bool) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": item["prompt"]}],
            "stream": False, **THINK_ARMS[think]}
    started = time.perf_counter()
    try:
        response = await client.post(f"{gateway}/v1/chat/completions", json=body)
        response.raise_for_status()
        choice = response.json()["choices"][0]
        usage = response.json().get("usage") or {}
        content = choice["message"].get("content") or ""
        finish = choice.get("finish_reason")
        return {"correct": think_correct(item, content, finish), "finish_reason": finish,
                "completion_tokens": usage.get("completion_tokens"), "content": content,
                "ms": (time.perf_counter() - started) * 1000, "error": None}
    except Exception as exc:
        return {"correct": None, "finish_reason": None, "completion_tokens": None, "content": "",
                "ms": (time.perf_counter() - started) * 1000,
                "error": f"{type(exc).__name__}: {exc}"}


async def label_think(items: list[dict], gateway: str, model: str, out: str,
                      concurrency: int) -> list[dict]:
    """Both arms per item, one line flushed as each item completes; undecidable errors are null."""
    gate = asyncio.Semaphore(concurrency)
    done: list[dict] = []

    async def one(item: dict, client: httpx.AsyncClient) -> None:
        async with gate:
            off = await ask_arm(client, gateway, model, item, False)
            on = await ask_arm(client, gateway, model, item, True)
        errored = off["error"] or (not off["correct"] and on["error"])
        label = None if errored else think_label(off["correct"], bool(on["correct"]))
        record = {"id": item["id"], "source": item["source"], "category": item["category"],
                  "history": [], "user": item["user"], "search": None, "think": label,
                  "must_contain": [], "answer": item["answer"], "off": off, "on": on,
                  "model": model}
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        done.append(record)
        print(f"  {item['id']}: off={yn(off['correct'])} on={yn(on['correct'])} "
              f"think={yn(label)}{' ERROR' if errored else ''}", flush=True)

    async with httpx.AsyncClient(timeout=900.0) as client:
        await asyncio.gather(*(one(item, client) for item in items))
    counts = {str(v): sum(r["think"] is v for r in done) for v in (True, False, None)}
    print(f"{out}: {len(done)} items; think labels {counts}; "
          f"errors {sum(bool(r['off']['error'] or r['on']['error']) for r in done)}")
    return done


async def run_label_think(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in open(args.gsm8k, encoding="utf-8") if line.strip()]
    items = think_items(rows, plansets.load_freshqa(Path(args.cache)), args.n, args.seed)
    model = args.model or await discover_model(args.gateway)
    await label_think(items, args.gateway, model, args.out, args.concurrency)
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
    selftest_datasets(check)
    selftest_label_think(check)
    print("\n".join(fails) if fails else "planeval selftest: PASS")
    return 1 if fails else 0


def plan(search: bool, think: bool, queries: list[str], ms: float = 100.0) -> dict:
    return {"id": "", "ms": ms, "error": None, "raw": None,
            **agent.resolve({"search": search, "queries": queries, "think": think},
                            "auto", "auto", "q")}


def selftest_datasets(check) -> None:
    """Null labels, per-source and per-category rows, half-hit rule, p95 gating."""
    items = [
        {"id": "a", "source": "mtrag", "category": "answerable", "history": [], "user": "q",
         "search": True, "think": None, "must_contain": ["kevin", "durant", "thunder", "2010"],
         "hit_fraction": 0.5},
        {"id": "b", "source": "mtrag", "category": "conversational", "history": [], "user": "thanks",
         "search": False, "think": None, "must_contain": []},
        {"id": "c", "source": "freshqa", "category": "never-changing", "history": [], "user": "q",
         "search": None, "think": None, "must_contain": []},
        {"id": "d", "source": "freshqa", "category": "never-changing", "history": [], "user": "q",
         "search": None, "think": True, "must_contain": []},
        {"id": "e", "source": "freshqa", "category": "fast-changing", "history": [], "user": "q",
         "search": True, "think": None, "must_contain": []},
    ]
    results = [{**plan(True, False, ["Kevin Durant 2010 contract"]), "id": "a"},
               {**plan(True, False, ["thanks"]), "id": "b"},
               {**plan(True, False, ["x"]), "id": "c"},
               {**plan(False, True, []), "id": "d"},
               {**plan(True, False, ["y"]), "id": "e"}]
    s = score(items, results)
    check("null search labels are skipped, counted separately",
          (s["search_agree"], s["search_n"], s["search_rate"], s["n"]) == (2, 3, 4, 5))
    check("null think labels are skipped", (s["think_agree"], s["think_n"]) == (1, 1))
    check("half rule: 3 of 4 is a hit",
          entity_hit(items[0], results[0]) is True and s["entity_n"] == 1)
    check("half rule: 1 of 4 is a miss",
          entity_hit(items[0], {**results[0], "queries": ["kevin"]}) is False)
    check("default rule still needs every entity",
          entity_hit({**items[0], "hit_fraction": 1.0}, results[0]) is False)
    rows = dict(breakdown(items, results, "category"))
    check("per-category search rate on unlabelled items",
          (rows["never-changing"]["search_rate"], rows["never-changing"]["search_n"],
           rows["never-changing"]["n"]) == (1, 0, 2))
    check("per-category agreement",
          (rows["conversational"]["search_agree"], rows["conversational"]["search_n"]) == (0, 1))
    src = dict(breakdown(items, results, "source"))
    check("per-source split", (src["mtrag"]["n"], src["freshqa"]["n"]) == (2, 3))
    check("p95 is None below 20 samples, nearest-rank at 20",
          p95([1.0] * 19) is None and p95([float(i) for i in range(1, 21)]) == 19.0)
    text = report(items, results, "fixture")
    check("report prints source and category tables with n<20",
          "| source | n |" in text and "| category | n |" in text and
          "| never-changing | 2 | 1/2 (50.0%) | 0/0 (-) | 1/1 (100.0%) | 0/0 (-) | 0 | 100 | n<20 |"
          in text)
    check("old label files print no breakdown tables",
          "| source |" not in report([{k: v for k, v in item.items()
                                       if k not in ("source", "category")} for item in items],
                                     results, "fixture"))
    with tempfile.TemporaryDirectory() as tmp:
        good, bad = Path(tmp) / "good.jsonl", Path(tmp) / "bad.jsonl"
        good.write_text("".join(json.dumps(item) + "\n" for item in items))
        bad.write_text(json.dumps({**items[0], "search": "yes"}) + "\n")
        check("null labels load", len(load_labels(str(good))) == 5)
        try:
            load_labels(str(bad))
            check("a non-boolean label is rejected", False)
        except SystemExit:
            pass


def selftest_label_think(check) -> None:
    """Three label outcomes plus an undecidable error, against a fake gateway."""
    gsm = [{"question": "Janet has 9 eggs and sells them at $2. How much?",
            "answer": "9 * 2 = 18\n#### 18"},
           {"question": "A robe takes 2 bolts plus half that. Total?", "answer": "2 + 1\n#### 3"}]
    fresh = [dict(zip(["id", "split", "question", "false_premise", "fact_type", "answer_0",
                       "answer_1"], row)) for row in
             (["245", "TEST", "What was the first animal to orbit the Earth?", "FALSE",
               "never-changing", "Laika", ""],
              ["250", "TEST", "What state is known as the Aloha State?", "FALSE", "never-changing",
               "Hawaii", "Hawai'i"],
              ["251", "TEST", "Who is the CEO of X?", "FALSE", "fast-changing", "Musk", ""],
              ["252", "DEV", "Who wrote Hamlet?", "FALSE", "never-changing", "Shakespeare", ""])]
    replies = {("Janet", False): "ANSWER: 18", ("Janet", True): "ANSWER: 18",
               ("robe", False): "ANSWER: 4", ("robe", True): "It is \\boxed{3}",
               ("orbit", False): "It was a dog.", ("orbit", True): "Probably a monkey.",
               ("Aloha", True): "hawaii"}
    seen: list[dict] = []

    async def gateway(reader, writer) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        length = next(int(line.split(b":", 1)[1]) for line in head.split(b"\r\n")
                      if line.lower().startswith(b"content-length:"))
        request = json.loads(await reader.readexactly(length))
        seen.append(request)
        think = "gw_thinking_budget" not in request
        prompt = request["messages"][0]["content"]
        key = next((k for k in replies if k[0] in prompt and k[1] == think), None)
        if key is None:
            writer.write(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n")
        else:
            payload = json.dumps({"choices": [{"message": {"content": replies[key],
                                                           "reasoning": "hmm"},
                                               "finish_reason": "stop"}],
                                  "usage": {"completion_tokens": 7}}).encode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                         b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
        await writer.drain()
        writer.close()

    async def main() -> tuple[list[dict], list[str]]:
        server = await asyncio.start_server(gateway, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "think.jsonl"
            from contextlib import redirect_stdout
            import io
            with redirect_stdout(io.StringIO()):
                done = await label_think(think_items(gsm, fresh, 2, 0),
                                         f"http://127.0.0.1:{port}", "m", str(out), 2)
            lines = out.read_text().splitlines()
        server.close()
        await server.wait_closed()
        return done, lines

    done, lines = asyncio.run(main())
    by_q = {r["user"].split()[0]: r for r in done if r["source"] == "gsm8k"}
    fresh_by_id = {r["id"]: r for r in done if r["source"] == "freshqa"}
    check("label-think writes one line per item, FreshQA TEST never-changing only",
          len(lines) == 4 and set(by_q) == {"Janet", "A"} and
          set(fresh_by_id) == {"think-freshqa-245", "think-freshqa-250"} and
          {r["category"] for r in done} == {"gsm8k", "freshqa-never"})
    check("off right -> think false", by_q["Janet"]["think"] is False)
    check("off wrong, on right (boxed, via the qualeval grader) -> think true",
          by_q["A"]["think"] is True and by_q["A"]["on"]["correct"] is True)
    check("both wrong -> think null", fresh_by_id["think-freshqa-245"]["think"] is None
          and fresh_by_id["think-freshqa-245"]["off"]["error"] is None)
    check("off arm error -> null with the error kept",
          fresh_by_id["think-freshqa-250"]["think"] is None
          and fresh_by_id["think-freshqa-250"]["off"]["error"])
    check("label records carry source, category, search null, usage tokens",
          all(r["search"] is None and r["must_contain"] == [] and r["history"] == []
              for r in done) and by_q["Janet"]["off"]["completion_tokens"] == 7)
    check("the gsm8k user text is the bare question, the prompt carries ANSWER:",
          by_q["Janet"]["user"] == gsm[0]["question"] and
          any(mkitems.ANSWER_INT in r["messages"][0]["content"] for r in seen))
    off = [r for r in seen if "gw_thinking_budget" in r]
    on = [r for r in seen if "gw_thinking_budget" not in r]
    check("off arm: budget 0, plain sampling, 1024 tokens, no stream, no search fields",
          len(off) == 4 and all(r["gw_thinking_budget"] == 0 and r["max_tokens"] == 1024 and
                                (r["temperature"], r["top_p"], r["top_k"], r["min_p"]) ==
                                (0.7, 0.8, 20, 0.0) and r["stream"] is False and
                                not any(k.startswith("gw_") and k != "gw_thinking_budget"
                                        for k in r) for r in off))
    check("on arm: field omitted, thinking sampling, 4096 tokens, run for every item",
          len(on) == 4 and all(r["max_tokens"] == 4096 and
                               (r["temperature"], r["top_p"], r["top_k"], r["min_p"]) ==
                               (0.6, 0.95, 20, 0.0) and r["stream"] is False and
                               not any(k.startswith("gw_") for k in r) for r in on))
    check("freshqa containment is case-insensitive over every gold answer",
          think_correct({"source": "freshqa", "answer": ["Hawaii", "Hawai'i"]}, "HAWAI'I", "stop"))


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    if sys.argv[1:2] == ["label-think"]:
        lt = argparse.ArgumentParser(prog="planeval.py label-think",
                                     description="measured think labels: off vs on, per item")
        lt.add_argument("--gateway", default="http://127.0.0.1:8080")
        lt.add_argument("--out", default=str(plansets.CACHE / "think.jsonl"))
        lt.add_argument("--model", default=None, help="discovered from /v1/models if omitted")
        lt.add_argument("--n", type=int, default=40, help="items per source")
        lt.add_argument("--seed", type=int, default=0)
        lt.add_argument("--gsm8k", default="data/gsm8k-test.jsonl")
        lt.add_argument("--cache", default=str(plansets.CACHE), help="FreshQA CSV cache")
        lt.add_argument("--concurrency", type=int, default=8)
        return asyncio.run(run_label_think(lt.parse_args(sys.argv[2:])))
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

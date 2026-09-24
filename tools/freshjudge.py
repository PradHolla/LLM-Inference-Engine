#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28"]
# ///
"""Grade recorded FreshQA replies against the gold answers with a Qwen judge, after the run.

  uv run tools/freshjudge.py --answers results/p6c-fresh-answers.jsonl --out results/p6c-fresh-judged.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys

import httpx

GOLD_DATE = "2026-04-21"   # the FreshQA sheet the gold answers come from
THINK_CHARS = 4000         # judge sees at most this much of the model's thinking
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["verdict", "reason"],
          "properties": {"verdict": {"type": "string", "enum": ["correct", "incorrect"]},
                         "reason": {"type": "string"}}}
LABEL = ("judge: Qwen3-8B fp8, thinking off, temperature 0, JSON schema; answers from the 6c app, "
         "vLLM 0.27.1 V1 runner, 32k ctx, A10G, one user")


def judge_prompt(item: dict, rec: dict) -> str:
    """The judge sees the question, the gold, the reply, and the thinking as context only."""
    rules = ("It is correct if its main answer agrees with any reference answer; paraphrases, "
             "spelling variants and equivalent numbers count. It is incorrect if it gives a different "
             "main answer, contradicts the reference, or hedges without committing to one.")
    if item["category"] == "false-premise":
        rules = ("The question contains a false premise. The reply is correct only if it points out "
                 "the false premise; answering as if the premise were true is incorrect.")
    return (f"Question: {item['user']}\n"
            f"Reference answer(s), recorded {GOLD_DATE}: {' | '.join(item['answer'])}\n\n"
            f"The model's final reply:\n<<<\n{rec['content'].strip()}\n>>>\n\n"
            f"The model's private reasoning, for context only; do not grade it:\n<<<\n"
            f"{(rec.get('reasoning') or '').strip()[:THINK_CHARS]}\n>>>\n\n"
            f"Is the model's final reply correct? {rules} "
            'Reply with JSON: {"verdict": "correct" or "incorrect", "reason": one short sentence}.')


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.]", " ", text.lower().replace("'", ""))).strip()


def contains(item: dict, reply: str) -> bool | None:
    """Second grader: any gold string inside the reply. None for false premise (golds are prose)."""
    if item["category"] == "false-premise":
        return None
    body = normalize(reply)
    return any(normalize(g) and normalize(g) in body for g in item["answer"])


async def judge_one(client: httpx.AsyncClient, gateway: str, model: str, item: dict,
                    rec: dict) -> dict:
    body = {"model": model, "stream": False, "max_tokens": 160, "temperature": 0.0,
            "messages": [{"role": "system", "content": "You grade replies to factual questions "
                          "against reference answers. You are strict and brief."},
                         {"role": "user", "content": judge_prompt(item, rec)}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "verdict", "schema": SCHEMA, "strict": True}},
            "gw_thinking_budget": 0, "gw_purpose": "judge"}
    try:
        r = await client.post(f"{gateway}/v1/chat/completions", json=body)
        r.raise_for_status()
        out = json.loads(r.json()["choices"][0]["message"]["content"])
        return {"verdict": out["verdict"] == "correct", "reason": out.get("reason", ""), "error": None}
    except Exception as exc:
        return {"verdict": None, "reason": "", "error": f"{type(exc).__name__}: {exc}"}


def med(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, nd=0, scale=1.0) -> str:
    return "-" if x is None else f"{x / scale:,.{nd}f}"


def pct(hits: int, n: int) -> str:
    return f"{hits}/{n} ({100 * hits / n:.0f}%)" if n else "-"


def report(rows: list[dict]) -> str:
    arms = sorted({r["arm"] for r in rows})
    cats = sorted({r["category"] for r in rows})
    out = [f"config: {LABEL}", f"gold answers recorded {GOLD_DATE}; fast-changing golds may be stale", ""]
    out.append("| arm | n | judge correct | containment correct | graders disagree | judge errors "
               "| first token p50 s | first answer token p50 s | e2e p50 s | tokens p50 | think chars p50 "
               "| chose to think |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        g = [r for r in rows if r["arm"] == arm and r["status"] == "ok"]
        judged = [r for r in g if r["verdict"] is not None]
        gradable = [r for r in g if r["contains"] is not None]       # independent of the judge
        both = [r for r in judged if r["contains"] is not None]
        chose = [r for r in g if r.get("plan")]
        out.append(f"| {arm} | {len(g)} | {pct(sum(r['verdict'] for r in judged), len(judged))} "
                   f"| {pct(sum(r['contains'] for r in gradable), len(gradable))} "
                   f"| {sum(r['verdict'] != r['contains'] for r in both)} | {len(g) - len(judged)} "
                   f"| {fmt(med(r['ttft_ms'] for r in g), 2, 1e3)} | {fmt(med(r['ttfc_ms'] for r in g), 1, 1e3)} "
                   f"| {fmt(med(r['e2e_ms'] for r in g), 1, 1e3)} | {fmt(med(r['tokens'] for r in g))} "
                   f"| {fmt(med(len(r.get('reasoning') or '') for r in g))} "
                   f"| {pct(sum(bool(r['plan'].get('think')) for r in chose), len(chose)) if chose else '-'} |")
    out += ["", "judge accuracy by arm and category", "",
            "| arm | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for arm in arms:
        cells = []
        for c in cats:
            g = [r for r in rows if r["arm"] == arm and r["category"] == c and r["verdict"] is not None]
            cells.append(pct(sum(r["verdict"] for r in g), len(g)))
        out.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(out)


async def run(a: argparse.Namespace) -> int:
    items = {json.loads(l)["id"]: json.loads(l) for l in open(a.items) if l.strip()}
    recs = [json.loads(l) for l in open(a.answers) if l.strip()]
    recs = [r for r in recs if r.get("phase") == "fresh" and r["qid"] in items]
    async with httpx.AsyncClient(timeout=120) as client:
        model = a.model or (await client.get(f"{a.gateway}/v1/models")).json()["data"][0]["id"]
        sem, rows = asyncio.Semaphore(a.concurrency), []
        out = open(a.out, "w")

        async def one(rec: dict) -> None:
            item = items[rec["qid"]]
            async with sem:
                verdict = (await judge_one(client, a.gateway, model, item, rec)
                           if rec["status"] == "ok" else
                           {"verdict": None, "reason": "", "error": "reply did not complete"})
            row = {**rec, **verdict, "contains": contains(item, rec["content"]),
                   "gold": item["answer"], "question": item["user"]}
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()

        await asyncio.gather(*(one(r) for r in recs))
        out.close()
    print(report(rows))
    disagree = [r for r in rows if r["verdict"] is not None and r["contains"] is not None
                and r["verdict"] != r["contains"]]
    with open(a.disagreements, "w") as f:
        for r in disagree:
            f.write(json.dumps({k: r[k] for k in ("qid", "arm", "category", "question", "gold",
                                                  "verdict", "contains", "reason")}
                               | {"reply": r["content"][:1500]}) + "\n")
    print(f"\n{len(disagree)} judge/containment disagreements written to {a.disagreements}")
    return 0


def selftest() -> int:
    fails = []

    def chk(name, got, want):
        if got != want:
            fails.append(f"  FAIL {name}: got {got!r}, want {want!r}")

    item = {"id": "f1", "category": "never-changing", "user": "Aloha State?", "answer": ["Hawaii", "Hawai'i"]}
    fp = {"id": "f2", "category": "false-premise", "user": "Why is the moon cheese?",
          "answer": ["The moon is not made of cheese."]}
    chk("containment finds a gold", contains(item, "It is **Hawaii**."), True)
    chk("containment handles the apostrophe variant", contains(item, "That is Hawai'i."), True)
    chk("containment misses a wrong answer", contains(item, "Alaska."), False)
    chk("containment abstains on false premise", contains(fp, "anything"), None)
    chk("false-premise prompt swaps the rule", "false premise" in judge_prompt(fp, {"content": "x"}), True)
    chk("prompt carries thinking as context", "private reasoning" in judge_prompt(item, {"content": "x",
        "reasoning": "hmm"}), True)
    chk("thinking is capped", judge_prompt(item, {"content": "x", "reasoning": "y" * 9000}).count("y")
        <= THINK_CHARS + 5, True)

    def handler(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        assert sent["gw_thinking_budget"] == 0 and sent["response_format"]["type"] == "json_schema"
        reply = sent["messages"][1]["content"]
        verdict = "correct" if "Hawaii" in reply.split("final reply")[1] else "incorrect"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
            {"verdict": verdict, "reason": "r"})}}]})

    async def fake() -> tuple[dict, dict, dict]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            good = await judge_one(c, "http://g", "m", item, {"content": "Hawaii."})
            bad = await judge_one(c, "http://g", "m", item, {"content": "Alaska."})
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "oops"}}]}))) as c:
            broken = await judge_one(c, "http://g", "m", item, {"content": "Hawaii."})
        return good, bad, broken

    good, bad, broken = asyncio.run(fake())
    chk("judge correct", good["verdict"], True)
    chk("judge incorrect", bad["verdict"], False)
    chk("unparseable judge reply is an error, not a verdict", (broken["verdict"], bool(broken["error"])),
        (None, True))
    rows = [{"arm": "a", "category": "never-changing", "status": "ok", "verdict": True, "contains": True,
             "ttft_ms": 1000, "ttfc_ms": 2000, "e2e_ms": 5000, "tokens": 100, "reasoning": "", "plan": None},
            {"arm": "a", "category": "never-changing", "status": "ok", "verdict": False, "contains": True,
             "ttft_ms": 3000, "ttfc_ms": 4000, "e2e_ms": 7000, "tokens": 300, "reasoning": "xx",
             "plan": {"think": True}}]
    text = report(rows)
    chk("report counts judge accuracy", "| 1/2 (50%) | 2/2 (100%) | 1 |" in text, True)
    rows[1]["verdict"] = None
    chk("containment still counts when the judge failed", "| 1/1 (100%) | 2/2 (100%) | 0 | 1 |"
        in report(rows), True)
    print("\n".join(fails) if fails else "freshjudge selftest: PASS")
    return 1 if fails else 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", default="results/p6c-fresh-answers.jsonl")
    ap.add_argument("--items", default="data/plansets/freshqa.jsonl")
    ap.add_argument("--gateway", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="results/p6c-fresh-judged.jsonl")
    ap.add_argument("--disagreements", default="results/p6c-fresh-disagreements.jsonl")
    ap.add_argument("--concurrency", type=int, default=4)
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())

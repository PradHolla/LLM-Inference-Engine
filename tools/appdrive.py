#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28", "tokenizers>=0.20"]
# ///
"""Drive the chat app's own HTTP API as the browser does; join each turn to the gateway trace.

  uv run tools/appdrive.py run --mode smoke|convo|levels --out results/p6b-app.jsonl
  uv run tools/appdrive.py report --app-out results/p6b-app.jsonl --gw-trace results/p6b-gw.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import random
import sys
import time

import httpx

LEVELS = ["off", "brief", "full"]

# P6B-1: one conversation with history that grows, search on every turn.
CONVO = [
    "I'm thinking about buying an electric car. What should I consider first?",
    "How does charging at home work, and what does installing a charger cost?",
    "How much does real-world range drop between summer and winter?",
    "Compare the Tesla Model 3 and the Hyundai Ioniq 6 on range and price.",
    "Are there federal tax credits for buying an EV right now?",
    "How long do EV batteries last, and what does a replacement cost?",
    "How do insurance costs compare with a similar gas car?",
    "Is it worth waiting a year for newer models?",
    "How does pricing work at public fast chargers?",
    "Given everything we discussed, give me a short list of pros and cons.",
]

# P6B-2: ordinary chat questions, single and multi-part. Each is asked at all three levels.
QUESTIONS = [
    "What's the difference between a Roth IRA and a traditional IRA?",
    "How long should I boil an egg for a jammy yolk?",
    "Explain how a heat pump works, and whether it makes sense in a cold climate like Minnesota.",
    "I have three days in Lisbon. Plan an itinerary and include one day trip.",
    "What are the main causes of inflation, and what can a central bank do about each one?",
    "Write a short, polite email declining a meeting invitation.",
    "What is the latest stable version of Python, and what are its headline features?",
    "If I invest $500 a month at a 7% annual return, roughly how much will I have after 20 years?",
    "Which is better for a beginner, React or Vue? Give three concrete reasons for each.",
    "Suggest a 30-minute weeknight dinner using chicken thighs and common pantry staples.",
    "Explain what caused the 2008 financial crisis in plain language, then list three regulatory changes that followed.",
    "What's the difference between a virus and a bacterium, and why don't antibiotics work on viruses?",
]

SMOKE = "How many times does the letter r appear in the word strawberry? Check carefully."


# FreshQA end to end: search always on (thinking Auto/Off/Full) plus a no-search, no-think baseline.
FRESH_ARMS = [("on", "auto"), ("on", "off"), ("on", "full"), ("off", "off")]


def fresh_pick(items: list[dict], per_category: int, seed: int) -> list[dict]:
    """A fixed-seed, category-stratified sample, in id order within each category."""
    rng, picked = random.Random(seed), []
    for category in sorted({it["category"] for it in items}):
        pool = [it for it in items if it["category"] == category]
        picked += sorted(rng.sample(pool, min(per_category, len(pool))), key=lambda it: it["id"])
    return picked


def level_order(i: int) -> list[str]:
    """Rotate the level order per question so no level is always first on a cold search."""
    return LEVELS[i % 3:] + LEVELS[:i % 3]


def parse_event(line: str) -> dict | None:
    """One app SSE line to its event dict, or None for keep-alives and junk."""
    if not line.startswith("data:"):
        return None
    try:
        ev = json.loads(line[5:].strip())
    except json.JSONDecodeError:
        return None
    return ev if isinstance(ev, dict) else None


async def send(client: httpx.AsyncClient, app: str, chat_id: int, content: str,
               thinking: str, search: bool) -> dict:
    """POST one message and time it the way a user would see it."""
    rec: dict = {"chat_id": chat_id, "thinking": thinking, "search": search,
                 "t_wall": time.time(), "ttft_ms": None, "ttfc_ms": None, "e2e_ms": None,
                 "reasoning": "", "content": "", "tokens": None, "message_id": None, "sources": None,
                 "status": "incomplete", "error": None}
    t0 = time.perf_counter()
    try:
        async with client.stream("POST", f"{app}/api/chats/{chat_id}/send",
                                 json={"content": content, "thinking": thinking,
                                       "search": search}) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                ev = parse_event(line)
                if ev is None:
                    continue
                now = (time.perf_counter() - t0) * 1e3
                kind = ev.get("type")
                if kind in ("reasoning", "content"):
                    rec["ttft_ms"] = rec["ttft_ms"] if rec["ttft_ms"] is not None else now
                    if kind == "content" and rec["ttfc_ms"] is None:
                        rec["ttfc_ms"] = now
                    rec[kind] += ev.get("text") or ""
                elif kind == "stats":
                    rec["stats"] = ev.get("stats")
                elif kind == "plan":
                    rec["plan"] = {k: ev.get(k) for k in ("search", "queries", "think", "fallback")}
                elif kind == "sources":
                    rec["sources"] = ev.get("sources")
                elif kind == "done":
                    rec["tokens"], rec["message_id"] = ev.get("tokens"), ev.get("message_id")
                    rec["status"] = "ok"
                elif kind == "error":
                    rec["status"], rec["error"] = "error", ev.get("message")
    except Exception as e:
        rec["status"], rec["error"] = "exception", f"{type(e).__name__}: {e}"
    rec["e2e_ms"] = (time.perf_counter() - t0) * 1e3
    return rec


async def new_chat(client: httpx.AsyncClient, app: str, title: str, thinking: str) -> int:
    r = await client.post(f"{app}/api/chats", json={"title": title, "thinking_default": thinking})
    r.raise_for_status()
    return int(r.json()["id"])


def flush(path: str, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def line(rec: dict) -> str:
    return (f"  {rec['phase']:<6} q={rec['qid']:<3} t={rec['turn_index']:<3} "
            f"{rec['thinking']:<5} {rec['status']:<9} ttft={fmt(rec['ttft_ms'])} "
            f"e2e={fmt(rec['e2e_ms'])} tok={rec['tokens']} think_chars={len(rec['reasoning'])}")


async def run(a: argparse.Namespace) -> int:
    bad = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0)) as client:
        if a.mode == "smoke":
            for level in LEVELS:
                chat_id = await new_chat(client, a.app, f"smoke-{level}", level)
                rec = await send(client, a.app, chat_id, SMOKE, level, False)
                rec.update(phase="smoke", qid=0, turn_index=1)
                flush(a.out, rec)
                print(line(rec), flush=True)
            return smoke_verdict([json.loads(x) for x in open(a.out)
                                  if json.loads(x).get("phase") == "smoke"][-3:])
        if a.mode == "convo":
            prompts = json.load(open(a.prompts)) if a.prompts else CONVO
            chat_id = await new_chat(client, a.app, a.title, a.convo_level)
            for i, q in enumerate(prompts[:a.turns], start=1):
                search = True if a.convo_search == "on" else a.convo_search
                rec = await send(client, a.app, chat_id, q, a.convo_level, search)
                rec.update(phase="convo", qid=i, turn_index=i)
                flush(a.out, rec)
                print(line(rec), flush=True)
                bad += rec["status"] != "ok"
                await asyncio.sleep(a.gap_s)
        if a.mode == "levels":
            for i, q in enumerate(QUESTIONS[:a.questions]):
                for level in level_order(i):
                    chat_id = await new_chat(client, a.app, f"p6b-q{i}-{level}", level)
                    rec = await send(client, a.app, chat_id, q, level, True)
                    rec.update(phase="levels", qid=i, turn_index=1)
                    flush(a.out, rec)
                    print(line(rec), flush=True)
                    bad += rec["status"] != "ok"
                    await asyncio.sleep(a.gap_s)
        if a.mode == "fresh":
            items = [json.loads(line_) for line_ in open(a.items) if line_.strip()]
            for i, it in enumerate(fresh_pick(items, a.per_category, a.seed)):
                for search, level in FRESH_ARMS[i % 4:] + FRESH_ARMS[:i % 4]:
                    chat_id = await new_chat(client, a.app, f"fresh-{it['id']}-{search}-{level}", level)
                    rec = await send(client, a.app, chat_id, it["user"], level, search)
                    rec.update(phase="fresh", qid=it["id"], category=it["category"], turn_index=1,
                               arm=f"search-{search}/think-{level}")
                    flush(a.out, rec)
                    print(line(rec), flush=True)
                    bad += rec["status"] != "ok"
                    await asyncio.sleep(a.gap_s)
    print(f"{a.mode}: {bad} failed sends")
    return 1 if bad else 0


def smoke_verdict(recs: list[dict]) -> int:
    """Pass/fail on the three behaviours a thinking level must show. Prints why."""
    by = {r["thinking"]: r for r in recs}
    fails = []
    for level in LEVELS:
        r = by.get(level)
        if r is None or r["status"] != "ok":
            fails.append(f"{level}: no completed reply ({r and r['error']})")
            continue
        if not r["content"].strip():
            fails.append(f"{level}: empty answer")
        if "<think>" in r["content"] or "</think>" in r["content"]:
            fails.append(f"{level}: think tags leaked into the answer; is --reasoning-parser qwen3 on?")
        if r["tokens"] is None:
            fails.append(f"{level}: no usage token count reached the app")
    if "off" in by and by["off"]["reasoning"]:
        fails.append("off: reasoning arrived with budget 0")
    if "full" in by and not by["full"]["reasoning"]:
        fails.append("full: no reasoning at all; is --reasoning-parser qwen3 on?")
    # 128 tokens is ~500-700 chars of English. Far past that means the budget did not bind.
    if "brief" in by and len(by["brief"]["reasoning"]) > 1500:
        fails.append(f"brief: {len(by['brief']['reasoning'])} reasoning chars, budget 128 did not bind")
    if "brief" in by and "full" in by and len(by["full"]["reasoning"]) <= len(by["brief"]["reasoning"]):
        print("  WARN full reasoned no longer than brief on the smoke prompt; check by hand")
    for f in fails:
        print(f"  FAIL {f}")
    print("smoke: PASS" if not fails else f"smoke: {len(fails)} FAILURES")
    return 1 if fails else 0


# ---------------------------------------------------------------- report

def fmt(x, nd: int = 0) -> str:
    return "-" if x is None else f"{x:,.{nd}f}"


def pc(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def p50(xs: list[float]) -> float | None:
    s = sorted(x for x in xs if x is not None)
    return s[(len(s) - 1) // 2] if s else None


def load_tokenizer(path: str | None):
    """Qwen's tokenizer.json from the HF cache, so thinking and answer can be split exactly."""
    paths = [path] if path else sorted(glob.glob(
        "/opt/llm/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/*/tokenizer.json"))
    if not paths or not paths[0]:
        return None
    from tokenizers import Tokenizer
    return Tokenizer.from_file(paths[0])


def count(tok, text: str) -> int:
    if tok is None:
        return round(len(text) / 4)
    return len(tok.encode(text, add_special_tokens=False).ids)


def join(app_recs: list[dict], gw_recs: list[dict]) -> list[dict]:
    """Attach the gateway's spans to each app turn on (chat_id, turn_index)."""
    # 6c planner calls share (chat_id, turn_index); join the answer. Absent purpose = answer.
    gw = {(g.get("chat_id"), g.get("turn_index")): g for g in gw_recs
          if g.get("purpose", "answer") == "answer"}
    out = []
    for r in app_recs:
        g = gw.get((r["chat_id"], r["turn_index"])) or {}
        spans = [g.get(k) for k in ("search_ms", "fetch_ms", "extract_ms")]
        out.append({**r, "gw": bool(g),
                    "retrieval_ms": sum(spans) if all(x is not None for x in spans) else None,
                    "trim_ms": g.get("trim_ms"), "upstream_ttft_ms": g.get("ttft_ms"),
                    "upstream_e2e_ms": g.get("e2e_ms"), "prompt_tokens": g.get("prompt_tokens"),
                    "cached_tokens": g.get("cached_tokens"), "injected_est": g.get("injected_tokens_est"),
                    "n_sources": g.get("n_sources"), "search_error": g.get("search_error"),
                    "budget": g.get("thinking_budget")})
    return out


def report(a: argparse.Namespace) -> int:
    app_recs = [json.loads(x) for x in open(a.app_out) if x.strip()]
    gw_recs = [json.loads(x) for x in open(a.gw_trace) if x.strip()]
    rows = join([r for r in app_recs if r.get("phase") in ("convo", "levels")], gw_recs)
    tok = load_tokenizer(a.tokenizer)
    for r in rows:
        r["think_tok"], r["answer_tok"] = count(tok, r["reasoning"]), count(tok, r["content"])
    unit = "tokens" if tok else "tokens ESTIMATED at 4 chars/token (no tokenizer found)"
    print(f"config: {a.label}")
    print(f"joined {sum(r['gw'] for r in rows)}/{len(rows)} app turns to gateway traces; "
          f"thinking split in {unit}")

    convo = [r for r in rows if r["phase"] == "convo" and r["status"] == "ok"]
    if convo:
        print(f"\n## P6B-1  one conversation, level {convo[0]['thinking']}, search on, concurrency 1  [{a.label}]")
        print("turn | prompt tok | cached | sources | injected est | retrieval ms | upstream TTFT ms"
              " | user TTFT ms | retrieval share of TTFT | first answer ms | user E2E ms"
              " | inference share of E2E")
        for r in convo:
            share = r["retrieval_ms"] / r["ttft_ms"] if r["retrieval_ms"] and r["ttft_ms"] else None
            inf = r["upstream_e2e_ms"] / r["e2e_ms"] if r["upstream_e2e_ms"] and r["e2e_ms"] else None
            r["share"], r["inf"] = share, inf
            print(f"{r['turn_index']:>4} | {fmt(r['prompt_tokens'])} | {fmt(r['cached_tokens'])} | "
                  f"{fmt(r['n_sources'])} | {fmt(r['injected_est'])} | {fmt(r['retrieval_ms'])} | "
                  f"{fmt(r['upstream_ttft_ms'])} | {fmt(r['ttft_ms'])} | {pc(share)} | "
                  f"{fmt(r['ttfc_ms'])} | {fmt(r['e2e_ms'])} | {pc(inf)}")
        late = [r for r in convo if r["turn_index"] >= 6]
        print(f"turns 6-10 p50 (n={len(late)}): user TTFT {fmt(p50([r['ttft_ms'] for r in late]))} ms, "
              f"retrieval share {pc(p50([r['share'] for r in late]))}, "
              f"inference share of E2E {pc(p50([r['inf'] for r in late]))}")

    lv = [r for r in rows if r["phase"] == "levels" and r["status"] == "ok"]
    if lv:
        print(f"\n## P6B-2  {len({r['qid'] for r in lv})} chat questions x 3 levels, turn 1, search on  [{a.label}]")
        print("level | n | E2E p50 s | E2E max s | first answer token p50 s | thinking tok p50 | thinking tok max"
              " | answer tok p50 | usage tok p50")
        for level in LEVELS:
            g = [r for r in lv if r["thinking"] == level]
            if not g:
                continue
            print(f"{level:<5} | {len(g)} | {fmt(p50([r['e2e_ms'] / 1e3 for r in g]), 1)} | "
                  f"{fmt(max(r['e2e_ms'] for r in g) / 1e3, 1)} | "
                  f"{fmt(p50([r['ttfc_ms'] and r['ttfc_ms'] / 1e3 for r in g]), 1)} | "
                  f"{fmt(p50([r['think_tok'] for r in g]))} | {fmt(max(r['think_tok'] for r in g))} | "
                  f"{fmt(p50([r['answer_tok'] for r in g]))} | {fmt(p50([r['tokens'] for r in g]))}")
        full = {r["qid"]: r["think_tok"] for r in lv if r["thinking"] == "full"}
        if full:
            n = len(full)
            print(f"full thinking over 128 tok (brief cut it off): {sum(v > 128 for v in full.values())}/{n}; "
                  f"over 1,000 tok: {sum(v > 1000 for v in full.values())}/{n}")
            print("per question, full thinking tokens: " +
                  ", ".join(f"q{k}={v}" for k, v in sorted(full.items())))
        # The split must add back up to usage, or the tokenizer is not the one that generated.
        ratios = [(r["think_tok"] + r["answer_tok"]) / r["tokens"] for r in lv if r["tokens"]]
        print(f"check: (thinking + answer) / usage.completion_tokens p50 = {fmt(p50(ratios), 3)} "
              "(should sit just under 1.0; the </think> and newlines are not in either text)")

    errs = [r for r in rows if r["status"] != "ok" or r.get("search_error")]
    for r in errs:
        print(f"  NOTE {r['phase']} q{r['qid']} {r['thinking']}: status={r['status']} "
              f"error={r['error']} search_error={r['search_error']}")
    return 0


# ---------------------------------------------------------------- selftest

def selftest() -> int:
    fails = []

    def chk(name, got, want):
        if got != want:
            fails.append(f"  FAIL {name}: got {got!r}, want {want!r}")

    chk("level order rotates", [level_order(i)[0] for i in range(3)], ["off", "brief", "full"])
    pool = [{"id": f"{c}-{k}", "category": c} for c in ("a", "b") for k in range(5)]
    pick = fresh_pick(pool, 3, 0)
    chk("fresh sample is stratified", sorted(it["category"] for it in pick), ["a"] * 3 + ["b"] * 3)
    chk("fresh sample is reproducible", [it["id"] for it in fresh_pick(pool, 3, 0)], [it["id"] for it in pick])
    chk("fresh arms: search on x3 thinking levels, plus a no-search baseline",
        sorted(FRESH_ARMS), [("off", "off"), ("on", "auto"), ("on", "full"), ("on", "off")])
    chk("every level once", sorted(level_order(4)), sorted(LEVELS))
    chk("parse data line", parse_event('data: {"type":"done","tokens":3}'), {"type": "done", "tokens": 3})
    chk("ignore ping", parse_event(": ping"), None)
    chk("p50 odd", p50([3, 1, 2]), 2)
    chk("p50 skips None", p50([None, 5]), 5)
    chk("p50 of nothing is None, not zero", pc(p50([])), "-")

    app = [{"phase": "convo", "chat_id": 7, "turn_index": 2, "status": "ok"}]
    gw = [{"chat_id": 7, "turn_index": 1, "search_ms": 9, "fetch_ms": 9, "extract_ms": 9},
          {"chat_id": 7, "turn_index": 2, "search_ms": 100, "fetch_ms": 200, "extract_ms": 5,
           "ttft_ms": 40, "e2e_ms": 900}]
    j = join(app, gw)[0]
    chk("join picks the matching turn", j["retrieval_ms"], 305)
    plan_row = {"chat_id": 7, "turn_index": 2, "purpose": "plan", "e2e_ms": 50}
    chk("join ignores a planner row sharing the key",
        join(app, [gw[1], plan_row])[0]["upstream_e2e_ms"], 900)
    chk("join carries upstream e2e", j["upstream_e2e_ms"], 900)
    chk("unjoined turn is flagged", join([{**app[0], "turn_index": 3}], gw)[0]["gw"], False)
    chk("missing span is None, not a partial sum",
        join(app, [{"chat_id": 7, "turn_index": 2, "search_ms": 1}])[0]["retrieval_ms"], None)

    ok = {"status": "ok", "content": "three", "tokens": 9, "error": None}
    good = [{**ok, "thinking": "off", "reasoning": ""},
            {**ok, "thinking": "brief", "reasoning": "x" * 400},
            {**ok, "thinking": "full", "reasoning": "x" * 2000}]
    chk("smoke passes a correct trio", smoke_verdict(good), 0)
    chk("smoke fails reasoning on off", smoke_verdict([{**good[0], "reasoning": "hm"}, *good[1:]]), 1)
    chk("smoke fails an unbounded brief", smoke_verdict([good[0], {**good[1], "reasoning": "x" * 5000},
                                                         good[2]]), 1)
    chk("smoke fails leaked tags", smoke_verdict([good[0], good[1], {**good[2], "content": "</think> 3"}]), 1)
    print("\n".join(fails) if fails else "appdrive selftest: PASS")
    return 1 if fails else 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--app", default="http://127.0.0.1:8090")
    r.add_argument("--mode", choices=["smoke", "convo", "levels", "fresh"], required=True)
    r.add_argument("--items", default="data/plansets/freshqa.jsonl", help="fresh mode: FreshQA label file")
    r.add_argument("--per-category", type=int, default=20, help="fresh mode: questions per category")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="results/p6b-app.jsonl")
    r.add_argument("--turns", type=int, default=len(CONVO))
    r.add_argument("--questions", type=int, default=len(QUESTIONS))
    r.add_argument("--convo-level", choices=LEVELS + ["auto"], default="brief")
    r.add_argument("--convo-search", choices=["on", "auto", "off"], default="on",
                   help="6c search mode; 'on' is sent as the legacy boolean true")
    r.add_argument("--prompts", default=None, help="JSON list of user messages replacing CONVO")
    r.add_argument("--title", default="p6b-convo", help="chat title; not 'New chat', so no title call")
    r.add_argument("--gap-s", type=float, default=1.5, help="pause between sends; Brave rate limits")
    p = sub.add_parser("report")
    p.add_argument("--app-out", default="results/p6b-app.jsonl")
    p.add_argument("--gw-trace", default="results/p6b-gw.jsonl")
    p.add_argument("--tokenizer", default=None, help="tokenizer.json; the HF cache is searched if omitted")
    p.add_argument("--label", default="Qwen3-8B fp8 weights, fp8 KV, vLLM 0.27.1 V1 runner, 16k ctx, "
                   "prefix cache on, A10G, via app + gateway")
    a = ap.parse_args()
    return asyncio.run(run(a)) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    sys.exit(main())

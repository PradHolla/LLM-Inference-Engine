#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28"]
# ///
"""
mchat.py -- convo.py's multi-conversation sibling: K interleaved chats for cache
thrashing (M3), or one conversation pushed past the 16k context limit (M6).

  uv run tools/mchat.py --mode thrash --chats 1,2,4,6,8 --turns 12 --out results/m3.jsonl
  uv run tools/mchat.py --mode overflow --strategy window --turns 60 --out results/m6-window.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable

try:
    import httpx
except ImportError:
    sys.exit("pip install httpx")

PROMPTS = [
    "Explain what a KV cache is in transformer inference.",
    "Why does decode read the whole model for every single token?",
    "What limits how many users a single GPU can serve at once?",
    "How does continuous batching differ from static batching?",
    "What does quantization actually change about the arithmetic?",
    "Why is prefill compute-bound but decode memory-bound?",
    "What is a prefix cache and when does it help?",
    "Explain speculative decoding without using the word draft.",
]


@dataclass
class Result:
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    status: str = "ok"
    error: str | None = None


@dataclass
class ThrashRow:
    k: int
    chat_id: int
    turn: int
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    prefix_hit_rate: float | None = None
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    completion_tokens: int | None = None
    status: str = "ok"
    error: str | None = None


@dataclass
class OverflowRow:
    turn: int
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    prefix_hit_rate: float | None = None
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    completion_tokens: int | None = None
    trim_strategy: str | None = None
    dropped_turns: int | None = None
    status: str = "ok"
    error: str | None = None


def hit_rate(before: tuple[float, float] | None, after: tuple[float, float] | None) -> float | None:
    """(hits_after - hits_before) / (queries_after - queries_before) across one request.
    `is not None`, not truthiness: a (0.0, 0.0) pair is falsy and would silently skip real data."""
    if before is not None and after is not None and after[0] > before[0]:
        return (after[1] - before[1]) / (after[0] - before[0])
    return None


async def prefix_stats(client: Any, metrics_url: str) -> tuple[float, float] | None:
    """vLLM's cumulative prefix-cache counters (queries, hits) from /metrics.
    A second, independent source alongside usage.prompt_tokens_details.cached_tokens."""
    try:
        r = await client.get(metrics_url.rstrip("/") + "/metrics", timeout=10)
        q = h = None
        for line in r.text.splitlines():
            if line.startswith("#"):
                continue
            name = line.split("{")[0].split(" ")[0]
            # external_ is the KV-connector's cache and sits at 0.0; without this guard
            # it overwrites the real counters with zero (see tools/convo.py, incident 20250904).
            if "external_" in name:
                continue
            if name.endswith("prefix_cache_queries_total"):
                q = float(line.rsplit(" ", 1)[1])
            elif name.endswith("prefix_cache_hits_total"):
                h = float(line.rsplit(" ", 1)[1])
        return (q, h) if q is not None and h is not None else None
    except Exception:
        return None


async def read_trace(client: Any, url: str) -> tuple[str | None, int | None]:
    """The gateway's own last trace: (trim_strategy, dropped_turns). None, None if unreadable."""
    try:
        r = await client.get(url.rstrip("/") + "/gateway/traces?n=1", timeout=10)
        traces = (r.json() or {}).get("traces") or []
        if not traces:
            return None, None
        t = traces[-1]
        return t.get("trim_strategy"), t.get("dropped_turns")
    except Exception:
        return None, None


async def one_turn(client: Any, url: str, model: str, messages: list[dict],
                    max_tokens: int, extra: dict | None = None) -> tuple[Result, str]:
    """One streamed turn. Token counts come from the server's usage block, never from
    counting SSE chunks -- one chunk stopped being one token the moment speculation is on."""
    body = {"model": model, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": max_tokens, "temperature": 0.0}
    if extra:
        body.update(extra)
    res = Result()
    text: list[str] = []
    t0 = time.perf_counter()
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=body,
                                 timeout=600) as r:
            if r.status_code != 200:
                res.status, res.error = "http_error", f"{r.status_code}: {(await r.aread())[:200]!r}"
                return res, ""
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                u = ev.get("usage") or {}
                if u.get("completion_tokens") is not None:
                    res.completion_tokens = int(u["completion_tokens"])
                    res.prompt_tokens = int(u.get("prompt_tokens") or 0)
                    det = u.get("prompt_tokens_details") or {}
                    if det.get("cached_tokens") is not None:
                        res.cached_tokens = int(det["cached_tokens"])
                ch = ev.get("choices") or []
                if not ch:
                    continue
                piece = (ch[0].get("delta") or {}).get("content") or ""
                if not piece:
                    continue
                if res.ttft_ms is None:
                    res.ttft_ms = (time.perf_counter() - t0) * 1e3
                text.append(piece)
        res.e2e_ms = (time.perf_counter() - t0) * 1e3
        if res.ttft_ms is None:
            res.status = "empty"
    except Exception as e:
        res.status, res.error = "exception", f"{type(e).__name__}: {e}"
    return res, "".join(text)


async def interleave_chats(n_chats: int, n_turns: int,
                            step: Callable[[int, int], Awaitable[None]]) -> list[tuple[int, int]]:
    """Every chat takes turn 1, then every chat takes turn 2, so K chats genuinely compete
    for cache within a turn. Returns the observed (turn, chat_id) call order."""
    order: list[tuple[int, int]] = []

    async def call(cid: int, turn: int) -> None:
        order.append((turn, cid))
        await step(cid, turn)

    for turn in range(1, n_turns + 1):
        await asyncio.gather(*(call(cid, turn) for cid in range(n_chats)))
    return order


def fmt(v: float | None, spec: str = ".1f") -> str:
    """Format a possibly-missing metric for the progress line, without crashing on None."""
    return format(v, spec) if v is not None else "n/a"


async def run_thrash(a: argparse.Namespace) -> int:
    out = open(a.out, "w")
    ks = [int(x) for x in a.chats.split(",")]
    async with httpx.AsyncClient() as client:
        for k in ks:
            histories: dict[int, list[dict]] = {cid: [] for cid in range(k)}

            async def step(cid: int, turn: int, k: int = k, histories=histories) -> None:
                q = PROMPTS[(turn - 1) % len(PROMPTS)]
                histories[cid].append({"role": "user", "content": q})
                before = await prefix_stats(client, a.metrics_url)
                res, reply = await one_turn(client, a.url, a.model, histories[cid], a.max_tokens)
                after = await prefix_stats(client, a.metrics_url)
                hr = hit_rate(before, after)
                row = ThrashRow(k=k, chat_id=cid, turn=turn, prompt_tokens=res.prompt_tokens,
                                cached_tokens=res.cached_tokens, prefix_hit_rate=hr,
                                ttft_ms=res.ttft_ms, e2e_ms=res.e2e_ms,
                                completion_tokens=res.completion_tokens,
                                status=res.status, error=res.error)
                out.write(json.dumps(asdict(row)) + "\n")
                out.flush()
                if res.status == "ok":
                    histories[cid].append({"role": "assistant", "content": reply})
                else:
                    histories[cid].pop()
                print(f"  k={k:>2} turn={turn:>3} chat={cid:>2} prompt={res.prompt_tokens} "
                      f"hit={fmt(hr, '.2f')} ttft={fmt(res.ttft_ms)}ms status={res.status}")

            await interleave_chats(k, a.turns, step)
    out.close()
    return 0


async def run_overflow(a: argparse.Namespace) -> int:
    out = open(a.out, "w")
    messages: list[dict] = []
    async with httpx.AsyncClient() as client:
        for i in range(1, a.turns + 1):
            q = PROMPTS[(i - 1) % len(PROMPTS)]
            messages.append({"role": "user", "content": q})
            before = await prefix_stats(client, a.metrics_url)
            res, reply = await one_turn(client, a.url, a.model, messages, a.max_tokens,
                                        extra={"gw_context": a.strategy})
            after = await prefix_stats(client, a.metrics_url)
            trim_strategy, dropped_turns = await read_trace(client, a.url)
            row = OverflowRow(turn=i, prompt_tokens=res.prompt_tokens, cached_tokens=res.cached_tokens,
                              prefix_hit_rate=hit_rate(before, after), ttft_ms=res.ttft_ms,
                              e2e_ms=res.e2e_ms, completion_tokens=res.completion_tokens,
                              trim_strategy=trim_strategy, dropped_turns=dropped_turns,
                              status=res.status, error=res.error)
            out.write(json.dumps(asdict(row)) + "\n")
            out.flush()
            if res.status == "ok":
                messages.append({"role": "assistant", "content": reply})
            else:
                messages.pop()
            print(f"  turn={i:>3} prompt={res.prompt_tokens} strategy={trim_strategy} "
                  f"dropped={dropped_turns} ttft={fmt(res.ttft_ms)}ms status={res.status}")
    out.close()
    return 0


class _FakeStreamCtx:
    """Stand-in for httpx.AsyncClient.stream()'s async context manager. No network."""

    def __init__(self, status_code: int, lines: list[str]):
        self.status_code = status_code
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b""


class _FakeClient:
    """Records the last forwarded body and replays canned SSE lines. Touches no network."""

    def __init__(self, lines: list[str] | None = None, status_code: int = 200,
                 raise_on_stream: bool = False):
        self.last_body: dict | None = None
        self._lines = lines or []
        self._status = status_code
        self._raise = raise_on_stream

    def stream(self, method: str, url: str, json: dict | None = None, timeout: float | None = None):
        self.last_body = json
        if self._raise:
            raise RuntimeError("connection refused")
        return _FakeStreamCtx(self._status, self._lines)


def selftest() -> int:
    fails: list[str] = []

    def chk(name: str, cond: bool) -> None:
        if not cond:
            fails.append(f"  FAIL {name}")

    # 1. K independent histories share no message objects
    histories = {cid: [] for cid in range(3)}
    histories[0].append({"role": "user", "content": "a"})
    chk("chat 0 mutation does not leak into chat 1", histories[1] == [])
    chk("histories are distinct list objects", histories[0] is not histories[1])

    # 2. interleaving order is genuinely round-robin, not just K*turns in count
    order: list[tuple[int, int]] = []

    async def step(cid: int, turn: int) -> None:
        order.append((turn, cid))
        await asyncio.sleep(0)
    asyncio.run(interleave_chats(3, 2, step))
    expected = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    chk(f"interleave order is round-robin, got {order}", order == expected)

    # 3. no prompt_tokens_details -> cached_tokens is None, not 0
    fc = _FakeClient(lines=[
        'data: {"choices":[{"delta":{"content":"hi"}}],'
        '"usage":{"completion_tokens":1,"prompt_tokens":5}}',
        "data: [DONE]",
    ])
    res, _ = asyncio.run(one_turn(fc, "http://x", "m", [{"role": "user", "content": "q"}], 10))
    chk("cached_tokens is None when prompt_tokens_details is absent", res.cached_tokens is None)

    # 4. usage's completion_tokens wins over the number of SSE chunks
    fc2 = _FakeClient(lines=[
        'data: {"choices":[{"delta":{"content":"ab"}}]}',
        'data: {"choices":[{"delta":{"content":"cd"}}],'
        '"usage":{"completion_tokens":7,"prompt_tokens":10,'
        '"prompt_tokens_details":{"cached_tokens":3}}}',
        "data: [DONE]",
    ])
    res2, _ = asyncio.run(one_turn(fc2, "http://x", "m", [{"role": "user", "content": "q"}], 10))
    chk(f"completion_tokens comes from usage (7), got {res2.completion_tokens}",
        res2.completion_tokens == 7)
    chk("cached_tokens comes from usage", res2.cached_tokens == 3)

    # 5. a failed request is recorded, never raised
    fc3 = _FakeClient(raise_on_stream=True)
    res3, _ = asyncio.run(one_turn(fc3, "http://x", "m", [{"role": "user", "content": "q"}], 10))
    chk("a raising client is recorded as status=exception, not raised", res3.status == "exception")

    # 6. overflow mode sends gw_context in the outgoing body
    fc4 = _FakeClient(lines=[
        'data: {"choices":[{"delta":{"content":"ok"}}],'
        '"usage":{"completion_tokens":1,"prompt_tokens":1}}',
        "data: [DONE]",
    ])
    asyncio.run(one_turn(fc4, "http://x", "m", [{"role": "user", "content": "q"}], 10,
                        extra={"gw_context": "window"}))
    chk("gw_context reaches the forwarded body", (fc4.last_body or {}).get("gw_context") == "window")

    print("\n".join(fails) if fails else "selftest: PASS")
    if fails:
        print(f"selftest: {len(fails)} FAILURES")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", default=None, help="served id; discovered if omitted")
    ap.add_argument("--mode", choices=("thrash", "overflow"), default="thrash")
    ap.add_argument("--chats", default="1,2,4,6,8", help="comma list of K, thrash only")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--strategy", choices=("none", "window", "summarize"), default="none",
                    help="gw_context value, overflow only")
    ap.add_argument("--metrics-url", default="http://localhost:8000",
                    help="engine base url for prefix-cache counters")
    ap.add_argument("--out", default="results/mchat.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.model:
        # bench.py's --model test 404s on vLLM; ask the server what it serves.
        r = httpx.get(f"{a.url}/v1/models", timeout=10)
        data = (r.json() or {}).get("data") or []
        if not data:
            print("could not resolve served model from /v1/models")
            return 1
        a.model = data[0]["id"]
    if a.mode == "thrash":
        print(f"mchat: thrash mode, chats={a.chats}, turns={a.turns}, model={a.model}")
        return asyncio.run(run_thrash(a))
    print(f"mchat: overflow mode, strategy={a.strategy}, turns={a.turns}, model={a.model}")
    return asyncio.run(run_overflow(a))


if __name__ == "__main__":
    sys.exit(main())

"""
context.py -- trim an over-budget conversation for the prompt sent upstream. Two
selectable strategies so their prefix-cache behavior can be measured, not assumed.

  python -m gateway.context --selftest
"""
from __future__ import annotations

import asyncio
import copy
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from gateway.search import CHARS_PER_TOKEN

Summarizer = Callable[[list[dict[str, Any]]], Awaitable[str]]


@dataclass
class TrimResult:
    """The trimmed message list plus what happened to produce it."""
    messages: list[dict[str, Any]]
    strategy: str
    dropped_turns: int
    summary_text: Optional[str]
    trim_ms: float
    est_tokens_before: int
    est_tokens_after: int


def _content_chars(content: Any) -> int:
    """Character count of a message's content, OpenAI multipart included."""
    if isinstance(content, str):
        return len(content)
    try:
        return len(json.dumps(content))
    except TypeError:
        return len(str(content))


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token count: total content characters divided by CHARS_PER_TOKEN."""
    chars = sum(_content_chars(m.get("content", "")) for m in messages)
    return int(chars / CHARS_PER_TOKEN)


def _has_system(messages: list[dict[str, Any]]) -> bool:
    return bool(messages) and messages[0].get("role") == "system"


def _window_drop(messages: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], int]:
    """Drop oldest non-system messages, in pairs where possible, until under budget."""
    has_sys = _has_system(messages)
    head = messages[:1] if has_sys else []
    body = messages[1:] if has_sys else list(messages)
    dropped = 0
    while body and estimate_tokens(head + body) > budget:
        if len(body) >= 2 and body[0].get("role") != body[1].get("role"):
            body = body[2:]
            dropped += 2
        else:
            body = body[1:]
            dropped += 1
        while body and body[0].get("role") == "assistant":
            body = body[1:]
            dropped += 1
    return head + body, dropped


def _synth_summary(n: int) -> str:
    return f"[Earlier conversation summarised: {n} messages omitted.]"


async def apply(messages: list[dict[str, Any]],
                 budget_tokens: int,
                 strategy: str = "none",
                 summarizer: Optional[Summarizer] = None) -> TrimResult:
    """Trim messages to budget_tokens using the given strategy. Never raises."""
    t0 = time.perf_counter()
    msgs = copy.deepcopy(messages) if messages else []
    before = estimate_tokens(msgs)

    if strategy == "none" or before <= budget_tokens:
        return TrimResult(msgs, strategy, 0, None,
                          (time.perf_counter() - t0) * 1e3, before, before)

    if strategy == "window":
        trimmed, dropped = _window_drop(msgs, budget_tokens)
        after = estimate_tokens(trimmed)
        return TrimResult(trimmed, strategy, dropped, None,
                          (time.perf_counter() - t0) * 1e3, before, after)

    if strategy == "summarize":
        has_sys = _has_system(msgs)
        head = msgs[:1] if has_sys else []
        body = msgs[1:] if has_sys else list(msgs)

        dropped_msgs: list[dict[str, Any]] = []
        remaining = list(body)
        while remaining and estimate_tokens(head + remaining) > budget_tokens:
            dropped_msgs.append(remaining[0])
            remaining = remaining[1:]

        if summarizer is not None and dropped_msgs:
            try:
                summary = await summarizer(dropped_msgs)
            except Exception:
                summary = _synth_summary(len(dropped_msgs))
        else:
            summary = _synth_summary(len(dropped_msgs)) if dropped_msgs else None

        if summary is None:
            result = head + remaining
        else:
            summary_msg = {"role": "system", "content": summary}
            result = head + [summary_msg] + remaining
            while remaining and estimate_tokens(result) > budget_tokens:
                dropped_msgs.append(remaining[0])
                remaining = remaining[1:]
                summary = _synth_summary(len(dropped_msgs)) if summarizer is None else summary
                summary_msg = {"role": "system", "content": summary}
                result = head + [summary_msg] + remaining

        after = estimate_tokens(result)
        return TrimResult(result, strategy, len(dropped_msgs), summary,
                          (time.perf_counter() - t0) * 1e3, before, after)

    return TrimResult(msgs, "none", 0, None,
                      (time.perf_counter() - t0) * 1e3, before, before)


def _mk(role: str, content: Any) -> dict[str, Any]:
    return {"role": role, "content": content}


def selftest() -> int:
    """Offline. Covers both strategies, the malformed-input contract, and no-mutation."""
    fails = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    convo = [_mk("system", "You are helpful.")]
    for i in range(20):
        convo.append(_mk("user", f"question {i} " * 20))
        convo.append(_mk("assistant", f"answer {i} " * 20))

    for strat in ("none", "window", "summarize"):
        r = asyncio.run(apply(convo, budget_tokens=10_000_000, strategy=strat))
        check(r.dropped_turns == 0, f"{strat}: under-budget dropped {r.dropped_turns}")
        check(r.messages == convo, f"{strat}: under-budget mutated messages")

    r = asyncio.run(apply(convo, budget_tokens=200, strategy="window"))
    check(estimate_tokens(r.messages) <= 200, "window: did not get under budget")
    check(r.messages[0]["content"] == "You are helpful.", "window: dropped system message")
    check(r.messages[1]["role"] != "assistant", "window: leading non-system is assistant")

    r_none = asyncio.run(apply(convo, budget_tokens=200, strategy="none"))
    check(r_none.dropped_turns == 0, "none: dropped messages despite being over budget")
    check(len(r_none.messages) == len(convo), "none: message count changed")

    r = asyncio.run(apply(convo, budget_tokens=300, strategy="summarize"))
    n_summaries = sum(1 for m in r.messages if isinstance(m.get("content"), str)
                      and m["content"].startswith("[Earlier conversation summarised"))
    check(n_summaries == 1, f"summarize: expected exactly one summary, got {n_summaries}")
    check(r.messages[1]["role"] == "system", "summarize: summary not role system")
    check(r.messages[0]["content"] == "You are helpful.", "summarize: system not preserved at head")
    check(estimate_tokens(r.messages) <= 300, "summarize: over budget including summary")
    check(r.summary_text is not None and r.summary_text in r.messages[1]["content"],
         "summarize: summary_text not reflected in inserted message")

    async def _fake_summarizer(dropped: list[dict[str, Any]]) -> str:
        return f"custom summary of {len(dropped)}"
    r = asyncio.run(apply(convo, budget_tokens=300, strategy="summarize", summarizer=_fake_summarizer))
    check(r.messages[1]["content"].startswith("custom summary of"),
         "summarize: custom summarizer output not used")

    for name, bad in (
        ("empty list", []),
        ("system only", [_mk("system", "hi")]),
        ("list content", [_mk("system", "s"), _mk("user", [{"type": "text", "text": "hi"}])]),
        ("no content key", [_mk("system", "s"), {"role": "user"}]),
    ):
        for strat in ("none", "window", "summarize"):
            try:
                r = asyncio.run(apply(bad, budget_tokens=1, strategy=strat))
                check(isinstance(r, TrimResult), f"malformed {name}/{strat}: bad return type")
            except Exception as e:
                fails.append(f"malformed {name}/{strat} raised: {type(e).__name__}: {e}")

    original = copy.deepcopy(convo)
    asyncio.run(apply(convo, budget_tokens=200, strategy="window"))
    asyncio.run(apply(convo, budget_tokens=300, strategy="summarize"))
    check(convo == original, "apply mutated the caller's input list or its dicts")

    r = asyncio.run(apply(convo, budget_tokens=200, strategy="window"))
    check(r.est_tokens_before is not None and r.est_tokens_after is not None,
         "token estimates not populated")
    check(r.est_tokens_after <= r.est_tokens_before, "est_tokens_after exceeds before")

    for f in fails:
        print(f"  FAIL {f}")
    print(f"selftest: {'PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    print("usage: python -m gateway.context --selftest")

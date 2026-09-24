"""The chat turn as a LangGraph graph: load_context -> [plan] -> [search] -> answer.

  async for part in agent.GRAPH.astream(state, stream_mode="custom", version="v2"): ...
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import time
from typing import Any, TypedDict
from urllib.parse import urlsplit

import httpx
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from gateway import search as gsearch
from . import budget, config, db, prompts

LOGGER = logging.getLogger(__name__)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "search": {"type": "boolean"},
        "queries": {"type": "array", "items": {"type": "string"}, "maxItems": config.MAX_QUERIES},
        "think": {"type": "boolean"},
    },
    "required": ["search", "queries", "think"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {"type": "json_schema",
                   "json_schema": {"name": "plan", "schema": PLAN_SCHEMA, "strict": True}}
PENDING: dict[int, asyncio.Task] = {}


class AgentState(TypedDict, total=False):
    chat_id: int
    turn_index: int
    model: str
    user_text: str
    history: list[dict]
    search_mode: str
    thinking: str
    today: str
    base: list[dict]
    plan: dict
    block: str


def _wire(message: dict) -> dict:
    return {"role": message["role"], "content": message["content"]}


def needs_plan(search_mode: str, thinking: str) -> bool:
    """Section 3's table: only search Off with a forced thinking level skips the planner."""
    return not (search_mode == "off" and thinking != config.AUTO)


def thinking_budget(thinking: str, think: bool) -> int | None:
    """gw_thinking_budget to send; None means omit the field (unbounded)."""
    if thinking == config.AUTO:
        return None if think else 0
    return config.THINKING_LEVELS[thinking]


def sampling(think: bool) -> dict:
    return dict(config.SAMPLING_THINK if think else config.SAMPLING_PLAIN)


def boundary(history: list[dict], user_text: str, today: str,
             user_tokens: int | None = None) -> int:
    """Leading history messages to summarise, sized for the worst case: think on, search on."""
    system_tokens = budget.message_tokens({"content": prompts.system_prompt(today)})
    if user_tokens is None:
        user_tokens = budget.message_tokens({"content": user_text})
    limit = budget.history_max(system_tokens, budget.summary_reserve(), user_tokens)
    return budget.summary_boundary([budget.message_tokens(m) for m in history], limit)


def parse_plan(text: str) -> dict:
    """Tolerant: take the outermost {...}, require the three typed fields. Raises ValueError."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in planner reply")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict) or not isinstance(data.get("search"), bool) \
            or not isinstance(data.get("think"), bool):
        raise ValueError("planner reply lacks boolean search/think")
    queries = data.get("queries", [])
    if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
        raise ValueError("planner queries is not a list of strings")
    return {"search": data["search"], "think": data["think"],
            "queries": [re.sub(r"\s+", " ", q).strip() for q in queries if q.strip()]}


def resolve(parsed: dict | None, search_mode: str, thinking: str, user_text: str) -> dict:
    """Apply the user's forced controls over the planner, or the section 3 fallback."""
    fallback = parsed is None
    if search_mode in ("on", "off"):
        search = search_mode == "on"
    else:
        search = True if fallback else parsed["search"]
    queries = ([] if fallback else parsed["queries"])[:config.MAX_QUERIES] if search else []
    if search and not queries:
        queries = [user_text.strip()]
    if thinking == config.AUTO:
        think = True if fallback else parsed["think"]
    else:
        think = thinking != "off"
    return {"search": search, "queries": queries, "think": think, "fallback": fallback}


def plan_messages(base: list[dict], user_text: str) -> list[dict]:
    """[system][summary][history][user][instruction]: the answer's prefix, instruction last."""
    return [*base, {"role": "user", "content": user_text},
            {"role": "system", "content": prompts.PLANNER_INSTRUCTION}]


def plan_body(messages: list[dict], model: str, chat_id: int | None,
              turn_index: int | None) -> dict:
    body = {"model": model, "messages": messages, "stream": False,
            "max_tokens": config.PLAN_MAX_TOKENS, "response_format": RESPONSE_FORMAT,
            **sampling(False), "gw_thinking_budget": 0, "gw_purpose": "plan"}
    if chat_id is not None:
        body["gw_chat_id"] = chat_id
    if turn_index is not None:
        body["gw_turn_index"] = turn_index
    return body


def _reply_text(payload: Any) -> str:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    message = choices[0].get("message") if choices and isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    for key in ("content", "reasoning", "reasoning_content"):
        if isinstance(message.get(key), str) and message[key].strip():
            return message[key]
    return ""


async def call_planner(messages: list[dict], model: str, chat_id: int | None = None,
                       turn_index: int | None = None,
                       gateway_url: str | None = None) -> tuple[dict | None, float, str | None]:
    """One planner call. Returns (parsed plan or None, elapsed ms, error). Never raises."""
    started = time.perf_counter()
    try:
        async with asyncio.timeout(config.PLAN_TIMEOUT_S):
            async with httpx.AsyncClient(timeout=config.PLAN_TIMEOUT_S) as client:
                response = await client.post(
                    f"{gateway_url or config.GATEWAY_URL}/v1/chat/completions",
                    json=plan_body(messages, model, chat_id, turn_index))
                response.raise_for_status()
        parsed = parse_plan(_reply_text(response.json()))
        return parsed, (time.perf_counter() - started) * 1000, None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        LOGGER.warning("planner fallback chat_id=%s turn_index=%s %s", chat_id, turn_index, error)
        return None, (time.perf_counter() - started) * 1000, error


async def _search_one(query: str, client: httpx.AsyncClient) -> gsearch.SearchOutcome:
    try:
        return await gsearch.run_search(query, client)
    except Exception as exc:
        return gsearch.SearchOutcome(query=query, error=f"{type(exc).__name__}")


def merge_sources(outcomes: list[gsearch.SearchOutcome]) -> list[gsearch.Source]:
    """Usable sources across every query, first occurrence of each URL kept, in order."""
    seen: set[str] = set()
    merged = []
    for outcome in outcomes:
        for source in outcome.sources:
            if source.ok and source.url not in seen:
                seen.add(source.url)
                merged.append(source)
    return merged


def cap_block(queries: list[str], sources: list[gsearch.Source],
              cap: int | None = None) -> tuple[str, list[gsearch.Source]]:
    """Render the block, shrinking every source equally until it fits the token cap."""
    cap = cap or config.SEARCH_CAP_TOKENS
    if not sources:
        return prompts.empty_search(queries), []
    chars = max(len(source.text) for source in sources)
    for _ in range(16):
        trimmed = [dataclasses.replace(source, text=source.text[:chars]) for source in sources]
        rendered = gsearch.render_block(gsearch.SearchOutcome(query="; ".join(queries),
                                                              sources=trimmed))
        block = prompts.search_block(queries, rendered)
        size = budget.count_tokens(block)
        if size <= cap or chars == 0:
            return block, trimmed
        chars = max(0, min(chars - 1, int(chars * cap / size * 0.95)))
    return block, trimmed


def source_cards(sources: list[gsearch.Source]) -> list[dict]:
    cards = []
    for source in sources:
        host = urlsplit(source.url).hostname or ""
        cards.append({"title": source.title or host, "url": source.url,
                      "site": host.removeprefix("www.")})
    return cards


async def load_context(state: AgentState) -> dict:
    write = get_stream_writer()
    history, today = state["history"], state["today"]
    k = boundary(history, state["user_text"], today)
    summary = None
    if k > 0:
        covered = int(history[k - 1]["id"])
        summary = await asyncio.to_thread(db.get_summary, covered)
        task = PENDING.get(covered)
        if summary is None and task is not None:
            await asyncio.wait({task}, timeout=config.SUMMARY_WAIT_S)
            if task.done() and not task.cancelled() and task.exception() is None:
                summary = task.result()
    base = [{"role": "system", "content": prompts.system_prompt(today)}]
    if summary is not None:
        base.append(prompts.summary_message(summary))
    base.extend(_wire(message) for message in history[k:])
    write({"type": "_stats", "summary_used": summary is not None,
           "history_tokens": sum(budget.message_tokens(m) for m in base[1:])})
    update: dict = {"base": base}
    if not needs_plan(state["search_mode"], state["thinking"]):
        update["plan"] = resolve({"search": False, "queries": [], "think": False},
                                 state["search_mode"], state["thinking"], state["user_text"])
    return update


async def plan(state: AgentState) -> dict:
    write = get_stream_writer()
    write({"type": "status", "stage": "planning"})
    parsed, ms, _ = await call_planner(plan_messages(state["base"], state["user_text"]),
                                       state["model"], state["chat_id"], state["turn_index"])
    decision = resolve(parsed, state["search_mode"], state["thinking"], state["user_text"])
    write({"type": "plan", **decision})
    write({"type": "_stats", "plan_ms": ms, "plan_fallback": decision["fallback"]})
    return {"plan": decision}


async def search(state: AgentState) -> dict:
    write = get_stream_writer()
    queries = state["plan"]["queries"]
    write({"type": "status", "stage": "searching", "query": "; ".join(queries),
           "queries": queries})
    started = time.perf_counter()
    async with httpx.AsyncClient() as client:
        outcomes = await asyncio.gather(*(_search_one(query, client) for query in queries))
    block, used = cap_block(queries, merge_sources(outcomes))
    write({"type": "sources", "sources": source_cards(used)})
    write({"type": "_stats", "search_ms": (time.perf_counter() - started) * 1000})
    return {"block": block}


def answer_messages(state: AgentState) -> list[dict]:
    messages = list(state["base"])
    if state.get("block"):
        messages.append({"role": "system", "content": state["block"]})
    messages.append({"role": "user", "content": state["user_text"]})
    return messages


def answer_body(state: AgentState) -> dict:
    think = state["plan"]["think"]
    messages = answer_messages(state)
    body = {"model": state["model"], "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}, **sampling(think),
            "max_tokens": budget.max_tokens_for(budget.prompt_tokens(messages)),
            "gw_events": True, "gw_purpose": "answer",
            "gw_chat_id": state["chat_id"], "gw_turn_index": state["turn_index"]}
    level = thinking_budget(state["thinking"], think)
    if level is not None:
        body["gw_thinking_budget"] = level
    return body


async def answer(state: AgentState) -> dict:
    write = get_stream_writer()
    body = answer_body(state)
    write({"type": "_stats", "budget_max_tokens": body["max_tokens"]})
    LOGGER.warning("gateway request chat_id=%s turn_index=%s gw_thinking_budget=%s max_tokens=%s",
                   state["chat_id"], state["turn_index"],
                   body.get("gw_thinking_budget", "omitted"), body["max_tokens"])
    async with httpx.AsyncClient(timeout=600.0) as client:
        async with client.stream("POST", f"{config.GATEWAY_URL}/v1/chat/completions",
                                 json=body) as response:
            response.raise_for_status()
            write({"type": "_open"})
            async for line in response.aiter_lines():
                _relay_line(line, write)
    return {}


def _relay_line(line: str, write) -> None:
    """Translate one gateway SSE line into writer events. Tokens only ever from usage."""
    if not line.startswith("data:"):
        return
    raw = line[5:].strip()
    if raw == "[DONE]":
        write({"type": "_done"})
        return
    try:
        chunk = json.loads(raw)
    except json.JSONDecodeError:
        return
    if not isinstance(chunk, dict):
        return
    if chunk.get("object") == "gw.event":
        kind = chunk.get("type")
        if kind == "status" and chunk.get("stage") == "generating":
            write({"type": "status", "stage": "generating"})
        elif kind == "stats" and isinstance(chunk.get("stats"), dict):
            write({"type": "_gw_stats", "stats": chunk["stats"]})
        return
    choices = chunk.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    delta = choice.get("delta") or {}
    if not isinstance(delta, dict):
        delta = {}
    thought = delta.get("reasoning")
    if not isinstance(thought, str):
        thought = delta.get("reasoning_content")
    text = delta.get("content")
    if isinstance(text, str) and text:
        write({"type": "content", "text": text})
    if isinstance(thought, str) and thought:
        write({"type": "reasoning", "text": thought})
    usage = chunk.get("usage") or {}
    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
        try:
            write({"type": "_usage", "completion_tokens": int(usage["completion_tokens"])})
        except (TypeError, ValueError):
            pass


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("load_context", load_context)
    graph.add_node("plan", plan)
    graph.add_node("search", search)
    graph.add_node("answer", answer)
    graph.add_edge(START, "load_context")
    graph.add_conditional_edges("load_context", lambda s: "answer" if "plan" in s else "plan",
                                ["plan", "answer"])
    graph.add_conditional_edges("plan", lambda s: "search" if s["plan"]["search"] else "answer",
                                ["search", "answer"])
    graph.add_edge("search", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


GRAPH = build_graph()


def summary_body(messages: list[dict], model: str) -> dict:
    return {"model": model, "messages": messages, "stream": False,
            "max_tokens": config.SUMMARY_MAX_TOKENS, **sampling(False),
            "gw_thinking_budget": 0, "gw_purpose": "summary"}


async def summarize(chat_id: int, history: list[dict], k: int, today: str, model: str) -> str:
    """Summarise history[:k] on top of the deepest earlier summary, store it, return the text."""
    covered = int(history[k - 1]["id"])
    previous = await asyncio.to_thread(db.latest_summary, [int(m["id"]) for m in history[:k - 1]])
    start = previous[0] + 1 if previous else 0
    messages = [{"role": "system", "content": prompts.system_prompt(today)}]
    if previous:
        messages.append(prompts.summary_message(previous[1]))
    messages.extend(_wire(message) for message in history[start:k])
    messages.append({"role": "system", "content": prompts.SUMMARY_INSTRUCTION})
    async with httpx.AsyncClient(timeout=config.SUMMARY_TIMEOUT_S) as client:
        response = await client.post(f"{config.GATEWAY_URL}/v1/chat/completions",
                                     json=summary_body(messages, model))
        response.raise_for_status()
    text = _reply_text(response.json()).strip()
    if not text:
        raise ValueError("summary response was empty")
    await asyncio.to_thread(db.save_summary, covered, chat_id, text)
    return await asyncio.to_thread(db.get_summary, covered) or text


async def schedule_summary(chat_id: int, path: list[dict], today: str,
                           model: str) -> asyncio.Task | None:
    """After an answer: if the NEXT turn crosses the boundary, write its summary in the background."""
    k = boundary(path, "", today, user_tokens=config.NEXT_USER_TOKENS)
    if k == 0:
        return None
    covered = int(path[k - 1]["id"])
    if covered in PENDING or await asyncio.to_thread(db.get_summary, covered) is not None:
        return None
    task = asyncio.create_task(summarize(chat_id, path, k, today, model))
    PENDING[covered] = task

    def _done(finished: asyncio.Task) -> None:
        PENDING.pop(covered, None)
        if not finished.cancelled() and finished.exception() is not None:
            LOGGER.warning("summary failed chat_id=%s covered_through=%s %r", chat_id, covered,
                           finished.exception())
    task.add_done_callback(_done)
    return task

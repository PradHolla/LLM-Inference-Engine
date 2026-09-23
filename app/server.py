"""Chat application API and static UI.

  uvicorn app.server:app --host 127.0.0.1 --port 8090
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from . import config, db


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
MODEL_ID = config.MODEL_ID
UI_DIR = Path(__file__).resolve().parent / "ui"


class CreateChatBody(BaseModel):
    title: str = "New chat"
    thinking_default: str | None = None


class RenameChatBody(BaseModel):
    title: str


class SendBody(BaseModel):
    content: str
    thinking: str | None = None
    search: bool = config.SEARCH_DEFAULT
    parent_id: int | None = None


class RegenerateBody(BaseModel):
    message_id: int
    thinking: str | None = None
    search: bool = config.SEARCH_DEFAULT


class HeadBody(BaseModel):
    message_id: int


def _row(row):
    return dict(row) if row is not None else None


def _message(chat_id: int, row) -> dict:
    message = dict(row)
    for column, key in (("sources_json", "sources"), ("stats_json", "stats")):
        raw = message.pop(column, None)
        message[key] = json.loads(raw) if raw else None
    message["stopped"] = bool(message["stopped"])
    message["sibling_ids"] = db.sibling_ids(chat_id, int(message["id"]))
    return message


def _chat_payload(chat_id: int) -> dict:
    chat = _require_chat(chat_id)
    return {"chat": _row(chat), "messages": [_message(chat_id, row)
                                               for row in db.get_branch(chat_id)]}


def _require_chat(chat_id: int):
    chat = db.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="chat not found")
    return chat


async def _discover_model() -> None:
    """Cache the gateway model id, falling back when discovery is unavailable."""
    global MODEL_ID
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{config.GATEWAY_URL}/v1/models")
            response.raise_for_status()
        payload = response.json()
        models = payload.get("data") if isinstance(payload, dict) else None
        model_id = models[0].get("id") if isinstance(models, list) and models else None
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("missing model id")
        MODEL_ID = model_id
        return
    except Exception:
        MODEL_ID = config.MODEL_ID
        LOGGER.warning("gateway model discovery failed; using fallback %s", MODEL_ID)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    await _discover_model()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


@app.get("/")
async def root():
    return RedirectResponse("/ui/")


@app.get("/api/chats")
async def chats_list():
    return [_row(chat) for chat in db.list_chats()]


@app.post("/api/chats")
async def chats_create(body: CreateChatBody):
    thinking = body.thinking_default or config.DEFAULT_THINKING
    if thinking not in config.THINKING_LEVELS:
        raise HTTPException(status_code=422, detail="invalid thinking_default")
    chat_id = db.create_chat(body.title.strip() or "New chat", thinking)
    return _row(db.get_chat(chat_id))


@app.get("/api/chats/{chat_id}")
async def chats_get(chat_id: int):
    return _chat_payload(chat_id)


@app.patch("/api/chats/{chat_id}")
async def chats_rename(chat_id: int, body: RenameChatBody):
    _require_chat(chat_id)
    db.rename_chat(chat_id, body.title)
    return _row(db.get_chat(chat_id))


@app.delete("/api/chats/{chat_id}")
async def chats_delete(chat_id: int):
    _require_chat(chat_id)
    db.delete_chat(chat_id)
    return {"ok": True}


@app.get("/api/config")
async def app_config():
    return {"thinking_levels": [{"id": name, **config.THINKING_OPTIONS[name]}
                                 for name in config.THINKING_LEVELS],
            "default_thinking": config.DEFAULT_THINKING,
            "search_default": config.SEARCH_DEFAULT}


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{config.GATEWAY_URL}/health")
        return {"gateway": response.is_success}
    except Exception:
        return {"gateway": False}


def _event(event_type: str, **fields: object) -> dict[str, str]:
    return {"data": json.dumps({"type": event_type, **fields})}


async def _protected_thread(function, *args):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    cancelled = False
    while True:
        try:
            return await asyncio.shield(task), cancelled
        except asyncio.CancelledError:
            cancelled = True
            if task.done():
                return task.result(), True


def _title_fallback(content: str) -> str:
    words = re.sub(r"\s+", " ", content).strip().split(" ")
    title = " ".join(words[:7]).strip(" .!?\t\n")
    return title[:64] or "New chat"


def _title_body(content: str) -> dict:
    return {"model": MODEL_ID, "messages": [
        {"role": "system", "content": "Give this conversation a concise title of a few words. Return only the title."},
        {"role": "user", "content": content},
    ], "stream": False, "max_tokens": 32, "temperature": 0.2,
        "gw_thinking_budget": 0, "gw_search": False}


async def _request_title(content: str) -> str:
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(f"{config.GATEWAY_URL}/v1/chat/completions",
                                     json=_title_body(content))
        response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    message = choices[0].get("message") if choices else None
    title = message.get("content") if isinstance(message, dict) else None
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title response was empty")
    return re.sub(r"\s+", " ", title).strip(" \"'`\n\t")[:80]


async def _make_title(content: str) -> str:
    try:
        return await _request_title(content)
    except Exception:
        return _title_fallback(content)


def _number(value) -> float | int | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _base_stats(thinking: str, searched: bool) -> dict:
    return {"ttft_ms": None, "e2e_ms": None, "engine_ttft_ms": None,
            "search_ms": None, "prompt_tokens": None, "cached_tokens": None,
            "completion_tokens": None, "decode_tok_s": None,
            "thinking_level": thinking, "searched": searched}


def _answer_stream(chat_id: int, user_id: int | None, assistant_parent_id: int | None,
                   messages: list[dict], thinking: str, searched: bool, user_text: str,
                   turn_index: int, previous_head_id: int | None = None):
    body = {"model": MODEL_ID, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "gw_search": searched, "gw_events": True, "gw_query": user_text,
            "gw_chat_id": chat_id, "gw_turn_index": turn_index}
    budget = config.THINKING_LEVELS[thinking]
    if budget is not None:
        body["gw_thinking_budget"] = budget
    LOGGER.warning("gateway request chat_id=%s turn_index=%s gw_thinking_budget=%s", chat_id,
                   turn_index, body.get("gw_thinking_budget", "omitted"))

    async def stream():
        content = ""
        reasoning = ""
        tokens: int | None = None
        sources: list[dict] | None = None
        gw_stats: dict = {}
        stats = _base_stats(thinking, searched)
        message_id: int | None = None
        stream_open = False
        complete = False
        cancelled = False
        error: str | None = None
        first_token_at: float | None = None
        started = time.perf_counter()
        yield _event("start", user_message_id=user_id,
                      assistant_parent_id=assistant_parent_id)
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream("POST", f"{config.GATEWAY_URL}/v1/chat/completions",
                                         json=body) as response:
                    response.raise_for_status()
                    stream_open = True
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            complete = True
                            continue
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(chunk, dict):
                            continue
                        if chunk.get("object") == "gw.event":
                            kind = chunk.get("type")
                            if kind == "search.started":
                                yield _event("status", stage="searching",
                                             query=chunk.get("query", ""))
                            elif kind == "sources":
                                raw_sources = chunk.get("sources")
                                sources = []
                                for item in raw_sources if isinstance(raw_sources, list) else []:
                                    if not isinstance(item, dict):
                                        continue
                                    url = str(item.get("url") or "")
                                    host = urlsplit(url).hostname or ""
                                    sources.append({"title": str(item.get("title") or host),
                                                    "url": url,
                                                    "site": host.removeprefix("www.")})
                                yield _event("sources", sources=sources)
                            elif kind == "status" and chunk.get("stage") == "generating":
                                yield _event("status", stage="generating")
                            elif kind == "stats" and isinstance(chunk.get("stats"), dict):
                                gw_stats.update(chunk["stats"])
                            continue
                        choices = chunk.get("choices") or []
                        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
                        delta = choice.get("delta") or {}
                        if not isinstance(delta, dict):
                            delta = {}
                        text = delta.get("content")
                        thought = delta.get("reasoning")
                        if not isinstance(thought, str):
                            thought = delta.get("reasoning_content")
                        if isinstance(text, str) and text:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            content += text
                            yield _event("content", text=text)
                        if isinstance(thought, str) and thought:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            reasoning += thought
                            yield _event("reasoning", text=thought)
                        usage = chunk.get("usage") or {}
                        if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                            try:
                                tokens = int(usage["completion_tokens"])
                            except (TypeError, ValueError):
                                pass
        except asyncio.CancelledError:
            cancelled = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            now = time.perf_counter()
            stats.update({key: _number(gw_stats.get(gateway_key)) for key, gateway_key in (
                ("engine_ttft_ms", "engine_ttft_ms"), ("search_ms", "search_ms"),
                ("prompt_tokens", "prompt_tokens"), ("cached_tokens", "cached_tokens"),
                ("completion_tokens", "completion_tokens"), ("decode_tok_s", "decode_tok_s"))})
            stats["ttft_ms"] = ((first_token_at - started) * 1000
                                if first_token_at is not None else None)
            stats["e2e_ms"] = (now - started) * 1000
            if stream_open:
                message_id, write_cancelled = await _protected_thread(
                    db.add_message, chat_id, assistant_parent_id, "assistant", content,
                    reasoning or None, tokens, sources, stats, bool(cancelled and not complete))
                cancelled = cancelled or write_cancelled
                LOGGER.info("gateway response chat_id=%s turn_index=%s app_rtt_ms=%.1f", chat_id,
                            turn_index, stats["e2e_ms"])
            elif user_id is not None:
                _, rollback_cancelled = await _protected_thread(
                    db.rollback_user_message, chat_id, user_id, previous_head_id)
                cancelled = cancelled or rollback_cancelled

        if cancelled:
            raise asyncio.CancelledError
        if error is not None:
            yield _event("error", message=error)
            return
        if not stream_open:
            return
        if sources is not None or gw_stats:
            yield _event("stats", stats=stats)
        if complete:
            yield _event("done", tokens=tokens, message_id=message_id)
            if turn_index == 1:
                current = await asyncio.to_thread(db.get_chat, chat_id)
                if current is not None and current["title"] == "New chat":
                    title = await _make_title(user_text)
                    await asyncio.to_thread(db.rename_chat, chat_id, title)
                    yield _event("title", title=title)

    return EventSourceResponse(stream(), ping=None)


@app.post("/api/chats/{chat_id}/send")
async def chats_send(chat_id: int, payload: SendBody):
    chat = _require_chat(chat_id)
    thinking = payload.thinking or config.DEFAULT_THINKING
    if thinking not in config.THINKING_LEVELS:
        raise HTTPException(status_code=422, detail="invalid thinking")
    if "parent_id" in payload.model_fields_set:
        parent_id = payload.parent_id
    else:
        parent_id = chat["head_message_id"]
    if parent_id is not None and not db.get_path(chat_id, parent_id):
        raise HTTPException(status_code=422, detail="parent message does not belong to this chat")
    user_id = db.add_message(chat_id, parent_id, "user", payload.content, None, None)
    branch = db.get_branch(chat_id)
    messages = [{"role": row["role"], "content": row["content"]} for row in branch]
    return _answer_stream(chat_id, user_id, user_id, messages, thinking, payload.search,
                          payload.content, sum(row["role"] == "user" for row in branch),
                          chat["head_message_id"])


@app.post("/api/chats/{chat_id}/regenerate")
async def chats_regenerate(chat_id: int, payload: RegenerateBody):
    chat = _require_chat(chat_id)
    original = db.get_message(chat_id, payload.message_id)
    if original is None or original["role"] != "assistant":
        raise HTTPException(status_code=422, detail="message_id must be an assistant message in this chat")
    parent_id = original["parent_id"]
    if parent_id is None:
        raise HTTPException(status_code=422, detail="assistant message has no user parent")
    path = db.get_path(chat_id, parent_id)
    if not path or path[-1]["role"] != "user":
        raise HTTPException(status_code=422, detail="assistant parent is not a user message")
    thinking = payload.thinking or chat["thinking_default"]
    if thinking not in config.THINKING_LEVELS:
        raise HTTPException(status_code=422, detail="invalid thinking")
    messages = [{"role": row["role"], "content": row["content"]} for row in path]
    return _answer_stream(chat_id, None, parent_id, messages, thinking, payload.search,
                          path[-1]["content"], sum(row["role"] == "user" for row in path))


@app.post("/api/chats/{chat_id}/head")
async def chats_head(chat_id: int, payload: HeadBody):
    _require_chat(chat_id)
    leaf_id = db.newest_leaf(chat_id, payload.message_id)
    if leaf_id is None:
        raise HTTPException(status_code=404, detail="message not found")
    db.set_head(chat_id, leaf_id)
    return _chat_payload(chat_id)


async def _selftest_async() -> list[str]:
    fails: list[str] = []
    received: list[dict] = []
    title_should_fail = False
    upstream_closed = asyncio.Event()
    old_db_path = db.DB_PATH
    old_gateway = config.GATEWAY_URL
    old_model = MODEL_ID

    def check(name: str, condition: bool) -> None:
        if not condition:
            fails.append(f"  FAIL {name}")

    async def gateway(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin1").split("\r\n")
            length = next((int(line.split(":", 1)[1]) for line in lines
                           if line.lower().startswith("content-length:")), 0)
            request = json.loads(await reader.readexactly(length))
            received.append(request)
            if not request.get("stream"):
                if title_should_fail:
                    payload = b'{"error":"title unavailable"}'
                    writer.write(b"HTTP/1.1 503 Unavailable\r\nContent-Type: application/json\r\n"
                                 b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
                else:
                    payload = b'{"choices":[{"message":{"content":"A useful title"}}]}'
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                                 b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
                await writer.drain()
                return

            user_text = request["messages"][-1]["content"]
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                         b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
            await writer.drain()
            def frame(data: bytes) -> bytes:
                return b"%x\r\n%s\r\n" % (len(data), data)

            parts = []
            if request.get("gw_search"):
                query = request.get("gw_query", user_text)
                parts.extend([
                    json.dumps({"object": "gw.event", "type": "search.started",
                                "query": query}).encode(),
                    json.dumps({"object": "gw.event", "type": "sources", "sources": [
                        {"title": "Fixture source", "url": "https://example.test/story"}
                    ]}).encode(),
                ])
            parts.extend([
                json.dumps({"object": "gw.event", "type": "status",
                            "stage": "generating"}).encode(),
                b'{"choices":[{"delta":{"reasoning":"thinking "}}]}',
                b'{"choices":[{"delta":{"reasoning_content":"fallback"}}]}',
                b'{"choices":[{"delta":{"content":"answer"}}]}',
                b'{"choices":[],"usage":{"completion_tokens":17}}',
                b"[DONE]",
            ])
            stats = {"engine_ttft_ms": 8.5, "search_ms": 6.0, "prompt_tokens": 42,
                     "cached_tokens": 11, "completion_tokens": 17, "decode_tok_s": 25.0}
            if user_text == "hold open":
                parts = parts[:4]
                for item in parts:
                    writer.write(frame(b"data: " + item + b"\n\n"))
                    await writer.drain()
                await reader.read()
                upstream_closed.set()
                return
            for item in parts:
                writer.write(frame(b"data: " + item + b"\n\n"))
                await writer.drain()
            writer.write(frame(b"data: " + json.dumps({"object": "gw.event", "type": "stats",
                "stats": stats}).encode() + b"\n\n"))
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass

    async def collect(response) -> list[dict]:
        return [item async for item in response.body_iterator]

    def payloads(items: list[dict]) -> list[dict]:
        return [json.loads(item["data"]) for item in items if item.get("data")]

    with tempfile.TemporaryDirectory() as tmp:
        db.DB_PATH = str(Path(tmp) / "chats.db")
        db.init_db()
        chat_id = db.create_chat("test", "brief")
        server = await asyncio.start_server(gateway, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        config.GATEWAY_URL = f"http://127.0.0.1:{port}"
        try:
            response = await chats_send(chat_id, SendBody(content="hello", thinking="off", search=True))
            events = payloads(await collect(response))
            done = next(event for event in events if event["type"] == "done")
            branch = db.get_branch(chat_id)
            first_request = received[0]
            check("send forwards the fixed sampling values",
                  first_request.get("temperature") == 0.6 and first_request.get("top_p") == 0.95 and
                  first_request.get("top_k") == 20 and first_request.get("min_p") == 0.0)
            check("off sends zero budget", first_request.get("gw_thinking_budget") == 0)
            check("turn index starts at one", first_request.get("gw_turn_index") == 1)
            check("app opts into gateway progress frames", first_request.get("gw_events") is True)
            check("tokens come from usage", branch[-1]["tokens"] == 17)
            check("done carries saved message", done.get("message_id") == branch[-1]["id"])
            check("thinking field and deprecated fallback are both read",
                  branch[-1]["thinking"] == "thinking fallback")
            saved_message = _message(chat_id, branch[-1])
            check("statistics and search sources persist", saved_message["sources"] == [
                {"title": "Fixture source", "url": "https://example.test/story",
                 "site": "example.test"}] and saved_message["stats"]["cached_tokens"] == 11)

            search_chat = db.create_chat("search pipeline", "brief")
            search_response = await chats_send(search_chat, SendBody(
                content="find it", thinking="brief", search=True))
            search_events = payloads(await collect(search_response))
            check("search pipeline translates in order", [event["type"] for event in search_events
                  if event["type"] in ("status", "sources", "reasoning", "content", "stats", "done")]
                  == ["status", "sources", "status", "reasoning", "reasoning", "content", "stats", "done"])

            title_chat = db.create_chat("New chat", "brief")
            title_response = await chats_send(title_chat, SendBody(
                content="name this conversation", thinking="off", search=False))
            title_events = payloads(await collect(title_response))
            title_request = received[-1]
            check("automatic title follows done", [event["type"] for event in title_events][-2:] ==
                  ["done", "title"])
            check("automatic title is saved", db.get_chat(title_chat)["title"] == "A useful title")
            check("title request excludes trace join fields", "gw_chat_id" not in title_request and
                  "gw_turn_index" not in title_request and title_request.get("gw_thinking_budget") == 0 and
                  title_request.get("gw_search") is False and title_request.get("max_tokens") <= 32)

            title_should_fail = True
            fallback_chat = db.create_chat("New chat", "brief")
            fallback_response = await chats_send(fallback_chat, SendBody(
                content="Fallback title uses the first few words", thinking="off", search=False))
            fallback_events = payloads(await collect(fallback_response))
            expected_fallback = "Fallback title uses the first few words"
            check("automatic title failure emits fallback", fallback_events[-1].get("title") ==
                  expected_fallback and db.get_chat(fallback_chat)["title"] == expected_fallback)
            title_should_fail = False

            tree_chat = db.create_chat("branch tree", "brief")
            initial = payloads(await collect(await chats_send(tree_chat, SendBody(
                content="original", thinking="off", search=False))))
            original_user = initial[0]["user_message_id"]
            original_assistant = next(event["message_id"] for event in initial
                                      if event["type"] == "done")
            edited = payloads(await collect(await chats_send(tree_chat, SendBody(
                content="edited", thinking="full", search=False, parent_id=None))))
            edited_request = received[-1]
            check("full thinking leaves budget absent", "gw_thinking_budget" not in edited_request)
            edited_user = edited[0]["user_message_id"]
            edited_assistant = next(event["message_id"] for event in edited if event["type"] == "done")
            check("edited user message creates a sibling branch",
                  db.sibling_ids(tree_chat, original_user) == [original_user, edited_user] and
                  [row["content"] for row in db.get_path(tree_chat, original_assistant)] ==
                  ["original", "answer"])
            regenerated = payloads(await collect(await chats_regenerate(tree_chat,
                RegenerateBody(message_id=edited_assistant, thinking="off", search=False))))
            regenerated_id = next(event["message_id"] for event in regenerated if event["type"] == "done")
            check("regenerate creates an assistant sibling",
                  db.sibling_ids(tree_chat, edited_assistant) == [edited_assistant, regenerated_id])
            await chats_head(tree_chat, HeadBody(message_id=edited_assistant))
            check("head selects the requested sibling path",
                  db.get_chat(tree_chat)["head_message_id"] == edited_assistant)
            await chats_head(tree_chat, HeadBody(message_id=original_user))
            check("head switch restores the original branch",
                  [row["id"] for row in db.get_branch(tree_chat)] == [original_user, original_assistant])

            stop_chat = db.create_chat("stop", "off")
            stop_response = await chats_send(stop_chat, SendBody(
                content="hold open", thinking="off", search=False))
            seen: list[dict] = []
            reasoning_seen = asyncio.Event()

            async def consume_until_cancelled():
                async for item in stop_response.body_iterator:
                    parsed = json.loads(item["data"])
                    seen.append(parsed)
                    if parsed["type"] == "reasoning":
                        reasoning_seen.set()

            consumer = asyncio.create_task(consume_until_cancelled())
            await asyncio.wait_for(reasoning_seen.wait(), 3.0)
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
            await asyncio.wait_for(upstream_closed.wait(), 3.0)
            stopped_branch = db.get_branch(stop_chat)
            check("cancel persists an assistant row", len(stopped_branch) == 2 and
                  stopped_branch[-1]["role"] == "assistant")
            check("cancel marks the partial row stopped", bool(stopped_branch[-1]["stopped"]))
            check("cancel keeps partial reasoning", (stopped_branch[-1]["thinking"] or "").startswith("thinking"))
            check("cancel moves head to the partial row", db.get_chat(stop_chat)["head_message_id"] ==
                  stopped_branch[-1]["id"])
            after_stop = payloads(await collect(await chats_send(stop_chat, SendBody(
                content="next turn", thinking="off", search=False))))
            next_branch = db.get_branch(stop_chat)
            roles = [row["role"] for row in next_branch]
            check("next send continues without repeated user turns",
                  roles[-2:] == ["user", "assistant"] and
                  all(roles[index:index + 2] != ["user", "user"] for index in range(len(roles) - 1)))

            before_ids = [message["id"] for message in branch]
            previous_head_id = db.get_chat(chat_id)["head_message_id"]
            config.GATEWAY_URL = "http://127.0.0.1:1"
            failed = await chats_send(chat_id, SendBody(content="unreachable", thinking="off", search=False))
            failed_events = payloads(await collect(failed))
            after = db.get_branch(chat_id)
            check("failed stream reports an error", failed_events[-1]["type"] == "error")
            check("failed stream leaves branch unchanged",
                  [message["id"] for message in after] == before_ids)
            check("failed stream restores the head",
                  db.get_chat(chat_id)["head_message_id"] == previous_head_id)
        finally:
            server.close()
            await server.wait_closed()
            db.DB_PATH = old_db_path
            config.GATEWAY_URL = old_gateway
            globals()["MODEL_ID"] = old_model
    return fails


def selftest() -> int:
    """Check streaming request construction and usage-token persistence."""
    fails = asyncio.run(_selftest_async())
    print("\n".join(fails) if fails else "selftest: PASS")
    if fails:
        print(f"selftest: {len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())

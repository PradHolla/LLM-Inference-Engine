"""Chat application API and static UI.

  uvicorn app.server:app --host 127.0.0.1 --port 8090
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re
import sys
import tempfile
import time
from contextlib import aclosing, asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from . import agent, config, db


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
    search: bool | str = config.DEFAULT_SEARCH_MODE
    parent_id: int | None = None


class RegenerateBody(BaseModel):
    message_id: int
    thinking: str | None = None
    search: bool | str = config.DEFAULT_SEARCH_MODE


class HeadBody(BaseModel):
    message_id: int


def _search_mode(value: bool | str) -> str:
    """A boolean from the pre-6c UI means on/off; otherwise one of auto, on, off."""
    if value is True or value is False:
        return "on" if value else "off"
    if value in config.SEARCH_MODES:
        return value
    raise HTTPException(status_code=422, detail="invalid search")


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
            "search_default": config.SEARCH_DEFAULT,
            "search_modes": [{"id": name, **config.SEARCH_OPTIONS[name]}
                             for name in config.SEARCH_MODES],
            "default_search_mode": config.DEFAULT_SEARCH_MODE}


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
        "gw_thinking_budget": 0, "gw_purpose": "title"}


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
            "thinking_level": thinking, "searched": searched,
            "think": None if thinking == config.AUTO else thinking != "off",
            "queries": None,
            "plan_ms": None, "plan_fallback": False, "summary_used": False,
            "history_tokens": None, "budget_max_tokens": None}


_END = object()


async def _pump(state: dict, queue: asyncio.Queue) -> None:
    """Run the graph in its own task so a Stop can cancel it outside sse-starlette's scope."""
    try:
        async with aclosing(agent.GRAPH.astream(state, stream_mode="custom",
                                                version="v2")) as parts:
            async for part in parts:
                queue.put_nowait(part)
    finally:
        queue.put_nowait(_END)


async def _settle(task: asyncio.Task) -> bool:
    """Cancel a task and wait until it has finished; True if we were cancelled meanwhile."""
    task.cancel()
    cancelled = False
    while not task.done():
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError:
            cancelled = True
    return cancelled


GATEWAY_STATS = ("engine_ttft_ms", "prompt_tokens", "cached_tokens", "completion_tokens",
                 "decode_tok_s")


def _answer_stream(chat_id: int, user_id: int | None, assistant_parent_id: int | None,
                   path: list, thinking: str, search_mode: str, turn_index: int,
                   previous_head_id: int | None = None):
    history = [{"id": int(row["id"]), "role": row["role"], "content": row["content"]}
               for row in path]
    today = datetime.date.today().isoformat()
    state = {"chat_id": chat_id, "turn_index": turn_index, "model": MODEL_ID,
             "user_text": history[-1]["content"], "history": history[:-1],
             "search_mode": search_mode, "thinking": thinking, "today": today}

    async def stream():
        content = ""
        reasoning = ""
        tokens: int | None = None
        sources: list[dict] | None = None
        gw_stats: dict = {}
        app_stats: dict = {}
        stats = _base_stats(thinking, False)
        message_id: int | None = None
        stream_open = False
        complete = False
        cancelled = False
        error: str | None = None
        first_token_at: float | None = None
        started = time.perf_counter()
        yield _event("start", user_message_id=user_id,
                      assistant_parent_id=assistant_parent_id)
        queue: asyncio.Queue = asyncio.Queue()
        runner = asyncio.create_task(_pump(state, queue))
        try:
            while (part := await queue.get()) is not _END:
                data = part.get("data") if part.get("type") == "custom" else None
                kind = data.get("type") if isinstance(data, dict) else None
                if kind == "_open":
                    stream_open = True
                elif kind == "_done":
                    complete = True
                elif kind == "_usage":
                    tokens = data["completion_tokens"]
                elif kind == "_gw_stats":
                    gw_stats.update(data["stats"])
                elif kind == "_stats":
                    app_stats.update({k: v for k, v in data.items() if k != "type"})
                elif isinstance(kind, str):
                    if kind in ("content", "reasoning"):
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        if kind == "content":
                            content += data["text"]
                        else:
                            reasoning += data["text"]
                    elif kind == "sources":
                        sources = data["sources"]
                    elif kind == "plan":
                        stats["searched"] = bool(data["search"])
                        stats["think"] = bool(data["think"])
                        stats["queries"] = list(data.get("queries") or [])
                    yield {"data": json.dumps(data)}
            await runner
        except asyncio.CancelledError:
            cancelled = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if not runner.done():
                cancelled = await _settle(runner) or cancelled
            now = time.perf_counter()
            stats.update({key: _number(gw_stats.get(key)) for key in GATEWAY_STATS})
            stats.update(app_stats)
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
        yield _event("stats", stats=stats)
        if complete:
            yield _event("done", tokens=tokens, message_id=message_id)
            try:
                await agent.schedule_summary(chat_id, history + [
                    {"id": message_id, "role": "assistant", "content": content}], today, MODEL_ID)
            except Exception as exc:
                LOGGER.warning("summary scheduling failed chat_id=%s %r", chat_id, exc)
            if turn_index == 1:
                current = await asyncio.to_thread(db.get_chat, chat_id)
                if current is not None and current["title"] == "New chat":
                    title = await _make_title(history[-1]["content"])
                    await asyncio.to_thread(db.rename_chat, chat_id, title)
                    yield _event("title", title=title)

    return EventSourceResponse(stream(), ping=None)


@app.post("/api/chats/{chat_id}/send")
async def chats_send(chat_id: int, payload: SendBody):
    chat = _require_chat(chat_id)
    thinking = payload.thinking or config.DEFAULT_THINKING
    if thinking not in config.THINKING_LEVELS:
        raise HTTPException(status_code=422, detail="invalid thinking")
    search_mode = _search_mode(payload.search)
    if "parent_id" in payload.model_fields_set:
        parent_id = payload.parent_id
    else:
        parent_id = chat["head_message_id"]
    if parent_id is not None and not db.get_path(chat_id, parent_id):
        raise HTTPException(status_code=422, detail="parent message does not belong to this chat")
    user_id = db.add_message(chat_id, parent_id, "user", payload.content, None, None)
    branch = db.get_branch(chat_id)
    return _answer_stream(chat_id, user_id, user_id, branch, thinking, search_mode,
                          sum(row["role"] == "user" for row in branch), chat["head_message_id"])


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
    return _answer_stream(chat_id, None, parent_id, path, thinking, _search_mode(payload.search),
                          sum(row["role"] == "user" for row in path))


@app.post("/api/chats/{chat_id}/head")
async def chats_head(chat_id: int, payload: HeadBody):
    _require_chat(chat_id)
    leaf_id = db.newest_leaf(chat_id, payload.message_id)
    if leaf_id is None:
        raise HTTPException(status_code=404, detail="message not found")
    db.set_head(chat_id, leaf_id)
    return _chat_payload(chat_id)


async def _selftest_async() -> list[str]:
    import anyio
    from gateway import search as gsearch
    from . import budget, prompts

    fails: list[str] = []
    received: list[dict] = []
    fake = {"title_fail": False, "plan": "json", "summary": "ok",
            "plan_reply": {"search": True, "queries": ["rewritten query"], "think": False}}
    upstream_closed = asyncio.Event()
    plan_closed = asyncio.Event()
    plan_arrived = asyncio.Event()
    search_started = asyncio.Event()
    search_cancelled = asyncio.Event()
    old = (db.DB_PATH, config.GATEWAY_URL, MODEL_ID, gsearch.run_search, config.PLAN_TIMEOUT_S,
           config.CONTEXT_WINDOW, list(budget._TOKENIZER))

    def check(name: str, condition: bool) -> None:
        if not condition:
            fails.append(f"  FAIL {name}")

    def respond_json(writer, status: bytes, payload: bytes) -> None:
        writer.write(b"HTTP/1.1 " + status + b"\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)

    def reply(text: str) -> bytes:
        return json.dumps({"choices": [{"message": {"content": text}}]}).encode()

    async def gateway(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin1").split("\r\n")
            length = next((int(line.split(":", 1)[1]) for line in lines
                           if line.lower().startswith("content-length:")), 0)
            request = json.loads(await reader.readexactly(length))
            received.append(request)
            purpose = request.get("gw_purpose")
            if not request.get("stream"):
                if purpose == "plan":
                    if fake["plan"] == "hold":
                        plan_arrived.set()
                        await reader.read()
                        plan_closed.set()
                        return
                    if fake["plan"] == "error":
                        respond_json(writer, b"503 Unavailable", b'{"error":"down"}')
                    else:
                        text = (json.dumps(fake["plan_reply"]) if fake["plan"] == "json"
                                else "I think you should search.")
                        respond_json(writer, b"200 OK", reply(text))
                elif purpose == "summary":
                    if fake["summary"] == "slow":
                        await asyncio.sleep(0.3)
                    if fake["summary"] == "fail":
                        respond_json(writer, b"503 Unavailable", b'{"error":"down"}')
                    else:
                        count = sum(r.get("gw_purpose") == "summary" for r in received)
                        respond_json(writer, b"200 OK", reply(f"SUMMARY-{count}"))
                elif fake["title_fail"]:
                    respond_json(writer, b"503 Unavailable", b'{"error":"title unavailable"}')
                else:
                    respond_json(writer, b"200 OK", reply("A useful title"))
                await writer.drain()
                return

            user_text = request["messages"][-1]["content"]
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                         b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
            await writer.drain()

            def frame(data: bytes) -> bytes:
                return b"%x\r\n%s\r\n" % (len(data), data)

            parts = [
                json.dumps({"object": "gw.event", "type": "status",
                            "stage": "generating"}).encode(),
                b'{"choices":[{"delta":{"reasoning":"thinking "}}]}',
                b'{"choices":[{"delta":{"reasoning_content":"fallback"}}]}',
                b'{"choices":[{"delta":{"content":"answer"}}]}',
                b'{"choices":[],"usage":{"completion_tokens":17}}',
                b"[DONE]",
            ]
            stats = {"engine_ttft_ms": 8.5, "search_ms": None, "prompt_tokens": 42,
                     "cached_tokens": 11, "completion_tokens": 17, "decode_tok_s": 25.0}
            if user_text == "hold open":
                for item in parts[:3]:
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

    searched_queries: list[str] = []

    async def fake_search(query, client, *args, **kwargs):
        searched_queries.append(query)
        if "hang" in query:
            search_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                search_cancelled.set()
                raise
        if "nothing" in query:
            return gsearch.SearchOutcome(query=query, search_ms=1.0, fetch_ms=0.0,
                                         extract_ms=0.0, error="no results")
        return gsearch.SearchOutcome(query=query, sources=[
            gsearch.Source(url="https://www.example.test/story", title="Fixture source",
                           text="fixture body " * 50, ok=True),
            gsearch.Source(url="https://blocked.test/", error="HTTP 403")],
            search_ms=1.0, fetch_ms=1.0, extract_ms=1.0, n_sources=1)

    async def collect(response) -> list[dict]:
        return [json.loads(item["data"]) async for item in response.body_iterator
                if item.get("data")]

    def types(events: list[dict]) -> list[str]:
        return [event["type"] for event in events]

    def answers() -> list[dict]:
        return [r for r in received if r.get("stream")]

    def plans() -> list[dict]:
        return [r for r in received if r.get("gw_purpose") == "plan"]

    async def cancel_after(response, stage: str, ready: asyncio.Event | None = None,
                           plain: bool = False) -> list[dict]:
        seen: list[dict] = []
        reached = asyncio.Event()

        async def consume():
            async for item in response.body_iterator:
                parsed = json.loads(item["data"])
                seen.append(parsed)
                if parsed["type"] == stage or parsed.get("stage") == stage:
                    reached.set()

        if plain:
            consumer = asyncio.create_task(consume())
            await asyncio.wait_for(reached.wait(), 3.0)
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
            return seen
        # sse-starlette cancels through an anyio scope, which re-raises on every later await.
        async with anyio.create_task_group() as group:
            group.start_soon(consume)
            await asyncio.wait_for(reached.wait(), 3.0)
            if ready is not None:
                await asyncio.wait_for(ready.wait(), 3.0)
            group.cancel_scope.cancel()
        return seen

    with tempfile.TemporaryDirectory() as tmp:
        db.DB_PATH = str(Path(tmp) / "chats.db")
        db.init_db()
        gsearch.run_search = fake_search
        server = await asyncio.start_server(gateway, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        config.GATEWAY_URL = f"http://127.0.0.1:{port}"
        today = datetime.date.today().isoformat()
        try:
            cfg = await app_config()
            check("config lists auto thinking first and keeps the legacy search boolean",
                  cfg["thinking_levels"][0]["id"] == "auto" and cfg["default_thinking"] == "auto"
                  and cfg["search_default"] is True and
                  [m["id"] for m in cfg["search_modes"]] == ["auto", "on", "off"])
            check("search accepts booleans and the three modes",
                  [_search_mode(SendBody.model_validate_json(f'{{"content":"x","search":{v}}}').search)
                   for v in ("true", "false", '"auto"', '"on"', '"off"')] ==
                  ["on", "off", "auto", "on", "off"] and
                  _search_mode(SendBody(content="x").search) == "auto")
            try:
                _search_mode("sometimes")
                check("an unknown search mode is rejected", False)
            except HTTPException as exc:
                check("an unknown search mode is rejected", exc.status_code == 422)

            chat_id = db.create_chat("test", "brief")
            fake["plan_reply"] = {"search": False, "queries": [], "think": True}
            events = await collect(await chats_send(chat_id, SendBody(
                content="hello", thinking="off", search=True)))
            check("search on runs plan, search, answer in order", types(events) == [
                "start", "status", "plan", "status", "sources", "status", "reasoning",
                "reasoning", "content", "stats", "done"] and
                [e.get("stage") for e in events if e["type"] == "status"] ==
                ["planning", "searching", "generating"])
            plan_request, answer_request = plans()[-1], answers()[-1]
            check("plan call is tagged, joinable, structured and thinking-off",
                  plan_request.get("gw_purpose") == "plan" and plan_request.get("gw_chat_id") ==
                  chat_id and plan_request.get("gw_turn_index") == 1 and
                  plan_request.get("gw_thinking_budget") == 0 and
                  plan_request.get("stream") is False and plan_request.get("max_tokens") == 96 and
                  plan_request.get("response_format", {}).get("type") == "json_schema" and
                  plan_request.get("temperature") == 0.7 and plan_request.get("top_p") == 0.8)
            check("plan prompt ends user message then instruction",
                  plan_request["messages"][-2] == {"role": "user", "content": "hello"} and
                  plan_request["messages"][-1]["content"] == prompts.PLANNER_INSTRUCTION)
            plan_event = next(e for e in events if e["type"] == "plan")
            check("forced search overrides the planner and uses the user text",
                  plan_event == {"type": "plan", "search": True, "queries": ["hello"],
                                 "think": False, "fallback": False} and searched_queries[-1] == "hello")
            check("off sends zero budget and non-thinking sampling",
                  answer_request.get("gw_thinking_budget") == 0 and
                  answer_request.get("temperature") == 0.7 and answer_request.get("top_p") == 0.8
                  and answer_request.get("top_k") == 20 and answer_request.get("min_p") == 0.0)
            check("answer is tagged and joinable, and never asks the gateway to search",
                  answer_request.get("gw_purpose") == "answer" and
                  answer_request.get("gw_turn_index") == 1 and
                  answer_request.get("gw_events") is True and "gw_search" not in answer_request
                  and "gw_query" not in answer_request)
            sent = answer_request["messages"]
            check("system prompt first with the date on its last line",
                  sent[0]["role"] == "system" and
                  sent[0]["content"].splitlines()[-1] == f"Today's date is {today}.")
            check("search block sits just before the user message, numbered like the cards",
                  sent[-1] == {"role": "user", "content": "hello"} and sent[-2]["role"] == "system"
                  and '"hello"' in sent[-2]["content"] and "[1] Fixture source" in sent[-2]["content"]
                  and "blocked.test" not in sent[-2]["content"])
            check("max_tokens is W minus the prompt minus the margin",
                  answer_request.get("max_tokens") ==
                  config.CONTEXT_WINDOW - budget.prompt_tokens(sent) - config.MAX_TOKENS_MARGIN)
            branch = db.get_branch(chat_id)
            done = next(e for e in events if e["type"] == "done")
            check("tokens come from usage", branch[-1]["tokens"] == 17)
            check("done carries saved message", done.get("message_id") == branch[-1]["id"])
            check("thinking field and deprecated fallback are both read",
                  branch[-1]["thinking"] == "thinking fallback")
            saved = _message(chat_id, branch[-1])
            check("statistics and search sources persist", saved["sources"] == [
                {"title": "Fixture source", "url": "https://www.example.test/story",
                 "site": "example.test"}] and saved["stats"]["cached_tokens"] == 11)
            check("stats gain the agent fields", saved["stats"]["plan_fallback"] is False and
                  isinstance(saved["stats"]["plan_ms"], float) and
                  saved["stats"]["summary_used"] is False and
                  saved["stats"]["budget_max_tokens"] == answer_request["max_tokens"] and
                  isinstance(saved["stats"]["history_tokens"], int) and
                  isinstance(saved["stats"]["search_ms"], float) and saved["stats"]["searched"])
            check("planned queries persist with the message", saved["stats"]["queries"] ==
                  next(e for e in events if e["type"] == "plan")["queries"] and saved["stats"]["queries"])

            # Section 3's table, one row at a time. Planner reply: search, one query, no think.
            fake["plan_reply"] = {"search": True, "queries": ["rewritten query"], "think": False}
            table = [
                ("auto", "auto", True, True, ["rewritten query"], 0, 0.7),
                ("auto", "full", True, True, ["rewritten query"], None, 0.6),
                ("on", "auto", True, True, ["rewritten query"], 0, 0.7),
                ("on", "brief", True, True, ["rewritten query"], 128, 0.6),
                ("off", "auto", True, False, [], 0, 0.7),
                ("off", "full", False, False, [], None, 0.6),
                ("off", "off", False, False, [], 0, 0.7),
            ]
            for search_mode, level, planned, searched, queries, level_budget, temp in table:
                row_chat = db.create_chat(f"{search_mode}-{level}", level)
                n_plans = len(plans())
                n_searches = len(searched_queries)
                row_events = await collect(await chats_send(row_chat, SendBody(
                    content="row question", thinking=level, search=search_mode)))
                request = answers()[-1]
                name = f"branch search={search_mode} thinking={level}"
                check(f"{name}: planner {'runs' if planned else 'skipped'}",
                      (len(plans()) - n_plans == 1) == planned and
                      ("plan" in types(row_events)) == planned and
                      any(e.get("stage") == "planning" for e in row_events) == planned)
                check(f"{name}: search {'runs' if searched else 'skipped'}",
                      searched_queries[n_searches:] == queries and
                      ("sources" in types(row_events)) == searched)
                check(f"{name}: thinking budget {level_budget}",
                      request.get("gw_thinking_budget", "omitted") ==
                      ("omitted" if level_budget is None else level_budget) and
                      request.get("temperature") == temp)
                check(f"{name}: completes", types(row_events)[-2:] == ["stats", "done"])

            fake["plan_reply"] = {"search": True, "queries": ["a", " ", "b", "c", "d"], "think": True}
            capped = await collect(await chats_send(db.create_chat("cap", "auto"), SendBody(
                content="many parts", thinking="auto", search="auto")))
            check("planner queries are trimmed to three and blanks dropped",
                  next(e for e in capped if e["type"] == "plan")["queries"] == ["a", "b", "c"]
                  and "gw_thinking_budget" not in answers()[-1])

            fake["plan_reply"] = {"search": True, "queries": ["nothing here"], "think": False}
            empty = await collect(await chats_send(db.create_chat("empty", "auto"), SendBody(
                content="find nothing", thinking="auto", search="auto")))
            check("empty search sends empty sources and the could-not-verify line",
                  next(e for e in empty if e["type"] == "sources")["sources"] == [] and
                  answers()[-1]["messages"][-2]["content"] == prompts.empty_search(["nothing here"]))

            for mode in ("invalid", "error"):
                fake["plan"] = mode
                fb_chat = db.create_chat(f"fallback-{mode}", "auto")
                fb = await collect(await chats_send(fb_chat, SendBody(
                    content="fallback question", thinking="auto", search="auto")))
                fb_plan = next(e for e in fb if e["type"] == "plan")
                check(f"planner {mode}: falls back to search on the user text, think on",
                      fb_plan == {"type": "plan", "search": True, "queries": ["fallback question"],
                                  "think": True, "fallback": True} and
                      "gw_thinking_budget" not in answers()[-1] and
                      searched_queries[-1] == "fallback question")
                check(f"planner {mode}: fallback recorded in stats",
                      _message(fb_chat, db.get_branch(fb_chat)[-1])["stats"]["plan_fallback"] is True)
            fake["plan"] = "invalid"
            off_fb = await collect(await chats_send(db.create_chat("fb-off", "auto"), SendBody(
                content="fallback question", thinking="auto", search="off")))
            check("planner fallback with search off keeps search off, thinks",
                  next(e for e in off_fb if e["type"] == "plan") == {
                      "type": "plan", "search": False, "queries": [], "think": True,
                      "fallback": True} and "sources" not in types(off_fb))
            fake["plan"] = "hold"
            config.PLAN_TIMEOUT_S = 0.3
            timed = await collect(await chats_send(db.create_chat("timeout", "auto"), SendBody(
                content="slow planner", thinking="auto", search="auto")))
            check("planner timeout falls back and still answers",
                  next(e for e in timed if e["type"] == "plan")["fallback"] is True and
                  types(timed)[-2:] == ["stats", "done"])
            await asyncio.wait_for(plan_closed.wait(), 3.0)
            plan_closed.clear()
            plan_arrived.clear()
            config.PLAN_TIMEOUT_S = 30.0

            stop_plan_chat = db.create_chat("stop-plan", "auto")
            seen = await cancel_after(await chats_send(stop_plan_chat, SendBody(
                content="stop while planning", thinking="auto", search="auto")), "planning",
                plan_arrived)
            await asyncio.wait_for(plan_closed.wait(), 3.0)
            check("stop mid-plan closes the planner's upstream request", plan_closed.is_set())
            check("stop mid-plan never reaches the answer", "plan" not in types(seen) and
                  answers()[-1]["messages"][-1]["content"] != "stop while planning")
            check("stop mid-plan leaves no half turn (stream never opened)",
                  db.get_branch(stop_plan_chat) == [] and
                  db.get_chat(stop_plan_chat)["head_message_id"] is None)
            config.PLAN_TIMEOUT_S = old[4]

            fake["plan"] = "json"
            fake["plan_reply"] = {"search": True, "queries": ["hang please"], "think": False}
            stop_search_chat = db.create_chat("stop-search", "auto")
            await cancel_after(await chats_send(stop_search_chat, SendBody(
                content="stop while searching", thinking="auto", search="auto")), "searching",
                search_started)
            await asyncio.wait_for(search_cancelled.wait(), 3.0)
            check("stop mid-search cancels the running search", search_cancelled.is_set())
            check("stop mid-search leaves no half turn", db.get_branch(stop_search_chat) == [])

            title_chat = db.create_chat("New chat", "brief")
            title_events = await collect(await chats_send(title_chat, SendBody(
                content="name this conversation", thinking="off", search=False)))
            title_request = received[-1]
            check("automatic title follows done", types(title_events)[-2:] == ["done", "title"])
            check("automatic title is saved", db.get_chat(title_chat)["title"] == "A useful title")
            check("title request is tagged and excludes trace join fields",
                  "gw_chat_id" not in title_request and "gw_turn_index" not in title_request and
                  title_request.get("gw_purpose") == "title" and "gw_search" not in title_request
                  and title_request.get("gw_thinking_budget") == 0 and
                  title_request.get("max_tokens") <= 32)

            fake["title_fail"] = True
            fallback_chat = db.create_chat("New chat", "brief")
            fallback_events = await collect(await chats_send(fallback_chat, SendBody(
                content="Fallback title uses the first few words", thinking="off", search=False)))
            expected_fallback = "Fallback title uses the first few words"
            check("automatic title failure emits fallback", fallback_events[-1].get("title") ==
                  expected_fallback and db.get_chat(fallback_chat)["title"] == expected_fallback)
            fake["title_fail"] = False

            tree_chat = db.create_chat("branch tree", "brief")
            initial = await collect(await chats_send(tree_chat, SendBody(
                content="original", thinking="off", search=False)))
            original_user = initial[0]["user_message_id"]
            original_assistant = next(e["message_id"] for e in initial if e["type"] == "done")
            edited = await collect(await chats_send(tree_chat, SendBody(
                content="edited", thinking="full", search=False, parent_id=None)))
            check("full thinking leaves budget absent", "gw_thinking_budget" not in answers()[-1]
                  and answers()[-1].get("temperature") == 0.6)
            edited_user = edited[0]["user_message_id"]
            edited_assistant = next(e["message_id"] for e in edited if e["type"] == "done")
            check("edited user message creates a sibling branch",
                  db.sibling_ids(tree_chat, original_user) == [original_user, edited_user] and
                  [row["content"] for row in db.get_path(tree_chat, original_assistant)] ==
                  ["original", "answer"])
            regenerated = await collect(await chats_regenerate(tree_chat, RegenerateBody(
                message_id=edited_assistant, thinking="off", search=False)))
            regenerated_id = next(e["message_id"] for e in regenerated if e["type"] == "done")
            check("regenerate creates an assistant sibling",
                  db.sibling_ids(tree_chat, edited_assistant) == [edited_assistant, regenerated_id])
            check("regenerate sends the path without the old answer",
                  answers()[-1]["messages"][-1] == {"role": "user", "content": "edited"})
            await chats_head(tree_chat, HeadBody(message_id=edited_assistant))
            check("head selects the requested sibling path",
                  db.get_chat(tree_chat)["head_message_id"] == edited_assistant)
            await chats_head(tree_chat, HeadBody(message_id=original_user))
            check("head switch restores the original branch",
                  [row["id"] for row in db.get_branch(tree_chat)] == [original_user, original_assistant])

            stop_chat = db.create_chat("stop", "off")
            await cancel_after(await chats_send(stop_chat, SendBody(
                content="hold open", thinking="off", search=False)), "reasoning")
            await asyncio.wait_for(upstream_closed.wait(), 3.0)
            stopped_branch = db.get_branch(stop_chat)
            check("stop mid-answer closes the upstream stream", upstream_closed.is_set())
            check("cancel persists an assistant row", len(stopped_branch) == 2 and
                  stopped_branch[-1]["role"] == "assistant")
            check("cancel marks the partial row stopped", bool(stopped_branch[-1]["stopped"]))
            check("cancel keeps partial reasoning",
                  (stopped_branch[-1]["thinking"] or "").startswith("thinking"))
            check("cancel moves head to the partial row", db.get_chat(stop_chat)["head_message_id"] ==
                  stopped_branch[-1]["id"])
            await collect(await chats_send(stop_chat, SendBody(
                content="next turn", thinking="off", search=False)))
            upstream_closed.clear()
            plain_chat = db.create_chat("stop plain", "off")
            await cancel_after(await chats_send(plain_chat, SendBody(
                content="hold open", thinking="off", search=False)), "reasoning", plain=True)
            await asyncio.wait_for(upstream_closed.wait(), 3.0)
            check("a plain task cancel also closes upstream and saves a stopped row",
                  upstream_closed.is_set() and bool(db.get_branch(plain_chat)[-1]["stopped"]))
            roles = [row["role"] for row in db.get_branch(stop_chat)]
            check("next send continues without repeated user turns",
                  roles[-2:] == ["user", "assistant"] and
                  all(roles[i:i + 2] != ["user", "user"] for i in range(len(roles) - 1)))

            # The summary boundary, with a small window and the chars/3.5 counter forced.
            budget._TOKENIZER[:] = [None]
            budget.count_tokens.cache_clear()
            config.CONTEXT_WINDOW = 12000
            long_chat = db.create_chat("long", "off")
            first_summary_request = len(received)
            turn_prompts: list[list[dict]] = []
            turn_stats: list[dict] = []
            for turn in range(1, 41):
                await collect(await chats_send(long_chat, SendBody(
                    content=f"turn {turn} " + "words " * 60, thinking="off", search=False)))
                await asyncio.gather(*list(agent.PENDING.values()), return_exceptions=True)
                turn_prompts.append(answers()[-1]["messages"])
                turn_stats.append(_message(long_chat, db.get_branch(long_chat)[-1])["stats"])
            summary_requests = [r for r in received[first_summary_request:]
                                if r.get("gw_purpose") == "summary"]
            summaries = [next((m["content"] for m in p if m["content"].startswith(
                "Summary of the earlier conversation")), None) for p in turn_prompts]
            used = [s for s in summaries if s is not None]
            check("summaries were written and then used", len(summary_requests) >= 2 and
                  len(used) >= 10 and all(st["summary_used"] == (s is not None)
                                          for st, s in zip(turn_stats, summaries)))
            check("summary calls are tagged, thinking-off, capped, carry no join ids",
                  all(r.get("gw_purpose") == "summary" and "gw_chat_id" not in r and
                      "gw_turn_index" not in r and r.get("gw_thinking_budget") == 0 and
                      r.get("max_tokens") == 320 and r.get("stream") is False and
                      r["messages"][-1]["content"] == prompts.SUMMARY_INSTRUCTION
                      for r in summary_requests))
            check("the second summary builds on the first",
                  len(summary_requests) >= 2 and any(m["content"].endswith("SUMMARY-1")
                                                     for m in summary_requests[1]["messages"]))
            same_block = [(a, b) for a, b, sa, sb in zip(turn_prompts, turn_prompts[1:],
                                                         summaries, summaries[1:])
                          if sa is not None and sa == sb]
            check("within a block the previous prompt is a byte-identical prefix of the next",
                  len(same_block) >= 8 and all(json.dumps(b[:len(a) - 1]) == json.dumps(a[:-1])
                                               for a, b in same_block))
            check("the summary text changes only at block jumps",
                  0 < len(set(used)) <= len(used) // 5)
            check("every answer prompt plus max_tokens fits the window",
                  all(budget.prompt_tokens(p) + st["budget_max_tokens"] <= config.CONTEXT_WINDOW
                      for p, st in zip(turn_prompts, turn_stats)))

            fake["summary"] = "fail"
            fail_chat = db.create_chat("summary fails", "off")
            dropped_turn = None
            for turn in range(1, 41):
                await collect(await chats_send(fail_chat, SendBody(
                    content=f"turn {turn} " + "words " * 60, thinking="off", search=False)))
                await asyncio.gather(*list(agent.PENDING.values()), return_exceptions=True)
                prompt = answers()[-1]["messages"]
                if prompt[1]["content"] != f"turn 1 " + "words " * 60:
                    dropped_turn = prompt
                    break
            check("a failed summary drops the block with no summary for that turn",
                  dropped_turn is not None and not any(
                      m["content"].startswith("Summary of the earlier") for m in dropped_turn) and
                  _message(fail_chat, db.get_branch(fail_chat)[-1])["stats"]["summary_used"] is False)

            fake["summary"] = "slow"
            slow_chat = db.create_chat("summary slow", "off")
            awaited = False
            for turn in range(1, 41):
                pending_before = bool(agent.PENDING)
                await collect(await chats_send(slow_chat, SendBody(
                    content=f"turn {turn} " + "words " * 60, thinking="off", search=False)))
                if pending_before and _message(slow_chat, db.get_branch(slow_chat)[-1])[
                        "stats"]["summary_used"] is True:
                    awaited = True
                    break
            check("a summary still being written is awaited by the next turn", awaited)
            await asyncio.gather(*list(agent.PENDING.values()), return_exceptions=True)
            fake["summary"] = "ok"
            config.CONTEXT_WINDOW = old[5]
            budget._TOKENIZER[:] = old[6]
            budget.count_tokens.cache_clear()

            before_ids = [message["id"] for message in branch]
            previous_head_id = db.get_chat(chat_id)["head_message_id"]
            config.GATEWAY_URL = "http://127.0.0.1:1"
            failed_events = await collect(await chats_send(chat_id, SendBody(
                content="unreachable", thinking="off", search=False)))
            after = db.get_branch(chat_id)
            check("failed stream reports an error", failed_events[-1]["type"] == "error")
            check("failed stream leaves branch unchanged",
                  [message["id"] for message in after] == before_ids)
            check("failed stream restores the head",
                  db.get_chat(chat_id)["head_message_id"] == previous_head_id)
        finally:
            server.close()
            await server.wait_closed()
            db.DB_PATH, config.GATEWAY_URL = old[0], old[1]
            globals()["MODEL_ID"] = old[2]
            gsearch.run_search, config.PLAN_TIMEOUT_S, config.CONTEXT_WINDOW = old[3], old[4], old[5]
            budget._TOKENIZER[:] = old[6]
            budget.count_tokens.cache_clear()
    return fails


def selftest() -> int:
    """Graph branches, fallback, Stop at each step, summaries, budget, persistence."""
    fails = asyncio.run(_selftest_async())
    print("\n".join(fails) if fails else "selftest: PASS")
    if fails:
        print(f"selftest: {len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())

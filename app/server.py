"""Chat application API and static UI.

  uvicorn app.server:app --host 127.0.0.1 --port 8090
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

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
    title: str
    thinking_default: str


class RenameChatBody(BaseModel):
    title: str


class SendBody(BaseModel):
    content: str
    thinking: str
    search: bool


def _row(row):
    return dict(row) if row is not None else None


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
    if body.thinking_default not in config.THINKING_LEVELS:
        raise HTTPException(status_code=422, detail="invalid thinking_default")
    chat_id = db.create_chat(body.title, body.thinking_default)
    return _row(db.get_chat(chat_id))


@app.get("/api/chats/{chat_id}")
async def chats_get(chat_id: int):
    chat = _require_chat(chat_id)
    return {"chat": _row(chat), "messages": [_row(message) for message in db.get_branch(chat_id)]}


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


@app.post("/api/chats/{chat_id}/send")
async def chats_send(chat_id: int, payload: SendBody):
    chat = _require_chat(chat_id)
    if payload.thinking not in config.THINKING_LEVELS:
        raise HTTPException(status_code=422, detail="invalid thinking")

    previous_head_id = chat["head_message_id"]
    user_id = db.add_message(chat_id, previous_head_id, "user", payload.content, None, None)
    branch = db.get_branch(chat_id)
    messages = [{"role": message["role"], "content": message["content"]} for message in branch]
    turn_index = sum(message["role"] == "user" for message in branch)
    body = {
        "model": MODEL_ID,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "gw_search": payload.search,
        "gw_chat_id": chat_id,
        "gw_turn_index": turn_index,
    }
    budget = config.THINKING_LEVELS[payload.thinking]
    if budget is not None:
        body["gw_thinking_budget"] = budget
    LOGGER.warning("gateway request chat_id=%s turn_index=%s gw_thinking_budget=%s", chat_id,
                   turn_index, body.get("gw_thinking_budget", "omitted"))

    async def stream():
        content = ""
        reasoning = ""
        tokens: int | None = None
        message_id: int | None = None
        stream_open = False
        complete = False
        error: str | None = None
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream("POST", f"{config.GATEWAY_URL}/v1/chat/completions",
                                         json=body) as response:
                    response.raise_for_status()
                    stream_open = True
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:]
                        if raw == "[DONE]":
                            complete = True
                            break
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(chunk, dict):
                            continue
                        choices = chunk.get("choices") or []
                        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
                        delta = choice.get("delta") or {}
                        if not isinstance(delta, dict):
                            delta = {}
                        text = delta.get("content")
                        if isinstance(text, str) and text:
                            content += text
                            yield _event("content", text=text)
                        thought = delta.get("reasoning")
                        if isinstance(thought, str) and thought:
                            reasoning += thought
                            yield _event("reasoning", text=thought)
                        usage = chunk.get("usage") or {}
                        if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                            try:
                                tokens = int(usage.get("completion_tokens"))
                            except (TypeError, ValueError):
                                pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if stream_open:
                message_id = await asyncio.to_thread(db.add_message, chat_id, user_id, "assistant",
                                                     content, reasoning or None, tokens)
                LOGGER.info("gateway response chat_id=%s turn_index=%s app_rtt_ms=%.1f", chat_id,
                            turn_index, (time.perf_counter() - started) * 1e3)
            else:
                await asyncio.to_thread(db.set_head, chat_id, previous_head_id)
                await asyncio.to_thread(db.delete_message, user_id)

        if error is not None:
            yield _event("error", message=error)
        elif complete:
            yield _event("done", tokens=tokens, message_id=message_id)

    return EventSourceResponse(stream(), ping=None)


async def _selftest_async() -> list[str]:
    fails: list[str] = []
    received: dict = {}
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
            received.update(json.loads(await reader.readexactly(length)))
            chunks = [
                b'data: {"choices":[{"delta":{"content":"one"}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":" two"}}]}\n\n',
                b'data: {"choices":[],"usage":{"completion_tokens":17}}\n\n',
                b"data: [DONE]\n\n",
            ]
            payload = b"".join(chunks)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                         b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    with tempfile.TemporaryDirectory() as tmp:
        db.DB_PATH = str(Path(tmp) / "chats.db")
        db.init_db()
        chat_id = db.create_chat("test", "brief")
        server = await asyncio.start_server(gateway, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        config.GATEWAY_URL = f"http://127.0.0.1:{port}"
        try:
            response = await chats_send(chat_id, SendBody(content="hello", thinking="off", search=False))
            events = [event async for event in response.body_iterator]
            done = json.loads(events[-1]["data"])
            branch = db.get_branch(chat_id)
            check("send forwards the fixed sampling values",
                  received.get("temperature") == 0.6 and received.get("top_p") == 0.95 and
                  received.get("top_k") == 20 and received.get("min_p") == 0.0)
            check("off sends zero budget", received.get("gw_thinking_budget") == 0)
            check("turn index starts at one", received.get("gw_turn_index") == 1)
            check("tokens come from usage", branch[-1]["tokens"] == 17)
            check("done carries saved message", done.get("message_id") == branch[-1]["id"])

            before_ids = [message["id"] for message in branch]
            previous_head_id = db.get_chat(chat_id)["head_message_id"]
            config.GATEWAY_URL = "http://127.0.0.1:1"
            failed = await chats_send(chat_id, SendBody(content="unreachable", thinking="off", search=False))
            failed_events = [event async for event in failed.body_iterator]
            after = db.get_branch(chat_id)
            check("failed stream reports an error", json.loads(failed_events[-1]["data"])["type"] == "error")
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

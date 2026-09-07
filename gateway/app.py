"""
app.py -- the gateway that REWRITES a request before it reaches vLLM: injects web
search results and trims context, then streams the response back byte-for-byte.
Reuses labbench/proxy.py's trace and streaming machinery; see code-notes.md.

  uvicorn gateway.app:app --port 8080
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from labbench.proxy import StreamAccounting, Trace, pct
from gateway.search import CHARS_PER_TOKEN, render_block, run_search

UPSTREAM = os.environ.get("GW_UPSTREAM", "http://localhost:8000")
TIMEOUT = float(os.environ.get("GW_TIMEOUT", "600"))
BUDGET_TOKENS = int(os.environ.get("GW_BUDGET_TOKENS", "12000"))
DEFAULT_CONTEXT_STRATEGY = os.environ.get("GW_CONTEXT", "none")
ALWAYS_SEARCH = os.environ.get("GW_ALWAYS_SEARCH") == "1"

# The gateway owns its own trace file. Reusing labbench.proxy's writer meant setting
# its module global, which silently redirected the lab bench's traces here too.
TRACE_PATH = os.environ.get("GW_TRACE", "results/gateway-traces.jsonl")


def write_trace(tr: GatewayTrace, path: str | None = None) -> None:
    """Flush per request. A run that writes at the end loses everything to a crash."""
    path = path or TRACE_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(tr)) + "\n")
        f.flush()


@dataclass
class GatewayTrace(Trace):
    """Extends the proxy Trace with the non-inference spans. Same jsonl file shape plus these."""
    search_ms: float | None = None
    fetch_ms: float | None = None
    extract_ms: float | None = None
    n_sources: int = 0
    trim_ms: float | None = None
    trim_strategy: str = "none"
    dropped_turns: int = 0
    injected_tokens_est: int = 0
    upstream_ms: float | None = None


@dataclass
class _NoTrim:
    """Fallback TrimResult when gateway/context.py is not importable -- no trimming."""
    messages: list[dict]
    strategy: str = "none"
    dropped_turns: int = 0
    summary_text: str | None = None
    trim_ms: float = 0.0
    est_tokens_before: int = 0
    est_tokens_after: int = 0


def assemble_messages(messages: list[dict], retrieved_block: str) -> list[dict]:
    """[history][retrieved][last user]. Retrieved sits immediately before the last
    message so vLLM's exact-prefix cache only re-prefills the volatile tail (13.8x)."""
    if not retrieved_block or not messages:
        return list(messages)
    *history, last = messages
    return [*history, {"role": "system", "content": retrieved_block}, last]


def _last_user_text(messages: list[dict]) -> str:
    """Content of the most recent user turn; used as the search query."""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


async def _trim(messages: list[dict], budget_tokens: int, strategy: str):
    """Delegate to gateway.context if present; otherwise a no-op passthrough."""
    try:
        from gateway import context
    except ImportError:
        return _NoTrim(messages=messages)
    return await context.apply(messages, budget_tokens, strategy=strategy)


async def _do_search(messages: list[dict], tr: GatewayTrace) -> str:
    """Run search and render its block. A raising search still answers the turn."""
    query = _last_user_text(messages)
    try:
        outcome = await run_search(query, CLIENT)
    except Exception:
        tr.n_sources = 0
        return ""
    tr.search_ms, tr.fetch_ms, tr.extract_ms = outcome.search_ms, outcome.fetch_ms, outcome.extract_ms
    tr.n_sources = outcome.n_sources
    return render_block(outcome)


CLIENT: httpx.AsyncClient | None = None
RECENT: list[GatewayTrace] = []
INFLIGHT: dict[str, StreamAccounting] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """One shared client for the process; see labbench/proxy.py for the +11 ms measurement."""
    global CLIENT
    CLIENT = httpx.AsyncClient(timeout=TIMEOUT,
                               limits=httpx.Limits(max_connections=256,
                                                   max_keepalive_connections=256))
    try:
        yield
    finally:
        await CLIENT.aclose()


app = FastAPI(lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat(req: Request):
    raw = await req.body()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return Response(content=json.dumps({"error": "invalid json body"}),
                        status_code=400, media_type="application/json")

    tr = GatewayTrace(request_id=uuid.uuid4().hex[:12], t_wall=time.time(), upstream=UPSTREAM)
    messages = body.get("messages") or []
    do_search = bool(body.get("gw_search")) or ALWAYS_SEARCH
    strategy = body.get("gw_context", DEFAULT_CONTEXT_STRATEGY)
    for k in [k for k in body if k.startswith("gw_")]:
        body.pop(k, None)

    retrieved_block = await _do_search(messages, tr) if do_search and messages else ""

    trimmed = await _trim(messages, BUDGET_TOKENS, strategy)
    tr.trim_ms, tr.trim_strategy, tr.dropped_turns = (trimmed.trim_ms, trimmed.strategy,
                                                      trimmed.dropped_turns)

    final_messages = assemble_messages(trimmed.messages, retrieved_block)
    tr.injected_tokens_est = int(len(retrieved_block) / CHARS_PER_TOKEN) if retrieved_block else 0
    body["messages"] = final_messages
    new_body = json.dumps(body).encode()

    if not body.get("stream"):
        return await _passthrough_json(new_body, tr)
    return StreamingResponse(_relay(new_body, tr), media_type="text/event-stream")


async def _passthrough_json(body: bytes, tr: GatewayTrace) -> Response:
    t0 = time.perf_counter()
    r = await CLIENT.post(f"{UPSTREAM}/v1/chat/completions", content=body,
                          headers={"content-type": "application/json"})
    tr.http_status = r.status_code
    tr.e2e_ms = tr.upstream_ms = (time.perf_counter() - t0) * 1e3
    tr.status = "ok" if r.status_code == 200 else "http_error"
    _record(tr)
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


async def _relay(body: bytes, tr: GatewayTrace):
    """Yield upstream bytes unchanged; account for them on the side."""
    t0 = time.perf_counter()
    acc = StreamAccounting(tr, t0)
    INFLIGHT[tr.request_id] = acc
    try:
        async with CLIENT.stream("POST", f"{UPSTREAM}/v1/chat/completions", content=body,
                                 headers={"content-type": "application/json"}) as r:
            tr.http_status = r.status_code
            if r.status_code != 200:
                tr.status, tr.error = "http_error", str(r.status_code)
                yield await r.aread()
                return
            async for blob in r.aiter_bytes():
                yield blob
                acc.feed(blob, time.perf_counter())
    except Exception as e:
        tr.status, tr.error = "exception", f"{type(e).__name__}: {e}"
    finally:
        INFLIGHT.pop(tr.request_id, None)
        finished = acc.finish(time.perf_counter())
        finished.upstream_ms = finished.ttft_ms
        _record(finished)


def _record(tr: GatewayTrace) -> None:
    RECENT.append(tr)
    del RECENT[:-200]
    write_trace(tr)


@app.get("/v1/models")
async def models():
    """Pass through unchanged, so a client discovers the served model via the gateway."""
    try:
        r = await CLIENT.get(f"{UPSTREAM}/v1/models")
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))
    except Exception as e:
        return Response(content=json.dumps({"error": f"{type(e).__name__}: {e}"}),
                        status_code=502, media_type="application/json")


@app.get("/health")
async def health():
    return {"ok": True, "upstream": UPSTREAM}


@app.get("/gateway/traces")
async def traces(n: int = 20):
    """Finished traces, then anything still streaming. Newest last either way."""
    now = time.perf_counter()
    live = [a.partial(now) for a in list(INFLIGHT.values())]
    return {"upstream": UPSTREAM, "traces": [asdict(t) for t in RECENT[-n:]] + live}


class _FakeResp:
    """Stand-in for an httpx.Response, used only by selftest."""

    def __init__(self, content: bytes, status_code: int = 200):
        self.content, self.status_code = content, status_code
        self.headers: dict[str, str] = {"content-type": "application/json"}


class _FakeStreamCtx:
    """Stand-in for the async context manager httpx.AsyncClient.stream() returns."""

    def __init__(self, status_code: int, chunks: list[bytes]):
        self.status_code, self.headers = status_code, {}
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c

    async def aread(self) -> bytes:
        return b"".join(self._chunks)


class _FakeUpstream:
    """Records the forwarded body and replays a canned reply. Touches no network."""

    def __init__(self, json_reply: dict | None = None, stream_chunks: list[bytes] | None = None):
        self.last_body: bytes = b""
        self._json_reply = json_reply or {"id": "x", "choices": []}
        self._stream_chunks = stream_chunks or []

    async def post(self, url, content=b"", headers=None):
        self.last_body = content
        return _FakeResp(json.dumps(self._json_reply).encode())

    def stream(self, method, url, content=b"", headers=None):
        self.last_body = content
        return _FakeStreamCtx(200, self._stream_chunks)

    async def get(self, url):
        return _FakeResp(json.dumps({"data": []}).encode())

    async def aclose(self) -> None:
        pass


def selftest() -> int:
    """No network, no vLLM, no Brave: monkeypatches CLIENT and run_search."""
    global CLIENT, run_search
    from fastapi.testclient import TestClient

    fails: list[str] = []

    def chk(name: str, cond: bool) -> None:
        if not cond:
            fails.append(f"  FAIL {name}")

    convo = [{"role": "system", "content": "sys"},
             {"role": "user", "content": "q1"},
             {"role": "assistant", "content": "a1"},
             {"role": "user", "content": "q2"}]
    out = assemble_messages(convo, "DOCS")
    chk("assemble: length grows by one", len(out) == len(convo) + 1)
    chk("assemble: retrieved block immediately before last", out[3] == {"role": "system", "content": "DOCS"})
    chk("assemble: last message preserved and last", out[4] == convo[-1])
    chk("assemble: history untouched ahead of it", out[:3] == convo[:3])
    chk("assemble: no-op with empty block", assemble_messages(convo, "") == convo)
    chk("assemble: no-op with empty messages", assemble_messages([], "DOCS") == [])

    global TRACE_PATH
    with tempfile.TemporaryDirectory() as tmp:
        TRACE_PATH = os.path.join(tmp, "gateway-traces.jsonl")
        with TestClient(app) as client:
            CLIENT = _FakeUpstream()

            r = client.get("/health")
            chk("health status 200", r.status_code == 200)
            chk("health body", r.json().get("ok") is True)

            body = {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                   "gw_search": False, "gw_context": "none", "gw_bogus": 1}
            r = client.post("/v1/chat/completions", json=body)
            chk("non-streaming 200", r.status_code == 200)
            fwd = json.loads(CLIENT.last_body)
            chk("gw_* stripped from forwarded body", all(not k.startswith("gw_") for k in fwd))
            chk("no search leaves messages unchanged", fwd["messages"] == body["messages"])

            _orig_run_search = run_search

            async def _raise(*a, **kw):
                raise RuntimeError("brave is down")
            run_search = _raise
            body2 = {"model": "m", "messages": [{"role": "user", "content": "who won"}],
                    "gw_search": True}
            r = client.post("/v1/chat/completions", json=body2)
            chk("search failure still answers", r.status_code == 200)
            t = RECENT[-1]
            chk("search failure gives n_sources 0", t.n_sources == 0)
            run_search = _orig_run_search

            chunks = [b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
                     b'data: {"choices":[{"delta":{"content":" there"}}],'
                     b'"usage":{"completion_tokens":2,"prompt_tokens":5}}\n\n',
                     b"data: [DONE]\n\n"]
            CLIENT = _FakeUpstream(stream_chunks=chunks)
            body3 = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
            with client.stream("POST", "/v1/chat/completions", json=body3) as r:
                got = b"".join(r.iter_bytes())
            chk("streaming byte-identical to upstream", got == b"".join(chunks))

            r = client.get("/gateway/traces?n=5")
            rows = r.json()["traces"]
            chk("trace rows recorded", len(rows) >= 2)
            last = rows[-1]
            chk("trace has span fields", all(f in last for f in
                ("search_ms", "fetch_ms", "extract_ms", "trim_ms", "upstream_ms", "n_sources")))

    print("\n".join(fails) if fails else "selftest: PASS")
    if fails:
        print(f"selftest: {len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print("usage: uvicorn gateway.app:app --port 8080  |  python -m gateway.app --selftest")

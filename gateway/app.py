"""
app.py -- the gateway that REWRITES a request before it reaches vLLM: injects web
search results and trims context, then streams the response back byte-for-byte.
Reuses labbench/proxy.py's trace and streaming machinery; see code-notes.md.

  uvicorn gateway.app:app --port 8080
"""
from __future__ import annotations

import asyncio
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
from gateway import splice
from gateway.load import LoadSensor
from gateway.search import CHARS_PER_TOKEN, render_block, run_search

UPSTREAM = os.environ.get("GW_UPSTREAM", "http://localhost:8000")
TIMEOUT = float(os.environ.get("GW_TIMEOUT", "600"))
BUDGET_TOKENS = int(os.environ.get("GW_BUDGET_TOKENS", "12000"))
# Q1b. Two SAFE budgets and a hysteresis band, never a value between them: P7-Q1m/n measured
# 512-1024 as a trough 11 points below both neighbours. See NOTES/predictions.md.
ADAPT_BIG = int(os.environ.get("GW_BUDGET_BIG", "2048"))
ADAPT_SMALL = int(os.environ.get("GW_BUDGET_SMALL", "128"))
Q_HIGH = float(os.environ.get("GW_Q_HIGH", "8"))
Q_LOW = float(os.environ.get("GW_Q_LOW", "2"))
ADAPT_STATE = {"small": False}
# The policy is a property of the SERVING layer, not the client, so Q1b's three arms are
# three gateway configurations driving one unchanged client.
BUDGET_POLICY = os.environ.get("GW_BUDGET_POLICY", "off")
BUDGET_DEFAULT = os.environ.get("GW_BUDGET_DEFAULT", "")
DEFAULT_CONTEXT_STRATEGY = os.environ.get("GW_CONTEXT", "none")
# Orchestration order is a Phase 7 parameter. "overlap" generates while the search is in
# flight and splices results in mid-stream; generate_then_retrieve is still a stub.
ORDER_DEFAULT = os.environ.get("GW_ORDER", "retrieve_then_generate")
ORDERS = ("retrieve_then_generate", "generate_then_retrieve", "overlap")
BUILDABLE_ORDERS = ("retrieve_then_generate", "overlap", "generate_then_retrieve")
METRICS_URL_ENV = os.environ.get("GW_METRICS", "http://localhost:8000")
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
    chat_id: int | None = None
    turn_index: int | None = None
    search_ms: float | None = None
    fetch_ms: float | None = None
    extract_ms: float | None = None
    n_sources: int = 0
    search_error: str | None = None
    trim_ms: float | None = None
    trim_strategy: str = "none"
    dropped_turns: int = 0
    injected_tokens_est: int = 0
    upstream_ms: float | None = None
    load_running: float | None = None
    load_waiting: float | None = None
    load_kv_usage: float | None = None
    load_stale: bool = False
    priority: int | None = None
    order: str = ORDER_DEFAULT
    order_honoured: bool = True
    thinking_budget: int | None = None
    budget_policy: str | None = None
    overlap_hidden_ms: float | None = None
    overlap_pre_tokens: int = 0
    splice_chars: int = 0
    reissue_prompt_tokens: int | None = None
    reissue_cached_tokens: int | None = None


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


def adaptive_budget(waiting: float | None) -> int:
    """Pick one of two safe budgets from queue depth. Hysteresis stops a queue oscillating
    around one threshold from flapping the budget across the trough every request."""
    if waiting is None:
        return ADAPT_BIG
    if ADAPT_STATE["small"]:
        if waiting < Q_LOW:
            ADAPT_STATE["small"] = False
    elif waiting > Q_HIGH:
        ADAPT_STATE["small"] = True
    return ADAPT_SMALL if ADAPT_STATE["small"] else ADAPT_BIG


def apply_thinking_budget(body: dict, budget: int | None) -> dict:
    """The single point where a thinking budget lands on the outgoing request.
    Phase 7 decides the number; 6a only applies whatever it is given."""
    if budget is None:
        return body
    kw = dict(body.get("chat_template_kwargs") or {})
    kw["enable_thinking"] = budget > 0
    body["chat_template_kwargs"] = kw
    # 6a set only enable_thinking, which is an on/off switch. The actual actuator is this
    # field, and without it every budget above 0 was a silent no-op. See code-notes.
    body["thinking_token_budget"] = budget
    return body


async def _trim(messages: list[dict], budget_tokens: int, strategy: str):
    """Delegate to gateway.context if present; otherwise a no-op passthrough."""
    try:
        from gateway import context
    except ImportError:
        return _NoTrim(messages=messages)
    return await context.apply(messages, budget_tokens, strategy=strategy)


async def _do_search(messages: list[dict], tr: GatewayTrace,
                     query: str | None = None, on_sources=None) -> str:
    """Run search and render its block. A raising search still answers the turn."""
    query = query or _last_user_text(messages)
    try:
        outcome = await run_search(query, CLIENT)
    except Exception as e:
        tr.n_sources, tr.search_error = 0, f"{type(e).__name__}"
        if on_sources is not None:
            on_sources([])
        return ""
    tr.search_ms, tr.fetch_ms, tr.extract_ms = outcome.search_ms, outcome.fetch_ms, outcome.extract_ms
    tr.n_sources = outcome.n_sources
    # Brave degrades to zero sources on a 429 as quietly as on a genuine miss, and the
    # free plan allows 1 query/second. Without this the two are indistinguishable.
    tr.search_error = outcome.error
    if on_sources is not None:
        on_sources([{"title": source.title, "url": source.url}
                    for source in outcome.sources if source.ok])
    return render_block(outcome)


def _render_prompt(messages: list[dict], enable_thinking: bool = True) -> str:
    """Render locally. The overlap path needs the exact bytes, not a messages array."""
    from gateway import prompt
    return prompt.render(messages, enable_thinking=enable_thinking,
                         add_generation_prompt=True, patch=True)


RENDER = _render_prompt


def _sse(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def _gw_event(event_type: str, **fields: object) -> bytes:
    return _sse({"object": "gw.event", "type": event_type, **fields})


def _gw_stats(tr: GatewayTrace) -> dict:
    spans = (tr.search_ms, tr.fetch_ms, tr.extract_ms)
    search_ms = sum(spans) if all(value is not None for value in spans) else None
    decode_tok_s = 1000.0 / tr.itl_ms_derived if tr.itl_ms_derived else None
    return {"engine_ttft_ms": tr.upstream_ms, "search_ms": search_ms,
            "prompt_tokens": tr.prompt_tokens, "cached_tokens": tr.cached_tokens,
            "completion_tokens": tr.completion_tokens, "decode_tok_s": decode_tok_s}


async def _relay_with_events(body: dict, tr: GatewayTrace, messages: list[dict],
                             do_search: bool, query: str | None, budget,
                             strategy: str):
    """Add opt-in progress frames around the default retrieve-then-generate relay."""
    retrieved_block = ""
    if do_search and messages:
        resolved_query = query or _last_user_text(messages)
        yield _gw_event("search.started", query=resolved_query)
        sources: list[dict[str, str]] = []
        retrieved_block = await _do_search(messages, tr, resolved_query,
                                            on_sources=lambda value: sources.extend(value))
        yield _gw_event("sources", sources=sources)

    trimmed = await _trim(messages, BUDGET_TOKENS, strategy)
    tr.trim_ms, tr.trim_strategy, tr.dropped_turns = (
        trimmed.trim_ms, trimmed.strategy, trimmed.dropped_turns)
    final_body = {key: value for key, value in body.items() if not key.startswith("gw_")}
    final_body["messages"] = assemble_messages(trimmed.messages, retrieved_block)
    tr.injected_tokens_est = int(len(retrieved_block) / CHARS_PER_TOKEN) if retrieved_block else 0
    apply_thinking_budget(final_body, budget)
    yield _gw_event("status", stage="generating")
    async for blob in _relay(json.dumps(final_body).encode(), tr):
        yield blob
    yield _gw_event("stats", stats=_gw_stats(tr))


def _chunk(cid: str, model: str, delta: dict, finish: str | None = None) -> bytes:
    """One chat-shaped SSE chunk. The overlap path speaks completions upstream and
    chat downstream, so every upstream token is retranslated on the way out."""
    return _sse({"id": cid, "object": "chat.completion.chunk", "model": model,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})


def _completion_body(base: dict, prompt_text: str) -> dict:
    """A /v1/completions body carrying over only what the engine accepts there."""
    keep = ("model", "max_tokens", "temperature", "top_p", "top_k", "seed", "stop",
            "priority", "thinking_token_budget", "repetition_penalty")
    body = {k: base[k] for k in keep if k in base}
    body["prompt"] = prompt_text
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    return body


def _usage_from(line: bytes) -> dict | None:
    """Pull a usage object out of one SSE line, if it carries one."""
    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
        return None
    try:
        return json.loads(line[6:]).get("usage")
    except (json.JSONDecodeError, AttributeError):
        return None


def _text_from(line: bytes) -> str:
    """The generated text in one completions SSE line. Empty when there is none."""
    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
        return ""
    try:
        ch = (json.loads(line[6:]).get("choices") or [{}])[0]
    except (json.JSONDecodeError, AttributeError):
        return ""
    return ch.get("text") or ""


async def _stream_completion(body: dict, on_text, stop_when=None) -> dict | None:
    """Stream /v1/completions, feeding text to on_text. Returns the final usage.
    stop_when() true ends the read early -- leaving the context closes the connection."""
    usage = None
    async with CLIENT.stream("POST", f"{UPSTREAM}/v1/completions",
                             content=json.dumps(body).encode(),
                             headers={"content-type": "application/json"}) as r:
        if r.status_code != 200:
            raise RuntimeError(f"upstream {r.status_code} on /v1/completions")
        buf = b""
        async for blob in r.aiter_bytes():
            buf += blob
            while b"\n\n" in buf:
                line, buf = buf.split(b"\n\n", 1)
                usage = _usage_from(line) or usage
                t = _text_from(line)
                if t:
                    on_text(t)
            if stop_when is not None and stop_when():
                return usage
    return usage


async def _relay_overlap(base: dict, tr: GatewayTrace, messages: list[dict],
                         search_task, enable_thinking: bool = True, defer_search=None):
    """Generate against the un-retrieved prompt while the search runs, then splice.
    A separate path from _relay, whose byte-fidelity the protocol gate validated."""
    t0 = time.perf_counter()
    cid, model = f"chatcmpl-{tr.request_id}", base.get("model", "")
    yield _chunk(cid, model, {"role": "assistant", "content": ""})

    generated: list[str] = []

    def collect(t: str) -> None:
        generated.append(t)

    try:
        prompt_sent = RENDER(messages, enable_thinking=enable_thinking)
        # Q3 overlaps: stop generating when the search lands. Q2 defers: generate first,
        # search after, which is the same splice on a different clock.
        usage = usage_pre = await _stream_completion(
            _completion_body(base, prompt_sent), collect,
            stop_when=search_task.done if search_task else None)
        pre = "".join(generated)
        tr.overlap_hidden_ms = (time.perf_counter() - t0) * 1e3
        tr.overlap_pre_tokens = len(pre) // int(CHARS_PER_TOKEN) if pre else 0
        if pre:
            yield _chunk(cid, model, {"content": pre})

        if search_task is None and defer_search is not None:
            search_task = asyncio.ensure_future(defer_search())
        block = await search_task if search_task is not None else ""
        if block:
            cont = splice.build(prompt_sent, pre, block)
            splice.verify(cont, prompt_sent, pre)
            tr.splice_chars = cont.spliced_chars
            tail: list[str] = []
            usage = await _stream_completion(_completion_body(base, cont.prompt), tail.append)
            if tail:
                yield _chunk(cid, model, {"content": "".join(tail)})
        if usage:
            tr.reissue_prompt_tokens = usage.get("prompt_tokens")
            det = usage.get("prompt_tokens_details") or {}
            tr.reissue_cached_tokens = det.get("cached_tokens")
        # Both phases generate, so the client's completion count is their SUM. Without this
        # the relay emits no usage at all and every token metric downstream reads nan.
        pre_ct = (usage_pre or {}).get("completion_tokens") or 0
        post_ct = 0 if usage is usage_pre else ((usage or {}).get("completion_tokens") or 0)
        total = pre_ct + post_ct
        yield _sse({"id": cid, "object": "chat.completion.chunk", "model": model,
                    "choices": [], "usage": {
                        "completion_tokens": total,
                        "prompt_tokens": (usage or {}).get("prompt_tokens"),
                        "total_tokens": total + ((usage or {}).get("prompt_tokens") or 0)}})
        tr.http_status, tr.status = 200, "ok"
    except Exception as e:
        tr.status, tr.error = "exception", f"{type(e).__name__}: {e}"
        yield _chunk(cid, model, {"content": ""}, finish="stop")
        yield b"data: [DONE]\n\n"
        tr.e2e_ms = (time.perf_counter() - t0) * 1e3
        _record(tr)
        return

    yield _chunk(cid, model, {}, finish="stop")
    yield b"data: [DONE]\n\n"
    tr.e2e_ms = tr.upstream_ms = (time.perf_counter() - t0) * 1e3
    _record(tr)


CLIENT: httpx.AsyncClient | None = None
RECENT: list[GatewayTrace] = []
SENSOR = LoadSensor(METRICS_URL_ENV)
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
    order = body.get("gw_order", ORDER_DEFAULT)
    priority = body.get("gw_priority")
    budget = body.get("gw_thinking_budget")
    squery = body.get("gw_query") or None
    gw_events = body.get("gw_events") is True
    chat_id = body.get("gw_chat_id")
    turn_index = body.get("gw_turn_index")
    for k in [k for k in body if k.startswith("gw_")]:
        body.pop(k, None)

    tr.order = order if order in ORDERS else ORDER_DEFAULT
    tr.order_honoured = tr.order in BUILDABLE_ORDERS
    tr.chat_id = chat_id
    tr.turn_index = turn_index
    if priority is not None:
        try:
            tr.priority = int(priority)
            body["priority"] = tr.priority
        except (TypeError, ValueError):
            tr.priority = None
    s = await SENSOR.sample(CLIENT)
    if budget == "adaptive" or (budget is None and BUDGET_POLICY == "adaptive"):
        budget = adaptive_budget(s.waiting)
        tr.budget_policy = "adaptive"
    elif budget is None and BUDGET_DEFAULT:
        budget = int(BUDGET_DEFAULT)
        tr.budget_policy = "fixed"
    tr.thinking_budget = budget
    tr.load_running, tr.load_waiting = s.running, s.waiting
    tr.load_kv_usage, tr.load_stale = s.kv_usage, s.stale

    trimmed_first = None
    if tr.order in ("overlap", "generate_then_retrieve") and do_search and messages \
            and body.get("stream"):
        # overlap starts the search now; generate_then_retrieve starts it only after the
        # model has had its turn. Same relay, same splice, different ordering.
        deferred = tr.order == "generate_then_retrieve"
        task = None if deferred else asyncio.ensure_future(_do_search(messages, tr, squery))
        trimmed_first = await _trim(messages, BUDGET_TOKENS, strategy)
        tr.trim_ms, tr.trim_strategy, tr.dropped_turns = (
            trimmed_first.trim_ms, trimmed_first.strategy, trimmed_first.dropped_turns)
        body.pop("stream_options", None)
        return StreamingResponse(
            _relay_overlap(body, tr, trimmed_first.messages, task,
                           enable_thinking=(budget is None or budget > 0),
                           defer_search=(lambda: _do_search(messages, tr, squery)) if deferred else None),
            media_type="text/event-stream")

    if gw_events and body.get("stream"):
        return StreamingResponse(
            _relay_with_events(body, tr, messages, do_search, squery, budget, strategy),
            media_type="text/event-stream")

    retrieved_block = await _do_search(messages, tr, squery) if do_search and messages else ""

    trimmed = await _trim(messages, BUDGET_TOKENS, strategy)
    tr.trim_ms, tr.trim_strategy, tr.dropped_turns = (trimmed.trim_ms, trimmed.strategy,
                                                      trimmed.dropped_turns)

    final_messages = assemble_messages(trimmed.messages, retrieved_block)
    tr.injected_tokens_est = int(len(retrieved_block) / CHARS_PER_TOKEN) if retrieved_block else 0
    body["messages"] = final_messages
    apply_thinking_budget(body, budget)
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


@app.get("/gateway/load")
async def load():
    """The live engine load a Phase 7 policy would key off. Reads, decides nothing."""
    s = await SENSOR.sample(CLIENT)
    return asdict(s)


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
        self.bodies: list[bytes] = []
        self._json_reply = json_reply or {"id": "x", "choices": []}
        self._stream_chunks = stream_chunks or []

    async def post(self, url, content=b"", headers=None):
        self.last_body = content
        self.bodies.append(content)
        return _FakeResp(json.dumps(self._json_reply).encode())

    def stream(self, method, url, content=b"", headers=None):
        self.last_body = content
        self.bodies.append(content)
        return _FakeStreamCtx(200, self._stream_chunks)

    async def get(self, url):
        return _FakeResp(json.dumps({"data": []}).encode())

    async def aclose(self) -> None:
        pass


def selftest() -> int:
    """No network, no vLLM, no Brave: monkeypatches CLIENT and run_search."""
    global CLIENT, run_search, render_block
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

    b = apply_thinking_budget({"messages": []}, None)
    chk("budget None leaves the body untouched", "chat_template_kwargs" not in b)
    b = apply_thinking_budget({"messages": []}, 0)
    chk("budget 0 disables thinking", b["chat_template_kwargs"]["enable_thinking"] is False)
    b = apply_thinking_budget({"chat_template_kwargs": {"keep": 1}}, 512)
    chk("budget >0 enables thinking", b["chat_template_kwargs"]["enable_thinking"] is True)
    chk("budget preserves other template kwargs", b["chat_template_kwargs"]["keep"] == 1)

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
            _t0 = len(RECENT)

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

            r = client.post("/v1/chat/completions", json={
                "model": "m", "messages": [{"role": "user", "content": "hi"}],
                "gw_priority": 3, "gw_order": "generate_then_retrieve",
                "gw_thinking_budget": 0})
            fwd = json.loads(CLIENT.last_body)
            t = RECENT[-1]
            chk("gw_priority reaches upstream as priority", fwd.get("priority") == 3)
            chk("gw_priority itself is stripped", "gw_priority" not in fwd)
            chk("priority recorded on the trace", t.priority == 3)
            chk("thinking budget applied to the forwarded body",
                fwd.get("chat_template_kwargs", {}).get("enable_thinking") is False)
            chk("thinking budget recorded", t.thinking_budget == 0)
            b = apply_thinking_budget({"messages": []}, 512)
            chk("the budget reaches upstream as thinking_token_budget",
                b.get("thinking_token_budget") == 512)
            chk("a positive budget leaves thinking enabled",
                b["chat_template_kwargs"]["enable_thinking"] is True)
            chk("order recorded", t.order == "generate_then_retrieve")
            chk("generate_then_retrieve is now a buildable order", t.order_honoured is True)
            # Q1b: the policy may only ever return one of the two SAFE budgets.
            ADAPT_STATE["small"] = False
            chk("idle queue picks the big budget", adaptive_budget(0.0) == ADAPT_BIG)
            chk("below the high-water mark holds big", adaptive_budget(Q_HIGH) == ADAPT_BIG)
            chk("above the high-water mark switches small",
                adaptive_budget(Q_HIGH + 1) == ADAPT_SMALL)
            chk("inside the band it HOLDS small, it does not drift",
                adaptive_budget(Q_LOW + 0.5) == ADAPT_SMALL)
            chk("below the low-water mark returns to big", adaptive_budget(0.0) == ADAPT_BIG)
            chk("a failed scrape is not read as an empty queue",
                adaptive_budget(None) == ADAPT_BIG)
            chk("the policy never returns a value in the measured trough",
                all(adaptive_budget(q) not in range(400, 1200) for q in (0, 5, 9, 20, 3, 1)))

            r = client.post("/v1/chat/completions", json={
                "model": "m", "messages": [{"role": "user", "content": "hi"}],
                "gw_priority": "not-an-int", "gw_order": "nonsense"})
            fwd = json.loads(CLIENT.last_body)
            t = RECENT[-1]
            chk("a bad priority is dropped, not forwarded", "priority" not in fwd)
            chk("a bad priority is not recorded", t.priority is None)
            chk("an unknown order falls back to the default", t.order == ORDER_DEFAULT)
            chk("the default order is honoured", t.order_honoured is True)
            chk("load recorded on the trace", t.load_stale in (True, False))

            r = client.get("/gateway/load")
            chk("load endpoint 200", r.status_code == 200)
            chk("load endpoint shape", set(("running", "waiting", "kv_usage", "stale"))
                <= set(r.json()))

            chunks = [b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
                     b'data: {"choices":[{"delta":{"content":" there"}}],'
                     b'"usage":{"completion_tokens":2,"prompt_tokens":5}}\n\n',
                     b"data: [DONE]\n\n"]
            CLIENT = _FakeUpstream(stream_chunks=chunks)
            body3 = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
            with client.stream("POST", "/v1/chat/completions", json=body3) as r:
                got = b"".join(r.iter_bytes())
            chk("streaming byte-identical to upstream", got == b"".join(chunks))

            # -- overlap path -----------------------------------------------------
            global RENDER
            _orig_render = RENDER
            RENDER = lambda msgs, enable_thinking=True: "PROMPT<|im_start|>assistant\n"
            comp = [b'data: {"choices":[{"text":"<think>\\nreason"}]}\n\n',
                    b'data: {"choices":[{"text":" more"}],'
                    b'"usage":{"prompt_tokens":9,"prompt_tokens_details":{"cached_tokens":8}}}\n\n',
                    b"data: [DONE]\n\n"]
            CLIENT = _FakeUpstream(stream_chunks=comp)

            # The REAL dataclass, not a stand-in: a hand-rolled fake silently lacks any
            # field added later, and _do_search reads them outside its try. Incident 41.
            from gateway.search import SearchOutcome

            seen_q = []

            async def _fixed_search(q, *a, **kw):
                seen_q.append(q)
                return SearchOutcome(query=q, search_ms=1.0, fetch_ms=1.0,
                                     extract_ms=1.0, n_sources=2)
            _orig_rs = run_search
            run_search = _fixed_search
            _orig_render_block = render_block
            render_block = lambda o: "SOURCES"

            with client.stream("POST", "/v1/chat/completions", json={
                    "model": "m", "messages": [{"role": "user", "content": "who won"}],
                    "gw_search": True, "gw_order": "overlap", "stream": True}) as r:
                got = b"".join(r.iter_bytes())
            t = RECENT[-1]
            chk("overlap is a buildable order", t.order_honoured is True and t.order == "overlap")
            chk("overlap emits chat-shaped SSE", b'"object": "chat.completion.chunk"' in got)
            chk("overlap terminates the stream", got.rstrip().endswith(b"data: [DONE]"))
            chk("overlap did not error", t.status == "ok")
            chk("search query defaults to the user turn", seen_q == ["who won"])

            sent = [json.loads(b) for b in CLIENT.bodies if b'"prompt"' in b]
            chk("overlap made two upstream calls", len(sent) == 2)
            if len(sent) == 2:
                first, second = sent[0]["prompt"], sent[1]["prompt"]
                # The property the whole design rests on, asserted end to end.
                chk("re-issue extends the first prompt plus what was generated",
                    second.startswith(first + "<think>\nreason"))
                chk("re-issue carries the retrieved block", "SOURCES" in second)
                chk("upstream is /v1/completions, not chat", "messages" not in sent[0])
            chk("cached tokens recorded from usage", t.reissue_cached_tokens == 8)
            chk("splice length recorded", t.splice_chars > 0)

            # Proves the actuator BINDS. A query field the gateway ignores would leave the
            # answer-format instruction in every search string and nothing would say so.
            seen_q.clear()
            CLIENT = _FakeUpstream(stream_chunks=comp)
            with client.stream("POST", "/v1/chat/completions", json={
                    "model": "m", "messages": [{"role": "user", "content": "who won\n\nANSWER: <integer>"}],
                    "gw_search": True, "gw_order": "overlap", "stream": True,
                    "gw_query": "who won the 1998 final"}) as r:
                b"".join(r.iter_bytes())
            chk("gw_query overrides the user turn as the search query",
                seen_q == ["who won the 1998 final"])
            chk("gw_query is stripped before the upstream call",
                b"gw_query" not in CLIENT.last_body)

            # A failing search must not take the turn down; it just never splices.
            async def _boom(*a, **kw):
                raise RuntimeError("brave is down")
            run_search = _boom
            CLIENT = _FakeUpstream(stream_chunks=comp)
            with client.stream("POST", "/v1/chat/completions", json={
                    "model": "m", "messages": [{"role": "user", "content": "q"}],
                    "gw_search": True, "gw_order": "overlap", "stream": True}) as r:
                got2 = b"".join(r.iter_bytes())
            t = RECENT[-1]
            chk("overlap survives a dead search", t.status == "ok")
            chk("a dead search splices nothing", t.splice_chars == 0)
            chk("a dead search still answers", b'"object": "chat.completion.chunk"' in got2)
            chk("a dead search makes one upstream call",
                len([b for b in CLIENT.bodies if b'"prompt"' in b]) == 1)

            run_search, render_block, RENDER = _orig_rs, _orig_render_block, _orig_render
            CLIENT = _FakeUpstream(stream_chunks=chunks)

            absent_body = {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                           "stream": True}
            with client.stream("POST", "/v1/chat/completions", json=absent_body) as r:
                absent_bytes = b"".join(r.iter_bytes())
            CLIENT = _FakeUpstream(stream_chunks=chunks)
            with client.stream("POST", "/v1/chat/completions",
                               json={**absent_body, "gw_events": False}) as r:
                false_bytes = b"".join(r.iter_bytes())
            chk("gw_events false preserves exact OpenAI bytes", false_bytes == absent_bytes ==
                b"".join(chunks))

            from gateway.search import Source

            async def _event_search(q, *a, **kw):
                return SearchOutcome(query=q, sources=[Source(url="https://example.test/a",
                    title="Example source", text="source body", ok=True)], search_ms=2.0,
                    fetch_ms=3.0, extract_ms=1.0, n_sources=1)

            run_search = _event_search
            CLIENT = _FakeUpstream(stream_chunks=chunks)
            with client.stream("POST", "/v1/chat/completions", json={
                    "model": "m", "messages": [{"role": "user", "content": "latest"}],
                    "stream": True, "gw_search": True, "gw_events": True}) as r:
                event_bytes = b"".join(r.iter_bytes())
            run_search = _orig_rs
            frames = []
            for line in event_bytes.splitlines():
                if line.startswith(b"data: ") and line != b"data: [DONE]":
                    try:
                        frames.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass
            gateway_frames = [frame for frame in frames if frame.get("object") == "gw.event"]
            chk("opt-in stream marks search start", gateway_frames[0].get("type") ==
                "search.started" and gateway_frames[0].get("query") == "latest")
            chk("opt-in stream includes prompt sources", any(
                frame.get("type") == "sources" and frame.get("sources") ==
                [{"title": "Example source", "url": "https://example.test/a"}]
                for frame in gateway_frames))
            chk("opt-in stats follow OpenAI DONE", event_bytes.find(b"data: [DONE]") <
                event_bytes.find(b'"type": "stats"'))
            chk("opt-in stream preserves the original OpenAI frames",
                all(frame in event_bytes for frame in chunks))

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

"""
proxy.py -- OpenAI-streaming passthrough that records what crossed the wire.
Relays bytes unchanged so it cannot alter a measurement, and accounts for SSE
events alongside them. See NOTES/code-notes.md for the byte-fidelity argument.

  uvicorn labbench.proxy:app --port 8080
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

UPSTREAM = os.environ.get("LABBENCH_UPSTREAM", "http://localhost:8000")
TRACE_PATH = os.environ.get("LABBENCH_TRACE", "results/labbench-traces.jsonl")
TIMEOUT = float(os.environ.get("LABBENCH_TIMEOUT", "600"))


@dataclass
class Trace:
    """One request's server-side record. Every field is measured next to the engine."""
    request_id: str
    t_wall: float
    upstream: str
    model: str = ""
    status: str = "ok"
    http_status: int = 0
    error: str | None = None
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    n_events: int = 0                 # SSE data events, [DONE] excluded
    n_content_events: int = 0         # events that carried text
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_event: float | None = None
    inter_event_p50_ms: float | None = None
    inter_event_p95_ms: float | None = None
    itl_ms_derived: float | None = None
    events: list[dict] = field(default_factory=list)


def pct(xs: list[float], p: float) -> float | None:
    """Nearest-rank percentile. None below 1/(1-p) samples -- a p95 of 7 points is the max."""
    if not xs or len(xs) < 1 / (1 - p):
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p * len(s))) - 1 if p * len(s) >= 1 else 0)]


class StreamAccounting:
    """Consumes SSE bytes and records timing plus the server's own token counts.
    Never treats an event as a token: speculative decoding puts several in one."""

    def __init__(self, trace: Trace, t_sent: float):
        self.tr = trace
        self.t0 = t_sent
        self.buf = b""
        self.last: float | None = None
        self.gaps: list[float] = []
        self.done = False

    def feed(self, blob: bytes, now: float) -> None:
        self.buf += blob
        while b"\n" in self.buf:
            raw, self.buf = self.buf.split(b"\n", 1)
            self._line(raw.strip(), now)

    def _line(self, raw: bytes, now: float) -> None:
        if not raw.startswith(b"data: "):
            return
        data = raw[6:]
        if data == b"[DONE]":
            self.done = True
            return
        try:
            ev = json.loads(data)
        except json.JSONDecodeError:
            return
        self.tr.n_events += 1
        self._usage(ev.get("usage"))
        if not self.tr.model:
            self.tr.model = ev.get("model", "") or ""
        text = self._text(ev)
        if not text:
            return
        self.tr.n_content_events += 1
        self.tr.events.append({"t_ms": round((now - self.t0) * 1e3, 3),
                               "bytes": len(raw), "chars": len(text)})
        if self.tr.ttft_ms is None:
            self.tr.ttft_ms = (now - self.t0) * 1e3
        elif self.last is not None:
            self.gaps.append((now - self.last) * 1e3)
        self.last = now

    @staticmethod
    def _text(ev: dict) -> str:
        ch = ev.get("choices") or []
        if not ch:
            return ""
        d = ch[0].get("delta") or {}
        return (d.get("content") or d.get("reasoning_content") or "") or ""

    def _usage(self, u: dict | None) -> None:
        if not u:
            return
        if u.get("completion_tokens"):
            self.tr.completion_tokens = int(u["completion_tokens"])
        if u.get("prompt_tokens"):
            self.tr.prompt_tokens = int(u["prompt_tokens"])
        det = u.get("prompt_tokens_details") or {}
        if det.get("cached_tokens") is not None:
            self.tr.cached_tokens = int(det["cached_tokens"])

    def _derive(self) -> None:
        tr = self.tr
        tr.inter_event_p50_ms = pct(self.gaps, 0.50)
        tr.inter_event_p95_ms = pct(self.gaps, 0.95)
        if tr.completion_tokens and tr.n_content_events:
            tr.tokens_per_event = tr.completion_tokens / tr.n_content_events
        if (tr.completion_tokens and tr.completion_tokens > 1
                and tr.ttft_ms is not None and tr.e2e_ms is not None):
            tr.itl_ms_derived = (tr.e2e_ms - tr.ttft_ms) / (tr.completion_tokens - 1)

    def partial(self, now: float) -> dict:
        """Live snapshot of a request still streaming, so TTFT is server-measured from
        the first token rather than only after the stream ends."""
        self.tr.e2e_ms = (now - self.t0) * 1e3
        self._derive()
        d = asdict(self.tr)
        d["status"] = "streaming"
        return d

    def finish(self, now: float) -> Trace:
        tr = self.tr
        tr.e2e_ms = (now - self.t0) * 1e3
        self._derive()
        if tr.ttft_ms is None and tr.status == "ok":
            tr.status = "empty"
        return tr


def write_trace(tr: Trace) -> None:
    """Flush per request. A run that writes at the end loses everything to a crash."""
    path = TRACE_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(tr)) + "\n")
        f.flush()


CLIENT: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """One shared client for the process. Per-request clients cost a TCP setup each,
    measured at +11 ms of TTFT against the mock."""
    global CLIENT
    CLIENT = httpx.AsyncClient(timeout=TIMEOUT,
                               limits=httpx.Limits(max_connections=256,
                                                   max_keepalive_connections=256))
    try:
        yield
    finally:
        await CLIENT.aclose()


app = FastAPI(lifespan=lifespan)
RECENT: list[Trace] = []
INFLIGHT: dict[str, StreamAccounting] = {}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.body()
    tr = Trace(request_id=uuid.uuid4().hex[:12], t_wall=time.time(), upstream=UPSTREAM)
    streaming = b'"stream"' in body and b'"stream": false' not in body
    if not streaming:
        return await _passthrough_json(body, tr)
    return StreamingResponse(_relay(body, tr), media_type="text/event-stream")


async def _passthrough_json(body: bytes, tr: Trace) -> Response:
    t0 = time.perf_counter()
    r = await CLIENT.post(f"{UPSTREAM}/v1/chat/completions", content=body,
                          headers={"content-type": "application/json"})
    tr.http_status, tr.e2e_ms = r.status_code, (time.perf_counter() - t0) * 1e3
    tr.status = "ok" if r.status_code == 200 else "http_error"
    _record(tr)
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


async def _relay(body: bytes, tr: Trace):
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
        _record(acc.finish(time.perf_counter()))


def _record(tr: Trace) -> None:
    RECENT.append(tr)
    del RECENT[:-200]
    write_trace(tr)


@app.get("/labbench/traces")
async def traces(n: int = 20):
    """Finished traces, then anything still streaming. Newest last either way."""
    now = time.perf_counter()
    live = [a.partial(now) for a in list(INFLIGHT.values())]
    return {"upstream": UPSTREAM, "traces": [asdict(t) for t in RECENT[-n:]] + live}


@app.get("/health")
async def health():
    return {"ok": True, "upstream": UPSTREAM}


def selftest() -> int:
    """Feed a synthetic speculative-decoding stream: 3 events, 7 tokens."""
    fails = []

    def chk(name, got, want):
        if got != want:
            fails.append(f"  FAIL {name}: got {got!r}, want {want!r}")

    tr = Trace(request_id="t", t_wall=0.0, upstream="test")
    acc = StreamAccounting(tr, 0.0)
    def ev(txt, usage=None):
        d = {"model": "m", "choices": [{"delta": {"content": txt}}]}
        if usage:
            d["usage"] = usage
        return b"data: " + json.dumps(d).encode() + b"\n\n"

    acc.feed(ev("Hello wor"), 0.100)
    acc.feed(ev("ld this is"), 0.140)
    acc.feed(ev(" spec"), 0.180)
    acc.feed(b'data: {"choices":[],"usage":{"completion_tokens":7,"prompt_tokens":11,'
             b'"prompt_tokens_details":{"cached_tokens":8}}}\n\n', 0.185)
    acc.feed(b"data: [DONE]\n\n", 0.190)
    out = acc.finish(0.190)

    chk("n_content_events", out.n_content_events, 3)
    chk("n_events", out.n_events, 4)
    chk("completion_tokens", out.completion_tokens, 7)
    chk("prompt_tokens", out.prompt_tokens, 11)
    chk("cached_tokens", out.cached_tokens, 8)
    chk("tokens_per_event", round(out.tokens_per_event, 4), round(7 / 3, 4))
    chk("ttft_ms", round(out.ttft_ms, 1), 100.0)
    # incident 32: no field may equal the event count when they differ
    if out.completion_tokens == out.n_content_events:
        fails.append("  FAIL tokens equal event count -- chunk counting has crept back")
    chk("itl_ms_derived", round(out.itl_ms_derived, 3), round((190 - 100) / 6, 3))

    tr4 = Trace(request_id="t4", t_wall=0.0, upstream="test")
    acc4 = StreamAccounting(tr4, 0.0)
    acc4.feed(ev("mid str"), 0.100)
    acc4.feed(ev("eam here"), 0.140)
    live = acc4.partial(0.150)
    chk("partial marked streaming", live["status"], "streaming")
    chk("partial has server ttft", round(live["ttft_ms"], 1), 100.0)
    chk("partial content events", live["n_content_events"], 2)
    # no usage chunk yet, so these stay unknown rather than being guessed from event count
    chk("partial tokens unknown", live["completion_tokens"], None)
    chk("partial tokens_per_event unknown", live["tokens_per_event"], None)
    chk("partial does not break finish", round(acc4.finish(0.150).e2e_ms, 1), 150.0)

    tr2 = Trace(request_id="t2", t_wall=0.0, upstream="test")
    acc2 = StreamAccounting(tr2, 0.0)
    acc2.feed(b"data: [DONE]\n\n", 0.01)
    out2 = acc2.finish(0.01)
    chk("empty stream status", out2.status, "empty")
    chk("empty stream tokens_per_event", out2.tokens_per_event, None)

    tr3 = Trace(request_id="t3", t_wall=0.0, upstream="test")
    acc3 = StreamAccounting(tr3, 0.0)
    half = ev("split")
    acc3.feed(half[:9], 0.05)
    acc3.feed(half[9:], 0.06)
    chk("split across byte chunks", tr3.n_content_events, 1)

    print("\n".join(fails) if fails else "labbench/proxy.py selftest: all checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(selftest())

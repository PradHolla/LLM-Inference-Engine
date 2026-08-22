"""
server.py -- Phase 2 step 3b: put the continuous batching engine behind an
OpenAI-compatible streaming HTTP API, so tools/bench.py can measure it open-loop
against Phase 1's 0.332 req/s the same way it measured the baseline.

Wire-compatible with tools/bench.py -- see baseline/server.py's _chunk() and
_stream_response() for the contract this file copies exactly: the `data:
{...}\\n\\n` SSE framing, the response envelope shape, the final `data:
[DONE]\\n\\n`, and the `vllm:`-prefixed /metrics counters infra/idle-shutdown.sh
reads. bench.py must work against this server with NO changes.

THE THREADING MODEL -- read this before touching anything below:

  ONE thread (the "engine thread", started in `lifespan`) owns the GPU. It is the
  ONLY thread that ever calls `engine.step()`, and therefore the only thread that
  ever touches `engine.pending`, `engine.rows`, `engine.cache`, `engine.mask`,
  `engine.nxt`, `engine.next_pos` -- any of Engine's state. Nothing outside
  `_engine_loop` and the functions it calls (`_drain_intake`, `_sweep_
  cancellations`, `_publish_metrics`) may read or write those attributes. The
  asyncio event loop thread (running every FastAPI handler) never touches the GPU,
  never touches `engine.*` state, and never blocks the engine thread.

  Two one-way channels cross the thread boundary, and it is only ever these two:

    intake (queue.Queue, asyncio thread -> engine thread): an HTTP handler builds
    a Request and calls `STATE.intake.put_nowait(r)`. `queue.Queue` is thread-safe
    by design, so this is the one operation on a Request that's allowed to happen
    from the asyncio thread before the engine thread has taken ownership of it.
    The engine thread drains it at the top of every loop iteration (`_drain_
    intake`) and is the one that calls `engine.submit()` -- never the handler.

    per-request token queue (asyncio.Queue, engine thread -> asyncio thread): the
    engine thread's on_token/on_complete callbacks (themselves called synchronously
    from inside `engine.step()`, i.e. running ON the engine thread) deliver text
    via `loop.call_soon_threadsafe(q.put_nowait, item)`, where `loop` is the
    specific request's asyncio event loop, captured with `asyncio.get_running_
    loop()` in the handler BEFORE the request is submitted. call_soon_threadsafe is
    the only sanctioned way to reach into the event loop from another thread --
    direct `q.put_nowait()` from off-loop is not thread-safe for asyncio.Queue,
    even though the name doesn't warn you.

  Everything else that looks like it crosses threads is a plain, GIL-atomic
  attribute read/write with a single writer, same reasoning as baseline/server.py's
  METRICS: `Request.cancelled` is written only by the asyncio thread (in a
  generator's `finally`) and read only by the engine thread; `METRICS`/`SNAPSHOT`
  fields are written only by the engine thread and read only by the asyncio thread
  (inverted from baseline, because generation happens on the engine thread here,
  not inside the request handler). Neither direction needs a lock: a plain
  attribute store/load is atomic under the GIL, so a reader sees either the old
  value or the new one, never a torn one -- there is nothing to race as long as
  only one thread ever writes.

  /metrics NEVER reads `engine.rows` / `engine.pending` / `engine.mask` directly --
  even though that would return the right numbers most of the time, it would be a
  second, unsanctioned way to touch engine-thread-owned state from the asyncio
  thread. Instead the engine thread copies the plain ints it needs into METRICS
  once per loop iteration (`_publish_metrics`), and /metrics only ever reads those
  copies.

  /v1/chat/completions --model ... --max-tokens ...
    python -m engine.server --model Qwen/Qwen3-8B --max-batch 8 --port 8000
"""
from __future__ import annotations

import os
# MUST precede `import torch` (transitively, via engine.continuous / engine.manual
# below) -- read once at CUDA allocator init, silently ignored after. Same
# requirement as every other engine/ module; set again here (redundantly safe,
# os.environ.setdefault is idempotent) in case a future refactor reorders imports
# and this file is no longer guaranteed to import engine.continuous first.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import asyncio
import itertools
import json
import logging
import queue
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.continuous import Engine, Request, collect_eos_ids, run_warmup
from engine.manual import make_prompt, pick_logits_kwarg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("engine.server")

GIB = 1 << 30

# How many tokens the one-time warmup pass (item H) prefills at, per admission
# shape 1..max_batch. Matches engine/continuous.py main()'s own --prompt-tokens
# default (512) -- not exposed as a CLI flag here, since item G's flag list is
# fixed and does not include one; the warmup shape only needs to be IN THE
# NEIGHBOURHOOD of real traffic for cuBLAS's kernel selection to transfer, not
# exact.
WARMUP_PROMPT_TOKENS = 512

_rid_counter = itertools.count(1)


@dataclass
class Config:
    """Populated by main() from argv BEFORE uvicorn.run() is called (item G).
    lifespan() reads it when the ASGI app actually starts -- by then argv has
    already been parsed, so there is no ordering hazard despite this looking like
    a global mutated after import."""
    model: str = "Qwen/Qwen3-8B"
    max_batch: int = 8
    compact_threshold: int = 128
    ignore_eos: bool = False


CONFIG = Config()


@dataclass
class ModelState:
    """Populated once in lifespan, before yield. tokenizer/model/fwd_kw/eos_ids are
    read-only after that point (every reader, on either thread, only ever reads
    them once startup has finished) so no lock is needed for those. `intake` is the
    thread-safe queue.Queue bridge described in the module docstring."""
    tokenizer: Any = None
    model: Any = None
    fwd_kw: str | None = None
    eos_ids: set[int] = field(default_factory=set)
    intake: "queue.Queue[Request] | None" = None
    engine_thread: threading.Thread | None = None
    stop_event: threading.Event | None = None
    gpu_name: str = ""
    weights_gib: float = 0.0


STATE = ModelState()


@dataclass
class Metrics:
    """Written ONLY by the engine thread -- either once per loop iteration
    (running/waiting/buffer_len/compactions, via _publish_metrics) or synchronously
    from inside engine.step()'s on_token/on_complete callbacks (prompt_tokens_total/
    generation_tokens_total). Read only by the asyncio thread, from the /metrics
    handler. Plain ints, single writer: safe without a lock for the same reason
    baseline/server.py's Metrics is, just with the writer/reader roles swapped
    (there, the event-loop thread writes because generation happens inline in the
    handler; here the engine thread writes because generation happens off-loop).

    running/waiting are also republished under engine:active_rows / engine:
    pending_depth for item F2 -- same numbers, non-vllm:-prefixed names so
    infra/idle-shutdown.sh's awk patterns (which only match `vllm:...`) can't
    mistake them for something they're not.
    """
    running: int = 0                 # len(engine.rows) -- vllm:num_requests_running
    waiting: int = 0                 # intake.qsize() + len(engine.pending) -- vllm:num_requests_waiting
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0
    buffer_len: int = 0              # current shared KV buffer length (item F2)
    compactions: int = 0             # total compaction events (item F2)


METRICS = Metrics()

# Set by the engine thread if engine.step() raises something that is NOT the
# specific, pre-mutation "ragged admission batch" RuntimeError (see
# _is_recoverable_admit_error below) -- e.g. a CUDA OOM mid-forward. Such an
# exception can leave engine.cache/engine.mask out of sync with each other (see
# _engine_loop's docstring), so the engine thread stops calling step() entirely
# rather than risk producing silently-wrong tokens from a corrupted buffer. Read
# by the asyncio thread (to reject new requests with 503, and to stop an in-flight
# stream from hanging forever); written only by the engine thread. threading.Event
# is itself thread-safe, so this needs no additional protection.
WEDGED = threading.Event()


# --------------------------------------------------------------------- engine thread
def _is_recoverable_admit_error(exc: BaseException) -> bool:
    """True for the two specific RuntimeErrors engine.continuous.Engine._admit()
    raises for a ragged-length admission batch (see its own comments: "this design
    assumes uniform --prompt-tokens prompts" / "buffer/admission invariant
    violated"). Matched by message text, not type, because continuous.py raises a
    bare RuntimeError for both -- there is no dedicated exception class to catch,
    and this file may not add one there (the only sanctioned edit to that file is
    the on_token parameter).

    KNOWN, ACCEPTED LIMITATION (see the file's own uncertainty list): a real HTTP
    server sees genuinely ragged prompt lengths -- unlike continuous.py's own
    offline harness, which deliberately keeps every prompt in a run identical
    (module docstring: "PROMPTS ARE UNIFORM ... ragged-prompt-length [is] a
    separate, unmeasured problem"). Two concurrently-pending requests whose
    prompts tokenize to different lengths WILL hit this if they land in the same
    admission batch. Both of _admit()'s RuntimeErrors are raised BEFORE any
    cache/mask mutation for that batch (verified by reading _admit() top to
    bottom: the uniformity check and the padn<0 check both precede every torch op
    that touches self.cache/self.mask), so recovering by dropping that batch's
    newcomers and continuing is safe -- engine state is provably untouched. The
    newcomers themselves are unrecoverably lost (they were already popped off
    engine.pending before the check that fails); their HTTP requests hang until
    the CLIENT's own timeout fires. That is a real, deliberately-not-fixed gap,
    not an oversight -- fixing it means teaching _admit() to pad a ragged batch,
    which is out of this step's scope and out of continuous.py's stated one.
    """
    return isinstance(exc, RuntimeError) and (
        "ragged prompt lengths" in str(exc) or "buffer/admission invariant" in str(exc)
    )


def _drain_intake(engine: Engine, intake: "queue.Queue[Request]") -> None:
    """Item B: the ONLY place engine.submit() is called. Runs on the engine
    thread, at the top of every loop iteration."""
    while True:
        try:
            r = intake.get_nowait()
        except queue.Empty:
            break
        if getattr(r, "cancelled", False):
            continue  # client already gone before this row ever reached engine.pending
        engine.submit(r)


def _sweep_cancellations(engine: Engine) -> None:
    """Item E. engine.pending and engine.rows are engine-thread-owned (module
    docstring) -- safe to mutate directly here because this function only ever
    runs ON the engine thread, same as _drain_intake.

    A request cancelled while still queued is dropped outright -- it never cost
    the engine anything. A request cancelled mid-decode cannot be ripped out of
    the shared batch mid-step (its K/V rows are interleaved with every other
    active row's, and evicting is a batched index_select over ALL rows, not a
    per-row op) -- so instead its max_new_tokens is capped at its current decode
    count. Engine._finished() (continuous.py) already treats decode_count >=
    max_new_tokens as done, so the row is picked up by the very next _evict() --
    i.e. evicted within one engine step of the disconnect being noticed, freeing
    its slot without producing another token nobody will read.
    """
    if engine.pending:
        engine.pending = [r for r in engine.pending if not getattr(r, "cancelled", False)]
    for r in engine.rows:
        if getattr(r, "cancelled", False) and r.finish_step is None:
            r.max_new_tokens = max(0, len(r.tokens) - 1)


def _publish_metrics(engine: Engine, intake: "queue.Queue[Request]") -> None:
    """Item F2 + the running/waiting half of item F. The ONLY place METRICS.running/
    waiting/buffer_len/compactions are written -- once per engine-loop iteration,
    from the engine thread. See Metrics' docstring for why this needs no lock."""
    METRICS.running = len(engine.rows)
    METRICS.waiting = intake.qsize() + len(engine.pending)
    METRICS.buffer_len = int(engine.mask.shape[-1]) if engine.mask is not None else 0
    METRICS.compactions = len(engine.compact_s)


def _on_token(r: Request, tok_id: int) -> None:
    """Engine.on_token -- called synchronously from inside engine.step() (_admit()
    for the free admission token, _decode() for every decode-loop token), i.e. on
    the engine thread. Wrapped in its own try/except so that a failure HERE (a bad
    token id, a closed loop) can never cascade into wedging the whole engine: by
    the time this runs, r.tokens/r.itls bookkeeping for this row is already
    complete (continuous.py appends before calling the hook), so swallowing an
    error here costs the client one missing delta chunk, not engine correctness.
    """
    try:
        if getattr(r, "cancelled", False):
            return  # no one is reading r.out_q -- don't bother decoding or scheduling
        METRICS.generation_tokens_total += 1
        text = STATE.tokenizer.decode([tok_id], skip_special_tokens=True)
        if text:
            r.loop.call_soon_threadsafe(r.out_q.put_nowait, text)
    except Exception:
        logger.exception("on_token failed for rid=%s -- token dropped, engine state unaffected", r.rid)


def _on_complete(r: Request) -> None:
    """Engine.on_complete -- called from _evict(), on the engine thread, for every
    row that finishes OR is force-finished by _sweep_cancellations. Mirrors
    baseline/server.py's implicit behaviour: prompt_tokens_total is only counted
    for requests that actually got delivered (baseline's own METRICS update sits
    downstream of a try/finally that a cancelled generator never reaches -- a
    disconnected client's tokens aren't counted there either). generation_tokens_
    total is symmetric with that in _on_token above.
    """
    try:
        if not getattr(r, "cancelled", False):
            METRICS.prompt_tokens_total += int(r.prompt_ids.shape[0])
            r.loop.call_soon_threadsafe(r.out_q.put_nowait, None)  # sentinel: stream is done
    except Exception:
        logger.exception("on_complete failed for rid=%s", r.rid)


def _engine_loop(engine: Engine, intake: "queue.Queue[Request]", stop_event: threading.Event) -> None:
    """Item A. The ONE thread that ever calls engine.step(). Runs until server
    shutdown (stop_event) or an unrecoverable exception (WEDGED) -- see module
    docstring and _is_recoverable_admit_error for what "unrecoverable" means here.
    """
    logger.info("engine thread starting")
    while not stop_event.is_set():
        try:
            _drain_intake(engine, intake)
            _sweep_cancellations(engine)
            if engine.pending or engine.rows:
                engine.step()
            else:
                time.sleep(0.001)
            _publish_metrics(engine, intake)
        except Exception as e:
            if _is_recoverable_admit_error(e):
                logger.warning(
                    "admission rejected a ragged-length batch (%s) -- those "
                    "newcomers are dropped; their HTTP requests will hang until "
                    "the client's own timeout fires. KNOWN LIMITATION: continuous.py's "
                    "_admit() assumes uniform prompt lengths within one admission "
                    "batch (see its module docstring). Engine state is NOT corrupted "
                    "by this specific error -- it raises before any cache/mask "
                    "mutation -- so the loop continues.", e,
                )
                continue
            logger.exception(
                "engine step failed -- WEDGING the engine thread. A mid-step "
                "exception (e.g. CUDA OOM) can leave engine.cache and engine.mask "
                "out of sync with each other (see this file's module docstring); "
                "continuing to call step() risks producing silently-wrong tokens "
                "rather than a loud failure, so this thread stops making progress "
                "instead. The process needs a restart to recover."
            )
            WEDGED.set()
            return
    logger.info("engine thread stopping (shutdown)")


# ------------------------------------------------------------------------- lifespan
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("loading %s ...", CONFIG.model)
    t0 = time.perf_counter()
    STATE.tokenizer = AutoTokenizer.from_pretrained(CONFIG.model)
    STATE.model = AutoModelForCausalLM.from_pretrained(
        CONFIG.model, dtype=torch.bfloat16, device_map="cuda"
    )
    STATE.model.eval()
    load_s = time.perf_counter() - t0

    STATE.weights_gib = torch.cuda.memory_allocated() / GIB
    STATE.gpu_name = torch.cuda.get_device_name(0)
    logger.info(
        "loaded %s in %.1fs | %.2f GiB allocated | %s",
        CONFIG.model, load_s, STATE.weights_gib, STATE.gpu_name,
    )
    print(
        f"[engine.server] {CONFIG.model} ready in {load_s:.1f}s -- "
        f"{STATE.weights_gib:.2f} GiB weights on {STATE.gpu_name}"
    )

    STATE.fwd_kw = pick_logits_kwarg(STATE.model)
    STATE.eos_ids = collect_eos_ids(STATE.tokenizer, STATE.model)
    logger.info("logits kwarg: %s", STATE.fwd_kw or "NOT FOUND -- logits explosion risk")
    logger.info("eos ids: %s", sorted(STATE.eos_ids) or "none found")

    device = torch.device("cuda:0")

    # Item H: warm up EVERY admission batch shape (1..max_batch) before binding
    # traffic, not just max_batch -- reusing continuous.py's own run_warmup rather
    # than re-deriving it. Measured 2026-08-21 in this project: a warmup at the
    # wrong (batch, seq) shape warms nothing, since cuBLAS selects kernels per
    # problem shape; the first REAL request would otherwise eat a 36%+ latency
    # outlier on top of the 60-105s model load this is also here to avoid exposing
    # to a live request.
    warm_body = make_prompt(WARMUP_PROMPT_TOKENS)
    warm_ids = STATE.tokenizer.apply_chat_template(
        [{"role": "user", "content": warm_body}],
        add_generation_prompt=True, tokenize=True, return_tensors="pt",
        enable_thinking=False,
    )
    # transformers 5.x returns a BatchEncoding (dict-like) from apply_chat_template
    # when tokenize=True; 4.x returned a bare tensor. VERIFIED in baseline/server.py
    # and engine/continuous.py's main().
    if not hasattr(warm_ids, "shape"):
        warm_ids = warm_ids["input_ids"]
    template_row = warm_ids.to(device)[0]
    run_warmup(STATE.model, STATE.fwd_kw, CONFIG.max_batch, template_row, STATE.eos_ids, device)

    STATE.intake = queue.Queue()
    STATE.stop_event = threading.Event()
    engine = Engine(
        STATE.model, STATE.fwd_kw, CONFIG.max_batch, device, STATE.eos_ids,
        ignore_eos=CONFIG.ignore_eos, on_complete=_on_complete, on_token=_on_token,
        compact_threshold=CONFIG.compact_threshold,
    )
    STATE.engine_thread = threading.Thread(
        target=_engine_loop, args=(engine, STATE.intake, STATE.stop_event),
        daemon=True, name="engine-gpu-thread",
    )
    STATE.engine_thread.start()
    print(
        f"[engine.server] engine thread started -- max_batch={CONFIG.max_batch} "
        f"compact_threshold={CONFIG.compact_threshold} ignore_eos={CONFIG.ignore_eos}"
    )

    yield

    if STATE.stop_event is not None:
        STATE.stop_event.set()
    if STATE.engine_thread is not None:
        STATE.engine_thread.join(timeout=5.0)
    # No CUDA context teardown beyond that: process exit frees it, same as
    # baseline/server.py -- this server never restarts the model mid-process.


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------- API models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = True
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


def _chunk(request_id: str, created: int, model_name: str,
           delta: dict[str, Any], finish_reason: str | None) -> str:
    """Copied verbatim from baseline/server.py's _chunk() -- this shape, not a
    reinterpretation of it, is the contract bench.py is written against."""
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_response(req: ChatCompletionRequest, r: Request,
                           prompt_tokens: int, t0: float) -> AsyncIterator[str]:
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    yield _chunk(request_id, created, req.model, {"role": "assistant"}, None)

    try:
        while True:
            # Bounded wait, not a bare await -- a wedged engine thread (WEDGED,
            # see module docstring) will never deliver the sentinel for a row
            # already in flight when it wedged, and without this the stream (and
            # the client waiting on it) would hang forever instead of ending.
            try:
                item = await asyncio.wait_for(r.out_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if WEDGED.is_set():
                    logger.error("request %s abandoned -- engine thread is wedged", request_id)
                    return
                continue
            if item is None:  # sentinel: engine finished this row
                break
            yield _chunk(request_id, created, req.model, {"content": item}, None)
    finally:
        # Item E, unconditional -- mirrors baseline/server.py's own cleanup
        # discipline. Runs whether this generator finished normally (sentinel
        # received) or Starlette cancelled it because the client disconnected. If
        # the exception path is why we're here, this is the last line of the
        # function that executes: the cancellation continues propagating past a
        # bare `finally` with no `except`, so everything below (the final chunk,
        # [DONE]) is correctly skipped in that case, same control-flow shape as
        # baseline/server.py's _generate_under_lock. If the row already finished
        # normally, the row is already gone from engine.rows/engine.pending, so
        # this flag has nothing left to affect -- a harmless no-op.
        r.cancelled = True

    generated_len = len(r.tokens)
    finish_reason = "length" if generated_len >= req.max_tokens else "stop"
    logger.info(
        "request %s: %d prompt tok, %d generated, finish=%s",
        request_id, prompt_tokens, generated_len, finish_reason,
    )

    yield _chunk(request_id, created, req.model, {}, finish_reason)
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> StreamingResponse:
    t0 = time.perf_counter()
    if not req.stream:
        # Streaming only, same restriction as baseline/server.py and for the same
        # reason: a non-streaming caller should get a clear 400, not a silent
        # multi-second hang waiting for a buffered response this server never
        # implements.
        raise HTTPException(
            status_code=400,
            detail="This server only supports stream=true. Non-streaming responses "
                   "are not implemented.",
        )
    if req.temperature > 0:
        # engine.continuous.Engine is greedy-argmax only -- there is no sampling
        # path to route temperature/top_p/top_k into (see Engine._admit/_decode:
        # both call .argmax(-1), unconditionally). Silently ignoring a non-zero
        # temperature would look like it worked and return the wrong distribution
        # of output -- exactly the "plausible but wrong" failure this project's
        # standards single out as the expensive kind. bench.py always sends 0.0,
        # so this never fires in the benchmark path.
        raise HTTPException(
            status_code=400,
            detail="This server only supports greedy decoding (temperature=0); "
                   "the continuous-batching engine is argmax-only.",
        )
    if WEDGED.is_set():
        raise HTTPException(status_code=503, detail="engine is wedged; restart the server")

    try:
        input_ids = STATE.tokenizer.apply_chat_template(
            [m.model_dump() for m in req.messages],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            **req.chat_template_kwargs,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid request: {e}") from None

    if not hasattr(input_ids, "shape"):
        input_ids = input_ids["input_ids"]

    prompt_tokens = int(input_ids.shape[-1])
    # Engine.Request.prompt_ids is 1-D [prompt_len] (see continuous.py's Request
    # docstring and its main()'s identical `template_row = input_ids.to(device)[0]`)
    # -- drop the batch dim baseline/server.py keeps for model.generate()'s sake.
    input_ids = input_ids.to(STATE.model.device)[0]

    # continuous.py's Request.max_new_tokens counts DECODE-LOOP tokens only -- the
    # admission forward produces one token for free and that free token is not
    # part of the decode budget (see continuous.py's module docstring, "TOKEN-COUNT
    # CONVENTION"). req.max_tokens is the OpenAI-style TOTAL output budget, so the
    # decode budget is one less.
    decode_budget = max(0, req.max_tokens - 1)

    r = Request(rid=next(_rid_counter), prompt_ids=input_ids, max_new_tokens=decode_budget)
    # Item E/C: attributes not part of continuous.py's Request dataclass, added
    # here because Request has no __slots__ -- plain instance attributes are safe.
    r.cancelled = False
    r.out_q = asyncio.Queue()
    r.loop = asyncio.get_running_loop()  # captured BEFORE submit -- item C

    STATE.intake.put_nowait(r)

    return StreamingResponse(
        _stream_response(req, r, prompt_tokens, t0),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "wedged" if WEDGED.is_set() else "ok",
        "model": CONFIG.model,
        "gpu": STATE.gpu_name,
        "weights_gib": round(STATE.weights_gib, 2),
    }


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    # Item F: the vllm:-prefixed lines are the ones infra/idle-shutdown.sh reads
    # and MUST keep these exact names -- do not rename without updating that
    # script's awk patterns too. Item F2: the engine:-prefixed lines are pure
    # diagnostics, deliberately NOT vllm:-prefixed so idle-shutdown.sh's
    # `/^vllm:.../` patterns can't accidentally match them.
    body = (
        "# HELP vllm:num_requests_running Requests currently admitted and decoding.\n"
        "# TYPE vllm:num_requests_running gauge\n"
        f"vllm:num_requests_running {METRICS.running}\n"
        "# HELP vllm:num_requests_waiting Requests submitted but not yet admitted.\n"
        "# TYPE vllm:num_requests_waiting gauge\n"
        f"vllm:num_requests_waiting {METRICS.waiting}\n"
        "# HELP vllm:prompt_tokens_total Cumulative prompt tokens processed.\n"
        "# TYPE vllm:prompt_tokens_total counter\n"
        f"vllm:prompt_tokens_total {METRICS.prompt_tokens_total}\n"
        "# HELP vllm:generation_tokens_total Cumulative tokens generated.\n"
        "# TYPE vllm:generation_tokens_total counter\n"
        f"vllm:generation_tokens_total {METRICS.generation_tokens_total}\n"
        "# HELP engine:active_rows Rows currently occupying a batch slot. Diagnostic -- not a vllm: metric, not read by infra/idle-shutdown.sh.\n"
        "# TYPE engine:active_rows gauge\n"
        f"engine:active_rows {METRICS.running}\n"
        "# HELP engine:pending_depth Requests submitted but not yet admitted. Diagnostic.\n"
        "# TYPE engine:pending_depth gauge\n"
        f"engine:pending_depth {METRICS.waiting}\n"
        "# HELP engine:buffer_len Current shared left-padded KV buffer length. Diagnostic.\n"
        "# TYPE engine:buffer_len gauge\n"
        f"engine:buffer_len {METRICS.buffer_len}\n"
        "# HELP engine:compactions_total Cumulative buffer compaction events. Diagnostic.\n"
        "# TYPE engine:compactions_total counter\n"
        f"engine:compactions_total {METRICS.compactions}\n"
    )
    return PlainTextResponse(body)


# --------------------------------------------------------------------------- CLI
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--compact-threshold", type=int, default=128,
                   help="trim the shared KV buffer once this many dead left-pad "
                        "positions accumulate; see engine/continuous.py's _compact()")
    p.add_argument("--ignore-eos", action="store_true", default=False,
                   help="do not stop generation on eos (default: OFF -- a real "
                        "server honours eos, unlike the offline engine's synthetic "
                        "benchmarks, which default this True)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        raise SystemExit(1)

    # This server is single-process, single-worker by design: the engine thread /
    # queue.Queue / asyncio.Queue plumbing above assumes exactly one Python
    # process owns the GPU and the intake queue. Do not run this under
    # `uvicorn ... --workers N>1` or a process manager that forks multiple
    # workers -- each worker would load its own copy of the model and compete for
    # the same GPU with no coordination between them.
    CONFIG.model = args.model
    CONFIG.max_batch = args.max_batch
    CONFIG.compact_threshold = args.compact_threshold
    CONFIG.ignore_eos = args.ignore_eos

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

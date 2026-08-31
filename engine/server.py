"""
server.py -- Phase 2 step 3b: continuous batching engine behind an OpenAI-compatible
streaming HTTP API for tools/bench.py; wire-compatible with baseline/server.py's SSE
framing. ONE engine thread owns the GPU; see docs for the full threading model.

  python -m engine.server --model Qwen/Qwen3-8B --max-batch 8 --port 8000
"""
from __future__ import annotations

import os
# MUST precede `import torch` (transitively) -- read once at CUDA init, ignored after.
# Set again here (setdefault is idempotent) in case a refactor reorders imports. See docs.
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

# How many tokens the one-time warmup pass (item H) prefills at, per admission shape
# 1..max_batch. Matches engine/continuous.py main()'s --prompt-tokens default. See docs.
WARMUP_PROMPT_TOKENS = 512

_rid_counter = itertools.count(1)


@dataclass
class Config:
    """Populated by main() from argv before uvicorn.run(). lifespan() reads it once
    the ASGI app starts, by which point argv is already parsed -- no ordering hazard
    despite looking like a global mutated after import."""
    model: str = "Qwen/Qwen3-8B"
    max_batch: int = 8
    compact_threshold: int = 128
    ignore_eos: bool = False


CONFIG = Config()


@dataclass
class ModelState:
    """Populated once in lifespan, before yield; tokenizer/model/fwd_kw/eos_ids are
    read-only after that (no lock needed). `intake` is the thread-safe queue.Queue
    bridge -- see the module-level threading-model doc."""
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
    """Written ONLY by the engine thread (per-loop-iteration or from on_token/
    on_complete), read only by the asyncio /metrics handler -- single writer, no
    lock needed. running/waiting are republished under engine:-prefixed names too."""
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
    """True for the two specific RuntimeErrors engine.continuous.Engine._admit() raises
    for a ragged-length admission batch, matched by message text (continuous.py has no
    dedicated exception class). See docs for why dropping that batch is safe."""
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
    """Item E. Runs only on the engine thread, like _drain_intake. A queued-and-
    cancelled request is dropped outright; a mid-decode cancellation caps its
    max_new_tokens so _finished()/_evict() reclaim its slot within one step. See docs."""
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
    """Engine.on_token -- called synchronously from inside engine.step(), on the
    engine thread. Wrapped in try/except so a failure here (bad token id, closed
    loop) costs one missing delta chunk, never engine correctness. See docs."""
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
    """Engine.on_complete -- called from _evict() for every finished or force-
    finished row. Mirrors baseline/server.py: prompt_tokens_total only counts
    delivered requests, symmetric with generation_tokens_total in _on_token."""
    try:
        if not getattr(r, "cancelled", False):
            METRICS.prompt_tokens_total += int(r.prompt_ids.shape[0])
            r.loop.call_soon_threadsafe(r.out_q.put_nowait, None)  # sentinel: stream is done
    except Exception:
        logger.exception("on_complete failed for rid=%s", r.rid)


def _engine_loop(engine: Engine, intake: "queue.Queue[Request]", stop_event: threading.Event) -> None:
    """Item A. The ONE thread that ever calls engine.step(). Runs until server
    shutdown (stop_event) or an unrecoverable exception (WEDGED) -- see
    _is_recoverable_admit_error for what "unrecoverable" means here."""
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
                    "batch (see NOTES/code-notes.md). Engine state is NOT corrupted "
                    "by this specific error -- it raises before any cache/mask "
                    "mutation -- so the loop continues.", e,
                )
                continue
            logger.exception(
                "engine step failed -- WEDGING the engine thread. A mid-step "
                "exception (e.g. CUDA OOM) can leave engine.cache and engine.mask "
                "out of sync with each other (see NOTES/code-notes.md); "
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

    # Item H: warm up EVERY admission shape (1..max_batch) before binding traffic --
    # a wrong-shape warmup warms nothing (cuBLAS picks kernels per shape). See docs.
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
            # Bounded wait, not a bare await: a wedged engine thread never delivers
            # the sentinel, so without this timeout the stream would hang forever.
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
        # Item E, unconditional cleanup mirroring baseline/server.py: runs whether
        # this generator finished normally or was cancelled by a disconnect. See docs.
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
        # Streaming only, same restriction as baseline/server.py: a non-streaming
        # caller gets a clear 400 instead of a silent hang. See docs.
        raise HTTPException(
            status_code=400,
            detail="This server only supports stream=true. Non-streaming responses "
                   "are not implemented.",
        )
    if req.temperature > 0:
        # Engine is greedy-argmax only (Engine._admit/_decode both call .argmax()
        # unconditionally) -- silently ignoring temperature would be plausible-but-wrong.
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
    # Engine.Request.prompt_ids is 1-D [prompt_len] (continuous.py) -- drop the batch
    # dim baseline/server.py keeps for model.generate()'s sake.
    input_ids = input_ids.to(STATE.model.device)[0]

    # continuous.py's max_new_tokens counts DECODE-LOOP tokens only (admission's free
    # token isn't part of the budget); req.max_tokens is the OpenAI TOTAL, so -1.
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
    # Item F: vllm:-prefixed lines are read by infra/idle-shutdown.sh -- keep these
    # exact names. engine:-prefixed lines are diagnostics-only, deliberately not vllm:.
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

    # Single-process, single-worker by design: the engine thread / queue plumbing
    # assumes exactly one process owns the GPU. Never run with --workers N>1.
    CONFIG.model = args.model
    CONFIG.max_batch = args.max_batch
    CONFIG.compact_threshold = args.compact_threshold
    CONFIG.ignore_eos = args.ignore_eos

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

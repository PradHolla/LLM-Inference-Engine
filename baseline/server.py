"""
server.py -- Phase 1: the deliberately bad baseline.

Plain HuggingFace `transformers`, `.generate()`, ONE request at a time. No batching,
no scheduler, no paged KV cache -- every limitation below is a choice, called out
where it happens, not an oversight. This exists to produce a bad number that later
phases beat. Do not optimize it; the naivety is the spec.

THE GLOBAL LOCK (see below) is the defining property: a single `asyncio.Lock` held
for the whole generation. Exactly one request touches the GPU at a time; everything
else queues behind it. Phase 2 replaces this with a real scheduler.

Wire-compatible with `tools/bench.py` -- see the SSE chunk shapes in `_chunk()` and
`_stream_response()`. That file is the spec for what a client is allowed to assume;
do not change the response shape without re-checking its parsing loop.

  MODEL_ID=Qwen/Qwen3-8B PORT=8000 SERVE_LOG=baseline/serve.jsonl \
    uvicorn baseline.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("baseline")

GIB = 1 << 30
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-8B")
PORT = int(os.environ.get("PORT", "8000"))
SERVE_LOG = os.environ.get("SERVE_LOG", "baseline/serve.jsonl")

# THE GLOBAL LOCK. Held for the ENTIRE generation (prefill + every decode step),
# not just the enqueue. Exactly one request is ever inside `model.generate()` at a
# time; every other request `await`s here, which is what turns concurrency into a
# queue. This single line is the whole architectural difference between this file
# and vLLM. Removing it -- and replacing it with a real batching scheduler -- is
# Phase 2, not a tweak to this one.
LOCK = asyncio.Lock()


@dataclass
class ModelState:
    """Populated once at startup, via lifespan. Never touched before `yield`."""
    tokenizer: Any = None
    model: Any = None
    gpu_name: str = ""
    weights_gib: float = 0.0


STATE = ModelState()


@dataclass
class Metrics:
    """In-process counters for /metrics. Mutated only on the event-loop thread
    (the background generation threads never touch these), so plain ints are
    safe -- no lock needed beyond the GIL's own atomicity of `+= 1`."""
    running: int = 0              # 0 or 1 -- this server never runs more than one request at once
    waiting: int = 0              # requests queued behind LOCK right now
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0


METRICS = Metrics()

SENTINEL = object()  # marks end-of-generation on the bridging asyncio.Queue


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("loading %s ...", MODEL_ID)
    t0 = time.perf_counter()
    STATE.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # dtype=, not the deprecated torch_dtype=. device_map="cuda" pins the whole
    # model onto the single GPU -- there is only one GPU on this box, and no
    # sharding logic exists in this baseline anyway.
    STATE.model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    )
    STATE.model.eval()
    load_s = time.perf_counter() - t0

    # The measured weight footprint -- compare this against roofline.py's
    # `weights = params * dtype_bytes` prediction. Any gap is activation buffers,
    # CUDA context, or a prediction that was wrong.
    STATE.weights_gib = torch.cuda.memory_allocated() / GIB
    STATE.gpu_name = torch.cuda.get_device_name(0)
    logger.info(
        "loaded %s in %.1fs | %.2f GiB allocated | %s",
        MODEL_ID, load_s, STATE.weights_gib, STATE.gpu_name,
    )
    print(
        f"[baseline] {MODEL_ID} ready in {load_s:.1f}s -- "
        f"{STATE.weights_gib:.2f} GiB weights on {STATE.gpu_name}"
    )
    yield
    # No explicit teardown: process exit frees the CUDA context. This baseline
    # never restarts the model mid-process, so there is nothing to release here.


app = FastAPI(lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = True
    # Passed straight through to apply_chat_template -- e.g. bench.py sends
    # {"enable_thinking": false} for Qwen3. We do not interpret it; the tokenizer
    # does.
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


def _chunk(request_id: str, created: int, model_name: str,
           delta: dict[str, Any], finish_reason: str | None) -> str:
    """One SSE frame in vLLM/OpenAI's streaming chat-completion shape. bench.py
    only reads choices[0].delta.content and the [DONE] sentinel, but the rest of
    the envelope is included so this baseline is a drop-in replacement for the
    real thing once later phases swap the server underneath it."""
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _run_generate(gen_kwargs: dict[str, Any], result: dict[str, Any]) -> None:
    """Runs on its own thread, invoked from `_stream_response`'s worker thread.
    `model.generate(streamer=...)` calls `streamer.put()` synchronously as each
    token is produced, from THIS thread -- that is what TextIteratorStreamer is
    for. Any exception (OOM, a bad generation kwarg) is captured here rather than
    raised across the thread boundary, since Python threads can't propagate
    exceptions to their caller."""
    try:
        result["output_ids"] = STATE.model.generate(**gen_kwargs)
    except Exception as e:  # surfaced to the request handler below, not swallowed
        result["error"] = e


async def _stream_response(req: ChatCompletionRequest, input_ids: torch.Tensor,
                           prompt_tokens: int, t0: float) -> AsyncIterator[str]:
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    METRICS.waiting += 1
    acquired = False
    try:
        async with LOCK:
            METRICS.waiting -= 1
            acquired = True
            METRICS.running += 1
            try:
                async for piece in _generate_under_lock(
                    req, input_ids, prompt_tokens, t0, request_id, created
                ):
                    yield piece
            finally:
                METRICS.running -= 1
    finally:
        if not acquired:
            # Cancelled (client disconnected) while still queued -- never
            # touched METRICS.waiting's decrement above, so do it here.
            METRICS.waiting -= 1


async def _generate_under_lock(req: ChatCompletionRequest, input_ids: torch.Tensor,
                               prompt_tokens: int, t0: float, request_id: str,
                               created: int) -> AsyncIterator[str]:
    """Everything in here runs while LOCK is held -- this is "the whole
    generation" the module docstring promises. Only one coroutine is ever
    inside this function body at a time across the whole process."""
    t_lock_acquired = time.perf_counter()
    queue_wait_s = t_lock_acquired - t0

    yield _chunk(request_id, created, req.model, {"role": "assistant"}, None)

    streamer = TextIteratorStreamer(
        STATE.tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs: dict[str, Any] = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=req.max_tokens,
        streamer=streamer,
        pad_token_id=STATE.tokenizer.pad_token_id or STATE.tokenizer.eos_token_id,
    )
    # Reproducibility beats variety for a benchmark: temperature 0 means greedy,
    # not "sampling with a temperature of 0" (which would divide by zero).
    if req.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=req.temperature)
    else:
        gen_kwargs.update(do_sample=False)

    loop = asyncio.get_running_loop()
    token_queue: asyncio.Queue[Any] = asyncio.Queue()
    result: dict[str, Any] = {}
    gen_thread_holder: list[threading.Thread] = []

    def worker() -> None:
        # `model.generate` runs on ITS OWN thread (started here) so that this
        # thread is free to drain the streamer concurrently -- `for text in
        # streamer` blocks on an internal queue.Queue.get() until generate()
        # produces the next token, and generate() can't do that while also
        # being the thread that's blocked reading its own output. Bridging to
        # asyncio happens via call_soon_threadsafe: this thread must never
        # touch the event loop directly, since it is not the loop's thread.
        gen_thread = threading.Thread(
            target=_run_generate, args=(gen_kwargs, result), daemon=True
        )
        gen_thread_holder.append(gen_thread)
        gen_thread.start()
        for text in streamer:
            if text:
                loop.call_soon_threadsafe(token_queue.put_nowait, text)
        gen_thread.join()
        loop.call_soon_threadsafe(token_queue.put_nowait, SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    t_first_token: float | None = None
    t_last_token: float | None = None
    try:
        while True:
            item = await token_queue.get()  # never blocks the loop -- it's a coroutine await
            if item is SENTINEL:
                break
            now = time.perf_counter()
            if t_first_token is None:
                t_first_token = now
            t_last_token = now
            yield _chunk(request_id, created, req.model, {"content": item}, None)
    finally:
        # If the client disconnects, Starlette cancels this generator -- but the
        # background thread above is still inside model.generate() on the GPU
        # and cannot be interrupted. Joining here (unconditionally; it's a
        # no-op if generation already finished normally) blocks the event loop
        # on purpose: releasing LOCK before the GPU is actually free would let a
        # second request start generate() on the same model instance while this
        # one is still running, which is exactly the invariant this whole file
        # exists to prevent. This is the one place a synchronous block is
        # correct rather than a bug.
        for t in gen_thread_holder:
            t.join()

    if "error" in result:
        logger.error("generation failed for %s: %s", request_id, result["error"])
        # Naive baseline: no retry, no structured error event -- just stop the
        # stream short. A scheduler that could requeue or degrade gracefully is
        # Phase 2's job, not this file's.
        return

    output_ids = result["output_ids"]
    generated_len = int(output_ids.shape[-1]) - prompt_tokens
    finish_reason = "length" if generated_len >= req.max_tokens else "stop"

    t_end = time.perf_counter()
    prefill_s = (t_first_token - t_lock_acquired) if t_first_token is not None \
        else (t_end - t_lock_acquired)
    decode_s = (t_last_token - t_first_token) \
        if (t_first_token is not None and t_last_token is not None) else 0.0
    total_s = t_end - t0
    decode_tok_s = (generated_len - 1) / decode_s if decode_s > 0 and generated_len > 1 else 0.0

    METRICS.prompt_tokens_total += prompt_tokens
    METRICS.generation_tokens_total += generated_len

    # The server's own view of where time went -- independent of bench.py's
    # client-side TTFT/ITL. When the two disagree, this file is how you tell
    # "the model was slow" apart from "the request sat in the queue" or "HTTP
    # overhead ate the difference."
    _log_request({
        "timestamp": time.time(),
        "prompt_tokens": prompt_tokens,
        "output_tokens": generated_len,
        "queue_wait_s": queue_wait_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": total_s,
        "decode_tok_s": decode_tok_s,
    })

    yield _chunk(request_id, created, req.model, {}, finish_reason)
    yield "data: [DONE]\n\n"


def _log_request(entry: dict[str, Any]) -> None:
    # Flushed every write, per request -- a long run that dies mid-sweep should
    # not lose the requests that already completed. Mirrors bench.py's own
    # per-request flush discipline on the client side.
    with open(SERVE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> StreamingResponse:
    t0 = time.perf_counter()
    if not req.stream:
        # Streaming only, deliberately -- see the module docstring. A
        # non-streaming caller gets a clear error instead of a silent 60s hang
        # waiting for the whole generation to buffer.
        raise HTTPException(
            status_code=400,
            detail="This baseline only supports stream=true. Non-streaming responses "
                   "are not implemented.",
        )

    # Tokenization is CPU-only and cheap; it happens BEFORE the lock so a bad
    # request (unknown chat_template_kwargs, malformed messages) fails fast
    # with a real 400 instead of a 200 whose body silently ends early -- once
    # StreamingResponse starts, the status code is already committed.
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

    # transformers 5.x returns a BatchEncoding (dict-like) from apply_chat_template
    # when tokenize=True; 4.x returned a bare tensor. Accept either, so this file is
    # not silently pinned to one major version of a dependency we did not choose.
    if not hasattr(input_ids, "shape"):
        input_ids = input_ids["input_ids"]

    prompt_tokens = int(input_ids.shape[-1])
    input_ids = input_ids.to(STATE.model.device)

    return StreamingResponse(
        _stream_response(req, input_ids, prompt_tokens, t0),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "gpu": STATE.gpu_name,
        "weights_gib": round(STATE.weights_gib, 2),
    }


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    # These are literally vLLM's metric names, even though this server is not
    # vLLM. infra/idle-shutdown.sh polls exactly these names to decide whether
    # the box is idle, and it has to work against this baseline in Phase 1 too
    # -- so the names are load-bearing, not cosmetic.
    body = (
        "# HELP vllm:num_requests_running Requests currently generating (this\n"
        "# baseline never runs more than one).\n"
        "# TYPE vllm:num_requests_running gauge\n"
        f"vllm:num_requests_running {METRICS.running}\n"
        "# HELP vllm:num_requests_waiting Requests queued behind the global lock.\n"
        "# TYPE vllm:num_requests_waiting gauge\n"
        f"vllm:num_requests_waiting {METRICS.waiting}\n"
        "# HELP vllm:prompt_tokens_total Cumulative prompt tokens processed.\n"
        "# TYPE vllm:prompt_tokens_total counter\n"
        f"vllm:prompt_tokens_total {METRICS.prompt_tokens_total}\n"
        "# HELP vllm:generation_tokens_total Cumulative tokens generated.\n"
        "# TYPE vllm:generation_tokens_total counter\n"
        f"vllm:generation_tokens_total {METRICS.generation_tokens_total}\n"
    )
    return PlainTextResponse(body)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)

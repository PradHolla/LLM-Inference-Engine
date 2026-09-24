#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
mock_server.py -- a fake vLLM, so bench.py can be validated with no GPU. Zero
dependencies; simulates a server with real capacity so a sweep against it produces
a genuine knee -- see NOTES/code-notes.md for what that simulation models.

  python tools/mock_server.py --port 8000
"""
from __future__ import annotations
import argparse, asyncio, json, time
import math
import re

ARGS = None
active = 0
REPLAY = []
REPLAY_CURSOR = 0
YEAR_OR_TICKER = re.compile(r"\b(?:19|20)\d{2}\b|(?<![A-Za-z])\$?[A-Z]{2,5}(?![A-Za-z])")
RECENCY = re.compile(r"\b(?:latest|today)\b", re.IGNORECASE)


def plan_json(user_text: str) -> str:
    """The 6c planner's reply: search on a year, ticker or recency word, think past 12 words."""
    search = bool(YEAR_OR_TICKER.search(user_text) or RECENCY.search(user_text))
    return json.dumps({"search": search, "queries": [user_text] if search else [],
                       "think": len(user_text.split()) > 12})


async def generate(writer: asyncio.StreamWriter, prompt_tokens: int, max_tokens: int,
                   reasoning: int = 0) -> None:
    global active

    def chunk(payload: dict) -> bytes:
        body = f"data: {json.dumps(payload)}\n\n".encode()
        return b"%x\r\n%s\r\n" % (len(body), body)

    # Prefill: compute-bound, proportional to prompt length.
    await asyncio.sleep(prompt_tokens * ARGS.prefill_ms_per_token / 1000)

    per, emitted = max(1, ARGS.tokens_per_chunk), 0
    async with SEM:                      # capacity limit -> queueing -> the knee
        active += 1
        try:
            while emitted < reasoning + max_tokens:
                # One sleep per STEP, not per token: a speculative step emits several
                # tokens for the price of one, which is what --tokens-per-chunk models.
                itl = ARGS.itl_ms * (1 + (active - 1) / ARGS.batch) / 1000
                await asyncio.sleep(itl)
                field = "reasoning" if emitted < reasoning else "content"
                limit = reasoning if field == "reasoning" else reasoning + max_tokens
                n = min(per, limit - emitted)
                text = "".join(f"tok{emitted + j} " for j in range(n))
                writer.write(chunk({"choices": [{"delta": {field: text},
                                                 "index": 0, "finish_reason": None}]}))
                await writer.drain()
                emitted += n
        finally:
            active -= 1

    writer.write(chunk({"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}))
    if ARGS.usage:
        writer.write(chunk({"choices": [], "usage": {
            "prompt_tokens": prompt_tokens, "completion_tokens": emitted,
            "total_tokens": prompt_tokens + emitted,
            "prompt_tokens_details": {"cached_tokens": ARGS.cached_tokens}}}))
    writer.write(b"%x\r\ndata: [DONE]\n\n\r\n" % len(b"data: [DONE]\n\n"))
    writer.write(b"0\r\n\r\n")
    await writer.drain()


async def generate_replay(writer: asyncio.StreamWriter, prompt_tokens: int,
                          max_tokens: int, record: dict, enabled: bool,
                          budget: int | None) -> None:
    global active

    def chunk(payload: dict) -> bytes:
        body = f"data: {json.dumps(payload)}\n\n".encode()
        return b"%x\r\n%s\r\n" % (len(body), body)

    await asyncio.sleep(prompt_tokens * ARGS.prefill_ms_per_token / 1000)
    reasoning = record.get("reasoning", "") if enabled else ""
    if not isinstance(reasoning, str):
        reasoning = ""
    if budget is not None:
        reasoning = reasoning[:max(0, budget) * 4]
    content = record.get("content", "")
    if not isinstance(content, str):
        content = ""
    content = content[:max(0, max_tokens) * 4]
    async with SEM:
        active += 1
        try:
            for field, value in (("reasoning", reasoning), ("content", content)):
                for offset in range(0, len(value), 16):
                    await asyncio.sleep(ARGS.itl_ms / 1000)
                    writer.write(chunk({"choices": [{"delta": {field: value[offset:offset + 16]},
                                                     "index": 0, "finish_reason": None}]}))
                    await writer.drain()
        finally:
            active -= 1

    prompt_count = math.ceil(prompt_tokens)
    completion_count = math.ceil(len(reasoning) / 4) + math.ceil(len(content) / 4)
    writer.write(chunk({"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}))
    if ARGS.usage:
        writer.write(chunk({"choices": [], "usage": {
            "prompt_tokens": prompt_count, "completion_tokens": completion_count,
            "total_tokens": prompt_count + completion_count,
            "prompt_tokens_details": {"cached_tokens": ARGS.cached_tokens}}}))
    writer.write(b"%x\r\ndata: [DONE]\n\n\r\n" % len(b"data: [DONE]\n\n"))
    writer.write(b"0\r\n\r\n")
    await writer.drain()


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    global REPLAY_CURSOR
    try:
        while True:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin1").split("\r\n")
            length = next((int(l.split(":", 1)[1]) for l in lines
                           if l.lower().startswith("content-length:")), 0)
            body = await reader.readexactly(length) if length else b""

            if "/metrics" in lines[0]:
                m = (f"vllm:num_requests_running{{model_name=\"mock\"}} {active}.0\n"
                     f"vllm:prompt_tokens_total{{model_name=\"mock\"}} {time.time():.0f}.0\n").encode()
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
                             b"Content-Length: %d\r\n\r\n%s" % (len(m), m))
                await writer.drain()
                continue

            req = json.loads(body or b"{}")
            prompt = "".join(m.get("content", "") for m in req.get("messages", []))
            if not req.get("stream", True):
                user_text = next((m.get("content", "") for m in reversed(req.get("messages", []))
                                  if m.get("role") == "user"), "")
                if req.get("response_format"):
                    title = plan_json(str(user_text))
                else:
                    title = " ".join(str(user_text).split()[:7]).strip(" .!?\n") or "New chat"
                payload = json.dumps({"choices": [{"message": {
                    "role": "assistant", "content": title}}]}).encode()
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                             b"Connection: close\r\nContent-Length: "
                             + str(len(payload)).encode() + b"\r\n\r\n" + payload)
                await writer.drain()
                return
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                         b"Transfer-Encoding: chunked\r\nCache-Control: no-cache\r\n\r\n")
            await writer.drain()
            # Reasoning follows vLLM's switches: enable_thinking off gives none, a
            # thinking_token_budget caps it.
            think = (req.get("chat_template_kwargs") or {}).get("enable_thinking", True)
            budget = req.get("thinking_token_budget")
            reasoning = 0 if not think else ARGS.reasoning if budget is None \
                else min(ARGS.reasoning, int(budget))
            prompt_tokens = max(1, len(prompt) // 4)
            replay_record = None
            if REPLAY:
                replay_record = REPLAY[REPLAY_CURSOR % len(REPLAY)]
                REPLAY_CURSOR += 1
                max_tokens = int(req["max_tokens"]) if req.get("max_tokens") is not None \
                    else math.ceil(len(str(replay_record.get("content", ""))) / 4)
            else:
                max_tokens = int(req.get("max_tokens", 128))
            if REPLAY:
                await generate_replay(writer, prompt_tokens, max_tokens, replay_record,
                                      bool(think), None if not think or budget is None
                                      else int(budget))
            else:
                await generate(writer, prompt_tokens, max_tokens, reasoning)
    except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        writer.close()


async def main() -> None:
    global ARGS, SEM, REPLAY
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8, help="requests that can decode at once")
    ap.add_argument("--itl-ms", type=float, default=25, help="inter-token latency at batch 1")
    ap.add_argument("--prefill-ms-per-token", type=float, default=0.25)
    ap.add_argument("--tokens-per-chunk", type=int, default=1,
                    help="tokens per SSE chunk; >1 simulates speculative decoding")
    ap.add_argument("--reasoning", type=int, default=0,
                    help="reasoning tokens emitted as delta.reasoning before the content")
    ap.add_argument("--usage", action="store_true",
                    help="emit a final usage chunk, as stream_options.include_usage does")
    ap.add_argument("--replay", help="replay real reasoning and content from an app JSONL file")
    ap.add_argument("--cached-tokens", type=int, default=0,
                    help="prompt_tokens_details.cached_tokens reported in that chunk")
    ARGS = ap.parse_args()
    if ARGS.replay:
        with open(ARGS.replay, encoding="utf-8") as source:
            REPLAY = [json.loads(line) for line in source if line.strip()]
        REPLAY = [row for row in REPLAY if isinstance(row.get("content"), str)]
        if not REPLAY:
            raise SystemExit("replay file contains no usable content records")
    SEM = asyncio.Semaphore(ARGS.batch)
    server = await asyncio.start_server(handle, "127.0.0.1", ARGS.port)
    print(f"mock vLLM on :{ARGS.port}  batch={ARGS.batch}  itl={ARGS.itl_ms}ms  "
          f"prefill={ARGS.prefill_ms_per_token}ms/tok" + (f"  replay={len(REPLAY)}" if REPLAY else ""))
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

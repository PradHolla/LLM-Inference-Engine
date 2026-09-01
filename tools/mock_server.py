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

ARGS = None
active = 0


async def generate(writer: asyncio.StreamWriter, prompt_tokens: int, max_tokens: int) -> None:
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
            while emitted < max_tokens:
                # One sleep per STEP, not per token: a speculative step emits several
                # tokens for the price of one, which is what --tokens-per-chunk models.
                itl = ARGS.itl_ms * (1 + (active - 1) / ARGS.batch) / 1000
                await asyncio.sleep(itl)
                n = min(per, max_tokens - emitted)
                text = "".join(f"tok{emitted + j} " for j in range(n))
                writer.write(chunk({"choices": [{"delta": {"content": text},
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


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
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
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                         b"Transfer-Encoding: chunked\r\nCache-Control: no-cache\r\n\r\n")
            await writer.drain()
            await generate(writer, max(1, len(prompt) // 4),
                           int(req.get("max_tokens", 128)))
    except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        writer.close()


async def main() -> None:
    global ARGS, SEM
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8, help="requests that can decode at once")
    ap.add_argument("--itl-ms", type=float, default=25, help="inter-token latency at batch 1")
    ap.add_argument("--prefill-ms-per-token", type=float, default=0.25)
    ap.add_argument("--tokens-per-chunk", type=int, default=1,
                    help="tokens per SSE chunk; >1 simulates speculative decoding")
    ap.add_argument("--usage", action="store_true",
                    help="emit a final usage chunk, as stream_options.include_usage does")
    ap.add_argument("--cached-tokens", type=int, default=0,
                    help="prompt_tokens_details.cached_tokens reported in that chunk")
    ARGS = ap.parse_args()
    SEM = asyncio.Semaphore(ARGS.batch)
    server = await asyncio.start_server(handle, "127.0.0.1", ARGS.port)
    print(f"mock vLLM on :{ARGS.port}  batch={ARGS.batch}  itl={ARGS.itl_ms}ms  "
          f"prefill={ARGS.prefill_ms_per_token}ms/tok")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

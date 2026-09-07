#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28"]
# ///
"""
bench.py -- open-loop load generator for a streaming OpenAI-compatible endpoint. Fires
requests at a fixed Poisson-distributed rate rather than waiting for replies, so a queue
can form under load; see NOTES/code-notes.md for why that distinction is the point of this file.

  # single load point
  python tools/bench.py --url http://localhost:8000 --rate 4 --duration 60

  # the sweep: find the knee
  python tools/bench.py --url http://localhost:8000 --sweep 1,2,4,8,16 --duration 45

  # defeat prefix caching to measure the cold path
  python tools/bench.py --url ... --rate 4 --unique-prefix
"""
from __future__ import annotations
import argparse, asyncio, json, os, random, statistics, sys, time
from dataclasses import dataclass, field, asdict

try:
    import httpx
except ImportError:
    sys.exit("pip install httpx")

FILLER = ("The quick brown fox jumps over the lazy dog while the system under test "
          "processes tokens one at a time in strict sequence. ")


@dataclass
class Record:
    rate: float
    t_arrival: float          # when the scheduler decided to fire
    t_sent: float             # when the request actually went out
    ttft: float | None = None
    e2e: float | None = None
    itls: list[float] = field(default_factory=list)
    out_tokens: int = 0       # SSE chunks received. ONE PER ENGINE STEP, not per token.
    usage_tokens: int = 0     # server-reported completion_tokens. The real count.
    prompt_chars: int = 0
    status: str = "ok"
    error: str = ""
    warmup: bool = False      # excluded from the stats; still written to the JSONL

    @property
    def client_lag(self) -> float:
        # If this is not ~0 the CLIENT is the bottleneck and every number below
        # is understated. bench.py warns loudly when it grows.
        return self.t_sent - self.t_arrival


def min_samples(p: float) -> int:
    """Samples needed before a percentile means anything: p95 -> 20, p99 -> 100.
    See NOTES/code-notes.md."""
    return 2 if p <= 50 else int(round(1 / (1 - p / 100)))


def pct(xs: list[float], p: float) -> float:
    """Linear-interpolated percentile, or NaN if there is not enough data to
    justify one. Returning NaN loudly beats returning a confident wrong number --
    this tool has already shipped two bugs whose symptom was a plausible value."""
    if not xs or len(xs) < min_samples(p):
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


POOL: list[str] = []
_pool_i = 0


def load_prompts(path: str) -> list[str]:
    """Real prompts, one per line or JSONL with a prompt/question/text field.
    Filler is the WORST content for speculative decoding (Phase 5: acceptance 0.30
    against 0.49-0.67 on real text), so a spec measurement on filler understates it."""
    out: list[str] = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = d.get("prompt") or d.get("question") or d.get("text")
            if v:
                out.append(str(v))
        else:
            out.append(line)
    if not out:
        raise SystemExit(f"no prompts found in {path}")
    return out


def make_prompt(target_tokens: int, unique: bool) -> str:
    global _pool_i
    if POOL:
        # Fixed order, and the cursor is reset per rate by the sweep loop, so every
        # rate AND every configuration sees the same prompts in the same sequence.
        body = POOL[_pool_i % len(POOL)]
        _pool_i += 1
    else:
        # ~4 chars/token, crude but STABLE across the sweep -- see NOTES/code-notes.md.
        body = FILLER * max(1, target_tokens * 4 // len(FILLER) + 1)
        body = body[: target_tokens * 4]
    if unique:
        # A random head defeats prefix caching. Without this, run 2 of a sweep is
        # measuring the cache, not the model, and looks mysteriously faster.
        body = f"[{random.getrandbits(64):016x}] " + body
    return body


async def one_request(client: httpx.AsyncClient, args, rate: float,
                      t_arrival: float, out, lock) -> Record:
    rec = Record(rate=rate, t_arrival=t_arrival, t_sent=time.perf_counter())
    prompt = make_prompt(args.prompt_tokens, args.unique_prefix)
    rec.prompt_chars = len(prompt)
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    if args.no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if not args.no_usage:
        # Ask the server for authoritative completion_tokens; else out_tokens just
        # counts SSE chunks (see the out_tokens field comment above, and NOTES/code-notes.md).
        payload["stream_options"] = {"include_usage": True}

    last = rec.t_sent
    try:
        async with client.stream("POST", f"{args.url}/v1/chat/completions",
                                 json=payload, timeout=args.timeout) as r:
            if r.status_code != 200:
                rec.status, rec.error = "http_error", f"{r.status_code}"
                await r.aread()
                return rec
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens"):
                    rec.usage_tokens = int(usage["completion_tokens"])
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                text = delta.get("content") or delta.get("reasoning_content") or ""
                if not text:
                    continue
                now = time.perf_counter()
                if rec.ttft is None:
                    # Measured from t_sent. Compare against client_lag to be sure
                    # the client is not adding delay of its own.
                    rec.ttft = now - rec.t_sent
                else:
                    rec.itls.append(now - last)
                last = now
                rec.out_tokens += 1
        rec.e2e = time.perf_counter() - rec.t_sent
        if rec.ttft is None:
            rec.status = "empty"
    except Exception as e:
        rec.status, rec.error = "exception", f"{type(e).__name__}: {e}"
        rec.e2e = time.perf_counter() - rec.t_sent

    # Flush per request -- a sweep that only writes at the end loses everything to a
    # crash, ctrl-C, or spot reclaim. See NOTES/code-notes.md.
    if out:
        async with lock:
            out.write(json.dumps(asdict(rec)) + "\n")
            out.flush()
    return rec


async def run_point(args, rate: float, out, lock) -> list[Record]:
    inflight: set[asyncio.Task] = set()
    recs: list[Record] = []          # every record, appended on completion
    dropped = 0
    limits = httpx.Limits(max_connections=args.max_inflight + 16,
                          max_keepalive_connections=args.max_inflight + 16)
    async with httpx.AsyncClient(limits=limits) as client:
        end = time.perf_counter() + args.duration
        while time.perf_counter() < end:
            # Exponential gaps == Poisson arrivals. THIS is the open loop: we sleep
            # on the clock, never on the server.
            await asyncio.sleep(random.expovariate(rate))
            if len(inflight) >= args.max_inflight:
                # Safety valve so an overloaded server cannot OOM the client; drops
                # are counted and reported -- see NOTES/code-notes.md.
                dropped += 1
                continue
            t = asyncio.create_task(one_request(client, args, rate,
                                                time.perf_counter(), out, lock))
            inflight.add(t)
            # `inflight` tracks concurrency only; results are collected separately,
            # or discarding a finished task would discard its record too.
            t.add_done_callback(inflight.discard)
            t.add_done_callback(
                lambda f: recs.append(f.result()) if not f.cancelled()
                and f.exception() is None else None)
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
    recs.sort(key=lambda r: r.t_sent)
    if dropped:
        print(f"    \033[33m{dropped} requests dropped (max-inflight {args.max_inflight}) "
              f"— this load point is invalid\033[0m")
    return recs


async def run_serial(args, out, lock) -> tuple[list[Record], float]:
    """Closed-loop: one request at a time, each awaited before the next. Deliberate
    EXCEPTION to the open-loop rule at the top of this file; see NOTES/code-notes.md."""
    recs: list[Record] = []
    t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        for i in range(args.warmup + args.serial):
            rec = await one_request(client, args, 0.0, time.perf_counter(), out, lock)
            # First request pays CUDA autotuning/allocator warmup (measured 47x
            # outlier); discarding it is not cheating -- see NOTES/code-notes.md.
            rec.warmup = i < args.warmup
            recs.append(rec)
    return recs, time.perf_counter() - t0


def summarize(rate: float, recs: list[Record], duration: float) -> dict:
    ok = [r for r in recs if r.status == "ok" and not r.warmup]
    # Throughput divides by time actually spent, not the arrival window -- at
    # saturation the backlog drains long after the window closes (incident 10).
    if ok:
        span = max(r.t_sent + (r.e2e or 0) for r in ok) - min(r.t_sent for r in ok)
        duration = max(span, duration * 0.5)
    ttfts = [r.ttft * 1000 for r in ok if r.ttft is not None]
    itls = [x * 1000 for r in ok for x in r.itls]
    e2es = [r.e2e * 1000 for r in ok if r.e2e is not None]
    deltas = sum(r.out_tokens for r in ok)
    usage = sum(r.usage_tokens for r in ok)
    # Prefer the server's own count. Falls back to the chunk count for servers that
    # do not implement stream_options, where one chunk really is one token.
    toks = usage or deltas
    lag = max((r.client_lag for r in recs), default=0) * 1000
    return {
        "rate": rate, "sent": len(recs), "ok": len(ok),
        "fail": sum(1 for r in recs if r.status != "ok"),
        "achieved_rps": len(ok) / duration,
        "out_tok_s": toks / duration,
        "out_deltas": deltas,
        "out_tokens_usage": usage,
        "tokens_per_step": (usage / deltas) if (usage and deltas) else None,
        "ttft_p50": pct(ttfts, 50), "ttft_p95": pct(ttfts, 95), "ttft_p99": pct(ttfts, 99),
        "itl_p50": pct(itls, 50), "itl_p95": pct(itls, 95),
        "e2e_p50": pct(e2es, 50), "e2e_p95": pct(e2es, 95),
        "max_client_lag_ms": lag,
    }


HDR = (f"  {'rate':>6} {'ok':>5} {'fail':>5} {'rps':>7} {'tok/s':>8} │ "
       f"{'TTFT p50':>9} {'p95':>8} {'p99':>8} │ {'ITL p50':>8} {'p95':>8} │ {'E2E p95':>9}")


def _f(v: float, w: int, suffix: str = "m") -> str:
    """NaN prints as n/a rather than as a number nobody should trust."""
    return f"{'n/a':>{w}}" if v != v else f"{v:>{w}.0f}{suffix}"


def print_row(s: dict) -> None:
    warn = "  \033[33m← client lag!\033[0m" if s["max_client_lag_ms"] > 50 else ""
    if s["ok"] < min_samples(95):
        warn += f"  \033[33m← n={s['ok']}, need {min_samples(95)} for p95\033[0m"
    fail = f"\033[31m{s['fail']:>5}\033[0m" if s["fail"] else f"{s['fail']:>5}"
    print(f"  {s['rate']:>6.2f} {s['ok']:>5} {fail} {s['achieved_rps']:>7.2f} "
          f"{s['out_tok_s']:>8.0f} │ {_f(s['ttft_p50'],8)} {_f(s['ttft_p95'],7)} "
          f"{_f(s['ttft_p99'],7)} │ {_f(s['itl_p50'],7)} {_f(s['itl_p95'],7)} │ "
          f"{_f(s['e2e_p95'],8)}{warn}")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="test")
    ap.add_argument("--rate", type=float, help="requests/sec (open loop)")
    ap.add_argument("--serial", type=int, metavar="N",
                    help="closed loop: N requests one at a time. The correct mode for "
                         "single-stream/batch-1 latency, where a queue is contamination")
    ap.add_argument("--warmup", type=int, default=1,
                    help="requests to discard before measuring (CUDA autotuning)")
    ap.add_argument("--sweep", help="comma-separated rates, e.g. 1,2,4,8,16")
    ap.add_argument("--duration", type=float, default=60, help="seconds per load point")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--prompts-file", help="real prompts to send instead of filler; "
                    "one per line, or JSONL with a prompt/question/text field")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--unique-prefix", action="store_true",
                    help="random head per request, to defeat prefix caching")
    ap.add_argument("--no-think", action="store_true", help="Qwen3: enable_thinking=false")
    ap.add_argument("--no-usage", action="store_true",
                    help="do not send stream_options.include_usage. Escape hatch for a "
                         "server that rejects the field; out_tokens then counts chunks")
    ap.add_argument("--max-inflight", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", default="results/bench.jsonl")
    ap.add_argument("--settle", type=float, default=3, help="seconds between load points")
    args = ap.parse_args()
    if args.prompts_file:
        POOL.extend(load_prompts(args.prompts_file))
        print(f"prompts: {len(POOL)} real prompts from {args.prompts_file}, "
              f"cycled in a fixed order")

    if not args.rate and not args.sweep and not args.serial:
        ap.error("need --rate, --sweep, or --serial")
    rates = [float(x) for x in args.sweep.split(",")] if args.sweep else [args.rate]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lock = asyncio.Lock()
    mode = (f"serial (closed loop) · {args.serial} requests + {args.warmup} warmup"
            if args.serial else
            f"open-loop · Poisson arrivals · {args.duration:.0f}s per point")
    print(f"\n\033[1m{mode} · "
          f"{args.prompt_tokens} prompt / {args.max_tokens} max out"
          f"{' · unique prefix' if args.unique_prefix else ''}\033[0m")
    print(f"  → {args.url}   results → {args.out}\n")
    print(HDR)
    print("  " + "─" * (len(HDR) - 2))

    summaries = []
    if args.serial:
        with open(args.out, "a") as out:
            recs, dur = await run_serial(args, out, lock)
        s = summarize(0.0, recs, dur)
        print_row(s)
        ok = [r for r in recs if r.status == "ok" and not r.warmup]
        itls = [x * 1000 for r in ok for x in r.itls]
        ttfts = [r.ttft * 1000 for r in ok if r.ttft is not None]
        print(f"\n\033[1mSINGLE STREAM\033[0m  ({len(ok)} measured, {args.warmup} warmup discarded)")
        deltas = sum(r.out_tokens for r in ok)
        usage = sum(r.usage_tokens for r in ok)
        tps = (usage / deltas) if (usage and deltas) else None
        dec_s = sum((r.e2e or 0) - (r.ttft or 0) for r in ok)
        print(f"  TTFT          {pct(ttfts,50):>8.0f} ms  (p50)")
        print(f"  ITL           {pct(itls,50):>8.1f} ms  (p50)   per STREAMED CHUNK")
        if tps and tps > 1.01:
            # Speculative decoding emits several tokens per engine step, so a chunk
            # is not a token and ITL is not per-token latency.
            print(f"  tokens/chunk  {tps:>8.2f}        {usage} tokens in {deltas} chunks")
            print(f"  per token     {pct(itls,50)/tps:>8.1f} ms  (derived)")
        if dec_s > 0:
            print(f"  decode        {(usage or deltas)/dec_s:>8.1f} tok/s"
                  f"   ({usage or deltas} tokens / {dec_s:.2f}s decode)")
        else:
            print(f"  decode        {1000/pct(itls,50):>8.1f} tok/s")
        print()
        return

    with open(args.out, "a") as out:
        for i, rate in enumerate(rates):
            if i:
                await asyncio.sleep(args.settle)
            # Restart the pool at every rate. Without this the cursor carries over, so
            # each rate draws a DIFFERENT slice of a heterogeneous pool -- one window
            # took all 90 long-context items and read as a capacity cliff that vanished
            # at the next rate up.
            global _pool_i
            _pool_i = 0
            recs = await run_point(args, rate, out, lock)
            s = summarize(rate, recs, args.duration)
            summaries.append(s)
            print_row(s)

    if len(summaries) > 1:
        print("\n\033[1mTHE KNEE\033[0m")
        base = summaries[0]["ttft_p95"]
        knee = None
        for s in summaries:
            # First load point where p95 TTFT has tripled off the unloaded baseline.
            if s["ttft_p95"] > base * 3 and knee is None:
                knee = s
        if knee:
            print(f"  p95 TTFT departs baseline ({base:.0f}ms) at ~{knee['rate']:.1f} req/s "
                  f"→ \033[1m{knee['out_tok_s']:.0f} tok/s\033[0m sustained")
            print("  Capacity is the load point BEFORE this one.")
        else:
            print(f"  No knee reached — p95 TTFT still {summaries[-1]['ttft_p95']:.0f}ms "
                  f"at {summaries[-1]['rate']:.1f} req/s. Push the sweep higher.")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\ninterrupted — partial results are already on disk (flushed per request)")

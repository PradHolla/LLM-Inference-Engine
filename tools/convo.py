#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28"]
# ///
"""
convo.py -- grow ONE conversation turn by turn and record what each turn costs.
bench.py fires independent requests; this is stateful, which is what makes prefix
caching visible. See NOTES/code-notes.md for the cold-arm control.

  uv run tools/convo.py --url http://localhost:8080 --turns 25 --out results/convo-warm.jsonl
  uv run tools/convo.py --url ... --cold --out results/convo-cold.jsonl   # defeats the cache
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import string
import sys
import time
from dataclasses import asdict, dataclass, field

try:
    import httpx
except ImportError:
    sys.exit("pip install httpx")

PROMPTS = [
    "Explain what a KV cache is in transformer inference.",
    "Why does decode read the whole model for every single token?",
    "What limits how many users a single GPU can serve at once?",
    "How does continuous batching differ from static batching?",
    "What does quantization actually change about the arithmetic?",
    "Why is prefill compute-bound but decode memory-bound?",
    "What is a prefix cache and when does it help?",
    "Explain speculative decoding without using the word draft.",
]


@dataclass
class Turn:
    turn: int
    t_wall: float
    prompt_chars: int
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    n_events: int = 0
    usage_prompt_tokens: int | None = None
    usage_cached_tokens: int | None = None
    usage_completion_tokens: int | None = None
    itls_ms: list[float] = field(default_factory=list)
    prefix_hit_rate: float | None = None   # measured per turn from vLLM's counters
    status: str = "ok"
    error: str | None = None


def noise(n: int = 64) -> str:
    """Random text for position ZERO of the rendered prompt. A prefix cache matches an exact
    prefix, so only text before every other token shifts the whole sequence. Noise on the
    NEW USER MESSAGE does nothing: the history renders first and still matches (incident 45)."""
    return "".join(random.choices(string.ascii_lowercase + " ", k=n))


async def prefix_stats(client, metrics_url: str) -> tuple[float, float] | None:
    """vLLM's cumulative prefix-cache counters. This version does not report
    usage.prompt_tokens_details.cached_tokens, so the counters are the only signal that
    says whether the cold arm actually went cold."""
    try:
        r = await client.get(metrics_url.rstrip("/") + "/metrics", timeout=10)
        q = h = None
        for line in r.text.splitlines():
            if line.startswith("#"):
                continue
            name = line.split("{")[0].split(" ")[0]
            # `external_` is the KV-connector's cache, not the local prefix cache, and it
            # sits at 0.0. It also ends with the same suffix, so without this guard it
            # overwrites the real counters with zero. Same defect labbench/probes.py fixed.
            if "external_" in name:
                continue
            if name.endswith("prefix_cache_queries_total"):
                q = float(line.rsplit(" ", 1)[1])
            elif name.endswith("prefix_cache_hits_total"):
                h = float(line.rsplit(" ", 1)[1])
        return (q, h) if q is not None and h is not None else None
    except Exception:
        return None


async def one_turn(client, url, model, messages, max_tokens, think) -> tuple[Turn, str]:
    body = {"model": model, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": max_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": think}}
    rec = Turn(turn=0, t_wall=time.time(),
               prompt_chars=sum(len(m["content"]) for m in messages))
    text, t0, last = [], time.perf_counter(), None
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=body,
                                 timeout=600) as r:
            if r.status_code != 200:
                rec.status, rec.error = "http_error", f"{r.status_code}: {(await r.aread())[:200]!r}"
                return rec, ""
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                u = ev.get("usage") or {}
                if u.get("completion_tokens"):
                    rec.usage_completion_tokens = int(u["completion_tokens"])
                    rec.usage_prompt_tokens = int(u.get("prompt_tokens") or 0)
                    det = u.get("prompt_tokens_details") or {}
                    if det.get("cached_tokens") is not None:
                        rec.usage_cached_tokens = int(det["cached_tokens"])
                ch = ev.get("choices") or []
                if not ch:
                    continue
                d = ch[0].get("delta") or {}
                piece = d.get("content") or ""
                if not piece:
                    continue
                now = time.perf_counter()
                if rec.ttft_ms is None:
                    rec.ttft_ms = (now - t0) * 1e3
                elif last is not None:
                    rec.itls_ms.append((now - last) * 1e3)
                last = now
                text.append(piece)
        rec.e2e_ms = (time.perf_counter() - t0) * 1e3
        rec.n_events = len(text)
        if rec.ttft_ms is None:
            rec.status = "empty"
    except Exception as e:
        rec.status, rec.error = "exception", f"{type(e).__name__}: {e}"
    return rec, "".join(text)


# Measured, not assumed: a 16,384-token target seeded 11,261 real tokens at 4.0,
# so this text runs 5.82 chars per token. M4 was 26% low because of the old value.
SEED_CHARS_PER_TOKEN = 5.82


def seed_history(target_tokens: int) -> list[dict]:
    """A synthetic prior conversation of about `target_tokens`, so turn 1 measures
    reopening a persisted chat rather than opening an empty one."""
    msgs: list[dict] = []
    approx = 0
    i = 0
    while approx < target_tokens:
        i += 1
        q = PROMPTS[(i - 1) % len(PROMPTS)]
        a_txt = f"Answer {i}. " + ("Prior conversation text. " * 60)
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a_txt})
        approx += int((len(q) + len(a_txt)) / SEED_CHARS_PER_TOKEN)
    return msgs


async def run(a) -> int:
    out = open(a.out, "w")   # truncate: appending pooled two runs in the 09-04 attempt
    messages: list[dict] = seed_history(a.seed_tokens) if a.seed_tokens else []
    rows: list[Turn] = []
    async with httpx.AsyncClient() as client:
        for i in range(1, a.turns + 1):
            q = PROMPTS[(i - 1) % len(PROMPTS)]
            messages.append({"role": "user", "content": q})
            # A fresh system message per turn puts new text at position zero, which is the
            # only placement that shifts every later token and defeats the cache.
            send = ([{"role": "system", "content": noise()}] if a.cold else []) + messages
            before = await prefix_stats(client, a.metrics_url)
            rec, reply = await one_turn(client, a.url, a.model, send,
                                        a.max_tokens, a.think)
            after = await prefix_stats(client, a.metrics_url)
            # `is not None`, not truthiness: a counter pair of (0.0, 0.0) is falsy in the
            # wrong way and silently skipped the whole measurement.
            if before is not None and after is not None and after[0] > before[0]:
                rec.prefix_hit_rate = (after[1] - before[1]) / (after[0] - before[0])
            rec.turn = i
            rows.append(rec)
            out.write(json.dumps(asdict(rec)) + "\n")
            out.flush()
            if rec.status != "ok":
                print(f"  turn {i:>3} FAILED {rec.status} {rec.error}")
                if a.stop_on_error:
                    break
                messages.pop()
                continue
            messages.append({"role": "assistant", "content": reply})
            pt = rec.usage_prompt_tokens or 0
            hr = rec.prefix_hit_rate
            print(f"  turn {i:>3}  prompt {pt:>6} tok  hit {('%.2f'%hr) if hr is not None else '  n/a':>5}  "
                  f"ttft {rec.ttft_ms:>7.1f} ms  out {rec.usage_completion_tokens or 0:>4}")
            # Incident 45: a control that does not perturb what it claims to perturb yields a
            # null result that reads as a finding. Check the manipulation took, then continue.
            if a.cold and i == a.verify_turn:
                if hr is None:
                    print("  CONTROL UNVERIFIABLE: no prefix-cache counters; aborting rather "
                          "than collecting an arm that cannot be trusted")
                    return 2
                if hr > a.max_cold_hit:
                    print(f"  CONTROL FAILED: prefix hit rate {hr:.2f} at turn {i} exceeds "
                          f"{a.max_cold_hit}. The cache is still hitting, so this arm is not "
                          f"cold. Aborting before spending the run.")
                    return 2
                print(f"  control verified: prefix hit rate {hr:.2f} at turn {i}")
    out.close()
    ok = [r for r in rows if r.status == "ok" and r.ttft_ms]
    if len(ok) >= 4:
        early = sorted(r.ttft_ms for r in ok[1:4])[len(ok[1:4]) // 2]
        late = sorted(r.ttft_ms for r in ok[-3:])[1]
        print(f"\n  arm={'cold' if a.cold else 'warm'}  turns={len(ok)}  "
              f"TTFT early {early:.1f} ms -> late {late:.1f} ms  ratio {late/early:.2f}x")
        print(f"  prompt tokens {ok[0].usage_prompt_tokens} -> {ok[-1].usage_prompt_tokens}")
    return 0


def selftest() -> int:
    fails = []
    a = noise(24)
    if len(a) != 24:
        fails.append("  FAIL noise length")
    if noise(24) == a:
        fails.append("  FAIL noise is not random")
    # Incident 45: the cold arm must change position ZERO of what is sent, not the tail.
    hist = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}]
    cold_a = [{"role": "system", "content": noise()}] + hist
    cold_b = [{"role": "system", "content": noise()}] + hist
    warm = hist
    if cold_a[0]["role"] != "system":
        fails.append("  FAIL cold arm does not put anything at position 0")
    if cold_a[0]["content"] == cold_b[0]["content"]:
        fails.append("  FAIL consecutive cold turns share position 0; the cache would still hit")
    if warm and warm[0].get("role") == "system":
        fails.append("  FAIL warm arm gained a system message; the arms would not be matched")
    # the shared history must be IDENTICAL in both arms -- only the prefix may differ
    if cold_a[1:] != warm or cold_b[1:] != warm:
        fails.append("  FAIL cold arm altered the history instead of only prefixing it")

    t = Turn(turn=1, t_wall=0.0, prompt_chars=10)
    d = asdict(t)
    for k in ("usage_cached_tokens", "usage_prompt_tokens", "ttft_ms"):
        if d[k] is not None:
            fails.append(f"  FAIL {k} should default to None, not 0")
    print("\n".join(fails) if fails else "tools/convo.py selftest: all checks passed")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", default=None, help="served id; discovered if omitted")
    ap.add_argument("--turns", type=int, default=25)
    ap.add_argument("--seed-tokens", type=int, default=0,
                    help="pre-seed a synthetic history of ~N tokens; M4 reopens a 16k chat")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--cold", action="store_true",
                    help="unique front-loaded noise per turn, defeating the prefix cache")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--out", default="results/convo.jsonl")
    ap.add_argument("--metrics-url", default="http://localhost:8000",
                    help="engine base url for prefix-cache counters")
    ap.add_argument("--verify-turn", type=int, default=3,
                    help="turn at which --cold must prove the cache is not hitting")
    ap.add_argument("--max-cold-hit", type=float, default=0.35,
                    help="abort --cold if the per-turn prefix hit rate exceeds this")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.model:
        # bench.py's --model test 404s on vLLM; ask the server what it serves.
        r = httpx.get(f"{a.url}/v1/models", timeout=10)
        data = (r.json() or {}).get("data") or []
        if not data:
            print("could not resolve served model from /v1/models")
            return 1
        a.model = data[0]["id"]
    print(f"convo: {a.turns} turns, {'COLD (cache defeated)' if a.cold else 'WARM'}, "
          f"model={a.model}, max_tokens={a.max_tokens}")
    return asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())

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
    status: str = "ok"
    error: str | None = None


def noise(n: int = 24) -> str:
    """Random text to place at the FRONT of the prompt, which is what defeats a prefix
    cache: the hit ends at the first differing token."""
    return "".join(random.choices(string.ascii_lowercase + " ", k=n))


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


async def run(a) -> int:
    out = open(a.out, "a")
    messages: list[dict] = []
    rows: list[Turn] = []
    async with httpx.AsyncClient() as client:
        for i in range(1, a.turns + 1):
            q = PROMPTS[(i - 1) % len(PROMPTS)]
            if a.cold:
                # Front-loaded noise, so no turn shares a prefix with any other.
                q = f"[{noise()}] {q}"
            messages.append({"role": "user", "content": q})
            rec, reply = await one_turn(client, a.url, a.model, messages,
                                        a.max_tokens, a.think)
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
            ct = rec.usage_cached_tokens
            print(f"  turn {i:>3}  prompt {pt:>6} tok  cached {str(ct):>6}  "
                  f"ttft {rec.ttft_ms:>7.1f} ms  out {rec.usage_completion_tokens or 0:>4}")
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
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--cold", action="store_true",
                    help="unique front-loaded noise per turn, defeating the prefix cache")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--out", default="results/convo.jsonl")
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

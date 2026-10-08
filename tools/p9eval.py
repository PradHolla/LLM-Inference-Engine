#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28", "langgraph==1.2.12", "tokenizers>=0.20"]
# ///
"""Phase 9: perplexity vs real quantization damage, and what a JSON schema costs the planner.

  uv run tools/p9eval.py ppl  --url http://localhost:8000 --label int4 --corpus k32 --out results/p9-ppl.jsonl
  uv run tools/p9eval.py json --url http://localhost:8000 --n 100 --out results/p9-json.jsonl
  uv run tools/p9eval.py report
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app import agent, prompts  # noqa: E402

THINK_OFF_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def served_model(client: httpx.Client, url: str) -> str:
    return client.get(f"{url}/v1/models").json()["data"][0]["id"]


def wikitext_chunks(path: str, n: int, chars: int) -> list[tuple[str, str, int]]:
    """(id, text, score_from_token): fixed character chunks, every token scored after the first."""
    text = Path(path).read_text()
    out, i = [], 0
    while len(out) < n and i < len(text):
        chunk = text[i:i + chars]
        if chunk.strip():
            out.append((f"wiki-{len(out):03d}", chunk, 1))
        i += chars
    return out


def k32_texts(results: str, items: str) -> list[tuple[str, str, str]]:
    """bf16's CORRECT no-think 32-step answers: (id, rendered prefix, answer). Only the answer is scored."""
    prompt = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(items)}
    out = []
    for line in open(results):
        r = json.loads(line)
        if str(r.get("k")) == "32" and not r.get("thinking") and r.get("correct") and r.get("text"):
            prefix = f"<|im_start|>user\n{prompt[r['id']]}<|im_end|>\n{THINK_OFF_PREFIX}"
            out.append((r["id"], prefix, r["text"]))
    return out


def tokenize(client: httpx.Client, url: str, model: str, text: str) -> list[int]:
    r = client.post(f"{url}/tokenize", json={"model": model, "prompt": text,
                                             "add_special_tokens": False})
    r.raise_for_status()
    return r.json()["tokens"]


def score(client: httpx.Client, url: str, model: str, text: str) -> list[tuple[float, int]]:
    """(logprob, rank) of every prompt token after the first, from vLLM's prompt_logprobs."""
    r = client.post(f"{url}/v1/completions", json={
        "model": model, "prompt": text, "max_tokens": 1, "temperature": 0.0,
        "prompt_logprobs": 1, "add_special_tokens": False})
    r.raise_for_status()
    plp = r.json()["choices"][0]["prompt_logprobs"]
    toks = tokenize(client, url, model, text)
    out = []
    for pos, entry in enumerate(plp):
        if entry is None:
            continue
        actual = entry.get(str(toks[pos])) or entry.get(toks[pos])
        out.append((actual["logprob"], actual.get("rank", 0)))
    return out


def run_ppl(a: argparse.Namespace) -> int:
    with httpx.Client(timeout=600.0) as client:
        model = served_model(client, a.url)
        if a.corpus == "wiki":
            units = [(cid, text, 1, None) for cid, text, _ in wikitext_chunks(a.wiki, a.n, a.chars)]
        else:
            units = []
            for cid, prefix, answer in k32_texts(a.results, a.items)[:a.n]:
                n_prefix = len(tokenize(client, a.url, model, prefix))
                full = tokenize(client, a.url, model, prefix + answer)
                if full[:n_prefix] != tokenize(client, a.url, model, prefix):
                    print(f"  skip {cid}: prefix tokens do not survive concatenation", flush=True)
                    continue
                units.append((cid, prefix + answer, n_prefix, None))
        for cid, text, start, _ in units:
            scored = score(client, a.url, model, text)
            kept = scored[start - 1:]          # entry k scores token k; entry 0 is None and dropped
            rec = {"label": a.label, "corpus": a.corpus, "id": cid, "tokens": len(kept),
                   "nll_sum": -sum(lp for lp, _ in kept),
                   "top1": sum(1 for _, rank in kept if rank == 1)}
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"  {a.label} {a.corpus}: {len(units)} units scored", flush=True)
    return 0


def planner_messages(item: dict, today: str) -> list[dict]:
    base = [{"role": "system", "content": prompts.system_prompt(today)},
            *({"role": m["role"], "content": m["content"]} for m in item["history"])]
    return agent.plan_messages(base, item["user"])


def one_json(client: httpx.Client, url: str, model: str, messages: list[dict],
             schema: bool) -> dict:
    body = {"model": model, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": agent.config.PLAN_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False}, **agent.sampling(False)}
    if schema:
        body["response_format"] = agent.RESPONSE_FORMAT
    t0 = time.perf_counter()
    first, n_chunks, usage, text = None, 0, {}, ""
    with client.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning") or "")
                if piece:
                    first = first if first is not None else time.perf_counter()
                    n_chunks += 1
                    text += piece
    end = time.perf_counter()
    out_tokens = usage.get("completion_tokens") or 0
    ttft = (first - t0) * 1000 if first else None
    tpot = ((end - first) * 1000 / (out_tokens - 1)) if first and out_tokens > 1 else None
    try:
        parses = isinstance(json.loads(text), dict)
    except ValueError:
        parses = False
    return {"schema": schema, "ttft_ms": ttft, "e2e_ms": (end - t0) * 1000, "tpot_ms": tpot,
            "out_tokens": out_tokens, "prompt_tokens": usage.get("prompt_tokens"),
            "parses_as_json": parses}


def run_json(a: argparse.Namespace) -> int:
    today = datetime.date.today().isoformat()
    items = [json.loads(l) for l in open(a.items)][:a.n]
    with httpx.Client(timeout=120.0) as client:
        model = served_model(client, a.url)
        # The first schema request pays any grammar compilation; measure it on its own.
        warm = one_json(client, a.url, model, planner_messages(items[0], today), False)
        cold = one_json(client, a.url, model, planner_messages(items[0], today), True)
        with open(a.out, "a") as f:
            f.write(json.dumps({"phase": "first", "plain": warm, "schema": cold}) + "\n")
        for i, item in enumerate(items):
            order = (True, False) if i % 2 == 0 else (False, True)
            for schema in order:
                rec = one_json(client, a.url, model, planner_messages(item, today), schema)
                rec.update(phase="pair", id=item["id"])
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        print(f"  json: {len(items)} pairs", flush=True)
    return 0


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def report(a: argparse.Namespace) -> int:
    lines = []
    if Path(a.ppl).exists():
        rows = [json.loads(l) for l in open(a.ppl)]
        lines.append("## Perplexity vs generation damage  [Qwen3-8B, A10G, vLLM 0.27.1, prompt_logprobs, "
                     "bf16 KV; k32 = bf16's correct no-think 32-step answers, answer tokens only]")
        lines.append("corpus | config | units | tokens | perplexity | vs bf16 | top-1 match")
        for corpus in ("wiki", "k32"):
            base = None
            for label in ("bf16", "fp8", "int4"):
                rs = [r for r in rows if r["corpus"] == corpus and r["label"] == label]
                if not rs:
                    continue
                tok = sum(r["tokens"] for r in rs)
                ppl = math.exp(sum(r["nll_sum"] for r in rs) / tok)
                base = base or ppl
                top1 = sum(r["top1"] for r in rs) / tok
                lines.append(f"{corpus} | {label} | {len(rs)} | {tok:,} | {ppl:.4f} | "
                             f"{(ppl / base - 1) * 100:+.2f}% | {top1 * 100:.2f}%")
    if Path(a.json_out).exists():
        rows = [json.loads(l) for l in open(a.json_out)]
        first = next((r for r in rows if r.get("phase") == "first"), None)
        pairs = [r for r in rows if r.get("phase") == "pair"]
        lines.append("\n## JSON schema cost  [Qwen3-8B fp8, fp8 KV, prefix caching OFF, thinking off, "
                     "concurrency 1, planner prompts]")
        lines.append("arm | n | ttft p50 | time per token p50 | out tokens p50 | parses as JSON")
        for schema in (False, True):
            rs = [r for r in pairs if r["schema"] == schema]
            lines.append(f"{'schema' if schema else 'plain'} | {len(rs)} | {med([r['ttft_ms'] for r in rs]):.1f} ms | "
                         f"{med([r['tpot_ms'] for r in rs]):.2f} ms | {med([r['out_tokens'] for r in rs])} | "
                         f"{sum(r['parses_as_json'] for r in rs)}/{len(rs)}")
        if first:
            lines.append(f"first schema request ttft {first['schema']['ttft_ms']:.1f} ms vs plain "
                         f"{first['plain']['ttft_ms']:.1f} ms (grammar compile, cold)")
    text = "\n".join(lines)
    print(text)
    Path(a.report).write_text(text + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ppl")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--label", required=True)
    p.add_argument("--corpus", choices=["wiki", "k32"], required=True)
    p.add_argument("--wiki", default="data/wikitext2-test.txt")
    p.add_argument("--results", default="results/phase4-bf16-a.jsonl")
    p.add_argument("--items", default="results/phase4-items.jsonl")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--chars", type=int, default=4000)
    p.add_argument("--out", default="results/p9-ppl.jsonl")
    j = sub.add_parser("json")
    j.add_argument("--url", default="http://localhost:8000")
    j.add_argument("--items", default="data/plansets/all.jsonl")
    j.add_argument("--n", type=int, default=100)
    j.add_argument("--out", default="results/p9-json.jsonl")
    r = sub.add_parser("report")
    r.add_argument("--ppl", default="results/p9-ppl.jsonl")
    r.add_argument("--json-out", default="results/p9-json.jsonl")
    r.add_argument("--report", default="results/p9-report.txt")
    a = ap.parse_args()
    return {"ppl": run_ppl, "json": run_json, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

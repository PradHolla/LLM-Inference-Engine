"""
server.py -- the lab bench API: probes, backend switching, load generation, static UI.
Adds routes to proxy.py's app, so /v1/chat/completions stays the same passthrough
bench.py is gated against. See NOTES/code-notes.md for the endpoint contract.

  uvicorn labbench.server:app --host 127.0.0.1 --port 8080
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import httpx
from fastapi import Body
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import backends, probes
from .proxy import RECENT, UPSTREAM, app

REPO = Path(__file__).resolve().parent.parent
UI_DIR = Path(__file__).resolve().parent / "ui"
SCRATCH = os.environ.get("LABBENCH_SCRATCH", "/tmp/labbench")

# Measured on this hardware, not assumed: the bf16 batch-1 prefill fit over
# 1,611-6,407 tokens (results/phase2-prefill-*.jsonl), and vLLM's edge over engine/manual.py.
PREFILL_MS_PER_TOKEN = 0.3093
VLLM_PREFILL_SPEEDUP = 1.21

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ROOFLINE_PATTERNS = {
    "kv_gib": r"= KV cache room\s+([\d.]+) GiB",
    "kv_tokens": r"max tokens in flight\s+([\d,]+)",
    "concurrent": r"([\d.]+)\s+concurrent requests at",
    "tok_s": r"realistic tok/s\s+([\d.]+)",
    "floor_ms_per_token": r"floor\s+ms/token\s+([\d.]+)",
}

SWITCHER = backends.Switcher()
# vLLM's prefix counters are cumulative for the server's life, so a lifetime ratio barely
# moves and shows nothing. The panel wants the rate since the last scrape.
_PREFIX_LAST: dict[str, float | None] = {"q": None, "h": None}
BOMBARD: dict = {"job": None, "running": False, "stdout": "", "returncode": None}


@app.get("/")
async def root():
    return RedirectResponse("/ui/")


@app.get("/labbench/state")
async def state():
    await SWITCHER.refresh()
    snap = SWITCHER.state.snapshot()
    snap["engine"] = await asyncio.to_thread(probes.engine_metrics, UPSTREAM)
    snap["gpu"] = await asyncio.to_thread(probes.gpu)
    sm = await asyncio.to_thread(probes.served_model, UPSTREAM)
    snap["config"]["served_model"] = sm["id"]
    snap["config"]["served_model_error"] = sm["error"]
    snap["engine"]["values"]["prefix_hit_rate"] = _windowed_prefix_rate(snap["engine"]["values"])
    return snap


def _windowed_prefix_rate(v: dict) -> float | None:
    """Hit rate since the previous scrape, falling back to lifetime on the first one."""
    q, h = v.get("prefix_queries"), v.get("prefix_hits")
    if q is None or h is None:
        return None
    lq, lh = _PREFIX_LAST["q"], _PREFIX_LAST["h"]
    _PREFIX_LAST.update(q=q, h=h)
    if lq is None or q <= lq:
        return v.get("prefix_hit_rate_lifetime")
    return (h - lh) / (q - lq)


@app.get("/labbench/journal")
async def journal(lines: int = 200):
    active = SWITCHER.state.active
    unit = backends.UNITS.get(active) if active else None
    if not unit:
        return {"unit": None, "lines": [], "error": "no backend active"}
    return await asyncio.to_thread(probes.journal, unit, max(1, min(lines, 5000)))


@app.get("/labbench/metrics/raw")
async def metrics_raw():
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(UPSTREAM.rstrip("/") + "/metrics")
        return {"text": r.text, "error": None}
    except Exception as e:
        return {"text": "", "error": f"{type(e).__name__}: {e}"}


_TOK = None


def _render(messages: list, enable_thinking: bool) -> tuple[str, int]:
    """Apply the model's own chat template locally. All Qwen3-8B variants share one
    tokenizer, so this matches whatever the engine renders."""
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(backends.MODEL)
    # tokenize=False returns a string; transformers 5 returns BatchEncoding otherwise.
    prompt = _TOK.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=enable_thinking)
    return prompt, len(_TOK(prompt, add_special_tokens=False)["input_ids"])


@app.post("/labbench/render")
async def render(body: dict = Body(...)):
    """The exact string the engine will prefill, rendered here rather than asked for."""
    msgs = body.get("messages") or []
    think = body.get("enable_thinking")
    try:
        prompt, n = await asyncio.to_thread(_render, msgs, True if think is None else bool(think))
        return {"prompt": prompt, "n_tokens": n, "error": None}
    except Exception as e:
        return {"prompt": "", "n_tokens": 0, "error": f"{type(e).__name__}: {e}"}


@app.post("/labbench/backend")
async def switch(body: dict = Body(...)):
    return SWITCHER.request(str(body.get("backend", "")), body.get("quantization"))


@app.get("/labbench/prediction")
async def prediction(context: int = 4096):
    quant = SWITCHER.state.quant
    dtype = {"bf16": "bf16", "fp8": "fp8", "int4": "int4"}.get(quant, "bf16")
    rc, out = await asyncio.to_thread(
        probes._run, ["uv", "run", "tools/roofline.py", "--model", "qwen3-8b",
                      "--gpu", "a10g", "--dtype", dtype, "--context", str(context)], 60.0)
    text = ANSI.sub("", out)
    if rc != 0:
        return {"values": {}, "text": text, "error": f"roofline exited {rc}"}
    vals: dict[str, float] = {}
    for key, pat in ROOFLINE_PATTERNS.items():
        m = re.search(pat, text)
        if m and m.groups():
            vals[key] = float(m.group(1).replace(",", ""))
    if vals.get("tok_s"):
        vals["itl_ms"] = 1000.0 / vals["tok_s"]
        vals["ttft_ms"] = (context * PREFILL_MS_PER_TOKEN / VLLM_PREFILL_SPEEDUP
                           + vals["itl_ms"])
    return {"values": vals, "text": text, "error": None,
            "note": "ttft_ms uses the measured bf16 prefill curve; fp8 prefill is unverified"}


@app.post("/labbench/bombard")
async def bombard_start(body: dict = Body(...)):
    if BOMBARD["running"]:
        return {"job": BOMBARD["job"], "error": "a job is already running"}
    n = max(1, min(int(body.get("n", 8)), 128))
    pt = max(1, min(int(body.get("prompt_tokens", 512)), 32768))
    mt = max(1, min(int(body.get("max_tokens", 64)), 4096))
    os.makedirs(SCRATCH, exist_ok=True)
    job = uuid.uuid4().hex[:8]
    # Straight at the engine, not through this proxy: the numbers must stay comparable
    # with everything already in results/.
    sm = await asyncio.to_thread(probes.served_model, UPSTREAM)
    argv = ["uv", "run", "tools/bench.py", "--url", UPSTREAM, "--serial", str(n),
            "--warmup", "1", "--prompt-tokens", str(pt), "--max-tokens", str(mt),
            "--out", f"{SCRATCH}/bombard-{job}.jsonl"]
    # bench.py defaults to --model test, which vLLM 404s. Runbook section 2 and its
    # troubleshooting table both say so; discovering it beats remembering it.
    if sm["id"]:
        argv += ["--model", sm["id"]]
    BOMBARD.update(job=job, running=True, stdout=f"$ {' '.join(argv)}\n", returncode=None)
    asyncio.create_task(_run_bombard(argv))
    return {"job": job, "error": None}


async def _run_bombard(argv: list[str]) -> None:
    try:
        p = await asyncio.create_subprocess_exec(
            *argv, cwd=str(REPO), stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT)
        assert p.stdout is not None
        async for line in p.stdout:
            BOMBARD["stdout"] += ANSI.sub("", line.decode("utf-8", "replace"))
        BOMBARD["returncode"] = await p.wait()
    except Exception as e:
        BOMBARD["stdout"] += f"\n{type(e).__name__}: {e}\n"
        BOMBARD["returncode"] = -1
    finally:
        BOMBARD["running"] = False


@app.get("/labbench/bombard")
async def bombard_status():
    return dict(BOMBARD)


if UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


def selftest() -> int:
    """Contract checks that need no server: route table and roofline parsing."""
    fails: list[str] = []

    def chk(name, got, want):
        if got != want:
            fails.append(f"  FAIL {name}: got {got!r}, want {want!r}")

    paths = {getattr(r, "path", None) for r in app.routes}
    for p in ("/v1/chat/completions", "/labbench/state", "/labbench/traces",
              "/labbench/journal", "/labbench/metrics/raw", "/labbench/render",
              "/labbench/backend", "/labbench/prediction", "/labbench/bombard"):
        if p not in paths:
            fails.append(f"  FAIL route missing: {p}")

    sample = """
MEMORY BUDGET
  = KV cache room           11.23 GiB
CAPACITY
  max tokens in flight     81,741       <- across ALL users combined
DECODE ROOFLINE
  floor  ms/token           13.65       = 8.2 GB / 600 GB/s
  realistic tok/s            47.6       (at 65% of peak bandwidth)
"""
    vals = {}
    for key, pat in ROOFLINE_PATTERNS.items():
        m = re.search(pat, sample)
        if m and m.groups():
            vals[key] = float(m.group(1).replace(",", ""))
    chk("parsed kv_gib", vals.get("kv_gib"), 11.23)
    chk("parsed kv_tokens (comma stripped)", vals.get("kv_tokens"), 81741.0)
    chk("parsed tok_s", vals.get("tok_s"), 47.6)
    chk("parsed floor", vals.get("floor_ms_per_token"), 13.65)
    itl = 1000.0 / vals["tok_s"]
    chk("derived itl_ms", round(itl, 2), 21.01)
    chk("derived ttft_ms at 16384",
        round(16384 * PREFILL_MS_PER_TOKEN / VLLM_PREFILL_SPEEDUP + itl, 1), 4209.1)

    chk("ansi stripped", ANSI.sub("", "\x1b[1mbold\x1b[0m"), "bold")

    _PREFIX_LAST.update(q=None, h=None)
    base = {"prefix_queries": 1000.0, "prefix_hits": 800.0, "prefix_hit_rate_lifetime": 0.8}
    chk("first scrape falls back to lifetime", _windowed_prefix_rate(dict(base)), 0.8)
    # 100 new queries, 90 of them hits -> the window is 0.9 even though lifetime is 0.81
    nxt = {"prefix_queries": 1100.0, "prefix_hits": 890.0, "prefix_hit_rate_lifetime": 0.809}
    chk("second scrape uses the window", round(_windowed_prefix_rate(nxt), 4), 0.9)
    chk("no new queries falls back rather than dividing by zero",
        _windowed_prefix_rate(dict(nxt)), 0.809)
    chk("missing counters -> None", _windowed_prefix_rate({}), None)
    _PREFIX_LAST.update(q=None, h=None)

    # render(): verify the call contract with a stub, since the real tokenizer is on the box.
    global _TOK
    seen = {}

    class StubTok:
        def apply_chat_template(self, msgs, **kw):
            seen.update(kw)
            return "<|im_start|>user\nhi<|im_end|>\n"

        def __call__(self, text, **kw):
            seen["encode_kw"] = kw
            return {"input_ids": list(range(len(text.split())))}

    _TOK = StubTok()
    prompt, n = _render([{"role": "user", "content": "hi"}], False)
    chk("render returns a string", isinstance(prompt, str), True)
    chk("tokenize=False passed (else transformers 5 returns BatchEncoding)",
        seen.get("tokenize"), False)
    chk("add_generation_prompt passed", seen.get("add_generation_prompt"), True)
    chk("enable_thinking forwarded", seen.get("enable_thinking"), False)
    chk("count excludes special tokens", seen["encode_kw"].get("add_special_tokens"), False)
    chk("n_tokens from encoding, not from the string", n, 2)
    _TOK = None
    chk("empty roofline yields no values",
        [k for k, p in ROOFLINE_PATTERNS.items()
         if (m := re.search(p, "")) and m.groups()], [])

    print("\n".join(fails) if fails else "labbench/server.py selftest: all checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(selftest())

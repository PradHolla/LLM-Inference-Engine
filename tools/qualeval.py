#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28"]
# ///
"""
qualeval.py -- the Phase 4 quality instrument. Protocol: NOTES/phase4-eval-design.md.

Deliberately NOT part of bench.py. bench.py is a load generator whose record schema every
Phase 1-3 result and tools/curve.py depend on; adding full completion text would bloat it by
kilobytes per request and break that schema. Same wire protocol, different job.

Three modes, and the split is the point:

  run       talk to a server, save EVERYTHING, grade nothing that cannot be regraded
  grade     re-score a saved run offline, no GPU, because a grading bug is inevitable
  compare   paired McNemar between two runs -- the actual statistical test

WHAT THIS INSTRUMENT CAN GET WRONG, all of which produce plausible numbers not errors:

  * SAMPLING SILENTLY ON. Qwen3's generation_config.json sets temperature 0.6 / top_p 0.95.
    If the request's temperature=0 does not override it, outputs are random, the bf16-vs-bf16
    noise floor swamps every effect, and the phase measures nothing. --check-determinism
    sends one item twice and diffs. Run it before every session.

  * BATCH SIZE NOT WHAT WAS PINNED. Quality depends on batch composition (incident 22), so
    every configuration must decode at the same batch size. The design pins --max-num-seqs
    server-side; this tool POLLS /metrics and records the achieved vllm:num_requests_running
    rather than trusting the flag took.

  * ANSWER EXTRACTED FROM THE THINKING BLOCK. A model writes "ANSWER: 42", reconsiders, and
    ends at 37. Taking the first match grades the abandoned answer. Last match, post-thinking
    content only.

  * A LOOSE FALLBACK. If the format is missing, this tool does NOT hunt for the last integer
    in the reply. A fallback that fires more often for one configuration applies a different
    grading standard to each. Unparseable is its own outcome and is reported, never healed.

  * TRUNCATION READ AS A WRONG ANSWER. finish_reason=="length" is recorded per item and
    reported beside accuracy, never folded into it.

  uv run tools/qualeval.py --check-determinism --url http://IP:8000
  uv run tools/qualeval.py run --url http://IP:8000 --config bf16-a --slices math,gsm8k
  uv run tools/qualeval.py grade results/phase4-bf16-a.jsonl
  uv run tools/qualeval.py compare results/phase4-bf16-a.jsonl results/phase4-fp8.jsonl
"""
import argparse, asyncio, json, math, os, random, re, sys, time
from dataclasses import dataclass, field, asdict

import httpx

MODEL = "Qwen/Qwen3-8B"

# (slice, thinking, default max_tokens). max_tokens for the thinking passes is a PLACEHOLDER
# until calibration measures the p99 of the thinking-token distribution -- see design 3d.
PASSES = [
    ("math",    True,  2048),
    ("math",    False, 256),
    ("gsm8k",   True,  1024),
    ("gsm8k",   False, 256),
    ("longctx", False, 64),
]

# [ \\t]* NOT \\s* around the colon, and [^\\n]+ not [^\\n]*. \\s crosses newlines, so the
# case-insensitive "Answer:" inside a "### Final Answer:" heading swallowed the line break
# and captured the WHOLE NEXT LINE ("ANSWER: 72") as the answer. normalize then rejected it
# and the item scored unparseable with a correct answer sitting one line below. Requiring at
# least one character on the SAME line makes the heading match nothing and the real marker win.
ANSWER_RE = re.compile(r"ANSWER[ \t]*:[ \t]*([^\n]+)", re.IGNORECASE)
# Qwen3 is heavily trained to close a maths answer with \boxed{}, and it does so even when
# the prompt demands "ANSWER: <integer>". Calibration measured 16% of COMPLETED, CORRECT
# responses ending in \boxed{N} with no ANSWER: line -- graded wrong by an earlier version of
# this file, with the model at 100% on everything it did state.
#
# This is NOT the loose fallback the design forbids. \boxed{} is an explicit, unambiguous
# answer declaration, structurally identical to ANSWER:, and accepting it makes grading MORE
# uniform across configurations rather than less. Hunting for a bare integer in prose remains
# forbidden. The marker that matched is recorded per item, so a configuration that changes
# convention is visible instead of silently mis-scored.
BOXED_RE = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")   # one nested level, for \\boxed{\\text{42}}
THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)


# --------------------------------------------------------------------------- grading

def post_thinking(text: str) -> str:
    """Everything after the last </think>. Unchanged if there is no thinking block."""
    hits = list(THINK_CLOSE.finditer(text))
    return text[hits[-1].end():] if hits else text


def extract(text: str) -> tuple[str | None, str | None]:
    """Last explicit answer marker in post-thinking content, and which marker it was.

    Both markers are scanned and the one appearing LATEST wins, because that is the model's
    final stated answer regardless of which convention it reached for.
    """
    body = post_thinking(text)
    hits = [(m.start(), m.group(1), "answer") for m in ANSWER_RE.finditer(body)]
    hits += [(m.start(), m.group(1), "boxed") for m in BOXED_RE.finditer(body)]
    if not hits:
        return None, None
    _, raw, marker = max(hits, key=lambda h: h[0])
    return raw.strip(), marker


def normalize(raw: str | None, slice_: str) -> str | None:
    """Strip pure FORMATTING. Never search for a value that was not offered as the answer.

    Permitted: markdown bold, currency, thousands separators, trailing punctuation, case.
    Not permitted: pulling an integer out of prose. That is the line between normalising a
    format and inventing an answer, and crossing it grades configurations differently.
    """
    if raw is None:
        return None
    s = raw.strip()
    s = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", s)   # \boxed{\text{18}}
    s = s.strip().strip("*").strip().strip("$").strip().rstrip(".").strip()
    if slice_ == "longctx":
        s = re.sub(r"[^A-Za-z0-9]", "", s).upper()
        return s or None
    s = s.replace(",", "").replace("$", "").replace(" ", "")
    m = re.fullmatch(r"(-?\d+)(?:\.0+)?", s)     # accept 18.0 as 18, reject 18.5
    return m.group(1) if m else None


def grade(rec: dict) -> dict:
    """Pure function of a saved record. Re-runnable offline; that is why text is saved."""
    raw, marker = extract(rec.get("text") or "")
    got = normalize(raw, rec["slice"])
    want = normalize(rec["answer"], rec["slice"])
    rec["extracted_raw"] = raw
    rec["marker"] = marker
    rec["extracted"] = got
    rec["parsed"] = got is not None
    rec["correct"] = (got == want) if got is not None else False
    rec["truncated"] = rec.get("finish_reason") == "length"
    return rec


# --------------------------------------------------------------------------- one request

@dataclass
class Rec:
    config: str = ""
    slice: str = ""
    id: str = ""
    thinking: bool = False
    k: int | None = None
    depth: float | None = None
    answer: str = ""
    prompt_chars: int = 0
    text: str = ""
    reasoning_chars: int = 0
    content_chars: int = 0
    think_path: str = "none"
    n_deltas: int = 0
    usage_completion: int | None = None
    usage_prompt: int | None = None
    finish_reason: str | None = None
    ttft: float | None = None
    e2e: float | None = None
    status: str = "ok"
    error: str | None = None
    attempts: int = 1


async def one(client, url, item, cfg, think, max_tokens, seed, attempts=3):
    """Retry transient transport errors. A dropped connection must not become a missing
    item: compare() joins on id, so lost items silently shrink the denominator, and they
    do not go missing at the same rate for every configuration. That is a biased sample,
    which is worse than a slow run."""
    for a in range(attempts):
        r = await _one(client, url, item, cfg, think, max_tokens, seed)
        if r.status != "exception":
            r.attempts = a + 1
            return r
        if a < attempts - 1:
            await asyncio.sleep(1.5 * (a + 1))
    r.attempts = attempts
    return r


async def _one(client, url, item, cfg, think, max_tokens, seed):
    r = Rec(config=cfg, slice=item["slice"], id=item["id"], thinking=think,
            k=item.get("k"), depth=item.get("depth"), answer=item["answer"],
            prompt_chars=len(item["prompt"]))
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": True,
        # Exact server-side token counts. Delta counting is kept as a cross-check; when the
        # two disagree the server is right and the delta convention is the thing to fix.
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": bool(think)},
    }
    t0 = time.perf_counter()
    reasoning, content = [], []
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                await resp.aread()
                r.status, r.error = "http_error", f"{resp.status_code}: {resp.text[:200]}"
                return r
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if u := chunk.get("usage"):
                    r.usage_completion = u.get("completion_tokens")
                    r.usage_prompt = u.get("prompt_tokens")
                for ch in chunk.get("choices") or []:
                    if ch.get("finish_reason"):
                        r.finish_reason = ch["finish_reason"]
                    d = ch.get("delta") or {}
                    rc, cc = d.get("reasoning_content"), d.get("content")
                    if rc:
                        reasoning.append(rc)
                    if cc:
                        content.append(cc)
                    if rc or cc:
                        r.n_deltas += 1
                        if r.ttft is None:
                            r.ttft = time.perf_counter() - t0
        r.e2e = time.perf_counter() - t0
    except Exception as e:
        r.status, r.error, r.e2e = "exception", f"{type(e).__name__}: {e}", time.perf_counter() - t0
        return r

    rtext, ctext = "".join(reasoning), "".join(content)
    # vLLM splits <think> into reasoning_content ONLY when --reasoning-parser is set.
    # Otherwise the block arrives inline. Handle both and RECORD WHICH, or the thinking-token
    # metric is silently zero on one of the two paths.
    if rtext:
        r.think_path = "reasoning_content"
        r.reasoning_chars, r.content_chars = len(rtext), len(ctext)
        r.text = f"<think>{rtext}</think>{ctext}"
    else:
        r.text = ctext
        if THINK_CLOSE.search(ctext):
            r.think_path = "inline_tags"
            head = ctext[:THINK_CLOSE.search(ctext).start()]
            r.reasoning_chars = len(head)
            r.content_chars = len(ctext) - len(head)
        else:
            r.think_path = "none"
            r.content_chars = len(ctext)
    if r.ttft is None:
        r.status = "empty"
    return r


# --------------------------------------------------------------------------- batch probe

async def probe_batch(url, stop, out):
    """Record the batch size actually achieved. The design pins --max-num-seqs; this checks
    the pin took. Trusting the flag is how a confound survives to the writeup."""
    async with httpx.AsyncClient(timeout=5) as c:
        while not stop.is_set():
            try:
                t = (await c.get(f"{url}/metrics")).text
                for ln in t.splitlines():
                    if ln.startswith("vllm:num_requests_running"):
                        out.append(float(ln.rsplit(" ", 1)[-1]))
                        break
            except Exception:
                pass
            try:
                # 4 Hz, not 1 Hz. At 1 Hz a short pass finishes inside a single interval and
                # the only sample taken is the one before any request landed -- which reads
                # zero and is indistinguishable from a broken metric.
                await asyncio.wait_for(stop.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                pass


# --------------------------------------------------------------------------- run

async def run_pass(args, items, slice_, think, max_tokens, fh):
    sel = [i for i in items if i["slice"] == slice_]
    rng = random.Random(args.order_seed)
    rng.shuffle(sel)                       # same order every configuration
    # Per-pass caps. Taken AFTER the shuffle, so a subsample is spread across k rather than
    # being the first N of one level, and so the SAME N items are chosen for every
    # configuration (order_seed is fixed). That is what lets a capped run still pair against
    # an uncapped one: compare joins on item id and drops the surplus.
    label = f"{slice_}/{'think' if think else 'nothink'}"
    cap = args.limit
    for spec in (args.limit_pass or "").split(","):
        if not spec.strip():
            continue
        sl, mode, n = spec.split(":")
        if sl == slice_ and (mode == "think") == bool(think):
            cap = int(n)
    # `is not None`, not truthiness: a cap of 0 means SKIP THIS PASS, and `if cap:` silently
    # treated it as "no cap" and ran all 180 items instead of none.
    if cap:
        sel = sel[:cap]
    elif cap == 0 and args.limit_pass and f"{slice_}:{'think' if think else 'nothink'}:0" \
            in args.limit_pass.replace(" ", ""):
        print(f"  {label:<18} skipped (capped at 0)", flush=True)
        return []
    print(f"  {label:<18} {len(sel):>4} items  max_tokens={max_tokens}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    done = [0]
    samples: list[float] = []
    stop = asyncio.Event()
    probe = asyncio.create_task(probe_batch(args.url, stop, samples))

    # max_keepalive MUST be set alongside max_connections. Left at its default of 20
    # against 40 max_connections, the pool evicts and closes connections under fast
    # turnover, and the next request reuses a dead one and dies instantly with ReadError.
    # Measured: 36% failures on the fast no-thinking pass against 1% on the slow thinking
    # pass -- the opposite of a network timeout, which is what this was first misdiagnosed
    # as. bench.py sets the two equal and never had the problem.
    limits = httpx.Limits(max_connections=args.concurrency + 8,
                          max_keepalive_connections=args.concurrency + 8)
    async with httpx.AsyncClient(limits=limits, timeout=args.timeout) as client:
        async def work(item):
            async with sem:
                r = await one(client, args.url, item, args.config, think, max_tokens, args.seed)
                rec = grade(asdict(r))
                async with lock:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()             # per item; a sweep that writes at the end loses all
                    done[0] += 1
                    if done[0] % 25 == 0:
                        print(f"    {done[0]}/{len(sel)}", flush=True)
                return rec
        recs = await asyncio.gather(*(work(i) for i in sel))

    stop.set()
    await probe
    ok = [r for r in recs if r["status"] == "ok"]
    acc = sum(r["correct"] for r in ok) / len(ok) if ok else float("nan")
    trunc = sum(r["truncated"] for r in ok) / len(ok) if ok else float("nan")
    unp = sum(not r["parsed"] for r in ok) / len(ok) if ok else float("nan")
    toks = [r["usage_completion"] for r in ok if r["usage_completion"]]
    paths = {r["think_path"] for r in ok}
    bmax = max(samples) if samples else float("nan")
    bmed = sorted(samples)[len(samples) // 2] if samples else float("nan")
    batch_note = ""
    if not samples:
        batch_note = "  BATCH PROBE TOOK NO SAMPLES -- /metrics unreachable?"
    elif bmax == 0:
        batch_note = (f"  BATCH PROBE READ ZERO on all {len(samples)} samples -- either the "
                      "metric name changed or the pass was shorter than the sampling window. "
                      "Achieved batch size is UNVERIFIED for this pass.")
    print(f"    acc {acc:6.1%}   truncated {trunc:5.1%}   unparseable {unp:5.1%}   "
          f"err {len(recs)-len(ok)}", flush=True)
    print(f"    completion tokens p50 {sorted(toks)[len(toks)//2] if toks else 0:>5}  "
          f"p99 {sorted(toks)[int(len(toks)*0.99)] if toks else 0:>5}  "
          f"batch med {bmed:.0f} max {bmax:.0f} (n={len(samples)})  "
          f"think_path {sorted(paths)}", flush=True)
    if batch_note:
        print(f"   {batch_note}", flush=True)
    if think and toks:
        p99 = sorted(toks)[int(len(toks) * 0.99)]
        if p99 >= max_tokens * 0.95:
            print(f"    WARNING: completion p99 {p99} is at the max_tokens ceiling "
                  f"{max_tokens}. Truncation is capping the measurement -- raise it.",
                  flush=True)
    return recs


async def cmd_run(args):
    items = [json.loads(l) for l in open(args.items) if l.strip()]
    want = set(args.slices.split(","))
    passes = [p for p in PASSES if p[0] in want]
    if not passes:
        print(f"no passes match --slices {args.slices}", file=sys.stderr)
        return 1
    out = args.out or f"results/phase4-{args.config}.jsonl"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(f"config {args.config}  ->  {out}")
    print(f"  {args.url}   concurrency {args.concurrency}\n")
    with open(out, "w") as fh:
        for slice_, think, mt in passes:
            if think and args.max_tokens_think:
                mt = args.max_tokens_think
            if not think and args.max_tokens_nothink:
                mt = args.max_tokens_nothink
            await run_pass(args, items, slice_, think, mt, fh)
    print(f"\nwrote {out}")
    return 0


# --------------------------------------------------------------------------- determinism

async def cmd_check_determinism(args):
    """If sampling is on, the noise floor swamps every effect and the phase measures nothing.
    Two identical requests, sequentially, at batch 1. Greedy decoding must give byte-identical
    output. This is the cheapest possible check on the most damaging possible failure."""
    items = [json.loads(l) for l in open(args.items) if l.strip()]
    item = next(i for i in items if i["slice"] == "math")
    print(f"determinism check on {item['id']}, 2 sequential requests at batch 1")
    async with httpx.AsyncClient(timeout=args.timeout) as c:
        a = await one(c, args.url, item, "check", True, 512, args.seed)
        b = await one(c, args.url, item, "check", True, 512, args.seed)
    if a.status != "ok" or b.status != "ok":
        print(f"FAILED to complete: {a.status}/{a.error} {b.status}/{b.error}")
        return 1
    same = a.text == b.text
    print(f"  lengths {len(a.text)} / {len(b.text)}   think_path {a.think_path}")
    print(f"  identical: {same}")
    if not same:
        for i, (x, y) in enumerate(zip(a.text, b.text)):
            if x != y:
                print(f"  first divergence at char {i}: {a.text[i:i+60]!r} vs {b.text[i:i+60]!r}")
                break
        print("\n  SAMPLING IS ON, or the server is nondeterministic at batch 1.")
        print("  Qwen3's generation_config.json sets temperature 0.6 / top_p 0.95; if the")
        print("  request temperature=0 is not overriding it, STOP -- every quality number")
        print("  from this server would be noise.")
        return 1
    print("  greedy decoding confirmed at batch 1")
    return 0


# --------------------------------------------------------------------------- grade / compare

def cmd_grade(args):
    recs = [grade(json.loads(l)) for l in open(args.file) if l.strip()]
    with open(args.file, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"regraded {len(recs)} records in {args.file}")
    summarize(recs)
    return 0


def summarize(recs):
    keys = sorted({(r["slice"], r["thinking"], r.get("k")) for r in recs},
                  key=lambda t: (t[0], not t[1], t[2] or 0))
    print(f"\n  {'slice':<10} {'think':<6} {'k':>4} {'n':>5} {'acc':>7} {'trunc':>7} {'unparsed':>9} {'tok p50':>8}")
    for sl, th, k in keys:
        g = [r for r in recs if r["slice"] == sl and r["thinking"] == th
             and r.get("k") == k and r["status"] == "ok"]
        if not g:
            continue
        toks = sorted(r["usage_completion"] for r in g if r["usage_completion"])
        print(f"  {sl:<10} {str(th):<6} {str(k or '-'):>4} {len(g):>5} "
              f"{sum(r['correct'] for r in g)/len(g):>6.1%} "
              f"{sum(r['truncated'] for r in g)/len(g):>6.1%} "
              f"{sum(not r['parsed'] for r in g)/len(g):>8.1%} "
              f"{toks[len(toks)//2] if toks else 0:>8}")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar. Under the null, b ~ Binomial(b+c, 0.5).

    Exact rather than the chi-square approximation because the discordant counts here are
    small -- often under 20 -- which is exactly where the approximation misbehaves.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def cmd_compare(args):
    A = {(r["slice"], r["id"], r["thinking"]): r
         for r in (json.loads(l) for l in open(args.a)) if r["status"] == "ok"}
    B = {(r["slice"], r["id"], r["thinking"]): r
         for r in (json.loads(l) for l in open(args.b)) if r["status"] == "ok"}
    keys = sorted(A.keys() & B.keys())
    na, nb = len(A), len(B)
    print(f"A = {args.a}  ({na} ok)")
    print(f"B = {args.b}  ({nb} ok)")
    # Warn whenever EITHER side lost records. The first version only warned when the pair
    # count differed from both totals, so a run where B alone dropped 3 items reported
    # nothing -- and a silently shrinking denominator is how a biased sample gets in.
    print(f"paired on {len(keys)} items"
          + (f"   WARNING: dropped {na-len(keys)} from A, {nb-len(keys)} from B"
             if len(keys) < max(na, nb) else ""))

    groups = {}
    for key in keys:
        r = A[key]
        groups.setdefault((r["slice"], r["thinking"], r.get("k")), []).append(key)
    groups["ALL"] = keys

    print(f"\n  {'group':<24} {'n':>5} {'accA':>7} {'accB':>7} {'b':>4} {'c':>4} "
          f"{'disc':>6} {'ansdiff':>8} {'McNemar p':>10}")
    rows = sorted((g for g in groups if g != "ALL"),
                  key=lambda t: (t[0], not t[1], t[2] or 0)) + ["ALL"]
    for g in rows:
        ks = groups[g]
        b = sum(1 for k in ks if A[k]["correct"] and not B[k]["correct"])
        c = sum(1 for k in ks if not A[k]["correct"] and B[k]["correct"])
        accA = sum(A[k]["correct"] for k in ks) / len(ks)
        accB = sum(B[k]["correct"] for k in ks) / len(ks)
        # answer-level disagreement is more sensitive than correctness-level: two different
        # wrong answers are concordant on correctness but the model still changed its mind.
        ansdiff = sum(1 for k in ks if A[k]["extracted"] != B[k]["extracted"]) / len(ks)
        name = "ALL" if g == "ALL" else f"{g[0]}/{'think' if g[1] else 'nothink'}" + (
            f"/k{g[2]}" if g[2] else "")
        print(f"  {name:<24} {len(ks):>5} {accA:>6.1%} {accB:>6.1%} {b:>4} {c:>4} "
              f"{(b+c)/len(ks):>5.1%} {ansdiff:>7.1%} {mcnemar_exact(b, c):>10.4f}")
    print("\n  b = A right, B wrong.  c = A wrong, B right.  disc = (b+c)/n.")
    print("  ansdiff = extracted answers differ at all -- compare THIS against the")
    print("  bf16-vs-bf16 control's ansdiff, which is the noise floor d0.")
    print("  McNemar tests whether B is LESS ACCURATE; it is unbiased under symmetric")
    print("  nondeterminism, so it does not need d0. See design section 2b.")
    return 0


# --------------------------------------------------------------------------- selftest

def cmd_selftest(args):
    cases = [
        ("ANSWER: 42", "math", "42", "plain"),
        ("ANSWER: 1,000", "math", "1000", "thousands separator"),
        ("ANSWER: $18", "math", "18", "currency"),
        ("ANSWER: -5", "math", "-5", "negative (2 of 1319 gsm8k answers are)"),
        ("**ANSWER: 42**", "math", "42", "markdown bold"),
        ("ANSWER: 42.", "math", "42", "trailing period"),
        ("ANSWER: 18.0", "math", "18", "integer written as float"),
        ("answer: 42", "math", "42", "case insensitive"),
        ("ANSWER: 42\nANSWER: 37", "math", "37", "LAST match wins"),
        ("<think>ANSWER: 42</think>\nANSWER: 37", "math", "37", "ignores thinking block"),
        ("<think>I get 42.</think>\nANSWER: 42", "math", "42", "answer after thinking"),
        ("The answer is 42.", "math", None, "NO fallback hunt for integers"),
        ("ANSWER: forty-two", "math", None, "words are not integers"),
        ("ANSWER: 18.5", "math", None, "non-integer rejected, not rounded"),
        ("", "math", None, "empty completion"),
        ("<think>reasoning ran out of tokens", "math", None, "truncated mid-thinking"),
        (r"### Final Answer\n$$\n\boxed{3}\n$$", "math", "3", "boxed, Qwen3's real habit"),
        ("### Final Answer:\nANSWER: 72", "math", "72", "heading 'Answer:' must not eat the next line"),
        ("**Final Answer:**\n\nANSWER: 3060", "math", "3060", "bold heading, blank line between"),
        ("Final Answer:\n", "math", None, "bare heading alone yields nothing"),
        (r"\boxed{108}", "math", "108", "boxed bare"),
        (r"\boxed{1,000}", "math", "1000", "boxed with separator"),
        (r"\boxed{\text{42}}", "math", "42", "boxed wrapping text"),
        (r"ANSWER: 42\n\boxed{37}", "math", "37", "latest marker wins, boxed after"),
        (r"\boxed{37}\nANSWER: 42", "math", "42", "latest marker wins, ANSWER after"),
        (r"<think>\boxed{42}</think>\nANSWER: 37", "math", "37", "boxed inside thinking ignored"),
        ("ANSWER: 6K0X", "longctx", "6K0X", "code"),
        ("ANSWER: 6k0x", "longctx", "6K0X", "code case-normalised"),
        ("ANSWER: `6K0X`", "longctx", "6K0X", "code in backticks"),
    ]
    bad = 0
    for text, sl, want, why in cases:
        got = normalize(extract(text)[0], sl)
        ok = got == want
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {why:<42} {text[:34]!r:<38} -> {got!r}")
        if not ok:
            print(f"        expected {want!r}")

    mc = [((0, 0), 1.0), ((5, 0), 0.0625), ((0, 5), 0.0625), ((10, 0), 0.001953125),
          ((3, 3), 1.0), ((8, 2), 0.109375)]
    print()
    for (b, c), want in mc:
        got = mcnemar_exact(b, c)
        ok = abs(got - want) < 1e-9
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  mcnemar_exact(b={b}, c={c}) = {got:.9f}"
              + ("" if ok else f"   expected {want}"))
    print(f"\n{'SELFTEST FAILED' if bad else 'selftest passed'}: {len(cases)+len(mc)} cases, {bad} bad")
    return 1 if bad else 0


# --------------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--url", default="http://localhost:8000")
        p.add_argument("--items", default="results/phase4-items.jsonl")
        p.add_argument("--seed", type=int, default=20260828)
        p.add_argument("--timeout", type=float, default=900)

    r = sub.add_parser("run"); common(r)
    r.add_argument("--config", required=True, help="label, e.g. bf16-a / fp8 / int4-w4a16")
    r.add_argument("--slices", default="math,gsm8k",
                   help="longctx needs its own server session: the design pins "
                        "--max-num-seqs 6 for it against 12 for the rest")
    r.add_argument("--concurrency", type=int, default=32,
                   help="deliberately ABOVE the server's --max-num-seqs pin, so the queue "
                        "keeps the batch saturated at exactly the pinned size")
    r.add_argument("--max-tokens-think", type=int, default=0,
                   help="override the thinking passes once calibration has measured p99")
    r.add_argument("--order-seed", type=int, default=7)
    r.add_argument("--limit", type=int, default=0,
                   help="cap EVERY pass at N. Calibration only; prefer --limit-pass")
    r.add_argument("--limit-pass", default="",
                   help="cap one pass: 'math:think:180'. Comma-separate several. The "
                        "thinking passes dominate cost (math/think is 60 percent of a run) "
                        "and sit at a 100 percent ceiling, where each item is maximally "
                        "informative, so halving them loses little")
    r.add_argument("--max-tokens-nothink", type=int, default=0)
    r.add_argument("--out", default="")

    d = sub.add_parser("check-determinism"); common(d)
    d.add_argument("--order-seed", type=int, default=7)

    g = sub.add_parser("grade"); g.add_argument("file")
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    sub.add_parser("selftest")

    args = ap.parse_args()
    if args.cmd == "run":
        return asyncio.run(cmd_run(args))
    if args.cmd == "check-determinism":
        return asyncio.run(cmd_check_determinism(args))
    if args.cmd == "grade":
        return cmd_grade(args)
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "selftest":
        return cmd_selftest(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

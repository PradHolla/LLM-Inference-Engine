"""
continuous.py -- Phase 2 step 3a: continuous batching, offline.

step 2 (engine/static_batch.py) put a number on static batching's flaw: a finished
row cannot free its slot, so a ragged workload ([256,128,64,64,32,32,16,16]) measured
29.7% utilisation -- 3,360 of 4,096 slot-steps computed tokens for sequences that had
already finished. Continuous batching is the fix: every step, finished rows are
EVICTED and queued requests are ADMITTED into the freed slots, so the batch composition
changes mid-flight instead of being fixed at launch.

The three mechanics this needs -- left-padded rows with per-row (buffer-decoupled)
RoPE position_ids, index_select eviction, and pad-then-cat admission -- are proven
token-for-token identical to batch-1 references in engine/cache_probe.py. This file
does not re-derive or alter that design; it batches admission (cache_probe admits one
newcomer at a time to keep its demo simple) and wraps it in a real FIFO scheduler.

  1. eviction:  index_select(0, keep_idx) on every layer's K/V, on mask, on nxt/next_pos
  2. admission: ALL newcomers that fit are prefilled together in ONE forward (batched,
     not greedy one-at-a-time -- GEMM efficiency rises with token count: the prefill
     sweep measured 8x412 tokens together at ~975ms vs ~1,187ms done singly, 18%
     faster), then left-padded with zero KV up to the current buffer length and cat'd
     onto the batch dimension
  3. decode:    one forward, one token for every currently active row

Order within a step matters and is fixed: EVICT (shrink) -> ADMIT (grow) -> DECODE.
Evicting first means admission always sees the true freed-slot count; growing before
shrinking would either admit into slots that are not yet free or require a second
eviction pass.

PROMPTS ARE UNIFORM, same as static_batch.py -- every request uses the same
--prompt-tokens template, built with engine.manual.make_prompt. Only OUTPUT length
(--lengths, cycled across --n-requests) is ragged. That keeps admission padding-free
*within* an admission batch (item C), and keeps ragged-output-length the only variable
this file measures, not ragged-prompt-length (a separate, unmeasured problem).

TOKEN-COUNT CONVENTION -- decode-loop-only, matching static_batch.py's
per_seq_max_tokens, NOT engine.manual's max_new_tokens (which counts the prefill token
too). This file's Request.max_new_tokens is deliberately the *static_batch* convention:
it counts tokens produced by the DECODE phase only, not the one token admission's own
prefill produces for free. That is what makes --lengths default
(256,128,64,64,32,32,16,16, "the same spread step 2c used") land on the SAME per-row
decode-step counts step 2c measured, so item H's static-equivalent arithmetic is
comparing like with like. Every completed request therefore ends up with
len(tokens) == max_new_tokens + 1 (the free admission token plus the requested decode
tokens) -- which is exactly what item E's `useful_tokens = sum(len(r.tokens))` expects:
no special-casing needed, the +1 falls out of the token list itself.

  /opt/llm/.venv/bin/python -m engine.continuous
  /opt/llm/.venv/bin/python -m engine.continuous --max-batch 16 --n-requests 128
"""
from __future__ import annotations

import os
# MUST precede `import torch` (transitively, via engine.manual/engine.cache_probe
# below) -- read once at CUDA allocator init, silently ignored after. See
# engine/manual.py's identical comment and NOTES/predictions.md for the measured
# +29% concurrency this buys.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Import the VERIFIED helpers rather than duplicating them. Importing engine.manual /
# engine.cache_probe pulls in torch -- which is why the env line above MUST come first.
from engine.manual import pick_logits_kwarg, make_prompt, pct, min_samples
from engine.cache_probe import get_kv, set_kv, n_layers

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def collect_eos_ids(tok, model) -> set[int]:
    """Duplicated from engine.manual / engine.static_batch -- not in this file's
    designated import list (item A), so copied rather than cross-imported, matching
    static_batch.py's own precedent for this exact function."""
    ids: set[int] = set()
    for src in (getattr(tok, "eos_token_id", None),
                getattr(getattr(model, "generation_config", None), "eos_token_id", None)):
        if src is None:
            continue
        if isinstance(src, int):
            ids.add(src)
        else:
            ids.update(int(x) for x in src)
    return ids


def _f(v: float, w: int, suffix: str = "") -> str:
    # Duplicated from engine/manual.py and engine/static_batch.py.
    return f"{'n/a':>{w}}" if v != v else f"{v:>{w}.1f}{suffix}"


# --------------------------------------------------------------------------- Request
@dataclass
class Request:
    rid: int
    prompt_ids: "torch.Tensor"          # 1-D, [prompt_len]
    max_new_tokens: int                 # decode-loop tokens only -- see module docstring
    admitted_step: int | None = None    # engine step at which this row entered the batch
    first_token_s: float | None = None  # TTFT: admission-forward start -> its own return
    finish_step: int | None = None      # engine step at which this row was marked done
    tokens: list[int] = field(default_factory=list)   # includes the admission token
    itls: list[float] = field(default_factory=list)   # one entry per decode step this
                                                        # row was active for


# ----------------------------------------------------------------------------- Engine
class Engine:
    """Fixed max_batch, FIFO pending queue, step() = evict -> admit -> decode.

    Buffer layout matches engine.cache_probe.padded_batch_run exactly: every row is
    left-padded into one shared [B, L] buffer (mask, K, V all share L on their sequence
    dimension), L growing by exactly 1 every decode step regardless of eviction/
    admission (those only ever touch the batch dimension, dim 0). position_ids is
    tracked per row and is deliberately decoupled from a row's physical position in the
    buffer -- only the mask and position_ids need to be right, per cache_probe's proof.
    """

    def __init__(self, model, fwd_kw: str | None, max_batch: int, device,
                 eos_ids: set[int], ignore_eos: bool = True,
                 on_complete: Callable[[Request], None] | None = None,
                 on_token: Callable[[Request, int], None] | None = None,
                 compact_threshold: int = 128):
        self.model = model
        self.fwd_kw = fwd_kw
        self.max_batch = max_batch
        self.device = device
        self.eos_ids = eos_ids
        self.ignore_eos = ignore_eos
        self.on_complete = on_complete
        # Fires once per token produced for a row -- the admission-forward's free
        # token AND every decode-loop token (see _admit/_decode below). Added for
        # Phase 2 step 3b (engine/server.py) so a live server can stream tokens out
        # as they're produced instead of only learning about a row at on_complete.
        self.on_token = on_token
        # Trim the buffer once this many dead left-pad positions accumulate. Low enough
        # to keep the buffer tight, high enough that the ~2.4 ms copy amortises.
        self.compact_threshold = compact_threshold

        self.pending: list[Request] = []       # FIFO: pop from front (index 0)
        self.rows: list[Request] = []          # parallel to cache/mask/nxt/next_pos dim 0
        self.cache = None                      # transformers Cache, or None pre-admission
        self.mask: "torch.Tensor | None" = None
        self.nxt: "torch.Tensor | None" = None
        self.next_pos: "torch.Tensor | None" = None

        self.completed: list[Request] = []
        self.step_n = 0
        self.prefill_s: list[float] = []       # one entry per ADMISSION event
        self.decode_s: list[float] = []        # one entry per DECODE step
        self.active_counts: list[int] = []     # len(rows) at each decode step -- slot_steps_used
        self.compact_s: list[float] = []       # one entry per compaction event
        self.trimmed = 0                       # total buffer positions reclaimed
        self.max_buffer = 0                    # high-water buffer length, bounded or not

        self.wall_start: float | None = None   # perf_counter at first admission's forward start
        self.wall_end: float | None = None     # perf_counter at last request's finishing decode

    def submit(self, r: Request) -> None:
        self.pending.append(r)

    def _kw(self) -> dict:
        return {self.fwd_kw: 1} if self.fwd_kw else {}

    def _finished(self, r: Request) -> bool:
        # tokens[0] is the free admission token (see module docstring) -- decode_count
        # excludes it so max_new_tokens compares like with like against static_batch.py.
        decode_count = len(r.tokens) - 1
        if decode_count >= r.max_new_tokens:
            return True
        if not self.ignore_eos and r.tokens and r.tokens[-1] in self.eos_ids:
            return True
        return False

    # ---- phase 1: evict -----------------------------------------------------------
    def _evict(self) -> None:
        if not self.rows:
            return
        finished_idx = [i for i, r in enumerate(self.rows) if self._finished(r)]
        if not finished_idx:
            return
        keep = [i for i in range(len(self.rows)) if i not in finished_idx]
        for i in finished_idx:
            r = self.rows[i]
            self.completed.append(r)
            if self.on_complete is not None:
                self.on_complete(r)
        if keep:
            idx = torch.tensor(keep, device=self.device)
            for li in range(n_layers(self.cache)):
                k, v = get_kv(self.cache, li)
                set_kv(self.cache, li, k.index_select(0, idx).contiguous(),
                       v.index_select(0, idx).contiguous())
            self.mask = self.mask.index_select(0, idx)
            self.nxt = self.nxt.index_select(0, idx)
            self.next_pos = self.next_pos.index_select(0, idx)
            self.rows = [self.rows[i] for i in keep]
        else:
            # every active row finished at once -- the buffer is empty until the next
            # admission rebuilds it from scratch (mirrors the cache=None startup path).
            self.cache = None
            self.mask = None
            self.nxt = None
            self.next_pos = None
            self.rows = []

    def _compact(self) -> None:
        """Trim dead left-padding off the FRONT of the buffer.

        Every row is right-aligned: a row of true length L occupies [cur_L - L, cur_L).
        So the first cur_L - max(true_len) positions are padding for EVERY active row at
        once, and slicing them off cannot touch valid data. next_pos already holds each
        row's true length, so the trim point costs nothing to find.

        Without this the buffer grows by one per decode step and only resets when the
        batch empties -- which under sustained load never happens. A server would drift
        its own ITL upward for as long as it stayed busy, making every measurement a
        function of run length.
        """
        if self.cache is None or not self.rows:
            return
        cur_L = get_kv(self.cache, 0)[0].shape[2]
        self.max_buffer = max(self.max_buffer, cur_L)
        needed = int(self.next_pos.max().item())
        waste = cur_L - needed
        if waste < self.compact_threshold:
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for li in range(n_layers(self.cache)):
            k, v = get_kv(self.cache, li)
            set_kv(self.cache, li, k[:, :, waste:, :].contiguous(),
                   v[:, :, waste:, :].contiguous())
        self.mask = self.mask[:, waste:].contiguous()
        torch.cuda.synchronize()
        self.compact_s.append(time.perf_counter() - t0)
        self.trimmed += waste

    # ---- phase 2: admit -------------------------------------------------------------
    # KNOWN LIMITATION, deliberately measured rather than fixed here:
    # the left-padded buffer grows by one every decode step and only resets when the
    # batch empties completely, which with a full queue never happens. Over a 64-request
    # run it climbs from 412 to roughly 1,020, and EVERY row attends over the whole
    # buffer -- including a row admitted at step 600, whose real content is 412 tokens
    # sitting behind 600 tokens of zero padding it still pays to read. Expect ITL to
    # drift upward across the run. For a long-lived server this is unbounded and would
    # need periodic compaction. A paged cache removes the problem entirely by never
    # requiring rows to share a buffer length; that is step 4.
    def _admit(self) -> None:
        free = self.max_batch - len(self.rows)
        if free <= 0 or not self.pending:
            return
        newcomers = self.pending[:free]
        self.pending = self.pending[free:]

        lens = {int(r.prompt_ids.shape[0]) for r in newcomers}
        if len(lens) != 1:
            # Design assumes uniform prompts (module docstring) so the admission batch
            # itself needs no internal padding. Fail loudly rather than silently
            # mis-padding -- see the file's uncertainty list for why this is an assert,
            # not a feature.
            raise RuntimeError(f"admission batch has ragged prompt lengths {lens}; "
                                f"this design assumes uniform --prompt-tokens prompts")
        new_L = lens.pop()
        n_new = len(newcomers)
        batch_ids = torch.stack([r.prompt_ids for r in newcomers], dim=0)

        kw = self._kw()
        if self.wall_start is None:
            torch.cuda.synchronize()
            self.wall_start = time.perf_counter()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache_position = torch.arange(0, new_L, device=self.device)
        out = self.model(input_ids=batch_ids, past_key_values=None, use_cache=True,
                          cache_position=cache_position, **kw)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.prefill_s.append(t1 - t0)
        ttft = t1 - t0

        past_new = out.past_key_values
        first_tok = out.logits[:, -1, :].argmax(-1)     # [n_new]
        first_tok_list = first_tok.tolist()

        for i, r in enumerate(newcomers):
            r.admitted_step = self.step_n
            r.first_token_s = ttft
            tok_id = int(first_tok_list[i])
            r.tokens.append(tok_id)
            if self.on_token is not None:
                self.on_token(r, tok_id)

        if self.cache is None:
            # First-ever admission: the buffer doesn't exist yet, nothing to pad against.
            self.cache = past_new
            self.mask = torch.ones(n_new, new_L, dtype=torch.long, device=self.device)
            self.nxt = first_tok
            self.next_pos = torch.full((n_new, 1), new_L, dtype=torch.long,
                                        device=self.device)
        else:
            cur_L = get_kv(self.cache, 0)[0].shape[2]
            padn = cur_L - new_L
            if padn < 0:
                # Cannot happen with uniform prompts: cur_L only grows (by decode steps)
                # from an initial value == new_L, so it is always >= new_L by the time a
                # second admission occurs. Fail loudly if that invariant is ever broken.
                raise RuntimeError(f"newcomer prefill length {new_L} exceeds current "
                                    f"buffer length {cur_L}; buffer/admission invariant "
                                    f"violated")
            for li in range(n_layers(self.cache)):
                k, v = get_kv(self.cache, li)
                k2, v2 = get_kv(past_new, li)
                if padn > 0:
                    zk = torch.zeros(n_new, k2.shape[1], padn, k2.shape[3],
                                      dtype=k2.dtype, device=self.device)
                    k2 = torch.cat([zk, k2], dim=2)
                    v2 = torch.cat([zk, v2], dim=2)
                set_kv(self.cache, li, torch.cat([k, k2], dim=0).contiguous(),
                       torch.cat([v, v2], dim=0).contiguous())
            newmask = torch.zeros(n_new, cur_L, dtype=self.mask.dtype, device=self.device)
            newmask[:, cur_L - new_L:] = 1
            self.mask = torch.cat([self.mask, newmask], dim=0)
            self.nxt = torch.cat([self.nxt, first_tok], dim=0)
            self.next_pos = torch.cat(
                [self.next_pos, torch.full((n_new, 1), new_L, dtype=torch.long,
                                            device=self.device)], dim=0)
        self.rows.extend(newcomers)

    # ---- phase 3: decode --------------------------------------------------------
    def _decode(self) -> None:
        if not self.rows:
            return
        kw = self._kw()
        self.mask = torch.cat(
            [self.mask, torch.ones(len(self.rows), 1, dtype=self.mask.dtype,
                                    device=self.device)], dim=-1)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model(input_ids=self.nxt.unsqueeze(-1), attention_mask=self.mask,
                          position_ids=self.next_pos, past_key_values=self.cache,
                          use_cache=True, **kw)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_dur = t1 - t0
        self.decode_s.append(step_dur)
        self.active_counts.append(len(self.rows))

        self.cache = out.past_key_values
        self.nxt = out.logits[:, -1, :].argmax(-1)
        self.next_pos = self.next_pos + 1

        next_tok_list = self.nxt.tolist()
        for i, r in enumerate(self.rows):
            tok_id = int(next_tok_list[i])
            r.tokens.append(tok_id)
            r.itls.append(step_dur)
            if self.on_token is not None:
                self.on_token(r, tok_id)
            if r.finish_step is None and self._finished(r):
                r.finish_step = self.step_n
                self.wall_end = t1

    @torch.inference_mode()
    def step(self) -> None:
        # inference_mode is NOT optional. Without it every forward builds an autograd
        # graph and retains activations, and the engine OOMs at 21.7 GiB on a workload
        # that should sit near 16. Every other module in engine/ has it; this one was
        # written without it and a fake-torch harness cannot detect the difference,
        # because a fake torch has no autograd to leak.
        self.step_n += 1
        self._evict()
        self._compact()
        self._admit()
        self._decode()

    def run(self) -> list[Request]:
        while self.pending or self.rows:
            self.step()
        return self.completed


# ---------------------------------------------------------------------------- metrics
def compute_metrics(engine: Engine, completed: list[Request]) -> dict:
    """Defined EXACTLY as item E specifies -- this arithmetic is the deliverable."""
    steps = len(engine.decode_s)
    useful_tokens = sum(len(r.tokens) for r in completed)
    slot_steps_used = sum(engine.active_counts)
    slot_steps_avail = steps * engine.max_batch
    decode_utilization = (slot_steps_used / slot_steps_avail
                           if slot_steps_avail else float("nan"))
    decode_time = sum(engine.decode_s)
    prefill_time = sum(engine.prefill_s)
    wall_s = (engine.wall_end - engine.wall_start
              if engine.wall_start is not None and engine.wall_end is not None
              else float("nan"))
    useful_tok_s_wall = useful_tokens / wall_s if wall_s == wall_s and wall_s > 0 else float("nan")
    useful_tok_s_decode = useful_tokens / decode_time if decode_time > 0 else float("nan")
    prefill_share = prefill_time / wall_s if wall_s == wall_s and wall_s > 0 else float("nan")
    return {
        "steps": steps,
        "useful_tokens": useful_tokens,
        "slot_steps_used": slot_steps_used,
        "slot_steps_avail": slot_steps_avail,
        "decode_utilization": decode_utilization,
        "decode_time": decode_time,
        "prefill_time": prefill_time,
        "wall_s": wall_s,
        "useful_tok_s_wall": useful_tok_s_wall,
        "useful_tok_s_decode": useful_tok_s_decode,
        "prefill_share": prefill_share,
        "compact_time": sum(engine.compact_s),
        "compactions": len(engine.compact_s),
        "buffer_positions_trimmed": engine.trimmed,
        "max_buffer_len": engine.max_buffer,
    }


def static_equivalent(lengths_cycled: list[int], max_batch: int) -> dict:
    """Item H -- pure arithmetic, no GPU. What static batching (step 2's design) would
    have cost for this SAME per-request length list: process requests in consecutive
    groups of max_batch, each group costing max(group lengths) decode steps (its own
    one-time prefill produces a free token per row too, uncounted here -- mirrors
    static_batch.py's compute_metrics, where tokens_produced only counts the decode
    loop). Group size is the count of requests actually in that slice, not padded out
    to max_batch -- matters only if --n-requests is not a multiple of --max-batch."""
    total_steps = 0
    slot_steps = 0
    useful_tokens = 0
    for i in range(0, len(lengths_cycled), max_batch):
        group = lengths_cycled[i:i + max_batch]
        steps = max(group)
        total_steps += steps
        slot_steps += len(group) * steps
        # +1 per request: prefill emits a token before the decode loop starts, and the
        # requester receives it. The Engine counts it in len(r.tokens), so static must
        # count it too or the A/B compares two different conventions. Worth 5.6% at the
        # smoke test's 18-token average and 1.3% at the real workload's 76.
        useful_tokens += sum(group) + len(group)
    utilization = useful_tokens / slot_steps if slot_steps else float("nan")
    return {"total_steps": total_steps, "slot_steps": slot_steps,
            "useful_tokens": useful_tokens, "utilization": utilization}


# --------------------------------------------------------------------------- reporting
@dataclass
class ReqRecord:
    """Per-request JSONL record. Separate from Request -- Request.prompt_ids is a
    tensor and not JSON-serializable; everything else is copied straight across."""
    rid: int
    max_new_tokens: int
    admitted_step: int | None
    finish_step: int | None
    ttft_s: float | None
    n_tokens: int
    tokens: list[int]
    itls: list[float]


def to_record(r: Request) -> ReqRecord:
    return ReqRecord(rid=r.rid, max_new_tokens=r.max_new_tokens,
                      admitted_step=r.admitted_step, finish_step=r.finish_step,
                      ttft_s=r.first_token_s, n_tokens=len(r.tokens),
                      tokens=r.tokens, itls=r.itls)


def write_jsonl(out, rec: ReqRecord) -> None:
    # Flushed per completed request, not buffered to the end -- a run that only writes
    # at the end loses everything to a crash, a Ctrl-C, or a spot reclaim.
    out.write(json.dumps(asdict(rec)) + "\n")
    out.flush()


def run_warmup(model, fwd_kw: str | None, max_batch: int, template_row: "torch.Tensor",
               eos_ids: set[int], device) -> None:
    """One short (8-decode-token) run at max_batch shape, discarded before any measured
    run. MEASURED 2026-08-21 in engine/manual.py: cuBLAS picks kernels per problem
    SHAPE, so warming at the wrong (batch, seq) shape warms nothing. This warms the
    INITIAL max_batch-sized admission and the max_batch-sized decode shape -- it does
    NOT separately warm the smaller incremental admission shapes (1..max_batch-1
    newcomers) that occur later in a real run once eviction starts freeing single
    slots; see this file's uncertainty list."""
    print(f"warming up (discarded)... shapes 1..{max_batch}", end=" ", flush=True)
    t0 = time.perf_counter()
    # Warm EVERY admission batch size, not just max_batch. Once eviction starts freeing
    # slots one or two at a time, admissions happen at n_new = 1..max_batch-1, and each
    # of those is a distinct GEMM shape cuBLAS has never selected a kernel for. Incident
    # 15 was exactly this: a 36% latency outlier that survived a warmup done at the
    # wrong shape. Costs ~max_batch extra prefills once; buys a clean prefill_time.
    for n_new in range(1, max_batch + 1):
        warm_engine = Engine(model, fwd_kw, n_new, device, eos_ids, ignore_eos=True)
        for i in range(n_new):
            warm_engine.submit(Request(rid=-(i + 1), prompt_ids=template_row,
                                       max_new_tokens=4))
        warm_engine.run()
    torch.cuda.synchronize()
    print(f"done ({time.perf_counter() - t0:.1f}s)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--n-requests", type=int, default=64)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--lengths", default="256,128,64,64,32,32,16,16",
                    help="comma list of per-request decode-loop output lengths, "
                         "CYCLED across --n-requests -- same spread step 2c used")
    p.add_argument("--ignore-eos", dest="ignore_eos", action="store_true", default=True,
                   help="do not stop on eos; run exactly the requested token counts "
                        "(default). Identical prompts plus greedy means eos would fire "
                        "on every row at the same offset -- see engine/static_batch.py")
    p.add_argument("--respect-eos", dest="ignore_eos", action="store_false")
    p.add_argument("--compact-threshold", type=int, default=128,
                   help="trim the buffer once this many dead left-pad positions "
                        "accumulate; a huge value disables compaction entirely")
    p.add_argument("--out", default="results/phase2-step3a.jsonl")
    p.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1
    device = torch.device("cuda:0")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                  device_map="cuda:0")
    model.eval()
    GIB = 1 << 30
    weights_gib = torch.cuda.memory_allocated() / GIB
    print(f"{args.model} loaded in {time.time()-t0:.1f}s -- {weights_gib:.2f} GiB weights "
          f"on {torch.cuda.get_device_name(0)}")

    fwd_kw = pick_logits_kwarg(model)
    print(f"logits kwarg: {fwd_kw or 'NOT FOUND -- logits explosion risk, see kvprobe.py'}")

    # Uniform prompts, no padding -- deliberate, see the module docstring. Always
    # no-think, same reasoning as static_batch.py: what this file measures (scheduler
    # bookkeeping, not generation content) does not depend on it, so there is no CLI
    # flag for it here.
    body = make_prompt(args.prompt_tokens)
    input_ids = tok.apply_chat_template(
        [{"role": "user", "content": body}],
        add_generation_prompt=True, tokenize=True, return_tensors="pt",
        enable_thinking=False,
    )
    # transformers 5.x returns a BatchEncoding from apply_chat_template(tokenize=True);
    # 4.x returned a bare tensor. VERIFIED in engine/manual.py and baseline/server.py.
    if not hasattr(input_ids, "shape"):
        input_ids = input_ids["input_ids"]
    template_row = input_ids.to(device)[0]
    prompt_len = int(template_row.shape[0])
    print(f"prompt: {args.prompt_tokens} target -> {prompt_len} actual tokens "
          f"(chat template adds a few), uniform across every request -- no padding")

    eos_ids = collect_eos_ids(tok, model)
    print(f"eos ids: {sorted(eos_ids) or 'none found'}")
    print(f"ITL/TTFT percentile sample floors: p50>={min_samples(50)}  p95>={min_samples(95)}  "
          f"p99>={min_samples(99)} samples (fewer -> reported as n/a, correctly)")

    lengths = [int(x) for x in args.lengths.split(",")]
    lengths_cycled = [lengths[i % len(lengths)] for i in range(args.n_requests)]
    print(f"lengths: base pattern {lengths}, cycled across {args.n_requests} requests")

    if args.warmup:
        run_warmup(model, fwd_kw, args.max_batch, template_row, eos_ids, device)

    requests = [Request(rid=i, prompt_ids=template_row, max_new_tokens=lengths_cycled[i])
                for i in range(args.n_requests)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a") as out:
        engine = Engine(model, fwd_kw, args.max_batch, device, eos_ids,
                    args.ignore_eos,
                         on_complete=lambda r: write_jsonl(out, to_record(r)),
                        compact_threshold=args.compact_threshold)
        for r in requests:
            engine.submit(r)

        print(f"\nrunning continuous batching -- max_batch={args.max_batch}, "
              f"{args.n_requests} requests...")
        t_run0 = time.time()
        completed = engine.run()
        print(f"done ({time.time()-t_run0:.1f}s wall, includes model-call overhead "
              f"outside the timed prefill/decode regions)")

    m = compute_metrics(engine, completed)
    ttft_ms = [r.first_token_s * 1000 for r in completed if r.first_token_s is not None]
    itl_ms = [d * 1000 for r in completed for d in r.itls]

    print(f"\n\033[1mcontinuous batching -- {len(completed)}/{args.n_requests} "
          f"requests completed\033[0m")
    print(f"  steps               {m['steps']:>10}")
    print(f"  useful_tokens       {m['useful_tokens']:>10}")
    print(f"  slot_steps_used     {m['slot_steps_used']:>10}")
    print(f"  slot_steps_avail    {m['slot_steps_avail']:>10}")
    print(f"  decode_utilization  {m['decode_utilization']*100:>9.1f}%")
    print(f"  decode_time_s       {m['decode_time']:>10.3f}")
    print(f"  prefill_time_s      {m['prefill_time']:>10.3f}")
    print(f"  wall_s              {_f(m['wall_s'],10)}")
    print(f"  useful_tok_s_wall   {_f(m['useful_tok_s_wall'],10)}   <- THE HEADLINE. "
          f"honest: admission stalls included")
    print(f"  useful_tok_s_decode {_f(m['useful_tok_s_decode'],10)}   <- optimistic: "
          f"ignores prefill stalls")
    print(f"  prefill_share       {m['prefill_share']*100:>9.1f}%   of wall time spent "
          f"stalled on admission prefills")
    print(f"  compactions         {m['compactions']:>10}   trimmed "
          f"{m['buffer_positions_trimmed']:,} dead buffer positions in "
          f"{m['compact_time']*1000:.0f} ms")
    print(f"  max_buffer_len      {m['max_buffer_len']:>10}   high-water. Unbounded "
          f"without compaction -- a busy server never empties its batch.")

    print(f"\n  TTFT   p50 {_f(pct(ttft_ms,50),8)} ms   p95 {_f(pct(ttft_ms,95),8)} ms   "
          f"p99 {_f(pct(ttft_ms,99),8)} ms")
    print(f"  ITL    p50 {_f(pct(itl_ms,50),8)} ms   p95 {_f(pct(itl_ms,95),8)} ms   "
          f"p99 {_f(pct(itl_ms,99),8)} ms")

    s = static_equivalent(lengths_cycled, args.max_batch)
    ratio = (m["decode_utilization"] / s["utilization"]
             if s["utilization"] and s["utilization"] == s["utilization"] else float("nan"))
    print(f"\n\033[1mstatic-batching equivalent, SAME lengths, groups of {args.max_batch}, "
          f"pure arithmetic (no GPU)\033[0m")
    print(f"  total_steps={s['total_steps']}  slot_steps={s['slot_steps']}  "
          f"useful_tokens={s['useful_tokens']}  utilization={s['utilization']*100:.1f}%")
    print(f"  continuous decode_utilization {m['decode_utilization']*100:.1f}% vs "
          f"static {s['utilization']*100:.1f}%  --  {ratio:.2f}x")

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

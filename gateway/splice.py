"""
splice.py -- build the continuation prompt for the overlap path: the exact bytes the
engine already holds, plus the search results that arrived mid-generation.

  python -m gateway.splice --selftest
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

BLOCK_SIZE = int(os.environ.get("GW_BLOCK_SIZE", "16"))
CUE = os.environ.get(
    "GW_SPLICE_CUE",
    "\n\nWait, the search results just arrived:\n\n{block}\n\nI should use these to answer.\n")


@dataclass
class Continuation:
    """A continuation prompt plus the accounting that makes its cost checkable."""
    prompt: str
    prefix_chars: int
    spliced_chars: int
    cue: str

    @property
    def total_chars(self) -> int:
        return len(self.prompt)


def render_cue(block: str, cue: str = CUE) -> str:
    """Fill the splice cue with the retrieved block. No block means no splice."""
    if not block:
        return ""
    return cue.format(block=block) if "{block}" in cue else cue + block


def build(prompt_sent: str, generated: str, block: str, cue: str = CUE) -> Continuation:
    """prompt_sent + generated is what the engine already has; only the cue is new."""
    prefix = prompt_sent + generated
    spliced = render_cue(block, cue)
    return Continuation(prompt=prefix + spliced, prefix_chars=len(prefix),
                        spliced_chars=len(spliced), cue=cue)


def verify(cont: Continuation, prompt_sent: str, generated: str) -> None:
    """The property the whole design rests on. Cheap, so it is never skipped."""
    prefix = prompt_sent + generated
    if not cont.prompt.startswith(prefix):
        raise RuntimeError("continuation does not extend what the engine holds; "
                           "the cache would miss from the first generated token")
    if cont.prefix_chars != len(prefix):
        raise RuntimeError(f"prefix_chars {cont.prefix_chars} != {len(prefix)}")


def partial_block_tokens(n_prefix_tokens: int, block_size: int = BLOCK_SIZE) -> int:
    """Tokens in the unfilled trailing block. Only full blocks are cacheable, so
    this is the recompute the splice pays even on a perfect prefix match."""
    return n_prefix_tokens % block_size


def selftest() -> int:
    """Offline. Asserts the cache property and the documented negative result."""
    fails = []

    sent, gen, block = "<|im_start|>assistant\n", "<think>\nreasoning so far", "SOURCES"
    c = build(sent, gen, block)
    try:
        verify(c, sent, gen)
    except RuntimeError as e:
        fails.append(f"verify rejected a correct continuation: {e}")
    if not c.prompt.startswith(sent + gen):
        fails.append("continuation does not start with what the engine holds")
    if block not in c.prompt:
        fails.append("the retrieved block never made it into the prompt")
    if c.spliced_chars != len(c.prompt) - len(sent + gen):
        fails.append("spliced_chars disagrees with the actual splice length")

    # A continuation that rewrites the prefix must be caught, not merely measured.
    bad = Continuation(prompt="different" + gen, prefix_chars=len(sent + gen), cue=CUE,
                       spliced_chars=0)
    try:
        verify(bad, sent, gen)
        fails.append("verify accepted a continuation that broke the prefix")
    except RuntimeError:
        pass

    if build(sent, gen, "").spliced_chars != 0:
        fails.append("an empty block should splice nothing")

    for n, want in ((32, 0), (33, 1), (47, 15), (48, 0)):
        got = partial_block_tokens(n)
        if got != want:
            fails.append(f"partial_block_tokens({n}) = {got}, expected {want}")

    if build(sent, gen, block).prompt != build(sent, gen, block).prompt:
        fails.append("build is not deterministic")

    for f in fails:
        print(f"  FAIL {f}")
    print(f"selftest: {'PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    return 1 if fails else 0


def gate(verbose: bool = True) -> int:
    """Tokenizer-backed. Asserts BOTH that the chat path is unusable and that ours holds."""
    from gateway.prompt import n_tokens, patched_template, tok

    fails = []
    msgs = [{"role": "user", "content": "Why is the sky blue?"}]
    gen = "<think>\nOkay, the user asks about Rayleigh scattering"

    def render(m, **kw):
        return tok().apply_chat_template(m, tokenize=False,
                                         chat_template=patched_template(), **kw)

    sent = render(msgs, add_generation_prompt=True, enable_thinking=True)
    engine_holds = sent + gen

    # Direction 1: the chat path must still be broken. If Qwen ever teaches the template
    # to render an unclosed think block, this fails and the simpler path becomes available.
    partial = msgs + [{"role": "assistant", "content": gen}]
    chat_render = render(partial, continue_final_message=True, enable_thinking=True)
    if chat_render == engine_holds:
        fails.append("continue_final_message now round-trips; reconsider the raw-prompt path")

    # Direction 2: ours must extend it exactly.
    c = build(sent, gen, "SOURCES")
    try:
        verify(c, sent, gen)
    except RuntimeError as e:
        fails.append(str(e))

    n_prefix = n_tokens(engine_holds)
    partial_tok = partial_block_tokens(n_prefix)
    if verbose:
        print(f"  chat path round-trips : {chat_render == engine_holds}  (must be False)")
        print(f"  ours extends exactly  : {c.prompt.startswith(engine_holds)}")
        print(f"  prefix                : {n_prefix} tokens, {partial_tok} in a partial block")
        print(f"  recompute at 0.2915 ms/token: {partial_tok * 0.2915:.1f} ms")
    if not 0 <= partial_tok < BLOCK_SIZE:
        fails.append(f"partial block {partial_tok} outside [0, {BLOCK_SIZE})")

    for f in fails:
        print(f"  FAIL {f}")
    print(f"gate: {'PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--gate" in sys.argv:
        raise SystemExit(gate())
    raise SystemExit(selftest() if "--selftest" in sys.argv else print(__doc__) or 0)

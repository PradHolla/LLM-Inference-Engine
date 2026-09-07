"""
prompt.py -- render the chat template locally and assert the property the prefix
cache depends on: turn N-1's prompt must be a literal prefix of turn N's.

  python -m gateway.prompt --selftest | --diagnose
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

MODEL = os.environ.get("GW_MODEL", "Qwen/Qwen3-8B")
_TOK = None

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def tok():
    """The model's own tokenizer. Loaded once; needs no GPU and no weights."""
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(MODEL)
    return _TOK


_HIST_ASSISTANT = "{{- '<|im_start|>' + message.role + '\\n' + content }}"
_HIST_FIXED = ("{%- if enable_thinking is defined and not enable_thinking %}"
               "{{- '<|im_start|>' + message.role + '\\n<think>\\n\\n</think>\\n\\n' + content }}"
               "{%- else %}" + _HIST_ASSISTANT + "{%- endif %}")


def patched_template() -> str:
    """Qwen3 issue 1826: with thinking off the template adds an empty think block to the
    generation prompt but not to history, so turn N is not a prefix of turn N+1."""
    t = tok().chat_template
    if t.count(_HIST_ASSISTANT) < 1:
        raise RuntimeError("template shape changed; the 1826 patch no longer applies")
    return t.replace(_HIST_ASSISTANT, _HIST_FIXED)


def render(messages: list[dict], enable_thinking: bool = False,
           add_generation_prompt: bool = True, patch: bool = False) -> str:
    """The exact string the engine prefills, rendered here rather than asked for."""
    kw = {"chat_template": patched_template()} if patch else {}
    return tok().apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=add_generation_prompt,
                                     enable_thinking=enable_thinking, **kw)


def n_tokens(text: str) -> int:
    """Token count of an already-rendered string."""
    return len(tok()(text, add_special_tokens=False)["input_ids"])


def strip_thinking(messages: list[dict]) -> list[dict]:
    """Remove think blocks from assistant history. Must be deterministic or the
    cache dies silently, so it is a pure regex over a copy."""
    out = []
    for m in messages:
        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
            m = {**m, "content": _THINK.sub("", m["content"]).lstrip()}
        else:
            m = dict(m)
        out.append(m)
    return out


@dataclass
class PrefixCheck:
    """Whether prev extends into cur, and where it stops if not."""
    holds: bool
    diverge_at: int | None
    prev_chars: int
    cur_chars: int
    prev_tail: str = ""
    cur_tail: str = ""


def check_prefix(prev: str, cur: str) -> PrefixCheck:
    """Is the earlier rendered prompt a literal prefix of the later one?"""
    if cur.startswith(prev):
        return PrefixCheck(True, None, len(prev), len(cur))
    i = 0
    for i, (a, b) in enumerate(zip(prev, cur)):
        if a != b:
            break
    else:
        i = min(len(prev), len(cur))
    return PrefixCheck(False, i, len(prev), len(cur),
                       prev_tail=prev[i:i + 90], cur_tail=cur[i:i + 90])


def conversation(turns: int, reply_chars: int = 1200) -> list[dict]:
    """A conversation shaped like tools/convo.py's: alternating turns, fixed replies."""
    msgs: list[dict] = []
    for i in range(1, turns + 1):
        msgs.append({"role": "user", "content": f"Question {i} about inference engineering?"})
        if i < turns:
            msgs.append({"role": "assistant",
                         "content": f"Answer {i}. " + ("Explanation text. " * (reply_chars // 18))})
    return msgs


def diagnose(turns: int = 6, enable_thinking: bool = False) -> int:
    """Does the rendered prompt actually extend turn to turn? Decides why the
    previous reply is not cached: our rendering, or the engine's block reuse."""
    print(f"model {MODEL}   enable_thinking={enable_thinking}\n")
    broken = 0
    prev_render = None
    for n in range(1, turns + 1):
        msgs = conversation(n)
        cur = render(msgs, enable_thinking)
        if prev_render is not None:
            c = check_prefix(prev_render, cur)
            tag = "extends" if c.holds else "BREAKS"
            print(f"turn {n-1} -> {n}   {tag}   prev {c.prev_chars} chars, cur {c.cur_chars}")
            if not c.holds:
                broken += 1
                print(f"    diverges at char {c.diverge_at} of {c.prev_chars}"
                      f"  ({100*c.diverge_at/c.prev_chars:.1f}% through the earlier prompt)")
                print(f"    turn {n-1} had: ...{c.prev_tail!r}")
                print(f"    turn {n}  has: ...{c.cur_tail!r}")
        prev_render = cur

    print()
    if broken:
        print("VERDICT: our own rendering breaks the prefix. The re-rendered history does not")
        print("  reproduce what turn N-1 sent, so the cache cannot match past that point.")
        print("  This is hypothesis B, and it is ours to fix.")
    else:
        print("VERDICT: rendering is a clean extension at every turn, so nothing here stops the")
        print("  cache matching through the previous reply. Whether it actually does is a")
        print("  separate question about the engine's block reuse, and needs the box to answer.")
    return broken


def selftest() -> int:
    """Offline apart from the tokenizer files, which are cached after first use."""
    fails = []

    a = [{"role": "assistant", "content": "<think>hidden reasoning</think>Visible answer."}]
    if strip_thinking(a)[0]["content"] != "Visible answer.":
        fails.append(f"strip_thinking wrong: {strip_thinking(a)[0]['content']!r}")
    if strip_thinking(a) is a or strip_thinking(a)[0] is a[0]:
        fails.append("strip_thinking mutated or aliased its input")
    if a[0]["content"] != "<think>hidden reasoning</think>Visible answer.":
        fails.append("strip_thinking mutated the caller's message")
    if strip_thinking(a) != strip_thinking(a):
        fails.append("strip_thinking is not deterministic")
    u = [{"role": "user", "content": "<think>not mine</think>keep"}]
    if strip_thinking(u)[0]["content"] != "<think>not mine</think>keep":
        fails.append("strip_thinking touched a user message")

    c = check_prefix("abc", "abcdef")
    if not c.holds or c.diverge_at is not None:
        fails.append("check_prefix missed a true extension")
    c = check_prefix("abcX", "abcdef")
    if c.holds or c.diverge_at != 3:
        fails.append(f"check_prefix wrong divergence: {c}")
    if not check_prefix("", "anything").holds:
        fails.append("empty prefix should always hold")

    m = conversation(3)
    if [x["role"] for x in m] != ["user", "assistant", "user", "assistant", "user"]:
        fails.append(f"conversation shape wrong: {[x['role'] for x in m]}")

    try:
        broke_stock = sum(not check_prefix(render(conversation(n - 1), False),
                                           render(conversation(n), False)).holds
                          for n in range(2, 7))
        broke_fixed = sum(not check_prefix(render(conversation(n - 1), False, patch=True),
                                           render(conversation(n), False, patch=True)).holds
                          for n in range(2, 7))
        if broke_stock == 0:
            fails.append("stock template no longer breaks; upstream may have fixed 1826")
        if broke_fixed != 0:
            fails.append(f"the 1826 patch does not restore the prefix: {broke_fixed}/5 break")
    except Exception as e:
        fails.append(f"patched_template raised: {type(e).__name__}: {e}")

    try:
        r = render([{"role": "user", "content": "hi"}])
        if "hi" not in r:
            fails.append("render dropped the message content")
        if n_tokens(r) < 1:
            fails.append("n_tokens returned nothing")
    except Exception as e:
        fails.append(f"render raised: {type(e).__name__}: {e}")

    for f in fails:
        print(f"  FAIL {f}")
    print(f"selftest: {'PASS' if not fails else str(len(fails)) + ' FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    if "--write-template" in sys.argv:
        dest = sys.argv[sys.argv.index("--write-template") + 1]
        body = patched_template()
        with open(dest, "w") as fh:
            fh.write(body)
        print(f"wrote patched template to {dest}, {len(body)} chars")
        raise SystemExit(0)
    if "--diagnose" in sys.argv:
        think = "--think" in sys.argv
        raise SystemExit(0 if diagnose(enable_thinking=think) == 0 else 2)
    print(__doc__)

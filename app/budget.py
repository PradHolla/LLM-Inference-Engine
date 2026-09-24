"""Token counting, the per-answer budget and the summary boundary (6c design section 6).

  uv run --group serve python -m app.budget --selftest
"""
from __future__ import annotations

import glob
import math
import os
import sys
from functools import lru_cache

from . import config

# Chat-template tokens around each message and after the last one; see code-notes.
MESSAGE_OVERHEAD = 6
PROMPT_OVERHEAD = 12
_TOKENIZER: list = []


def _tokenizer_paths() -> list[str]:
    if config.TOKENIZER_PATH:
        return [config.TOKENIZER_PATH]
    roots = [os.environ.get("HF_HOME", ""), "/opt/llm/hf-cache",
             os.path.expanduser("~/.cache/huggingface")]
    pattern = "hub/models--Qwen--Qwen3-8B/snapshots/*/tokenizer.json"
    return [path for root in roots if root
            for path in sorted(glob.glob(os.path.join(root, pattern)))]


def tokenizer():
    """Qwen's tokenizer.json when present, else None (the laptop with the mock)."""
    if not _TOKENIZER:
        loaded = None
        for path in _tokenizer_paths():
            try:
                from tokenizers import Tokenizer
                loaded = Tokenizer.from_file(path)
                break
            except Exception:
                continue
        _TOKENIZER.append(loaded)
    return _TOKENIZER[0]


def counting_mode() -> str:
    return "tokenizer" if tokenizer() is not None else f"chars/{config.CHARS_PER_TOKEN_FALLBACK}"


@lru_cache(maxsize=8192)
def count_tokens(text: str) -> int:
    """Exact with the tokenizer, otherwise chars/3.5 rounded up (overstates, the safe side)."""
    if not text:
        return 0
    tok = tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False).ids)
    return math.ceil(len(text) / config.CHARS_PER_TOKEN_FALLBACK)


def message_tokens(message: dict) -> int:
    content = message.get("content")
    return count_tokens(content if isinstance(content, str) else str(content or "")) + MESSAGE_OVERHEAD


def prompt_tokens(messages: list[dict]) -> int:
    """Upper-bound estimate of the rendered prompt for a messages array."""
    return sum(message_tokens(message) for message in messages) + PROMPT_OVERHEAD


def reserve_out(think: bool) -> int:
    return config.RESERVE_OUT_THINK if think else config.RESERVE_OUT_PLAIN


def history_max(system_tokens: int, summary_tokens: int, user_tokens: int,
                think: bool = True, search: bool = True) -> int:
    """W - reserve_out - search_cap - system - summary - user_message."""
    return (config.CONTEXT_WINDOW - reserve_out(think)
            - (config.SEARCH_CAP_TOKENS if search else 0)
            - system_tokens - summary_tokens - user_tokens)


def summary_reserve() -> int:
    """Worst-case summary size, so the boundary does not depend on whether one exists yet."""
    return config.SUMMARY_MAX_TOKENS + 64 + MESSAGE_OVERHEAD


def max_tokens_for(prompt_token_count: int) -> int:
    """W - prompt - margin, floored at 1 so the request is still well-formed."""
    return max(1, config.CONTEXT_WINDOW - prompt_token_count - config.MAX_TOKENS_MARGIN)


def summary_boundary(tokens: list[int], limit: int, block: int | None = None,
                     min_keep: int | None = None) -> int:
    """How many leading history messages a summary replaces. Moves in whole blocks."""
    block = block or config.SUMMARY_BLOCK
    min_keep = config.SUMMARY_MIN_KEEP if min_keep is None else min_keep
    n = len(tokens)
    suffix = [0] * (n + 1)
    for index in range(n - 1, -1, -1):
        suffix[index] = suffix[index + 1] + tokens[index]
    if suffix[0] <= limit:
        return 0
    need = next(k for k in range(n + 1) if suffix[k] <= limit)
    want = math.ceil(need / block) * block
    avail = max(0, (n - min_keep) // block) * block
    k = min(want, avail)
    # Fit beats cache stability: a prompt that overflows fails, one that misses is slow.
    while k < n and suffix[k] > limit:
        k = min(n, k + 2)
    return k


def selftest() -> int:
    """Budget arithmetic and boundary quantisation, with the chars fallback forced."""
    fails: list[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            fails.append(f"  FAIL {name}")

    saved = list(_TOKENIZER)
    _TOKENIZER[:] = [None]
    count_tokens.cache_clear()
    try:
        check("chars fallback rounds up at 3.5", count_tokens("abcdefg") == 2 and
              count_tokens("abcdefgh") == 3 and count_tokens("") == 0)
        check("message carries template overhead",
              message_tokens({"role": "user", "content": "abcdefg"}) == 2 + MESSAGE_OVERHEAD)
        check("prompt adds the generation overhead",
              prompt_tokens([{"role": "user", "content": "abcdefg"}] * 2) ==
              2 * (2 + MESSAGE_OVERHEAD) + PROMPT_OVERHEAD)
        W = config.CONTEXT_WINDOW
        check("history_max worst case", history_max(300, 100, 50) ==
              W - 6144 - 3000 - 300 - 100 - 50)
        check("history_max think off, no search", history_max(300, 0, 50, think=False,
              search=False) == W - 2048 - 300 - 50)
        check("max_tokens is W minus prompt minus margin", max_tokens_for(1000) == W - 1000 - 64)
        check("max_tokens never below one", max_tokens_for(W * 2) == 1)
        check("prompt plus max_tokens never exceeds W",
              all(p + max_tokens_for(p) <= W for p in (0, 1, 5000, W - 65, W - 64)))
        check("under the limit summarises nothing", summary_boundary([100] * 10, 5000) == 0)
        check("boundary rounds up to a block", summary_boundary([100] * 40, 3500, block=20,
              min_keep=6) == 20)
        check("boundary respects the recent tail", summary_boundary([100] * 24, 1500, block=20,
              min_keep=6) == 10)
        check("fit wins over the block when the tail cannot fit",
              summary_boundary([10, 10, 10, 10, 5000, 5000], 6000, block=20, min_keep=6) == 6)
        k_seen = []
        for turns in range(1, 80):
            k_seen.append(summary_boundary([300] * (2 * turns), 12000, block=20, min_keep=6))
        jumps = sum(1 for a, b in zip(k_seen, k_seen[1:]) if a != b)
        check("boundary moves in jumps, not every turn", 0 < jumps <= len(k_seen) // 8)
        check("boundary is always even (starts the kept tail on a user turn)",
              all(k % 2 == 0 for k in k_seen))
        check("boundary never shrinks as the chat grows",
              all(b >= a for a, b in zip(k_seen, k_seen[1:])))
    finally:
        _TOKENIZER[:] = saved
        count_tokens.cache_clear()

    print("\n".join(fails) if fails else "selftest: PASS")
    if fails:
        print(f"selftest: {len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())

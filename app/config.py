import os


GATEWAY_URL = "http://127.0.0.1:8080"
DB_PATH = os.environ.get("APP_DB_PATH", "app/chats.db")
MODEL_ID = "Qwen/Qwen3-8B"   # fallback only, see D4
HOST = "127.0.0.1"
PORT = 8090

# Measured on the math slice in Phase 7. There is deliberately NOTHING between 128 and
# unbounded: budgets near the model's natural reasoning length get cut off mid-thought.
AUTO = "auto"
THINKING_LEVELS = {
    "auto":  AUTO,   # the planner picks off or full, never a cap (6c D5)
    "off":   0,      # send gw_thinking_budget = 0
    "brief": 128,    # send gw_thinking_budget = 128
    "full":  None,   # send NO gw_thinking_budget field at all
}
DEFAULT_THINKING = "auto"
THINKING_OPTIONS = {
    "auto": {"label": "Auto", "description": "The assistant decides whether to reason."},
    "off": {"label": "Off", "description": "Fast replies without a reasoning block."},
    "brief": {"label": "Brief", "description": "A short, bounded reasoning pass."},
    "full": {"label": "Full", "description": "Unbounded reasoning for harder questions."},
}
SEARCH_MODES = ("auto", "on", "off")
DEFAULT_SEARCH_MODE = "auto"
SEARCH_OPTIONS = {
    "auto": {"label": "Auto", "description": "Search the web when the question needs it."},
    "on": {"label": "On", "description": "Always search the web."},
    "off": {"label": "Off", "description": "Never search the web."},
}
SEARCH_DEFAULT = True   # legacy boolean kept for the pre-6c frontend

# Qwen3-8B model card values. Never greedy (incident 53).
SAMPLING_THINK = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
SAMPLING_PLAIN = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}

# Budget, 6c design section 6. W matches the app server's --max-model-len.
CONTEXT_WINDOW = 32768
RESERVE_OUT_THINK = 6144
RESERVE_OUT_PLAIN = 2048
SEARCH_CAP_TOKENS = 3000
MAX_TOKENS_MARGIN = 64
SUMMARY_BLOCK = 20
SUMMARY_MIN_KEEP = 6
SUMMARY_MAX_TOKENS = 320
NEXT_USER_TOKENS = 256
PLAN_MAX_TOKENS = 96
PLAN_TIMEOUT_S = 8.0
SUMMARY_TIMEOUT_S = 60.0
SUMMARY_WAIT_S = 10.0    # the next turn waits this long for a pending summary, then drops the block
MAX_QUERIES = 3
TOKENIZER_PATH = os.environ.get("APP_TOKENIZER", "")
CHARS_PER_TOKEN_FALLBACK = 3.5

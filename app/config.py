import os


GATEWAY_URL = "http://127.0.0.1:8080"
DB_PATH = os.environ.get("APP_DB_PATH", "app/chats.db")
MODEL_ID = "Qwen/Qwen3-8B"   # fallback only, see D4
HOST = "127.0.0.1"
PORT = 8090

# Measured on the math slice in Phase 7. There is deliberately NOTHING between 128 and
# unbounded: budgets near the model's natural reasoning length get cut off mid-thought.
THINKING_LEVELS = {
    "off":   0,      # send gw_thinking_budget = 0
    "brief": 128,    # send gw_thinking_budget = 128
    "full":  None,   # send NO gw_thinking_budget field at all
}
DEFAULT_THINKING = "brief"
THINKING_OPTIONS = {
    "off": {"label": "Off", "description": "Fast replies without a reasoning block."},
    "brief": {"label": "Brief", "description": "A short, bounded reasoning pass."},
    "full": {"label": "Full", "description": "Unbounded reasoning for harder questions."},
}
SEARCH_DEFAULT = True

#!/usr/bin/env bash
# Phase 6c session steps, one sub-command each. Runs ON the box after `app-run.sh up`.
#   sudo systemd-run --unit=p6c-<step> --collect /bin/bash /opt/llm/infra/p6c-runs.sh <step>
#   steps: wipe | g1 | planeval | replay | levels
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv          # absolute: systemd-run is root, ~ is /root
PY=/opt/llm/.venv/bin/python
export HF_HOME=/opt/llm/hf-cache HF_HUB_OFFLINE=1
ts() { date -u +%H:%M:%S; }
die() { echo "ABORT: $*"; exit 1; }
need_stack() {
    curl -sf -m 5 http://localhost:8080/health >/dev/null || die "no gateway on :8080; run app-run.sh up"
    curl -sf -m 5 http://localhost:8090/api/health >/dev/null || die "no app on :8090; run app-run.sh up"
}

case "${1:-}" in
wipe)
    # The owner asked for an empty app. Keep a copy rather than destroy it outright.
    sudo systemctl stop llm-app >/dev/null 2>&1
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    if [ -f app/chats.db ]; then
        "$PY" -c "import sqlite3; s=sqlite3.connect('app/chats.db'); d=sqlite3.connect('app/chats.db.bak-$stamp'); s.backup(d); d.close()" \
            || die "backup failed; nothing deleted"
        rm -f app/chats.db app/chats.db-wal app/chats.db-shm
        echo "[$(ts)] chats wiped; backup at /opt/llm/app/chats.db.bak-$stamp"
    else
        echo "[$(ts)] no chats.db; nothing to wipe"
    fi
    ;;
g1)
    need_stack
    "$PY" - <<'PY' | tee results/p6c-g1.txt
import json, httpx
model = httpx.get("http://localhost:8080/v1/models", timeout=10).json()["data"][0]["id"]
schema = {"type": "object", "properties": {"fruit": {"type": "string"}, "n": {"type": "integer"}},
          "required": ["fruit", "n"], "additionalProperties": False}
body = {"model": model, "messages": [{"role": "user", "content": "Name a fruit you like."}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "g1", "schema": schema, "strict": True}},
        "max_tokens": 64, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0,
        "gw_thinking_budget": 0, "gw_purpose": "plan"}
reply = httpx.post("http://localhost:8080/v1/chat/completions", json=body, timeout=60).json()
text = reply["choices"][0]["message"].get("content") or ""
print("G1 raw reply:", repr(text[:200]))
try:
    data = json.loads(text)
    ok = isinstance(data, dict) and set(data) == {"fruit", "n"} and isinstance(data["n"], int)
except json.JSONDecodeError:
    ok = False
# The prompt never mentions JSON, so a schema-exact reply means the engine enforced it.
print("G1_ENFORCED" if ok else "G1_NOT_ENFORCED")
PY
    ;;
planeval)
    need_stack
    today=$(date -u +%F)
    echo "[$(ts)] planeval on all.jsonl (581 items)"
    rm -f results/p6c-planeval.jsonl
    "$UV" run tools/planeval.py --labels data/plansets/all.jsonl --today "$today" \
        --out results/p6c-planeval.jsonl > results/p6c-planeval.txt 2>&1
    echo "  rc=$?"
    echo "[$(ts)] label-think (40 GSM8K + 40 FreshQA never-changing, off vs on)"
    rm -f results/p6c-think-labels.jsonl
    "$UV" run tools/planeval.py label-think --out results/p6c-think-labels.jsonl \
        > results/p6c-think-labels.txt 2>&1
    echo "  rc=$?"
    echo "[$(ts)] planeval on the measured think labels"
    rm -f results/p6c-planeval-think.jsonl
    "$UV" run tools/planeval.py --labels results/p6c-think-labels.jsonl --today "$today" \
        --out results/p6c-planeval-think.jsonl > results/p6c-planeval-think.txt 2>&1
    echo "  rc=$?"
    echo "PLANEVAL_DONE"
    ;;
replay)
    need_stack
    rm -f results/p6c-replay53.jsonl
    "$UV" run tools/appdrive.py run --mode convo --prompts data/plansets/replay53.json \
        --convo-level auto --convo-search auto --title p6c-replay53 --gap-s 1.5 \
        --out results/p6c-replay53.jsonl
    echo "REPLAY_DONE rc=$?"
    ;;
levels)
    need_stack
    rm -f results/p6c-levels.jsonl
    "$UV" run tools/appdrive.py run --mode levels --out results/p6c-levels.jsonl
    echo "LEVELS_DONE rc=$?"
    ;;
*)
    echo "usage: $0 wipe | g1 | planeval | replay | levels"; exit 2 ;;
esac
exit 0

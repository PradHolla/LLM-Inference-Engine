#!/usr/bin/env bash
# Phase 6d session steps: warm-up A/B, turn-1 recheck, load sweep, long conversation.
# Every step talks to a SCRATCH stack (gateway :8082, apps :8091/:8092, scratch DBs), never the
# owner's app on :8090 or its chats.db.   Runs ON the box after `app-run.sh up`:
#   sudo systemd-run --unit=p6d-<step> --collect /bin/bash /opt/llm/infra/p6d-runs.sh <step>
#   steps: scratch | warmcheck | turn1 | load | loadsearch | long | report
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv          # absolute: systemd-run is root, ~ is /root
PY=/opt/llm/.venv/bin/python
GWT=results/p6d-gw.jsonl
ts() { date -u +%H:%M:%S; }
die() { echo "ABORT: $*"; exit 1; }
up() { curl -sf -m 5 "$1" >/dev/null; }
wait_up() { local i; for i in $(seq 1 40); do up "$1" && return 0; sleep 1; done; return 1; }

start_gateway() {   # name port trace; both scratch apps get their own, so chat ids never collide
    sudo systemctl stop "$1" >/dev/null 2>&1
    sudo systemd-run --unit="$1" --collect --working-directory=/opt/llm \
        --setenv=HF_HOME=/opt/llm/hf-cache --setenv=HF_HUB_OFFLINE=1 \
        --setenv=GW_UPSTREAM=http://localhost:8000 --setenv=GW_METRICS=http://localhost:8000 \
        --setenv=GW_TRACE="$3" \
        "$PY" -m uvicorn gateway.app:app --host 127.0.0.1 --port "$2" >/dev/null 2>&1 \
        || die "$1 failed to start"
    wait_up "http://localhost:$2/health" || die "$1 never became ready"
}

start_app() {   # name port warm db gateway-port
    sudo systemctl stop "$1" >/dev/null 2>&1
    sudo systemd-run --unit="$1" --collect --working-directory=/opt/llm \
        --setenv=PYTHONUNBUFFERED=1 --setenv=APP_DB_PATH="$4" \
        --setenv=APP_GATEWAY_URL="http://127.0.0.1:$5" --setenv=APP_WARM="$3" \
        "$PY" -m uvicorn app.server:app --host 127.0.0.1 --port "$2" >/dev/null 2>&1 \
        || die "$1 failed to start"
    wait_up "http://localhost:$2/api/health" || die "$1 never became ready"
}

case "${1:-}" in
scratch)
    up http://localhost:8000/v1/models || die "no vLLM on :8000; run app-run.sh up"
    start_gateway gateway-p6d 8082 $GWT
    start_gateway gateway-p6d-cold 8083 results/p6d-gw-cold.jsonl
    rm -f /opt/llm/app/p6d-warm.db* /opt/llm/app/p6d-cold.db*
    start_app app-p6d-warm 8091 1 /opt/llm/app/p6d-warm.db 8082
    start_app app-p6d-cold 8092 0 /opt/llm/app/p6d-cold.db 8083
    echo "[$(ts)] SCRATCH_UP warm app :8091 -> gateway :8082 -> $GWT; cold app :8092 -> :8083"
    ;;
warmcheck)
    # Same ten prompts, thinking Auto, search off: once with the warm-up, once without.
    rm -f results/p6d-warm.jsonl results/p6d-cold.jsonl
    "$UV" run tools/appdrive.py run --mode convo --app http://127.0.0.1:8091 --convo-level auto \
        --convo-search off --title p6d-warm --gap-s 5 --out results/p6d-warm.jsonl
    "$UV" run tools/appdrive.py run --mode convo --app http://127.0.0.1:8092 --convo-level auto \
        --convo-search off --title p6d-cold --gap-s 5 --out results/p6d-cold.jsonl
    echo "WARMCHECK_DONE"
    ;;
turn1)
    # P6C-2's FreshQA questions, search on / thinking Off only, against its 1.83 s first token.
    rm -f results/p6d-turn1.jsonl
    "$UV" run tools/appdrive.py run --mode fresh --app http://127.0.0.1:8091 --per-category 20 \
        --fresh-arms on/off --gap-s 1.0 --out results/p6d-turn1.jsonl
    echo "TURN1_DONE rc=$?"
    ;;
load)
    rm -f results/p6d-load.jsonl
    "$UV" run tools/appdrive.py run --mode load --app http://127.0.0.1:8091 \
        --items data/plansets/all.jsonl --turns 4 --rates 2,4,8,12 --hold-s 240 --think-s 15 \
        --load-thinking auto --load-search off --out results/p6d-load.jsonl
    echo "LOAD_DONE rc=$?"
    ;;
loadsearch)
    rm -f results/p6d-loadsearch.jsonl
    "$UV" run tools/appdrive.py run --mode load --app http://127.0.0.1:8091 \
        --items data/plansets/all.jsonl --turns 4 --rates "${RATE:-4}" --hold-s 240 --think-s 15 \
        --load-thinking auto --load-search auto --seed 1 --out results/p6d-loadsearch.jsonl
    echo "LOADSEARCH_DONE rc=$?"
    ;;
long)
    rm -f results/p6d-long.jsonl
    "$UV" run tools/appdrive.py run --mode convo --app http://127.0.0.1:8091 --prompts long \
        --turns 40 --convo-level off --convo-search off --title p6d-long --gap-s 2 \
        --out results/p6d-long.jsonl
    echo "LONG_DONE rc=$?"
    ;;
report)
    "$UV" run tools/appdrive.py loadreport --app-out results/p6d-load.jsonl --gw-trace $GWT \
        > results/p6d-load-report.txt 2>&1
    [ -f results/p6d-loadsearch.jsonl ] && "$UV" run tools/appdrive.py loadreport \
        --app-out results/p6d-loadsearch.jsonl --gw-trace $GWT > results/p6d-loadsearch-report.txt 2>&1
    echo "REPORT_DONE"
    ;;
*)
    echo "usage: $0 scratch | warmcheck | turn1 | load | loadsearch | long | report"; exit 2 ;;
esac
exit 0

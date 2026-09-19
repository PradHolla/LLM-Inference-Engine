#!/usr/bin/env bash
# Phase 7 runs, one sub-command each. Every run gates its own preconditions and flushes
# per request, so a failure costs one arm rather than the session.
#   ./infra/phase7-runs.sh q4b | q1b | q2 | q3 | q4
# Always under systemd-run, never a foreground ssh session.
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv
PY=/opt/llm/.venv/bin/python
export HF_HOME=/opt/llm/hf-cache HF_HUB_OFFLINE=1
MODEL=Qwen/Qwen3-8B
# The model card forbids greedy decoding in thinking mode, and P7-S confirmed the trough
# survives correct sampling. Every run below uses the card's values.
QSAMP=(--temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0)
ts() { date -u +%H:%M:%S; }
die() { echo "ABORT: $*"; exit 1; }

require_vllm() {
    curl -sf -m 5 http://localhost:8000/health >/dev/null || die "no vLLM on :8000"
    systemctl show vllm --property=Environment --value 2>/dev/null \
        | tr ' ' '\n' | grep -q '^VLLM_USE_V2_MODEL_RUNNER=0$' \
        || die "vllm unit lacks VLLM_USE_V2_MODEL_RUNNER=0; the budget would not bind"
    echo "  [$(ts)] vLLM up, V2 runner disabled"
}

# Stop, WAIT for the port, relaunch, then wait for readiness and BRANCH on it.
restart_gateway() {
    local label="$1"; shift
    sudo systemctl stop gateway >/dev/null 2>&1
    local i
    for i in $(seq 1 20); do
        ss -ltn 2>/dev/null | grep -q ':8080 ' || break
        sleep 1
    done
    sudo systemd-run --unit=gateway --collect --working-directory=/opt/llm \
        --setenv=GW_UPSTREAM=http://localhost:8000 \
        --setenv=GW_METRICS=http://localhost:8000 \
        --setenv=GW_TRACE="results/p7gw-$label.jsonl" \
        "$@" \
        "$PY" -m uvicorn gateway.app:app --host 127.0.0.1 --port 8080 \
        >/dev/null 2>&1 || die "gateway unit failed to start for $label"
    local up=0
    for i in $(seq 1 40); do
        if curl -sf -m 2 http://localhost:8080/health >/dev/null 2>&1; then up=1; break; fi
        sleep 1
    done
    [ "$up" = 1 ] || die "gateway never became ready for arm $label"
    echo "  [$(ts)] gateway up for arm $label"
}

# vLLM with a chosen reasoning_end_str. Model load is 60-105 s and that is real.
restart_vllm_endstr() {
    local label="$1" endstr="$2"
    sudo systemctl stop vllm >/dev/null 2>&1
    VLLM_USE_V2_MODEL_RUNNER=0 KV_PIN=10213733807 ./infra/vllm-launch.sh "$label" \
        --model "$MODEL" --quantization fp8 --max-model-len 16384 \
        --kv-cache-dtype fp8 --enable-prefix-caching --reasoning-parser qwen3 \
        --reasoning-config "{\"reasoning_start_str\": \"<think>\", \"reasoning_end_str\": $endstr}" \
        || die "vllm relaunch failed for $label"
    require_vllm
}

ladder() {   # $1 out-prefix, $2 url, rest: extra qualeval args
    local prefix="$1" url="$2"; shift 2
    local b label arg
    for b in 128 512 2048; do
        label="b$b"
        echo "  [$(ts)] arm $prefix-$label"
        "$UV" run tools/qualeval.py run --url "$url" --config "$prefix-$label" \
            --slices math --concurrency 32 --max-tokens-think 6144 \
            --max-tokens-nothink 2048 --limit-pass math:think:180,math:nothink:0 \
            "${QSAMP[@]}" --thinking-budget "$b" --out "results/$prefix-$label.jsonl" "$@"
        echo "    rc=$?"
    done
}

case "${1:-}" in

# ---- Q1 section 4b: does Qwen's trained stop phrase beat the bare tag? -------------
q4b)
    echo "[$(ts)] Q4B stop-instruction A/B"
    restart_vllm_endstr p7q4b-A '"</think>"'
    ladder p7q4bA http://localhost:8000
    restart_vllm_endstr p7q4b-B '"Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n"'
    ladder p7q4bB http://localhost:8000
    ;;

# ---- Q1b: fixed vs load-adaptive budget at a MATCHED arrival rate ------------------
q1b)
    echo "[$(ts)] Q1B adaptive vs fixed, open-loop at matched rate"
    require_vllm
    # A math request at budget 2048 runs ~50 s e2e, so capacity is near 0.6 req/s at this
    # concurrency. Sit just at it: below and no queue forms and the policy never switches,
    # far above and the queue diverges and every arm is equally broken.
    RATE="${RATE:-0.6}"
    for arm in big small adaptive; do
        case "$arm" in
            big)      env=(--setenv=GW_BUDGET_DEFAULT=2048) ;;
            small)    env=(--setenv=GW_BUDGET_DEFAULT=128) ;;
            adaptive) env=(--setenv=GW_BUDGET_POLICY=adaptive
                           --setenv=GW_BUDGET_BIG=2048 --setenv=GW_BUDGET_SMALL=128
                           --setenv=GW_Q_HIGH=8 --setenv=GW_Q_LOW=2) ;;
        esac
        restart_gateway "q1b-$arm" "${env[@]}"
        echo "  [$(ts)] arm $arm at $RATE req/s"
        "$UV" run tools/qualeval.py run --url http://localhost:8080 \
            --config "p7q1b-$arm" --slices math --rate "$RATE" --concurrency 32 \
            --max-tokens-think 6144 --max-tokens-nothink 2048 \
            --limit-pass math:think:180,math:nothink:0 \
            "${QSAMP[@]}" --out "results/p7q1b-$arm.jsonl"
        echo "    rc=$?"
    done
    ;;

# ---- Q2: search-then-think against think-then-search -------------------------------
q2)
    echo "[$(ts)] Q2 retrieval order"
    require_vllm
    [ -s /opt/llm/.brave-key ] || die "no /opt/llm/.brave-key; both arms would skip the search"
    for order in retrieve_then_generate generate_then_retrieve; do
        restart_gateway "q2-$order" --setenv=GW_ORDER="$order" --setenv=GW_ALWAYS_SEARCH=1
        echo "  [$(ts)] arm $order"
        "$UV" run tools/qualeval.py run --url http://localhost:8080 \
            --config "p7q2-$order" --slices math --concurrency 8 \
            --max-tokens-think 6144 --max-tokens-nothink 2048 \
            --limit-pass math:think:60,math:nothink:0 \
            "${QSAMP[@]}" --thinking-budget 2048 --out "results/p7q2-$order.jsonl"
        echo "    rc=$?"
    done
    ;;

# ---- Q3: can the thinking overlap the search round-trip? ---------------------------
q3)
    echo "[$(ts)] Q3 overlap"
    require_vllm
    [ -s /opt/llm/.brave-key ] || die "no /opt/llm/.brave-key; overlap has nothing to hide"
    for order in retrieve_then_generate overlap; do
        restart_gateway "q3-$order" --setenv=GW_ORDER="$order" --setenv=GW_ALWAYS_SEARCH=1
        echo "  [$(ts)] arm $order"
        "$UV" run tools/qualeval.py run --url http://localhost:8080 \
            --config "p7q3-$order" --slices math --concurrency 8 \
            --max-tokens-think 6144 --max-tokens-nothink 2048 \
            --limit-pass math:think:60,math:nothink:0 \
            "${QSAMP[@]}" --thinking-budget 2048 --out "results/p7q3-$order.jsonl"
        echo "    rc=$?"
    done
    ;;

# ---- Q4: does priority actually reorder the queue? ---------------------------------
q4)
    echo "[$(ts)] Q4 priority"
    require_vllm
    ./infra/p7-priority.sh
    echo "  rc=$?"
    ;;

# ---- Sampling: is the runaway tail greedy repetition, as the model card warns? ------
samp)
    echo "[$(ts)] SAMP greedy against Qwen3's recommended sampling"
    require_vllm
    # Only the arms that can discriminate: one inside the trough, one either side of it.
    for b in 128 1024 none; do
        if [ "$b" = none ]; then label=unbounded; ARG=""; else label="b$b"; ARG="--thinking-budget $b"; fi
        for mode in greedy qwen; do
            if [ "$mode" = greedy ]; then
                SAMP=(--temperature 0.0)
            else
                SAMP=(--temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0)
            fi
            echo "  [$(ts)] arm $label/$mode"
            "$UV" run tools/qualeval.py run --url http://localhost:8000 \
                --config "p7samp-$label-$mode" --slices math --concurrency 32 \
                --max-tokens-think 6144 --max-tokens-nothink 2048 \
                --limit-pass math:think:180,math:nothink:0 \
                "${SAMP[@]}" $ARG --out "results/p7samp-$label-$mode.jsonl"
            echo "    rc=$?"
        done
    done
    ;;

*) echo "usage: $0 q4b|q1b|q2|q3|q4|samp"; exit 2 ;;
esac

echo "[$(ts)] ${1} DONE"
exit 0

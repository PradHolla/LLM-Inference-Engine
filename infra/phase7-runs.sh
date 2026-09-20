#!/usr/bin/env bash
# Phase 7 runs, one sub-command each. Every run gates its own preconditions and flushes
# per request, so a failure costs one arm rather than the session.
#   ./infra/phase7-runs.sh q4b | q1b | probe | q3 | q4
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
    if [ "${1:-strict}" = strict ]; then
        sudo journalctl -u vllm --no-pager -o cat | grep -o "reasoning_end_str=[^,)]*" | tail -1 \
            | grep -q "reasoning_end_str='</think>'" \
            || die "server is not on the bare </think> stop string; q4b leaves it on Qwen's phrase, which costs 42 points at small budgets"
    fi
    echo "  [$(ts)] vLLM up, V2 runner disabled, stop string checked"
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
    require_vllm lax
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
    echo "  [$(ts)] restoring the default server so the next run is not silently poisoned"
    restart_vllm_endstr p7 '"</think>"'
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

# ---- Q2/Q3: does the order of retrieval and generation matter? ---------------------
# One run, three arms. Q3 is overlap against retrieve_then_generate; Q2 is
# retrieve_then_generate against generate_then_retrieve. Same records serve both.
probe)
    echo "[$(ts)] search probe on the retrieval slice"
    [ -s /opt/llm/.brave-key ] || die "no /opt/llm/.brave-key"
    [ -s results/p7-retrieval-items.jsonl ] || "$UV" run tools/mkretrieval.py \
        --out results/p7-retrieval-items.jsonl || die "could not build the slice"
    rm -f results/p7-search-probe.jsonl
    "$UV" run tools/p7search.py --items results/p7-retrieval-items.jsonl \
        --out results/p7-search-probe.jsonl
    echo "  probe rc=$?"
    ;;

q3)
    echo "[$(ts)] Q2/Q3 retrieval order and overlap"
    require_vllm
    [ -s /opt/llm/.brave-key ] || die "no /opt/llm/.brave-key; overlap has nothing to hide"
    [ -s results/p7-retrieval-items.jsonl ] || die "no retrieval slice; run: $0 probe"
    # The probe is the gate, not a formality: Q2 was voided once by a slice that returned
    # no pages, and a 1 qps plan produces the same null from 429s. Refuse to spend the GPU.
    "$PY" - <<'PY' || die "probe missing or too many zero-source items; run: $0 probe"
import json, sys
try:
    r = [json.loads(l) for l in open("results/p7-search-probe.jsonl") if l.strip()]
except OSError:
    sys.exit(1)
z = sum(1 for x in r if x["n_sources"] == 0)
print(f"  probe: {len(r)} items, {z} zero-source ({z/max(1,len(r)):.1%})")
sys.exit(0 if r and z / len(r) <= 0.10 else 1)
PY
    for order in retrieve_then_generate overlap generate_then_retrieve; do
        restart_gateway "q3-$order" --setenv=GW_ORDER="$order" --setenv=GW_ALWAYS_SEARCH=1
        echo "  [$(ts)] arm $order"
        # concurrency 8 is safe only because the probe read x-ratelimit-policy off a live
        # response: this key is 50;w=1, not the documented free tier's 1/s.
        "$UV" run tools/qualeval.py run --url http://localhost:8080 \
            --config "p7q3-$order" --slices retrieval \
            --items results/p7-retrieval-items.jsonl --concurrency 8 \
            --max-tokens-think 6144 --max-tokens-nothink 512 \
            --limit-pass retrieval:think:60,retrieval:nothink:0 \
            "${QSAMP[@]}" --thinking-budget 2048 --out "results/p7q3-$order.jsonl"
        echo "    rc=$?"
    done
    ;;

# ---- Q4: does priority actually reorder the queue? ---------------------------------
q4)
    echo "[$(ts)] Q4 priority"
    require_vllm
    ./infra/p7-priority.sh; rc=$?
    echo "  rc=$rc"
    [ "$rc" = 0 ] || die "p7-priority.sh failed rc=$rc (verdict UNTESTED means no queue formed)"
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

*) echo "usage: $0 q4b|q1b|probe|q3|q4|samp"; exit 2 ;;
esac

echo "[$(ts)] ${1} DONE"
exit 0

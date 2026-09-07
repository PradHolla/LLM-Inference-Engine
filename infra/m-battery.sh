#!/usr/bin/env bash
# The Phase 6 app battery, M1-M9, in one unattended pass. Restarts vLLM only twice
# (A -> B for M7, B -> C for M8); the gateway restarts once more within config A,
# for M1's forced search. See NOTES/code-notes.md for the per-measurement rationale.
#   ./m-battery.sh [--dry-run] [--only M1,M2] [--skip M7,M8]
set -uo pipefail
cd /opt/llm || exit 1

UV=/home/ubuntu/.local/bin/uv
LAUNCH=/opt/llm/infra/vllm-launch.sh
RESULTS=/opt/llm/results
PROMPTS=/opt/llm/results/phase4-items.jsonl
# M7 and M8 ask about APP traffic: long retrieved context in, short grounded answer
# out. phase4-items is short arithmetic, the opposite shape.
PROMPTS_APP=/opt/llm/results/longctx-prompts.jsonl

# Built on the box: sync carries code, not results, so shipping this file would mean
# a manual copy that silently goes stale. Regenerated from the item set every run.
TEMPLATE=/opt/llm/qwen3-patched.jinja

# Qwen3 issue 1826: with thinking off the stock template adds an empty think block to the
# generation prompt but not to history, so turn N is not a prefix of turn N+1 and the cache
# dies. Patched at the server so it applies to every client, not per request.
build_template() {
    if [ "$DRY_RUN" = 1 ]; then
        echo "  [template] $UV run python -m gateway.prompt --write-template $TEMPLATE"
        return 0
    fi
    "$UV" run python -m gateway.prompt --write-template "$TEMPLATE"
}

build_app_prompts() {
    if [ "$DRY_RUN" = 1 ]; then
        echo "  [prompts] $UV run tools/mkprompts.py --slice longctx --out $PROMPTS_APP --min 50"
        return 0
    fi
    "$UV" run tools/mkprompts.py --items /opt/llm/results/phase4-items.jsonl \
        --slice longctx --out "$PROMPTS_APP" --min 50
}
MODEL=Qwen/Qwen3-8B
EAGLE='{"model":"RedHatAI/Qwen3-8B-speculator.eagle3","method":"eagle3","num_speculative_tokens":2}'

DRY_RUN=0
ONLY=""
SKIP=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --only=*) ONLY="${1#--only=}"; shift ;;
        --skip) SKIP="$2"; shift 2 ;;
        --skip=*) SKIP="${1#--skip=}"; shift ;;
        -h|--help)
            echo "usage: $0 [--dry-run] [--only M1,M2] [--skip M7,M8]"
            exit 0
            ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# --only/--skip are comma lists matched against a single M<N> label. No associative
# arrays (bash 3.2 target), so membership is a substring match on a ",list," fence.
want() {
    local m="$1"
    if [ -n "$ONLY" ]; then
        case ",$ONLY," in
            *",$m,"*) : ;;
            *) return 1 ;;
        esac
    fi
    if [ -n "$SKIP" ]; then
        case ",$SKIP," in
            *",$m,"*) return 1 ;;
        esac
    fi
    return 0
}

SUMMARY=()

record_skip() {
    local label="$1" outfile="$2" reason="$3"
    echo "[$(ts)] SKIP $label: $reason"
    SUMMARY+=("$label|skipped|0|-|$outfile")
    return 0
}

# Runs one measurement's tool invocation as literal argv ("$@"), never a pre-built
# flag string -- a scalar holding flags is exactly what word-split unpredictably
# (incident 42). Bookkeeps duration and row count; never aborts the battery itself.
run_tool() {
    local label="$1" outfile="$2"; shift 2
    if [ "$DRY_RUN" = 1 ]; then
        echo "  [$label] $*"
        SUMMARY+=("$label|dryrun|0|-|$outfile")
        return 0
    fi
    echo "[$(ts)] START $label"
    local t0 rc dur rows status
    t0=$(date +%s)
    "$@"
    rc=$?
    dur=$(( $(date +%s) - t0 ))
    rows=0
    [ -f "$outfile" ] && rows=$(wc -l < "$outfile" | tr -d ' ')
    status=ok
    [ "$rc" != 0 ] && status=failed
    echo "[$(ts)] END $label status=$status duration=${dur}s rows=$rows"
    SUMMARY+=("$label|$status|$dur|$rows|$outfile")
    return 0
}

# Launches one vLLM config through vllm-launch.sh, which already waits for the GPU
# to be released and gates on health. Branch on ITS exit status; never let one
# config's failure abort the rest of the battery.
# vLLM's KV profiler is not deterministic: identical commands have produced 9.28,
# 9.51 and 10.28 GiB here, and a 14.8% budget difference moved measured capacity 40%.
# A and C are pinned to the SAME bytes so M8 compares KV dtype and nothing else.
# B is left to profile because the draft model needs room this pin does not leave it.
KV_PIN_BYTES=10213733807

launch_server() {
    local label="$1"; shift
    local logfile="/tmp/m-battery-launch-$label.log"
    local pin=""
    if [ "$label" = A ] || [ "$label" = C ]; then pin="$KV_PIN_BYTES"; fi
    if [ "$DRY_RUN" = 1 ]; then
        echo "  [launch $label] KV_PIN=${pin:-unset} $LAUNCH $label $*"
        return 0
    fi
    echo "[$(ts)] LAUNCH $label (KV_PIN=${pin:-unset}): $*"
    KV_PIN="$pin" "$LAUNCH" "$label" "$@" > "$logfile" 2>&1
    local rc=$?
    if [ "$rc" != 0 ]; then
        echo "[$(ts)] LAUNCH $label FAILED (exit $rc); see $logfile"
        tail -20 "$logfile" | sed 's/^/    /'
        return 1
    fi
    local kv gb
    kv=$(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$logfile" | tail -1)
    gb=$(grep -oE 'Available KV cache memory: [0-9.]+ GiB' "$logfile" | tail -1)
    echo "[$(ts)] LAUNCH $label READY  kv=[${kv:-n/a}]  mem=[${gb:-n/a}]"
    return 0
}

stop_gateway() {
    if [ "$DRY_RUN" = 1 ]; then
        echo "  [gateway] sudo systemctl stop labbench gateway"
        return 0
    fi
    # session-up.sh leaves the lab bench on :8080 and it answers /health too, so
    # without this the gateway fails to bind and the battery measures the wrong service.
    sudo systemctl stop labbench 2>/dev/null
    sudo systemctl stop gateway 2>/dev/null
    return 0
}

# Poll on the condition, not a fixed sleep, and BRANCH on whether it came up or the
# attempts were exhausted -- a loop that cannot tell those apart is incident 30's bug.
wait_gateway() {
    local i ok
    ok=0
    for i in $(seq 1 30); do
        if curl -sf -m 3 http://localhost:8080/gateway/load >/dev/null 2>&1; then
            ok=1
            break
        fi
        sleep 2
    done
    if [ "$ok" = 1 ]; then
        echo "[$(ts)] gateway confirmed on :8080 (answered /gateway/load, which the lab bench does not serve)"
        return 0
    fi
    echo "[$(ts)] gateway did not answer /gateway/load after 60s"
    return 1
}

# label/trace/always_search are written as two full call variants below rather than
# assembled from a conditional flag string, per this file's no-flags-in-a-variable rule.
start_gateway() {
    local label="$1" trace="$2" always="$3"
    stop_gateway
    if [ "$DRY_RUN" = 1 ]; then
        if [ "$always" = 1 ]; then
            echo "  [gateway $label] sudo systemd-run --unit=gateway --collect --working-directory=/opt/llm --setenv=GW_UPSTREAM=http://localhost:8000 --setenv=GW_TRACE=$trace --setenv=GW_METRICS=http://localhost:8000 --setenv=GW_ALWAYS_SEARCH=1 /opt/llm/.venv/bin/python -m uvicorn gateway.app:app --host 127.0.0.1 --port 8080"
        else
            echo "  [gateway $label] sudo systemd-run --unit=gateway --collect --working-directory=/opt/llm --setenv=GW_UPSTREAM=http://localhost:8000 --setenv=GW_TRACE=$trace --setenv=GW_METRICS=http://localhost:8000 /opt/llm/.venv/bin/python -m uvicorn gateway.app:app --host 127.0.0.1 --port 8080"
        fi
        return 0
    fi
    echo "[$(ts)] START gateway $label trace=$trace always_search=$always"
    if [ "$always" = 1 ]; then
        sudo systemd-run --unit=gateway --collect --working-directory=/opt/llm \
          --setenv=GW_UPSTREAM=http://localhost:8000 \
          --setenv=GW_TRACE="$trace" \
          --setenv=GW_METRICS=http://localhost:8000 \
          --setenv=GW_ALWAYS_SEARCH=1 \
          /opt/llm/.venv/bin/python -m uvicorn gateway.app:app --host 127.0.0.1 --port 8080 >/dev/null 2>&1
    else
        sudo systemd-run --unit=gateway --collect --working-directory=/opt/llm \
          --setenv=GW_UPSTREAM=http://localhost:8000 \
          --setenv=GW_TRACE="$trace" \
          --setenv=GW_METRICS=http://localhost:8000 \
          /opt/llm/.venv/bin/python -m uvicorn gateway.app:app --host 127.0.0.1 --port 8080 >/dev/null 2>&1
    fi
    wait_gateway
    return $?
}

GW_DEFAULT_UP=0
ensure_default_gateway() {
    if [ "$GW_DEFAULT_UP" = 1 ]; then
        return 0
    fi
    if start_gateway A "$RESULTS/gateway-traces-A.jsonl" 0; then
        GW_DEFAULT_UP=1
        return 0
    fi
    GW_DEFAULT_UP=0
    return 1
}

[ "$DRY_RUN" = 1 ] || mkdir -p "$RESULTS"

echo "############################################################"
echo "# Phase 6 app battery -- M1 through M9"
echo "# dry-run=$DRY_RUN  only=[${ONLY:-all}]  skip=[${SKIP:-none}]"
echo "############################################################"

# ============================================================
# CONFIG A: fp8 weights, fp16 KV, --max-model-len 16384, no speculation
# Serves M1 M2 M3 M4 M5 M6 M9, plus the config-A control arm shared by M7 and M8.
# ============================================================
NEED_A=0
for m in M1 M2 M3 M4 M5 M6 M7 M8 M9; do
    if want "$m"; then NEED_A=1; fi
done

CONFIG_A_OK=0
echo
echo "########## CONFIG A: fp8 weights, fp16 KV, max-model-len 16384, spec off"
if [ "$NEED_A" = 1 ]; then
    build_template || echo "[$(ts)] WARNING: patched template not built"
    if launch_server A --model "$MODEL" --quantization fp8 --max-model-len 16384 \
           --chat-template "$TEMPLATE"; then
        CONFIG_A_OK=1
    else
        echo "[$(ts)] CONFIG A FAILED -- skipping M1 M2 M3 M4 M5 M6 M9 and the M7/M8 control arm"
    fi
else
    echo "config A not needed for the selected measurements"
fi

# ---------- M1: gateway with forced search, 12 conversational turns ----------
if want M1; then
    if [ "$CONFIG_A_OK" = 1 ]; then
        if start_gateway A-m1 "$RESULTS/gateway-traces-A-m1.jsonl" 1; then
            run_tool M1 "$RESULTS/m1-search.jsonl" \
                $UV run tools/convo.py --url http://localhost:8080 --turns 12 \
                --max-tokens 300 --out "$RESULTS/m1-search.jsonl"
            echo "  gateway trace (search_ms/fetch_ms/extract_ms/n_sources): $RESULTS/gateway-traces-A-m1.jsonl"
        else
            record_skip M1 "$RESULTS/m1-search.jsonl" "always-search gateway did not come up"
        fi
    else
        record_skip M1 "$RESULTS/m1-search.jsonl" "config A failed to launch"
    fi
else
    record_skip M1 "$RESULTS/m1-search.jsonl" "not selected"
fi

# ---------- M2: convo.py 30 turns, warm and cold, through the gateway ----------
if want M2; then
    if [ "$CONFIG_A_OK" = 1 ] && ensure_default_gateway; then
        run_tool M2-warm "$RESULTS/m2-warm.jsonl" \
            $UV run tools/convo.py --url http://localhost:8080 --turns 30 \
            --out "$RESULTS/m2-warm.jsonl"
        run_tool M2-cold "$RESULTS/m2-cold.jsonl" \
            $UV run tools/convo.py --url http://localhost:8080 --turns 30 --cold \
            --out "$RESULTS/m2-cold.jsonl"
    else
        record_skip M2-warm "$RESULTS/m2-warm.jsonl" "config A or gateway unavailable"
        record_skip M2-cold "$RESULTS/m2-cold.jsonl" "config A or gateway unavailable"
    fi
else
    record_skip M2-warm "$RESULTS/m2-warm.jsonl" "not selected"
    record_skip M2-cold "$RESULTS/m2-cold.jsonl" "not selected"
fi

# ---------- M3: mchat.py thrash, several live chats ----------
if want M3; then
    if [ "$CONFIG_A_OK" = 1 ] && ensure_default_gateway; then
        run_tool M3 "$RESULTS/m3-thrash.jsonl" \
            $UV run tools/mchat.py --url http://localhost:8080 --mode thrash \
            --chats 2,4,8,12,16 --turns 29 --out "$RESULTS/m3-thrash.jsonl"
    else
        record_skip M3 "$RESULTS/m3-thrash.jsonl" "config A or gateway unavailable"
    fi
else
    record_skip M3 "$RESULTS/m3-thrash.jsonl" "not selected"
fi

# ---------- M4: cold-open cost of a persisted ~16k-token chat ----------
if want M4; then
    if [ "$CONFIG_A_OK" = 1 ] && ensure_default_gateway; then
        # --seed-tokens pre-loads a synthetic ~16k history so turn 1 measures REOPENING
        # a persisted chat. Without it this measured a short first turn: a plausible
        # number two orders of magnitude off, and no error.
        run_tool M4 "$RESULTS/m4-coldopen.jsonl" \
            $UV run tools/convo.py --url http://localhost:8080 --cold --turns 1 \
            --seed-tokens 16384 --max-tokens 64 --out "$RESULTS/m4-coldopen.jsonl"
    else
        record_skip M4 "$RESULTS/m4-coldopen.jsonl" "config A or gateway unavailable"
    fi
else
    record_skip M4 "$RESULTS/m4-coldopen.jsonl" "not selected"
fi

# ---------- M5: thinking stripped (default) vs kept (--think), 25 turns ----------
if want M5; then
    if [ "$CONFIG_A_OK" = 1 ] && ensure_default_gateway; then
        run_tool M5-strip "$RESULTS/m5-strip.jsonl" \
            $UV run tools/convo.py --url http://localhost:8080 --turns 25 \
            --out "$RESULTS/m5-strip.jsonl"
        run_tool M5-think "$RESULTS/m5-think.jsonl" \
            $UV run tools/convo.py --url http://localhost:8080 --turns 25 --think \
            --out "$RESULTS/m5-think.jsonl"
    else
        record_skip M5-strip "$RESULTS/m5-strip.jsonl" "config A or gateway unavailable"
        record_skip M5-think "$RESULTS/m5-think.jsonl" "config A or gateway unavailable"
    fi
else
    record_skip M5-strip "$RESULTS/m5-strip.jsonl" "not selected"
    record_skip M5-think "$RESULTS/m5-think.jsonl" "not selected"
fi

# ---------- M6: overflow at turn 60, sliding window vs summarize-and-restart ----------
if want M6; then
    if [ "$CONFIG_A_OK" = 1 ] && ensure_default_gateway; then
        run_tool M6-window "$RESULTS/m6-window.jsonl" \
            $UV run tools/mchat.py --url http://localhost:8080 --mode overflow \
            --turns 60 --strategy window --out "$RESULTS/m6-window.jsonl"
        run_tool M6-summarize "$RESULTS/m6-summarize.jsonl" \
            $UV run tools/mchat.py --url http://localhost:8080 --mode overflow \
            --turns 60 --strategy summarize --out "$RESULTS/m6-summarize.jsonl"
    else
        record_skip M6-window "$RESULTS/m6-window.jsonl" "config A or gateway unavailable"
        record_skip M6-summarize "$RESULTS/m6-summarize.jsonl" "config A or gateway unavailable"
    fi
else
    record_skip M6-window "$RESULTS/m6-window.jsonl" "not selected"
    record_skip M6-summarize "$RESULTS/m6-summarize.jsonl" "not selected"
fi

# ---------- M9: end-to-end criteria -- arrival-rate sweep plus one convo run, ----------
# ---------- both through the gateway, while config A (the app's actual deployed ----------
# ---------- config) is still up. Grouped here, not after B/C, so this stays a ----------
# ---------- two-restart battery (A->B, B->C) rather than three.               ----------
if want M9; then
    if [ "$CONFIG_A_OK" = 1 ] && ensure_default_gateway; then
        run_tool M9-sweep "$RESULTS/m9-sweep.jsonl" \
            $UV run tools/bench.py --url http://localhost:8080 --model "$MODEL" \
            --sweep 1,2,4,6,8 --duration 45 --prompts-file "$PROMPTS" \
            --max-tokens 128 --out "$RESULTS/m9-sweep.jsonl"
        run_tool M9-convo "$RESULTS/m9-convo.jsonl" \
            $UV run tools/convo.py --url http://localhost:8080 --turns 25 \
            --out "$RESULTS/m9-convo.jsonl"
    else
        record_skip M9-sweep "$RESULTS/m9-sweep.jsonl" "config A or gateway unavailable"
        record_skip M9-convo "$RESULTS/m9-convo.jsonl" "config A or gateway unavailable"
    fi
else
    record_skip M9-sweep "$RESULTS/m9-sweep.jsonl" "not selected"
    record_skip M9-convo "$RESULTS/m9-convo.jsonl" "not selected"
fi

# ---------- M7/M8 shared control: config-A capacity, raw vLLM, same rates as the ----------
# ---------- ablation arms below. One run serves both comparisons -- config A does ----------
# ---------- not change between M7's and M8's control need, so a second identical ----------
# ---------- sweep would just double GPU time for no new number.                  ----------
if want M7 || want M8; then
        build_app_prompts || echo "[$(ts)] WARNING: app prompts not built"
    if [ "$CONFIG_A_OK" = 1 ]; then
        run_tool M7-M8-control "$RESULTS/m7-m8-control.jsonl" \
            $UV run tools/bench.py --url http://localhost:8000 --model "$MODEL" \
            --sweep 0.25,0.5,1,2 --duration 45 --prompts-file "$PROMPTS_APP" --max-tokens 32 \
            --out "$RESULTS/m7-m8-control.jsonl"
    else
        record_skip M7-M8-control "$RESULTS/m7-m8-control.jsonl" "config A failed to launch"
    fi
else
    record_skip M7-M8-control "$RESULTS/m7-m8-control.jsonl" "not selected"
fi

stop_gateway
GW_DEFAULT_UP=0

# ============================================================
# CONFIG B: config A plus EAGLE3, num_speculative_tokens 2 -- M7
# ============================================================
CONFIG_B_OK=0
if want M7; then
    echo
    echo "########## CONFIG B: config A plus EAGLE3 speculative decoding (k=2)"
    if launch_server B --model "$MODEL" --quantization fp8 --max-model-len 16384 \
           --chat-template "$TEMPLATE" \
            --speculative-config "$EAGLE"; then
        CONFIG_B_OK=1
    else
        echo "[$(ts)] CONFIG B FAILED -- skipping M7's speculative arm"
    fi
    if [ "$CONFIG_B_OK" = 1 ]; then
        run_tool M7-eagle "$RESULTS/m7-eagle.jsonl" \
            $UV run tools/bench.py --url http://localhost:8000 --model "$MODEL" \
            --sweep 0.25,0.5,1,2 --duration 45 --prompts-file "$PROMPTS_APP" --max-tokens 32 \
            --out "$RESULTS/m7-eagle.jsonl"
    else
        record_skip M7-eagle "$RESULTS/m7-eagle.jsonl" "config B failed to launch"
    fi
else
    record_skip M7-eagle "$RESULTS/m7-eagle.jsonl" "not selected"
fi

# ============================================================
# CONFIG C: config A but --kv-cache-dtype fp8 -- M8, capacity only
# ============================================================
CONFIG_C_OK=0
if want M8; then
    echo
    echo "########## CONFIG C: config A, --kv-cache-dtype fp8"
    if launch_server C --model "$MODEL" --quantization fp8 --max-model-len 16384 \
           --chat-template "$TEMPLATE" \
            --kv-cache-dtype fp8; then
        CONFIG_C_OK=1
    else
        echo "[$(ts)] CONFIG C FAILED -- skipping M8's fp8-KV arm"
    fi
    if [ "$CONFIG_C_OK" = 1 ]; then
        run_tool M8-fp8kv "$RESULTS/m8-fp8kv.jsonl" \
            $UV run tools/bench.py --url http://localhost:8000 --model "$MODEL" \
            --sweep 0.25,0.5,1,2 --duration 45 --prompts-file "$PROMPTS_APP" --max-tokens 32 \
            --out "$RESULTS/m8-fp8kv.jsonl"
    else
        record_skip M8-fp8kv "$RESULTS/m8-fp8kv.jsonl" "config C failed to launch"
    fi
else
    record_skip M8-fp8kv "$RESULTS/m8-fp8kv.jsonl" "not selected"
fi

echo
echo "############################################################"
echo "# SUMMARY"
echo "############################################################"
printf '%-16s %-10s %10s %10s  %s\n' "measurement" "status" "duration_s" "rows" "output_file"
printf '%s\n' "${SUMMARY[@]}" | sort | while IFS='|' read -r label status dur rows outfile; do
    printf '%-16s %-10s %10s %10s  %s\n' "$label" "$status" "$dur" "$rows" "$outfile"
done

echo
# M8 quality: paired 1,210-item eval, fp16 KV vs fp8 KV. Runs LAST because
# kvquality-run.sh launches and tears down its own two servers.
if want M8-quality; then
    echo "[$(ts)] START M8-quality (two arms, ~70 min)"
    if [ "$DRY_RUN" = 1 ]; then
        echo "  [M8-quality] /opt/llm/infra/kvquality-run.sh"
    else
        sudo systemctl stop gateway 2>/dev/null
        /opt/llm/infra/kvquality-run.sh && echo "[$(ts)] END M8-quality status=ok" \
            || echo "[$(ts)] END M8-quality status=failed"
    fi
else
    echo "[$(ts)] SKIP M8-quality: not selected"
fi

echo M_BATTERY_DONE
exit 0

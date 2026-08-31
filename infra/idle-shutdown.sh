#!/usr/bin/env bash
# Shut a GPU inference box down after sustained idleness. Install as a root
# cron job running every minute (wiring at the bottom of user-data.sh).
# To prevent shutdown during long CPU-only work: touch "$HOLD"
# Adapted from the sql-reasoning-llm guardrail -- see NOTES/code-notes.md.

APP_DIR="${APP_DIR:-/opt/app}"
THRESHOLD_PCT="${IDLE_GPU_PCT:-5}"
IDLE_MINUTES="${IDLE_SHUTDOWN_MINUTES:-30}"
BUSY_MINUTES="${IDLE_BUSY_MINUTES:-180}"
VLLM_PORT="${VLLM_PORT:-8000}"
STATE=/var/run/gpu-idle-count
TOKSTATE=/var/run/vllm-tokens-last
HOLD="$APP_DIR/.no-autoshutdown"

# Servers are judged by request activity, not memory held: vLLM keeps its KV
# cache resident at 0% util for its whole life. See NOTES/code-notes.md.

# Manual hold, but only while fresh -- a stale hold is treated as forgotten
# rather than intended (see NOTES/code-notes.md for the incident behind this).
HOLD_MAX_H="${HOLD_MAX_H:-6}"
if [ -f "$HOLD" ]; then
    HOLD_AGE_H=$(( ( $(date +%s) - $(stat -c %Y "$HOLD" 2>/dev/null || echo 0) ) / 3600 ))
    if [ "$HOLD_AGE_H" -lt "$HOLD_MAX_H" ]; then
        echo 0 > "$STATE"
        exit 0
    fi
    logger -t idle-shutdown "hold file is ${HOLD_AGE_H}h old (limit ${HOLD_MAX_H}h) - treating as stale, resuming idle checks"
fi

# Highest utilization across all GPUs. If nvidia-smi is missing or the driver
# isn't up yet, do nothing rather than risk shutting down a healthy box.
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
if [ -z "$UTIL" ]; then
    exit 0
fi

# --- Request activity -------------------------------------------------------
# Any movement in vLLM's cumulative token counters means real work happened.
SERVED=0
METRICS=$(curl -sf --max-time 2 "http://127.0.0.1:${VLLM_PORT}/metrics" 2>/dev/null || true)
if [ -n "$METRICS" ]; then
    TOKENS=$(printf '%s\n' "$METRICS" \
        | awk '/^vllm:(prompt|generation)_tokens_total/ {s+=$2} END {printf "%.0f", s+0}')
    LAST=$(cat "$TOKSTATE" 2>/dev/null || echo "")
    echo "${TOKENS:-0}" > "$TOKSTATE"
    # First poll after boot has no baseline -- treat as no activity, not as work.
    if [ -n "$LAST" ] && [ "${TOKENS:-0}" != "$LAST" ]; then
        SERVED=1
    fi
    # In-flight right now also counts (a single long generation can span polls
    # without the counter ticking, since it only increments on completion).
    INFLIGHT=$(printf '%s\n' "$METRICS" \
        | awk '/^vllm:num_requests_(running|waiting)/ {s+=$2} END {printf "%.0f", s+0}')
    [ "${INFLIGHT:-0}" -gt 0 ] && SERVED=1
fi

# --- Non-server GPU jobs ----------------------------------------------------
# Still judged by the original memory-based check; servers are not (NOTES/code-notes.md).
JOB_PROCS=0
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
for pid in $GPU_PIDS; do
    [ -r "/proc/$pid/cmdline" ] || continue
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    case "$cmd" in
        *vllm*|*api_server*|*engine.server*) continue ;;  # server: judged by activity
    esac
    JOB_PROCS=$((JOB_PROCS + 1))
done
JOB_PROCS=${JOB_PROCS:-0}

# Is a human still connected? `grep -c` can exit 1 and double-count on empty
# input -- see NOTES/code-notes.md for why `|| true` + ${VAR:-0} is required here.
SESSIONS=$(who 2>/dev/null | grep -c . || true)
SESSIONS=${SESSIONS:-0}

# --- Count idle minutes -----------------------------------------------------
COUNT=$(cat "$STATE" 2>/dev/null || echo 0)
if [ "$UTIL" -lt "$THRESHOLD_PCT" ] && [ "$SERVED" -eq 0 ]; then
    COUNT=$((COUNT + 1))
else
    COUNT=0
fi
echo "$COUNT" > "$STATE"

# --- Pick the rope ----------------------------------------------------------
if [ "$JOB_PROCS" -gt 0 ] || [ "$SESSIONS" -gt 0 ]; then
    LIMIT="$BUSY_MINUTES"
    REASON="in use (jobs=${JOB_PROCS}, sessions=${SESSIONS})"
else
    LIMIT="$IDLE_MINUTES"
    REASON="nothing running (idle vllm server does not count)"
fi

if [ "$COUNT" -ge "$LIMIT" ]; then
    logger -t idle-shutdown "GPU idle ${COUNT}m, ${REASON} - shutting down"
    wall "GPU idle ${COUNT} minutes (${REASON}). Shutting down. Prevent with: touch ${HOLD}" 2>/dev/null
    sleep 10
    /sbin/shutdown -h now
fi

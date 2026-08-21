#!/usr/bin/env bash
# Shut a GPU inference box down after sustained idleness.
#
# Adapted from the sql-reasoning-llm guardrail. The graded-idleness idea and both
# bug fixes are carried over verbatim. ONE THING IS DIFFERENT AND IT MATTERS --
# see "THE INFERENCE ADAPTATION" below.
#
# Install as a root cron job running every minute (wiring at the bottom).
# On a g5.xlarge a forgotten box is ~$170/week; this caps that at the threshold.
#
# To prevent shutdown during long CPU-only work:  touch "$HOLD"

APP_DIR="${APP_DIR:-/opt/app}"
THRESHOLD_PCT="${IDLE_GPU_PCT:-5}"
IDLE_MINUTES="${IDLE_SHUTDOWN_MINUTES:-30}"
BUSY_MINUTES="${IDLE_BUSY_MINUTES:-180}"
VLLM_PORT="${VLLM_PORT:-8000}"
STATE=/var/run/gpu-idle-count
TOKSTATE=/var/run/vllm-tokens-last
HOLD="$APP_DIR/.no-autoshutdown"

# ---------------------------------------------------------------------------
# THE INFERENCE ADAPTATION
#
# The original grades idleness by "does any process still hold GPU memory" --
# correct for training, where a job between batches reads 0% util but is alive.
#
# On an inference box that heuristic DEGENERATES TO ALWAYS-BUSY. vLLM allocates
# its KV cache at startup and holds it for the entire life of the server, at 0%
# util, whether it served a million requests or none. So GPU_PROCS is > 0 from
# the moment the server boots until it dies. Every idle box would take the
# 180-minute rope and the 30-minute limit would never once fire -- a silent 6x
# on the thing this script exists to prevent.
#
# Fix: a server holding memory is not evidence of use. Judge a server by REQUEST
# ACTIVITY instead, and keep the memory heuristic only for non-server jobs
# (benchmarks, eval sweeps, ad-hoc scripts) where the original logic is right.
#
# And measure activity with the CUMULATIVE token counter, not the instantaneous
# num_requests_running gauge. Polling a gauge once a minute misses essentially
# every request that has ever completed; a monotonic counter cannot miss one.
# ---------------------------------------------------------------------------

# Manual hold -- reset the counter and do nothing.
if [ -f "$HOLD" ]; then
    echo 0 > "$STATE"
    exit 0
fi

# Highest utilization across all GPUs. If nvidia-smi is missing or the driver
# isn't up yet, do nothing rather than risk shutting down a healthy box.
# (Carried over: without this the script kills the box during boot.)
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
if [ -z "$UTIL" ]; then
    exit 0
fi

# --- Request activity -------------------------------------------------------
# Any movement in vLLM's cumulative token counters since the last poll means the
# server did real work in the last minute.
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
# A benchmark sweep or eval script sitting in a CPU-only phase reads 0% util but
# is genuinely working. Those still get the original treatment. Servers do not.
#
# nvidia-smi's process_name is usually just "python3", so match on the real
# cmdline from /proc instead.
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

# Is a human still connected?
#
# Carried over: `grep -c` prints the count AND exits 1 when that count is zero,
# so `|| echo 0` would append a SECOND line, making "$SESSIONS" the two-line
# string "0\n0" and the -gt below die with "integer expression expected" --
# silently falling through to the aggressive limit. `|| true` + ${VAR:-0}.
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

# ---------------------------------------------------------------------------
# WIRING (goes in user-data.sh, not here):
#
#   install -m 755 infra/idle-shutdown.sh /usr/local/bin/idle-shutdown.sh
#   cat > /etc/cron.d/llm-idle <<EOF
#   APP_DIR=/opt/llm
#   IDLE_SHUTDOWN_MINUTES=30
#   IDLE_GPU_PCT=5
#   VLLM_PORT=8000
#   * * * * * root /usr/local/bin/idle-shutdown.sh
#   EOF
#   chmod 644 /etc/cron.d/llm-idle
#
# AND at launch time, so `shutdown -h now` STOPS the box instead of destroying
# it (on-demand keeps the root volume; spot cannot and must terminate):
#
#   --instance-initiated-shutdown-behavior stop      # on-demand
#   --instance-initiated-shutdown-behavior terminate # spot
#
# NOTE: a stopped instance still bills EBS (~$16/mo for 200 GB gp3).
# Terminate boxes you are actually done with.
# ---------------------------------------------------------------------------

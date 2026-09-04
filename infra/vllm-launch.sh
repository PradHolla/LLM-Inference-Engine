#!/usr/bin/env bash
# Launch vLLM on the box with a given argument set, wait for health, and print
# the configuration the server actually resolved (KV size, speculative config,
# rejection sampling method -- see NOTES/code-notes.md for why). Runs ON the box.
#   ./vllm-launch.sh <label> [vllm args...]
set -uo pipefail
LABEL="${1:?usage: vllm-launch.sh <label> [args...]}"; shift
# KV_PIN pins --kv-cache-memory. vLLM sizes KV by profiling free memory at startup and that
# profile is NOT deterministic: identical commands have taken 9.28, 9.51 and 10.28 GiB on
# this box, and the greedy one OOMed during CUDA graph capture. Pinning also removes the
# 10-15% run-to-run KV variance that makes capacity comparisons unattributable.
PIN_ARGS=""
[ -n "${KV_PIN:-}" ] && PIN_ARGS="--kv-cache-memory $KV_PIN"
VENV=/opt/llm/.venv-vllm

sudo systemctl stop vllm 2>/dev/null
for i in $(seq 1 30); do
    ss -ltn 2>/dev/null | grep -q ':8000 ' || break
    sleep 1
done
# The port closing and the driver reclaiming VRAM are DIFFERENT moments. Launching in the
# gap makes vLLM profile against memory the previous server still holds, and it then OOMs
# during graph capture. Same guard as labbench/backends.py; this script never had it.
gpufree=0
for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$used" ] && { gpufree=1; break; }
    [ "$used" -lt 1024 ] && { gpufree=1; break; }
    sleep 2
done
if [ "$gpufree" != 1 ]; then
    echo "LAUNCH_ABORT $LABEL: GPU still holds ${used} MiB after 120s; refusing to profile against it"
    exit 1
fi

sudo systemd-run --unit=vllm --collect --working-directory=/opt/llm \
  --setenv=HF_HOME=/opt/llm/hf-cache --setenv=HF_HUB_OFFLINE=1 --setenv=PYTHONUNBUFFERED=1 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --setenv=PATH=$VENV/bin:/usr/local/bin:/usr/bin:/bin \
  $VENV/bin/python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port 8000 $PIN_ARGS "$@" >/dev/null 2>&1
INVID=$(systemctl show vllm --property=InvocationID --value 2>/dev/null)

# Read THIS invocation only. `journalctl -u vllm` spans every launch, so in a sweep a
# failure can print a previous run's traceback and a success can scrape its KV numbers.
invlog() {
    local inv="$INVID"
    [ -n "$inv" ] || inv=$(systemctl show vllm --property=InvocationID --value 2>/dev/null)
    if [ -n "$inv" ]; then
        sudo journalctl "_SYSTEMD_INVOCATION_ID=$inv" --no-pager -o cat 2>/dev/null
    else
        sudo journalctl -u vllm --no-pager -o cat 2>/dev/null
    fi
}

ok=0
for i in $(seq 1 120); do
    if curl -sf -m 3 http://localhost:8000/health >/dev/null 2>&1; then ok=1; break; fi
    st=$(systemctl is-active vllm)
    if [ "$st" = "failed" ] || [ "$st" = "inactive" ]; then
        echo "LAUNCH_FAILED $LABEL (unit $st after ${i}s)"
        FULL="/tmp/vllm-fail-$LABEL.log"
        invlog > "$FULL"
        echo "  full log: $FULL ($(wc -l < "$FULL") lines)"
        # vLLM ends its traceback with "See root cause above", so a tail keeps the wrong
        # end. Grep the signatures the real cause actually uses.
        echo "  --- root cause candidates:"
        grep -nEi "ValueError|RuntimeError: [^E]|out of memory|free memory|KV cache|To serve at least|is larger than|decrease|No available memory" \
            "$FULL" | grep -viE "See root cause|Engine core initialization failed" \
            | head -12 | sed 's/^/    /'
        echo "  --- last 12 lines:"
        tail -12 "$FULL" | sed 's/^/    /'
        exit 1
    fi
    sleep 5
done
[ "$ok" = 1 ] || { echo "LAUNCH_TIMEOUT $LABEL"; invlog | tail -20 | sed 's/^/    /'; exit 1; }

L=$(invlog)
echo "### $LABEL READY"
echo "  args: $*"
echo "$L" | grep -oE 'GPU KV cache size: [0-9,]+ tokens'                  | tail -1 | sed 's/^/  /'
echo "$L" | grep -oE 'Available KV cache memory: [0-9.]+ GiB'             | tail -1 | sed 's/^/  /'
echo "$L" | grep -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' | tail -1 | sed 's/^/  /'
echo "$L" | grep -oE 'speculative_config=SpeculativeConfig\([^)]*\)|speculative_config=None' | tail -1 | sed 's/^/  /'
echo "$L" | grep -oE "rejection_sample_method='[a-z]*'"                   | tail -1 | sed 's/^/  /'
echo "  vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
# vLLM prints this on EVERY startup, success included, and the launcher ignored it for
# three phases while the reproducibility problem it solves went unfixed.
REC=$(echo "$L" | grep -oE '\-\-kv-cache-memory=[0-9]+' | head -1 | cut -d= -f2)
if [ -n "$REC" ]; then
    echo "  kv pin recommended by the engine: --kv-cache-memory $REC ($(awk -v b="$REC" 'BEGIN{printf "%.2f", b/1073741824}') GiB)"
    mkdir -p results && echo "$REC" > "results/kv-pin-$LABEL.txt"
    [ -n "${KV_PIN:-}" ] && echo "  (this run was pinned at $KV_PIN)"
fi

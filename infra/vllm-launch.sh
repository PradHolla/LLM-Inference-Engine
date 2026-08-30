#!/usr/bin/env bash
# Launch vLLM on the box with a given argument set, wait for health, and print the
# configuration the server actually resolved. Runs ON the box.
#
# Every Phase 5 comparison is a matched pair, and Phase 3's trap was that two identical
# launches produced KV budgets 28% apart. So the KV size, the resolved speculative
# config, and the rejection sampling method are captured from THIS run's own log, never
# assumed. `rejection_sample_method` matters because 'synthetic' fabricates acceptance
# and would report beautiful fictional numbers with no error.
#
#   ./vllm-launch.sh <label> [vllm args...]
set -uo pipefail
LABEL="${1:?usage: vllm-launch.sh <label> [args...]}"; shift
VENV=/opt/llm/.venv-vllm

sudo systemctl stop vllm 2>/dev/null
for i in $(seq 1 30); do
    ss -ltn 2>/dev/null | grep -q ':8000 ' || break
    sleep 1
done

sudo systemd-run --unit=vllm --collect --working-directory=/opt/llm \
  --setenv=HF_HOME=/opt/llm/hf-cache --setenv=HF_HUB_OFFLINE=1 --setenv=PYTHONUNBUFFERED=1 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --setenv=PATH=$VENV/bin:/usr/local/bin:/usr/bin:/bin \
  $VENV/bin/python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port 8000 "$@" >/dev/null 2>&1

ok=0
for i in $(seq 1 120); do
    if curl -sf -m 3 http://localhost:8000/health >/dev/null 2>&1; then ok=1; break; fi
    st=$(systemctl is-active vllm)
    if [ "$st" = "failed" ] || [ "$st" = "inactive" ]; then
        echo "LAUNCH_FAILED $LABEL (unit $st after ${i}s)"
        sudo journalctl -u vllm --no-pager -o cat | tail -30
        exit 1
    fi
    sleep 5
done
[ "$ok" = 1 ] || { echo "LAUNCH_TIMEOUT $LABEL"; exit 1; }

L=$(sudo journalctl -u vllm --no-pager -o cat)
echo "### $LABEL READY"
echo "  args: $*"
echo "$L" | grep -oE 'GPU KV cache size: [0-9,]+ tokens'                  | tail -1 | sed 's/^/  /'
echo "$L" | grep -oE 'Available KV cache memory: [0-9.]+ GiB'             | tail -1 | sed 's/^/  /'
echo "$L" | grep -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' | tail -1 | sed 's/^/  /'
echo "$L" | grep -oE 'speculative_config=SpeculativeConfig\([^)]*\)|speculative_config=None' | tail -1 | sed 's/^/  /'
echo "$L" | grep -oE "rejection_sample_method='[a-z]*'"                   | tail -1 | sed 's/^/  /'
echo "  vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

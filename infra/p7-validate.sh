#!/usr/bin/env bash
# Phase 7 instrument validation. Launches the phase's one configuration, records what
# the server actually resolved, then runs the three checks the offline fakes cannot.
# Run 1 left VLLM_USE_V2_MODEL_RUNNER unset and the engine answered: V2 is the 0.27.1
# default and rejects thinking_token_budget with a 400. It is now set explicitly.
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv          # absolute: systemd-run is root, ~ is /root
MODEL=Qwen/Qwen3-8B
KV_PIN_BYTES=10213733807
LABEL=p7v
mkdir -p results

echo "=== launching $LABEL ==="
VLLM_USE_V2_MODEL_RUNNER=0 KV_PIN="$KV_PIN_BYTES" ./infra/vllm-launch.sh "$LABEL" \
    --model "$MODEL" --quantization fp8 --max-model-len 16384 \
    --kv-cache-dtype fp8 --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}'
rc=$?
if [ "$rc" != 0 ]; then echo "LAUNCH FAILED rc=$rc"; exit 1; fi

# vllm-launch.sh greps five chosen patterns and is blind to the rest (incident 43).
# These two decide whether the budget can bind at all, so read them separately.
echo
echo "=== what the server resolved ==="
INV=$(systemctl show vllm --property=InvocationID --value 2>/dev/null)
LOG=$(sudo journalctl "_SYSTEMD_INVOCATION_ID=$INV" --no-pager -o cat 2>/dev/null)
echo "$LOG" | grep -oiE "model runner[^,]*|V2ModelRunner|GPUModelRunner|use_v2[^,]*" | sort -u | head -5 | sed 's/^/  runner: /'
echo "$LOG" | grep -oiE "reasoning[_-]parser[^,]*|ReasoningConfig\([^)]*\)" | sort -u | head -3 | sed 's/^/  reasoning: /'
echo "$LOG" | grep -oiE "async[_ ]scheduling[^,]*" | sort -u | head -2 | sed 's/^/  sched: /'
echo "  env VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-<unset>}"

echo
echo "=== checks ==="
HF_HOME=/opt/llm/hf-cache HF_HUB_OFFLINE=1 \
  "$UV" run tools/p7validate.py --url http://localhost:8000 --out results/p7-validate.jsonl
crc=$?

echo
echo "=== done, rc=$crc ==="
exit "$crc"

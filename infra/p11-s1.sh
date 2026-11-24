#!/usr/bin/env bash
# Phase 11 S1 on the box: launch one model with tool calling, run the probes, then five dev
# tasks per arm through the real harness. Run detached, one model per invocation:
#   sudo systemd-run --unit=p11-s1-qwen --collect --working-directory=/opt/llm /bin/bash /opt/llm/infra/p11-s1.sh qwen
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv
QWEN=Qwen/Qwen3-8B
XLAM=Salesforce/Llama-xLAM-2-8b-fc-r
# The same five dev tasks for every arm: database, workflow, fuzzy text, the stage none-rule, privacy.
TYPES=handle_time,case_routing,knowledge_qa,wrong_stage_rectification,private_customer_information
COMMON=(--quantization fp8 --max-model-len 32768 --kv-cache-dtype fp8 --enable-prefix-caching
        --enable-prompt-tokens-details --enable-auto-tool-choice)

tasks() {
    local arm="$1" model="$2" protocol="${3:-react_text_v0}" rid="p11-s1-$1"
    [ "$protocol" = tools_v1 ] && rid="p11-s1b-$1-tools"
    $UV run tools/crmeval.py run --url http://localhost:8000 --arm "$arm" --model "$model" --org b2b \
        --split dev --types "$TYPES" --limit 1 --concurrency 1 --run-id "$rid" --protocol "$protocol" \
        || echo "P11_TASKS_FAIL $arm"
}

launch_qwen() {
    VLLM_USE_V2_MODEL_RUNNER=0 KV_PIN=10213733807 ./infra/vllm-launch.sh p11-qwen --model "$QWEN" "${COMMON[@]}" \
        --reasoning-parser qwen3 \
        --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}' \
        --tool-call-parser hermes
}

case "${1:-}" in
  qwen-tools)
    launch_qwen || { echo "P11_LAUNCH_FAIL qwen"; exit 1; }
    tasks q-off "$QWEN" tools_v1
    tasks q-think "$QWEN" tools_v1
    ;;
  qwen)
    launch_qwen || { echo "P11_LAUNCH_FAIL qwen"; exit 1; }
    $UV run tools/p11probe.py qwen --url http://localhost:8000 --out results/p11-s1-probe-qwen.json \
        || echo "P11_PROBE_FAIL qwen"
    tasks q-off "$QWEN"
    tasks q-think "$QWEN"
    ;;
  xlam)
    snaps=$(ls /opt/llm/hf-cache/hub/models--Salesforce--Llama-xLAM-2-8b-fc-r/snapshots/*/model-0000?-of-00004.safetensors 2>/dev/null | wc -l)
    if [ "$snaps" -ne 4 ] || systemctl is-active --quiet xlam-dl; then
        echo "P11_LAUNCH_FAIL xlam: download incomplete ($snaps of 4 shards, xlam-dl $(systemctl is-active xlam-dl))"
        exit 1
    fi
    VLLM_USE_V2_MODEL_RUNNER=0 KV_PIN=10213733807 ./infra/vllm-launch.sh p11-xlam --model "$XLAM" "${COMMON[@]}" \
        --tool-call-parser xlam \
        || { echo "P11_LAUNCH_FAIL xlam"; exit 1; }
    $UV run tools/p11probe.py xlam --url http://localhost:8000 --out results/p11-s1-probe-xlam.json \
        || echo "P11_PROBE_FAIL xlam"
    tasks xlam "$XLAM"
    ;;
  *) echo "usage: p11-s1.sh qwen|qwen-tools|xlam"; exit 2 ;;
esac
echo "P11_S1_DONE ${1}"
exit 0

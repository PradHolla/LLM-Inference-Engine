#!/usr/bin/env bash
# Priority actuator check. vLLM's own bench generates the load -- it handles concurrency,
# arrival rates and unique prompts already, and reinventing that cost four iterations.
# Our tool only fires matched high/low pairs into the queue it creates.
#   ./p7-priority.sh
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv
BENCH=/opt/llm/.venv-vllm/bin/vllm
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[$(ts)] starting load: vllm bench serve, random dataset, well past capacity"
sudo systemd-run --unit=p7load --collect --working-directory=/opt/llm \
    --setenv=HF_HOME=/opt/llm/hf-cache --setenv=HF_HUB_OFFLINE=1 \
    "$BENCH" bench serve --backend openai-chat --model Qwen/Qwen3-8B \
        --endpoint /v1/chat/completions \
        --dataset-name random --num-prompts 400 --request-rate 30 \
        --max-concurrency 128 >/dev/null 2>&1
sleep 8

echo "[$(ts)] measuring matched pairs"
"$UV" run tools/p7prio.py --url http://localhost:8000 --pairs 6 \
    --out /opt/llm/results/p7-priority.jsonl
rc=$?

echo "[$(ts)] stopping load"
sudo systemctl stop p7load 2>/dev/null
echo "[$(ts)] P7_PRIORITY_DONE rc=$rc"
exit 0

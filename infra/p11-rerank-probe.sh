#!/usr/bin/env bash
# Measure bge-reranker-base on an otherwise idle GPU, then relaunch Qwen. Run detached:
#   sudo systemd-run --unit=p11-rerank-probe --collect --working-directory=/opt/llm /bin/bash /opt/llm/infra/p11-rerank-probe.sh
# Needs /tmp/probe.json from: uv run tools/rerank_gpu_probe.py prep /tmp/probe.json
set -uo pipefail
cd /opt/llm || exit 1
PY=/opt/llm/.venv-vllm/bin/python
UV=/home/ubuntu/.local/bin/uv
export HF_HOME=/opt/llm/hf-cache

[ -s /tmp/probe.json ] || { echo "PROBE_FAIL: /tmp/probe.json missing"; exit 1; }
HF_HUB_OFFLINE=0 $PY -c "from huggingface_hub import snapshot_download as d; \
d('BAAI/bge-reranker-base', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])" \
    || { echo "PROBE_FAIL: download"; exit 1; }
echo "PROBE downloaded"

systemctl stop vllm
freed=no
for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 1000 ] && { freed=yes; break; }
    sleep 2
done
echo "PROBE gpu freed=$freed used=${used}MiB"

if [ "$freed" = yes ]; then
    HF_HUB_OFFLINE=1 $UV run --no-project --python "$PY" tools/rerank_gpu_probe.py gpu /tmp/probe.json \
        || echo "PROBE_FAIL: gpu probe"
fi

/bin/bash /opt/llm/infra/p11-app.sh launch qwen
up=$(curl -s -m 5 http://localhost:8000/v1/models | grep -o '"id":"[^"]*"' | head -1)
echo "PROBE_DONE qwen: ${up:-NOT UP}"
exit 0

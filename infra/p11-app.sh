#!/usr/bin/env bash
# The CRM demo on the box: vLLM with tool calling for one model, and the app with the live agent.
#   sudo systemd-run --unit=p11-app-up --collect --working-directory=/opt/llm /bin/bash /opt/llm/infra/p11-app.sh up
#   /opt/llm/infra/p11-app.sh launch qwen|xlam    # also what the app's model picker runs
#   /opt/llm/infra/p11-app.sh app                 # (re)start only the app, live mode, on 127.0.0.1:8090
set -uo pipefail
cd /opt/llm || exit 1
PY=/opt/llm/.venv/bin/python
QWEN=Qwen/Qwen3-8B
XLAM=Salesforce/Llama-xLAM-2-8b-fc-r
COMMON=(--quantization fp8 --max-model-len 32768 --kv-cache-dtype fp8 --enable-prefix-caching
        --enable-prompt-tokens-details --enable-auto-tool-choice)

launch() {
    case "$1" in
      qwen) VLLM_USE_V2_MODEL_RUNNER=0 KV_PIN=10213733807 ./infra/vllm-launch.sh p11-qwen --model "$QWEN" \
                "${COMMON[@]}" --reasoning-parser qwen3 \
                --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}' \
                --tool-call-parser hermes ;;
      xlam) VLLM_USE_V2_MODEL_RUNNER=0 KV_PIN=10213733807 ./infra/vllm-launch.sh p11-xlam --model "$XLAM" \
                "${COMMON[@]}" --tool-call-parser xlam ;;
      *) echo "usage: launch qwen|xlam"; return 2 ;;
    esac
}

FASTEMBED_PIN=fastembed==0.9.1
FASTEMBED_CACHE=/opt/llm/hf-cache/fastembed

start_app() {
    # Hybrid knowledge search needs fastembed; without it the app falls back to BM25, so a failure here only warns.
    if ! "$PY" -c "import fastembed" 2>/dev/null; then
        /home/ubuntu/.local/bin/uv pip install --python "$PY" "$FASTEMBED_PIN" || echo "P11_WARN: fastembed install failed"
    fi
    FASTEMBED_CACHE_PATH="$FASTEMBED_CACHE" "$PY" -m crm.retrieval build --org all || echo "P11_WARN: embedding cache build failed"
    systemctl stop llm-app 2>/dev/null
    for i in $(seq 1 20); do ss -ltn | grep -q ':8090 ' || break; sleep 1; done
    systemd-run --unit=llm-app --collect --working-directory=/opt/llm \
        --setenv=PYTHONUNBUFFERED=1 --setenv=FASTEMBED_CACHE_PATH="$FASTEMBED_CACHE" --setenv=CRM_AGENT_LIVE=1 --setenv=CRM_VLLM_URL=http://localhost:8000 \
        "$PY" -m uvicorn app.server:app --host 127.0.0.1 --port 8090 >/dev/null 2>&1
    ok=0
    for i in $(seq 1 30); do
        if curl -sf -m 3 http://localhost:8090/api/crm/models >/dev/null; then ok=1; break; fi
        sleep 1
    done
    [ "$ok" = 1 ] || { echo "P11_APP_FAIL: app never answered on 8090"; return 1; }
    echo "P11_APP_UP $(curl -s http://localhost:8090/api/crm/models | head -c 200)"
}

case "${1:-}" in
  launch) launch "${2:-}" || { echo "P11_LAUNCH_FAIL ${2:-}"; exit 1; } ;;
  app) start_app || exit 1 ;;
  up)
    served=$(curl -s -m 3 http://localhost:8000/v1/models | grep -o "$QWEN" | head -1)
    if [ "$served" != "$QWEN" ]; then launch qwen || { echo "P11_LAUNCH_FAIL qwen"; exit 1; }; fi
    start_app || exit 1
    ;;
  *) echo "usage: p11-app.sh up | app | launch qwen|xlam"; exit 2 ;;
esac
echo "P11_APP_DONE ${1}"
exit 0

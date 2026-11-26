#!/usr/bin/env bash
# The CRM demo on the box: vLLM with tool calling for one model, and the app with the live agent.
#   sudo systemd-run --unit=p11-app-up --collect --working-directory=/opt/llm /bin/bash /opt/llm/infra/p11-app.sh up
#   /opt/llm/infra/p11-app.sh launch qwen|xlam    # also what the app's model picker runs
#   /opt/llm/infra/p11-app.sh app                 # (re)start only the app, live mode, on 127.0.0.1:8090
#   /opt/llm/infra/p11-app.sh rerank              # (re)start only the GPU reranker on 127.0.0.1:8010
#   /opt/llm/infra/p11-app.sh gateway             # (re)start only the chat gateway on 127.0.0.1:8080
set -uo pipefail
cd /opt/llm || exit 1
PY=/opt/llm/.venv/bin/python
QWEN=Qwen/Qwen3-8B
XLAM=Salesforce/Llama-xLAM-2-8b-fc-r
COMMON=(--quantization fp8 --max-model-len 32768 --kv-cache-dtype fp8 --enable-prefix-caching
        --enable-prompt-tokens-details --enable-auto-tool-choice)

# KV pin lowered by 1.25 GiB (from 10213733807) to leave room for the GPU reranker (1,119 MiB measured).
KV_PIN_BYTES=8871556527
RERANK_URL=http://127.0.0.1:8010

start_rerank() {
    # GPU reranker for knowledge search; if it fails, search falls back to the CPU reranker, so this only warns.
    systemctl stop p11-rerank 2>/dev/null
    systemd-run --unit=p11-rerank --collect --working-directory=/opt/llm \
        --setenv=HF_HOME=/opt/llm/hf-cache --setenv=HF_HUB_OFFLINE=1 --setenv=PYTHONUNBUFFERED=1 \
        /opt/llm/.venv-vllm/bin/python -m uvicorn crm.rerank_gpu:app --host 127.0.0.1 --port 8010 >/dev/null 2>&1
    ok=0
    for i in $(seq 1 60); do
        if curl -sf -m 3 "$RERANK_URL/health" >/dev/null; then ok=1; break; fi
        sleep 2
    done
    if [ "$ok" = 1 ]; then echo "P11_RERANK_UP $(curl -s "$RERANK_URL/health")"
    else echo "P11_WARN: GPU reranker not up; knowledge search uses the CPU reranker"; fi
    return 0
}

start_gateway() {
    # Chat gateway for the app's chat mode; if it fails, the CRM demo still comes up, so this only warns.
    systemctl stop gateway 2>/dev/null
    for i in $(seq 1 20); do ss -ltn | grep -q ':8080 ' || break; sleep 1; done
    mkdir -p /opt/llm/logs
    systemd-run --unit=gateway --collect --working-directory=/opt/llm \
        --setenv=HF_HOME=/opt/llm/hf-cache --setenv=HF_HUB_OFFLINE=1 \
        --setenv=GW_UPSTREAM=http://localhost:8000 --setenv=GW_METRICS=http://localhost:8000 \
        --setenv=GW_TRACE=/opt/llm/logs/p11-gw-trace.jsonl \
        "$PY" -m uvicorn gateway.app:app --host 127.0.0.1 --port 8080 >/dev/null 2>&1
    ok=0
    for i in $(seq 1 40); do
        if curl -sf -m 3 http://localhost:8080/health >/dev/null; then ok=1; break; fi
        sleep 1
    done
    if [ "$ok" = 1 ]; then echo "P11_GATEWAY_UP"
    else echo "P11_WARN: gateway not up; chat mode will not work"; fi
    return 0
}

launch() {
    # vllm-launch.sh refuses to start while the GPU holds >= 1 GiB, and the reranker holds 1.1 GiB: stop it first.
    systemctl stop p11-rerank 2>/dev/null
    local rc=0
    case "$1" in
      qwen) VLLM_USE_V2_MODEL_RUNNER=0 KV_PIN=$KV_PIN_BYTES ./infra/vllm-launch.sh p11-qwen --model "$QWEN" \
                "${COMMON[@]}" --reasoning-parser qwen3 \
                --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}' \
                --tool-call-parser hermes || rc=1 ;;
      xlam) VLLM_USE_V2_MODEL_RUNNER=0 KV_PIN=$KV_PIN_BYTES ./infra/vllm-launch.sh p11-xlam --model "$XLAM" \
                "${COMMON[@]}" --tool-call-parser xlam || rc=1 ;;
      *) echo "usage: launch qwen|xlam"; return 2 ;;
    esac
    [ "$rc" = 0 ] || return 1
    start_rerank
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
        --setenv=CRM_RERANK_URL="$RERANK_URL" \
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
  rerank) start_rerank ;;
  gateway) start_gateway ;;
  up)
    served=$(curl -s -m 3 http://localhost:8000/v1/models | grep -o "$QWEN" | head -1)
    if [ "$served" != "$QWEN" ]; then launch qwen || { echo "P11_LAUNCH_FAIL qwen"; exit 1; }; fi
    start_gateway
    start_app || exit 1
    ;;
  *) echo "usage: p11-app.sh up | app | rerank | gateway | launch qwen|xlam"; exit 2 ;;
esac
echo "P11_APP_DONE ${1}"
exit 0

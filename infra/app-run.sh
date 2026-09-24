#!/usr/bin/env bash
# Phase 6b: vLLM, gateway and app on the box, then drive the app through its own API.
# Runs ON the box, always inside a unit, never a foreground ssh session:
#   sudo systemd-run --unit=app-up --collect /bin/bash /opt/llm/infra/app-run.sh up
#   ./infra/app-run.sh up | labbench | smoke | drive | report | down
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv          # absolute: systemd-run is root, ~ is /root
PY=/opt/llm/.venv/bin/python
MODEL=Qwen/Qwen3-8B
# P6B's files are scored and committed; later sessions write their own.
GWTRACE=results/app-gw.jsonl
APPOUT=results/app-drive.jsonl
export HF_HOME=/opt/llm/hf-cache HF_HUB_OFFLINE=1
ts() { date -u +%H:%M:%S; }
die() { echo "ABORT: $*"; exit 1; }

# The Phase 7 server with a 32k window (6c design O1; Phase 7 and P6B ran 16k). phase7-runbook.md.
# --enable-prompt-tokens-details only adds usage.prompt_tokens_details.cached_tokens per request.
launch_vllm() {
    VLLM_USE_V2_MODEL_RUNNER=0 KV_PIN=10213733807 ./infra/vllm-launch.sh app \
        --model "$MODEL" --quantization fp8 --max-model-len 32768 \
        --kv-cache-dtype fp8 --enable-prefix-caching --reasoning-parser qwen3 \
        --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}' \
        --enable-prompt-tokens-details \
        || die "vllm launch failed"
}

# Both of these fail SILENTLY if wrong: the levels look fine and behave identically, or the
# thinking block stays empty. So they kill the run instead of being remembered.
require_vllm() {
    # Capture, then match. `curl | grep -q` under pipefail fails a HEALTHY server: grep exits
    # on the first match and curl dies of SIGPIPE writing the rest.
    local env inv endstr stream
    curl -sf -m 5 http://localhost:8000/health >/dev/null || die "no vLLM on :8000"
    env=$(systemctl show vllm --property=Environment --value 2>/dev/null)
    [[ " $env " == *" VLLM_USE_V2_MODEL_RUNNER=0 "* ]] \
        || die "vllm unit lacks VLLM_USE_V2_MODEL_RUNNER=0; the thinking budget would not bind"
    inv=$(systemctl show vllm --property=InvocationID --value 2>/dev/null)
    endstr=$(sudo journalctl "_SYSTEMD_INVOCATION_ID=$inv" --no-pager -o cat \
        | grep -o "reasoning_end_str=[^,)]*" | tail -1)
    [ "$endstr" = "reasoning_end_str='</think>'" ] \
        || die "server is not on the bare </think> stop string (found: ${endstr:-nothing})"
    stream=$(curl -sN -m 60 http://localhost:8000/v1/chat/completions \
        -H 'content-type: application/json' \
        -d "{\"model\":\"$MODEL\",\"stream\":true,\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")
    [[ "$stream" =~ \"reasoning\":\ ?\" ]] \
        || die "no delta.reasoning in a thinking stream; --reasoning-parser qwen3 is not working"
    echo "  [$(ts)] vLLM up: V2 runner off, bare </think>, reasoning arrives in delta.reasoning"
}

wait_http() {   # $1 url, $2 tries
    local i
    for i in $(seq 1 "$2"); do
        curl -sf -m 2 "$1" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

wait_port_free() {
    local i
    for i in $(seq 1 20); do
        ss -ltn 2>/dev/null | grep -q ":$1 " || return 0
        sleep 1
    done
    return 1
}

start_gateway() {
    sudo systemctl stop gateway >/dev/null 2>&1
    wait_port_free 8080 || die "port 8080 still bound after the gateway stopped"
    sudo systemd-run --unit=gateway --collect --working-directory=/opt/llm \
        --setenv=HF_HOME=/opt/llm/hf-cache --setenv=HF_HUB_OFFLINE=1 \
        --setenv=GW_UPSTREAM=http://localhost:8000 --setenv=GW_METRICS=http://localhost:8000 \
        --setenv=GW_TRACE="$GWTRACE" \
        "$PY" -m uvicorn gateway.app:app --host 127.0.0.1 --port 8080 \
        >/dev/null 2>&1 || die "gateway unit failed to start"
    wait_http http://localhost:8080/health 40 || die "gateway never became ready"
    echo "  [$(ts)] gateway up, tracing to $GWTRACE"
}

start_app() {
    sudo systemctl stop llm-app >/dev/null 2>&1
    wait_port_free 8090 || die "port 8090 still bound after the app stopped"
    sudo systemd-run --unit=llm-app --collect --working-directory=/opt/llm \
        --setenv=PYTHONUNBUFFERED=1 \
        "$PY" -m uvicorn app.server:app --host 127.0.0.1 --port 8090 \
        >/dev/null 2>&1 || die "app unit failed to start"
    wait_http http://localhost:8090/api/health 30 || die "app never became ready"
    [[ "$(curl -sf -m 5 http://localhost:8090/api/health)" == *'"gateway":true'* ]] \
        || die "app is up but cannot reach the gateway"
    echo "  [$(ts)] app up on 127.0.0.1:8090"
}

require_stack() {
    require_vllm
    wait_http http://localhost:8080/health 3 || die "no gateway on :8080; run: $0 up"
    wait_http http://localhost:8090/api/health 3 || die "no app on :8090; run: $0 up"
}

smoke() {
    echo "[$(ts)] smoke: one question at each thinking level, search off"
    "$UV" run tools/appdrive.py run --mode smoke --out results/app-smoke.jsonl \
        || die "smoke failed; the app is not fit to measure"
}

case "${1:-}" in
up)
    echo "[$(ts)] 6b up"
    [ -s /opt/llm/.brave-key ] || die "no /opt/llm/.brave-key; search would answer with nothing"
    # The box venv predates the app. Install as ubuntu, who owns the venv, not as root.
    # langgraph is pinned to match pyproject.toml's serve group (6c).
    "$PY" -c "import sse_starlette, langgraph, tokenizers" 2>/dev/null \
        || sudo -u ubuntu "$UV" pip install --python "$PY" sse-starlette langgraph==1.2.12 tokenizers \
        || die "app dependencies missing from $PY and could not be installed"
    "$PY" -c "import importlib.metadata as m; print('  app deps:', *(f'{p} {m.version(p)}' for p in ('sse-starlette', 'fastapi', 'uvicorn', 'langgraph', 'tokenizers')))" \
        || die "app dependencies do not import"
    sudo systemctl stop llm-app gateway >/dev/null 2>&1
    launch_vllm
    require_vllm
    start_gateway
    start_app
    smoke
    echo "UP_OK  browser: ssh -L 8090:localhost:8090 ubuntu@<ip>, then http://localhost:8090"
    ;;
smoke)
    require_stack
    smoke
    ;;
drive)
    require_stack
    echo "[$(ts)] P6B-1: one conversation, 10 turns, brief, search on"
    "$UV" run tools/appdrive.py run --mode convo --out "$APPOUT" || echo "  convo had failed sends"
    echo "[$(ts)] P6B-2: 12 questions x 3 levels, search on"
    "$UV" run tools/appdrive.py run --mode levels --out "$APPOUT" || echo "  levels had failed sends"
    "$UV" run tools/appdrive.py report --app-out "$APPOUT" --gw-trace "$GWTRACE" \
        | tee results/app-report.txt
    echo "DRIVE_DONE"
    ;;
report)
    "$UV" run tools/appdrive.py report --app-out "$APPOUT" --gw-trace "$GWTRACE" \
        | tee results/app-report.txt
    ;;
labbench)
    # :8081 because the gateway holds :8080. Watches the same vLLM the app uses; its backend
    # switcher would relaunch vLLM without this recipe's flags, so leave it alone.
    require_vllm
    sudo systemctl stop llm-labbench >/dev/null 2>&1
    wait_port_free 8081 || die "port 8081 still bound after the lab bench stopped"
    sudo systemd-run --unit=llm-labbench --collect --working-directory=/opt/llm \
        --setenv=HF_HOME=/opt/llm/hf-cache --setenv=HF_HUB_OFFLINE=1 \
        --setenv=LABBENCH_UPSTREAM=http://localhost:8000 --setenv=LABBENCH_UV="$UV" \
        "$PY" -m uvicorn labbench.server:app --host 127.0.0.1 --port 8081 \
        >/dev/null 2>&1 || die "lab bench unit failed to start"
    wait_http http://localhost:8081/health 30 || die "lab bench never became ready"
    echo "LABBENCH_OK  browser: add -L 8081:localhost:8081 to the tunnel, then http://localhost:8081"
    ;;
down)
    sudo systemctl stop llm-labbench llm-app gateway vllm >/dev/null 2>&1
    echo "[$(ts)] lab bench, app, gateway and vLLM stopped; the box is still running -- ./infra/down.sh"
    ;;
*)
    echo "usage: $0 up | labbench | smoke | drive | report | down"; exit 2 ;;
esac
exit 0

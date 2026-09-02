#!/usr/bin/env bash
# Protocol gate: bench.py straight at the engine vs through the lab bench proxy.
# The two must agree, or the proxy is changing a measurement. Runs ON the box.
#   ./labbench-gate.sh
set -uo pipefail
cd /opt/llm
UV=/home/ubuntu/.local/bin/uv

# bench.py defaults to --model test and vLLM 404s it (runbook section 2). Ask, don't assume.
MODEL=$(curl -s http://localhost:8000/v1/models \
        | python3 -c 'import json,sys; d=json.load(sys.stdin).get("data") or []; print(d[0]["id"] if d else "")' 2>/dev/null)
if [ -z "$MODEL" ]; then echo "GATE_ABORT: could not resolve the served model from /v1/models"; exit 1; fi
echo "served model: $MODEL"

for t in "direct 8000" "proxied 8080"; do
    set -- $t
    echo "=== $1 (port $2)"
    $UV run tools/bench.py --url "http://localhost:$2" --model "$MODEL" \
        --serial 12 --warmup 1 --prompt-tokens 512 --max-tokens 64 \
        --out "results/gate-$1.jsonl" 2>&1 | tail -10
done
echo GATE_DONE

#!/usr/bin/env bash
# Phase 7 Q1a: the accuracy-and-latency-versus-thinking-budget curve on gsm8k.
# Assumes the P7V server is already up with VLLM_USE_V2_MODEL_RUNNER=0. Flushes one
# result file per arm so a crash costs one arm, not the sweep.
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv          # absolute: systemd-run is root
MAXTOK=3000
mkdir -p results

if ! curl -sf -m 5 http://localhost:8000/health >/dev/null; then
    echo "ABORT: no server on :8000"; exit 1
fi
# The budget silently does nothing on Model Runner V2, so refuse to spend an hour
# producing a flat curve that would read as "budgets do not matter".
ENVLINE=$(systemctl show vllm --property=Environment --value 2>/dev/null)
case "$ENVLINE" in
    *VLLM_USE_V2_MODEL_RUNNER=0*) echo "  runner: V2 disabled, budget can bind" ;;
    *) echo "ABORT: server unit does not have VLLM_USE_V2_MODEL_RUNNER=0"; exit 1 ;;
esac

for b in 0 128 256 512 1024 2048 none; do
    if [ "$b" = none ]; then
        label=unbounded; ARG=""
    else
        label="b$b"; ARG="--thinking-budget $b"
    fi
    out="results/p7q1a-$label.jsonl"
    echo
    echo "=== arm $label -> $out ==="
    HF_HOME=/opt/llm/hf-cache HF_HUB_OFFLINE=1 \
      "$UV" run tools/qualeval.py run --url http://localhost:8000 \
        --config "q1a-$label" --slices gsm8k --concurrency 32 \
        --max-tokens-think "$MAXTOK" $ARG --out "$out"
    rc=$?
    echo "  arm $label rc=$rc"
    [ "$rc" = 0 ] && "$UV" run tools/qualeval.py grade "$out" 2>&1 | tail -6
done

echo
echo "=== all arms done ==="
exit 0

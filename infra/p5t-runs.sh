#!/usr/bin/env bash
# Speculative decoding vs sampling temperature: Phase 5's P5-I recipe (fp8, EAGLE3 k=3,
# concurrency 1, 12 items per slice) at T = 0, 0.3, 0.6, 1.0. Runs ON the box; replaces the
# app's vLLM, so run it AFTER anything that needs the app stack.
#   sudo systemd-run --unit=p5t --collect /bin/bash /opt/llm/infra/p5t-runs.sh all
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv
EAGLE='{"model":"RedHatAI/Qwen3-8B-speculator.eagle3","method":"eagle3","num_speculative_tokens":3}'
TEMPS="${TEMPS:-0 0.3 0.6 1.0}"
ts() { date -u +%H:%M:%S; }
die() { echo "ABORT: $*"; exit 1; }

runs() {   # arm: spec | control
    local arm=$1 slice t label sampling
    for slice in gsm8k math; do
        for t in $TEMPS; do
            label="p5t-$arm-$slice-t$t"
            sampling=(--temperature "$t")
            # T > 0 uses the Qwen3 thinking card's other values; T = 0 is P5-I verbatim.
            [ "$t" != 0 ] && sampling+=(--top-p 0.95 --top-k 20 --min-p 0)
            rm -f "results/$label.jsonl"
            local cmd=("$UV" run tools/qualeval.py run --url http://localhost:8000 --config "$label"
                       --items results/phase4-items.jsonl --concurrency 1 --out "results/$label.jsonl"
                       --slices "$slice" --max-tokens-think 5120
                       --limit-pass "$slice:think:12,$slice:nothink:0" "${sampling[@]}")
            echo "[$(ts)] $label"
            if [ "$arm" = spec ]; then
                "$UV" run tools/specmon.py wrap --url http://localhost:8000 --label "$label" \
                    --out results/p5t-spec.jsonl -- "${cmd[@]}" > "results/$label.log" 2>&1 \
                    || echo "  $label failed rc=$?"
            else
                "${cmd[@]}" > "results/$label.log" 2>&1 || echo "  $label failed rc=$?"
            fi
            tail -2 "results/$label.log" | sed 's/^/  /'
        done
    done
}

case "${1:-}" in
spec)
    ./infra/vllm-launch.sh p5t-spec --model Qwen/Qwen3-8B --quantization fp8 \
        --max-model-len 16384 --speculative-config "$EAGLE" > /tmp/p5t-spec-launch.log 2>&1 \
        || { tail -20 /tmp/p5t-spec-launch.log; die "spec launch"; }
    grep -oE 'speculative_config=[A-Za-z]+|GPU KV cache size: [0-9,]+ tokens' /tmp/p5t-spec-launch.log | tail -2
    runs spec
    echo "SPEC_DONE"
    ;;
control)
    ./infra/vllm-launch.sh p5t-control --model Qwen/Qwen3-8B --quantization fp8 \
        --max-model-len 16384 > /tmp/p5t-control-launch.log 2>&1 \
        || { tail -20 /tmp/p5t-control-launch.log; die "control launch"; }
    grep -oE 'speculative_config=[A-Za-z]+|GPU KV cache size: [0-9,]+ tokens' /tmp/p5t-control-launch.log | tail -2
    runs control
    echo "CONTROL_DONE"
    ;;
all)
    bash "$0" spec || die "spec arm"
    bash "$0" control || die "control arm"
    echo "P5T_DONE"
    ;;
*)
    echo "usage: $0 spec | control | all"; exit 2 ;;
esac
exit 0

#!/usr/bin/env bash
# Phase 9: perplexity vs real quantization damage (bf16 / fp8 / int4), and the JSON schema's cost
# to the planner. Runs ON the box with no vLLM server up. Each config smoke-tests 5 units first.
#   sudo systemd-run --unit=p9 --collect /bin/bash /opt/llm/infra/p9-runs.sh all
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv          # absolute: systemd-run is root, ~ is /root
export HF_HOME=/opt/llm/hf-cache PYTHONUNBUFFERED=1
export PATH="/opt/llm/.venv-vllm/bin:$PATH"   # vLLM JIT-builds kernels with ninja (incident 38)
ts() { date -u +%H:%M:%S; }
die() { echo "ABORT: $*"; exit 1; }

ppl_config() {   # label, then vllm-launch args
    local label=$1; shift
    HF_HUB_OFFLINE=1 ./infra/vllm-launch.sh "p9-$label" "$@" --max-model-len 8192 \
        > "/tmp/p9-$label-launch.log" 2>&1 || { tail -20 "/tmp/p9-$label-launch.log"; die "launch $label"; }
    for corpus in wiki k32; do
        rm -f "/tmp/p9-smoke-$label.jsonl"
        "$UV" run tools/p9eval.py ppl --label "$label" --corpus "$corpus" --n 5 \
            --out "/tmp/p9-smoke-$label.jsonl" || die "smoke $label $corpus"
        [ "$(wc -l < "/tmp/p9-smoke-$label.jsonl")" -ge 4 ] || die "smoke $label $corpus scored too few units"
        "$UV" run tools/p9eval.py ppl --label "$label" --corpus "$corpus" --out results/p9-ppl.jsonl \
            || die "ppl $label $corpus"
    done
}

case "${1:-}" in
data)
    if [ ! -s data/wikitext2-test.txt ]; then
        HF_HUB_OFFLINE=0 /opt/llm/.venv-eval/bin/python - <<'PY' || die "wikitext download"
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
open("data/wikitext2-test.txt", "w").write("".join(ds["text"]))
PY
    fi
    echo "  wikitext: $(wc -c < data/wikitext2-test.txt) chars"
    [ "$(wc -c < data/wikitext2-test.txt)" -gt 400000 ] || die "wikitext too small"
    [ -s results/phase4-bf16-a.jsonl ] && [ -s results/phase4-items.jsonl ] || die "phase4 inputs missing"
    echo "DATA_DONE"
    ;;
ppl)
    rm -f results/p9-ppl.jsonl
    echo "[$(ts)] bf16";  ppl_config bf16 --model Qwen/Qwen3-8B
    echo "[$(ts)] fp8";   ppl_config fp8  --model Qwen/Qwen3-8B --quantization fp8
    echo "[$(ts)] int4";  ppl_config int4 --model RedHatAI/Qwen3-8B-quantized.w4a16
    echo "PPL_DONE"
    ;;
json)
    # The app's server, with prefix caching OFF so both arms of each pair prefill cold.
    rm -f results/p9-json.jsonl /tmp/p9-json-smoke.jsonl
    VLLM_USE_V2_MODEL_RUNNER=0 KV_PIN=10213733807 HF_HUB_OFFLINE=1 ./infra/vllm-launch.sh p9-json \
        --model Qwen/Qwen3-8B --quantization fp8 --max-model-len 32768 --kv-cache-dtype fp8 \
        --no-enable-prefix-caching --reasoning-parser qwen3 \
        --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}' \
        > /tmp/p9-json-launch.log 2>&1 || { tail -20 /tmp/p9-json-launch.log; die "launch json"; }
    "$UV" run tools/p9eval.py json --n 3 --out /tmp/p9-json-smoke.jsonl || die "json smoke"
    [ "$(grep -c '"phase": "pair"' /tmp/p9-json-smoke.jsonl)" -eq 6 ] || die "json smoke wrote the wrong count"
    "$UV" run tools/p9eval.py json --n 100 --out results/p9-json.jsonl || die "json"
    echo "JSON_DONE"
    ;;
report)
    "$UV" run tools/p9eval.py report || die "report"
    echo "REPORT_DONE"
    ;;
all)
    for step in data ppl json report; do
        echo "[$(ts)] STEP $step"
        bash "$0" "$step" || die "step $step failed"
    done
    sudo systemctl stop vllm >/dev/null 2>&1
    echo "P9_DONE"
    ;;
*)
    echo "usage: $0 data | ppl | json | report | all"; exit 2 ;;
esac
exit 0

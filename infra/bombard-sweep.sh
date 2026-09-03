#!/usr/bin/env bash
# Bombard sweep: {bf16,fp8,int4} x {spec off, eagle3 k=2}, vLLM only. Runs ON the box.
# Each config records the KV budget from its OWN startup log, because a capacity
# comparison is not attributable until both budgets are known (Phase 3, 28% start spread).
#   ./bombard-sweep.sh [duration_per_rate]
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv
DUR="${1:-45}"
OUT=results/phase6-bombard.jsonl
# Real prompts, not filler. Phase 5 measured filler as the worst possible content for
# speculation, so a spec sweep on filler measures the technique at its weakest.
PROMPTS="${PROMPTS:-results/phase4-items.jsonl}"
SUM=results/phase6-bombard-summary.txt
EAGLE='{"model":"RedHatAI/Qwen3-8B-speculator.eagle3","method":"eagle3","num_speculative_tokens":2}'
: > "$SUM"

[ -x "$UV" ] || { echo "ABORT: uv missing at $UV"; exit 1; }
[ -s "$PROMPTS" ] || { echo "ABORT: prompt file $PROMPTS missing or empty"; exit 1; }
echo "prompts: $(wc -l < "$PROMPTS") from $PROMPTS"

for quant in bf16 fp8 int4; do
  for spec in off on; do
    label="$quant-spec$spec"
    model=Qwen/Qwen3-8B; qflag=""
    [ "$quant" = fp8 ]  && qflag="--quantization fp8"
    [ "$quant" = int4 ] && model=RedHatAI/Qwen3-8B-quantized.w4a16
    sflag=""; [ "$spec" = on ] && sflag="--speculative-config $EAGLE"

    echo "########## $label"
    if ! ./infra/vllm-launch.sh "$label" --model "$model" --max-model-len 16384 \
           $qflag $sflag > /tmp/launch-$label.log 2>&1; then
        echo "LAUNCH FAILED for $label"; tail -20 /tmp/launch-$label.log
        { echo "=== $label  LAUNCH FAILED"; } >> "$SUM"
        continue
    fi
    kv=$(grep -oE 'GPU KV cache size: [0-9,]+ tokens' /tmp/launch-$label.log | tail -1)
    sp=$(grep -oE 'speculative_config=[A-Za-z]+' /tmp/launch-$label.log | tail -1)
    echo "  $kv | $sp"

    served=$(curl -s http://localhost:8000/v1/models \
             | python3 -c 'import json,sys;d=json.load(sys.stdin).get("data") or [];print(d[0]["id"] if d else "")')
    [ -n "$served" ] || { echo "  ABORT: no served model"; continue; }

    { echo "=== $label"; echo "  $kv"; echo "  $sp"; } >> "$SUM"

    echo "--- batch 1 latency"
    $UV run tools/bench.py --url http://localhost:8000 --model "$served" \
        --serial 12 --warmup 1 --prompts-file "$PROMPTS" --max-tokens 128 \
        --out "$OUT" 2>&1 | tail -8 | tee -a "$SUM"

    echo "--- open loop 2 and 6 req/s"
    $UV run tools/bench.py --url http://localhost:8000 --model "$served" \
        --sweep 2,6 --duration "$DUR" --prompts-file "$PROMPTS" --max-tokens 128 \
        --out "$OUT" 2>&1 | tail -8 | tee -a "$SUM"
  done
done
echo SWEEP_DONE

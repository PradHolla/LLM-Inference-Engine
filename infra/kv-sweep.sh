#!/usr/bin/env bash
# Test 1 of the fp8-KV question: does halving KV bytes per token remove the speculation
# crossover? Four cells, all fresh in one session, because identical launches differ ~10%
# in KV budget and a matched comparison cannot span sessions. Runs ON the box.
#   ./kv-sweep.sh [duration_per_rate]
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv
DUR="${1:-45}"
OUT=results/phase6-kvdtype.jsonl
SUM=results/phase6-kvdtype-summary.txt
PROMPTS="${PROMPTS:-results/phase4-items.jsonl}"
EAGLE='{"model":"RedHatAI/Qwen3-8B-speculator.eagle3","method":"eagle3","num_speculative_tokens":2}'
: > "$SUM"

[ -x "$UV" ] || { echo "ABORT: uv missing at $UV"; exit 1; }
[ -s "$PROMPTS" ] || { echo "ABORT: prompts missing at $PROMPTS"; exit 1; }

for kvd in auto fp8; do
  for spec in off on; do
    label="fp8w-kv$kvd-spec$spec"
    kflag=""; [ "$kvd" = fp8 ] && kflag="--kv-cache-dtype fp8"
    sflag=""; [ "$spec" = on ] && sflag="--speculative-config $EAGLE"

    echo "########## $label"
    if ! ./infra/vllm-launch.sh "$label" --model Qwen/Qwen3-8B --max-model-len 16384 \
           --quantization fp8 $kflag $sflag > /tmp/launch-$label.log 2>&1; then
        echo "LAUNCH FAILED $label"; tail -25 /tmp/launch-$label.log
        { echo "=== $label  LAUNCH FAILED"; tail -25 /tmp/launch-$label.log; } >> "$SUM"
        continue
    fi
    kv=$(grep -oE 'GPU KV cache size: [0-9,]+ tokens' /tmp/launch-$label.log | tail -1)
    gb=$(grep -oE 'Available KV cache memory: [0-9.]+ GiB' /tmp/launch-$label.log | tail -1)
    sp=$(grep -oE 'speculative_config=[A-Za-z]+' /tmp/launch-$label.log | tail -1)
    echo "  $kv | $gb | $sp"
    { echo "=== $label"; echo "  $kv"; echo "  $gb"; echo "  $sp"; } >> "$SUM"

    served=$(curl -s http://localhost:8000/v1/models \
             | python3 -c 'import json,sys;d=json.load(sys.stdin).get("data") or [];print(d[0]["id"] if d else "")')
    [ -n "$served" ] || { echo "  ABORT: no served model"; continue; }

    $UV run tools/bench.py --url http://localhost:8000 --model "$served" \
        --serial 12 --warmup 1 --prompts-file "$PROMPTS" --max-tokens 128 \
        --out "$OUT" 2>&1 | tail -8 | tee -a "$SUM"
    $UV run tools/bench.py --url http://localhost:8000 --model "$served" \
        --sweep 2,6 --duration "$DUR" --prompts-file "$PROMPTS" --max-tokens 128 \
        --out "$OUT" 2>&1 | tail -8 | tee -a "$SUM"
  done
done
echo KV_SWEEP_DONE

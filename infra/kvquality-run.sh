#!/usr/bin/env bash
# Test 2 of the fp8-KV question: paired quality eval, fp16 KV against fp8 KV, fp8 weights,
# speculation off in both arms. Phase 4 protocol. Runs ON the box.
#   ./kvquality-run.sh
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv
SLICES="math,gsm8k,longctx"
[ -x "$UV" ] || { echo "ABORT: uv missing"; exit 1; }

# Optional arg: run one arm only, so a failed arm can be redone without repeating the
# hour the other one already cost.
ARMS="${1:-kvfp16 kvfp8}"
for arm in $ARMS; do
    kflag=""; [ "$arm" = kvfp8 ] && kflag="--kv-cache-dtype fp8"
    echo "########## arm $arm"
    if ! ./infra/vllm-launch.sh "qual-$arm" --model Qwen/Qwen3-8B --max-model-len 16384 \
           --quantization fp8 $kflag; then
        echo "ABORT: launch failed for $arm"; exit 1
    fi
    # concurrency 12: Phase 4 used 12 for thinking-heavy runs; 32 risks the rejection
    # sampler's fp32 logits buffer that OOMed in Phase 5.
    $UV run tools/qualeval.py run --url http://localhost:8000 \
        --config "$arm" --slices "$SLICES" --concurrency 12 \
        --out "results/phase6-qual-$arm.jsonl" || { echo "ABORT: eval failed for $arm"; exit 1; }
done

echo "########## compare"
$UV run tools/qualeval.py compare results/phase6-qual-kvfp16.jsonl results/phase6-qual-kvfp8.jsonl \
    | tee results/phase6-qual-compare.txt
echo QUAL_DONE

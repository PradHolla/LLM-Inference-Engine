#!/usr/bin/env bash
# L4 vs A10G: the Phase 6B sweep (spec off) on a g6.2xlarge, replaying the A10G's exact arrivals
# and prompts, plus forced-Marlin fp8 as the native-fp8 control and a 4k-token prefill probe.
#   sudo systemd-run --unit=l4 --collect /bin/bash /opt/llm/infra/l4-runs.sh all
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv          # absolute: systemd-run is root, ~ is /root
export HF_HOME=/opt/llm/hf-cache PYTHONUNBUFFERED=1
export PATH="/opt/llm/.venv-vllm/bin:$PATH"   # vLLM JIT-builds kernels with ninja (incident 38)
PROMPTS=results/phase4-items.jsonl
A10G=results/phase6-bombard.jsonl
KV_BYTES_PER_TOKEN=147456             # 36 layers x K,V x 8 heads x 128 x 2 bytes
ts() { date -u +%H:%M:%S; }
die() { echo "ABORT: $*"; exit 1; }
prompt_tokens() {
    curl -s localhost:8000/metrics | awk '/^vllm:prompt_tokens_total/{s+=$2} END{printf "%d", s}'
}

run_cfg() {   # cfg, A10G KV tokens, A10G replay blocks, then vllm-launch args
    local cfg=$1 kvtok=$2 blocks=$3; shift 3
    local lab="l4-$cfg"
    rm -f "results/$lab-"{bombard.jsonl,prefill.jsonl,smi.csv,marks.txt,launch.txt,prompt-tokens.txt}
    nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu \
        --format=csv -lms 500 > "results/$lab-smi.csv" 2>/dev/null &
    SMI=$!                                  # global: a local is gone by the time EXIT fires
    trap 'kill $SMI 2>/dev/null' EXIT       # each config is its own process, so EXIT covers die too
    if ! KV_PIN=$((kvtok * KV_BYTES_PER_TOKEN)) HF_HUB_OFFLINE=1 ./infra/vllm-launch.sh "$lab" "$@" \
            > "/tmp/$lab-launch.log" 2>&1; then
        echo "  pinned launch failed, retrying with KV profiled:"; tail -12 "/tmp/$lab-launch.log"
        if ! HF_HUB_OFFLINE=1 ./infra/vllm-launch.sh "$lab" "$@" > "/tmp/$lab-launch.log" 2>&1; then
            tail -20 "/tmp/$lab-launch.log"; die "launch $cfg"
        fi
    fi
    sed -n '/READY/,$p' "/tmp/$lab-launch.log" > "results/$lab-launch.txt"
    cat "results/$lab-launch.txt"

    local want="" got
    [ "$cfg" = fp8 ]  && want=CutlassFP8ScaledMMLinearKernel
    [ "$cfg" = fp8m ] && want=MarlinFP8ScaledMMLinearKernel
    if [ -n "$want" ]; then
        got=$(grep -oE 'Selected [A-Za-z0-9]+ for Fp8[A-Za-z]+' "results/$lab-launch.txt" | awk '{print $2}' | sort -u)
        if [ "$got" != "$want" ]; then
            die "$cfg selected '${got:-no kernel line}', expected $want"
        fi
    fi

    local served
    served=$(curl -s localhost:8000/v1/models | python3 -c 'import json,sys;d=json.load(sys.stdin).get("data") or [];print(d[0]["id"] if d else "")')
    [ -n "$served" ] || die "$cfg: no served model"
    rm -f "/tmp/$lab-smoke.jsonl"
    "$UV" run tools/bench.py --model "$served" --serial 2 --warmup 0 --prompts-file "$PROMPTS" \
        --max-tokens 16 --out "/tmp/$lab-smoke.jsonl" > /dev/null 2>&1 || die "$cfg smoke"
    [ "$(grep -c '"status": "ok"' "/tmp/$lab-smoke.jsonl")" -eq 2 ] || die "$cfg smoke: not 2 ok"

    echo "serial $(date '+%Y/%m/%d %H:%M:%S')" >> "results/$lab-marks.txt"
    "$UV" run tools/bench.py --model "$served" --serial 12 --warmup 1 --prompts-file "$PROMPTS" \
        --max-tokens 128 --out "results/$lab-bombard.jsonl" 2>&1 | tail -6 || die "$cfg serial"
    echo "replay $(date '+%Y/%m/%d %H:%M:%S')" >> "results/$lab-marks.txt"
    "$UV" run tools/bench.py --model "$served" --replay "$A10G:$blocks" --prompts-file "$PROMPTS" \
        --duration 45 --max-tokens 128 --out "results/$lab-bombard.jsonl" 2>&1 | tail -6 || die "$cfg replay"
    echo "prefill $(date '+%Y/%m/%d %H:%M:%S')" >> "results/$lab-marks.txt"
    local before after
    before=$(prompt_tokens)
    "$UV" run tools/bench.py --model "$served" --serial 8 --warmup 1 --prompt-tokens 4096 \
        --unique-prefix --max-tokens 1 --out "results/$lab-prefill.jsonl" 2>&1 | tail -4 || die "$cfg prefill"
    after=$(prompt_tokens)
    awk -v a="$after" -v b="$before" 'BEGIN{printf "%.1f\n", (a-b)/9}' > "results/$lab-prompt-tokens.txt"
    echo "  prefill prompt tokens per request: $(cat "results/$lab-prompt-tokens.txt")"
    echo "done $(date '+%Y/%m/%d %H:%M:%S')" >> "results/$lab-marks.txt"
}

case "${1:-}" in
check)
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,power.limit --format=csv
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | grep -q "L4, 8.9" || die "not an L4 (sm89)"
    /opt/llm/.venv-vllm/bin/python -c 'import torch; f,t=torch.cuda.mem_get_info(); print(f"  CUDA free {f/2**20:.0f} MiB of {t/2**20:.0f} MiB")'
    for m in models--Qwen--Qwen3-8B models--RedHatAI--Qwen3-8B-quantized.w4a16; do
        [ -d "hf-cache/hub/$m" ] || die "missing $m"
    done
    # The box's copies came from the image; replay indexes into them, so they must match the repo's.
    echo "2a929b01947be1fbfb87b94a5419898c7c0bc336aeb59dd3db6486e213596828  $A10G
01ba4591767bdbedf1bba1c2b817ff7230909ada87bd6468bde15bbbeed34b8e  $PROMPTS" \
        | sha256sum -c --quiet || die "replay inputs differ from the repo's"
    echo "CHECK_DONE"
    ;;
fp8)  run_cfg fp8  74880  4,5  --model Qwen/Qwen3-8B --quantization fp8 --max-model-len 16384 ;;
fp8m) export VLLM_TEST_FORCE_FP8_MARLIN=1     # read by vllm-launch.sh into the unit's env
      run_cfg fp8m 74880  4,5  --model Qwen/Qwen3-8B --quantization fp8 --max-model-len 16384 ;;
bf16) run_cfg bf16 33312  1,2  --model Qwen/Qwen3-8B --max-model-len 16384 ;;
int4) run_cfg int4 101920 10,11 --model RedHatAI/Qwen3-8B-quantized.w4a16 --max-model-len 16384 ;;
report)
    "$UV" run tools/l4report.py || die "report"
    echo "REPORT_DONE"
    ;;
all)
    bash "$0" check || die "check"
    for cfg in fp8 fp8m bf16 int4; do
        echo "[$(ts)] CONFIG $cfg"
        bash "$0" "$cfg" || echo "CONFIG_FAILED $cfg"
    done
    sudo systemctl stop vllm >/dev/null 2>&1
    bash "$0" report
    echo "L4_DONE"
    ;;
*)
    echo "usage: $0 check | fp8 | fp8m | bf16 | int4 | report | all"; exit 2 ;;
esac
exit 0

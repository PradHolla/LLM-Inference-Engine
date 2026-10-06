#!/usr/bin/env bash
# Phase 8: profile one decode step of our engine (PyTorch profiler + Nsight Systems), with vLLM
# as the reference. Runs ON the box with no vLLM server up.
#   sudo systemd-run --unit=p8 --collect /bin/bash /opt/llm/infra/p8-runs.sh all
set -uo pipefail
cd /opt/llm || exit 1
PY=/opt/llm/.venv/bin/python
VLLM=/opt/llm/.venv-vllm/bin/vllm
export HF_HOME=/opt/llm/hf-cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
ts() { date -u +%H:%M:%S; }
die() { echo "ABORT: $*"; exit 1; }

find_nsys() {
    command -v nsys 2>/dev/null && return 0
    ls -1d /opt/nvidia/nsight-systems*/bin/nsys /usr/local/cuda*/bin/nsys 2>/dev/null | head -1
}

case "${1:-}" in
install)
    N=$(find_nsys)
    if [ -z "$N" ]; then
        echo "[$(ts)] nsys not found; installing nsight-systems-cli from NVIDIA's devtools repo"
        # The devtools repo is signed with NVIDIA's older 7fa2af80 key, not the CUDA keyring's.
        KEY=/usr/share/keyrings/nvidia-devtools.gpg
        wget -q -O - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
            | gpg --dearmor --yes -o "$KEY" || die "devtools key fetch failed"
        echo "deb [signed-by=$KEY] https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/ /" \
            > /etc/apt/sources.list.d/nvidia-devtools.list
        apt-get update -q >/tmp/apt-update.log 2>&1 || { tail -5 /tmp/apt-update.log; die "apt update"; }
        apt-get install -y -q nsight-systems-cli >/tmp/apt-nsys.log 2>&1 \
            || { tail -10 /tmp/apt-nsys.log; die "nsight-systems-cli install"; }
        N=$(find_nsys)
    fi
    [ -n "$N" ] || die "nsys still not found"
    ln -sf "$N" /usr/local/bin/nsys
    nsys --version
    "$PY" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
    echo "INSTALL_DONE"
    ;;
time)
    rm -f results/p8-time.jsonl
    "$PY" -m engine.profile_step --mode time --batches 1,8 --out results/p8-time || die "time"
    "$PY" -m engine.profile_step --mode time --batches 8 --masked --out results/p8-time || die "time masked"
    echo "TIME_DONE"
    ;;
torch)
    rm -f results/p8-torch.jsonl results/p8-torch-*.json results/p8-torch-*-ops.txt
    "$PY" -m engine.profile_step --mode torch --batches 1,8 --out results/p8-torch || die "torch"
    "$PY" -m engine.profile_step --mode torch --batches 8 --masked --out results/p8-torch || die "torch masked"
    gzip -9 -f results/p8-torch-*.json
    echo "TORCH_DONE"
    ;;
nsys)
    for spec in "1:" "8:" "8:--masked"; do
        B=${spec%%:*}; extra=${spec#*:}; name=p8-nsys-engine-B$B${extra:+-masked}
        echo "[$(ts)] $name"
        rm -f "results/$name.nsys-rep" "results/$name.sqlite"
        nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,nvtx,osrt \
            --force-overwrite=true -o "results/$name" \
            "$PY" -m engine.profile_step --mode nsys --batches "$B" $extra --out results/p8-nsys \
            > "results/$name.log" 2>&1 || { tail -15 "results/$name.log"; die "$name"; }
        nsys stats --force-export=true --report nvtx_sum,nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_api_sum \
            "results/$name.nsys-rep" > "results/$name-stats.txt" 2>&1
        head -40 "results/$name-stats.txt" | sed 's/^/  /'
    done
    echo "NSYS_DONE"
    ;;
vllm)
    echo "[$(ts)] vLLM's own profiler options in this version (looked up, not assumed):"
    "$VLLM" bench latency --help=all 2>/dev/null | grep -iE "profil|input-len|output-len|batch-size|num-iters" | sed 's/^/  /'
    name=p8-nsys-vllm-B1
    rm -f "results/$name.nsys-rep"
    # Load + compile + graph capture take ~2-3 min; record 20 s of steady decode after that.
    VLLM_WORKER_MULTIPROC_METHOD=spawn nsys profile --trace-fork-before-exec=true \
        --cuda-graph-trace=node -t cuda,nvtx --delay "${DELAY:-240}" --duration 20 \
        --force-overwrite=true -o "results/$name" \
        "$VLLM" bench latency --model Qwen/Qwen3-8B --dtype bfloat16 --max-model-len 4096 \
        --input-len 412 --output-len 256 --batch-size 1 --num-iters-warmup 3 --num-iters 60 \
        > "results/$name.log" 2>&1 || { tail -20 "results/$name.log"; die "$name"; }
    grep -iE "avg latency|latency" "results/$name.log" | tail -3 | sed 's/^/  /'
    nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum \
        "results/$name.nsys-rep" > "results/$name-stats.txt" 2>&1
    head -40 "results/$name-stats.txt" | sed 's/^/  /'
    echo "VLLM_DONE"
    ;;
all)
    for step in install time torch nsys vllm; do
        echo "[$(ts)] STEP $step"
        bash "$0" "$step" || die "step $step failed"
    done
    echo "P8_DONE"
    ;;
*)
    echo "usage: $0 install | time | torch | nsys | vllm | all"; exit 2 ;;
esac
exit 0

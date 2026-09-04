#!/usr/bin/env bash
# Run one vLLM ablation: restart with a flag changed, wait for health, RECORD THE KV
# SIZE, then sweep. See infra/vllm-runbook.md for why each piece is there.
#   ./infra/vllm-ablate.sh <name> "<vllm-flags>" "<bench-flags>" "<rates>"
#   ./infra/vllm-ablate.sh A-nocache "--no-enable-prefix-caching" "--unique-prefix" "3,4,5,6,7"
set -uo pipefail

NAME="${1:?usage: vllm-ablate.sh <name> <vllm-flags> <bench-flags> <rates>}"
VLLM_FLAGS="${2:-}"
BENCH_FLAGS="${3:-}"
RATES="${4:-3,4,5,6,7}"

KEY="${KEY:-$HOME/.ssh/llm-inference.pem}"
IP="${IP:?set IP to the box public address}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DURATION="${DURATION:-60}"
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@$IP"

echo "################ $NAME ################"
echo "  vllm flags : ${VLLM_FLAGS:-<defaults>}"
echo "  bench flags: ${BENCH_FLAGS:-<none>}"

# PATH must include the venv bin or FlashInfer's JIT cannot find ninja and the engine
# dies during warmup with a FileNotFoundError that names the tool, not the cause.
# Stop, then wait for the driver to reclaim VRAM. The port closing and the memory freeing
# are different moments, and launching in the gap makes vLLM profile against memory the
# previous server still holds (incident 39, and it cost an hour on 2026-09-04).
$SSH "sudo systemctl stop vllm 2>/dev/null; sudo systemctl reset-failed vllm 2>/dev/null;
      for i in \$(seq 1 60); do
          u=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
          [ -z \"\$u\" ] && break
          [ \"\$u\" -lt 1024 ] && break
          sleep 2
      done;
      sudo systemd-run --unit=vllm --collect --working-directory=/opt/llm \
        --setenv=HF_HOME=/opt/llm/hf-cache --setenv=HF_HUB_OFFLINE=1 \
        --setenv=PYTHONUNBUFFERED=1 \
        --setenv=PATH=/opt/llm/.venv-vllm/bin:/usr/local/bin:/usr/bin:/bin \
        /opt/llm/.venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
          --model $MODEL --max-model-len 4096 --host 0.0.0.0 --port 8000 $VLLM_FLAGS" >/dev/null 2>&1

# Poll on the condition, never sleep a fixed guess. Startup is 2-3 min.
for i in $(seq 1 70); do
  curl -sf -m 3 "http://$IP:8000/health" >/dev/null 2>&1 && { READY=1; break; }
  $SSH 'systemctl is-active --quiet vllm' 2>/dev/null || { echo "  UNIT DIED -- check: journalctl -u vllm"; exit 1; }
  sleep 10
done
[ "${READY:-0}" = "1" ] || { echo "  TIMED OUT waiting for health"; exit 1; }

# vLLM's KV size varies between identical runs (see NOTES/code-notes.md) -- record it
# with every result so a capacity difference isn't wrongly attributed to the flag.
echo "  $($SSH 'journalctl -u vllm --no-pager -o cat | grep "GPU KV cache size" | tail -1' 2>/dev/null)"

# --model is REQUIRED: bench.py defaults to "test", which our own servers ignore and
# vLLM rejects with a 404 that surfaces as every request failing and metrics as nan.
uv run tools/bench.py --url "http://$IP:8000" --model "$MODEL" $BENCH_FLAGS \
  --sweep "$RATES" --duration "$DURATION" --prompt-tokens 512 --max-tokens 64 --no-think \
  --out "results/phase3-ablate-$NAME.jsonl" 2>&1 | tail -18

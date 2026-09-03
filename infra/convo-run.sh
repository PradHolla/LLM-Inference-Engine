#!/usr/bin/env bash
# The P6L-2 controlled run: one conversation grown turn by turn, twice.
# WARM lets the prefix cache work. COLD defeats it with front-loaded noise, and is the
# control -- without it a flat warm curve could just mean prefill is cheap. Runs ON the box.
#   ./convo-run.sh [turns] [max_tokens]
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv
TURNS="${1:-25}"; MAXTOK="${2:-300}"

[ -x "$UV" ] || { echo "ABORT: uv not found at $UV"; exit 1; }
curl -sf -m 5 http://localhost:8080/v1/models >/dev/null 2>&1 \
    || { echo "ABORT: gateway not serving /v1/models on :8080"; exit 1; }

echo "### engine config for this run"
sudo journalctl -u vllm --no-pager -o cat 2>/dev/null \
  | grep -oE 'GPU KV cache size: [0-9,]+ tokens|speculative_config=[A-Za-z]+' | tail -2 | sed 's/^/  /'

for arm in warm cold; do
    flag=""; [ "$arm" = cold ] && flag="--cold"
    out="results/convo-$arm.jsonl"
    rm -f "$out"
    echo
    echo "########## $arm arm: $TURNS turns, max_tokens=$MAXTOK"
    $UV run tools/convo.py --url http://localhost:8080 --turns "$TURNS" \
        --max-tokens "$MAXTOK" $flag --out "$out" || { echo "ARM $arm FAILED"; exit 1; }
done
echo
echo CONVO_RUN_DONE

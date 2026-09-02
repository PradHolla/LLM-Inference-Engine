#!/usr/bin/env bash
# Launch the Phase 6a lab bench on the box as a systemd unit, bound to localhost only.
# Reach it with: ssh -L 8080:localhost:8080 <box>   then open http://localhost:8080/ui/
#   ./labbench-launch.sh
set -uo pipefail
REPO=/opt/llm
PORT="${PORT:-8080}"

sudo systemctl stop labbench 2>/dev/null
sudo systemctl reset-failed labbench 2>/dev/null

# --host 127.0.0.1 is the access control: the tunnel is the only way in, so there is
# no auth to build and no GPU endpoint reachable from the internet.
sudo systemd-run --unit=labbench --collect --working-directory=$REPO \
  --setenv=PYTHONPATH=$REPO \
  --setenv=LABBENCH_UPSTREAM=http://localhost:8000 \
  --setenv=LABBENCH_TRACE=$REPO/results/labbench-traces.jsonl \
  --setenv=LABBENCH_SCRATCH=$REPO/results/labbench \
  --setenv=LABBENCH_UV=/home/ubuntu/.local/bin/uv \
  --setenv=PATH=$REPO/.venv/bin:/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin \
  $REPO/.venv/bin/python -m uvicorn labbench.server:app \
    --host 127.0.0.1 --port "$PORT" >/dev/null 2>&1

ok=0
for i in $(seq 1 30); do
    if curl -sf -m 3 "http://localhost:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
    st=$(systemctl is-active labbench)
    if [ "$st" = "failed" ] || [ "$st" = "inactive" ]; then
        echo "LABBENCH_FAILED (unit $st after ${i}s)"
        sudo journalctl -u labbench --no-pager -o cat | tail -30
        exit 1
    fi
    sleep 1
done
[ "$ok" = 1 ] || { echo "LABBENCH_TIMEOUT"; sudo journalctl -u labbench --no-pager -o cat | tail -30; exit 1; }
echo "labbench ready on 127.0.0.1:$PORT"
echo "  tunnel: ssh -L $PORT:localhost:$PORT <box>   then http://localhost:$PORT/ui/"

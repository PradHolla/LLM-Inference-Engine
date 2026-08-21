#!/usr/bin/env bash
# cloud-init bootstrap. Runs once as root on first boot.
# Watch progress: sudo tail -f /var/log/cloud-init-output.log
set -uxo pipefail

APP_DIR=/opt/llm
IDLE_SHUTDOWN_MINUTES=30
IDLE_BUSY_MINUTES=180
IDLE_GPU_PCT=5
VLLM_PORT=8000

mkdir -p "$APP_DIR" /var/log/llm
chown -R ubuntu:ubuntu "$APP_DIR" /var/log/llm

# Model weights live on the root volume, which survives a stop. Re-downloading
# 16 GB on every start is 5 wasted minutes and pointless HF egress.
mkdir -p "$APP_DIR/hf-cache"
chown -R ubuntu:ubuntu "$APP_DIR/hf-cache"
cat > /etc/profile.d/llm.sh <<EOF
export HF_HOME=$APP_DIR/hf-cache
export APP_DIR=$APP_DIR
export VLLM_PORT=$VLLM_PORT
EOF

# --- idle shutdown ----------------------------------------------------------
# Cron, not a systemd timer: cron is already running on every Ubuntu AMI, so
# there is nothing to enable and nothing to debug at boot.
cat > /usr/local/bin/idle-shutdown.sh <<'IDLESCRIPT'
__IDLE_SHUTDOWN_BODY__
IDLESCRIPT
chmod 755 /usr/local/bin/idle-shutdown.sh

cat > /etc/cron.d/llm-idle <<EOF
APP_DIR=$APP_DIR
IDLE_SHUTDOWN_MINUTES=$IDLE_SHUTDOWN_MINUTES
IDLE_BUSY_MINUTES=$IDLE_BUSY_MINUTES
IDLE_GPU_PCT=$IDLE_GPU_PCT
VLLM_PORT=$VLLM_PORT
* * * * * root /usr/local/bin/idle-shutdown.sh
EOF
chmod 644 /etc/cron.d/llm-idle

# --- python env -------------------------------------------------------------
sudo -u ubuntu bash <<'UBUNTU'
set -eux
export HF_HOME=/opt/llm/hf-cache
cd /opt/llm
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv
. .venv/bin/activate
uv pip install --torch-backend=auto vllm
uv pip install fastapi uvicorn httpx numpy pandas
uv pip install transformers accelerate       # Phase 1 baseline runs on these, not vLLM
UBUNTU

echo "BOOTSTRAP COMPLETE" | tee /var/log/llm/bootstrap-done

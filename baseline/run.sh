#!/usr/bin/env bash
# Launches the Phase 1 baseline server via the project venv.
#
# HF_HOME must be set INLINE here, not sourced from /etc/profile.d/llm.sh --
# a detached process (nohup, this script run from a disconnected SSH session,
# a systemd unit) never loads a login shell's profile. Relying on the profile
# file would silently revert HF_HOME to ~/.cache/huggingface, re-downloading
# ~16 GB and eventually filling the root volume.
set -euo pipefail

export HF_HOME=/opt/llm/hf-cache
cd /opt/llm

# exec replaces this shell with uvicorn so signals (Ctrl-C, systemd stop,
# idle-shutdown's `shutdown -h now`) reach the server directly instead of a
# wrapper process that has to be told twice.
exec .venv/bin/python -m uvicorn baseline.server:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}"

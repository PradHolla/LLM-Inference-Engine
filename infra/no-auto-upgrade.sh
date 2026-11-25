#!/usr/bin/env bash
# Stop Ubuntu's daily unattended upgrade from restarting vLLM and every other service mid-run (2026-11-25: it
# killed a practice eval at 15:43). Run once per session on the box, after any image restore:
#   sudo bash /opt/llm/infra/no-auto-upgrade.sh
set -uo pipefail
for t in apt-daily.timer apt-daily-upgrade.timer; do
    systemctl disable --now "$t" >/dev/null 2>&1 && echo "disabled $t" || echo "WARN: could not disable $t"
done
# needrestart: list services that would need a restart, never restart them.
conf=/etc/needrestart/conf.d/90-no-auto-restart.conf
mkdir -p /etc/needrestart/conf.d
echo "\$nrconf{restart} = 'l';" > "$conf" && echo "needrestart set to list-only ($conf)"
busy=$(pgrep -f "/usr/bin/unattended-upgrade$" | head -1)  # not the always-running shutdown helper
echo "unattended-upgrade running now: ${busy:-no}"
exit 0

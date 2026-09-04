#!/usr/bin/env bash
# One command each way, correct flags baked in. Written because hand-typed rsync and ssh
# invocations failed three times this week for reasons unrelated to the work.
#   ./sync.sh push          code up (labbench, tools, infra)
#   ./sync.sh pull          results down
#   ./sync.sh run '<cmd>'   run a command on the box, from /opt/llm
set -uo pipefail
REGION="${AWS_REGION:-us-east-1}"; PROJECT="llm-inference"
PEM="$HOME/.ssh/${KEY_NAME:-llm-inference}.pem"
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO" || exit 1
IP=$(aws ec2 describe-instances --region "$REGION" \
     --filters "Name=tag:Project,Values=$PROJECT" "Name=instance-state-name,Values=running" \
     --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null)
[ -n "$IP" ] && [ "$IP" != "None" ] || { echo "no running instance"; exit 1; }
RS=(rsync -az -e "ssh -i $PEM -o StrictHostKeyChecking=no" --exclude '__pycache__')

case "${1:-}" in
  push)
    for d in labbench tools infra; do
        "${RS[@]}" "$d/" "ubuntu@$IP:/opt/llm/$d/" || exit 1
    done
    # Record which commit the box is running, so a result can be traced to a source state.
    ssh -i "$PEM" -o StrictHostKeyChecking=no "ubuntu@$IP" \
        "mkdir -p /opt/llm/results && echo '$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)' > /opt/llm/results/DEPLOYED_SHA"
    echo "pushed to $IP at $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
    ;;
  pull)
    "${RS[@]}" "ubuntu@$IP:/opt/llm/results/" results/ || exit 1
    echo "pulled results from $IP (box was running $(cat results/DEPLOYED_SHA 2>/dev/null || echo unknown))"
    ;;
  run)
    [ -n "${2:-}" ] || { echo "usage: sync.sh run '<command>'"; exit 2; }
    ssh -i "$PEM" -o StrictHostKeyChecking=no "ubuntu@$IP" "cd /opt/llm && $2"
    ;;
  *) echo "usage: sync.sh push|pull|run '<cmd>'"; exit 2 ;;
esac

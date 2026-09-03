#!/usr/bin/env bash
# End a session cleanly: fetch results, drop the hold, stop the box. Never terminates.
#   ./infra/session-down.sh
set -uo pipefail
REGION="${AWS_REGION:-us-east-1}"; PROJECT="llm-inference"
KEY="${KEY_NAME:-llm-inference}"; PEM="$HOME/.ssh/$KEY.pem"
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO" || exit 1

IID=$(aws ec2 describe-instances --region "$REGION" \
      --filters "Name=tag:Project,Values=$PROJECT" "Name=instance-state-name,Values=running" \
      --query 'Reservations[].Instances[0].InstanceId' --output text 2>/dev/null)
if [ -z "$IID" ] || [ "$IID" = "None" ]; then echo "no running instance; nothing to do"; exit 0; fi
IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
     --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "=== fetching results from $IP"
rsync -az -e "ssh -i $PEM -o StrictHostKeyChecking=no" \
      "ubuntu@$IP:/opt/llm/results/" results/ 2>/dev/null && echo "  fetched" || echo "  WARNING: fetch failed"

echo "=== removing the hold file (the expiry is a backstop, not the plan)"
ssh -i "$PEM" -o StrictHostKeyChecking=no "ubuntu@$IP" \
    'rm -f /opt/llm/.no-autoshutdown; ls /opt/llm/.no-autoshutdown >/dev/null 2>&1 \
     && echo "  STILL PRESENT" || echo "  removed, confirmed absent"' 2>/dev/null

echo "=== stopping (never terminating; the root volume holds the model cache)"
./infra/down.sh

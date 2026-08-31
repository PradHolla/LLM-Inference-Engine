#!/usr/bin/env bash
# Stop the project box. Stop, never terminate: the root volume holds the HF
# cache and everything else. Terminate only when the project is over.
set -euo pipefail
REGION="${AWS_REGION:-us-east-1}"; PROJECT="llm-inference"
IID=$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Project,Values=$PROJECT" \
              "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[0].InstanceId' --output text)
[ -n "$IID" ] && [ "$IID" != "None" ] || { echo "nothing running"; exit 0; }
aws ec2 stop-instances --region "$REGION" --instance-ids "$IID" >/dev/null
echo "  $IID  stopping"

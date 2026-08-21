#!/usr/bin/env bash
# Start the stopped project box and print how to reach it.
set -euo pipefail
REGION="${AWS_REGION:-us-east-1}"; PROJECT="llm-inference"
KEY_NAME="${KEY_NAME:-$PROJECT}"; VLLM_PORT="${VLLM_PORT:-8000}"

IID=$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Project,Values=$PROJECT" \
              "Name=instance-state-name,Values=stopped,running" \
    --query 'Reservations[].Instances[0].InstanceId' --output text)
[ -n "$IID" ] && [ "$IID" != "None" ] || { echo "no instance found - run infra/launch.sh"; exit 1; }

aws ec2 start-instances --region "$REGION" --instance-ids "$IID" >/dev/null
aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
     --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

# The public IP changes on every stop/start (no Elastic IP attached, because an
# unattached EIP bills by the hour and a project box is stopped most of the time).
# Re-authorize the SG for wherever you are now.
MYIP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
SG=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
     --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text)
for port in 22 "$VLLM_PORT"; do
    aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" \
        --protocol tcp --port "$port" --cidr "$MYIP/32" >/dev/null 2>&1 || true
done

echo "  $IID  up"
echo "  ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@$IP"
echo "  http://$IP:$VLLM_PORT"

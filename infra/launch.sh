#!/usr/bin/env bash
# Provision the inference box for this project. Idempotent-ish: refuses to run
# if an instance already exists for this project (use up.sh instead).
#   ./infra/launch.sh              # on-demand g5.xlarge  (default; use this)
#   USE_SPOT=1 ./infra/launch.sh   # spot -- capacity escape hatch, see NOTES/PROJECT.md
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
PROJECT="llm-inference"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.2xlarge}"
# The cheapest AZ is per-instance-type, not per-region, and can invert between
# types (see NOTES/code-notes.md). Resolved from live spot data below, not hardcoded.
AZ="${AZ:-}"
ROOT_GB="${ROOT_GB:-200}"
USE_SPOT="${USE_SPOT:-0}"
KEY_NAME="${KEY_NAME:-$PROJECT}"
SG_NAME="${SG_NAME:-$PROJECT-sg}"
VLLM_PORT="${VLLM_PORT:-8000}"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- guard: don't double-provision -----------------------------------------
EXISTING=$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Project,Values=$PROJECT" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text)
if [ -n "$EXISTING" ]; then
    die "instance already exists for this project: $EXISTING (use infra/up.sh)"
fi

# --- quota sanity check ------------------------------------------------------
# Shared account-wide G quota with the unrelated sql-llm project (NOTES/code-notes.md).
NEED=$(aws ec2 describe-instance-types --region "$REGION" --instance-types "$INSTANCE_TYPE" \
        --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' --output text)
QCODE="L-DB2E81BA"; [ "$USE_SPOT" = "1" ] && QCODE="L-3819A6DF"
LIMIT=$(aws service-quotas get-service-quota --region "$REGION" --service-code ec2 \
        --quota-code "$QCODE" --query 'Quota.Value' --output text | cut -d. -f1)
INUSE=0
for t in $(aws ec2 describe-instances --region "$REGION" \
            --filters "Name=instance-state-name,Values=pending,running" \
            --query 'Reservations[].Instances[?starts_with(InstanceType,`g`)].InstanceType' \
            --output text); do
    v=$(aws ec2 describe-instance-types --region "$REGION" --instance-types "$t" \
        --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' --output text)
    INUSE=$((INUSE + v))
done
say "G quota ($QCODE): ${INUSE}/${LIMIT} vCPU in use, this needs ${NEED}"
if [ $((INUSE + NEED)) -gt "$LIMIT" ]; then
    die "would exceed quota. Stop the other project's box, pick a smaller type, or raise $QCODE."
fi

# --- AMI: resolve latest via SSM so this never goes stale --------------------
AMI=$(aws ssm get-parameter --region "$REGION" \
    --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id \
    --query 'Parameter.Value' --output text)
say "AMI $AMI (Deep Learning Base OSS Nvidia Driver, Ubuntu 24.04)"

# --- key pair ----------------------------------------------------------------
if ! aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" >/dev/null 2>&1; then
    say "creating key pair $KEY_NAME -> ~/.ssh/$KEY_NAME.pem"
    aws ec2 create-key-pair --region "$REGION" --key-name "$KEY_NAME" \
        --query 'KeyMaterial' --output text > "$HOME/.ssh/$KEY_NAME.pem"
    chmod 400 "$HOME/.ssh/$KEY_NAME.pem"
fi

# --- security group, locked to this machine's public IP ----------------------
VPC=$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true \
      --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 describe-security-groups --region "$REGION" \
     --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC" \
     --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")
if [ "$SG" = "None" ] || [ -z "$SG" ]; then
    say "creating security group $SG_NAME"
    SG=$(aws ec2 create-security-group --region "$REGION" --group-name "$SG_NAME" \
         --description "$PROJECT inference box" --vpc-id "$VPC" \
         --query 'GroupId' --output text)
fi
MYIP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
say "authorizing $MYIP/32 on 22 and $VLLM_PORT"
for port in 22 "$VLLM_PORT"; do
    aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" \
        --protocol tcp --port "$port" --cidr "$MYIP/32" >/dev/null 2>&1 || true
done

if [ -z "$AZ" ]; then
    if [ "$USE_SPOT" = "1" ]; then
        AZ=$(aws ec2 describe-spot-price-history --region "$REGION" \
             --instance-types "$INSTANCE_TYPE" --product-descriptions "Linux/UNIX" \
             --start-time "$(date -u -v-6H +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -u -d '6 hours ago' +%Y-%m-%dT%H:%M:%S)" \
             --query 'SpotPriceHistory[].[AvailabilityZone,SpotPrice]' --output text \
             | sort -k1,1 -k2,2n | awk '!s[$1]++' | sort -k2,2n | head -1 | cut -f1)
        say "cheapest spot AZ for $INSTANCE_TYPE right now: $AZ"
    fi
    AZ="${AZ:-us-east-1c}"
fi

SUBNET=$(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=vpc-id,Values=$VPC" "Name=availability-zone,Values=$AZ" \
    --query 'Subnets[0].SubnetId' --output text)
[ "$SUBNET" = "None" ] && die "no default subnet in $AZ"

# --- the stop-vs-terminate split --------------------------------------------
# on-demand STOPS (root volume survives); spot TERMINATES. See NOTES/code-notes.md.
SHUTDOWN_BEHAVIOR="stop"; DELETE_ROOT="false"; MARKET_ARGS=()
if [ "$USE_SPOT" = "1" ]; then
    SHUTDOWN_BEHAVIOR="terminate"; DELETE_ROOT="true"
    MARKET_ARGS=(--instance-market-options \
        'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
    say "SPOT mode: box is disposable. Anything not synced to S3 will be lost."
fi

# Splice idle-shutdown.sh into the bootstrap at launch time so its logic has
# exactly one source of truth on disk.
HERE="$(cd "$(dirname "$0")" && pwd)"
UD=$(mktemp -t llm-userdata)
trap 'rm -f "$UD"' EXIT
python3 - "$HERE" "$UD" <<'PY'
import sys, pathlib
here, out = pathlib.Path(sys.argv[1]), sys.argv[2]
ud = (here/"user-data.sh").read_text()
body = (here/"idle-shutdown.sh").read_text().rstrip()
assert "__IDLE_SHUTDOWN_BODY__" in ud, "placeholder missing from user-data.sh"
pathlib.Path(out).write_text(ud.replace("__IDLE_SHUTDOWN_BODY__", body))
PY
# EC2 caps user-data at 16 KB.
SZ=$(wc -c < "$UD")
[ "$SZ" -lt 16384 ] || die "user-data is ${SZ} bytes, over the 16 KB EC2 limit"

say "launching $INSTANCE_TYPE in $AZ (shutdown -> $SHUTDOWN_BEHAVIOR)"
IID=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" --security-group-ids "$SG" --subnet-id "$SUBNET" \
    --instance-initiated-shutdown-behavior "$SHUTDOWN_BEHAVIOR" \
    --metadata-options 'HttpTokens=required,HttpEndpoint=enabled' \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$ROOT_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":$DELETE_ROOT}}]" \
    --user-data "file://$UD" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=$PROJECT},{Key=Name,Value=$PROJECT}]" \
    ${MARKET_ARGS[@]+"${MARKET_ARGS[@]}"} \
    --query 'Instances[0].InstanceId' --output text)

say "waiting for $IID"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
     --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

cat <<EOF

  instance   $IID  ($INSTANCE_TYPE, $AZ)
  ssh        ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@$IP
  vllm       http://$IP:$VLLM_PORT
  shutdown   $SHUTDOWN_BEHAVIOR after 30m idle / 180m with a job or session
  hold       touch /opt/llm/.no-autoshutdown

  Bootstrap runs for a few minutes. Watch: sudo tail -f /var/log/cloud-init-output.log
EOF

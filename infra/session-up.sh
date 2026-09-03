#!/usr/bin/env bash
# Bring the entire lab bench up from a stopped box, in one command. Idempotent: safe to
# re-run at any point, it skips whatever is already done.
#   ./infra/session-up.sh                      # fp8, 16384 ctx
#   ./infra/session-up.sh --quant bf16 --ctx 8192
set -uo pipefail
REGION="${AWS_REGION:-us-east-1}"; PROJECT="llm-inference"
KEY="${KEY_NAME:-llm-inference}"; PEM="$HOME/.ssh/$KEY.pem"
QUANT=fp8; CTX=16384; SPEC=""
while [ $# -gt 0 ]; do
    case "$1" in
        --quant) QUANT="$2"; shift 2 ;;
        --ctx)   CTX="$2";   shift 2 ;;
        --spec)  SPEC="--speculative-config {\"model\":\"RedHatAI/Qwen3-8B-speculator.eagle3\",\"method\":\"eagle3\",\"num_speculative_tokens\":2}"; shift ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1
say() { printf '\n=== %s\n' "$*"; }
die() { printf '\nFAILED: %s\n' "$*" >&2; exit 1; }

say "instance state"
IID=$(aws ec2 describe-instances --region "$REGION" \
      --filters "Name=tag:Project,Values=$PROJECT" \
                "Name=instance-state-name,Values=stopped,stopping,pending,running" \
      --query 'Reservations[].Instances[0].InstanceId' --output text 2>/dev/null)
[ -n "$IID" ] && [ "$IID" != "None" ] || die "no instance found; run infra/launch.sh"
STATE=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
        --query 'Reservations[0].Instances[0].State.Name' --output text)
echo "  $IID is $STATE"

if [ "$STATE" = "stopping" ]; then
    echo "  waiting for it to finish stopping before starting again"
    aws ec2 wait instance-stopped --region "$REGION" --instance-ids "$IID" || die "wait stopped"
    STATE=stopped
fi

if [ "$STATE" != "running" ]; then
    say "starting (g5 capacity in this AZ is intermittent; retrying)"
    up=0
    for i in $(seq 1 25); do
        if ./infra/up.sh >/tmp/session-up.log 2>&1; then up=1; break; fi
        echo "  attempt $i: $(tail -1 /tmp/session-up.log | cut -c1-90)"
        sleep 60
    done
    [ "$up" = 1 ] || die "no capacity after 25 attempts"
else
    ./infra/up.sh >/tmp/session-up.log 2>&1 || true   # re-authorizes the SG for this IP
fi
IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
     --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
[ -n "$IP" ] && [ "$IP" != "None" ] || die "no public IP"
echo "  ip $IP"

sshbox() { ssh -i "$PEM" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "ubuntu@$IP" "$@"; }

say "waiting for ssh"
ready=0
for i in $(seq 1 40); do
    if sshbox 'echo ok' 2>/dev/null | grep -q ok; then ready=1; break; fi
    sleep 5
done
[ "$ready" = 1 ] || die "ssh never came up at $IP (check checkip against the security group)"
echo "  ssh ok"

say "deploying code"
for d in labbench tools infra; do
    rsync -az -e "ssh -i $PEM -o StrictHostKeyChecking=no" --exclude '__pycache__' \
        "$d/" "ubuntu@$IP:/opt/llm/$d/" || die "rsync $d"
done
sshbox 'chmod +x /opt/llm/infra/*.sh; touch /opt/llm/.no-autoshutdown' || die "chmod/hold"
echo "  code deployed, hold placed"

say "engine: $QUANT at $CTX ctx"
if sshbox "curl -sf -m 3 http://localhost:8000/health >/dev/null 2>&1"; then
    echo "  already healthy, leaving it alone (use --quant to force a relaunch via the UI)"
else
    MODEL=Qwen/Qwen3-8B; QFLAG="--quantization fp8"
    [ "$QUANT" = bf16 ] && QFLAG=""
    [ "$QUANT" = int4 ] && { MODEL=RedHatAI/Qwen3-8B-quantized.w4a16; QFLAG=""; }
    sshbox "cd /opt/llm && ./infra/vllm-launch.sh session-$QUANT-$CTX \
              --model $MODEL --max-model-len $CTX $QFLAG $SPEC" || die "vllm launch"
fi

say "lab bench"
sshbox 'cd /opt/llm && ./infra/labbench-launch.sh' || die "labbench launch"

say "verifying every endpoint the UI needs"
bad=0
for p in /ui/ /ui/app.css /ui/app.js /ui/vendor/react.production.min.js \
         /v1/models /labbench/state /labbench/traces; do
    code=$(sshbox "curl -s -o /dev/null -w '%{http_code}' http://localhost:8080$p")
    printf '  %-42s %s\n' "$p" "$code"
    [ "$code" = 200 ] || bad=1
done
[ "$bad" = 0 ] || die "some endpoints are not serving"

cat <<TXT

=== READY
  tunnel:  ssh -i $PEM -L 8080:localhost:8080 ubuntu@$IP
  open:    http://localhost:8080/ui/
  down:    ./infra/session-down.sh

  The box is billing at \$1.21/hr and .no-autoshutdown is held.
  session-down.sh removes the hold and stops the instance.
TXT

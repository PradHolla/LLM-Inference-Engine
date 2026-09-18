#!/usr/bin/env bash
# Image the project box so it can be TERMINATED rather than left stopped at 200 GB of gp3.
# An AMI also moves: a stopped instance is pinned to its AZ, an image is not.
#   ./bake-ami.sh bake            image the stopped box and tag it
#   ./bake-ami.sh list            images owned here, newest first
#   ./bake-ami.sh wait <ami-id>   block until it is available
#   ./bake-ami.sh rm <ami-id>     deregister AND delete its snapshots
set -euo pipefail
REGION="${AWS_REGION:-us-east-1}"; PROJECT="llm-inference"
HIST="$(cd "$(dirname "$0")" && pwd)/ami-history.txt"

instance() {
    aws ec2 describe-instances --region "$REGION" \
        --filters "Name=tag:Project,Values=$PROJECT" \
                  "Name=instance-state-name,Values=stopped,running" \
        --query 'Reservations[].Instances[0].[InstanceId,State.Name]' --output text
}

case "${1:-}" in
  bake)
    read -r IID STATE <<<"$(instance)"
    [ -n "${IID:-}" ] && [ "$IID" != "None" ] || { echo "no instance found"; exit 1; }
    # Only image a stopped box. Imaging a running one either reboots it -- killing a
    # server mid-run -- or takes a crash-consistent snapshot with the HF cache half
    # written. Stopped is the only state that is both safe and free of GPU time.
    [ "$STATE" = "stopped" ] || { echo "$IID is $STATE; stop it first (infra/down.sh)"; exit 1; }

    SHA="$(git -C "$(dirname "$HIST")/.." rev-parse --short HEAD 2>/dev/null || echo nogit)"
    STAMP="$(date -u +%Y%m%d-%H%M)"
    NAME="$PROJECT-$STAMP-$SHA"
    TAGS="Tags=[{Key=Project,Value=$PROJECT},{Key=Commit,Value=$SHA}]"
    AMI=$(aws ec2 create-image --region "$REGION" --instance-id "$IID" --name "$NAME" \
        --description "$PROJECT box imaged from $IID at $SHA" \
        --tag-specifications "ResourceType=image,$TAGS" "ResourceType=snapshot,$TAGS" \
        --query ImageId --output text)
    echo "$STAMP  $AMI  $NAME  from $IID" >> "$HIST"
    echo "  $AMI  pending"
    echo "  $0 wait $AMI"
    ;;
  list)
    aws ec2 describe-images --region "$REGION" --owners self \
        --query 'reverse(sort_by(Images,&CreationDate))[].{ami:ImageId,name:Name,state:State,created:CreationDate}' \
        --output table
    ;;
  wait)
    [ -n "${2:-}" ] || { echo "usage: $0 wait <ami-id>"; exit 2; }
    aws ec2 wait image-available --region "$REGION" --image-ids "$2"
    echo "  $2  available"
    ;;
  rm)
    [ -n "${2:-}" ] || { echo "usage: $0 rm <ami-id>"; exit 2; }
    # Deregistering alone leaves the snapshots billing forever -- they are NOT removed
    # with the image. This flag is the whole reason this mode exists as a script.
    aws ec2 deregister-image --region "$REGION" --image-id "$2" --delete-associated-snapshots
    echo "$(date -u +%Y%m%d-%H%M)  $2  removed with snapshots" >> "$HIST"
    echo "  $2  deregistered, snapshots deleted"
    ;;
  *) echo "usage: $0 bake|list|wait <ami-id>|rm <ami-id>"; exit 2 ;;
esac
exit 0

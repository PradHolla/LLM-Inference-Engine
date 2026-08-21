#!/usr/bin/env bash
# Spot interruption watcher. Polls IMDS every 5s; on a reclaim notice you have
# ~2 minutes to save anything you care about.
#
# Only needed when USE_SPOT=1. On-demand boxes stop and keep their volume.
#
# WHAT IS WORTH RESCUING IS DIFFERENT HERE than on the training project. There
# are no checkpoints. What dies with a spot box is BENCHMARK RESULTS -- a 40
# minute latency-vs-throughput sweep that has to be re-run from zero. So:
#
#   Layer 1 (this script): graceful-interruption sweep of $WATCH_DIR.
#   Layer 2 (the one that matters): bench.py appends each completed request to
#           JSONL and syncs periodically. The watcher only covers the graceful
#           case; an ungraceful kill loses everything since the last flush.
#           Never buffer a whole sweep in memory and write at the end.
set -uo pipefail

BUCKET="s3://REPLACE-ME-llm-inference-results"   # hardcoded on purpose, see below
WATCH_DIR="${WATCH_DIR:-/opt/llm/results}"
IMDS="http://169.254.169.254/latest"

sync_out() {
    # NEVER --delete. A fresh box with an empty results/ would mirror that
    # emptiness upward and destroy the only copy of everything.
    aws s3 sync "$WATCH_DIR" "$BUCKET/results/" --only-show-errors || true
    #                                                                ^^^^^^^
    # A failed sync must never kill the run. Log it and carry on.
}

while true; do
    # IMDSv2: without a token you poll blind forever, getting 401s that look
    # exactly like the 404 that means "nothing wrong".
    TOKEN=$(curl -sf -X PUT "$IMDS/api/token" \
            -H "X-aws-ec2-metadata-token-ttl-seconds: 300" --max-time 2 || true)
    if [ -n "$TOKEN" ]; then
        CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
               -H "X-aws-ec2-metadata-token: $TOKEN" \
               "$IMDS/meta-data/spot/instance-action" || echo 000)
        # 200 = reclaimed in ~2 min. 404 = normal. Anything else = don't panic.
        if [ "$CODE" = "200" ]; then
            logger -t spot-rescue "interruption notice - syncing $WATCH_DIR"
            sync_out
            logger -t spot-rescue "sync complete"
            exit 0
        fi
    fi
    sleep 5
done

# ---------------------------------------------------------------------------
# WHY THE BUCKET IS HARDCODED
#   Detached jobs (nohup, systemd, cron) do not load a login shell, so anything
#   set in ~/.bashrc or /etc/profile.d is simply absent. An env-var bucket
#   silently becomes "s3:///results/" and the rescue writes nowhere.
#
# WIRING (user-data.sh, only when USE_SPOT=1):
#   install -m 755 infra/spot-rescue.sh /usr/local/bin/spot-rescue.sh
#   cat > /etc/systemd/system/spot-rescue.service <<EOF
#   [Unit]
#   Description=Spot interruption watcher
#   [Service]
#   ExecStart=/usr/local/bin/spot-rescue.sh
#   Restart=always
#   [Install]
#   WantedBy=multi-user.target
#   EOF
#   systemctl enable --now spot-rescue
#
# The instance needs an IAM instance profile with s3:PutObject on the bucket.
#
# ALSO WORTH CACHING TO S3 ON SPOT: the model weights. A new spot box re-pulls
# 16 GB from HuggingFace every time. Same-region S3 is several times faster and
# has no egress cost.
# ---------------------------------------------------------------------------

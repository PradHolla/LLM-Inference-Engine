#!/usr/bin/env bash
# Spot interruption watcher: polls IMDS every 5s and syncs $WATCH_DIR to S3 on
# a reclaim notice. Only needed when USE_SPOT=1 -- on-demand boxes stop and
# keep their volume. See NOTES/code-notes.md for what this covers and what it misses.
set -uo pipefail

BUCKET="s3://REPLACE-ME-llm-inference-results"   # hardcoded on purpose -- see NOTES/code-notes.md
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

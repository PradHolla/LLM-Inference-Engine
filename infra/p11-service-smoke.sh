#!/usr/bin/env bash
# Service desk agent smoke test on the app's 9 curated showcase tasks: xLAM (as served now), then Qwen
# off and on. Leaves Qwen served for the app. Run detached:
#   sudo systemd-run --unit=p11-svc-smoke --collect --working-directory=/opt/llm /bin/bash /opt/llm/infra/p11-service-smoke.sh
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv
IDX=1483,1489,1497,1412,1459,1384,773,765,699
TYPES=case_routing,policy_violation_identification,knowledge_qa

run() {  # arm, served model
    $UV run tools/crmeval.py run --url http://localhost:8000 --arm "$1" --model "$2" --org b2b --split dev \
        --types "$TYPES" --idx "$IDX" --concurrency 3 --run-id "p11-svc-$1" --protocol service_v1 \
        || echo "P11_SVC_FAIL $1"
}

served=$(curl -s -m 5 http://localhost:8000/v1/models | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
echo "served at start: $served"
if [ "$served" = "Salesforce/Llama-xLAM-2-8b-fc-r" ]; then
    run xlam-t0 "$served"
    /bin/bash /opt/llm/infra/p11-app.sh launch qwen || { echo "P11_SVC_FAIL qwen launch"; exit 1; }
else
    echo "xLAM not served; skipping its arm"
fi
run q-off Qwen/Qwen3-8B
run q-think Qwen/Qwen3-8B
echo "P11_SVC_DONE"
exit 0

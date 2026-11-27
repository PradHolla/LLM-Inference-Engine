#!/usr/bin/env bash
# P11-INJ on the box: the detector's false-alarm scan, the attack bench (3 arms), and the held-out check on q-guard.
#   sudo systemd-run --unit=p11-inject-scan --collect --working-directory=/opt/llm /bin/bash /opt/llm/infra/p11-inject.sh scan
#   ... p11-inject.sh bench      # practice routing + policy, clean and attacked, q-judge / q-guard-code / q-guard
#   ... p11-inject.sh heldout    # held-out routing + policy (1 pass) and the 16 dev privacy requests, q-guard
set -uo pipefail
cd /opt/llm || exit 1
export HOME=/home/ubuntu CRM_RERANK_URL=http://127.0.0.1:8010 FASTEMBED_CACHE_PATH=/opt/llm/hf-cache/fastembed
UV=/home/ubuntu/.local/bin/uv
URL=http://localhost:8000
what="${1:?usage: p11-inject.sh scan | bench | heldout}"
fail=0
case "$what" in
    scan)
        $UV run tools/injectbench.py scan --url "$URL" --out results/p11-inject-scan.jsonl --concurrency 8 || fail=1
        $UV run tools/injectbench.py summary results/p11-inject-scan.jsonl || fail=1
        ;;
    bench)
        $UV run tools/injectbench.py run --url "$URL" --arms q-judge,q-guard-code,q-guard --templates all \
            --hide-extra-test 20 --out results/p11-inject.jsonl --concurrency 8 || fail=1
        $UV run tools/injectbench.py summary results/p11-inject.jsonl || fail=1
        ;;
    heldout)
        $UV run tools/crmeval.py run --url "$URL" --arm q-guard --model Qwen/Qwen3-8B --org b2b --split test \
            --types case_routing,policy_violation_identification --concurrency 8 --run-seed 0 \
            --run-id p11-heldout-q-guard --protocol service_v1 || fail=1
        $UV run tools/crmeval.py run --url "$URL" --arm q-guard --model Qwen/Qwen3-8B --org b2b --split dev \
            --types private_customer_information --concurrency 4 --run-seed 0 \
            --run-id p11-privacy-dev-q-guard --protocol service_v1 || fail=1
        ;;
    *) echo "usage: p11-inject.sh scan | bench | heldout"; exit 2 ;;
esac
if [ "$fail" = 0 ]; then echo "P11_INJ_DONE $what"; else echo "P11_INJ_FAIL $what"; fi
exit 0

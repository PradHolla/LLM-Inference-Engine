#!/usr/bin/env bash
# P11-KG5 on the box: the knowledge practice set, 3 passes for one arm (run one unit per arm, in parallel).
#   sudo systemd-run --unit=p11-kg5-q-judge --collect --working-directory=/opt/llm /bin/bash /opt/llm/infra/p11-kg5.sh q-judge
#   ... p11-kg5.sh privacy      # the 16 dev privacy requests, q-judge
#   ... p11-kg5.sh ktest        # the held-out knowledge test, 3 passes, q-judge
set -uo pipefail
cd /opt/llm || exit 1
export HOME=/home/ubuntu CRM_RERANK_URL=http://127.0.0.1:8010 FASTEMBED_CACHE_PATH=/opt/llm/hf-cache/fastembed
UV=/home/ubuntu/.local/bin/uv
arm="${1:?usage: p11-kg5.sh <arm> | privacy}"
fail=0
if [ "$arm" = ktest ]; then  # the held-out knowledge test, 80 questions x 3 passes, final q-judge
    for seed in 0 1 2; do
        $UV run tools/crmeval.py run --url http://localhost:8000 --arm q-judge --model Qwen/Qwen3-8B --org b2b \
            --split test --types knowledge_qa --concurrency 4 --run-seed "$seed" \
            --run-id "p11-ktest-q-judge-kg5-s$seed" --protocol service_v1 || fail=1
    done
elif [ "$arm" = privacy ]; then
    $UV run tools/crmeval.py run --url http://localhost:8000 --arm q-judge --model Qwen/Qwen3-8B --org b2b \
        --split dev --types private_customer_information --concurrency 4 --run-seed 0 \
        --run-id p11-privacy-dev-q-judge-kg5 --protocol service_v1 || fail=1
else
    for seed in 0 1 2; do
        $UV run tools/crmeval.py run --url http://localhost:8000 --arm "$arm" --model Qwen/Qwen3-8B --org b2b \
            --tasks-file crm/kqdev_b2b.json --split kqdev --types knowledge_qa --concurrency 4 --run-seed "$seed" \
            --run-id "p11-kqdev-$arm-kg5-s$seed" --protocol service_v1 || fail=1
    done
fi
if [ "$fail" = 0 ]; then echo "P11_KG5_DONE $arm"; else echo "P11_KG5_FAIL $arm"; fi
exit 0

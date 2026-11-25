#!/usr/bin/env bash
# Service desk agent evaluation on the box (Qwen must be served). Run detached:
#   sudo systemd-run --unit=p11-eval --collect --working-directory=/opt/llm /bin/bash /opt/llm/infra/p11-eval.sh practice
#   sudo systemd-run --unit=p11-eval --collect --working-directory=/opt/llm /bin/bash /opt/llm/infra/p11-eval.sh test q-b128
# practice: the 60 dev tasks x 3 thinking settings x 3 passes (seeds 0-2). test: the 240 test tasks, one arm,
# 3 passes. privacy: the 16 dev private-customer-information tasks in customer mode.
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv
TYPES=case_routing,policy_violation_identification,knowledge_qa
export FASTEMBED_CACHE_PATH=/opt/llm/hf-cache/fastembed
export CRM_RERANK_URL=http://127.0.0.1:8010  # GPU reranker (p11-app.sh rerank); search falls back to CPU if down

run() {  # split, arm, seed, run id
    $UV run tools/crmeval.py run --url http://localhost:8000 --arm "$2" --model Qwen/Qwen3-8B --org b2b \
        --split "$1" --types "$TYPES" --concurrency 4 --run-seed "$3" --run-id "$4" --protocol service_v1 \
        || echo "P11_EVAL_FAIL $4"
}

served=$(curl -s -m 5 http://localhost:8000/v1/models | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
[ "$served" = "Qwen/Qwen3-8B" ] || { echo "P11_EVAL_FAIL: served model is '$served', not Qwen"; exit 1; }
# Search would silently fall back to the CPU reranker and measure a different setup: refuse instead.
curl -sf -m 5 "$CRM_RERANK_URL/health" >/dev/null || { echo "P11_EVAL_FAIL: GPU reranker not up at $CRM_RERANK_URL"; exit 1; }

case "${1:-}" in
  practice)
    for seed in 0 1 2; do
        for arm in q-off q-b128 q-b128-first; do run dev "$arm" "$seed" "p11-eval-dev-$arm-s$seed"; done
    done
    ;;
  test)  # the held-out score: one arm, chosen on dev beforehand, 3 passes
    arm="${2:?usage: p11-eval.sh test <arm>}"
    for seed in 0 1 2; do run test "$arm" "$seed" "p11-eval-test-$arm-s$seed"; done
    ;;
  privacy)  # customer mode: the benchmark's private-information requests (dev), scoped to the logged-in customer
    arm="${2:?usage: p11-eval.sh privacy <arm>}"
    $UV run tools/crmeval.py run --url http://localhost:8000 --arm "$arm" --model Qwen/Qwen3-8B --org b2b \
        --split dev --types private_customer_information --concurrency 4 --run-seed 0 \
        --run-id "p11-privacy-dev-$arm" --protocol service_v1 || echo "P11_EVAL_FAIL privacy"
    ;;
  *) echo "usage: p11-eval.sh practice | test <arm> | privacy <arm>"; exit 2 ;;
esac
echo "P11_EVAL_DONE ${1}"
exit 0

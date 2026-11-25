#!/usr/bin/env bash
# Service desk agent evaluation on the box (Qwen must be served). Run detached:
#   sudo systemd-run --unit=p11-eval --collect --working-directory=/opt/llm /bin/bash /opt/llm/infra/p11-eval.sh practice
#   sudo systemd-run --unit=p11-eval --collect --working-directory=/opt/llm /bin/bash /opt/llm/infra/p11-eval.sh test q-b128
# practice: the 60 dev tasks x 3 thinking settings x 3 passes (seeds 0-2). test: the 240 test tasks, one arm, one pass.
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv
TYPES=case_routing,policy_violation_identification,knowledge_qa
export FASTEMBED_CACHE_PATH=/opt/llm/hf-cache/fastembed

run() {  # split, arm, seed, run id
    $UV run tools/crmeval.py run --url http://localhost:8000 --arm "$2" --model Qwen/Qwen3-8B --org b2b \
        --split "$1" --types "$TYPES" --concurrency 4 --run-seed "$3" --run-id "$4" --protocol service_v1 \
        || echo "P11_EVAL_FAIL $4"
}

served=$(curl -s -m 5 http://localhost:8000/v1/models | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
[ "$served" = "Qwen/Qwen3-8B" ] || { echo "P11_EVAL_FAIL: served model is '$served', not Qwen"; exit 1; }

case "${1:-}" in
  practice)
    for seed in 0 1 2; do
        for arm in q-off q-b128 q-b128-first; do run dev "$arm" "$seed" "p11-eval-dev-$arm-s$seed"; done
    done
    ;;
  test)
    arm="${2:?usage: p11-eval.sh test <arm>}"
    run test "$arm" 0 "p11-eval-test-$arm"
    ;;
  *) echo "usage: p11-eval.sh practice | test <arm>"; exit 2 ;;
esac
echo "P11_EVAL_DONE ${1}"
exit 0

#!/usr/bin/env bash
# One eval session on the box, in stages. Run detached, e.g.:
#   sudo systemd-run --unit=p11-session --collect --working-directory=/opt/llm /bin/bash /opt/llm/infra/p11-session.sh prepare
#   sudo systemd-run --unit=p11-session --collect --working-directory=/opt/llm /bin/bash /opt/llm/infra/p11-session.sh test q-off
# prepare: launch Qwen + GPU reranker, 12-question smoke (gated), then the 60 dev questions x 1 pass for q-off and
# q-b128. test <arm>: the 240 test questions x 3 passes. Each stage ends with P11_SESSION_DONE or P11_SESSION_FAIL.
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv
export FASTEMBED_CACHE_PATH=/opt/llm/hf-cache/fastembed CRM_RERANK_URL=http://127.0.0.1:8010
SHOW=1483,1489,1497,1412,1459,1384,717,762,767,699,765,773

fail() { echo "P11_SESSION_FAIL $*"; exit 1; }

case "${1:-}" in
  prepare)
    served=$(curl -s -m 5 http://localhost:8000/v1/models | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    if [ "$served" != "Qwen/Qwen3-8B" ]; then
        /bin/bash /opt/llm/infra/p11-app.sh launch qwen || fail "qwen launch"
    fi
    curl -sf -m 5 "$CRM_RERANK_URL/health" >/dev/null || /bin/bash /opt/llm/infra/p11-app.sh rerank
    curl -sf -m 5 "$CRM_RERANK_URL/health" >/dev/null || fail "GPU reranker not up"
    echo "P11_SESSION serving: Qwen + reranker up"

    rm -f results/p11-smoke-q-off.*.jsonl
    $UV run tools/crmeval.py run --url http://localhost:8000 --arm q-off --model Qwen/Qwen3-8B --org b2b --split dev \
        --types case_routing,policy_violation_identification,knowledge_qa --idx "$SHOW" --concurrency 4 \
        --run-id p11-smoke-q-off --protocol service_v1 || fail "smoke run"
    # Gate: 12 records, every one ended by respond, at least 2/3 routing and 2/3 policy right.
    $UV run --no-project python - <<'PY' || fail "smoke gate"
import json, sys
t = [json.loads(l) for l in open("results/p11-smoke-q-off.tasks.jsonl")]
ends = sum(x["end_reason"] == "respond" for x in t)
ok = lambda job: sum(x["reward"] == 1 for x in t if x["task_type"] == job)
r, p = ok("case_routing"), ok("policy_violation_identification")
print(f"P11_SESSION smoke: {len(t)} records, {ends} ended by respond, routing {r}/3, policy {p}/3")
sys.exit(0 if len(t) == 12 and ends == 12 and r >= 2 and p >= 2 else 1)
PY

    for arm in q-off q-b128; do
        $UV run tools/crmeval.py run --url http://localhost:8000 --arm "$arm" --model Qwen/Qwen3-8B --org b2b \
            --split dev --types case_routing,policy_violation_identification,knowledge_qa --concurrency 4 \
            --run-seed 0 --run-id "p11-cmp-dev-$arm-s0" --protocol service_v1 || fail "dev comparison $arm"
    done
    echo "P11_SESSION_DONE prepare"
    ;;
  kqdev)  # the synthetic knowledge practice set (crm/kqdev_b2b.json), one pass, then check before ktest
    arm="${2:?usage: p11-session.sh kqdev <arm>}"
    served=$(curl -s -m 5 http://localhost:8000/v1/models | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    if [ "$served" != "Qwen/Qwen3-8B" ]; then /bin/bash /opt/llm/infra/p11-app.sh launch qwen || fail "qwen launch"; fi
    curl -sf -m 5 "$CRM_RERANK_URL/health" >/dev/null || /bin/bash /opt/llm/infra/p11-app.sh rerank
    curl -sf -m 5 "$CRM_RERANK_URL/health" >/dev/null || fail "GPU reranker not up"
    $UV run tools/crmeval.py run --url http://localhost:8000 --arm "$arm" --model Qwen/Qwen3-8B --org b2b \
        --tasks-file crm/kqdev_b2b.json --split kqdev --types knowledge_qa --concurrency 4 --run-seed 0 \
        --run-id "p11-kqdev-$arm-s0" --protocol service_v1 || fail "kqdev run"
    n=$(wc -l < "results/p11-kqdev-$arm-s0.tasks.jsonl")
    [ "$n" -eq 40 ] || fail "kqdev produced $n of 40 records"
    echo "P11_SESSION_DONE kqdev $arm ($n records)"
    ;;
  ktest)  # knowledge only on the held-out test set, 3 passes
    arm="${2:?usage: p11-session.sh ktest <arm>}"
    for seed in 0 1 2; do
        $UV run tools/crmeval.py run --url http://localhost:8000 --arm "$arm" --model Qwen/Qwen3-8B --org b2b \
            --split test --types knowledge_qa --concurrency 4 --run-seed "$seed" \
            --run-id "p11-ktest-$arm-s$seed" --protocol service_v1 || fail "ktest seed $seed"
    done
    n=$(cat results/p11-ktest-"$arm"-s*.tasks.jsonl | wc -l)
    [ "$n" -eq 240 ] || fail "ktest produced $n of 240 records"
    echo "P11_SESSION_DONE ktest $arm ($n records)"
    ;;
  test)
    arm="${2:?usage: p11-session.sh test <arm>}"
    /bin/bash /opt/llm/infra/p11-eval.sh test "$arm" | grep -E "P11_EVAL|mean_reward" | tail -20
    n=$(cat results/p11-eval-test-"$arm"-s*.tasks.jsonl 2>/dev/null | wc -l)
    [ "$n" -eq 720 ] || fail "test produced $n of 720 records"
    echo "P11_SESSION_DONE test $arm ($n records)"
    ;;
  *) echo "usage: p11-session.sh prepare | kqdev <arm> | ktest <arm> | test <arm>"; exit 2 ;;
esac
exit 0

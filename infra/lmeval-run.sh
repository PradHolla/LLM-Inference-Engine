#!/usr/bin/env bash
# lm-eval against a local OpenAI-chat endpoint. URL/TASKS/LIMIT/TAG from the environment.
#   URL=http://localhost:8000/v1/chat/completions TASKS=nq_open LIMIT=20 TAG=sanity ./lmeval-run.sh
set -uo pipefail
cd /opt/llm || exit 1
LM=/opt/llm/.venv-eval/bin/lm-eval
URL="${URL:-http://localhost:8000/v1/chat/completions}"
TASKS="${TASKS:-nq_open}"
LIMIT="${LIMIT:-20}"
TAG="${TAG:-run}"
# nq_open/triviaqa stop at the first newline, which a thinking model emits inside <think>
# before writing anything -- every response came back empty. They also pin temperature 0.0,
# which Qwen3's card forbids. Both are overridden here.
GEN="${GEN:-until=[\"<|endoftext|>\"] temperature=0.6 top_p=0.95 top_k=20 max_gen_toks=2560}"
export HF_HOME=/opt/llm/hf-cache HF_HUB_OFFLINE=0 HF_DATASETS_TRUST_REMOTE_CODE=1
echo "[$(date -u +%H:%M:%S)] lm-eval tasks=$TASKS limit=$LIMIT url=$URL"
"$LM" --model local-chat-completions \
    --model_args "base_url=$URL,model=Qwen/Qwen3-8B,num_concurrent=8,max_retries=2,tokenized_requests=False" \
    --apply_chat_template --fewshot_as_multiturn --gen_kwargs $GEN \
    --tasks "$TASKS" --limit "$LIMIT" --log_samples \
    --output_path "results/lmeval-$TAG" 2>&1 | tail -25
rc=${PIPESTATUS[0]}
echo "[$(date -u +%H:%M:%S)] LMEVAL_DONE rc=$rc"
exit "$rc"

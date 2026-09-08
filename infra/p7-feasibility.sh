#!/usr/bin/env bash
# Phase 7 actuator check: does THIS vLLM support a thinking budget, priority scheduling
# and tool calls? Composes the flag set from what --help actually offers, because an
# unknown argument makes vLLM refuse to start at all. Runs ON the box.
#   ./p7-feasibility.sh
set -uo pipefail
cd /opt/llm || exit 1
UV=/home/ubuntu/.local/bin/uv
VLLM=/opt/llm/.venv-vllm/bin/vllm
RESULTS=/opt/llm/results
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "### vLLM version"
"$VLLM" --version 2>&1 | tail -1

HELP=$(mktemp)
"$VLLM" serve --help=all > "$HELP" 2>&1

has_flag() { grep -q -- "$1" "$HELP"; }

# POSITIVE CONTROL. Reading `--help` instead of `--help=all` reported every actuator
# ABSENT while the server itself demanded them. A detector that cannot find flags known
# to exist must declare ITSELF broken, not declare the features missing -- absence of
# evidence from a broken instrument is not evidence of absence.
for known in --model --max-model-len --quantization; do
    if ! has_flag "$known"; then
        echo "ABORT: flag detection is broken -- cannot find $known, which certainly exists."
        echo "  The help output is $(wc -l < "$HELP") lines; it is probably a summary."
        rm -f "$HELP"
        exit 1
    fi
done
echo "  detector positive control: found --model, --max-model-len, --quantization"

echo
echo "### which actuator flags does this build offer?"
ARGS=""
for f in --reasoning-parser --scheduling-policy --enable-auto-tool-choice --tool-call-parser; do
    if has_flag "$f"; then printf "  %-28s present\n" "$f"; else printf "  %-28s ABSENT\n" "$f"; fi
done

# Read the parser name out of --help rather than guessing it. vLLM prints its valid
# choices, and this build is the only authority on what it accepts -- the same reason the
# lab bench reads /v1/models instead of being told a model id.
# --reasoning-parser takes a free-form value with no enumerated choices, so it cannot be
# discovered. Try qwen3; a wrong value makes the launch fail, and the bare retry below
# separates "this flag is wrong" from "the server is broken".
RPARSER=qwen3
TPARSER=$(grep -oE '\--tool-call-parser \{[^}]*\}' "$HELP" | head -1 \
          | sed 's/.*{//; s/}//' | tr ',' '\n' | grep -iE 'hermes|qwen' | head -1)
[ -z "$TPARSER" ] && TPARSER=hermes
echo "  reasoning parser chosen: ${RPARSER:-<none offered>}"
echo "  tool-call parser chosen: ${TPARSER}"

# Compose only from flags that exist. A missing one is a finding, not a reason to abort.
if has_flag --reasoning-parser && [ -n "$RPARSER" ]; then ARGS="$ARGS --reasoning-parser $RPARSER"; fi
has_flag --scheduling-policy       && ARGS="$ARGS --scheduling-policy priority"
has_flag --enable-auto-tool-choice && ARGS="$ARGS --enable-auto-tool-choice"
has_flag --tool-call-parser        && ARGS="$ARGS --tool-call-parser $TPARSER"
echo "  composed: ${ARGS:-<none>}"

echo
echo "### launching"
# shellcheck disable=SC2086
if ! KV_PIN=10213733807 /opt/llm/infra/vllm-launch.sh p7 \
        --model Qwen/Qwen3-8B --quantization fp8 --max-model-len 16384 $ARGS; then
    echo "[$(ts)] launch FAILED with the composed flags; retrying with none to separate"
    echo "  a rejected flag from a broken build"
    if ! KV_PIN=10213733807 /opt/llm/infra/vllm-launch.sh p7-bare \
            --model Qwen/Qwen3-8B --quantization fp8 --max-model-len 16384; then
        echo "[$(ts)] ABORT: the server will not start even without the new flags"
        rm -f "$HELP"
        exit 1
    fi
    echo "[$(ts)] bare launch succeeded, so one of the composed flags is the problem"
fi
rm -f "$HELP"

echo
echo "### probes"
"$UV" run tools/p7probe.py --url http://localhost:8000 --out "$RESULTS/p7-feasibility.jsonl"
echo
echo "[$(ts)] P7_FEASIBILITY_DONE"
exit 0

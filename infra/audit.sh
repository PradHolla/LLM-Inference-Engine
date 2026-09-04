#!/usr/bin/env bash
# Mechanical sweep for the bug classes this project has actually hit. Written because the
# same defect kept living in one file and not its sibling: the external_ metric overwrite,
# the GPU-release wait, the invocation-scoped log. A checklist in CLAUDE.md did not stop
# that happening three times in one day; a grep does.
#   ./audit.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
hits=0
report() { printf '\n--- %s\n' "$1"; }
flag() { hits=$((hits+1)); printf '  FLAG %s\n' "$1"; }

report "1. Prometheus parsing must exclude external_ / _created / _by_reason (incidents 41, and today's convo bug)"
for f in $(grep -rl "prefix_cache\|num_requests_running\|prom" --include=*.py . 2>/dev/null); do
    if grep -q "prefix_cache\|num_requests" "$f"; then
        grep -q "external_" "$f" || flag "$f parses vllm metrics but never excludes external_"
    fi
done

report "2. Numeric checks on values that can legitimately be 0 must use 'is not None' (incident 28)"
grep -rn "if [a-z_]*\(count\|tokens\|rate\|usage\|waiting\|running\|limit\|cap\|hit\)[a-z_]*:" --include=*.py . 2>/dev/null \
  | grep -v "is not None\|is None\|#" | head -6 | while read -r l; do flag "truthiness on a numeric: $l"; done

report "3. Shell scripts must not end on a bare 'test && cmd' (incident 44)"
for f in infra/*.sh; do
    last=$(grep -vE '^\s*(#|$)' "$f" | tail -1)
    case "$last" in
        \[*\]\ \&\&*) flag "$f ends on a conditional whose exit status becomes the script's: $last" ;;
    esac
done

report "4. Bounded loops must set a flag the caller branches on, not rely on '&& break' (incident 30)"
grep -rn "&& break" infra/*.sh 2>/dev/null | while read -r l; do
    flag "'&& break' without an explicit success flag: $l"
done

report "5. Result files opened for append silently pool separate runs (incident 45)"
grep -rn 'open(.*\.out.*"a"\|open(.*, *"a")' --include=*.py tools/ labbench/ 2>/dev/null \
  | while read -r l; do flag "append-mode result file: $l"; done

report "6. Scripts run under systemd must not use home-relative paths (incident 38)"
grep -rn '~/\|\$HOME' infra/*.sh 2>/dev/null | grep -v "^\s*#" | grep -v "PEM=\|KEY" \
  | while read -r l; do flag "home-relative path in a box script: $l"; done

report "7. Anything launching vLLM must wait for GPU release, not just the port (incident 39)"
for f in $(grep -rl "systemd-run --unit=vllm" infra/ 2>/dev/null); do
    grep -q "memory.used" "$f" || flag "$f launches vllm without waiting for VRAM to free"
done

report "8. Token counts must come from usage, never from counting chunks (incident 32)"
grep -rn "out_tokens += 1\|tokens += 1\|len(chunks)" --include=*.py tools/ labbench/ 2>/dev/null \
  | while read -r l; do flag "possible chunk-as-token counting: $l"; done

report "9. Percentiles need 1/(1-p) samples (incident 11)"
for f in $(grep -rl "def pct\|percentile" --include=*.py tools/ labbench/ 2>/dev/null); do
    grep -q "1 - p\|1-p\|len(s) <\|len(xs) <" "$f" || flag "$f computes percentiles without a sample-count guard"
done

printf '\n=== %d flag(s)\n' "$hits"
exit 0

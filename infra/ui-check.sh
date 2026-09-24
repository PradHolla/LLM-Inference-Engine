#!/usr/bin/env bash
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1
LOGDIR="${TMPDIR:-/tmp}/ui-check-$$"
mkdir -p "$LOGDIR"
fail=0
note() { printf '  %-45s %s\n' "$1" "$2"; }

run_check() {
    local name="$1"; shift
    local log="$LOGDIR/${name// /_}.log"
    if "$@" >"$log" 2>&1; then
        note "$name" PASS
    else
        note "$name" FAIL
        cat "$log"
        fail=1
    fi
}

run_check "chat TypeScript" npm --prefix app/web run typecheck
run_check "chat production build" npm --prefix app/web run build
run_check "chat streaming and citation checks" npm --prefix app/web run check
run_check "labbench TypeScript" npm --prefix labbench/web run typecheck
run_check "labbench production build" npm --prefix labbench/web run build
run_check "UI artifacts and render invariants" node infra/ui-render-check.js

if [ "$fail" = 0 ]; then
    echo "  UI CHECK PASSED"
else
    echo "  UI CHECK FAILED"
fi
exit "$fail"

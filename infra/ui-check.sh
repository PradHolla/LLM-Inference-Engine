#!/usr/bin/env bash
# Pre-deploy gate for the lab bench UI. `node --check` validates syntax only, which is
# how a page that parsed cleanly rendered blank: a component was referenced after its
# definition had been deleted. Run this before every deploy.
#   ./infra/ui-check.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO" || exit 1
UI=labbench/ui
fail=0
note() { printf '  %-52s %s\n' "$1" "$2"; }

command -v node >/dev/null 2>&1 && { node --check "$UI/app.js" >/dev/null 2>&1 \
    && note "javascript parses" OK || { note "javascript parses" FAIL; fail=1; }; } \
  || note "javascript parses" "SKIPPED (no node)"

python3 - "$UI/app.js" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
defined = set(re.findall(r'function\s+([A-Z][A-Za-z0-9_]*)', s))
defined |= set(re.findall(r'const\s+([A-Z][A-Za-z0-9_]*)\s*=', s))
# React names arrive via a multi-line destructure
head = s[:2000]
defined |= {x.strip() for x in re.findall(r'[{,]\s*([A-Za-z][A-Za-z0-9_]*)', head)}
used = set(re.findall(r'<\$\{([A-Z][A-Za-z0-9_]*)\}', s))
missing = sorted(used - defined)
print(f"  {'every referenced component is defined':<52} "
      + ("OK" if not missing else "FAIL " + ", ".join(missing)))
sys.exit(1 if missing else 0)
PY
[ $? -eq 0 ] || fail=1

grep -q "dangerouslySetInnerHTML" <(grep -v '^\s*//' "$UI/app.js") \
  && { note "no dangerouslySetInnerHTML" FAIL; fail=1; } || note "no dangerouslySetInnerHTML" OK

if grep -qE 'https?://' "$UI/index.html" || grep -qE 'https?://[^l]' <(grep -v localhost "$UI/app.js"); then
    note "no external urls" FAIL; fail=1
else
    note "no external urls" OK
fi

# Emoji specifically, not all non-ASCII: CLAUDE.md section 8 permits arrows, em-dashes
# and math symbols as typography. A blanket non-ASCII test flags the chevrons and fails.
python3 - "$UI/app.js" "$UI/app.css" "$UI/index.html" <<'PYEMOJI'
import sys
RANGES = [(0x1F300,0x1FAFF),(0x1F000,0x1F2FF),(0x2600,0x27BF),(0xFE0F,0xFE0F),(0x1F1E6,0x1F1FF)]
bad = []
for path in sys.argv[1:]:
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        for ch in line:
            if any(a <= ord(ch) <= b for a, b in RANGES):
                bad.append(f"{path}:{n}:{ch!r}")
print(f"  {'no emoji (arrows and symbols allowed)':<52} "
      + ("OK" if not bad else "FAIL " + ", ".join(bad[:4])))
sys.exit(1 if bad else 0)
PYEMOJI
[ $? -eq 0 ] || fail=1

for f in index.html app.js app.css vendor/react.production.min.js \
         vendor/react-dom.production.min.js vendor/htm.js; do
    [ -s "$UI/$f" ] || { note "present: $f" FAIL; fail=1; }
done
note "all six files present and non-empty" OK

if command -v node >/dev/null 2>&1; then
    node infra/ui-render-check.js || fail=1
else
    note "components actually render" "SKIPPED (no node)"
fi

[ "$fail" = 0 ] && echo "  UI CHECK PASSED" || echo "  UI CHECK FAILED"
exit $fail

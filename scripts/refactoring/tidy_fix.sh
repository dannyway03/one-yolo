#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTPUT_DIR="$PROJECT_ROOT/results/clang-tidy"
FIXES="$OUTPUT_DIR/fixes.yaml"
LOG="$OUTPUT_DIR/clang-tidy.log"

cd "$PROJECT_ROOT"

[[ -f "$FIXES" ]] || { echo "No fixes file: $FIXES"; exit 1; }

echo "Applying fixes..."

tmpdir=$(mktemp -d)
cp "$FIXES" "$tmpdir/"

clang-apply-replacements-20 \
    -style=file \
    -format \
    -remove-change-desc-files \
    "$tmpdir"

rm -rf "$tmpdir"
echo "  → done"

echo ""
echo "=== Unfixed warnings ==="

python3 - "$FIXES" <<'EOF'
import sys
import yaml

def offset_to_line(filepath, offset):
    try:
        with open(filepath, "rb") as f:
            return f.read(offset).count(b"\n") + 1
    except OSError:
        return "?"

with open(sys.argv[1]) as f:
    data = yaml.safe_load(f) or {}

unfixed = [d for d in data.get("Diagnostics", [])
           if not d["DiagnosticMessage"].get("Replacements")]

if not unfixed:
    print("  (none)")
else:
    for d in unfixed:
        msg  = d["DiagnosticMessage"]
        fp   = msg["FilePath"]
        line = offset_to_line(fp, msg["FileOffset"])
        print(f"{fp}:{line}: [{d['DiagnosticName']}] {msg['Message']}")
EOF
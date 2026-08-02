#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTPUT_DIR="$PROJECT_ROOT/results/clang-tidy"
CONFIG="$PROJECT_ROOT/.clang-tidy"
FIXES="$OUTPUT_DIR/fixes.yaml"
LOG="$OUTPUT_DIR/clang-tidy.log"
COMPILE_DB="$PROJECT_ROOT/build/Release/compile_commands.json"

# Default search targets; override by passing dirs/files as arguments
if [[ $# -eq 0 ]]; then
    TARGETS=(src include samples tools)
else
    TARGETS=("$@")
fi

usage() {
    cat <<EOF
Usage: $0 [dir-or-file ...]

Run clang-tidy-20 on project sources and headers, excluding nlohmann.
Defaults to: src include samples tools

Output:
  log   : $LOG
  fixes : $FIXES
EOF
    exit 1
}

[[ -f "$CONFIG" ]]      || { echo "ERROR: .clang-tidy not found at $CONFIG"; exit 1; }
[[ -f "$COMPILE_DB" ]]  || { echo "ERROR: compile_commands.json not found at $COMPILE_DB"; exit 1; }

mkdir -p "$OUTPUT_DIR"
rm -f "$FIXES"
cd "$PROJECT_ROOT"

mapfile -t FILES < <(
    find "${TARGETS[@]}" \( -name "*.cpp" -o -name "*.h" \) | grep -v nlohmann
)

echo "=== clang-tidy-20 ==="
printf "  targets      : %s\n" "${TARGETS[*]}"
printf "  files found  : %d\n" "${#FILES[@]}"
echo   "  compile db   : $COMPILE_DB"
echo   "  log          : $LOG"
echo   "  fixes        : $FIXES"
echo

printf '%s\n' "${FILES[@]}" | xargs clang-tidy-20 \
    --config-file="$CONFIG" \
    -p "$COMPILE_DB" \
    --header-filter='^(?!.*/nlohmann/).*' \
    --export-fixes="$FIXES" \
    2>&1 | tee "$LOG"

echo
if [[ -f "$FIXES" ]]; then
    echo "Fixes written to: $FIXES"
else
    echo "No fixes generated."
fi
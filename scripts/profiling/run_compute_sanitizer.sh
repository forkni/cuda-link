#!/usr/bin/env bash
# Correctness gate — run before any perf profiling.
# Writes memcheck.log to benchmarks/results/sanitizer/<UTC-ts>/ and mirrors _latest.

set -euo pipefail

ts=$(date -u +"%Y-%m-%d_%H%M%S")
out="benchmarks/results/sanitizer/$ts"
mkdir -p "$out"

echo "==> compute-sanitizer memcheck → $out/memcheck.log"

compute-sanitizer \
    --tool memcheck \
    --leak-check full \
    --target-processes all \
    python benchmarks/bench_sweep.py --quick \
    2>&1 | tee "$out/memcheck.log"

exit_code=${PIPESTATUS[0]}

# Update _latest symlink
latest="benchmarks/results/sanitizer/_latest"
rm -f "$latest"
ln -s "$(realpath "$out")" "$latest"

if [ "$exit_code" -eq 0 ]; then
    echo "==> PASS: no sanitizer violations"
else
    echo "==> FAIL: sanitizer reported violations (exit $exit_code)"
fi
exit "$exit_code"

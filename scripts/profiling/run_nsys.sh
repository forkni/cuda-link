#!/usr/bin/env bash
# nsys profile — capture cross-process CUDA + NVTX timeline.
# spawn children each produce their own .nsys-rep; open both in nsys-ui for wall-clock alignment.
#
# Usage:
#   ./scripts/profiling/run_nsys.sh
#   nsys-ui benchmarks/results/nsys/_latest/run.nsys-rep

set -euo pipefail

ts=$(date -u +"%Y-%m-%d_%H%M%S")
out="benchmarks/results/nsys/$ts"
mkdir -p "$out"

export CUDALINK_NVTX=1
export CUDALINK_NVTX_VERBOSE=1

echo "==> nsys profile → $out/run.nsys-rep"

nsys profile \
    --trace=cuda,nvtx \
    --output "$out/run" \
    python benchmarks/bench_sweep.py --quick

# Analyze
echo "==> nsys analyze → $out/analyze.txt"
nsys analyze "$out/run.nsys-rep" > "$out/analyze.txt" 2>&1 || true

# Update _latest symlink
latest="benchmarks/results/nsys/_latest"
rm -f "$latest"
ln -s "$(realpath "$out")" "$latest"

echo "==> Reports in $out"
echo "    Open: nsys-ui benchmarks/results/nsys/_latest/run.nsys-rep"
echo "    Analyze summary: benchmarks/results/nsys/_latest/analyze.txt"

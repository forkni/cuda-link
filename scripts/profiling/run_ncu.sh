#!/usr/bin/env bash
# ncu kernel profiling — Sender export path (SpeedOfLight + MemoryWorkloadAnalysis).
#
# --replay-mode kernel  : required; CUDA Graphs on the export path makes range/app replay
#                         re-execute graphs and break IPC state.
# --clock-control base  : WDDM cannot lock clocks via nvidia-smi -pm 1; base is reproducible.
# --launch-count 5      : bounded to avoid WDDM TDR during long replay sessions.

set -euo pipefail

ts=$(date -u +"%Y-%m-%d_%H%M%S")
out="benchmarks/results/ncu/$ts"
mkdir -p "$out"

export CUDALINK_NVTX=1

echo "==> ncu (sender path) → $out/sender.ncu-rep"

ncu \
    --section SpeedOfLight,MemoryWorkloadAnalysis \
    --clock-control base \
    --launch-skip 5 --launch-count 5 \
    --replay-mode kernel \
    --target-processes all \
    --nvtx-include "cudalink.sender.export_frame@*" \
    -o "$out/sender" \
    python benchmarks/bench_sweep.py --quick

# Update _latest symlink
latest="benchmarks/results/ncu/_latest"
rm -f "$latest"
ln -s "$(realpath "$out")" "$latest"

echo "==> Report: $out/sender.ncu-rep"
echo "    Open: ncu-ui benchmarks/results/ncu/_latest/sender.ncu-rep"

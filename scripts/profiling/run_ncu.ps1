#!/usr/bin/env pwsh
# ncu kernel profiling — Sender export path (SpeedOfLight + MemoryWorkloadAnalysis).
#
# --replay-mode kernel  : required; CUDA Graphs on the export path makes range/app replay
#                         re-execute graphs and break IPC state.
# --clock-control base  : WDDM cannot lock clocks via nvidia-smi -pm 1; base is reproducible.
# --launch-count 5      : bounded to avoid WDDM TDR during long replay sessions.
# --nvtx-include        : focus collection on the Sender export phase only.
#
# Optional parameters:
#   -Set <name>   Replace default --section with --set <name> (e.g. "full" for all counters).
#                 Use with caution: --set full greatly increases replay time and TDR risk.
#                 If TDRs occur, reduce --launch-count to 2 or 3.

param(
    [string]$Set = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ts  = (Get-Date -Format "yyyy-MM-dd_HHmmss")
$out = "benchmarks/results/ncu/$ts"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$env:CUDALINK_NVTX = "1"
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = "1"  # suppress forrtl error 200 on WM_CLOSE

$sectionFlag = if ($Set) { "--set" }     else { "--section" }
$sectionVal  = if ($Set) { $Set }        else { "SpeedOfLight --section MemoryWorkloadAnalysis" }

Write-Host "==> ncu (sender path) → $out/sender.ncu-rep  [$sectionFlag $sectionVal]"

ncu `
    $sectionFlag $sectionVal `
    --clock-control base `
    --cache-control all `
    --import-source yes `
    --launch-skip 5 --launch-count 5 `
    --replay-mode kernel `
    --target-processes all `
    --nvtx `
    --nvtx-include "cudalink.exporter.slot*@*" `
    -o "$out/sender" `
    python benchmarks/bench_sweep.py --quick

# Update _latest
$latest = "benchmarks/results/ncu/_latest"
if (Test-Path $latest) { Remove-Item $latest -Recurse -Force }
Copy-Item -Recurse $out $latest

Write-Host "==> Report: $out/sender.ncu-rep"
Write-Host "    Open: ncu-ui benchmarks/results/ncu/_latest/sender.ncu-rep"

#!/usr/bin/env pwsh
# ncu kernel profiling — Receiver import path (SpeedOfLight + MemoryWorkloadAnalysis).
# Variant of run_ncu.ps1 filtered to the receiver NVTX phase instead of the sender.
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

Write-Host "==> ncu (receiver path) → $out/receiver.ncu-rep  [$sectionFlag $sectionVal]"

ncu `
    $sectionFlag $sectionVal `
    --clock-control base `
    --cache-control all `
    --import-source yes `
    --launch-skip 5 --launch-count 5 `
    --replay-mode kernel `
    --target-processes all `
    --nvtx `
    --nvtx-include "cudalink.receiver.import_frame@*" `
    -o "$out/receiver" `
    python benchmarks/bench_sweep.py --quick

# Update _latest (shares same _latest with run_ncu.ps1 under ncu/)
$latest = "benchmarks/results/ncu/_latest"
if (Test-Path $latest) { Remove-Item $latest -Recurse -Force }
Copy-Item -Recurse $out $latest

Write-Host "==> Report: $out/receiver.ncu-rep"
Write-Host "    Open: ncu-ui benchmarks/results/ncu/_latest/receiver.ncu-rep"

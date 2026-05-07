#!/usr/bin/env pwsh
# ncu kernel profiling — Receiver import path (SpeedOfLight + MemoryWorkloadAnalysis).
# Variant of run_ncu.ps1 filtered to the receiver NVTX phase instead of the sender.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ts  = (Get-Date -Format "yyyy-MM-dd_HHmmss")
$out = "benchmarks/results/ncu/$ts"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$env:CUDALINK_NVTX = "1"

Write-Host "==> ncu (receiver path) → $out/receiver.ncu-rep"

ncu `
    --section SpeedOfLight,MemoryWorkloadAnalysis `
    --clock-control base `
    --launch-skip 5 --launch-count 5 `
    --replay-mode kernel `
    --target-processes all `
    --nvtx-include "cudalink.receiver.import_frame@*" `
    -o "$out/receiver" `
    python benchmarks/bench_sweep.py --quick

# Update _latest (shares same _latest with run_ncu.ps1 under ncu/)
$latest = "benchmarks/results/ncu/_latest"
if (Test-Path $latest) { Remove-Item $latest -Recurse -Force }
Copy-Item -Recurse $out $latest

Write-Host "==> Report: $out/receiver.ncu-rep"
Write-Host "    Open: ncu-ui benchmarks/results/ncu/_latest/receiver.ncu-rep"

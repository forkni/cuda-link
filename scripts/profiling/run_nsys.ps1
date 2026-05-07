#!/usr/bin/env pwsh
# nsys profile — capture cross-process CUDA + NVTX timeline.
# spawn children each produce their own .nsys-rep; open both in nsys-ui for wall-clock alignment.
#
# Usage:
#   ./scripts/profiling/run_nsys.ps1
#   nsys-ui benchmarks/results/nsys/_latest/producer.nsys-rep benchmarks/results/nsys/_latest/consumer.nsys-rep

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ts  = (Get-Date -Format "yyyy-MM-dd_HHmmss")
$out = "benchmarks/results/nsys/$ts"
New-Item -ItemType Directory -Force -Path $out | Out-Null

Write-Host "==> nsys profile → $out/run.nsys-rep"
Write-Host "    Set CUDALINK_NVTX=1 (already exported by this script)"

$env:CUDALINK_NVTX = "1"
$env:CUDALINK_NVTX_VERBOSE = "1"

nsys profile `
    --trace=cuda,nvtx,osrt `
    --output "$out/run" `
    python benchmarks/bench_sweep.py --quick

# Analyze
Write-Host "==> nsys analyze → $out/analyze.txt"
nsys analyze "$out/run.nsys-rep" > "$out/analyze.txt" 2>&1

# Update _latest
$latest = "benchmarks/results/nsys/_latest"
if (Test-Path $latest) { Remove-Item $latest -Recurse -Force }
Copy-Item -Recurse $out $latest

Write-Host "==> Reports in $out"
Write-Host "    Open: nsys-ui benchmarks/results/nsys/_latest/run.nsys-rep"
Write-Host "    Analyze summary: benchmarks/results/nsys/_latest/analyze.txt"

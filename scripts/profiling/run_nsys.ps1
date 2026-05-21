#!/usr/bin/env pwsh
# nsys profile — capture CUDA + NVTX timeline for cuda-link export path.
#
# Two targets:
#   profile_export  — Python-only Exporter harness (no consumer process).
#                     Used for Cells A-nsys (graphs OFF) and B-nsys (graphs ON).
#   bench_sweep     — Full IPC roundtrip (producer + consumer spawn processes).
#                     Used for Cells C-nsys (graphs OFF) and D-nsys (graphs ON).
#
# Usage:
#   # Cell A-nsys: Python-only, graphs OFF
#   ./scripts/profiling/run_nsys.ps1 -Target profile_export -Graphs 0
#
#   # Cell B-nsys: Python-only, graphs ON
#   ./scripts/profiling/run_nsys.ps1 -Target profile_export -Graphs 1
#
#   # Cell C-nsys: Full IPC roundtrip, graphs OFF
#   ./scripts/profiling/run_nsys.ps1 -Target bench_sweep -Graphs 0
#
#   # Cell D-nsys: Full IPC roundtrip, graphs ON (default)
#   ./scripts/profiling/run_nsys.ps1 -Target bench_sweep -Graphs 1
#
#   # Open result:
#   nsys-ui benchmarks/results/nsys/_latest/run.nsys-rep

param(
    [ValidateSet("profile_export", "bench_sweep")]
    [string]$Target = "bench_sweep",

    [ValidateSet(0, 1)]
    [int]$Graphs = 1,

    [int]$Frames = 4000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$cellLabel = "graphs$(if ($Graphs -eq 1) { 'ON' } else { 'OFF' })"
$ts  = (Get-Date -Format "yyyy-MM-dd_HHmmss")
$out = "benchmarks/results/nsys/${Target}_${cellLabel}_${ts}"
New-Item -ItemType Directory -Force -Path $out | Out-Null

Write-Host "==> Target  : $Target"
Write-Host "==> Graphs  : $(if ($Graphs -eq 1) { 'ON' } else { 'OFF' })"
Write-Host "==> Output  : $out/run.nsys-rep"

$env:CUDALINK_NVTX            = "1"
$env:CUDALINK_NVTX_VERBOSE    = "1"
$env:CUDALINK_USE_GRAPHS      = "$Graphs"
$env:CUDALINK_TD_USE_GRAPHS   = "$Graphs"

if ($Target -eq "profile_export") {
    $cmd = "python scripts/profiling/profile_export.py --frames $Frames --export-profile"
} else {
    # bench_sweep --quick runs a single cell (512x512, uint8, 200 frames) for a quick smoke;
    # omit --quick for a full sweep across resolutions. For nsys timing use --quick to keep
    # trace size manageable — or pass --resolutions 1080 for a single 1080p cell.
    $cmd = "python benchmarks/bench_sweep.py --quick"
}

Write-Host "==> Command : $cmd"

$cmdParts = $cmd -split " "
nsys profile `
    --force-overwrite=true `
    --trace=cuda,nvtx,wddm `
    --wddm-memory-trace=false `
    --wddm-additional-events=true `
    --wddm-backtraces=true `
    --output "$out/run" `
    @cmdParts

# Analyze
Write-Host "==> nsys analyze → $out/analyze.txt"
nsys analyze "$out/run.nsys-rep" > "$out/analyze.txt" 2>&1

# Stats summary
Write-Host "==> nsys stats → $out/stats.txt"
nsys stats --report gputrace --format csv --output "$out/stats" "$out/run.nsys-rep" 2>&1

# Update _latest symlink (copy on Windows)
$latest = "benchmarks/results/nsys/_latest"
if (Test-Path $latest) { Remove-Item $latest -Recurse -Force }
Copy-Item -Recurse $out $latest

Write-Host "==> Reports in $out"
Write-Host "    Open in GUI : nsys-ui benchmarks/results/nsys/_latest/run.nsys-rep"
Write-Host "    Stats CSV   : benchmarks/results/nsys/_latest/stats_gputrace.csv"
Write-Host "    Analysis    : benchmarks/results/nsys/_latest/analyze.txt"

#!/usr/bin/env pwsh
# ncu kernel profiling — Sender export path driven by the TD pipeline.
#
# Drives td_exporter/example_sender_python.py while TouchDesigner (CUDA_Link_Example.toe)
# runs as the consumer (Receiver-A).  Uses the v5 async-flush-probe env stack so ncu
# kernel data is directly comparable to benchmarks/results/nsys/td_pipeline_v5_*.
#
# --replay-mode kernel  : required; CUDA Graphs on the export path makes range/app replay
#                         re-execute graphs and break IPC state (PROFILING.md §5).
# --clock-control base  : WDDM cannot lock clocks via nvidia-smi -pm 1.
# --launch-count        : 5 for the HW pass (safe), 2 for -Set full (TDR budget on WDDM).
# --nvtx-include        : focus collection on the Sender export phase only.
#
# Correctness gate note:
#   run_compute_sanitizer.ps1 requires the missing bench_sweep.py and is skipped here.
#   Relies on the v5 nsys capture's clean state as the proxy correctness baseline.
#   Restore bench_sweep.py in a follow-up task to re-arm the sanitizer runner.
#
# Capture protocol (order matters — avoids nsys CUPTI hook interaction with IPC handles):
#   1. Launch TouchDesigner with CUDA_Link_Example.toe OUTSIDE this script.
#      Wait until Receiver-A reports 60 fps in the TD viewer.
#   2. In a fresh PowerShell at repo root, run this script (HW pass):
#        ./scripts/profiling/run_ncu_td_pipeline.ps1
#   3. Let the sender stream ~30 s of steady state, then Ctrl+C to finalize.
#   4. Repeat with -Set full (SW pass, separate timestamp dir):
#        ./scripts/profiling/run_ncu_td_pipeline.ps1 -Set full
#   5. Close TouchDesigner.
#
# Optional parameters:
#   -Set <name>        Replace default --section with --set <name> (e.g. "full" for all
#                      counters).  When full, --launch-count is capped at 2 to limit TDR
#                      exposure on WDDM.  If TDRs still occur, use -LaunchCount 1.
#   -LaunchCount <n>   Override the automatic launch-count selection (0 = auto).

param(
    [string]$Set         = "",
    [int]   $LaunchCount = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ts  = (Get-Date -Format "yyyy-MM-dd_HHmmss")
$out = "benchmarks/results/ncu/$ts"
New-Item -ItemType Directory -Force -Path $out | Out-Null

# v5 async-flush-probe env — matches scripts/probes/v5_capture_producer.cmd:37-46
$env:CUDALINK_TD_PERSIST_STREAM  = "1"
$env:CUDALINK_TD_STREAM_PRIO     = "normal"
$env:CUDALINK_EXPORT_SYNC        = "0"
$env:CUDALINK_EXPORT_FLUSH_PROBE = "1"
$env:CUDALINK_LIB_STREAM_PRIO   = "high"
$env:CUDALINK_NVTX               = "1"
$env:CUDALINK_NVTX_VERBOSE       = "1"
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = "1"  # suppress forrtl error 200 on WM_CLOSE

# Section and launch-count selection
$sectionFlag  = if ($Set)          { "--set"     } else { "--section" }
$sectionVal   = if ($Set)          { $Set        } else { "SpeedOfLight --section MemoryWorkloadAnalysis" }

$autoCount    = if ($Set -eq "full") { 2 } else { 5 }
$resolvedCount = if ($LaunchCount -gt 0) { $LaunchCount } else { $autoCount }

Write-Host "==> ncu (TD pipeline — sender path) → $out/sender.ncu-rep"
Write-Host "    [$sectionFlag $sectionVal]  --launch-count $resolvedCount"
Write-Host "    Env: EXPORT_SYNC=0  FLUSH_PROBE=1  PERSIST_STREAM=1  STREAM_PRIO=normal"
Write-Host ""
Write-Host "    Prerequisites:"
Write-Host "      1. TouchDesigner with CUDA_Link_Example.toe already open and streaming."
Write-Host "      2. Receiver-A active at 60 fps in the TD viewer."
Write-Host "    Send Ctrl+C after ~30 s of steady-state frames to finalize the report."
Write-Host ""

ncu `
    $sectionFlag $sectionVal `
    --clock-control base `
    --cache-control all `
    --import-source yes `
    --launch-skip 5 --launch-count $resolvedCount `
    --replay-mode kernel `
    --target-processes all `
    --nvtx `
    --nvtx-include "cudalink.exporter.slot*@*" `
    -o "$out/sender" `
    python td_exporter/example_sender_python.py

# Update _latest
$latest = "benchmarks/results/ncu/_latest"
if (Test-Path $latest) { Remove-Item $latest -Recurse -Force }
Copy-Item -Recurse $out $latest

Write-Host ""
Write-Host "==> Report: $out/sender.ncu-rep"
Write-Host "    Open: ncu-ui benchmarks/results/ncu/_latest/sender.ncu-rep"
Write-Host ""
Write-Host "    SOL classification (docs/PROFILING.md §3):"
Write-Host "      DRAM >= 80% → memory-bandwidth bound (expected for D2D copy)"
Write-Host "      SM   >= 60% → compute-bound (unexpected; investigate)"
Write-Host "      Both < 30% → latency- or congestion-bound (investigate)"
Write-Host ""
Write-Host "    After SW pass (-Set full), write findings to:"
Write-Host "      benchmarks/results/ncu/$ts/findings.md"

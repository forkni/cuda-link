#!/usr/bin/env pwsh
# Correctness gate — run before any perf profiling.
# Writes memcheck.log to benchmarks/results/sanitizer/<UTC-ts>/ and mirrors _latest.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ts  = (Get-Date -Format "yyyy-MM-dd_HHmmss")
$out = "benchmarks/results/sanitizer/$ts"
New-Item -ItemType Directory -Force -Path $out | Out-Null

Write-Host "==> compute-sanitizer memcheck → $out/memcheck.log"

compute-sanitizer `
    --tool memcheck `
    --leak-check full `
    --target-processes all `
    python benchmarks/bench_sweep.py --quick `
    2>&1 | Tee-Object -FilePath "$out/memcheck.log"

$exitCode = $LASTEXITCODE

# Update _latest pointer (copy, Windows has no guaranteed symlink permission)
$latest = "benchmarks/results/sanitizer/_latest"
if (Test-Path $latest) { Remove-Item $latest -Recurse -Force }
Copy-Item -Recurse $out $latest

if ($exitCode -eq 0) {
    Write-Host "==> PASS: no sanitizer violations"
} else {
    Write-Host "==> FAIL: sanitizer reported violations (exit $exitCode)"
}
exit $exitCode

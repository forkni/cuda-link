# Sample the Windows "GPU Engine" counters, per process: the numbers Task Manager shows.
#
# Task Manager's "3D" value is the share of time a process has work on the 3D queue. It
# includes UI drawing and present waits, so it can read 40-50 % while NVML SM % (see
# gpu_load_probe.py) stays near 10 %. See docs/BENCHMARKS.md "GPU load".
#
# Usage:
#   powershell -File scripts/profiling/gpu_engine_probe.ps1 -Seconds 30 -Tag G1_tox [-Out results.md]
#   -Series -ProcessIds "1234,5678" prints one 3D value per second per process instead of the summary.
param(
    [int]$Seconds = 30,
    [string]$Tag = "probe",
    [string]$Out = "",
    [switch]$Series,
    [string]$ProcessIds = ""   # comma list, e.g. "1234,5678"
)

$counter = '\GPU Engine(*)\Utilization Percentage'

if ($Series) {
    $row = for ($i = 0; $i -lt $Seconds; $i++) {
        $s = (Get-Counter '\GPU Engine(*engtype_3D)\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples
        ($ProcessIds.Split(',', [System.StringSplitOptions]::RemoveEmptyEntries) | ForEach-Object {
            $p = $_
            $v = ($s | Where-Object InstanceName -like "pid_${p}_*" | Measure-Object CookedValue -Sum).Sum
            "{0}:{1:N0}" -f $p, $v
        }) -join '/'
    }
    $row -join ' '
    return
}

$per = @{}   # "pid engtype" -> samples
$tot = New-Object System.Collections.Generic.List[double]
for ($i = 0; $i -lt $Seconds; $i++) {
    $s = (Get-Counter $counter -ErrorAction SilentlyContinue).CounterSamples
    $sum3d = 0.0
    foreach ($c in $s) {
        if ($c.InstanceName -match 'pid_(\d+)_.*engtype_(\w+)$') {
            $k = "$($Matches[1]) $($Matches[2])"
            if (-not $per.ContainsKey($k)) { $per[$k] = New-Object System.Collections.Generic.List[double] }
            $per[$k].Add($c.CookedValue)
            if ($Matches[2] -eq '3D') { $sum3d += $c.CookedValue }
        }
    }
    $tot.Add($sum3d)
}

function Get-Quantile($v, $f) {
    $a = @($v | Sort-Object)
    $a[[math]::Min($a.Count - 1, [int]($a.Count * $f))]
}
function Format-Q($v) {
    $mx = ($v | Measure-Object -Maximum).Maximum
    "{0:N1}/{1:N1}/{2:N1}" -f (Get-Quantile $v 0.5), (Get-Quantile $v 0.95), $mx
}

$lines = @("- $Tag ($Seconds s, Windows GPU Engine %, p50/p95/max): 3D total $(Format-Q $tot)")
$rows = foreach ($k in $per.Keys) {
    $v = $per[$k]
    $mx = ($v | Measure-Object -Maximum).Maximum
    if ($mx -ge 1) {
        $p, $e = $k.Split(' ')
        $n = (Get-Process -Id $p -ErrorAction SilentlyContinue).ProcessName
        [pscustomobject]@{ Name = "$n($p) $e"; Q = (Format-Q $v); Max = $mx }
    }
}
foreach ($r in ($rows | Sort-Object Max -Descending)) { $lines += "  - $($r.Name): $($r.Q)" }
$lines | ForEach-Object { Write-Output $_ }
if ($Out) { $lines | Add-Content -Encoding utf8 $Out }

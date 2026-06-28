# Quick consistency check before submission / interview.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$report = "data\evaluation\report.json"
$refine = "data\evaluation\refinement_comparison.json"
$pers = "data\evaluation\personalization_comparison.json"

$ok = $true
Write-Host "=== Evaluation Consistency Check ===" -ForegroundColor Cyan

function Test-Json($path) {
    if (-not (Test-Path $path)) {
        Write-Host "MISSING $path" -ForegroundColor Red
        return $null
    }
    return Get-Content $path -Raw | ConvertFrom-Json
}

$r = Test-Json $report
$rf = Test-Json $refine
$p = Test-Json $pers

foreach ($file in @($report, $refine, $pers)) {
    if (-not (Test-Path $file)) { $ok = $false; continue }
    $j = Get-Content $file -Raw | ConvertFrom-Json
    if ($j.evaluated_queries -ne 2076) {
        Write-Host "FAIL $file evaluated_queries=$($j.evaluated_queries) (expected 2076)" -ForegroundColor Red
        $ok = $false
    } else {
        Write-Host "OK   $file -> 2076 queries" -ForegroundColor Green
    }
}

if ($r -and $rf) {
    $reportBm25 = $r.metrics.bm25.MAP
    $compareBm25 = $rf.models.bm25.before.MAP
    $delta = [math]::Abs($reportBm25 - $compareBm25)
    if ($reportBm25 -lt 0.1) {
        Write-Host "FAIL report.json bm25 MAP=$reportBm25 (stale - rebuild required)" -ForegroundColor Red
        Write-Host "     comparison before MAP=$compareBm25" -ForegroundColor Yellow
        Write-Host "     Fix: .\scripts\rebuild_baseline_report.ps1" -ForegroundColor Yellow
        $ok = $false
    } elseif ($delta -gt 0.05) {
        Write-Host "WARN report vs comparison bm25 MAP differ by $([math]::Round($delta,4))" -ForegroundColor Yellow
    } else {
        Write-Host "OK   baseline MAP aligned (bm25 report=$reportBm25 compare=$compareBm25)" -ForegroundColor Green
    }
}

foreach ($port in @(8000, 8001, 8003, 8004)) {
    try {
        $null = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 5 -UseBasicParsing
        Write-Host "OK   service port $port" -ForegroundColor Green
    } catch {
        Write-Host "DOWN service port $port" -ForegroundColor Yellow
    }
}

if ($ok) {
    Write-Host ""
    Write-Host "All critical checks passed." -ForegroundColor Green
    exit 0
}
Write-Host ""
Write-Host "Fix issues above before submission." -ForegroundColor Red
exit 1

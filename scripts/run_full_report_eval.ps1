# Full evaluation for Arabic report - ALL 2076 qrels queries (NO --query-limit)
# Requires: preprocessing :8001, retrieval :8003, query_refinement :8004

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUNBUFFERED = "1"

Write-Host "=== IR Full Report Evaluation (2076 queries) ===" -ForegroundColor Cyan
Write-Host "Ensure these are running in separate terminals:"
Write-Host "  py preprocessing_service\main.py"
Write-Host "  py retrieval_service\main.py"
Write-Host "  py query_refinement\main.py"
Write-Host ""

foreach ($port in @(8001, 8003, 8004)) {
    try {
        $null = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 30 -UseBasicParsing
        Write-Host "OK  port $port" -ForegroundColor Green
    } catch {
        Write-Host "FAIL port $port - start the service first." -ForegroundColor Red
        exit 1
    }
}

$reportPath = "data\evaluation\report.json"
$needsBaseline = $true
if (Test-Path $reportPath) {
    $reportJson = Get-Content $reportPath -Raw | ConvertFrom-Json
    $bm25Map = $reportJson.metrics.bm25.MAP
    $hasFullRun = ($reportJson.evaluated_queries -eq 2076) -and ($bm25Map -ge 0.1)
    if ($hasFullRun) {
        $needsBaseline = $false
        Write-Host ""
        Write-Host "Baseline report.json valid (2076 queries, bm25 MAP=$bm25Map) - skipping Step 1." -ForegroundColor Green
    } elseif ($reportJson.evaluated_queries -eq 2076) {
        Write-Host ""
        Write-Host "WARNING: report.json has 2076 queries but bm25 MAP=$bm25Map (stale). Re-running baseline." -ForegroundColor Yellow
    }
}

if ($needsBaseline) {
    Write-Host ""
    Write-Host "Step 1/4: baseline metrics - all 6 models, 2076 queries..." -ForegroundColor Yellow
    py -u scripts\run_evaluation.py --workers 1
    if (-not $?) { exit 1 }
}

Write-Host ""
Write-Host "Step 2/4: before/after refinement + personalization - 2076 queries each..." -ForegroundColor Yellow
Write-Host "Skipping if comparison files already valid (run with -ForceComparisons to redo)." -ForegroundColor DarkGray
$forceComparisons = $args -contains "-ForceComparisons"
$skipComparisons = $false
if (-not $forceComparisons) {
    $refPath = "data\evaluation\refinement_comparison.json"
    $persPath = "data\evaluation\personalization_comparison.json"
    if ((Test-Path $refPath) -and (Test-Path $persPath)) {
        $refJson = Get-Content $refPath -Raw | ConvertFrom-Json
        $persJson = Get-Content $persPath -Raw | ConvertFrom-Json
        if ($refJson.evaluated_queries -eq 2076 -and $persJson.evaluated_queries -eq 2076) {
            $skipComparisons = $true
            Write-Host "Comparison JSON files already complete - skipping Step 2." -ForegroundColor Green
        }
    }
}
if (-not $skipComparisons) {
    py -u scripts\run_evaluation.py --compare-all-extras --workers 1
    if (-not $?) { exit 1 }
}

Write-Host ""
Write-Host "Step 3/4: regenerate all charts..." -ForegroundColor Yellow
py scripts\plot_evaluation_charts.py
if (-not $?) { exit 1 }
py scripts\plot_evaluation_charts.py --comparison data\evaluation\refinement_comparison.json
if (-not $?) { exit 1 }
if (Test-Path "data\evaluation\personalization_comparison.json") {
    py scripts\plot_evaluation_charts.py --personalization-comparison data\evaluation\personalization_comparison.json
    if (-not $?) { exit 1 }
}

Write-Host ""
Write-Host "Step 4/4: markdown summary table for interview..." -ForegroundColor Yellow
py scripts\generate_eval_summary.py
if (-not $?) { exit 1 }

Write-Host ""
Write-Host "Done. Verify ALL JSON files show evaluated_queries: 2076" -ForegroundColor Green
$files = @(
    "data\evaluation\report.json",
    "data\evaluation\refinement_comparison.json",
    "data\evaluation\personalization_comparison.json"
)
foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "--- $file ---" -ForegroundColor Cyan
        Select-String -Path $file -Pattern "evaluated_queries"
    }
}

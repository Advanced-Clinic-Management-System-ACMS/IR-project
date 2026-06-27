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
    if ($reportJson.evaluated_queries -eq 2076) {
        $needsBaseline = $false
        Write-Host ""
        Write-Host "Baseline report.json already has 2076 queries - skipping Step 1." -ForegroundColor Green
    }
}

if ($needsBaseline) {
    Write-Host ""
    Write-Host "Step 1/4: baseline metrics - all 6 models, 2076 queries..." -ForegroundColor Yellow
    py -u scripts\run_evaluation.py --workers 6
    if (-not $?) { exit 1 }
}

Write-Host ""
Write-Host "Step 2/4: before/after refinement + personalization - 2076 queries each..." -ForegroundColor Yellow
py -u scripts\run_evaluation.py --compare-all-extras --workers 6
if (-not $?) { exit 1 }

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

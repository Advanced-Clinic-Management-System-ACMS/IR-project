# Rebuild baseline report.json when metrics are stale (MAP too low vs comparison files).
# Requires: preprocessing :8001, retrieval :8003

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUNBUFFERED = "1"

Write-Host "=== Rebuild Baseline Evaluation (6 models x 2076 queries) ===" -ForegroundColor Cyan
Write-Host "Use workers=1 to avoid RAM exhaustion on retrieval service." -ForegroundColor Yellow
Write-Host ""

foreach ($port in @(8001, 8003)) {
    try {
        $null = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 30 -UseBasicParsing
        Write-Host "OK  port $port" -ForegroundColor Green
    } catch {
        Write-Host "FAIL port $port - start services first:" -ForegroundColor Red
        Write-Host "  py preprocessing_service\main.py"
        Write-Host "  py retrieval_service\main.py"
        exit 1
    }
}

$reportPath = "data\evaluation\report.json"
if (Test-Path $reportPath) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backup = "data\evaluation\report.backup.$stamp.json"
    Copy-Item $reportPath $backup
    Write-Host "Backed up old report to $backup" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Running baseline evaluation (expect several hours)..." -ForegroundColor Yellow
py -u scripts\run_evaluation.py --workers 1 --output data\evaluation\report.json
if (-not $?) { exit 1 }

py scripts\plot_evaluation_charts.py
py scripts\generate_eval_summary.py

Write-Host ""
Write-Host "Done. Verify bm25 MAP is ~0.32 (not ~0.01)." -ForegroundColor Green
Select-String -Path $reportPath -Pattern '"MAP"'

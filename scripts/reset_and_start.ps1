# Stop hung IR services on ports 8001/8003/8004/8000 and restart cleanly.
$ErrorActionPreference = "SilentlyContinue"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "=== Stopping old processes on IR ports ===" -ForegroundColor Yellow
foreach ($port in @(8000, 8001, 8002, 8003, 8004, 8005)) {
    $lines = netstat -ano | Select-String ":$port\s.*LISTENING"
    foreach ($line in $lines) {
        $pid = ($line -split '\s+')[-1]
        if ($pid -match '^\d+$') {
            Write-Host "Killing PID $pid on port $port"
            taskkill /PID $pid /F | Out-Null
        }
    }
}
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "=== Start these in SEPARATE terminals (wait for each before next) ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Terminal 1:" -ForegroundColor Green
Write-Host '  cd "d:\five year\ir"'
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  py preprocessing_service\main.py"
Write-Host ""
Write-Host "Terminal 2 (WAIT until you see 'Uvicorn running on 8003'):" -ForegroundColor Green
Write-Host '  cd "d:\five year\ir"'
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  py retrieval_service\main.py"
Write-Host ""
Write-Host "Terminal 3:" -ForegroundColor Green
Write-Host '  cd "d:\five year\ir"'
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  py query_refinement\main.py"
Write-Host ""
Write-Host "Terminal 4 (query refinement):" -ForegroundColor Green
Write-Host '  cd "d:\five year\ir"'
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  py query_refinement\main.py"
Write-Host ""
Write-Host "Terminal 5 (optional evaluation API):" -ForegroundColor Green
Write-Host '  cd "d:\five year\ir"'
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  py evaluation_service\main.py"
Write-Host ""
Write-Host "Terminal 6 (UI + validation):" -ForegroundColor Green
Write-Host '  cd "d:\five year\ir"'
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  py ui_gateway\main.py"
Write-Host "  .\scripts\validate_evaluation.ps1"
Write-Host ""
Write-Host "Rebuild stale baseline (if MAP ~0.01):" -ForegroundColor Yellow
Write-Host "  .\scripts\rebuild_baseline_report.ps1"
Write-Host ""

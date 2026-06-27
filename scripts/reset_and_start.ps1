# Stop hung IR services on ports 8001/8003/8004/8000 and restart cleanly.
$ErrorActionPreference = "SilentlyContinue"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "=== Stopping old processes on IR ports ===" -ForegroundColor Yellow
foreach ($port in @(8000, 8001, 8003, 8004)) {
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
Write-Host "Terminal 4 (ONLY after 8003 health works):" -ForegroundColor Green
Write-Host '  cd "d:\five year\ir"'
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  Invoke-WebRequest http://127.0.0.1:8003/health -UseBasicParsing"
Write-Host "  py -u scripts\run_evaluation.py --compare-all-extras --workers 1"
Write-Host ""

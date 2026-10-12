$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "The .venv is missing. Run setup_laptop_dev.ps1 first." -ForegroundColor Red
    exit 1
}
& $Python (Join-Path $ProjectRoot "scripts\run_laptop_tests.py")
exit $LASTEXITCODE

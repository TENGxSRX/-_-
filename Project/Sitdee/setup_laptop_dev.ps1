param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

Write-Host "=== Office posture: Acer lightweight setup ===" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"
Write-Host "This setup installs only the lightweight requirements. It never downloads datasets."

$pythonCommand = $null
$pythonArgs = @()
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($null -ne $pyLauncher) {
    & py -3.12 --version *> $null
    if ($LASTEXITCODE -eq 0) {
        $pythonCommand = "py"
        $pythonArgs = @("-3.12")
    }
}
if ($null -eq $pythonCommand) {
    $pythonLauncher = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pythonLauncher) {
        $pythonVersion = & python --version 2>&1
        if ($pythonVersion -match "Python 3\.12\.") {
            $pythonCommand = "python"
        }
    }
}
if ($null -eq $pythonCommand) {
    Write-Host "FAIL: Python 3.12 (64-bit) was not found. Install it, then run this script again." -ForegroundColor Red
    exit 1
}

$venvPath = Join-Path $ProjectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating .venv with Python 3.12..." -ForegroundColor Yellow
    & $pythonCommand @pythonArgs -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw "Could not create .venv" }
} else {
    $venvVersion = & $venvPython --version 2>&1
    if ($venvVersion -notmatch "Python 3\.12\.") {
        Write-Host "FAIL: existing .venv is not Python 3.12. It was not deleted or changed." -ForegroundColor Red
        exit 1
    }
}
& $venvPython -c "import struct, sys; assert sys.version_info[:2] == (3, 12); assert struct.calcsize('P') * 8 == 64"
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: .venv must use 64-bit Python 3.12. It was not deleted or changed." -ForegroundColor Red
    exit 1
}

Write-Host "Installing lightweight requirements..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $venvPython -m pip install -r (Join-Path $ProjectRoot "requirements-laptop.txt")
if ($LASTEXITCODE -ne 0) { throw "lightweight dependency installation failed" }

Write-Host "Running non-destructive health check..." -ForegroundColor Yellow
& $venvPython (Join-Path $ProjectRoot "scripts\laptop_health_check.py")
$healthCode = $LASTEXITCODE

$testCode = 0
if (-not $SkipTests) {
    Write-Host "Running Acer lightweight tests..." -ForegroundColor Yellow
    & $venvPython (Join-Path $ProjectRoot "scripts\run_laptop_tests.py")
    $testCode = $LASTEXITCODE
}

Write-Host ""
Write-Host "=== SETUP SUMMARY ===" -ForegroundColor Cyan
if ($healthCode -eq 0) {
    Write-Host "Health check: PASS" -ForegroundColor Green
} else {
    Write-Host "Health check: NEEDS FIX" -ForegroundColor Red
}
if ($SkipTests) {
    Write-Host "Lightweight tests: SKIPPED"
} elseif ($testCode -eq 0) {
    Write-Host "Lightweight tests: PASS" -ForegroundColor Green
} else {
    Write-Host "Lightweight tests: NEEDS FIX" -ForegroundColor Red
}
Write-Host "No dataset was downloaded. No raw/model file was deleted."
if (($healthCode -eq 0) -and (($SkipTests) -or ($testCode -eq 0))) {
    Write-Host "LAPTOP_SETUP_READY" -ForegroundColor Green
    exit 0
}
Write-Host "LAPTOP_SETUP_NEEDS_FIX" -ForegroundColor Red
exit 1

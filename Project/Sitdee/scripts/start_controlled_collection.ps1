param(
    [Parameter(Mandatory=$true)] [string]$SubjectId,
    [Parameter(Mandatory=$true)] [string]$ClipId,
    [ValidateSet('HIKVISION_MAIN','HIKVISION_SUB')]
    [string]$Profile = 'HIKVISION_MAIN',
    [ValidateSet('RTSP_DESKTOP','RTSP_BALANCED','RTSP_LOW_LATENCY')]
    [string]$Preset = 'RTSP_LOW_LATENCY'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Project virtual environment was not found.' }
$envFile = Join-Path $ProjectRoot '.env'
$hasLocalConfig = Test-Path -LiteralPath $envFile
$hasProcessConfig = [bool]$env:POSTURE_RTSP_URL -or [bool]$env:POSTURE_CAMERA_HOST
if (-not $hasLocalConfig -and -not $hasProcessConfig) {
    throw 'RTSP configuration is missing. Configure local .env or process environment variables; no connection was attempted.'
}
& $Python (Join-Path $ProjectRoot 'scripts\collect_controlled_posture.py') --source rtsp --profile $Profile --preset $Preset --subject-id $SubjectId --clip-id $ClipId
exit $LASTEXITCODE

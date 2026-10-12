param(
    [string]$OutputPath = "outputs\laptop_dev_bundle_v5.zip"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $ProjectRoot

$resolvedOutput = [IO.Path]::GetFullPath((Join-Path $ProjectRoot $OutputPath))
$stage = Join-Path ([IO.Path]::GetTempPath()) ("office-posture-laptop-" + [Guid]::NewGuid().ToString("N"))
$files = @(
    "README.md",
    ".gitignore",
    "pytest.ini",
    "requirements-laptop.txt",
    "setup_laptop_dev.ps1",
    "start_laptop_dev.bat",
    "run_rtsp_posture.bat",
    "configs\posture_engine.yaml",
    "configs\laptop_low_spec.yaml",
    "configs\rtsp_profiles.yaml",
    "configs\seat_zones.example.yaml",
    "configs\seat_layout_example.yaml",
    "configs\seat_layout_hikvision.example.yaml",
    ".env.example",
    "metadata\pose_landmarker_heavy.task",
    "outputs\models\rgb_native_v4\model.joblib",
    "outputs\models\rgb_native_v4\feature_config.json",
    "outputs\models\rgb_native_v4\labels.json",
    "outputs\models\rgb_native_v4\metadata.json",
    "scripts\laptop_health_check.py",
    "scripts\run_laptop_tests.py",
    "scripts\run_laptop_tests.ps1",
    "src",
    "scripts",
    "tests",
    "fixtures",
    "reports\laptop_file_manifest.md",
    "reports\laptop_file_manifest_th.md",
    "reports\acer_python_setup_th.md",
    "reports\two_machine_workflow_th.md",
    "reports\acer_daily_workflow_th.md",
    "reports\git_repository_audit.md",
    "reports\current_system_architecture.md",
    "reports\current_system_architecture_th.md",
    "reports\multi_person_readiness.md",
    "reports\multi_person_readiness_th.md",
    "reports\privacy_design_th.md",
    "reports\git_bootstrap_th.md",
    "reports\project_status_for_instructor_th.md",
    "reports\room_measurement_checklist_th.md",
    "reports\room_camera_planner.md",
    "reports\overnight_system_prep_progress.md",
    "reports\low_spec_performance_plan_th.md",
    "reports\overnight_system_prep_integrity.md",
    "reports\desktop_multiperson_integration_plan_th.md",
    "reports\acer_desktop_integration_audit_th.md",
    "reports\acer_desktop_integration_status_th.md",
    "reports\acer_to_desktop_merge_plan_th.md",
    "reports\multiperson_mediapipe_audit_th.md",
    "reports\multiperson_prototype_v1_th.md",
    "reports\multiperson_real_test_analysis_th.md",
    "reports\seat_aware_multiperson_v2_th.md",
    "reports\hikvision_network_deployment_th.md",
    "reports\hikvision_camera_test_checklist_th.md",
    "reports\rtsp_bandwidth_estimation_th.md",
    "reports\rtsp_preparation_progress.md",
    "reports\rtsp_ai_server_preparation_th.md",
    "reports\final_project_status.md",
    "reports\final_project_status_th.md",
    "outputs\demo\test_fixture_dashboard.html"
)

try {
    New-Item -ItemType Directory -Path $stage -Force | Out-Null
    foreach ($relative in $files) {
        $source = Join-Path $ProjectRoot $relative
        if (-not (Test-Path -LiteralPath $source)) {
            throw "Bundle source is missing: $relative"
        }
        $destination = Join-Path $stage $relative
        if ((Get-Item -LiteralPath $source).PSIsContainer) {
            New-Item -ItemType Directory -Path $destination -Force | Out-Null
            Get-ChildItem -LiteralPath $source -Force | Copy-Item -Destination $destination -Recurse -Force
        } else {
            $destinationParent = Split-Path -Parent $destination
            New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
            Copy-Item -LiteralPath $source -Destination $destination -Force
        }
    }
    Get-ChildItem -LiteralPath $stage -Directory -Recurse -Force |
        Where-Object { $_.Name -in @("__pycache__", ".pytest_cache") -or $_.Name -like ".pytest-runtime*" } |
        Sort-Object FullName -Descending |
        Remove-Item -Recurse -Force
    $outputParent = Split-Path -Parent $resolvedOutput
    New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
    if (Test-Path -LiteralPath $resolvedOutput) {
        throw "Bundle output already exists; choose a new versioned -OutputPath instead of overwriting it: $resolvedOutput"
    }
    Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $resolvedOutput -CompressionLevel Optimal
    $item = Get-Item -LiteralPath $resolvedOutput
    Write-Host ("Created code-only Acer bundle: {0} ({1:N2} MiB)" -f $item.FullName, ($item.Length / 1MB))
    Write-Host "Excluded by construction: datasets, validation videos, live logs, generated recordings, .venv, and training artifacts."
} finally {
    if (Test-Path -LiteralPath $stage) {
        Remove-Item -LiteralPath $stage -Recurse -Force
    }
}

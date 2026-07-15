[CmdletBinding()]
param(
    [switch]$Gpu,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$AppRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runtime = Join-Path $AppRoot ".runtime"
$Python = Join-Path $Runtime "Scripts\python.exe"

Set-Location $AppRoot
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.11+ is required. Install Python, then run this script again."
}

if (-not (Test-Path $Python)) {
    Write-Host "Creating private Python runtime..." -ForegroundColor Cyan
    python -m venv $Runtime
    if ($LASTEXITCODE -ne 0) { throw "Cannot create the private runtime." }
}

Write-Host "Installing application dependencies into $Runtime..." -ForegroundColor Cyan
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

# setup.ps1 clones DiariZen/3D-Speaker and installs their editable packages.
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $AppRoot "scripts\setup.ps1") -SkipModels -PythonExe $Python
if ($LASTEXITCODE -ne 0) { throw "DiariZen/CAM++ setup failed." }

if ($Gpu) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $AppRoot "scripts\install_gpu.ps1") -SkipDependencies -PythonExe $Python
    if ($LASTEXITCODE -ne 0) { throw "GPU runtime setup failed." }
}

if (-not $SkipModels) {
    & $Python scripts\download_models.py
    if ($LASTEXITCODE -ne 0) { throw "Model download failed." }
}

Write-Host "`nTest runtime is ready." -ForegroundColor Green
Write-Host "Run the app with: powershell -ExecutionPolicy Bypass -File scripts\run_test.ps1"

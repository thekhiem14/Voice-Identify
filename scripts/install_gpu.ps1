param(
    [switch]$SkipDependencies,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $PSScriptRoot
Set-Location $AppRoot

Write-Host "IMPORTANT: Close every running 'python app.py' window before continuing."
Write-Host "Checking PyTorch CUDA 12.8..."

$Py = $PythonExe
& $Py -m pip install --upgrade pip
if (-not $SkipDependencies) {
    & $Py -m pip install -r requirements.txt
}

& $Py -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() and str(torch.__version__).endswith('+cu128') else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing PyTorch CUDA 12.8 for NVIDIA GPU..."
    & $Py -m pip install --force-reinstall `
        torch==2.11.0+cu128 `
        torchaudio==2.11.0+cu128 `
        --index-url https://download.pytorch.org/whl/cu128
}

& $Py -c "import torch; assert torch.cuda.is_available(), f'CUDA unavailable: torch={torch.__version__}, build={torch.version.cuda}'; p=torch.cuda.get_device_properties(0); print(f'CUDA OK: {torch.cuda.get_device_name(0)} | VRAM={p.total_memory/1024**3:.1f} GB | torch={torch.__version__} | runtime={torch.version.cuda}')"
& $Py -c "import noisereduce, nara_wpe, sherpa_onnx; print('Enhancement + Gipformer Python dependencies: OK')"

$SherpaVersion = "1.13.3"
$SherpaArchive = Join-Path $AppRoot "data\cache\sherpa-onnx-v$SherpaVersion-win-x64-cuda.tar.bz2"
$SherpaInstall = Join-Path $AppRoot "models\sherpa-onnx-cuda"
$SherpaFolder = Join-Path $SherpaInstall "sherpa-onnx-v$SherpaVersion-cuda-12.x-cudnn-9.x-win-x64-cuda"
$SherpaUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/v$SherpaVersion/sherpa-onnx-v$SherpaVersion-cuda-12.x-cudnn-9.x-win-x64-cuda.tar.bz2"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $SherpaArchive) | Out-Null
New-Item -ItemType Directory -Force -Path $SherpaInstall | Out-Null
if (-not (Test-Path $SherpaArchive)) {
    Write-Host "Downloading official sherpa-onnx CUDA bundle..."
    & curl.exe --ssl-no-revoke -L --retry 3 -o $SherpaArchive $SherpaUrl
    if ($LASTEXITCODE -ne 0) { throw "Cannot download sherpa-onnx CUDA bundle." }
}
if (-not (Test-Path (Join-Path $SherpaFolder "lib\onnxruntime_providers_cuda.dll"))) {
    Write-Host "Extracting sherpa-onnx CUDA bundle..."
    tar -xf $SherpaArchive -C $SherpaInstall
    if ($LASTEXITCODE -ne 0) { throw "Cannot extract sherpa-onnx CUDA bundle." }
}

$PythonSherpaLib = & $Py -c "from pathlib import Path; import sherpa_onnx; print(Path(sherpa_onnx.__file__).resolve().parent / 'lib')"
if (-not (Test-Path $PythonSherpaLib)) { throw "Cannot locate sherpa_onnx Python lib directory." }
Write-Host "Installing sherpa-onnx CUDA DLLs into Python package..."
Copy-Item -Force (Join-Path $SherpaFolder "lib\*.dll") $PythonSherpaLib

& $Py -c "from pathlib import Path; import sherpa_onnx; p=Path(sherpa_onnx.__file__).resolve().parent/'lib'/'onnxruntime_providers_cuda.dll'; assert p.exists(), f'Missing {p}'; print(f'sherpa-onnx CUDA provider: {p}')"

Write-Host "GPU setup complete. Start the app with: python app.py"

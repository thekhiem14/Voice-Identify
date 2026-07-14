param(
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $PSScriptRoot
$Models = Join-Path $AppRoot "models"
$DiariZen = Join-Path $Models "diarizen\DiariZen"
$SpeakerLab = Join-Path $Models "3D-Speaker"

Set-Location $AppRoot
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -c "import torch; assert torch.cuda.is_available(), f'PyTorch CUDA is not available: torch={torch.__version__}, build={torch.version.cuda}'; print(f'CUDA OK: {torch.cuda.get_device_name(0)} | torch={torch.__version__} | CUDA runtime={torch.version.cuda}')"
python -c "import noisereduce, nara_wpe, sherpa_onnx; print('Enhancement and Gipformer dependencies: OK')"

if (-not (Test-Path (Join-Path $DiariZen ".git"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $DiariZen) | Out-Null
    git clone --recursive https://github.com/BUTSpeechFIT/DiariZen.git $DiariZen
}
python -m pip install -e $DiariZen
python -m pip install -e (Join-Path $DiariZen "pyannote-audio")

if (-not (Test-Path (Join-Path $SpeakerLab ".git"))) {
    git clone https://github.com/modelscope/3D-Speaker.git $SpeakerLab
}

if (-not $SkipModels) {
    python scripts/download_models.py
}

# Install/verify the official sherpa-onnx CUDA provider after the Python wheel.
& (Join-Path $PSScriptRoot "install_gpu.ps1") -SkipDependencies

Write-Host "Setup complete. Start the app with: python app.py"

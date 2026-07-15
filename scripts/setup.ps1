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

python -c "import torch; x=torch.ones(2); print(f'PyTorch CPU OK: torch={torch.__version__}, sum={x.sum().item()}')"
python -c "import noisereduce, nara_wpe, sherpa_onnx; print('CPU enhancement and Gipformer dependencies: OK')"

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

Write-Host "Setup complete. Start the app with: python app.py"
Write-Host "Optional GPU mode: run scripts/install_gpu.ps1, then enable GPU in the UI."

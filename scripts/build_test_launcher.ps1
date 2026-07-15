$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Destination = if($env:VOICE_TEST_DIR){$env:VOICE_TEST_DIR}else{"D:\VoiceIdentityStudio-Test"}
python -m pip install "pyinstaller>=6.11,<7"
Push-Location $Root
try {
    python -m PyInstaller --noconfirm --clean --onefile --windowed --name VoiceIdentityStudio-Setup --distpath $Destination bootstrap_launcher.py
    if($LASTEXITCODE -ne 0){throw "Launcher build failed."}
    Write-Host "Created: $Destination\VoiceIdentityStudio-Setup.exe" -ForegroundColor Green
} finally { Pop-Location }

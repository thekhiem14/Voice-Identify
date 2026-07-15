[CmdletBinding()]
param(
    [string]$OutputDir = "D:\VoiceIdentityStudio",
    [switch]$SkipModelCopy,
    [switch]$DebugConsole
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StageDir = Join-Path $ProjectRoot "build\package"
$BundleName = "VoiceIdentityStudio"

function Invoke-RoboCopy([string]$Source, [string]$Destination) {
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    & robocopy $Source $Destination /E /XD .git __pycache__ /XF *.pyc
    if ($LASTEXITCODE -gt 7) {
        throw "Copy failed from $Source (robocopy exit code $LASTEXITCODE)."
    }
}

Push-Location $ProjectRoot
try {
    python -m pip install "pyinstaller>=6.11,<7"
    if ($LASTEXITCODE -ne 0) { throw "Unable to install PyInstaller." }

    Remove-Item -Recurse -Force $StageDir -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $StageDir | Out-Null

    $packArgs = @(
        "pack", "app.py", "-D", "-y", "-n", $BundleName,
        "--distpath", $StageDir,
        "--product-name", "Voice Identity Studio",
        "--file-description", "Vietnamese meeting transcription and voice identification",
        "--hidden-import", "torch",
        "--hidden-import", "torchaudio",
        "--hidden-import", "sherpa_onnx",
        "--hidden-import", "soundfile",
        "--hidden-import", "numpy"
    )
    if ($DebugConsole) { $packArgs += @("--debug-console", "console") }

    # The hidden imports above trigger PyInstaller's hooks, which collect the
    # native extensions for these lazily imported packages.
    & flet @packArgs
    if ($LASTEXITCODE -ne 0) { throw "The EXE build failed." }

    $BuiltBundle = Join-Path $StageDir $BundleName
    if (-not (Test-Path (Join-Path $BuiltBundle "$BundleName.exe"))) {
        throw "Build finished but $BundleName.exe was not found."
    }
    Invoke-RoboCopy $BuiltBundle $OutputDir

    if (-not $SkipModelCopy) {
        foreach ($model in "cam_plus", "diarizen", "gipformer") {
            $source = Join-Path $ProjectRoot "models\$model"
            if (-not (Test-Path $source)) { throw "Required model folder missing: $source" }
            Invoke-RoboCopy $source (Join-Path $OutputDir "models\$model")
        }
    }

    Write-Host "`nPortable bundle is ready: $(Join-Path $OutputDir "$BundleName.exe")" -ForegroundColor Green
    Write-Host "Models and writable data stay in: $OutputDir\models and $OutputDir\data"
}
finally {
    Pop-Location
}

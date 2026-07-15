[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"
$AppRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $AppRoot ".runtime\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Private runtime not found. Run scripts\bootstrap_test.ps1 first."
}
Set-Location $AppRoot
& $Python app.py @Arguments
exit $LASTEXITCODE

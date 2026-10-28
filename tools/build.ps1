<#
.SYNOPSIS
    Build the double-clickable ShotFinder release folder and zip.

.DESCRIPTION
    Cleans the previous build, renders the icon, runs PyInstaller with the
    committed spec, audits the DLL imports, then zips the one-folder build to
    dist\ShotFinder-0.1.1-win64.zip. Models are not included: the app downloads
    them on first run.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\build.ps1
#>

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$version = "0.1.1"
$python = Join-Path $root ".venv\Scripts\python.exe"
$zip = Join-Path $root "dist\ShotFinder-$version-win64.zip"
$folder = Join-Path $root "dist\ShotFinder"

Set-Location $root

if (-not (Test-Path $python)) {
    throw "Python venv not found at $python. Run: py -3.11 -m venv .venv; .\.venv\Scripts\pip install -e .[build]"
}

# Windows PowerShell turns any native stderr line into a terminating error
# when $ErrorActionPreference is Stop, and PyInstaller logs to stderr. Run the
# tool with Continue and fail on the exit code instead.
function Invoke-Tool {
    param([string[]]$Arguments)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $python @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    return $code
}

if ((Invoke-Tool @("-c", "import PyInstaller")) -ne 0) {
    throw "PyInstaller is missing. Run: .\.venv\Scripts\pip install -e .[build]"
}

Write-Host "== Rendering icon" -ForegroundColor Cyan
if ((Invoke-Tool @((Join-Path $root "tools\make_icon.py"))) -ne 0) {
    throw "Icon rendering failed"
}

Write-Host "== Cleaning dist and build" -ForegroundColor Cyan
$running = Get-Process ShotFinder -ErrorAction SilentlyContinue
if ($running) {
    throw "ShotFinder is running (pid $($running.Id -join ', ')). Quit it from the tray, then build again."
}
Remove-Item -Recurse -Force (Join-Path $root "dist"), (Join-Path $root "build") -ErrorAction SilentlyContinue

Write-Host "== Building with PyInstaller" -ForegroundColor Cyan
if ((Invoke-Tool @("-m", "PyInstaller", "--noconfirm", "--clean", (Join-Path $root "shotfinder.spec"))) -ne 0) {
    throw "PyInstaller failed"
}

if (-not (Test-Path (Join-Path $folder "ShotFinder.exe"))) {
    throw "Build finished without dist\ShotFinder\ShotFinder.exe"
}

Write-Host "== Checking runtime dependencies" -ForegroundColor Cyan
$depsJson = Join-Path $root "dist\runtime-deps.json"
if ((Invoke-Tool @((Join-Path $root "tools\check_runtime_deps.py"), $folder, "--json", $depsJson)) -ne 0) {
    throw "The bundle imports DLLs that are neither bundled nor part of Windows (see the report above)."
}

Write-Host "== Zipping" -ForegroundColor Cyan
New-Item -ItemType Directory -Force (Split-Path -Parent $zip) | Out-Null
Compress-Archive -Path $folder -DestinationPath $zip -CompressionLevel Optimal
if (-not (Test-Path $zip)) { throw "Compress-Archive did not create $zip" }

$zipMb = (Get-Item $zip).Length / 1MB
$folderMb = (Get-ChildItem $folder -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
$sha = (Get-FileHash $zip -Algorithm SHA256).Hash

Write-Host ""
Write-Host "Built ShotFinder $version" -ForegroundColor Green
Write-Host ("  zip      {0}  ({1:N1} MB)" -f $zip, $zipMb)
Write-Host ("  unpacked {0}  ({1:N1} MB)" -f $folder, $folderMb)
Write-Host ("  sha256   {0}" -f $sha)
Write-Host ("  deps     {0}" -f $depsJson)

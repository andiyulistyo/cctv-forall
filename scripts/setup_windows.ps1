<#
.SYNOPSIS
  One-time setup for running the dashboard natively on Windows (AMD Ryzen or
  Intel Core), accelerated with OpenVINO.

.DESCRIPTION
  Docker Desktop on Windows cannot pass an integrated GPU into a container, so
  a native install is the only way to get hardware acceleration on these
  machines. The script picks a profile from the CPU vendor:

    AMD Ryzen  -> OpenVINO CPU plugin (INT8, uses AVX-512/VNNI on Zen 4)
    Intel Core -> OpenVINO GPU plugin (FP16 on the integrated HD/Iris graphics)

.EXAMPLE
  .\scripts\setup_windows.ps1
  .\scripts\setup_windows.ps1 -Hardware intel -YoloModel yolo11n.pt
#>
[CmdletBinding()]
param(
    # amd | intel | auto
    [ValidateSet('auto', 'amd', 'intel')]
    [string]$Hardware = 'auto',
    # Base weights to download and export. Default depends on the profile.
    [string]$YoloModel = '',
    [int]$ImgSz = 0,
    # Skip the (slow) frontend build.
    [switch]$SkipFrontend
)

$ErrorActionPreference = 'Stop'

$Root    = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root 'backend'
$Venv    = Join-Path $Backend '.venv'
$Py      = Join-Path $Venv 'Scripts\python.exe'

# $ErrorActionPreference only governs PowerShell cmdlets -- a native program
# that exits non-zero does NOT stop the script. Without this every pip/python
# failure below would scroll past and the script would still say "complete",
# leaving a half-built venv that only breaks later at runtime.
function Invoke-Native {
    param([Parameter(Mandatory)][string]$What,
          [Parameter(Mandatory)][string]$Exe,
          [Parameter(ValueFromRemainingArguments)][string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed (exit $LASTEXITCODE): $Exe $($Arguments -join ' ')"
    }
}

Write-Host '==> Checking the machine' -ForegroundColor Cyan
$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1)
Write-Host "    $($cpu.Name)"
Write-Host "    $($cpu.NumberOfCores) physical cores / $($cpu.NumberOfLogicalProcessors) logical"

if ($Hardware -eq 'auto') {
    if ($cpu.Name -match 'AMD|Ryzen') { $Hardware = 'amd' }
    elseif ($cpu.Name -match 'Intel')  { $Hardware = 'intel' }
    else {
        Write-Warning "Unrecognised CPU; defaulting to the AMD (CPU plugin) profile."
        $Hardware = 'amd'
    }
}
Write-Host "    profile: $Hardware" -ForegroundColor Yellow

# Profile defaults. The Intel box in mind here is a 2-core Kaby Lake i7, which
# needs the small model at a reduced input size; the Ryzen has cores to spare.
if ($Hardware -eq 'intel') {
    if (-not $YoloModel) { $YoloModel = 'yolo11n.pt' }
    if ($ImgSz -le 0)    { $ImgSz = 480 }
    $ExportArgs = @('--half')          # FP16 is the iGPU's native precision
    $EnvExample = '.env.intel.example'
    $WarmEasyOcr = $false              # ANPR is off in that profile
} else {
    if (-not $YoloModel) { $YoloModel = 'yolo11s.pt' }
    if ($ImgSz -le 0)    { $ImgSz = 640 }
    $ExportArgs = @('--int8')          # VNNI integer units on Zen 4
    $EnvExample = '.env.amd.example'
    $WarmEasyOcr = $true
}

Write-Host '==> Checking prerequisites' -ForegroundColor Cyan
# The pinned wheels (numpy, opencv, easyocr) and openvino only publish for
# 3.11-3.13. Picking whatever "python" happens to be on PATH is how you end up
# watching pip try to compile numpy from source.
$SupportedPython = @('3.12', '3.11', '3.13')
$python = $null
$launcher = Get-Command py -ErrorAction SilentlyContinue
# "py -0p" lists the installed versions on stdout, so we can pick one without
# probing each in turn. That matters: probing meant redirecting a native
# command's stderr, and in Windows PowerShell 5.1 that wraps every stderr line
# in an ErrorRecord which $ErrorActionPreference='Stop' then turns into a
# terminating error -- so a missing 3.12 would kill the script instead of
# falling through to 3.11.
$installed = @()
if ($launcher) {
    $listing = & $launcher.Source -0p
    if ($LASTEXITCODE -eq 0) {
        $installed = @($listing | ForEach-Object {
            if ($_ -match '-V:(\d+\.\d+)') { $Matches[1] }
        })
    }
}
foreach ($v in $SupportedPython) {
    if ($installed -contains $v) { $python = @($launcher.Source, "-$v"); break }
    $cmd = Get-Command "python$v" -ErrorAction SilentlyContinue
    if ($cmd) { $python = @($cmd.Source); break }
}
if (-not $python) {
    # Fall back to plain "python", but only if its version is one we support.
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        $ver = & $cmd.Source -c "import sys; print('%d.%d' % sys.version_info[:2])"
        if ($SupportedPython -contains $ver) { $python = @($cmd.Source) }
        else {
            throw ("Found Python $ver on PATH, but the pinned dependencies only " +
                   "have wheels for $($SupportedPython -join ', '). Install one " +
                   "(winget install Python.Python.3.12) and re-run.")
        }
    }
}
if (-not $python) {
    throw "No supported Python found. Install 3.12 (winget install Python.Python.3.12) and re-run."
}
Write-Host "    using $($python -join ' ') ($(& $python[0] @($python[1..($python.Length-1)]) -V))"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning 'ffmpeg not on PATH. OpenCV ships its own for decoding, but the CLI is handy for testing (winget install Gyan.FFmpeg).'
}

Write-Host '==> Creating the virtualenv at backend\.venv' -ForegroundColor Cyan
if (-not (Test-Path $Py)) {
    Invoke-Native 'venv creation' $python[0] @($python[1..($python.Length - 1)]) -m venv $Venv
}
Invoke-Native 'pip self-upgrade' $Py -m pip install --upgrade pip wheel

Write-Host '==> Installing PyTorch (CPU build)' -ForegroundColor Cyan
# Inference runs through OpenVINO; torch is still needed by ultralytics for
# pre/post-processing and by EasyOCR, so the small CPU wheel is enough.
Invoke-Native 'torch install' $Py -m pip install torch torchvision `
    --index-url https://download.pytorch.org/whl/cpu

Write-Host '==> Installing the backend requirements' -ForegroundColor Cyan
Invoke-Native 'requirements.txt install' $Py -m pip install -r (Join-Path $Backend 'requirements.txt')

Write-Host '==> Installing the OpenVINO runtime' -ForegroundColor Cyan
Invoke-Native 'requirements-openvino.txt install' $Py -m pip install -r (Join-Path $Backend 'requirements-openvino.txt')

Write-Host '==> Checking the OpenVINO runtime loads' -ForegroundColor Cyan
# The probe reports its own failure on stdout rather than letting Python write a
# traceback to stderr. Capturing native stderr in PowerShell 5.1 (2>&1 or 2>file)
# wraps each line in an ErrorRecord, which under $ErrorActionPreference='Stop'
# aborts the script with "NativeCommandError" and hides the message we need.
# Single-quoted on the Python side on purpose: PowerShell drops embedded double
# quotes when it hands an argument to a native command.
$probe = @'
import sys
try:
    import openvino as ov
except Exception as exc:
    print('FAILED|%s: %s' % (type(exc).__name__, exc))
    sys.exit(2)
print('OK|' + ','.join(ov.Core().available_devices))
'@
$result = & $Py -c $probe
if ($LASTEXITCODE -ne 0 -or "$result" -notmatch '^OK\|') {
    Write-Host "    $result" -ForegroundColor Red
    if ("$result" -match 'DLL load failed') {
        # openvino's native extension links against the MSVC runtime, which is
        # not part of a stock Windows install and not shipped in the wheel.
        throw ("OpenVINO installed but its native module will not load. This is " +
               "almost always the missing Microsoft Visual C++ Redistributable: " +
               "  winget install --id Microsoft.VCRedist.2015+.x64 -e" + [Environment]::NewLine +
               "Install it, reboot, and re-run this script.")
    }
    throw "OpenVINO installed but 'import openvino' failed -- see the message above."
}
$devices = ("$result" -split '\|', 2)[1]
Write-Host "    OpenVINO devices: $devices" -ForegroundColor Yellow
if ($Hardware -eq 'intel' -and $devices -notmatch 'GPU') {
    Write-Warning 'No OpenVINO GPU device found. Update the Intel Graphics driver, then re-run. Falling back to the CPU plugin for now.'
}

Write-Host '==> Fetching model weights into data\weights' -ForegroundColor Cyan
$Weights = Join-Path $Root 'data\weights'
New-Item -ItemType Directory -Force -Path (Join-Path $Weights 'face') | Out-Null
if (-not (Test-Path (Join-Path $Weights $YoloModel))) {
    Push-Location $Weights
    try {
        $env:YOLO_CONFIG_DIR = $Weights
        Invoke-Native "download of $YoloModel" $Py -c "from ultralytics import YOLO; YOLO('$YoloModel')"
    } finally { Pop-Location }
}
$faceModels = @{
    'yunet.onnx' = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
    'sface.onnx' = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx'
}
foreach ($name in $faceModels.Keys) {
    $dest = Join-Path $Weights "face\$name"
    if (-not (Test-Path $dest)) { Invoke-WebRequest -Uri $faceModels[$name] -OutFile $dest }
}

Write-Host "==> Exporting $YoloModel to OpenVINO IR (imgsz=$ImgSz $ExportArgs)" -ForegroundColor Cyan
Push-Location $Backend
try {
    Invoke-Native 'OpenVINO export' $Py (Join-Path $Root 'scripts\export_openvino.py') `
        --model (Join-Path $Weights $YoloModel) --imgsz $ImgSz @ExportArgs
} finally { Pop-Location }

if ($WarmEasyOcr) {
    Write-Host '==> Warming up EasyOCR (downloads its models once)' -ForegroundColor Cyan
    Invoke-Native 'EasyOCR warmup' $Py -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"
}

if (-not $SkipFrontend) {
    Write-Host '==> Building the frontend' -ForegroundColor Cyan
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Push-Location (Join-Path $Root 'frontend')
        try {
            Invoke-Native 'npm install' 'npm.cmd' install
            Invoke-Native 'npm run build' 'npm.cmd' run build
        } finally { Pop-Location }
    } else {
        Write-Warning 'npm not found - skipping. Install Node, then: cd frontend; npm install; npm run build'
    }
}

Write-Host '==> Creating .env' -ForegroundColor Cyan
$envPath = Join-Path $Root '.env'
if (-not (Test-Path $envPath)) {
    Copy-Item (Join-Path $Root $EnvExample) $envPath
    Write-Host "    wrote .env from $EnvExample - change ADMIN_PASSWORD and JWT_SECRET."
} else {
    Write-Host "    .env already exists, leaving it alone."
    Write-Host "    compare it against $EnvExample for the $Hardware settings."
}

Write-Host @"

Setup complete.

  Start the app:      .\scripts\run_windows.ps1
  Confirm the device: curl http://localhost:8000/api/health
  Measure throughput: backend\.venv\Scripts\python scripts\benchmark.py ``
                        --model data\weights\$([IO.Path]::GetFileNameWithoutExtension($YoloModel))_openvino_model ``
                        --imgsz $ImgSz

"@ -ForegroundColor Green

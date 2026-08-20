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
$python = $null
foreach ($candidate in @('python3.12', 'python3.11', 'python')) {
    $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($cmd) { $python = $cmd.Source; break }
}
if (-not $python) { throw 'No python found. Install Python 3.11 or 3.12 from python.org.' }
Write-Host "    using $python ($(& $python -V))"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning 'ffmpeg not on PATH. OpenCV ships its own for decoding, but the CLI is handy for testing (winget install Gyan.FFmpeg).'
}

Write-Host '==> Creating the virtualenv at backend\.venv' -ForegroundColor Cyan
if (-not (Test-Path $Py)) { & $python -m venv $Venv }
& $Py -m pip install --upgrade pip wheel | Out-Null

Write-Host '==> Installing PyTorch (CPU build)' -ForegroundColor Cyan
# Inference runs through OpenVINO; torch is still needed by ultralytics for
# pre/post-processing and by EasyOCR, so the small CPU wheel is enough.
& $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Write-Host '==> Installing the backend requirements' -ForegroundColor Cyan
& $Py -m pip install -r (Join-Path $Backend 'requirements.txt')

Write-Host '==> Installing the OpenVINO runtime' -ForegroundColor Cyan
& $Py -m pip install -r (Join-Path $Backend 'requirements-openvino.txt')

$devices = & $Py -c "import openvino as ov; print(','.join(ov.Core().available_devices))"
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
        & $Py -c "from ultralytics import YOLO; YOLO('$YoloModel')"
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
    & $Py (Join-Path $Root 'scripts\export_openvino.py') `
        --model (Join-Path $Weights $YoloModel) --imgsz $ImgSz @ExportArgs
} finally { Pop-Location }

if ($WarmEasyOcr) {
    Write-Host '==> Warming up EasyOCR (downloads its models once)' -ForegroundColor Cyan
    & $Py -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)" | Out-Null
}

if (-not $SkipFrontend) {
    Write-Host '==> Building the frontend' -ForegroundColor Cyan
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Push-Location (Join-Path $Root 'frontend')
        try { npm install; npm run build } finally { Pop-Location }
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

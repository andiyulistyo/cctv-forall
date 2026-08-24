<#
.SYNOPSIS
  One-time setup for running the dashboard natively on Windows, on an NVIDIA
  GPU (CUDA) or on the CPU/iGPU via OpenVINO.

.DESCRIPTION
  Docker Desktop on Windows cannot pass an integrated GPU into a container, so
  a native install is the only way to get hardware acceleration on these
  machines. The script picks a profile from the hardware:

    NVIDIA GPU -> CUDA (torch fp16, yolo12m, EasyOCR and the plate detector
                  on the GPU as well)
    AMD Ryzen  -> OpenVINO CPU plugin (INT8, uses AVX-512/VNNI on Zen 4)
    Intel Core -> OpenVINO GPU plugin (FP16 on the integrated HD/Iris graphics)

  The GPU is checked before the CPU vendor: an NVIDIA laptop almost always has
  an AMD or Intel CPU too, and choosing on the CPU there would configure the
  OpenVINO CPU plugin and leave the discrete card idle.

.EXAMPLE
  .\scripts\setup_windows.ps1
  .\scripts\setup_windows.ps1 -Hardware nvidia
  .\scripts\setup_windows.ps1 -Hardware intel -YoloModel yolo11n.pt
#>
[CmdletBinding()]
param(
    # nvidia | amd | intel | auto
    [ValidateSet('auto', 'nvidia', 'amd', 'intel')]
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

# A discrete NVIDIA card beats anything the CPU side can offer, so look there
# first. nvidia-smi ships with the driver; if it runs and lists a GPU, we have
# one. (Get-CimInstance Win32_VideoController would also see it, but it reports
# the card even when no usable driver is installed.)
$nvidiaGpu = $null
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $names = & $smi.Source --query-gpu=name --format=csv,noheader
    if ($LASTEXITCODE -eq 0 -and $names) { $nvidiaGpu = @($names)[0].Trim() }
}
if ($nvidiaGpu) { Write-Host "    $nvidiaGpu" }

if ($Hardware -eq 'auto') {
    if     ($nvidiaGpu)                    { $Hardware = 'nvidia' }
    elseif ($cpu.Name -match 'AMD|Ryzen')  { $Hardware = 'amd' }
    elseif ($cpu.Name -match 'Intel')      { $Hardware = 'intel' }
    else {
        Write-Warning "Unrecognised CPU; defaulting to the AMD (CPU plugin) profile."
        $Hardware = 'amd'
    }
}
if ($Hardware -eq 'nvidia' -and -not $nvidiaGpu) {
    throw ("-Hardware nvidia was requested but nvidia-smi found no GPU. Install " +
           "the NVIDIA driver, or re-run without -Hardware to auto-detect.")
}
Write-Host "    profile: $Hardware" -ForegroundColor Yellow

# Profile defaults. The Intel box in mind here is a 2-core Kaby Lake i7, which
# needs the small model at a reduced input size; the Ryzen has cores to spare.
# Only the OpenVINO profiles export an IR; $UseOpenVino gates every step that
# is specific to them.
$UseOpenVino = $Hardware -ne 'nvidia'

if ($Hardware -eq 'nvidia') {
    # The GPU has headroom to spare, and at imgsz 640 the medium model costs
    # essentially nothing over the small one -- the bottleneck at that point is
    # the CPU-side letterbox/NMS, not the matmuls.
    #
    # v12 only here. Its area-attention blocks trade throughput for accuracy
    # (measured on an RTX 5070 Laptop, fp16, imgsz 640: yolo12m 44.6 fps vs
    # yolo11m 51.9), which a discrete GPU can absorb and the CPU/OpenVINO
    # profiles below cannot -- so those stay on v11.
    if (-not $YoloModel) { $YoloModel = 'yolo12m.pt' }
    if ($ImgSz -le 0)    { $ImgSz = 640 }
    $ExportArgs = @()
    $EnvExample = '.env.nvidia.example'
    $WarmEasyOcr = $true
} elseif ($Hardware -eq 'intel') {
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

if ($Hardware -eq 'nvidia') {
    Write-Host '==> Installing PyTorch (CUDA 13 build)' -ForegroundColor Cyan
    # cu130, not the default wheel: RTX 50-series is sm_120 (Blackwell) and the
    # cu128-and-earlier builds have no kernels for it -- they install happily and
    # then fail at the first launch. CUDA is backward compatible with older
    # cards, so this is also the right wheel for a 40- or 30-series.
    Invoke-Native 'torch install' $Py -m pip install --force-reinstall torch torchvision `
        --index-url https://download.pytorch.org/whl/cu130
} else {
    Write-Host '==> Installing PyTorch (CPU build)' -ForegroundColor Cyan
    # Inference runs through OpenVINO; torch is still needed by ultralytics for
    # pre/post-processing and by EasyOCR, so the small CPU wheel is enough.
    Invoke-Native 'torch install' $Py -m pip install torch torchvision `
        --index-url https://download.pytorch.org/whl/cpu
}

Write-Host '==> Installing the backend requirements' -ForegroundColor Cyan
Invoke-Native 'requirements.txt install' $Py -m pip install -r (Join-Path $Backend 'requirements.txt')

if ($Hardware -eq 'nvidia') {
    Write-Host '==> Checking torch can reach the GPU' -ForegroundColor Cyan
    # Same shape as the OpenVINO probe below, and for the same reason: report
    # failure on stdout instead of letting Python write a traceback to stderr.
    $cudaProbe = @'
import sys
try:
    import torch
except Exception as exc:
    print('FAILED|%s: %s' % (type(exc).__name__, exc))
    sys.exit(2)
if not torch.cuda.is_available():
    print('FAILED|torch %s reports no CUDA device' % torch.__version__)
    sys.exit(2)
p = torch.cuda.get_device_properties(0)
print('OK|%s|%d MiB|sm_%d%d|torch %s' % (p.name, p.total_memory // (1024*1024),
                                         p.major, p.minor, torch.__version__))
'@
    $cudaResult = & $Py -c $cudaProbe
    if ($LASTEXITCODE -ne 0 -or "$cudaResult" -notmatch '^OK\|') {
        Write-Host "    $cudaResult" -ForegroundColor Red
        throw ("PyTorch is installed but cannot use the GPU. Check that the " +
               "NVIDIA driver is current (nvidia-smi works), then re-run. " +
               "A '+cpu' torch version in the message above means the CPU wheel " +
               "is still installed and the cu130 install did not take effect.")
    }
    Write-Host "    $(("$cudaResult" -split '\|', 2)[1])" -ForegroundColor Yellow

    Write-Host '==> Installing the GPU extras (ONNX Runtime for face recognition)' -ForegroundColor Cyan
    # YuNet/SFace are ONNX models and cv2.dnn has no CUDA backend, so without
    # this face recognition is the slowest step in the pipeline by a wide
    # margin. No separate CUDA Toolkit needed: onnxruntime-gpu links against
    # CUDA 13 + cuDNN 9, which the cu130 torch wheel already ships.
    Invoke-Native 'requirements-cuda.txt install' $Py -m pip install -r (Join-Path $Backend 'requirements-cuda.txt')
} else {

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

# Dedicated plate detector for ANPR. Without it alpr.py falls back to a
# heuristic ROI (the bottom 45% of the vehicle box), which is the single
# biggest source of misreads. A YOLO11 fine-tune, so ultralytics loads it
# as-is, and next to the vehicle model on a GPU it costs almost nothing.
if ($Hardware -eq 'nvidia') {
    $plateModel = 'license-plate-finetune-v1s.pt'
    $plateDest = Join-Path $Weights $plateModel
    if (-not (Test-Path $plateDest)) {
        Write-Host "    downloading $plateModel"
        Invoke-WebRequest -OutFile $plateDest -Uri `
            "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/$plateModel"
    }
}

if ($UseOpenVino) {
    Write-Host "==> Exporting $YoloModel to OpenVINO IR (imgsz=$ImgSz $ExportArgs)" -ForegroundColor Cyan
    Push-Location $Backend
    try {
        Invoke-Native 'OpenVINO export' $Py (Join-Path $Root 'scripts\export_openvino.py') `
            --model (Join-Path $Weights $YoloModel) --imgsz $ImgSz @ExportArgs
    } finally { Pop-Location }
} else {
    # The CUDA profile runs the .pt weights directly in fp16. A TensorRT engine
    # is faster still, but only by the share of the frame time that is actually
    # inference -- read scripts/export_tensorrt.py before spending time on it.
    Write-Host "==> Using $YoloModel directly (torch fp16 on the GPU)" -ForegroundColor Cyan
}

if ($WarmEasyOcr) {
    Write-Host '==> Warming up EasyOCR (downloads its models once)' -ForegroundColor Cyan
    # EasyOCR is a torch model, so on the CUDA profile it runs on the GPU too.
    # The download is the same either way, but warming it there also proves the
    # GPU path works before the first live stream depends on it.
    $ocrGpu = if ($Hardware -eq 'nvidia') { 'True' } else { 'False' }
    Invoke-Native 'EasyOCR warmup' $Py -c "import easyocr; easyocr.Reader(['en'], gpu=$ocrGpu, verbose=False)"
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

# The OpenVINO profiles benchmark the exported IR directory; the CUDA profile
# benchmarks the weights themselves.
$BenchModel = if ($UseOpenVino) {
    "$([IO.Path]::GetFileNameWithoutExtension($YoloModel))_openvino_model"
} else { $YoloModel }

Write-Host @"

Setup complete.

  Start the app:      .\scripts\run_windows.ps1
  Confirm the device: curl http://localhost:8000/api/health
  Measure throughput: backend\.venv\Scripts\python scripts\benchmark.py ``
                        --model data\weights\$BenchModel ``
                        --imgsz $ImgSz

"@ -ForegroundColor Green

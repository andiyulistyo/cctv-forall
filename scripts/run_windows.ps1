<#
.SYNOPSIS
  Start the dashboard natively on Windows (OpenVINO accelerated).

.EXAMPLE
  .\scripts\run_windows.ps1                 # http://localhost:8000
  $env:PORT = 9000; .\scripts\run_windows.ps1
  .\scripts\run_windows.ps1 --reload        # extra args go to uvicorn
#>
[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$UvicornArgs)

$ErrorActionPreference = 'Stop'

$Root    = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root 'backend'
$Uvicorn = Join-Path $Backend '.venv\Scripts\uvicorn.exe'

if (-not (Test-Path $Uvicorn)) {
    throw 'backend\.venv is missing. Run .\scripts\setup_windows.ps1 first.'
}

$httpHost = if ($env:HOST) { $env:HOST } else { '0.0.0.0' }
$port     = if ($env:PORT) { $env:PORT } else { '8000' }

# Settings come from .env at the repo root (see .env.amd.example /
# .env.intel.example). Only one uvicorn worker: the detection manager owns the
# child processes and the shared frame state, which must live in a single
# parent process.
Push-Location $Backend
try {
    & $Uvicorn app.main:app --host $httpHost --port $port --workers 1 @UvicornArgs
} finally {
    Pop-Location
}

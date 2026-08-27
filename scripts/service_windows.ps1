<#
.SYNOPSIS
  Supervisor the autostart Scheduled Task runs: starts uvicorn, writes its
  output to a log file, and restarts it if it dies.

.DESCRIPTION
  This is run_windows.ps1 hardened for unattended boots. It differs in three
  ways, all of which matter when nobody is at the keyboard:

    * output goes to data\logs\backend.log instead of a console that no longer
      exists,
    * it waits for a network profile to appear before the first start, because
      at boot the RTSP cameras are usually still unreachable when services
      start,
    * a crash is restarted here after a few seconds instead of waiting for the
      Task Scheduler's one-minute retry. Repeated crashes fall through to the
      Task Scheduler on purpose (see the restart budget below).

  Install it with .\scripts\install_autostart_windows.ps1; run it by hand only
  to reproduce what the task does.

.EXAMPLE
  .\scripts\service_windows.ps1
  .\scripts\service_windows.ps1 -Port 9000 -BindAddress 127.0.0.1
#>
[CmdletBinding()]
param(
    # $Host is a PowerShell automatic variable, hence the name.
    [string]$BindAddress = $(if ($env:HOST) { $env:HOST } else { '0.0.0.0' }),
    [int]$Port           = $(if ($env:PORT) { [int]$env:PORT } else { 8000 }),
    # Give up on the network after this long and start anyway: a machine with
    # only local video files must still come up with the cable unplugged.
    [int]$NetworkWaitSeconds = 60,
    # Rotate the log once it passes this size.
    [int]$MaxLogMB = 20
)

$ErrorActionPreference = 'Stop'

$Root    = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root 'backend'
$Uvicorn = Join-Path $Backend '.venv\Scripts\uvicorn.exe'
$LogDir  = Join-Path $Root 'data\logs'
$LogFile = Join-Path $LogDir 'backend.log'

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

# Everything written to the log goes through one long-lived StreamWriter, so
# that rotation can happen *mid-run*. Checking the size only between restarts
# would be no check at all on the installation this exists for: a box in a
# cabinet that comes up once and then runs for months, where a healthy uvicorn
# never restarts and its access log is the only thing that grows.
$script:LogWriter = $null
$script:LogBytes  = 0

function Open-Log {
    if ($script:LogWriter) { return }
    # Read the size before the writer creates the file, or an existing log
    # restarts its budget at zero on every supervisor start.
    $script:LogBytes = if (Test-Path $LogFile) { (Get-Item $LogFile).Length } else { 0 }
    $encoding = New-Object System.Text.UTF8Encoding $false
    $script:LogWriter = New-Object System.IO.StreamWriter -ArgumentList $LogFile, $true, $encoding
    # A crash must not cost the last few lines -- they are the ones that say why.
    $script:LogWriter.AutoFlush = $true
}

function Close-Log {
    if (-not $script:LogWriter) { return }
    $script:LogWriter.Dispose()
    $script:LogWriter = $null
}

# Keep one previous log around. Safe to call at any point: the handle is closed
# before the rename and reopened after, so nothing is ever written to a file
# that has just been moved out from under it.
function Invoke-LogRotation {
    if ($script:LogBytes -lt ($MaxLogMB * 1MB)) { return }
    Close-Log
    $previous = "$LogFile.1"
    if (Test-Path $previous) { Remove-Item $previous -Force }
    if (Test-Path $LogFile) { Move-Item $LogFile $previous -Force }
    Open-Log
    # Built first, then written: as arguments to WriteLine() the -f and its
    # operands parse as two method arguments, and the format never sees {1}.
    # Straight to the writer rather than through Write-Log, which would
    # re-enter the rotation check this line is announcing.
    $stamp  = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $marker = "$stamp [service] rotated; previous log is $(Split-Path -Leaf $previous)"
    $script:LogWriter.WriteLine($marker)
    $script:LogBytes += $marker.Length + 2
}

function Write-LogLine {
    param([string]$Text)
    Open-Log
    $script:LogWriter.WriteLine($Text)
    # +2 for the CRLF. Multi-byte UTF-8 makes this a slight undercount, which
    # only ever rotates a little late -- cheaper than stat-ing the file per line.
    $script:LogBytes += $Text.Length + 2
    Invoke-LogRotation
}

function Write-Log {
    param([Parameter(Mandatory)][string]$Message)
    Write-LogLine ('{0} [service] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message)
}

# At boot the NIC is often not up yet. Nothing here fails permanently without
# it -- the workers retry -- but waiting keeps the log free of a burst of
# unreachable-camera errors on every restart.
function Wait-ForNetwork {
    $deadline = (Get-Date).AddSeconds($NetworkWaitSeconds)
    while ((Get-Date) -lt $deadline) {
        $profiles = @(Get-NetConnectionProfile -ErrorAction SilentlyContinue)
        if ($profiles.Count -gt 0) {
            Write-Log "network up ($($profiles[0].Name))"
            return
        }
        Start-Sleep -Seconds 2
    }
    Write-Log "no network after ${NetworkWaitSeconds}s; starting anyway"
}

Write-Log "starting supervisor (root=$Root, bind=${BindAddress}:${Port})"

if (-not (Test-Path $Uvicorn)) {
    Write-Log "backend\.venv is missing -- run .\scripts\setup_windows.ps1 first"
    exit 1
}

Wait-ForNetwork

# Settings come from .env at the repo root. Only one uvicorn worker: the
# detection manager owns the child processes and the shared frame state, which
# must live in a single parent process.
Push-Location $Backend
try {
    # Restart budget: a crash loop is almost always a bad .env or a port
    # already in use, and hammering it every few seconds only fills the disk
    # with the same traceback. After enough failures in a short window we exit
    # non-zero and let the Task Scheduler's slower retry take over.
    $failures = New-Object System.Collections.Generic.Queue[datetime]
    $window   = [TimeSpan]::FromMinutes(5)
    $budget   = 5
    $backoff  = @(5, 10, 20, 30, 60)

    while ($true) {
        Write-Log "launching uvicorn on ${BindAddress}:${Port}"
        $started = Get-Date

        # uvicorn logs to stderr. Merging it into the pipeline needs
        # $ErrorActionPreference relaxed -- with 'Stop' a native command's
        # stderr becomes a terminating NativeCommandError on the first line
        # uvicorn prints. ToString() strips the ErrorRecord formatting the log
        # would otherwise carry on every one of those lines.
        $ErrorActionPreference = 'Continue'
        & $Uvicorn app.main:app --host $BindAddress --port $Port --workers 1 2>&1 |
            ForEach-Object { Write-LogLine $_.ToString() }
        $code = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'

        $ranFor = [int]((Get-Date) - $started).TotalSeconds
        Write-Log "uvicorn exited with $code after ${ranFor}s"

        $failures.Enqueue((Get-Date))
        while ($failures.Count -gt 0 -and ((Get-Date) - $failures.Peek()) -gt $window) {
            [void]$failures.Dequeue()
        }
        if ($failures.Count -ge $budget) {
            Write-Log "$budget restarts within $($window.TotalMinutes) min; giving up so the Task Scheduler can retry"
            exit 1
        }

        $delay = $backoff[[Math]::Min($failures.Count - 1, $backoff.Count - 1)]
        Write-Log "restarting in ${delay}s"
        Start-Sleep -Seconds $delay
    }
} finally {
    Pop-Location
    Close-Log
}

<#
.SYNOPSIS
  Restart the backend that the autostart Scheduled Task runs -- properly.

.DESCRIPTION
  Stop-ScheduledTask followed by Start-ScheduledTask looks like a restart and
  is not one. Stopping the task kills the supervisor PowerShell it launched,
  but uvicorn and the detection workers underneath it are *orphaned rather than
  killed*: they keep running, and they keep holding the port. The task then
  starts a fresh supervisor whose uvicorn cannot bind, exits 1, is restarted a
  few times, and gives up -- leaving the old build serving happily while every
  message says the restart succeeded.

  That is not hypothetical; it is what this script was written after. So:

    1. stop the task,
    2. kill whatever is still holding the port, walking up to the top of its
       process tree first (the listener is a grandchild of uvicorn.exe, and
       killing the listener alone leaves the rest of the tree behind),
    3. wait for the port to actually come free,
    4. re-enable the task if a previous crash-loop left it disabled,
    5. start it, and wait for the port to come back.

  Elevation is required and is checked up front, because the task is registered
  with RunLevel Highest: its processes are elevated, and a non-elevated shell
  can neither kill them nor touch the task. Failing at step 2 of 5 with half
  the backend dead is a worse outcome than refusing at step 0.

.EXAMPLE
  .\scripts\restart_windows.ps1
  .\scripts\restart_windows.ps1 -Port 9000
#>
[CmdletBinding()]
param(
    [string]$TaskName = 'DetectionDashboard',
    # Read from the task's own action arguments when not given -- that is the
    # only place that still knows what the install was told.
    [int]$Port = 0,
    [int]$TimeoutSeconds = 60
)

$ErrorActionPreference = 'Stop'

$Root    = Split-Path -Parent $PSScriptRoot
$LogFile = Join-Path $Root 'data\logs\backend.log'

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal $identity).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-Listener {
    param([int]$TcpPort)
    Get-NetTCPConnection -LocalPort $TcpPort -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
}

# Walk up from a process to the highest ancestor that is still part of the
# backend. Killing the listener alone is not enough: it is a grandchild of
# uvicorn.exe, and the processes above it survive and confuse the next start.
function Get-TreeRoot {
    param([int]$ProcessId)
    $ours = @('python.exe', 'pythonw.exe', 'uvicorn.exe')
    $top  = $ProcessId
    for ($i = 0; $i -lt 8; $i++) {
        $p = Get-CimInstance Win32_Process -Filter "ProcessId=$top" -ErrorAction SilentlyContinue
        if (-not $p) { break }
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($p.ParentProcessId)" -ErrorAction SilentlyContinue
        if (-not $parent -or $ours -notcontains $parent.Name) { break }
        $top = $parent.ProcessId
    }
    $top
}

function Wait-ForPort {
    param([int]$TcpPort, [bool]$WantListening, [int]$Seconds)
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        $listening = [bool](Get-Listener -TcpPort $TcpPort)
        if ($listening -eq $WantListening) { return $true }
        Start-Sleep -Milliseconds 500
    }
    $false
}

# --- checks -----------------------------------------------------------------

if (-not (Test-Admin)) {
    Write-Host 'This needs an elevated PowerShell.' -ForegroundColor Red
    Write-Host ''
    Write-Host "  The '$TaskName' task runs with RunLevel Highest, so the backend's"
    Write-Host '  processes are elevated. An ordinary shell cannot stop the task or'
    Write-Host '  kill those processes -- it gets "Access is denied" on every one.'
    Write-Host ''
    Write-Host '  Right-click PowerShell -> Run as administrator, then:'
    Write-Host "    cd $Root" -ForegroundColor Cyan
    Write-Host '    .\scripts\restart_windows.ps1' -ForegroundColor Cyan
    exit 1
}

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) {
    throw "Task '$TaskName' is not registered. Install it with .\scripts\install_autostart_windows.ps1, or just run .\scripts\run_windows.ps1 in the foreground."
}

if ($Port -eq 0) {
    $arguments = @($task.Actions)[0].Arguments
    $Port = if ($arguments -match '-Port\s+(\d+)') { [int]$Matches[1] } else { 8000 }
}

Write-Host "Restarting '$TaskName' on port $Port" -ForegroundColor Cyan

# --- 1. stop the task -------------------------------------------------------

if ($task.State -eq 'Running') {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Write-Host '  [1/5] task stopped'
} else {
    Write-Host "  [1/5] task was not running (State: $($task.State))"
}

# --- 2. kill whatever still holds the port ----------------------------------

$listener = Get-Listener -TcpPort $Port
if ($listener) {
    $root = Get-TreeRoot -ProcessId $listener.OwningProcess
    $name = (Get-Process -Id $root -ErrorAction SilentlyContinue).ProcessName
    Write-Host "  [2/5] port $Port still held by pid $($listener.OwningProcess); killing tree from pid $root ($name)"
    # /T for the whole tree, /F because a worker wedged in a native call will
    # not answer anything politer.
    & taskkill /PID $root /T /F 2>&1 | Out-Null
} else {
    Write-Host "  [2/5] nothing was holding port $Port"
}

# --- 3. confirm it is actually free -----------------------------------------

if (-not (Wait-ForPort -TcpPort $Port -WantListening $false -Seconds 20)) {
    $stuck = Get-Listener -TcpPort $Port
    throw ("Port $Port is still held by pid $($stuck.OwningProcess) after 20s. " +
           'Starting the task now would only crash-loop it. Stop that process, then run this again.')
}
Write-Host "  [3/5] port $Port is free"

# --- 4. re-arm the task if a crash-loop disabled it -------------------------

$task = Get-ScheduledTask -TaskName $TaskName
if (-not $task.Settings.Enabled) {
    Enable-ScheduledTask -TaskName $TaskName | Out-Null
    Write-Host '  [4/5] task was disabled (earlier crash-loop?); re-enabled' -ForegroundColor Yellow
} else {
    Write-Host '  [4/5] task is enabled'
}

# --- 5. start, and wait for it to actually serve ----------------------------

Start-ScheduledTask -TaskName $TaskName
if (Wait-ForPort -TcpPort $Port -WantListening $true -Seconds $TimeoutSeconds) {
    $up = Get-Listener -TcpPort $Port
    Write-Host "  [5/5] up -- listening on $Port (pid $($up.OwningProcess))" -ForegroundColor Green
    Write-Host ''
    Write-Host "  Open http://localhost:$Port  (hard-refresh with Ctrl+Shift+R after a frontend rebuild)"
} else {
    Write-Host "  [5/5] nothing is listening on $Port after ${TimeoutSeconds}s" -ForegroundColor Red
    Write-Host ''
    Write-Host '  The last few log lines usually say why:' -ForegroundColor Yellow
    if (Test-Path $LogFile) { Get-Content $LogFile -Tail 20 }
    exit 1
}

Write-Host ''
Write-Host "  Watch it run: Get-Content `"$LogFile`" -Tail 50 -Wait"

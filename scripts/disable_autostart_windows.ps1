<#
.SYNOPSIS
  Switch the autostart off -- and back on -- while Windows is running, without
  throwing the Scheduled Task away.

.DESCRIPTION
  There are two separate things called "auto start" in this project, and this
  script is about the outer one:

    * the Scheduled Task that starts the backend at boot  <- this script
    * the per-source "Auto start" checkbox in the dashboard, which decides
      which cameras come back once the backend is up

  Turning the task off is not the same as removing it.
  install_autostart_windows.ps1 -Uninstall deletes it, so getting it back means
  registering it again and re-deciding the account, the port and the firewall
  rule. This leaves all of that in place and only flips the switch -- which is
  what you want for maintenance: a demo on the same port, a driver update, a
  camera being rewired, an evening when the machine should stay quiet.

  Disabling also stops what is running now, because a backend that keeps
  serving until the next reboot is not "off" in any sense that helps. Use
  -KeepRunning when you only mean "don't come back after the next boot".

  Rights: the same ones that registered the task. A boot-triggered task is
  machine-wide, so it needs an elevated PowerShell; a -Trigger Logon task does
  not.

.EXAMPLE
  .\scripts\disable_autostart_windows.ps1                 # off now and at boot
  .\scripts\disable_autostart_windows.ps1 -KeepRunning    # off at boot, leave it up
  .\scripts\disable_autostart_windows.ps1 -Enable         # back on, start now
  .\scripts\disable_autostart_windows.ps1 -Enable -NoStart
  .\scripts\disable_autostart_windows.ps1 -Status
#>
[CmdletBinding(DefaultParameterSetName = 'Disable')]
param(
    # Leave the backend that is running now alone; only stop it coming back.
    [Parameter(ParameterSetName = 'Disable')]
    [switch]$KeepRunning,

    [Parameter(ParameterSetName = 'Enable', Mandatory = $true)]
    [switch]$Enable,

    # Re-enable for the next boot, but don't start the backend right now.
    [Parameter(ParameterSetName = 'Enable')]
    [switch]$NoStart,

    [Parameter(ParameterSetName = 'Status', Mandatory = $true)]
    [switch]$Status,

    # How long to wait for the port to be released after stopping the task.
    [int]$TimeoutSeconds = 20,

    [string]$TaskName = 'DetectionDashboard'
)

$ErrorActionPreference = 'Stop'

$Root    = Split-Path -Parent $PSScriptRoot
$LogFile = Join-Path $Root 'data\logs\backend.log'

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal $identity).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-TaskOrExit {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $task) {
        Write-Host "Task '$TaskName' is not registered, so there is nothing to switch." -ForegroundColor Yellow
        Write-Host 'Install it first: .\scripts\install_autostart_windows.ps1'
        exit 0
    }
    $task
}

# The port lives in the task's own action arguments ("-Port 9000"), which is
# the only place that still knows what the install was told. Asking for it
# again here would just be a second chance to get it wrong.
function Get-TaskPort {
    param($Task)
    $arguments = @($Task.Actions)[0].Arguments
    if ($arguments -match '-Port\s+(\d+)') { return [int]$Matches[1] }
    if ($env:PORT) { return [int]$env:PORT }
    return 8000
}

# Boot-triggered or logon-triggered, in the words the messages below need. The
# install script offers both, and telling someone with a logon task that it
# comes back "at the next boot" is how you end up debugging a machine that was
# never going to start it.
function Get-TriggerWhen {
    param($Task)
    $boot = @($Task.Triggers | Where-Object { $_.CimClass.CimClassName -eq 'MSFT_TaskBootTrigger' })
    if ($boot.Count) { return 'at boot' }
    return 'at logon'
}

function Get-Listener {
    param([int]$TaskPort)
    Get-NetTCPConnection -LocalPort $TaskPort -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
}

function Write-State {
    param($Task, [int]$TaskPort)
    $info = Get-ScheduledTaskInfo -TaskName $TaskName
    # The switch and the process are two different questions, and State answers
    # only the second: a task disabled while an instance is still up reports
    # "Running", which read on its own says the opposite of what happened.
    $armed  = [bool]$Task.Settings.Enabled
    $colour = if ($armed) { 'Green' } else { 'Yellow' }
    $when   = Get-TriggerWhen -Task $Task
    Write-Host "Task     : $TaskName" -ForegroundColor Cyan
    Write-Host "Autostart: $(if ($armed) { "enabled ($when)" } else { 'disabled' })" -ForegroundColor $colour
    Write-Host "State    : $($Task.State)"
    Write-Host "Runs as  : $($Task.Principal.UserId) ($($Task.Principal.LogonType))"
    Write-Host "Last run : $($info.LastRunTime)  result 0x$('{0:X}' -f $info.LastTaskResult)"
    Write-Host "Next run : $($info.NextRunTime)"
    $listener = Get-Listener -TaskPort $TaskPort
    if ($listener) {
        $owner = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
        Write-Host "Backend  : listening on $TaskPort (pid $($listener.OwningProcess) $($owner.ProcessName))" -ForegroundColor Green
    } else {
        Write-Host "Backend  : nothing listening on $TaskPort" -ForegroundColor Yellow
    }
    Write-Host "Log      : $LogFile"
}

# --- Status -----------------------------------------------------------------

if ($Status) {
    $task = Get-TaskOrExit
    Write-State -Task $task -TaskPort (Get-TaskPort -Task $task)
    exit 0
}

# --- Enable -----------------------------------------------------------------

if ($Enable) {
    $task = Get-TaskOrExit
    $port = Get-TaskPort -Task $task
    try {
        Enable-ScheduledTask -TaskName $TaskName | Out-Null
    } catch {
        throw ("Could not enable '$TaskName': $($_.Exception.Message)" +
               $(if (Test-Admin) { '' } else { ' Try an elevated PowerShell.' }))
    }
    $next = if ((Get-TriggerWhen -Task $task) -eq 'at boot') { 'at the next boot' } else { 'at your next logon' }
    Write-Host "Enabled '$TaskName' -- it will start again $next." -ForegroundColor Green

    if ($NoStart) {
        Write-Host "Not starting it now (-NoStart)."
        exit 0
    }
    $listener = Get-Listener -TaskPort $port
    if ($listener) {
        Write-Host "    port $port is already in use (pid $($listener.OwningProcess)); not starting the task now." -ForegroundColor Yellow
        exit 0
    }
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "    started -- give it a moment, then open http://localhost:$port" -ForegroundColor Green
    Write-Host "    watch it come up: Get-Content `"$LogFile`" -Tail 50 -Wait"
    exit 0
}

# --- Disable ----------------------------------------------------------------

$task = Get-TaskOrExit
$port = Get-TaskPort -Task $task

try {
    Disable-ScheduledTask -TaskName $TaskName | Out-Null
} catch {
    throw ("Could not disable '$TaskName': $($_.Exception.Message)" +
           $(if (Test-Admin) { '' } else {
               ' A boot-triggered task is machine-wide -- right-click PowerShell -> Run as administrator.' }))
}
$next = if ((Get-TriggerWhen -Task $task) -eq 'at boot') { 'at the next boot' } else { 'at your next logon' }
Write-Host "Disabled '$TaskName' -- it will not start $next." -ForegroundColor Green

if ($KeepRunning) {
    Write-Host "Left the running backend alone (-KeepRunning)." -ForegroundColor Yellow
} else {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    # Stopping the task tears down its process tree -- powershell.exe, uvicorn
    # and the detection workers under it. Confirm rather than assume: a worker
    # wedged in a native call is exactly the case where "stopped" is a lie, and
    # the next thing anyone does is bind this port again.
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Listener -TaskPort $port) -and (Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 500
    }
    $listener = Get-Listener -TaskPort $port
    if (-not $listener) {
        Write-Host "    stopped -- nothing is listening on $port any more." -ForegroundColor Green
    } else {
        $owner = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
        Write-Warning ("Port $port is still held by pid $($listener.OwningProcess) " +
                       "($($owner.ProcessName)) after ${TimeoutSeconds}s. If it was started by hand " +
                       "(run_windows.ps1) the task never owned it; otherwise stop it with " +
                       "Stop-Process -Id $($listener.OwningProcess).")
    }
}

Write-Host ''
Write-Host 'The task, its account and any firewall rule are all still in place.'
Write-Host 'Turn it back on with:'
Write-Host "  .\scripts\disable_autostart_windows.ps1 -Enable"
Write-Host 'Or remove it for good with:'
Write-Host "  .\scripts\install_autostart_windows.ps1 -Uninstall"

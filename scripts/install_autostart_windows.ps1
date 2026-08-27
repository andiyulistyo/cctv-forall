<#
.SYNOPSIS
  Register (or remove) the Windows Scheduled Task that starts the dashboard
  automatically when the machine boots.

.DESCRIPTION
  A Scheduled Task, not a Startup-folder shortcut: the dashboard has to come
  back after a power cut whether or not anyone logs in, and only a task can
  start before logon, restart itself after a crash, and survive the console
  window being closed. The task runs scripts\service_windows.ps1, which keeps
  uvicorn alive and logs to data\logs\backend.log.

  Two accounts are on offer:

    -RunAs CurrentUser  (default) the task runs as you, logged in or not
                        (S4U: no password is stored). Keeps whatever your
                        account can reach -- user-scoped CUDA/OpenVINO
                        installs, a home-directory model cache, mapped paths.
    -RunAs System       the task runs as LocalSystem. Independent of your
                        account entirely, but it cannot see anything installed
                        only for your user.

  Both need an elevated PowerShell, because a boot trigger is machine-wide.
  Without admin rights use -Trigger Logon, which registers a task that starts
  when you sign in.

.EXAMPLE
  .\scripts\install_autostart_windows.ps1
  .\scripts\install_autostart_windows.ps1 -Port 9000 -OpenFirewall
  .\scripts\install_autostart_windows.ps1 -Trigger Logon      # no admin needed
  .\scripts\install_autostart_windows.ps1 -Status
  .\scripts\install_autostart_windows.ps1 -Uninstall
#>
[CmdletBinding(DefaultParameterSetName = 'Install')]
param(
    [Parameter(ParameterSetName = 'Install')]
    [ValidateSet('CurrentUser', 'System')]
    [string]$RunAs = 'CurrentUser',

    # Startup = before/without logon (needs admin). Logon = when you sign in.
    [Parameter(ParameterSetName = 'Install')]
    [ValidateSet('Startup', 'Logon')]
    [string]$Trigger = 'Startup',

    [Parameter(ParameterSetName = 'Install')]
    [string]$BindAddress = '0.0.0.0',

    [Parameter(ParameterSetName = 'Install')]
    [int]$Port = 8000,

    # Let the drivers and the network stack settle before the first camera
    # connection attempt.
    [Parameter(ParameterSetName = 'Install')]
    [int]$DelaySeconds = 30,

    # Allow other machines on the LAN to reach the dashboard. Off by default:
    # opening a port is a change to the machine, not to this project.
    [Parameter(ParameterSetName = 'Install')]
    [switch]$OpenFirewall,

    # Register the task but leave it stopped until the next boot.
    [Parameter(ParameterSetName = 'Install')]
    [switch]$NoStart,

    [Parameter(ParameterSetName = 'Uninstall')]
    [switch]$Uninstall,

    [Parameter(ParameterSetName = 'Status')]
    [switch]$Status,

    [string]$TaskName = 'DetectionDashboard'
)

$ErrorActionPreference = 'Stop'

$Root         = Split-Path -Parent $PSScriptRoot
$ServicePs1   = Join-Path $PSScriptRoot 'service_windows.ps1'
$Uvicorn      = Join-Path $Root 'backend\.venv\Scripts\uvicorn.exe'
$LogFile      = Join-Path $Root 'data\logs\backend.log'
$PowerShell   = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$FirewallRule = "$TaskName (TCP $Port)"

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal $identity).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-Task {
    Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
}

# --- Status -----------------------------------------------------------------

if ($Status) {
    $task = Get-Task
    if (-not $task) {
        Write-Host "Task '$TaskName' is not registered." -ForegroundColor Yellow
        exit 0
    }
    $info = Get-ScheduledTaskInfo -TaskName $TaskName
    Write-Host "Task     : $TaskName" -ForegroundColor Cyan
    Write-Host "State    : $($task.State)"
    Write-Host "Runs as  : $($task.Principal.UserId) ($($task.Principal.LogonType))"
    Write-Host "Last run : $($info.LastRunTime)  result 0x$('{0:X}' -f $info.LastTaskResult)"
    Write-Host "Next run : $($info.NextRunTime)"
    Write-Host "Log      : $LogFile"
    exit 0
}

# --- Uninstall --------------------------------------------------------------

if ($Uninstall) {
    if (-not (Get-Task)) {
        Write-Host "Task '$TaskName' is not registered; nothing to remove." -ForegroundColor Yellow
    } else {
        # Said here and nowhere else: someone reaching for -Uninstall to free
        # the port for an afternoon wants disable_autostart_windows.ps1 and a
        # one-word undo, not to pick the account and the firewall rule all over
        # again tomorrow.
        Write-Host 'Tip: .\scripts\disable_autostart_windows.ps1 switches it off without deleting it.' -ForegroundColor DarkGray
        if (-not (Test-Admin)) {
            # A task the current user registered can be removed without
            # elevation; a machine-wide one cannot. Try, and say which it was.
            Write-Host 'Not elevated -- this only works for a -Trigger Logon task.' -ForegroundColor Yellow
        }
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'." -ForegroundColor Green
    }
    if (Get-NetFirewallRule -DisplayName $FirewallRule -ErrorAction SilentlyContinue) {
        Remove-NetFirewallRule -DisplayName $FirewallRule
        Write-Host "Removed firewall rule '$FirewallRule'." -ForegroundColor Green
    }
    Write-Host "The log at $LogFile was left in place."
    exit 0
}

# --- Install ----------------------------------------------------------------

$needsAdmin = ($Trigger -eq 'Startup') -or ($RunAs -eq 'System') -or $OpenFirewall
if ($needsAdmin -and -not (Test-Admin)) {
    throw ("This needs an elevated PowerShell (a boot trigger, the SYSTEM account and " +
           "firewall rules are all machine-wide). Right-click PowerShell -> Run as " +
           "administrator and run it again, or use -Trigger Logon to start the dashboard " +
           "when you sign in instead.")
}

if (-not (Test-Path $ServicePs1)) { throw "Missing $ServicePs1." }
if (-not (Test-Path $Uvicorn)) {
    Write-Warning "backend\.venv is missing. Run .\scripts\setup_windows.ps1 before the next boot, or the task will only write an error to $LogFile."
}

Write-Host '==> Building the task' -ForegroundColor Cyan

$arguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden ' +
             "-File `"$ServicePs1`" -BindAddress $BindAddress -Port $Port"
$action = New-ScheduledTaskAction -Execute $PowerShell -Argument $arguments -WorkingDirectory $Root

$currentUser = "$env:USERDOMAIN\$env:USERNAME"
if ($Trigger -eq 'Startup') {
    $taskTrigger = New-ScheduledTaskTrigger -AtStartup
} else {
    $taskTrigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
}
$taskTrigger.Delay = "PT${DelaySeconds}S"

if ($RunAs -eq 'System') {
    $principal = New-ScheduledTaskPrincipal -UserId 'NT AUTHORITY\SYSTEM' `
        -LogonType ServiceAccount -RunLevel Highest
} elseif ($Trigger -eq 'Startup') {
    # S4U runs the task as this user with no logon session and no stored
    # password -- the reason the dashboard can come up before anyone signs in.
    $principal = New-ScheduledTaskPrincipal -UserId $currentUser `
        -LogonType S4U -RunLevel Highest
} else {
    $principal = New-ScheduledTaskPrincipal -UserId $currentUser `
        -LogonType Interactive -RunLevel Limited
}

# StartWhenAvailable covers a boot the task missed (the machine was asleep);
# the restart pair is the outer net under service_windows.ps1's own retries;
# ExecutionTimeLimit 0 means "never kill it" -- the default of three days would
# otherwise stop the dashboard mid-week.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -DontStopOnIdleEnd `
    -MultipleInstances IgnoreNew `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

$description = "Detection Dashboard (FastAPI + YOLO) on http://localhost:$Port -- " +
               "runs $ServicePs1, logs to $LogFile"

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $taskTrigger `
    -Principal $principal -Settings $settings -Description $description -Force | Out-Null

Write-Host "    registered '$TaskName' ($Trigger trigger, +${DelaySeconds}s, as $($principal.UserId))" -ForegroundColor Green

if ($OpenFirewall) {
    if (Get-NetFirewallRule -DisplayName $FirewallRule -ErrorAction SilentlyContinue) {
        Remove-NetFirewallRule -DisplayName $FirewallRule
    }
    # Private/Domain only: a public profile is coffee-shop Wi-Fi, and the
    # dashboard is not meant to be exposed there.
    New-NetFirewallRule -DisplayName $FirewallRule -Direction Inbound -Action Allow `
        -Protocol TCP -LocalPort $Port -Profile Private, Domain | Out-Null
    Write-Host "    opened TCP $Port for private and domain networks" -ForegroundColor Green
}

if (-not $NoStart) {
    $listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($listening) {
        Write-Host "    port $Port is already in use (run_windows.ps1 still running?); not starting the task now" -ForegroundColor Yellow
    } else {
        Start-ScheduledTask -TaskName $TaskName
        Write-Host "    started -- give it a moment, then open http://localhost:$Port" -ForegroundColor Green
    }
}

Write-Host ''
Write-Host 'Useful commands:'
Write-Host "  .\scripts\install_autostart_windows.ps1 -Status"
Write-Host "  Get-Content `"$LogFile`" -Tail 50 -Wait"
Write-Host "  .\scripts\disable_autostart_windows.ps1            # switch it off for now"
Write-Host "  .\scripts\disable_autostart_windows.ps1 -Enable    # ...and back on"
Write-Host "  .\scripts\install_autostart_windows.ps1 -Uninstall # remove it for good"

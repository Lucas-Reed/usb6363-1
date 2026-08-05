[CmdletBinding()]
param(
    # Start：只补齐尚未运行的服务；Restart：停止后全部重启；
    # Stop：停止全部服务；Status：只查看状态，不改变任何进程。
    [ValidateSet("Start", "Restart", "Stop", "Status")]
    [string]$Action = "Restart"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

# 用脚本自身的位置确定项目目录，所以从任意 PowerShell 目录运行都可以。
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDirectory = Join-Path $ProjectRoot "data\runtime_logs"
$PythonExecutable = (Get-Command python -ErrorAction Stop).Source

# 端口和入口文件是双重身份检查。停止进程前两者必须同时匹配，避免误杀其他软件。
$Services = @(
    [pscustomobject]@{ Name = "usb6363-core"; Port = 8765; Script = "usb6363_server.py" };
    [pscustomobject]@{ Name = "two-peak-viewer"; Port = 8766; Script = "two_peak_viewer.py" };
    [pscustomobject]@{ Name = "power-drift-webui"; Port = 8767; Script = "power_drift_webui.py" };
    [pscustomobject]@{ Name = "ai-stream-console"; Port = 8768; Script = "ai_stream_console.py" }
)

function Get-PortProcessInfo {
    param([Parameter(Mandatory)]$Service)

    $listener = Get-NetTCPConnection `
        -LocalPort $Service.Port `
        -State Listen `
        -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $listener) {
        return $null
    }

    $processInfo = Get-CimInstance `
        Win32_Process `
        -Filter "ProcessId=$($listener.OwningProcess)" `
        -ErrorAction SilentlyContinue
    $commandLine = if ($null -eq $processInfo) { "" } else { [string]$processInfo.CommandLine }
    return [pscustomobject]@{
        ProcessId = [int]$listener.OwningProcess
        CommandLine = $commandLine
        Expected = $commandLine -like "*$($Service.Script)*"
    }
}

function Test-PortListening {
    param([Parameter(Mandatory)][int]$Port)

    return $null -ne (Get-NetTCPConnection `
        -LocalPort $Port `
        -State Listen `
        -ErrorAction SilentlyContinue | Select-Object -First 1)
}

function Invoke-OptionalPost {
    param(
        [Parameter(Mandatory)][string]$Url,
        [hashtable]$Body = @{},
        [int]$TimeoutSeconds = 6
    )

    try {
        $json = $Body | ConvertTo-Json -Depth 8
        Invoke-RestMethod `
            -Method Post `
            -Uri $Url `
            -ContentType "application/json; charset=utf-8" `
            -Body ([System.Text.Encoding]::UTF8.GetBytes($json)) `
            -TimeoutSec $TimeoutSeconds | Out-Null
        Write-Host "[OK] POST $Url"
    }
    catch {
        # 停止接口均设计为尽力而为。服务未运行或任务已经停止时，不阻止后续进程退出。
        Write-Warning "POST $Url failed: $($_.Exception.Message)"
    }
}

function Get-UnifiedRestoreSettings {
    # 只保存健康运行中的统一流。已经停止或带错误的流不会被自动重新启动。
    if (-not (Test-PortListening -Port 8765)) {
        return $null
    }

    try {
        $status = Invoke-RestMethod `
            -Uri "http://127.0.0.1:8765/api/ai/unified/status" `
            -TimeoutSec 5
        if (-not [bool]$status.running) {
            return $null
        }
        $settings = $status.settings
        return [ordered]@{
            channels = @($settings.channels)
            samples_per_frame = [int]$settings.samples_per_frame
            rate = [double]$settings.rate_per_channel
            terminal_config = [string]$settings.terminal_config
            min_val = [double]$settings.min_val
            max_val = [double]$settings.max_val
            timeout = [double]$settings.timeout
            trigger_enabled = [bool]$settings.trigger_enabled
            trigger_source = [string]$settings.trigger_source
            trigger_edge = [string]$settings.trigger_edge
            resync_every_frames = [int]$settings.resync_every_frames
        }
    }
    catch {
        Write-Warning "Could not read unified stream settings: $($_.Exception.Message)"
        return $null
    }
}

function Stop-RunningTasks {
    # 先停可能写 AO 的任务，再停记录器，最后才释放底层 AI task。
    if (Test-PortListening -Port 8766) {
        Invoke-OptionalPost -Url "http://127.0.0.1:8766/api/power_lock/stop"
        Invoke-OptionalPost -Url "http://127.0.0.1:8766/api/ao_scan/stop"
        Invoke-OptionalPost -Url "http://127.0.0.1:8766/api/test_sync/stop"
        Invoke-OptionalPost -Url "http://127.0.0.1:8766/api/trend/stop"
        # unified_stream 模式下，这个接口只让查看器脱离，不会提前停止统一流。
        Invoke-OptionalPost `
            -Url "http://127.0.0.1:8766/api/stream/stop" `
            -Body @{ stream_source = "unified_stream" }
    }
    if (Test-PortListening -Port 8767) {
        Invoke-OptionalPost -Url "http://127.0.0.1:8767/api/stop" -TimeoutSeconds 12
    }
    if (Test-PortListening -Port 8765) {
        Invoke-OptionalPost -Url "http://127.0.0.1:8765/api/ai/unified/stop" -TimeoutSeconds 12
        Invoke-OptionalPost -Url "http://127.0.0.1:8765/api/ai/frame_stream/stop" -TimeoutSeconds 12
        Invoke-OptionalPost -Url "http://127.0.0.1:8765/api/ai/clear" -TimeoutSeconds 12
    }
}

function Stop-LabServices {
    Stop-RunningTasks

    # 从上层 UI 向底层 core 逆序停止，避免上层在 core 退出后继续轮询产生噪声。
    foreach ($service in @($Services | Sort-Object Port -Descending)) {
        $info = Get-PortProcessInfo -Service $service
        if ($null -eq $info) {
            Write-Host "[--] $($service.Name) is already stopped."
            continue
        }
        if (-not $info.Expected) {
            throw "Port $($service.Port) is owned by an unexpected process: $($info.CommandLine)"
        }

        Stop-Process -Id $info.ProcessId -Force
        $processObject = Get-Process -Id $info.ProcessId -ErrorAction SilentlyContinue
        if ($null -ne $processObject) {
            $processObject.WaitForExit(5000) | Out-Null
        }
        Write-Host "[OK] Stopped $($service.Name) (PID $($info.ProcessId))."
    }
}

function Wait-ServicePort {
    param(
        [Parameter(Mandatory)]$Service,
        [int]$TimeoutSeconds = 12
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-PortListening -Port $Service.Port) {
            return $true
        }
        Start-Sleep -Milliseconds 150
    }
    return $false
}

function Start-LabService {
    param([Parameter(Mandatory)]$Service)

    $existing = Get-PortProcessInfo -Service $Service
    if ($null -ne $existing) {
        if ($existing.Expected) {
            Write-Host "[--] $($Service.Name) is already running (PID $($existing.ProcessId))."
            return
        }
        throw "Port $($Service.Port) is owned by an unexpected process: $($existing.CommandLine)"
    }

    New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
    $scriptPath = Join-Path $ProjectRoot $Service.Script
    $stdoutPath = Join-Path $LogDirectory "$($Service.Name).out.log"
    $stderrPath = Join-Path $LogDirectory "$($Service.Name).err.log"
    $processObject = Start-Process `
        -FilePath $PythonExecutable `
        -ArgumentList @("-B", "-u", $scriptPath) `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru

    if (-not (Wait-ServicePort -Service $Service)) {
        $errorTail = if (Test-Path $stderrPath) {
            (Get-Content -LiteralPath $stderrPath -Tail 20 -ErrorAction SilentlyContinue) -join [Environment]::NewLine
        } else {
            "No error log was created."
        }
        throw "$($Service.Name) did not listen on port $($Service.Port).`n$errorTail"
    }
    Write-Host "[OK] Started $($Service.Name) on http://127.0.0.1:$($Service.Port) (PID $($processObject.Id))."
}

function Start-LabServices {
    param($UnifiedRestoreSettings)

    foreach ($service in @($Services | Sort-Object Port)) {
        Start-LabService -Service $service
    }

    if ($null -ne $UnifiedRestoreSettings) {
        try {
            $json = $UnifiedRestoreSettings | ConvertTo-Json -Depth 8
            Invoke-RestMethod `
                -Method Post `
                -Uri "http://127.0.0.1:8765/api/ai/unified/start" `
                -ContentType "application/json; charset=utf-8" `
                -Body ([System.Text.Encoding]::UTF8.GetBytes($json)) `
                -TimeoutSec 20 | Out-Null
            Write-Host "[OK] Restored the previously running unified AI stream."
        }
        catch {
            Write-Warning "Services are running, but unified stream restore failed: $($_.Exception.Message)"
        }
    }
    else {
        Write-Host "[--] Unified AI stream was not running before this action; it was not auto-started."
    }
}

function Show-LabServiceStatus {
    $rows = foreach ($service in $Services) {
        $info = Get-PortProcessInfo -Service $service
        [pscustomobject]@{
            Service = $service.Name
            Port = $service.Port
            State = if ($null -eq $info) { "STOPPED" } elseif ($info.Expected) { "RUNNING" } else { "FOREIGN" }
            PID = if ($null -eq $info) { "--" } else { $info.ProcessId }
        }
    }
    $rows | Format-Table -AutoSize

    if (Test-PortListening -Port 8765) {
        try {
            $unified = Invoke-RestMethod `
                -Uri "http://127.0.0.1:8765/api/ai/unified/status" `
                -TimeoutSec 5
            Write-Host "Unified AI: running=$($unified.running), frame_id=$($unified.frame_id), error=$($unified.error)"
        }
        catch {
            Write-Warning "Could not query unified AI status: $($_.Exception.Message)"
        }
    }
}

Set-Location $ProjectRoot
Write-Host "USB-6363 service manager: $Action"

switch ($Action) {
    "Start" {
        Start-LabServices -UnifiedRestoreSettings $null
        Show-LabServiceStatus
    }
    "Restart" {
        $restoreSettings = Get-UnifiedRestoreSettings
        Stop-LabServices
        Start-LabServices -UnifiedRestoreSettings $restoreSettings
        Show-LabServiceStatus
    }
    "Stop" {
        Stop-LabServices
        Show-LabServiceStatus
    }
    "Status" {
        Show-LabServiceStatus
    }
}

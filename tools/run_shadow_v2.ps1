<#
Launch the 14-run shadow_tuning_v2 batch (fixedtime + maxpressure, --movement_flags),
each run in its own PowerShell window that stays open after the run ends.

    powershell -ExecutionPolicy Bypass -File tools\run_shadow_v2.ps1 [-MaxParallel 4] [-DryRun]

Each window tees its output to logs\shadow_tuning_v2\<test_name>.console.log and, on
exit, writes <test_name>.done (exit code + seconds) so this launcher can track it.
#>
param(
    [int]$MaxParallel = 4,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$RunGroup = "shadow_tuning_v2"
$DataDir = "dataset/llm_light/Hangzhou/4_4"
$LogDir = Join-Path $RepoRoot "logs\$RunGroup"
$S2Scenario = "$DataDir/scenarios/surge_s2_route150_2_3.json"
$S2Config = "anon_4_4_hangzhou_real_s2_route150_2_3.sumocfg"

function Read-S2Scenario {
    Get-Content (Join-Path $RepoRoot $S2Scenario) -Raw | ConvertFrom-Json
}

# The s2 json is the v1 schema: no top-level horizon/surges, only "generator".
function Get-S2Steps($scenario) {
    if ($null -ne $scenario.horizon) { return [int]$scenario.horizon }
    if ($null -ne $scenario.generator.horizon) { return [int]$scenario.generator.horizon }
    return [int]$scenario.simulation_steps
}

function Get-S2SurgesBlock($scenario) {
    if ($null -ne $scenario.surges) { return $scenario.surges }
    return $scenario.generator
}

function New-ScenarioRows($s2Steps) {
    $c3 = "--blockage_scenario $DataDir/scenarios/c3_through_north_2_3_1200s.json"
    $c4 = "--blockage_scenario $DataDir/scenarios/c4_two_lane_west_2_2_1200s.json"
    # Longest first so the 9000-step runs don't finish last.
    @(
        @{ Name = "cross_r1200"; Config = "anon_4_4_hangzhou_real_cross_2_3_r1200.sumocfg"; Steps = 9000; Extra = "" }
        @{ Name = "h2_steady"; Config = "anon_4_4_hangzhou_real_h2_steady.sumocfg"; Steps = 7200; Extra = "" }
        @{ Name = "route_s2"; Config = $S2Config; Steps = $s2Steps; Extra = "" }
        @{ Name = "cross_1h"; Config = "anon_4_4_hangzhou_real_cross_2_3_1h.sumocfg"; Steps = 3600; Extra = "" }
        @{ Name = "hzreal_none"; Config = "anon_4_4_hangzhou_real.sumocfg"; Steps = 3600; Extra = "" }
        @{ Name = "hzreal_c3"; Config = "anon_4_4_hangzhou_real.sumocfg"; Steps = 3600; Extra = $c3 }
        @{ Name = "hzreal_c4"; Config = "anon_4_4_hangzhou_real.sumocfg"; Steps = 3600; Extra = $c4 }
    )
}

function New-RunList($s2Steps) {
    $controllers = @(
        @{ Name = "fixedtime"; Prefix = "ft" }
        @{ Name = "maxpressure"; Prefix = "mp" }
    )
    foreach ($row in New-ScenarioRows $s2Steps) {
        foreach ($controller in $controllers) {
            $testName = "$($controller.Prefix)_$($row.Name)"
            $command = "python runner_baselines.py --controller $($controller.Name) --movement_flags" +
                " --run_group $RunGroup --test_name $testName" +
                " --simulation_config $DataDir/$($row.Config)" +
                " --simulation_steps $($row.Steps) --seed 1"
            if ($row.Extra) { $command += " $($row.Extra)" }
            [pscustomobject]@{
                TestName = $testName
                Controller = $controller.Name
                Config = $row.Config
                Steps = $row.Steps
                Extra = $(if ($row.Extra) { $row.Extra } else { "-" })
                Command = $command
            }
        }
    }
}

function Write-ChildScript($run) {
    $consoleLog = Join-Path $LogDir "$($run.TestName).console.log"
    $doneFile = Join-Path $LogDir "$($run.TestName).done"
    $childPath = Join-Path $LogDir "$($run.TestName).launch.ps1"
    # cmd /c merges stderr as plain text (PS 5.1 would wrap it in ErrorRecords)
    # and its exit code is python's.
    $child = @"
`$host.UI.RawUI.WindowTitle = '$($run.TestName)'
Set-Location '$RepoRoot'
`$env:PYTHONUNBUFFERED = '1'
Write-Host '$($run.Command)'
`$started = Get-Date
cmd /c "$($run.Command) 2>&1" | Tee-Object -FilePath '$consoleLog'
`$exitCode = `$LASTEXITCODE
`$seconds = [int]((Get-Date) - `$started).TotalSeconds
"`$exitCode `$seconds" | Set-Content -Encoding ascii '$doneFile'
Write-Host ""
Write-Host "[$($run.TestName)] exit code `$exitCode after `$seconds s"
"@
    Set-Content -Path $childPath -Value $child -Encoding utf8
    return $childPath
}

function Start-Run($run) {
    $doneFile = Join-Path $LogDir "$($run.TestName).done"
    if (Test-Path $doneFile) { Remove-Item $doneFile }
    $childPath = Write-ChildScript $run
    Start-Process powershell.exe -WorkingDirectory $RepoRoot -ArgumentList @(
        "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$childPath`""
    ) | Out-Null
    Write-Host ("[{0:HH:mm:ss}] launched {1}" -f (Get-Date), $run.TestName)
}

function Test-RunDone($run) {
    Test-Path (Join-Path $LogDir "$($run.TestName).done")
}

# The windows stay open (-NoExit), so "alive" means "no .done file yet".
function Wait-ForFreeSlot($running) {
    while ($true) {
        $alive = @($running | Where-Object { -not (Test-RunDone $_) })
        if ($alive.Count -lt $MaxParallel) { return }
        Start-Sleep -Seconds 10
    }
}

function Wait-ForAll($running) {
    while (@($running | Where-Object { -not (Test-RunDone $_) }).Count -gt 0) {
        Start-Sleep -Seconds 10
    }
}

function Read-RunResult($run) {
    $parts = (Get-Content (Join-Path $LogDir "$($run.TestName).done") -Raw).Trim() -split " "
    [pscustomobject]@{
        TestName = $run.TestName
        ExitCode = [int]$parts[0]
        WallClock = [timespan]::FromSeconds([int]$parts[1]).ToString("hh\:mm\:ss")
    }
}

function Test-CoreBudget {
    $cores = [Environment]::ProcessorCount
    Write-Host "Logical cores: $cores, MaxParallel: $MaxParallel"
    if ($MaxParallel -gt $cores - 2) {
        Write-Warning "MaxParallel ($MaxParallel) > cores - 2 ($($cores - 2)); the machine may be oversubscribed."
    }
}

$s2ScenarioJson = Read-S2Scenario
$runs = @(New-RunList (Get-S2Steps $s2ScenarioJson))

Write-Host "route_s2 scenario surges block ($S2Scenario):"
Write-Host (Get-S2SurgesBlock $s2ScenarioJson | ConvertTo-Json -Depth 10)
Write-Host ""
Test-CoreBudget
$runs | Format-Table TestName, Controller, Config, Steps, Extra -AutoSize | Out-String -Width 400 | Write-Host

if ($DryRun) {
    Write-Host "Dry run: commands that would be launched, in order:"
    foreach ($run in $runs) { Write-Host "  $($run.Command)" }
    return
}

if (-not (Test-Path (Join-Path $DataDir $S2Config))) {
    throw "Missing $DataDir/$S2Config"
}
New-Item -ItemType Directory -Force $LogDir | Out-Null

$running = @()
foreach ($run in $runs) {
    Wait-ForFreeSlot $running
    Start-Run $run
    $running += $run
}
Write-Host "All runs launched; waiting for them to finish..."
Wait-ForAll $running

Write-Host ""
Write-Host "Summary:"
$runs | ForEach-Object { Read-RunResult $_ } | Format-Table -AutoSize | Out-String -Width 200 | Write-Host

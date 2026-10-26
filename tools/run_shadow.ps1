<#
Launch a shadow-tuning batch (fixedtime + maxpressure, --movement_flags --flag_rule v2)
over every scenario row x seed, each run in its own PowerShell window that stays open
after the run ends.

    powershell -ExecutionPolicy Bypass -File tools\run_shadow.ps1 -RunGroup shadow_tuning_v3 -Seeds 1,2 [-MaxParallel 8] [-DryRun]

Each window tees its output to logs\<RunGroup>\<test_name>.console.log (test_name ends
in _s<seed>) and, on exit, writes <test_name>.done (exit code + seconds) so this
launcher can track it.
#>
param(
    [Parameter(Mandatory = $true)][string]$RunGroup,
    [Parameter(Mandatory = $true)][int[]]$Seeds,
    [int]$MaxParallel = 8,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$DataDir = "dataset/llm_light/Hangzhou/4_4"
$LogDir = Join-Path $RepoRoot "logs\$RunGroup"
$FlagRule = "v2"

function New-ScenarioRows {
    $c3 = "--blockage_scenario $DataDir/scenarios/c3_through_north_2_3_1200s.json"
    $c4 = "--blockage_scenario $DataDir/scenarios/c4_two_lane_west_2_2_1200s.json"
    # Longest first so the 9000-step runs don't finish last.
    @(
        @{ Name = "cross_r1200"; Config = "anon_4_4_hangzhou_real_cross_2_3_r1200.sumocfg"; Steps = 9000; Extra = "" }
        @{ Name = "h2_steady"; Config = "anon_4_4_hangzhou_real_h2_steady.sumocfg"; Steps = 7200; Extra = "" }
        @{ Name = "cross_1h"; Config = "anon_4_4_hangzhou_real_cross_2_3_1h.sumocfg"; Steps = 3600; Extra = "" }
        @{ Name = "hzreal_none"; Config = "anon_4_4_hangzhou_real.sumocfg"; Steps = 3600; Extra = "" }
        @{ Name = "hzreal_c3"; Config = "anon_4_4_hangzhou_real.sumocfg"; Steps = 3600; Extra = $c3 }
        @{ Name = "hzreal_c4"; Config = "anon_4_4_hangzhou_real.sumocfg"; Steps = 3600; Extra = $c4 }
    )
}

function New-RunList {
    $controllers = @(
        @{ Name = "fixedtime"; Prefix = "ft" }
        @{ Name = "maxpressure"; Prefix = "mp" }
    )
    foreach ($row in New-ScenarioRows) {
        foreach ($seed in $Seeds) {
            foreach ($controller in $controllers) {
                $testName = "$($controller.Prefix)_$($row.Name)_s$seed"
                $command = "python runner_baselines.py --controller $($controller.Name)" +
                    " --movement_flags --flag_rule $FlagRule" +
                    " --run_group $RunGroup --test_name $testName" +
                    " --simulation_config $DataDir/$($row.Config)" +
                    " --simulation_steps $($row.Steps) --seed $seed"
                if ($row.Extra) { $command += " $($row.Extra)" }
                [pscustomobject]@{
                    TestName = $testName
                    Controller = $controller.Name
                    Seed = $seed
                    Config = $row.Config
                    Steps = $row.Steps
                    Extra = $(if ($row.Extra) { $row.Extra } else { "-" })
                    Command = $command
                }
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

# Fail here rather than in every window if the runner doesn't know --flag_rule.
function Assert-RunnerSupportsFlagRule {
    $help = cmd /c "python runner_baselines.py --help 2>&1" | Out-String
    if ($help -notmatch "--flag_rule") {
        throw "runner_baselines.py does not accept --flag_rule; nothing launched."
    }
}

$runs = @(New-RunList)

Write-Host "Run group: $RunGroup, seeds: $($Seeds -join ','), flag rule: $FlagRule, runs: $($runs.Count)"
Test-CoreBudget
$runs | Format-Table TestName, Controller, Seed, Config, Steps, Extra -AutoSize | Out-String -Width 400 | Write-Host

if ($DryRun) {
    Write-Host "Dry run: commands that would be launched, in order:"
    foreach ($run in $runs) { Write-Host "  $($run.Command)" }
    return
}

Assert-RunnerSupportsFlagRule
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

<#
.SYNOPSIS
    Phase 4 noise run, unattended: check_env -> generate -> evaluate -> rejudge.

.DESCRIPTION
    Every step runs with --resume, which means "continue or start, never throw away": re-running
    this script with the same arguments after a crash (or a reboot) picks up where it stopped.
    Answers and grades are saved after each one, so a crash loses at most the one in flight.

    A step that exits non-zero is retried up to -Retries times (after -RetryDelaySeconds, which
    gives a crashed Ollama time to come back). The script stops at the first step that still
    fails, so a config refusal does not run into later steps.

    Output goes to the console and, with a timestamp per line, to results/<Name>.log
    (results/ is gitignored).

.EXAMPLE
    .\scripts\night_run.ps1 -Name 20260928-noise-draft -Repeats 6 -Samples 3 `
        -Frozen evals/frozen/phase4-draft.json

.EXAMPLE
    # Dry run on two items
    .\scripts\night_run.ps1 -Name 20260927-dry -Repeats 2 -Samples 1 `
        -Frozen evals/frozen/phase4-draft.json -Items ver-01,fact-01
#>
param(
    [Parameter(Mandatory)] [string] $Name,
    [Parameter(Mandatory)] [int] $Repeats,
    [Parameter(Mandatory)] [int] $Samples,
    [Parameter(Mandatory)] [string] $Frozen,
    [string] $Dataset = "evals/dataset.draft.jsonl",
    [int] $RejudgeRepeat = 1,
    [string[]] $Items = @(),
    [int] $Retries = 2,
    [int] $RetryDelaySeconds = 60
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$log = Join-Path $root "results/$Name.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

function Write-Log([string] $line) {
    $stamped = "{0:yyyy-MM-dd HH:mm:ss}  {1}" -f (Get-Date), $line
    Write-Host $stamped
    Add-Content -Path $log -Value $stamped -Encoding utf8
}

function Invoke-Step([string] $label, [string[]] $arguments) {
    for ($attempt = 1; $attempt -le $Retries + 1; $attempt++) {
        Write-Log "=== $label (attempt $attempt of $($Retries + 1)): uv $($arguments -join ' ')"
        # -u: unbuffered, so the log shows progress as it happens, not at the end.
        $ErrorActionPreference = "Continue"  # stderr lines are output here, not errors
        & uv @arguments 2>&1 | ForEach-Object { Write-Log "$_" }
        $code = $LASTEXITCODE
        $ErrorActionPreference = "Stop"
        if ($code -eq 0) {
            Write-Log "=== $label done"
            return
        }
        Write-Log "=== $label exited with $code"
        if ($attempt -le $Retries) {
            Write-Log "=== retrying in $RetryDelaySeconds s"
            Start-Sleep -Seconds $RetryDelaySeconds
        }
    }
    Write-Log "=== STOPPED: $label failed $($Retries + 1) times. Fix the cause, then re-run this same command; finished work is kept."
    exit 1
}

$frozenArgs = @("--frozen", $Frozen)
$filter = if ($Items.Count -gt 0) { @("--items", ($Items -join ",")) } else { @() }

Write-Log "night run $Name  repeats $Repeats  samples $Samples  frozen $Frozen  dataset $Dataset"
Invoke-Step "check_env" @("run", "python", "-u", "scripts/check_env.py")
Invoke-Step "generate" (@("run", "python", "-u", "-m", "evals.run_experiment", "generate",
        "--name", $Name, "--repeats", "$Repeats", "--dataset", $Dataset, "--resume") +
    $frozenArgs + $filter)
Invoke-Step "evaluate" (@("run", "python", "-u", "-m", "evals.run_experiment", "evaluate",
        "--name", $Name, "--resume") + $frozenArgs)
Invoke-Step "rejudge" (@("run", "python", "-u", "-m", "evals.run_experiment", "rejudge",
        "--name", $Name, "--repeat", "$RejudgeRepeat", "--samples", "$Samples", "--resume") +
    $frozenArgs)
Write-Log "=== all steps done. Next: uv run python -m evals.stats noise results/$Name.json"

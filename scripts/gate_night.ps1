<#
.SYNOPSIS
    Phase 5 gate validation, unattended: for each candidate generate -> evaluate, then `stats gate`.

.DESCRIPTION
    Each candidate is a label and a checkout directory: "." for this checkout (main), or a git
    worktree of a throwaway branch (the broken candidates never exist in this checkout). Each
    step runs IN that directory, so it uses that checkout's code, .env and results/ folder; the
    finished results file is then copied into this checkout's results/ and gated here against
    evals/baseline.json with this checkout's gate code.

    No --frozen: the prompt and code of a broken candidate differ by design. What must not
    differ (the instrument) is checked by the gate. A checkout with tracked changes is refused
    before anything runs, because the gate would refuse its results hours later.

    Every step runs with --resume: re-running this script with the same arguments after a crash
    picks up where it stopped. A failing step is retried -Retries times, then the script stops.
    A gate FAIL does not stop the script: two candidates are expected to fail.

    Log: results/<Name>.log in this checkout (gitignored).

.EXAMPLE
    .\scripts\gate_night.ps1 -Name 20260929-gate -Repeats 3 `
        -Candidates "nochange=.", "nocite=..\mini-rag-exp-nocite", "nofilter=..\mini-rag-exp-nofilter"
#>
param(
    [Parameter(Mandatory)] [string] $Name,
    [Parameter(Mandatory)] [int] $Repeats,
    [Parameter(Mandatory)] [string[]] $Candidates,
    [string] $Dataset = "evals/dataset.draft.jsonl",
    [string] $Baseline = "evals/baseline.json",
    [int] $Retries = 2,
    [int] $RetryDelaySeconds = 60
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$root = (Resolve-Path (Split-Path -Parent $PSScriptRoot)).Path
Set-Location $root
$log = Join-Path $root "results/$Name.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

. (Join-Path $PSScriptRoot "_steps.ps1")  # Write-Log, Invoke-Step

# label -> absolute checkout directory, in the given order
$runs = [ordered]@{}
foreach ($spec in $Candidates) {
    $label, $dir = $spec -split "=", 2
    if (-not $label -or -not $dir) { throw "candidate '$spec' is not label=directory" }
    $runs[$label] = (Resolve-Path (Join-Path $root $dir)).Path
}

Write-Log "gate night $Name  repeats $Repeats  dataset $Dataset  baseline $Baseline"
foreach ($label in $runs.Keys) {
    $dir = $runs[$label]
    $branch = git -C $dir rev-parse --abbrev-ref HEAD
    $commit = git -C $dir rev-parse --short HEAD
    # Tracked changes only; untracked files (.env is gitignored anyway) do not count.
    $changed = @(git -C $dir status --porcelain | Where-Object { $_ -notmatch '^\?\?' })
    Write-Log "candidate $label  $dir  branch $branch  commit $commit"
    if ($changed.Count -gt 0) {
        Write-Log "=== STOPPED: $dir has tracked changes; commit them first:"
        $changed | ForEach-Object { Write-Log "  $_" }
        exit 1
    }
}

foreach ($label in $runs.Keys) {
    $dir = $runs[$label]
    $run = "$Name-$label"
    Push-Location $dir
    try {
        Invoke-Step "$label check_env" @("run", "python", "-u", "scripts/check_env.py")
        Invoke-Step "$label generate" @("run", "python", "-u", "-m", "evals.run_experiment",
            "generate", "--name", $run, "--repeats", "$Repeats", "--dataset", $Dataset, "--resume")
        Invoke-Step "$label evaluate" @("run", "python", "-u", "-m", "evals.run_experiment",
            "evaluate", "--name", $run, "--resume")
    }
    finally {
        Pop-Location
    }
    if ($dir -ne $root) {
        Copy-Item (Join-Path $dir "results/$run.json") (Join-Path $root "results/$run.json") -Force
        Write-Log "copied results/$run.json from $dir"
    }
}

$summary = @()
foreach ($label in $runs.Keys) {
    $run = "$Name-$label"
    Write-Log "=== gate ${label}: uv run python -m evals.stats gate $Baseline results/$run.json"
    $ErrorActionPreference = "Continue"
    & uv run python -u -m evals.stats gate $Baseline "results/$run.json" 2>&1 |
        ForEach-Object { Write-Log "$_" }
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    $summary += "  $label -> exit $code ($(if ($code -eq 0) { 'PASS' } else { 'FAIL or refused' }))"
}
Write-Log "=== all candidates done; gate exit codes (compare with the DECISIONS 41 predictions):"
$summary | ForEach-Object { Write-Log $_ }

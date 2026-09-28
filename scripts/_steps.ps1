<#
.SYNOPSIS
    Shared by night_run.ps1 and gate_night.ps1 (dot-sourced): timestamped log, retried steps.

.DESCRIPTION
    The caller sets $log (log file path), $Retries and $RetryDelaySeconds before calling these;
    the functions read them from the caller's scope.
#>

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

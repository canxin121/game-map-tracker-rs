[CmdletBinding()]
param(
    [string]$CudaRoot,

    [string]$ComputeCap,

    [switch]$Release,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CargoArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$rawCargoArgs = @($CargoArgs)
$invocationLine = [string]$MyInvocation.Line
$processCommandLine = ""
try {
    $processCommandLine = [string](Get-CimInstance Win32_Process -Filter "ProcessId = $PID").CommandLine
} catch {
    $processCommandLine = ""
}
$gnuReleaseRequested =
    ($invocationLine -match '(^|\s)--release(?=\s|$)') -or
    ($processCommandLine -match '(^|\s)--release(?=\s|$)') -or
    ($CudaRoot -eq "--release") -or
    ($ComputeCap -eq "--release") -or
    ($rawCargoArgs -contains "--release")

if ($gnuReleaseRequested) {
    $Release = $true
    $rawCargoArgs = @($rawCargoArgs | Where-Object { $_ -ne "--release" })
    if ($CudaRoot -eq "--release") {
        $CudaRoot = ""
    }
    if ($ComputeCap -eq "--release") {
        $ComputeCap = ""
    }
}

$invokeParams = @{
    Command = "build"
    Release = $Release
}
if (-not [string]::IsNullOrWhiteSpace($CudaRoot)) {
    $invokeParams.CudaRoot = $CudaRoot
}
if (-not [string]::IsNullOrWhiteSpace($ComputeCap)) {
    $invokeParams.ComputeCap = $ComputeCap
}
& (Join-Path $PSScriptRoot "Invoke-CudaCargo.ps1") @invokeParams @rawCargoArgs
exit $LASTEXITCODE

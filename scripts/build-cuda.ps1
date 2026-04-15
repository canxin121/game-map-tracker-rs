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

$invokeParams = @{
    Command = "build"
}
if (-not [string]::IsNullOrWhiteSpace($CudaRoot)) {
    $invokeParams.CudaRoot = $CudaRoot
}
if (-not [string]::IsNullOrWhiteSpace($ComputeCap)) {
    $invokeParams.ComputeCap = $ComputeCap
}
$forwardArgs = @()
if ($Release) {
    $forwardArgs += "--release"
}
if (@($CargoArgs).Count -gt 0) {
    $forwardArgs += $CargoArgs
}

& (Join-Path $PSScriptRoot "Invoke-CudaCargo.ps1") @invokeParams @forwardArgs
exit $LASTEXITCODE

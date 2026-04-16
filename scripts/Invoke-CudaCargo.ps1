[CmdletBinding()]
param(
    [ValidateSet("build", "check", "run", "test")]
    [string]$Command = "build",

    [string]$CudaRoot,

    [string]$ComputeCap,

    [switch]$Release,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CargoArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Find-VsDevCmd {
    $vswherePath = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswherePath)) {
        throw "Missing vswhere.exe at $vswherePath"
    }

    $installationPath = & $vswherePath -latest -products * -property installationPath
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($installationPath)) {
        throw "Failed to locate a Visual Studio Build Tools installation via vswhere."
    }

    $vsDevCmd = Join-Path $installationPath.Trim() "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path -LiteralPath $vsDevCmd)) {
        throw "Missing VsDevCmd.bat at $vsDevCmd"
    }

    (Resolve-Path -LiteralPath $vsDevCmd).Path
}

function Resolve-CudaRoot {
    param(
        [string]$RequestedRoot
    )

    $candidates = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($RequestedRoot)) {
        $candidates.Add($RequestedRoot)
    }

    foreach ($name in @("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR")) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            $candidates.Add($value)
        }
    }

    $defaultCudaDir = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path -LiteralPath $defaultCudaDir) {
        $versions = Get-ChildItem -LiteralPath $defaultCudaDir -Directory |
            Sort-Object {
                try {
                    [Version]($_.Name.TrimStart("v"))
                } catch {
                    [Version]"0.0"
                }
            } -Descending
        foreach ($entry in $versions) {
            $candidates.Add($entry.FullName)
        }
    }

    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }

        $resolved = (Resolve-Path -LiteralPath $candidate -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($null -eq $resolved) {
            continue
        }

        $cudaHeader = Join-Path $resolved.Path "include\cuda.h"
        if (Test-Path -LiteralPath $cudaHeader) {
            return $resolved.Path
        }
    }

    throw "Failed to locate a CUDA toolkit root. Set -CudaRoot or one of CUDA_HOME / CUDA_PATH / CUDA_ROOT / CUDA_TOOLKIT_ROOT_DIR."
}

function Import-BatchEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BatchFile,

        [string[]]$Arguments = @()
    )

    $argText = if ($Arguments.Count -gt 0) {
        " " + ($Arguments -join " ")
    } else {
        ""
    }
    $commandText = ('"{0}"{1} >nul && set' -f $BatchFile, $argText)
    $environmentDump = & cmd.exe /d /s /c $commandText
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to import environment from $BatchFile"
    }

    foreach ($line in $environmentDump) {
        if ($line -notmatch "^(.*?)=(.*)$") {
            continue
        }
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}

function Add-EnvironmentFlag {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$Flag
    )

    $current = [Environment]::GetEnvironmentVariable($Name)
    $parts = @()
    if (-not [string]::IsNullOrWhiteSpace($current)) {
        $parts = $current -split "\s+" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    }

    if ($parts -contains $Flag) {
        return
    }

    $next = if ($parts.Count -gt 0) {
        "$Flag $current"
    } else {
        $Flag
    }
    [Environment]::SetEnvironmentVariable($Name, $next.Trim())
}

function Resolve-TargetRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot
    )

    $targetDir = [Environment]::GetEnvironmentVariable("CARGO_TARGET_DIR")
    if ([string]::IsNullOrWhiteSpace($targetDir)) {
        return (Join-Path $RepoRoot "target")
    }

    if ([System.IO.Path]::IsPathRooted($targetDir)) {
        return $targetDir
    }

    Join-Path $RepoRoot $targetDir
}

function Assert-BuildOutputUnlocked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,

        [Parameter(Mandatory = $true)]
        [ValidateSet("build", "check", "run", "test")]
        [string]$Command,

        [Parameter(Mandatory = $true)]
        [bool]$Release
    )

    if ($Command -notin @("build", "run")) {
        return
    }

    $profileDir = if ($Release) { "release" } else { "debug" }
    $targetRoot = Resolve-TargetRoot -RepoRoot $RepoRoot
    $binaryPath = [System.IO.Path]::GetFullPath(
        (Join-Path (Join-Path $targetRoot $profileDir) "game-map-tracker-rs.exe")
    )

    $runningProcesses = @(Get-Process -Name "game-map-tracker-rs" -ErrorAction SilentlyContinue | Where-Object {
        try {
            -not [string]::IsNullOrWhiteSpace($_.Path) -and
                ([System.IO.Path]::GetFullPath($_.Path) -eq $binaryPath)
        } catch {
            $false
        }
    })

    if ($runningProcesses.Count -eq 0) {
        return
    }

    $processIds = ($runningProcesses | Select-Object -ExpandProperty Id) -join ", "
    throw "The build output is currently locked by a running app instance: $binaryPath (PID: $processIds). Close the app or set CARGO_TARGET_DIR to another directory, then rerun the build."
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$vsDevCmd = Find-VsDevCmd
$resolvedCudaRoot = Resolve-CudaRoot -RequestedRoot $CudaRoot

Import-BatchEnvironment -BatchFile $vsDevCmd -Arguments @("-no_logo", "-arch=x64", "-host_arch=x64")

[Environment]::SetEnvironmentVariable("CUDA_HOME", $resolvedCudaRoot)
[Environment]::SetEnvironmentVariable("CUDA_PATH", $resolvedCudaRoot)
[Environment]::SetEnvironmentVariable("CUDA_ROOT", $resolvedCudaRoot)
[Environment]::SetEnvironmentVariable("CUDA_TOOLKIT_ROOT_DIR", $resolvedCudaRoot)

$clCommand = Get-Command cl.exe -ErrorAction Stop
$nvccCcbin = Split-Path -Parent $clCommand.Source
[Environment]::SetEnvironmentVariable("NVCC_CCBIN", $nvccCcbin)

Add-EnvironmentFlag -Name "CL" -Flag "/Zc:preprocessor"
Add-EnvironmentFlag -Name "NVCC_PREPEND_FLAGS" -Flag "-Xcompiler=/Zc:preprocessor"

if (-not [string]::IsNullOrWhiteSpace($ComputeCap)) {
    [Environment]::SetEnvironmentVariable("CUDA_COMPUTE_CAP", $ComputeCap)
}

$fullCargoArgs = @($Command, "--features", "ai-candle-cuda")
if ($Release) {
    $fullCargoArgs += "--release"
}
foreach ($cargoArg in @($CargoArgs)) {
    if (-not [string]::IsNullOrWhiteSpace($cargoArg)) {
        $fullCargoArgs += $cargoArg
    }
}

Write-Host "Visual Studio env: $vsDevCmd"
Write-Host "CUDA root: $resolvedCudaRoot"
Write-Host "NVCC_CCBIN: $nvccCcbin"
if (-not [string]::IsNullOrWhiteSpace($ComputeCap)) {
    Write-Host "CUDA_COMPUTE_CAP: $ComputeCap"
}
Write-Host ("cargo " + ($fullCargoArgs -join " "))

Push-Location $repoRoot
try {
    Assert-BuildOutputUnlocked -RepoRoot $repoRoot -Command $Command -Release $Release.IsPresent
    & cargo @fullCargoArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}

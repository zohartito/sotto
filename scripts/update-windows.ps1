<#
.SYNOPSIS
  Update Sotto from GitHub once it has quit, then start it again.

.DESCRIPTION
  Started hidden by the tray's "Check for updates…" (it can also be run by
  hand). Windows keeps the files of a running Python in use, so this waits
  until no process runs from Sotto's environment, then pulls the update with
  git (fast-forward only), re-runs install-windows.ps1 -SkipSetup for the
  pinned packages and shortcut, and starts Sotto from the Start Menu shortcut.
  The outcome goes to update-status.txt in the data folder, which the tray
  reports once at its next start; the full output is in update.log.
#>
param(
  [string]$DataDir = '',
  [string]$VenvDir = '',
  [int]$WaitSeconds = 120
)

Set-StrictMode -Version 2
$ErrorActionPreference = 'Continue'

$Root = Split-Path -Parent $PSScriptRoot
if (-not $DataDir) { $DataDir = Join-Path $env:LOCALAPPDATA 'sotto-alpha' }
if (-not $VenvDir) { $VenvDir = Join-Path $Root 'venv-alpha' }
$Log = Join-Path $DataDir 'update.log'
$Status = Join-Path $DataDir 'update-status.txt'
$Shortcut = Join-Path ([Environment]::GetFolderPath('Programs')) 'Sotto.lnk'
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
Set-Content -LiteralPath $Log -Value "update started $(Get-Date -Format s)" -Encoding UTF8

function Running-From-Venv {
  $prefix = [IO.Path]::GetFullPath($VenvDir).TrimEnd('\') + '\'
  return @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
      $path = $null
      try { $path = $_.Path } catch { }
      $path -and $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
    })
}

$deadline = (Get-Date).AddSeconds($WaitSeconds)
while (@(Running-From-Venv).Count -gt 0 -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 300 }
if (@(Running-From-Venv).Count -gt 0) {
  Add-Content -LiteralPath $Log -Value 'Sotto did not quit; nothing was changed.' -Encoding UTF8
  Set-Content -LiteralPath $Status -Value 'failed Sotto did not quit, so nothing was changed.' -Encoding UTF8
  exit 1
}

$ok = $false
git -C $Root pull --ff-only *>> $Log
if ($LASTEXITCODE -eq 0) {
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root 'scripts\install-windows.ps1') `
      -SkipSetup -DataDir $DataDir -VenvDir $VenvDir *>> $Log
  $ok = ($LASTEXITCODE -eq 0)
}
$version = (git -C $Root rev-parse --short HEAD)
if ($ok) {
  Set-Content -LiteralPath $Status -Value "ok $version" -Encoding UTF8
} else {
  Set-Content -LiteralPath $Status -Value "failed The update did not finish; see update.log in the data folder." -Encoding UTF8
}

if (Test-Path -LiteralPath $Shortcut -PathType Leaf) {
  Start-Process -FilePath $Shortcut
} else {
  Start-Process -FilePath (Join-Path $VenvDir 'Scripts\pythonw.exe') -WorkingDirectory $Root `
      -ArgumentList @("`"$(Join-Path $Root 'win_launch.py')`"", '--data-dir', "`"$DataDir`"")
}

<#
.SYNOPSIS
  Update Sotto from GitHub once it has quit, then start it again.

.DESCRIPTION
  Started hidden by the tray's "Check for updates..." (it can also be run by
  hand). Windows keeps the files of a running Python in use, so this waits
  until no process runs from Sotto's environment. Then, like
  install-mac.sh --update, it fetches the update and installs the new
  version's pinned packages (the CPU or CUDA set recorded at install) while
  this copy's source stays as it was; pip has no rollback, so a failed install
  puts the previously installed packages back and removes any it added. Only
  after the packages are in does it fast-forward the source and re-run
  install-windows.ps1 -SkipSetup for the shortcut and the install record.

  Sotto is started again either way: the new version, or the unchanged
  previous one, whose tray then reports the failure. The outcome goes to
  update-status.txt in the data folder, which the tray reports once at its
  next start; the full output is in update.log.

  get.ps1 runs the update's own copy of this script on an existing copy:
  -SourceDir names that copy, -NoRestart leaves starting Sotto to get.ps1
  and -ShortcutDir is its Start Menu folder.
#>
param(
  [string]$DataDir = '',
  [string]$VenvDir = '',
  [int]$WaitSeconds = 120,
  [string]$SourceDir = '',
  [switch]$NoRestart,
  [string]$ShortcutDir = ''
)

Set-StrictMode -Version 2
$ErrorActionPreference = 'Continue'

$Root = if ($SourceDir) { [IO.Path]::GetFullPath($SourceDir) } else { Split-Path -Parent $PSScriptRoot }
if (-not $ShortcutDir) { $ShortcutDir = [Environment]::GetFolderPath('Programs') }
if (-not $DataDir) { $DataDir = Join-Path $env:LOCALAPPDATA 'sotto-alpha' }
if (-not $VenvDir) { $VenvDir = Join-Path $Root 'venv-alpha' }
$Log = Join-Path $DataDir 'update.log'
$Status = Join-Path $DataDir 'update-status.txt'
$Shortcut = Join-Path $ShortcutDir 'Sotto.lnk'
$Launcher = Join-Path $Root 'win_launch.py'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$Manifest = Join-Path $VenvDir 'sotto-install.json'
$Next = $null  # the new version's package lists, in a temporary folder
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
Set-Content -LiteralPath $Log -Value "update started $(Get-Date -Format s)" -Encoding UTF8

function Say([string]$Text) { Add-Content -LiteralPath $Log -Value $Text -Encoding UTF8 }
# Native output goes into the UTF-8 log line by line (`*>>` in Windows
# PowerShell 5.1 would append UTF-16 to it); $LASTEXITCODE stays the command's.
function Logged([scriptblock]$Block) {
  & $Block 2>&1 | ForEach-Object { "$_" } | Add-Content -LiteralPath $Log -Encoding UTF8
}

function Running-From-Venv {
  $prefix = [IO.Path]::GetFullPath($VenvDir).TrimEnd('\') + '\'
  return @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
      $path = $null
      try { $path = $_.Path } catch { }
      $path -and $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
    })
}

# Start this copy again: through the Start Menu shortcut only when it starts
# this copy (it can belong to another one), else with this copy's launcher.
function Start-Sotto {
  if (Test-Path -LiteralPath $Shortcut -PathType Leaf) {
    $link = (New-Object -ComObject WScript.Shell).CreateShortcut($Shortcut)
    if ($link.Arguments -and $link.Arguments.IndexOf($Launcher, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
      Start-Process -FilePath $Shortcut
      return
    }
  }
  Start-Process -FilePath (Join-Path $VenvDir 'Scripts\pythonw.exe') -WorkingDirectory $Root `
      -ArgumentList @("`"$Launcher`"", '--data-dir', "`"$DataDir`"")
}

function Finish([bool]$Ok, [string]$Text) {
  if ($Next) { Remove-Item -LiteralPath $Next -Recurse -Force -ErrorAction SilentlyContinue }
  if ($Ok) {
    Say "OK updated to $Text"
    Set-Content -LiteralPath $Status -Value "ok $Text" -Encoding UTF8
  } else {
    Say "X $Text"
    Set-Content -LiteralPath $Status -Value "failed $Text See update.log in the data folder." -Encoding UTF8
  }
  if ($NoRestart) {
    # Run from get.ps1 in a console: say it there too.
    if ($Ok) { Write-Host "OK updated to $Text" } else { Write-Host "X $Text (details: $Log)" -ForegroundColor Red }
  } else {
    Start-Sotto
  }
  if ($Ok) { exit 0 }
  exit 1
}

# Installed distribution names, normalized (PEP 503); $null when pip fails.
function Package-Names {
  $listed = & $VenvPython -m pip list --disable-pip-version-check --format=json 2>$null
  if ($LASTEXITCODE -ne 0) { return $null }
  try { $names = @(("$listed" | ConvertFrom-Json) | ForEach-Object { ($_.name -replace '[-_.]+', '-').ToLowerInvariant() }) }
  catch { return $null }
  return ,$names
}

# pip cannot undo a half-finished install: put back the set recorded before it
# (pip install -r only adds), remove what the update added, then check it.
function Restore-Packages([string]$Previous) {
  Say '== restoring the previous packages'
  if (@(Get-Content -LiteralPath $Previous).Count -gt 0) {
    Logged { & $VenvPython -m pip install --disable-pip-version-check -r $Previous }
    if ($LASTEXITCODE -ne 0) { return $false }
  }
  $now = Package-Names
  if ($null -eq $now) { return $false }
  $added = @($now | Where-Object { $PreviousNames -notcontains $_ -and @('pip', 'setuptools', 'wheel') -notcontains $_ })
  if ($added.Count -gt 0) {
    Say "== removing what the update added: $($added -join ' ')"
    Logged { & $VenvPython -m pip uninstall --disable-pip-version-check --yes @added }
    if ($LASTEXITCODE -ne 0) { return $false }
  }
  Logged { & $VenvPython -m pip check --disable-pip-version-check }
  return ($LASTEXITCODE -eq 0)
}

$deadline = (Get-Date).AddSeconds($WaitSeconds)
while (@(Running-From-Venv).Count -gt 0 -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 300 }
if (@(Running-From-Venv).Count -gt 0) {
  Say 'Sotto did not quit; nothing was changed.'
  Set-Content -LiteralPath $Status -Value 'failed Sotto did not quit, so nothing was changed.' -Encoding UTF8
  exit 1
}

# ---------------------------------------------------------------- what to install
Logged { git -C $Root fetch --quiet }
if ($LASTEXITCODE -ne 0) { Finish $false 'Could not download the update (git fetch failed); nothing was changed.' }
$upstream = "$(git -C $Root rev-parse '@{u}' 2>$null)".Trim()
if ($LASTEXITCODE -ne 0 -or -not $upstream) { Finish $false 'This copy does not follow a GitHub branch; nothing was changed.' }
Logged { git -C $Root merge-base --is-ancestor HEAD $upstream }
if ($LASTEXITCODE -ne 0) { Finish $false 'This copy has its own commits, so it cannot simply move forward; nothing was changed.' }
$changes = @(git -C $Root status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0 -or $changes.Count -gt 0) { Finish $false 'This copy has local changes; nothing was changed.' }
$version = "$(git -C $Root rev-parse --short $upstream)".Trim()

$flavor = ''
if (Test-Path -LiteralPath $Manifest -PathType Leaf) {
  try {
    $record = Get-Content -LiteralPath $Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($record.PSObject.Properties['flavor'] -and @('cpu', 'cuda') -contains $record.flavor) { $flavor = $record.flavor }
  } catch { }
}
if (-not $flavor) {
  # No install record: keep the set that is there.
  & $VenvPython -m pip show --disable-pip-version-check nvidia-cublas-cu12 *> $null
  $flavor = if ($LASTEXITCODE -eq 0) { 'cuda' } else { 'cpu' }
}
if ($flavor -eq 'cuda') { $Requirements = 'requirements-alpha-windows-cuda.txt'; $FlavorSwitch = '-Cuda' }
else { $Requirements = 'requirements-alpha-windows.txt'; $FlavorSwitch = '-Cpu' }
Say "== $version, dependency set: $flavor"

$Next = Join-Path ([IO.Path]::GetTempPath()) "sotto-update-$PID"
Remove-Item -LiteralPath $Next -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $Next | Out-Null
$lists = @('requirements-alpha-windows.txt', 'requirements-alpha-windows-cuda.txt',
           'constraints-alpha-windows.txt', 'constraints-alpha-windows-cuda.txt')
$archive = Join-Path $Next 'next.tar'
Logged { git -C $Root archive --format=tar -o $archive $upstream @lists }
if ($LASTEXITCODE -eq 0) { Logged { & (Join-Path $env:SystemRoot 'System32\tar.exe') -xf $archive -C $Next } }
if (@($lists | Where-Object { -not (Test-Path -LiteralPath (Join-Path $Next $_) -PathType Leaf) }).Count -gt 0) {
  Finish $false "Could not read the update's package lists; nothing was changed."
}

# ---------------------------------------------------------------- packages first
$previous = Join-Path $Next 'previous-packages.txt'
$frozen = @(& $VenvPython -m pip freeze --disable-pip-version-check 2>$null)
if ($LASTEXITCODE -ne 0) { Finish $false 'Could not list the installed packages (pip freeze failed); nothing was changed.' }
[IO.File]::WriteAllLines($previous, [string[]]$frozen)
$PreviousNames = Package-Names
if ($null -eq $PreviousNames) { Finish $false 'Could not list the installed packages (pip list failed); nothing was changed.' }

Say "== packages ($Requirements)"
Logged { & $VenvPython -m pip install --disable-pip-version-check -r (Join-Path $Next $Requirements) }
$installed = ($LASTEXITCODE -eq 0)
if ($installed) {
  Logged { & $VenvPython -m pip check --disable-pip-version-check }
  $installed = ($LASTEXITCODE -eq 0)
}
if (-not $installed) {
  if (Restore-Packages $previous) {
    Finish $false "Installing the update's packages failed; the previous packages are back and this copy was not changed. Check the network and update again."
  }
  Finish $false "Installing the update's packages failed and the previous ones could not be put back; this copy's source was not changed. Run scripts\install-windows.ps1 again (it needs the network)."
}

# ---------------------------------------------------------------- then the source
Say '== source'
Logged { git -C $Root merge --ff-only --quiet $upstream }
if ($LASTEXITCODE -ne 0) {
  if (Restore-Packages $previous) {
    Finish $false 'git could not move this copy forward; the source was not switched and the previous packages are back.'
  }
  Finish $false "git could not move this copy forward and the previous packages could not be put back. Run scripts\install-windows.ps1 again (it needs the network)."
}

# Shortcut and install record, keeping the recorded dependency set.
Logged { & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root 'scripts\install-windows.ps1') `
    -SkipSetup $FlavorSwitch -DataDir $DataDir -VenvDir $VenvDir -ShortcutDir $ShortcutDir }
if ($LASTEXITCODE -ne 0) { Finish $false "Sotto was updated to $version, but its setup did not finish." }
Finish $true $version

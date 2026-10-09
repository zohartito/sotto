<#
.SYNOPSIS
  Install (or remove) the Sotto Windows source alpha for the current user.

.DESCRIPTION
  Checks for Windows x64 and Python 3.13 (64-bit), creates venv-alpha in this
  folder, installs the pinned CPU or CUDA dependency set (CUDA when
  nvidia-smi sees a GPU; -Cpu / -Cuda override), prepares the alpha data
  folder, downloads the pinned speech model and voice detection into it
  (sotto_win.py setup; -SkipSetup leaves that to the first launch), and adds a
  Start Menu shortcut "Sotto" that starts the tray app without a console
  window, with SOTTO_DATA_DIR applied.

  Nothing is installed system-wide: no admin rights, services or scheduled
  tasks. Python itself is never installed for you; the script says exactly
  what to install instead. Re-running is safe. -Uninstall removes only what
  this script recorded creating (and the login entry the Settings window may
  have added); your data folder stays unless you add -RemoveData.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install-windows.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install-windows.ps1 -WhatIf
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install-windows.ps1 -Uninstall
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
  [switch]$Cpu,
  [switch]$Cuda,
  [string]$DataDir = '',
  [string]$Python = '',
  [string]$VenvDir = '',
  [string]$ShortcutDir = '',
  [switch]$SkipSetup,
  [switch]$Uninstall,
  [switch]$RemoveData,
  # Tests point this at a throwaway key; the default is the per-user login list.
  [string]$RunKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
)

Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$Launcher = Join-Path $Root 'win_launch.py'
if (-not $DataDir) { $DataDir = Join-Path $env:LOCALAPPDATA 'sotto-alpha' }
if (-not $VenvDir) { $VenvDir = Join-Path $Root 'venv-alpha' }
if (-not $ShortcutDir) { $ShortcutDir = [Environment]::GetFolderPath('Programs') }
$DataDir = [IO.Path]::GetFullPath($DataDir)
$VenvDir = [IO.Path]::GetFullPath($VenvDir)
$Shortcut = Join-Path $ShortcutDir 'Sotto.lnk'
$Manifest = Join-Path $VenvDir 'sotto-install.json'
$LegacyData = Join-Path $env:APPDATA 'sotto'

function Say([string]$Text) { Write-Host $Text }
# Windows PowerShell 5.1 turns redirected native stderr into terminating
# errors under 'Stop'; probes that may fail run through this instead.
function Quiet([scriptblock]$Block) {
  $saved = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { & $Block 2>$null } finally { $ErrorActionPreference = $saved }
}
function Fail([string]$Text) { Write-Host ''; Write-Host "X $Text" -ForegroundColor Red; exit 1 }
# Long native steps (venv, pip): their warnings on stderr must stay warnings.
function Native([scriptblock]$Block) {
  $saved = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { & $Block } finally { $ErrorActionPreference = $saved }
}

function Read-Manifest {
  if (Test-Path -LiteralPath $Manifest -PathType Leaf) {
    return Get-Content -LiteralPath $Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
  }
  return $null
}

function Write-Manifest([bool]$CreatedVenv, [bool]$CreatedData, [string]$Flavor, [string]$Python) {
  if ($PSCmdlet.ShouldProcess($Manifest, 'Record what this script created')) {
    [ordered]@{
      schema = 1; created_venv = $CreatedVenv; created_data = $CreatedData; data_dir = $DataDir
      shortcut = $Shortcut; flavor = $Flavor; python = $Python
    } | ConvertTo-Json | Set-Content -LiteralPath $Manifest -Encoding UTF8
  }
}

function Points-Here([string]$Command) {
  return $Command -and $Command.IndexOf($Launcher, [StringComparison]::OrdinalIgnoreCase) -ge 0
}

function Running-From-Venv {
  $prefix = $VenvDir.TrimEnd('\') + '\'
  return @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
      $path = $null
      try { $path = $_.Path } catch { }
      $path -and $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
    })
}

# ---------------------------------------------------------------- uninstall
if ($Uninstall) {
  $record = Read-Manifest
  $running = @(Running-From-Venv)  # an empty result would otherwise unroll to $null
  if ($running.Count -gt 0) {
    Fail "Sotto is still running from $VenvDir (process $($running[0].Id)). Quit it from its tray menu, then run -Uninstall again."
  }
  if (Test-Path -LiteralPath $Shortcut -PathType Leaf) {
    $link = (New-Object -ComObject WScript.Shell).CreateShortcut($Shortcut)
    if (Points-Here $link.Arguments) {
      if ($PSCmdlet.ShouldProcess($Shortcut, 'Remove Start Menu shortcut')) {
        Remove-Item -LiteralPath $Shortcut -Force
        Say "removed Start Menu shortcut $Shortcut"
      }
    } else { Say "kept $Shortcut (it does not start this copy of Sotto)" }
  }
  $run = Get-ItemProperty -Path $RunKey -Name 'Sotto' -ErrorAction SilentlyContinue
  if ($run -and (Points-Here $run.Sotto)) {
    if ($PSCmdlet.ShouldProcess("$RunKey\Sotto", 'Remove launch-at-login entry')) {
      Remove-ItemProperty -Path $RunKey -Name 'Sotto'
      Say 'removed the launch-at-login entry'
    }
  }
  $isVenv = Test-Path -LiteralPath (Join-Path $VenvDir 'pyvenv.cfg') -PathType Leaf
  if ($record -and $record.created_venv -and $isVenv) {
    if ($PSCmdlet.ShouldProcess($VenvDir, 'Remove the Python environment')) {
      Remove-Item -LiteralPath $VenvDir -Recurse -Force
      Say "removed $VenvDir"
    }
  } elseif (Test-Path -LiteralPath $VenvDir) {
    Say "kept $VenvDir (this script did not create it)"
  }
  if ($RemoveData) {
    if (-not ($record -and $record.created_data -and $record.data_dir -eq $DataDir)) {
      Say "kept $DataDir (this script did not create it; delete it yourself if you want)"
    } elseif ($DataDir -eq $LegacyData) {
      Say "kept $DataDir (never removed automatically)"
    } elseif ((Test-Path -LiteralPath $DataDir) -and $PSCmdlet.ShouldProcess($DataDir, 'Remove Sotto data: history, recordings, models, dictionary')) {
      Remove-Item -LiteralPath $DataDir -Recurse -Force
      Say "removed $DataDir"
    }
  } elseif (Test-Path -LiteralPath $DataDir) {
    Say "kept your data in $DataDir (history, recordings, models, dictionary); add -RemoveData to delete it"
  }
  Say 'uninstall finished'
  exit 0
}

# ---------------------------------------------------------------- checks
if ($Cpu -and $Cuda) { Fail 'Choose at most one of -Cpu and -Cuda.' }
$arch = $env:PROCESSOR_ARCHITEW6432
if (-not $arch) { $arch = $env:PROCESSOR_ARCHITECTURE }
if (-not [Environment]::Is64BitOperatingSystem -or $arch -ne 'AMD64') {
  Fail "Sotto's Windows alpha supports Windows x64 only (this PC reports $arch)."
}
Say "OK Windows x64 ($([Environment]::OSVersion.VersionString))"

# No double quotes inside: Windows PowerShell 5.1 does not escape them for native commands.
$probe = "import struct, sys; print(sys.executable); print('%d.%d.%d' % sys.version_info[:3]); print(struct.calcsize('P') * 8)"
function Probe-Python([string[]]$Command) {
  try {
    $exe = $Command[0]
    $rest = @()
    if ($Command.Count -gt 1) { $rest = $Command[1..($Command.Count - 1)] }
    $out = Quiet { & $exe @rest -c $probe }
    if ($LASTEXITCODE -ne 0 -or -not $out -or @($out).Count -lt 3) { return $null }
    return @{ Exe = $out[0]; Version = $out[1]; Bits = $out[2] }
  } catch { return $null }
}
if ($Python) { $found = Probe-Python @($Python) } else { $found = Probe-Python @('py', '-3.13') }
if (-not $found -and -not $Python) { $found = Probe-Python @('python') }
$instructions = "Install Python 3.13 (64-bit) from https://www.python.org/downloads/windows/ " +
  "(the 'Windows installer (64-bit)'), keep 'py launcher' and 'tcl/tk and IDLE' ticked, " +
  "then run this script again. Or pass -Python C:\path\to\python.exe."
if (-not $found) { Fail "Python 3.13 was not found. $instructions" }
if (-not $found.Version.StartsWith('3.13.') -or $found.Bits -ne '64') {
  Fail "Found Python $($found.Version) ($($found.Bits)-bit) at $($found.Exe); Sotto needs 3.13 64-bit. $instructions"
}
$BasePython = $found.Exe
Say "OK Python $($found.Version) 64-bit ($BasePython)"
Quiet { & $BasePython -c 'import tkinter' } | Out-Null
if ($LASTEXITCODE -ne 0) {
  Say "! This Python has no tkinter: the tray works, but Settings and Correct need it. Re-run the python.org installer, choose Modify, tick 'tcl/tk and IDLE'."
}

if ($Cuda) { $flavor = 'cuda'; $why = '-Cuda' }
elseif ($Cpu) { $flavor = 'cpu'; $why = '-Cpu' }
else {
  $flavor = 'cpu'; $why = 'no NVIDIA GPU found'
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpus = @(Quiet { & nvidia-smi -L } | Where-Object { $_ -like 'GPU *' })
    if ($LASTEXITCODE -eq 0 -and $gpus.Count -gt 0) { $flavor = 'cuda'; $why = "nvidia-smi: $($gpus[0])" }
  }
}
if ($flavor -eq 'cuda') { $Requirements = Join-Path $Root 'requirements-alpha-windows-cuda.txt' }
else { $Requirements = Join-Path $Root 'requirements-alpha-windows.txt' }
Say "OK dependency set: $flavor ($why)"

# ---------------------------------------------------------------- install
$record = Read-Manifest
$createdVenv = [bool]($record -and $record.created_venv)
$createdData = [bool]($record -and $record.created_data -and $record.data_dir -eq $DataDir)
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$VenvPythonW = Join-Path $VenvDir 'Scripts\pythonw.exe'

if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
  $existing = Probe-Python @($VenvPython)
  if (-not $existing -or -not $existing.Version.StartsWith('3.13.')) {
    Fail "$VenvDir holds a different Python; remove it (or pass -VenvDir) and run again."
  }
  Say "OK reusing $VenvDir"
} elseif ((Test-Path -LiteralPath $VenvDir) -and @(Get-ChildItem -LiteralPath $VenvDir -Force).Count -gt 0) {
  # Never adopt (and later delete) a folder this script did not make.
  Fail "$VenvDir already exists but is not a Windows Python environment; pass -VenvDir with a new folder."
} elseif ($PSCmdlet.ShouldProcess($VenvDir, 'Create Python environment')) {
  Native { & $BasePython -m venv $VenvDir }
  if ($LASTEXITCODE -ne 0) { Fail "Could not create $VenvDir." }
  $createdVenv = $true
  Write-Manifest $createdVenv $createdData $flavor $BasePython  # ownership survives a failed pip run
  Say "OK created $VenvDir"
}

if ($PSCmdlet.ShouldProcess($VenvDir, "Install pinned packages from $(Split-Path -Leaf $Requirements)")) {
  # An exact pip, not whichever one this Python shipped (F25); tests/test_dependency_pins.py keeps every copy in step.
  Native { & $VenvPython -m pip install --disable-pip-version-check pip==26.2.1 }
  if ($LASTEXITCODE -ne 0) { Fail 'Could not install pip 26.2.1 (see above); check the network and run again.' }
  Native { & $VenvPython -m pip install --disable-pip-version-check -r $Requirements }
  if ($LASTEXITCODE -ne 0) { Fail 'pip could not install the pinned packages (see above); check the network and run again.' }
  Native { & $VenvPython -m pip check --disable-pip-version-check }
  if ($LASTEXITCODE -ne 0) { Fail 'pip check found broken requirements (see above).' }
  Say 'OK packages installed and consistent'
}

if (Test-Path -LiteralPath $DataDir) {
  Say "OK data folder $DataDir"
} elseif ($PSCmdlet.ShouldProcess($DataDir, 'Create the alpha data folder')) {
  New-Item -ItemType Directory -Path $DataDir | Out-Null
  $createdData = $true
  Say "OK created data folder $DataDir"
}

if (-not $SkipSetup -and $PSCmdlet.ShouldProcess($DataDir, 'Download the pinned speech model (about 1.6 GB, once) and voice detection')) {
  # Set only for the setup child and put back afterwards: a script run from an
  # interactive shell must not leave them in that shell.
  $savedData, $savedHf = $env:SOTTO_DATA_DIR, $env:SOTTO_HF_HOME
  try {
    $env:SOTTO_DATA_DIR = $DataDir
    $env:SOTTO_HF_HOME = Join-Path $DataDir 'huggingface'
    Native { & $VenvPython (Join-Path $Root 'sotto_win.py') setup }
    $setupExit = $LASTEXITCODE
  } finally {
    $env:SOTTO_DATA_DIR, $env:SOTTO_HF_HOME = $savedData, $savedHf
  }
  if ($setupExit -ne 0) {
    Say '! The model download did not finish; Sotto downloads it on first launch (or run this script again).'
  } elseif (Test-Path -LiteralPath (Join-Path $DataDir 'models\silero_vad.onnx') -PathType Leaf) {
    Say 'OK speech model and voice detection ready'
  } else {
    Say '! Speech model ready, but voice detection was not installed; run this script again later.'
  }
}

if ($PSCmdlet.ShouldProcess($Shortcut, 'Create Start Menu shortcut "Sotto" (tray app, no console)')) {
  if (-not (Test-Path -LiteralPath $ShortcutDir)) { New-Item -ItemType Directory -Path $ShortcutDir | Out-Null }
  $link = (New-Object -ComObject WScript.Shell).CreateShortcut($Shortcut)
  $link.TargetPath = $VenvPythonW
  $link.Arguments = "`"$Launcher`" --data-dir `"$DataDir`""
  $link.WorkingDirectory = $Root
  $link.Description = 'Sotto: local push-to-talk dictation'
  $link.Save()
  Say "OK Start Menu shortcut $Shortcut"
}

Write-Manifest $createdVenv $createdData $flavor $BasePython

Say ''
Say 'Sotto is installed. Start it from the Start Menu: Sotto (a tray icon appears, then a'
Say '"Ready" notification; if the model was not downloaded above, the first launch does it).'
Say 'Settings (tray menu) can turn on launch at login.'
Say 'Console run and checks, in PowerShell:'
Say "  `$env:SOTTO_DATA_DIR = `"$DataDir`"; `$env:SOTTO_HF_HOME = `"$DataDir\huggingface`""
Say "  & `"$VenvPython`" `"$(Join-Path $Root 'sotto_win.py')`" doctor"
Say "  & `"$VenvPython`" `"$(Join-Path $Root 'sotto_win.py')`""
Say "Remove: powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Uninstall"
exit 0

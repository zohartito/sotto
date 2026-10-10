# Install Sotto on Windows with one line (in PowerShell):
#
#   irm https://raw.githubusercontent.com/zohartito/sotto/main/scripts/get.ps1 | iex
#
# Downloads Sotto with git into %USERPROFILE%\sotto (or $env:SOTTO_SOURCE), or
# updates that copy (the new packages first, the source only after them), then
# runs its scripts\install-windows.ps1 and starts Sotto.
# No admin rights. Missing tools are named with the command that installs them.

# A folder has a long name and, on most volumes, an 8.3 short one
# (C:\Users\RUNNER~1); a running process can use either. Paths are
# compared by their long names: Windows PowerShell's GetFullPath expands only
# a path that exists and PowerShell 7's never does, so the part that exists
# goes through GetLongPathName.
if (-not ('SottoPaths.Native' -as [type])) {
  Add-Type -Namespace SottoPaths -Name Native -MemberDefinition @'
[DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
public static extern uint GetLongPathNameW(string shortPath, System.Text.StringBuilder longPath, uint size);
'@
}
function Long-Path([string]$Path) {
  $full = [IO.Path]::GetFullPath($Path)
  if ($full.Length -gt 3) { $full = $full.TrimEnd('\') }
  $existing, $rest = $full, ''
  while ($existing -and -not ([IO.Directory]::Exists($existing) -or [IO.File]::Exists($existing))) {
    $rest = '\' + [IO.Path]::GetFileName($existing) + $rest
    $existing = [IO.Path]::GetDirectoryName($existing)
  }
  $buffer = New-Object Text.StringBuilder 32768
  if (-not $existing -or [SottoPaths.Native]::GetLongPathNameW($existing, $buffer, $buffer.Capacity) -eq 0) { return $full }
  if ($rest) { return $buffer.ToString().TrimEnd('\') + $rest }
  return $buffer.ToString()
}

# Same repository? Ignores https vs either ssh form (git@github.com:... and
# ssh://git@github.com/...), a trailing .git and a trailing slash.
function Test-SameRepo([string]$First, [string]$Second) {
  $normalized = foreach ($url in $First, $Second) {
    $https = $url -replace '^git@github\.com:', 'https://github.com/' -replace '^ssh://git@github\.com/', 'https://github.com/'
    ($https.TrimEnd('/')) -replace '\.git$', ''
  }
  return $normalized[0] -eq $normalized[1]
}

# The one-liner runs this script inside the user's shell (iex), where exit
# would close that shell: a failure is said, remembered here, and becomes an
# exit code only when the script runs as a file.
function Fail-Install([string]$Text) {
  Write-Host "X $Text" -ForegroundColor Red
  $script:SottoInstallFailed = $true
}

function Install-Sotto {
  $ErrorActionPreference = 'Stop'
  $repo = if ($env:SOTTO_REPO) { $env:SOTTO_REPO } else { 'https://github.com/zohartito/sotto.git' }
  $dest = if ($env:SOTTO_SOURCE) { $env:SOTTO_SOURCE } else { Join-Path $env:USERPROFILE 'sotto' }
  $shortcutDir = [Environment]::GetFolderPath('Programs')

  if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail-Install 'Sotto needs Git: winget install --id Git.Git -e  (then open a new PowerShell and run this line again)'
    return
  }
  if (Test-Path -LiteralPath (Join-Path $dest '.git')) {
    $origin = ''
    try { $origin = git -C $dest remote get-url origin 2>$null } catch { $origin = '' }
    if (-not (Test-SameRepo $origin $repo)) {
      Fail-Install "$dest is a git copy of $origin, not Sotto. Choose another folder with `$env:SOTTO_SOURCE."
      return
    }
    # Pulling and reinstalling under a running Sotto would swap its source and
    # packages while they are in use.
    $prefix = (Long-Path $dest).TrimEnd('\') + '\'
    $running = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $path = $null
        try { $path = $_.Path } catch { }
        $path -and (Long-Path $path).StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
      })
    if ($running.Count -gt 0) {
      Fail-Install "Sotto is running from $dest (process $($running[0].Id)). Quit it from its tray menu (or update it there with Check for updates), then run this line again."
      return
    }
    Write-Host "== Updating Sotto in $dest"
    git -C $dest fetch --quiet
    if ($LASTEXITCODE -ne 0) { Fail-Install "Could not download the update into $dest."; return }
    # Ask the update's own installer whether it will run here (its Start Menu
    # check) before the source moves: a refusal after the pull would leave new
    # source on old packages.
    $check = Join-Path ([IO.Path]::GetTempPath()) "sotto-get-$PID"
    Remove-Item -LiteralPath $check -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $check | Out-Null
    try {
      $archive = Join-Path $check 'installer.tar'
      git -C $dest archive --format=tar -o $archive '@{u}' scripts/install-windows.ps1 scripts/update-windows.ps1
      if ($LASTEXITCODE -eq 0) { & (Join-Path $env:SystemRoot 'System32\tar.exe') -xf $archive -C $check }
      $installer = Join-Path $check 'scripts\install-windows.ps1'
      $updater = Join-Path $check 'scripts\update-windows.ps1'
      if (-not (Test-Path -LiteralPath $installer -PathType Leaf) -or -not (Test-Path -LiteralPath $updater -PathType Leaf)) {
        Fail-Install "Could not read the update's installer; $dest was not changed."; return
      }
      & powershell -NoProfile -ExecutionPolicy Bypass -File $installer -CheckOnly -SourceDir $dest -ShortcutDir $shortcutDir
      if ($LASTEXITCODE -ne 0) { Fail-Install "$dest was not updated; see the message above."; return }
      $behind = "$(git -C $dest rev-parse HEAD)".Trim() -ne "$(git -C $dest rev-parse '@{u}')".Trim()
      if ($behind -and (Test-Path -LiteralPath (Join-Path $dest 'venv-alpha\Scripts\python.exe') -PathType Leaf)) {
        # An installed copy: the update's own updater installs the new
        # packages first, moves the source only after them, and puts the
        # previous packages back if anything fails.
        & powershell -NoProfile -ExecutionPolicy Bypass -File $updater -SourceDir $dest -NoRestart `
          -WaitSeconds 0 -ShortcutDir $shortcutDir
        if ($LASTEXITCODE -ne 0) { Fail-Install "$dest was not updated; see the message above."; return }
      }
    } finally {
      Remove-Item -LiteralPath $check -Recurse -Force -ErrorAction SilentlyContinue
    }
    # Already moved by the updater, or nothing installed yet: no packages to keep in step.
    git -C $dest merge --ff-only --quiet '@{u}'
    if ($LASTEXITCODE -ne 0) { Fail-Install "Could not update $dest (local changes?)."; return }
  } elseif (Test-Path -LiteralPath $dest) {
    Fail-Install "$dest exists and is not a copy of Sotto. Choose another folder with `$env:SOTTO_SOURCE."
    return
  } else {
    Write-Host "== Downloading Sotto into $dest"
    git clone --quiet $repo $dest
    if ($LASTEXITCODE -ne 0) { Fail-Install 'Could not download Sotto.'; return }
  }

  if ($env:SOTTO_GET_DRY_RUN) { "dry run: would run $dest\scripts\install-windows.ps1"; return }
  # The installer checks Windows, Python 3.13 and the GPU, and says what is missing.
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $dest 'scripts\install-windows.ps1') -ShortcutDir $shortcutDir
  if ($LASTEXITCODE -ne 0) { Fail-Install 'The installer stopped; see the messages above.'; return }
  $shortcut = Join-Path $shortcutDir 'Sotto.lnk'
  if (Test-Path -LiteralPath $shortcut) { Start-Process -FilePath $shortcut }
}

$script:SottoInstallFailed = $false
Install-Sotto
if ($script:SottoInstallFailed) {
  $global:LASTEXITCODE = 1
  # Only when run as a file (powershell -File get.ps1): a function defined from
  # text pasted through iex has no file, and there exit would close the shell.
  if (${function:Install-Sotto}.File) { exit 1 }
} else {
  $global:LASTEXITCODE = 0
}

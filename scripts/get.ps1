# Install Sotto on Windows with one line (in PowerShell):
#
#   irm https://raw.githubusercontent.com/zohartito/sotto/main/scripts/get.ps1 | iex
#
# Downloads Sotto with git into %USERPROFILE%\sotto (or $env:SOTTO_SOURCE), or
# updates that copy, then runs its scripts\install-windows.ps1 and starts Sotto.
# No admin rights. Missing tools are named with the command that installs them.

# Same repository? Ignores https vs either ssh form (git@github.com:... and
# ssh://git@github.com/...), a trailing .git and a trailing slash.
function Test-SameRepo([string]$First, [string]$Second) {
  $normalized = foreach ($url in $First, $Second) {
    $https = $url -replace '^git@github\.com:', 'https://github.com/' -replace '^ssh://git@github\.com/', 'https://github.com/'
    ($https.TrimEnd('/')) -replace '\.git$', ''
  }
  return $normalized[0] -eq $normalized[1]
}

function Install-Sotto {
  $ErrorActionPreference = 'Stop'
  $repo = if ($env:SOTTO_REPO) { $env:SOTTO_REPO } else { 'https://github.com/zohartito/sotto.git' }
  $dest = if ($env:SOTTO_SOURCE) { $env:SOTTO_SOURCE } else { Join-Path $env:USERPROFILE 'sotto' }
  $shortcutDir = [Environment]::GetFolderPath('Programs')

  if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host 'X Sotto needs Git: winget install --id Git.Git -e  (then open a new PowerShell and run this line again)' -ForegroundColor Red
    return
  }
  if (Test-Path -LiteralPath (Join-Path $dest '.git')) {
    $origin = ''
    try { $origin = git -C $dest remote get-url origin 2>$null } catch { $origin = '' }
    if (-not (Test-SameRepo $origin $repo)) {
      Write-Host "X $dest is a git copy of $origin, not Sotto. Choose another folder with `$env:SOTTO_SOURCE." -ForegroundColor Red
      return
    }
    # Pulling and reinstalling under a running Sotto would swap its source and
    # packages while they are in use.
    $prefix = [IO.Path]::GetFullPath($dest).TrimEnd('\') + '\'
    $running = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $path = $null
        try { $path = $_.Path } catch { }
        $path -and $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
      })
    if ($running.Count -gt 0) {
      Write-Host "X Sotto is running from $dest (process $($running[0].Id)). Quit it from its tray menu (or update it there with Check for updates), then run this line again." -ForegroundColor Red
      return
    }
    Write-Host "== Updating Sotto in $dest"
    git -C $dest fetch --quiet
    if ($LASTEXITCODE -ne 0) { Write-Host "X Could not download the update into $dest." -ForegroundColor Red; return }
    # Ask the update's own installer whether it will run here (its Start Menu
    # check) before the source moves: a refusal after the pull would leave new
    # source on old packages.
    $check = Join-Path ([IO.Path]::GetTempPath()) "sotto-get-$PID"
    Remove-Item -LiteralPath $check -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $check | Out-Null
    try {
      $archive = Join-Path $check 'installer.tar'
      git -C $dest archive --format=tar -o $archive '@{u}' scripts/install-windows.ps1
      if ($LASTEXITCODE -eq 0) { & (Join-Path $env:SystemRoot 'System32\tar.exe') -xf $archive -C $check }
      $installer = Join-Path $check 'scripts\install-windows.ps1'
      if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
        Write-Host "X Could not read the update's installer; $dest was not changed." -ForegroundColor Red; return
      }
      & powershell -NoProfile -ExecutionPolicy Bypass -File $installer -CheckOnly -SourceDir $dest -ShortcutDir $shortcutDir
      if ($LASTEXITCODE -ne 0) { Write-Host "X $dest was not updated; see the message above." -ForegroundColor Red; return }
    } finally {
      Remove-Item -LiteralPath $check -Recurse -Force -ErrorAction SilentlyContinue
    }
    git -C $dest merge --ff-only --quiet '@{u}'
    if ($LASTEXITCODE -ne 0) { Write-Host "X Could not update $dest (local changes?)." -ForegroundColor Red; return }
  } elseif (Test-Path -LiteralPath $dest) {
    Write-Host "X $dest exists and is not a copy of Sotto. Choose another folder with `$env:SOTTO_SOURCE." -ForegroundColor Red
    return
  } else {
    Write-Host "== Downloading Sotto into $dest"
    git clone --quiet $repo $dest
    if ($LASTEXITCODE -ne 0) { Write-Host 'X Could not download Sotto.' -ForegroundColor Red; return }
  }

  if ($env:SOTTO_GET_DRY_RUN) { "dry run: would run $dest\scripts\install-windows.ps1"; return }
  # The installer checks Windows, Python 3.13 and the GPU, and says what is missing.
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $dest 'scripts\install-windows.ps1') -ShortcutDir $shortcutDir
  if ($LASTEXITCODE -ne 0) { Write-Host 'X The installer stopped; see the messages above.' -ForegroundColor Red; return }
  $shortcut = Join-Path $shortcutDir 'Sotto.lnk'
  if (Test-Path -LiteralPath $shortcut) { Start-Process -FilePath $shortcut }
}

Install-Sotto

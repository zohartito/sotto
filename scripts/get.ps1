# Install Sotto on Windows with one line (in PowerShell):
#
#   irm https://raw.githubusercontent.com/zohartito/sotto/main/scripts/get.ps1 | iex
#
# Downloads Sotto with git into %USERPROFILE%\sotto (or $env:SOTTO_SOURCE), or
# updates that copy, then runs its scripts\install-windows.ps1 and starts Sotto.
# No admin rights. Missing tools are named with the command that installs them.

# Same repository? Ignores https vs ssh form, a trailing .git and a trailing slash.
function Test-SameRepo([string]$First, [string]$Second) {
  $normalized = foreach ($url in $First, $Second) {
    (($url -replace '^git@github\.com:', 'https://github.com/').TrimEnd('/')) -replace '\.git$', ''
  }
  return $normalized[0] -eq $normalized[1]
}

function Install-Sotto {
  $ErrorActionPreference = 'Stop'
  $repo = if ($env:SOTTO_REPO) { $env:SOTTO_REPO } else { 'https://github.com/zohartito/sotto.git' }
  $dest = if ($env:SOTTO_SOURCE) { $env:SOTTO_SOURCE } else { Join-Path $env:USERPROFILE 'sotto' }

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
    Write-Host "== Updating Sotto in $dest"
    git -C $dest pull --ff-only
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
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $dest 'scripts\install-windows.ps1')
  if ($LASTEXITCODE -ne 0) { Write-Host 'X The installer stopped; see the messages above.' -ForegroundColor Red; return }
  $shortcut = Join-Path ([Environment]::GetFolderPath('Programs')) 'Sotto.lnk'
  if (Test-Path -LiteralPath $shortcut) { Start-Process -FilePath $shortcut }
}

Install-Sotto

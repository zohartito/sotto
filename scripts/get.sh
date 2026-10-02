#!/bin/bash
# Install Sotto on a Mac with one line:
#
#   curl -fsSL https://raw.githubusercontent.com/zohartito/sotto/main/scripts/get.sh | bash
#
# Downloads Sotto with git into ~/sotto (or $SOTTO_SOURCE), or updates that
# copy, then runs its scripts/install-mac.sh and opens Sotto. Nothing needs
# sudo. Missing tools are named with the command that installs them.
# Everything sits inside main(), so a partly downloaded script never runs.
set -euo pipefail

main() {
    local repo="${SOTTO_REPO:-https://github.com/zohartito/sotto.git}"
    local dest="${SOTTO_SOURCE:-$HOME/sotto}"
    local python="${SOTTO_PYTHON:-}"

    fail() { printf '✗ %s\n' "$*" >&2; exit 1; }

    [ "$(uname -s)" = Darwin ] || fail "This installer is for macOS. On Windows, see the README."
    [ "$(uname -m)" = arm64 ] || fail "Sotto needs an Apple Silicon Mac."
    if ! xcrun --find clang >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
        fail "First install Apple's command line tools: xcode-select --install  (then run this line again)"
    fi
    if [ -z "$python" ]; then
        for candidate in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
            if command -v "$candidate" >/dev/null 2>&1; then python="$(command -v "$candidate")"; break; fi
        done
    fi
    if [ -z "$python" ]; then
        if command -v brew >/dev/null 2>&1; then
            fail "Sotto needs Python 3.12: brew install python@3.12  (then run this line again)"
        fi
        fail "Sotto needs Python 3.12: install it from https://www.python.org/downloads/ (then run this line again)"
    fi

    if [ -d "$dest/.git" ]; then
        echo "== Updating Sotto in $dest"
        git -C "$dest" pull --ff-only < /dev/null || fail "Could not update $dest (local changes?)."
    elif [ -e "$dest" ]; then
        fail "$dest exists and is not a copy of Sotto. Choose another folder with SOTTO_SOURCE=..."
    else
        echo "== Downloading Sotto into $dest"
        git clone --quiet "$repo" "$dest" < /dev/null
    fi

    if [ -n "${SOTTO_GET_DRY_RUN:-}" ]; then
        echo "dry run: would run $dest/scripts/install-mac.sh --python $python"
        return 0
    fi
    bash "$dest/scripts/install-mac.sh" --python "$python" < /dev/null
    open -a Sotto || true
}

main "$@"

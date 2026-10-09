#!/bin/bash
# Install Sotto on a Mac with one line:
#
#   curl -fsSL https://raw.githubusercontent.com/zohartito/sotto/main/scripts/get.sh | bash
#
# Downloads Sotto with git into ~/sotto (or $SOTTO_SOURCE) and runs its
# scripts/install-mac.sh, or updates that copy with install-mac.sh --update
# (the new packages first, the source only after them), then opens Sotto. Nothing needs
# sudo. Missing tools are named with the command that installs them.
# Everything sits inside main(), so a partly downloaded script never runs.
set -euo pipefail

# Same repository? Ignores https vs either ssh form (git@github.com:… and
# ssh://git@github.com/…), a trailing .git and a trailing slash.
same_repo() {
    local normalized=()
    local url
    for url in "$1" "$2"; do
        url="${url/#git@github.com:/https://github.com/}"
        url="${url/#ssh:\/\/git@github.com\//https://github.com/}"
        url="${url%/}"
        normalized+=("${url%.git}")
    done
    [ "${normalized[0]}" = "${normalized[1]}" ]
}

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
        # The first native one: an Intel Homebrew's python3.12 may come first on PATH.
        local candidate found
        for candidate in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
            found="$(command -v "$candidate" 2>/dev/null)" || continue
            if [ "$("$found" -c 'import platform; print(platform.machine())' 2>/dev/null)" = arm64 ]; then
                python="$found"
                break
            fi
        done
    fi
    if [ -z "$python" ]; then
        if command -v brew >/dev/null 2>&1; then
            fail "Sotto needs Python 3.12: brew install python@3.12  (then run this line again)"
        fi
        fail "Sotto needs Python 3.12: install it from https://www.python.org/downloads/ (then run this line again)"
    fi

    local update=""
    if [ -d "$dest/.git" ]; then
        local origin
        origin="$(git -C "$dest" remote get-url origin 2>/dev/null || true)"
        same_repo "$origin" "$repo" \
            || fail "$dest is a git copy of ${origin:-an unknown project}, not Sotto. Choose another folder with SOTTO_SOURCE=..."
        # Updating under a running Sotto would swap its source and packages
        # while they are in use. The app runs its sotto.py by resolved path.
        local script listing running
        script="$(cd "$dest" && pwd -P)/sotto.py"
        listing="$(ps -axww -o pid= -o command=)" || listing=""
        running="$(printf '%s\n' "$listing" | awk -v script="$script" \
            'index($0 " ", " " script " ") && !found { found = $1 } END { print found }')"
        [ -z "$running" ] || fail "Sotto is running from $dest (process $running). Quit it from its menu (or update it there with Check for Updates), then run this line again."
        echo "== Updating Sotto in $dest"
        # install-mac.sh --update fetches, installs the new packages, and moves
        # the source only after them (putting the packages back on a failure).
        update="--update"
    elif [ -e "$dest" ]; then
        fail "$dest exists and is not a copy of Sotto. Choose another folder with SOTTO_SOURCE=..."
    else
        echo "== Downloading Sotto into $dest"
        git clone --quiet "$repo" "$dest" < /dev/null
    fi

    if [ -n "${SOTTO_GET_DRY_RUN:-}" ]; then
        echo "dry run: would run $dest/scripts/install-mac.sh ${update:+$update }--python $python"
        return 0
    fi
    # shellcheck disable=SC2086  # $update is one word or nothing
    bash "$dest/scripts/install-mac.sh" $update --python "$python" < /dev/null
    open -a Sotto || true
}

main "$@"

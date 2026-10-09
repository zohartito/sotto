#!/bin/bash
# Install, update or remove Sotto.app (in /Applications, or ~/Applications when that
# is not writable) from this source folder.
#
#   scripts/install-mac.sh              install (or repair) the app
#   scripts/install-mac.sh --update     pull the latest source (git clones), then reinstall
#   scripts/install-mac.sh --uninstall  remove Sotto.app and its login item; keeps your data
#
# Options: --data-dir DIR (default ~/Library/Application Support/sotto-alpha),
#          --python PATH (a native arm64 Python 3.12).
# Nothing here needs sudo, changes privacy settings or touches another Sotto.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
DATA_DIR="${SOTTO_DATA_DIR:-$HOME/Library/Application Support/sotto-alpha}"
PYTHON="${SOTTO_PYTHON:-}"
MODE=install

while [ $# -gt 0 ]; do
    case "$1" in
        --update) MODE=update ;;
        --uninstall) MODE=uninstall ;;
        --data-dir) DATA_DIR="$2"; shift ;;
        --python) PYTHON="$2"; shift ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

fail() { echo "✗ $*" >&2; exit 1; }
VENV="$ROOT/venv-alpha"

if [ "$MODE" = uninstall ]; then
    if [ -x "$VENV/bin/python" ]; then
        "$VENV/bin/python" "$ROOT/scripts/install_app.py" --uninstall
    else
        for app in /Applications/Sotto.app "$HOME/Applications/Sotto.app"; do
            # Only ever remove a Sotto.app this installer built.
            if [ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$app/Contents/Info.plist" 2>/dev/null)" = org.sotto.alpha ]; then
                rm -r "$app"
            fi
        done
        launchctl bootout "gui/$(id -u)/org.sotto.alpha" 2>/dev/null || true
        rm -f "$HOME/Library/LaunchAgents/org.sotto.alpha.plist"
    fi
    echo "Your recordings, settings and models are still in: $DATA_DIR"
    echo "Delete that folder yourself if you want them gone."
    exit 0
fi

[ "$(uname -s)" = Darwin ] || fail "Sotto's Mac installer runs on macOS only (see docs/windows-alpha.md for Windows)."
[ "$(uname -m)" = arm64 ] || fail "Sotto needs an Apple Silicon Mac."
MACOS_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
[ "$MACOS_MAJOR" -ge 14 ] || fail "Sotto needs macOS 14 or later (this Mac runs $(sw_vers -productVersion))."
xcrun --find clang >/dev/null 2>&1 || fail "Install Apple's command line tools first: xcode-select --install"

# A Sotto running from this copy (its sotto.py on the command line) would have
# its source and packages swapped while in use. Check for Updates runs this
# from inside Sotto, names itself in SOTTO_UPDATE_FROM_PID and restarts after.
running_sotto() {
    local listing
    listing="$(ps -axww -o pid= -o command=)" || return 0
    printf '%s\n' "$listing" | awk -v script="$ROOT/sotto.py" -v self="${SOTTO_UPDATE_FROM_PID:-}" \
        '$1 != self && index($0 " ", " " script " ") && !found { found = $1 } END { print found }'
}
RUNNING="$(running_sotto)"
[ -z "$RUNNING" ] || fail "Sotto is running from this copy (process $RUNNING). Quit it from its menu (or update it there with Check for Updates), then run this again."

UPSTREAM=""
REQUIREMENTS="$ROOT/requirements-alpha.txt"
if [ "$MODE" = update ]; then
    if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        # Fetch first and install the new version's packages before switching the
        # source, so a failed download leaves this copy's code exactly as it was.
        git -C "$ROOT" fetch --quiet || fail "Could not download the update (git fetch failed); nothing was changed."
        UPSTREAM="$(git -C "$ROOT" rev-parse '@{u}')" || fail "This copy does not follow a GitHub branch."
        git -C "$ROOT" merge-base --is-ancestor HEAD "$UPSTREAM" \
            || fail "This copy has its own commits, so it cannot simply move forward; nothing was changed."
        NEXT_REQUIREMENTS="$(mktemp -d)"
        trap 'rm -rf "$NEXT_REQUIREMENTS"' EXIT
        for name in requirements-alpha.txt requirements.txt constraints-alpha.txt; do
            git -C "$ROOT" show "$UPSTREAM:$name" > "$NEXT_REQUIREMENTS/$name" \
                || fail "The update has no $name; nothing was changed."
        done
        REQUIREMENTS="$NEXT_REQUIREMENTS/requirements-alpha.txt"
    else
        fail "This folder is not a git clone, so it cannot update itself. Download the new source archive and run scripts/install-mac.sh from it."
    fi
fi

if [ -z "$PYTHON" ]; then
    for candidate in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
        if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$(command -v "$candidate")"; break; fi
    done
fi
[ -n "$PYTHON" ] || fail "Python 3.12 not found. Install it (e.g. brew install python@3.12) or pass --python PATH."
[ "$("$PYTHON" -c 'import platform; print(platform.machine())')" = arm64 ] || fail "$PYTHON is not a native arm64 Python."
[ "$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = 3.12 ] || fail "$PYTHON is not Python 3.12."

echo "== Python environment ($VENV)"
[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
# An exact pip, not whatever is newest today (F25); tests/test_dependency_pins.py keeps every copy in step.
"$VENV/bin/python" -m pip install --quiet pip==26.2.1
if [ -z "$UPSTREAM" ]; then
    "$VENV/bin/python" -m pip install --quiet -r "$REQUIREMENTS" \
        || fail "Installing packages failed. Run this script again to retry."
    "$VENV/bin/python" -m pip check || fail "pip check found broken requirements (see above). Run this script again to retry."
else
    # Distribution names in a pip freeze, normalized (PEP 503) and sorted.
    names() {
        sed -E -e '/^[[:space:]]*(#|-|$)/d' -e 's/[[:space:]]*[=@<>!~;[].*//' "$1" \
            | tr '[:upper:]' '[:lower:]' | sed -E 's/[-_.]+/-/g' | LC_ALL=C sort -u
    }
    # pip cannot undo a half-finished install: put back the set recorded before
    # it (pip install -r only adds), remove what the update added, then check it.
    restore_packages() {
        echo "== Restoring the previous packages"
        if grep -q '[^[:space:]]' "$PREVIOUS_PACKAGES"; then
            "$VENV/bin/python" -m pip install --quiet -r "$PREVIOUS_PACKAGES" || return 1
        fi
        "$VENV/bin/python" -m pip freeze > "$NEXT_REQUIREMENTS/now-packages.txt" || return 1
        local added
        added="$(LC_ALL=C comm -13 <(names "$PREVIOUS_PACKAGES") <(names "$NEXT_REQUIREMENTS/now-packages.txt"))"
        if [ -n "$added" ]; then
            echo "== Removing what the update added:" $added
            # One distribution name per word.
            # shellcheck disable=SC2086
            "$VENV/bin/python" -m pip uninstall --quiet --yes $added || return 1
        fi
        "$VENV/bin/python" -m pip check
    }
    # Any failure between the first package change and the source switch, or a
    # stop (Ctrl-C, or Check for Updates giving up), puts the packages back. A
    # second stop interrupts only the rollback's current step.
    abandon_update() {
        trap : INT TERM HUP
        if restore_packages; then
            fail "$1; the previous packages are back and the source was not switched. Run this script again to retry."
        fi
        fail "$1, and the previous packages could not be restored; the source was not switched. Run scripts/install-mac.sh --update again (it needs the network)."
    }
    PREVIOUS_PACKAGES="$NEXT_REQUIREMENTS/previous-packages.txt"
    "$VENV/bin/python" -m pip freeze > "$PREVIOUS_PACKAGES" \
        || fail "Could not list the installed packages (pip freeze failed); nothing was changed."
    trap 'abandon_update "The update was stopped"' INT TERM HUP
    "$VENV/bin/python" -m pip install --quiet -r "$REQUIREMENTS" \
        || abandon_update "Installing the update's packages failed"
    "$VENV/bin/python" -m pip check || abandon_update "The update's packages do not fit together (pip check, above)"
    echo "== Source"
    git -C "$ROOT" merge --ff-only --quiet "$UPSTREAM" \
        || abandon_update "git could not move this copy forward (local changes?)"
    trap - INT TERM HUP
fi

export SOTTO_DATA_DIR="$DATA_DIR"
export SOTTO_HF_HOME="$DATA_DIR/huggingface"
echo "== Speech model (first time: about 1.6 GB into $SOTTO_HF_HOME)"
"$VENV/bin/python" "$ROOT/sotto.py" setup

echo "== Sotto.app"
"$VENV/bin/python" "$ROOT/scripts/install_app.py" --data-dir "$DATA_DIR"

cat <<EOF

Done. Open Sotto like any app (Spotlight, Launchpad, or: open -a Sotto).
The first time, allow "Sotto" to use the Microphone and turn it on under
System Settings → Privacy & Security → Accessibility. Then hold right Option,
speak, and release. Choose another key, languages and "Start Sotto when I log
in" under Settings… in Sotto's menu. Update later with: scripts/install-mac.sh --update
EOF

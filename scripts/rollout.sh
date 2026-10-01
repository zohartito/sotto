#!/bin/bash
# Sealed-release rollout: stage -> activate -> preflight -> install config ->
# restart -> verify.
#
# Receipts are sealed to runtime_source_digest(), so EVERY source change needs
# a fresh learning-preflight under the new release before the app restarts —
# skipping it crash-loops the app at warmup (bit 4 rollouts before this
# script existed). Rollback needs the same preflight under the LKG release,
# which this script's failure path performs automatically.
#
# EVERY failure after activation rolls back, not just a boot timeout: once
# `current` points at the new release, a failed preflight has left receipts
# that are invalid for it, and the next launch crash-loops.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RELEASES="$HOME/Library/Application Support/sotto/releases"
LOG="$HOME/Library/Logs/Sotto/app-error.log"
LABEL="com.zohartito.sotto.app"
DOMAIN="gui/$(id -u)"
INSTALLED_PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PLIST_TEMPLATE="launchd/$LABEL.plist"
BACKUP_DIR="$HOME/Library/Application Support/sotto/rollout-backups"
PLIST_BACKUP=""

plist_env() {
    env PATH=/usr/bin:/bin \
        SOTTO_FFMPEG=/opt/homebrew/bin/ffmpeg \
        SOTTO_HF_HOME="$HOME/Library/Application Support/sotto/huggingface" \
        HF_HOME="$HOME/Library/Application Support/sotto/huggingface" \
        SOTTO_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        PYTHONDONTWRITEBYTECODE=1 "$@"
}

preflight() {
    plist_env /usr/bin/python3 -I -S -B \
        "$RELEASES/current/source/sealed_release_bootstrap.py" \
        --release-root "$RELEASES" --entry sotto.py learning-preflight
}

# Bytes already in the log, so verify_boot reads only what THIS restart wrote.
log_size() { [ -f "$LOG" ] && wc -c < "$LOG" | tr -d ' ' || echo 0; }

verify_boot() {
    # Two independent facts, both specific to this release: a live process
    # started from THIS digest's bootstrap, and the app's own readiness banner
    # written AFTER the restart.
    #
    # Never key off incidental output. The first version of this script looked
    # for "releases/$digest/source/sotto.py" in the log, which a healthy boot
    # only ever emitted because a PyObjC warning happened to quote sotto.py's
    # path — so it passed by accident, and any warning filter would have failed
    # every rollout. It also matched any sealed_release_bootstrap process, and
    # the silver worker runs one of those too.
    local digest="$1" from="$2" deadline=$((SECONDS + 180))
    local proc="releases/$digest/source/sealed_release_bootstrap.py"
    while ((SECONDS < deadline)); do
        if pgrep -f "$proc" >/dev/null \
            && tail -c "+$((from + 1))" "$LOG" 2>/dev/null | grep -q "listening on"; then
            sleep 5   # survive the warmup window, not just reach it
            pgrep -f "$proc" >/dev/null && return 0
        fi
        sleep 3
    done
    return 1
}

prepare_plist() {
    local stamp
    if [ ! -f "$INSTALLED_PLIST" ] || [ -L "$INSTALLED_PLIST" ]; then
        echo "REFUSING: installed LaunchAgent is missing or unsafe: $INSTALLED_PLIST" >&2
        return 1
    fi
    /usr/bin/plutil -lint "$REPO/$PLIST_TEMPLATE" >/dev/null
    /bin/mkdir -p "$BACKUP_DIR"
    /bin/chmod 700 "$BACKUP_DIR"
    stamp=$(/bin/date -u +%Y%m%dT%H%M%SZ)
    PLIST_BACKUP="$BACKUP_DIR/$LABEL.$stamp.plist"
    /bin/cp -p "$INSTALLED_PLIST" "$PLIST_BACKUP"
    /bin/chmod 600 "$PLIST_BACKUP"
}

install_current_plist() {
    plist_env /usr/bin/python3 -I -S -B "$RELEASES/current/source/launchd_templates.py" \
        --template "$RELEASES/current/source/$PLIST_TEMPLATE" --output "$INSTALLED_PLIST"
    /usr/bin/plutil -lint "$INSTALLED_PLIST" >/dev/null
}

reload_job() {
    # bootout makes launchd forget the old ProgramArguments.  kickstart alone
    # only restarts the old in-memory definition and silently ignores a changed
    # sealed template.
    local attempt unload_deadline
    /bin/launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    # launchd's ExitTimeOut is five seconds, so a five-second poll window can
    # race the normal boundary and falsely report that the definition is still
    # loaded. Give graceful AppKit/model teardown a bounded margin before the
    # fail-closed check; never force-kill or bootstrap over the old definition.
    unload_deadline=$((SECONDS + 20))
    while ((SECONDS < unload_deadline)); do
        if ! /bin/launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
            break
        fi
        /bin/sleep 1
    done
    if /bin/launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
        echo "REFUSING: old LaunchAgent definition did not unload" >&2
        return 1
    fi
    for attempt in 1 2 3; do
        if /bin/launchctl bootstrap "$DOMAIN" "$INSTALLED_PLIST"; then
            return 0
        fi
        # A failed bootstrap can still have registered the job.  Let the
        # normal boot verification judge it instead of colliding on a retry.
        if /bin/launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
            return 0
        fi
        /bin/sleep 1
    done
    return 1
}

rollback() {
    "$REPO/venv/bin/python" - <<EOF
import pathlib, sys
sys.path.insert(0, "$REPO")
from sealed_release import rollback_release
print("rolled back to", rollback_release(pathlib.Path("$RELEASES")).name)
EOF
    # A failed LKG preflight must not abort the restart: a running app with
    # stale receipts is recoverable, a stopped one needs manual rescue.
    preflight || echo "WARNING: LKG preflight failed — app may not warm up" >&2
    if [ -n "$PLIST_BACKUP" ] && [ -f "$PLIST_BACKUP" ]; then
        /usr/bin/install -m 600 "$PLIST_BACKUP" "$INSTALLED_PLIST"
    fi
    reload_job
    echo "LKG restored; investigate $LOG" >&2
}

echo "== syntax check"
"$REPO/venv/bin/python" -m compileall -q "$REPO"/*.py "$REPO/tests"

echo "== validate + back up LaunchAgent"
prepare_plist

echo "== stage + activate"
DIGEST=$("$REPO/venv/bin/python" - <<EOF
import pathlib, sys
sys.path.insert(0, "$REPO")
from sealed_release import stage_release, activate_release
releases = pathlib.Path("$RELEASES")
bundle = stage_release(releases, source_root=pathlib.Path("$REPO"))
activate_release(releases, bundle.name)
print(bundle.name)
EOF
)
echo "activated $DIGEST"

echo "== preflight receipts under new release"
if ! preflight; then
    echo "PREFLIGHT FAILED under $DIGEST — activated release has no valid receipts" >&2
    rollback
    exit 1
fi

echo "== install LaunchAgent + restart"
LOG_BYTES=$(log_size)
if ! install_current_plist || ! reload_job; then
    echo "LAUNCHAGENT RELOAD FAILED for $LABEL" >&2
    rollback
    exit 1
fi

echo "== verify boot"
if verify_boot "$DIGEST" "$LOG_BYTES"; then
    echo "OK: release $DIGEST is live"
    exit 0
fi

echo "BOOT FAILED — rolling back to LKG" >&2
rollback
exit 1

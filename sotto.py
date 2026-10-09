#!/usr/bin/env python3
"""sotto — hold a key, speak, release; the words land at your cursor. All local.

Gestures (on whichever trigger key you pick; right Option by default):
    hold        push-to-talk — release to transcribe
    double-tap  hands-free — records until the next single tap
    lone tap    ignored (no accidental blips)

    python sotto.py run                      # menu bar app with the pill
    python sotto.py run --trigger fn         # another trigger key
    python sotto.py doctor                   # permission + device checks

The installed Sotto.app asks for the Microphone and Accessibility itself; a
terminal run needs both granted to the terminal. Speech engines (Whisper via
mlx-whisper, Parakeet, Nemotron) run on this Mac; model files download once,
at pinned revisions, and audio never leaves the Mac.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import pipeline
# The output guards live in pipeline.py (shared with Windows); these names stay
# importable from sotto for existing callers.
from pipeline import (LOOP_MAX_UNIT_WORDS, LOOP_MIN_PREFIX_CHARS, LOOP_MIN_REPEATS,  # noqa: F401
                      NO_SPEECH_TEXT, _find_repetition_loop, apply_voice_cleanup,
                      looks_hallucinated, reads_as_no_speech, salvage_repetition_loop, screen)
try:  # Keep content-free CLI/admin helpers importable on non-macOS test hosts.
    import Quartz
    from AppKit import NSPasteboard, NSPasteboardItem, NSPasteboardTypeString
except ImportError:  # pragma: no cover - capture itself remains macOS-only
    class _UnavailableMacAPI:
        # Constants are declared at module load; capture methods fail only if
        # somebody actually starts the macOS-only runtime.
        def __getattr__(self, name): return 0
    Quartz = _UnavailableMacAPI()
    NSPasteboard = NSPasteboardItem = NSPasteboardTypeString = _UnavailableMacAPI()

SAMPLE_RATE = 16_000
LONG_CAPTURE_S = 30.0           # beyond one whisper window: collapse dead air
SILENCE_RESTART_AFTER_S = 600.0  # min process age before a dead-mic restart
HOLD_THRESHOLD_S = 0.35
DOUBLE_TAP_WINDOW_S = 0.40
HANDS_FREE_MAX_S = 600.0        # watchdog: force-finish a forgotten open mic
# Clipboard restore after the synthetic ⌘V. Reading the pasteboard does not
# change its changeCount and there is no public "the app has read it" signal,
# so the restore is a bounded wait: long enough for an app that is briefly busy
# at paste time (Electron hitches, a tab mid-layout) to still read Sotto's text
# — the old 0.6 s handed such apps the user's OLD clipboard — and short enough
# that a user ⌘V a few seconds later gets their own clipboard back. The restore
# still runs only if changeCount is unchanged (a user ⌘C in the window wins),
# and overlapping pastes carry the original forward (see inject()).
RESTORE_DELAY_S = 3.0
# nspasteboard.org marker on Sotto's own paste write: clipboard managers skip
# it, so dictations do not pile up in their history (the restore is untouched).
TRANSIENT_PASTEBOARD_TYPE = "org.nspasteboard.TransientType"
DELIVERY_POLL_S = 0.15           # re-check a held key this often before pasting
DELIVERY_WAIT_MAX_S = 30.0       # key held, no recording: give up, the text stays in History
# The drop note for a dictation History could not keep (F16a/F16b, N16).
NOT_IN_HISTORY_EITHER = "History could not save it either; please dictate again."
SOTTO_EVENT_TAG = 0x534F5454     # "SOTT" in kCGEventSourceUserData on every key event Sotto posts
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
HISTORY_KEEP = 200   # every stored transcript is listed; the menu scrolls
APP_DRAIN_TIMEOUT = 1.0
RESTART_DRAIN_DEADLINE_S = 20.0  # a wedged native call must not block recovery forever
QUIT_DRAIN_S = 10.0              # Quit finishes the dictation in progress, up to this long
# An installed update restarts once dictation is done: the longest recording
# (the hands-free watchdog) plus time to transcribe and paste it, then anyway.
UPDATE_DRAIN_DEADLINE_S = HANDS_FREE_MAX_S + 120.0
RESTART_RELEASE_WAIT_S = 2.0     # bounded wait for the async mic teardown before exec
FAST_MARGIN_FRAMES = 500         # Fast: encode the speech plus 5 s, not a padded 30 s
MODEL_REWARM_AFTER_S = 240.0


class ShutdownBoundary:
    """One-way, event-only shutdown gate for the interactive runtime.

    Signal handlers deliberately only set ``event``.  Capture/UI teardown and
    queue accounting happen on normal Python/AppKit execution paths, so a
    signal cannot interrupt a SQLite write or start a second mutation.
    """

    def __init__(self) -> None:
        self.event = threading.Event()

    def requested(self) -> bool:
        return self.event.is_set()

    def request(self, *_unused) -> None:
        self.event.set()

    def enqueue(self, jobs: queue.Queue, job: tuple) -> bool:
        """Queue work only while live; race with shutdown is discarded below."""
        if self.requested():
            return False
        jobs.put(job)
        if self.requested():
            self.discard_queued(jobs)
            return False
        return True

    @staticmethod
    def discard_queued(jobs: queue.Queue) -> int:
        """Drain queued work with matching task accounting, never executing it."""
        discarded = 0
        while True:
            try:
                jobs.get_nowait()
            except queue.Empty:
                return discarded
            else:
                jobs.task_done()
                discarded += 1

    def stop_capture(self, capture) -> None:
        """Discard an active recording; it must never become a history row."""
        self.request()
        try:
            capture.abort()
        finally:
            close = getattr(capture, "shutdown", None)
            if callable(close):
                close()
            else:
                capture.release_soon()


@contextmanager
def shutdown_signal_handlers(boundary: ShutdownBoundary):
    """Install event-only SIGTERM/SIGINT handlers and restore embedded callers."""
    previous: dict[int, object] = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, boundary.request)
    try:
        yield boundary
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def commit_if_running(boundary: ShutdownBoundary, commit) -> bool:
    """Execute a post-inference side effect only while shutdown is clear."""
    if boundary.requested():
        return False
    commit()
    return True


def finalize_primary_live_delivery(*, append, adaptive_runtime, appended_publication: dict | None,
                                  shutdown: ShutdownBoundary, inject) -> object | None:
    """Persist and authorize a primary result before any cursor side effect.

    The caller supplies the already-prepared History append closure.  This
    narrow seam makes the order testable without a macOS event loop: append,
    adaptive registration, exact comparator publication acknowledgement, then
    (only while still live) cursor scheduling.
    """
    row=append()
    if adaptive_runtime is not None and row is not None and not shutdown.requested():
        try:
            adaptive_runtime.register_live(row)
            if appended_publication is not None and not adaptive_runtime.acknowledge_comparator_publication(appended_publication):
                raise RuntimeError("comparator publication unavailable")
        except Exception:
            if appended_publication is not None:
                adaptive_runtime.cancel_comparator_publication(appended_publication)
                adaptive_runtime.fail_comparator_publication(appended_publication)
            raise
    if row is not None and inject is not None and not shutdown.requested():
        inject()
    return row

# Sotto keeps its small, required Whisper model on the Mac itself.  The general
# Hugging Face cache may live on removable storage, but dictation must still
# start when that drive is absent.  Set SOTTO_HF_HOME to override this location.
from sotto_paths import MODEL_CACHE_DIR

SOTTO_HF_HOME = MODEL_CACHE_DIR
os.environ["HF_HOME"] = str(SOTTO_HF_HOME)
os.environ["HF_HUB_CACHE"] = str(SOTTO_HF_HOME / "hub")
# No telemetry (README): the Hub client otherwise adds the torch version and
# the calling agent to the user agent of every model download.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# trigger name -> (modifier flag mask, virtual keycode, per-side device bit).
# The NX_DEVICE*KEYMASK bits identify WHICH side of a paired modifier is down —
# the aggregate mask alone lies when the other side's twin is held. (Verified
# live on this machine: right-option down -> flags contain 0x40, not 0x20.)
# CGEventSourceKeyState does NOT track modifier keycodes; never use it here.
TRIGGERS = {
    "fn": (Quartz.kCGEventFlagMaskSecondaryFn, 63, 0),
    "right-cmd": (Quartz.kCGEventFlagMaskCommand, 54, 0x0010),
    "right-option": (Quartz.kCGEventFlagMaskAlternate, 61, 0x0040),
    "right-shift": (Quartz.kCGEventFlagMaskShift, 60, 0x0004),
    "right-ctrl": (Quartz.kCGEventFlagMaskControl, 62, 0x2000),
    "left-cmd": (Quartz.kCGEventFlagMaskCommand, 55, 0x0008),
    "left-option": (Quartz.kCGEventFlagMaskAlternate, 58, 0x0020),
    "left-shift": (Quartz.kCGEventFlagMaskShift, 56, 0x0002),
    "left-ctrl": (Quartz.kCGEventFlagMaskControl, 59, 0x0001),
}
TRIGGER_LABELS = {
    "right-option": "Right Option ⌥", "left-option": "Left Option ⌥",
    "right-cmd": "Right Command ⌘", "left-cmd": "Left Command ⌘",
    "right-ctrl": "Right Control ⌃", "left-ctrl": "Left Control ⌃",
    "right-shift": "Right Shift ⇧", "left-shift": "Left Shift ⇧", "fn": "Fn / Globe",
}


def glossary_source(explicit: str | None) -> Path | None:
    """--glossary wins; otherwise glossary.txt in the data folder, so Sotto.app
    (which takes no arguments) can be taught names and terms too."""
    if explicit:
        return Path(explicit)
    from sotto_paths import DATA_DIR
    candidate = DATA_DIR / "glossary.txt"
    return candidate if candidate.is_file() else None


def hands_free_hint(trigger: str) -> str:
    """How to end hands-free, short enough for the overlay: "tap right ⌘ to stop"."""
    if trigger == "fn":
        return "tap fn to stop"
    words = TRIGGER_LABELS.get(trigger, trigger).split()
    return f"tap {words[0].lower()} {words[-1]} to stop"


# Toggle hotkey: a NORMAL key plus modifiers. Unlike the bare-modifier
# trigger, real keycodes survive remote-desktop sessions (AnyDesk from an
# iPad), and toggle semantics don't depend on release timing over a network.
KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8,
    "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38, "k": 40, "n": 45,
    "m": 46, "space": 49, "f13": 105, "f14": 107, "f15": 113, "f16": 106,
}
MODIFIER_FLAGS = {
    "cmd": Quartz.kCGEventFlagMaskCommand,
    "ctrl": Quartz.kCGEventFlagMaskControl,
    "opt": Quartz.kCGEventFlagMaskAlternate,
    "alt": Quartz.kCGEventFlagMaskAlternate,
    "shift": Quartz.kCGEventFlagMaskShift,
}
ALL_MODIFIERS = (Quartz.kCGEventFlagMaskCommand | Quartz.kCGEventFlagMaskControl
                 | Quartz.kCGEventFlagMaskAlternate
                 | Quartz.kCGEventFlagMaskShift)


def parse_hotkey(spec: str) -> tuple[int, int] | None:
    """'ctrl-opt-d' -> (required modifier flags, keycode). None if unparsable."""
    parts = [p.strip().lower() for p in spec.split("-") if p.strip()]
    if not parts:
        return None
    key = parts[-1]
    if key not in KEYCODES:
        return None
    flags = 0
    for part in parts[:-1]:
        if part not in MODIFIER_FLAGS:
            return None
        flags |= MODIFIER_FLAGS[part]
    return flags, KEYCODES[key]


def log(msg: str) -> None:
    """Never raises: the transcription worker logs from inside its own error
    handler, so a full disk or a rotation race must not end the thread (N35)."""
    try:
        print(msg, file=sys.stderr, flush=True)
        keep_app_log_small()
    except (OSError, ValueError):
        pass


LAUNCHD_LABEL = "com.zohartito.sotto.app"  # must match launchd/ and scripts/rollout.sh


def app_mode() -> bool:
    """Started by the installed Sotto.app launcher (not a terminal)."""
    return os.environ.get("SOTTO_LAUNCHER") == "app"


_app_log: dict = {"path": None, "max_bytes": 0}  # set once logs are routed to a file
_app_log_lock = threading.Lock()  # every thread logs; one of them rotates at a time


def route_app_logs(data_dir: Path, max_bytes: int = 2_000_000) -> Path:
    """Sotto.app has no terminal: send stdout/stderr to a small rotating log.
    Logs hold counts and timings only, never what was said."""
    folder = data_dir / "logs"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = folder / "sotto.log"
    _rotate_if_large(path, max_bytes)
    _point_output_at(path)
    _app_log.update(path=path, max_bytes=max_bytes)
    return path


def _rotate_if_large(path: Path, max_bytes: int) -> bool:
    """Move a log past max_bytes to sotto.log.1 (replacing the older one)."""
    if path.exists() and path.stat().st_size > max_bytes:
        os.replace(path, path.with_suffix(".log.1"))
        return True
    return False


def _point_output_at(path: Path) -> None:
    """Send this process's stdout and stderr to the log file at path."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    for stream in (1, 2):
        os.dup2(descriptor, stream)
    os.close(descriptor)


def keep_app_log_small() -> None:
    """Rotate a routed log that grew past its limit while Sotto runs; the old
    file keeps the open descriptors, so point output at a fresh one."""
    path = _app_log["path"]
    if path is None:
        return
    with _app_log_lock:
        if _rotate_if_large(path, _app_log["max_bytes"]):
            _point_output_at(path)


def sotto_icon_path() -> Path | None:
    """Sotto.app's icon, found from the launcher that started this process."""
    executable = os.environ.get("SOTTO_APP_EXECUTABLE", "").strip()
    icon = Path(executable).parent.parent / "Resources" / "Sotto.icns" if executable else None
    return icon if icon is not None and icon.is_file() else None


ACCESSIBILITY_PROBE = (
    "import ctypes, sys; ax = ctypes.CDLL('/System/Library/Frameworks/ApplicationServices.framework/"
    "ApplicationServices'); ax.AXIsProcessTrusted.restype = ctypes.c_bool; "
    "sys.exit(0 if ax.AXIsProcessTrusted() else 1)")


def accessibility_granted_now() -> bool:
    """AXIsProcessTrusted() asks tccd once per process and then repeats that
    answer, so polling it never sees the switch turn on. A fresh child asks
    again, and macOS attributes it to the same app (Sotto)."""
    import subprocess
    try:
        return subprocess.run([sys.executable, "-I", "-S", "-c", ACCESSIBILITY_PROBE],
                              timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def explain_permission(title: str, message: str, pane: str, *, granted=None, register=None) -> bool:
    """In Sotto.app there is no terminal to read: say what is missing and
    offer the exact System Settings pane. With ``granted`` the dialog waits
    and returns True once the permission is on; Quit returns False."""
    import ui
    return ui.permission_dialog(
        title, message, f"x-apple.systempreferences:com.apple.preference.security?{pane}",
        granted=granted, register=register, icon_path=sotto_icon_path())


def launchd_owns_this_process() -> bool:
    """An installed job with the same name must never control a source run."""
    import subprocess
    try:
        result = subprocess.run(
            ["/bin/launchctl", "print", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    match = re.search(r"^\s*pid = (\d+)\s*$", result.stdout, re.MULTILINE)
    return result.returncode == 0 and match is not None and int(match[1]) == os.getpid()


def launched_as_sealed_job() -> bool:
    """Does anything say launchd started this process from a sealed release?

    launchd sets XPC_SERVICE_NAME to the job label, and the sealed bootstrap
    runs ``<releases>/<digest>/source/<entry>``.  Either one means a restart
    must go through launchd: an exec here would bypass the bootstrap's
    verification and ``-I -S -B`` isolation."""
    if os.environ.get("XPC_SERVICE_NAME") == LAUNCHD_LABEL:
        return True
    entry = Path(sys.argv[0]).absolute() if sys.argv and sys.argv[0] else None
    return entry is not None and entry.parent.name == "source" and entry.parent.parent.parent.name == "releases"


class RestartController:
    """Restart our supervisor or request a drained foreground exec."""

    def __init__(self, shutdown: ShutdownBoundary):
        self.shutdown = shutdown
        self.managed = launchd_owns_this_process()
        # Fail closed: an unconfirmed launchd job never execs itself.
        self.exec_allowed = not self.managed and not launched_as_sealed_job()
        self.foreground_pending = False
        self.argv = [sys.executable, *sys.argv]
        self._on_failure = None

    def request(self, on_failure=None) -> bool:
        if self.shutdown.requested():
            return False
        if not self.managed and not self.exec_allowed:
            raise RuntimeError("Sotto could not confirm its launchd job; quit and launch it again")
        self._on_failure = on_failure
        if self.managed:
            # launchd stops this very process during kickstart -k; keep the
            # main thread free so that SIGTERM still shuts down gracefully.
            threading.Thread(target=self._kickstart, daemon=True).start()
        else:
            self.foreground_pending = True
            self.shutdown.request()
        return True

    def _kickstart(self) -> None:
        import subprocess
        try:
            result = subprocess.run(
                ["/bin/launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"],
                capture_output=True, text=True, timeout=30)
            failed = result.returncode != 0
        except (OSError, subprocess.TimeoutExpired):
            failed = True
        if failed:
            self.failed("launchd could not restart Sotto; quit and launch it again")

    def failed(self, message: str) -> None:
        """Report a restart that failed after request() already returned."""
        callback, self._on_failure = self._on_failure, None
        log(f"! restart failed: {message}")
        if callback is not None:
            callback(message)

    def exec_foreground(self) -> None:
        if not self.foreground_pending:
            return
        if app_mode() and os.environ.get("SOTTO_APP_EXECUTABLE", "").strip():
            relaunch_app_after_exit()
            return
        os.execv(sys.executable, self.argv)


APP_RELAUNCH_SCRIPT = (
    'while /bin/kill -0 "$1" 2>/dev/null; do /bin/sleep 0.1; done; '
    '/bin/launchctl kickstart "gui/$(/usr/bin/id -u)/$2" 2>/dev/null || /usr/bin/open -n "$3"')


def relaunch_app_after_exit() -> None:
    """Sotto.app restarts as a fresh process: an in-place exec keeps the pid,
    and macOS then leaves the new menu bar icon blank and unclickable
    (2026-09-30). A detached helper waits for the launcher to exit, then
    starts the login item if it is loaded, otherwise opens the app again."""
    import subprocess
    import login_item
    bundle = Path(os.environ["SOTTO_APP_EXECUTABLE"].strip()).parents[2]
    subprocess.Popen(
        ["/bin/sh", "-c", APP_RELAUNCH_SCRIPT, "sotto-relaunch", str(os.getppid()), login_item.LABEL, str(bundle)],
        start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, close_fds=True)
    log("restarting Sotto.app")


def persist_engine_and_restart(mode: str, restart, on_failure=None) -> None:
    """A failed restart must not leave the next launch on an untested choice."""
    from speech_config import load_engine_mode, save_engine_mode
    previous = load_engine_mode()
    save_engine_mode(mode)

    def restore(message: str) -> None:
        save_engine_mode(previous)
        if on_failure is not None:
            on_failure(message)

    try:
        if not restart(on_failure=restore):
            raise RuntimeError("Sotto is already shutting down")
    except Exception:
        save_engine_mode(previous)
        raise


def microphone_permission(*, request: bool = False) -> bool:
    """Request audio permission explicitly, before model download/capture."""
    from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
    status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)
    if status == 3:  # AVAuthorizationStatusAuthorized
        return True
    if status == 0 and request:  # NotDetermined
        completed = threading.Event()
        answer = []

        def received(allowed):
            answer.append(bool(allowed))
            completed.set()

        AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeAudio, received)
        completed.wait(30)
        return bool(answer and answer[0])
    return False


def totals_path():
    import progress
    from sotto_paths import DATA_DIR
    return DATA_DIR / progress.TOTALS_NAME


def record_totals(text: str, seconds: float) -> None:
    """Lifetime counts for Your progress: words and seconds, never text."""
    import progress
    try:
        progress.record(totals_path(), text=text, seconds=seconds)
    except (OSError, ValueError) as exc:  # unreadable or malformed: keep the file, skip this one
        log(f"! progress totals not saved ({type(exc).__name__})")


def seed_totals(store) -> None:
    import progress
    try:
        progress.ensure_totals(totals_path(), store.entries(limit=HISTORY_KEEP))
    except OSError as exc:
        log(f"! progress totals not started ({type(exc).__name__})")


def apply_personal_dictionary(text: str, preprocessing: dict) -> str:
    """pipeline.apply_personal_dictionary, logging to Sotto's log."""
    return pipeline.apply_personal_dictionary(text, preprocessing, log)


def live_preview(text: str) -> str:
    """The pill's live words with the user's spellings, so they match what gets pasted."""
    if not text:
        return text
    import dictionary
    try:
        return dictionary.apply(text, dictionary.load(dictionary.DICTIONARY_PATH))[0]
    except Exception:  # display only: a broken dictionary leaves the words as heard
        return text


class PendingDeliveries:
    """Pastes and undos handed to the main thread but not yet delivered.

    The transcription queue counts a job done once its paste is scheduled,
    before the main loop runs it, so Quit also waits for this to reach zero.
    add() runs on the worker, finish() on the main thread; hence the lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._count = 0

    def add(self) -> None:
        with self._lock:
            self._count += 1

    def finish(self) -> None:
        with self._lock:
            self._count = max(0, self._count - 1)

    def count(self) -> int:
        with self._lock:
            return self._count


def main_thread_dispatch(has_ui: bool, call_after, deliveries: PendingDeliveries | None = None):
    """Return (ui_call, deliver_call), both scheduling work on the main thread.

    ui_call updates the menu bar and pill, so it does nothing under
    --no-overlay. deliver_call (pasting, voice undo) always runs: hiding the
    UI must never stop text from reaching the cursor. Each delivery is counted
    in deliveries until the scheduled method calls deliveries.finish().
    """
    def deliver_call(method, *call_args) -> None:
        if deliveries is not None:
            deliveries.add()
        call_after(method, *call_args)

    def ui_call(method, *call_args) -> None:
        if has_ui:
            call_after(method, *call_args)

    return ui_call, deliver_call


def wait_until_idle(busy, *, poll_s: float = 0.5, sleep=time.sleep,
                    deadline_s: float | None = None, clock=time.monotonic) -> bool:
    """Block the calling worker thread until busy() is False, or until
    deadline_s has passed. Returns whether it became idle.

    Side effects: sleeps. Used before an update restart and Quit, because
    shutdown discards any recording or transcription still in progress.
    """
    give_up = None if deadline_s is None else clock() + deadline_s
    while busy():
        if give_up is not None and clock() >= give_up:
            return False
        sleep(poll_s)
    return True


def dictation_in_flight(capture, jobs, deliveries: PendingDeliveries) -> bool:
    """Whether a dictation is still recording, transcribing, or waiting to be
    pasted — the three stages Quit lets finish before shutting down."""
    return capture.is_active() or jobs.unfinished_tasks > 0 or deliveries.count() > 0


def end_capture_now(engine, capture, on_finish) -> str | None:
    """End the recording in progress so it is transcribed, not dropped.

    Returns "gesture" when the gesture engine ended it, "orphan" when the
    capture outlived the engine's state (a gesture callback raised after the
    mic started) and was ended directly, or None when nothing was recording.
    Side effects: may stop the capture and queue it for transcription.
    """
    if engine.force_finish():
        return "gesture"
    if capture.is_active():
        on_finish()
        return "orphan"
    return None


INSTALL_TIMEOUT_S = 3600.0  # an engine download on a slow link, but never forever


def run_installer(argv: list[str], *, run=subprocess.run) -> tuple[bool, str]:
    """Run an optional-engine installer; (succeeded, its last output). Never
    raises, so the caller always clears its "installing" state."""
    try:
        result = run(argv, capture_output=True, text=True, timeout=INSTALL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {INSTALL_TIMEOUT_S / 60:.0f} minutes"
    except OSError as exc:
        return False, f"could not start ({exc.strerror or exc})"
    return result.returncode == 0, (result.stderr or result.stdout).strip()[-200:]


def desktop_audio_path(folder: Path, timestamp: float) -> Path:
    """Where "Save audio to Desktop" writes a recording: named by the second
    it was captured, with -2, -3, … so an earlier save is never overwritten.

    Side effects: creates the file empty with exclusive create, so two saves
    running at once can never pick the same name.
    """
    import datetime
    stem = f"sotto-{datetime.datetime.fromtimestamp(timestamp):%Y%m%d-%H%M%S}"
    candidate, number = folder / f"{stem}.wav", 2
    while True:
        try:
            with open(candidate, "xb"):
                return candidate
        except FileExistsError:
            candidate, number = folder / f"{stem}-{number}.wav", number + 1


def save_audio_copy(source: Path, folder: Path, timestamp: float) -> Path:
    """Copy a recording into folder under a fresh desktop_audio_path name.

    Side effects: writes one new file. The source is opened first, so a
    missing recording raises FileNotFoundError without leaving a file behind.
    """
    with open(source, "rb") as audio:
        target = desktop_audio_path(folder, timestamp)
        with open(target, "wb") as saved:
            shutil.copyfileobj(audio, saved)
    return target


class CaptureGate:
    """Whether a key press may start a new recording.

    close() is called before an update restart: it waits for a start already
    under way (so that recording is visible as active), then refuses new ones,
    so "wait until idle, then restart" cannot discard a recording that began
    in between. reopen() undoes it if the restart does not happen.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._open = True

    @contextmanager
    def starting(self):
        """Hold while deciding and starting a recording; yields whether it may start."""
        with self._lock:
            yield self._open

    def close(self) -> None:
        with self._lock:
            self._open = False

    def reopen(self) -> None:
        with self._lock:
            self._open = True


def choose_language(probabilities: dict, allowed) -> str:
    """Automatic's pick. Prefer the user's languages, but speech that is
    clearly another language (the allowed ones score < 0.25 while it scores
    >= 0.5) is written in that language: forcing it into an allowed one makes
    Whisper translate it. Measured: en/he speech scores >= 0.989 in-set;
    other languages <= 0.112 in-set with >= 0.863 for themselves."""
    top = max(probabilities, key=probabilities.get)
    if not allowed:
        return top
    best = max(allowed, key=lambda code: probabilities.get(code, 0.0))
    if top not in allowed and probabilities.get(best, 0.0) < 0.25 and probabilities[top] >= 0.5:
        return top
    return best


class ModelUnavailable(RuntimeError):
    """A plain, actionable reason the Whisper model cannot be used yet."""


def _complete_whisper_snapshot(path) -> str:
    """An interrupted download can leave a snapshot folder without weights."""
    folder = Path(path)
    if (folder / "config.json").is_file() and any(
            (folder / name).is_file() for name in ("weights.safetensors", "weights.npz")):
        return str(folder)
    raise FileNotFoundError(f"incomplete model snapshot: {folder}")


def resolve_pinned_snapshot(repo: str, allow_patterns: list, complete) -> str:
    """The local snapshot of ``repo`` at its pinned commit, cache-first.

    A cached snapshot is used without any Hub request; a first run that is
    not offline downloads exactly the pinned bytes; offline flags forbid it.
    ``complete`` rejects a folder an interrupted download left partial."""
    from offline_runtime import offline_requested
    from huggingface_hub import snapshot_download
    from speech_config import MODEL_REVISIONS
    options = {"repo_id": repo, "revision": MODEL_REVISIONS.get(repo),
               "cache_dir": str(SOTTO_HF_HOME / "hub"), "allow_patterns": allow_patterns}
    try:
        return complete(snapshot_download(**options, local_files_only=True))
    except FileNotFoundError:
        if offline_requested():
            raise ModelUnavailable(
                f"{repo} is not downloaded yet. Run once without SOTTO_OFFLINE, "
                "HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE to download it.") from None
    try:
        return complete(snapshot_download(**options))
    except Exception as exc:
        raise ModelUnavailable(
            f"could not download {repo} ({type(exc).__name__}: {str(exc)[:160]}). "
            "Check internet access and free disk space, then run again.") from None


def parakeet_cached() -> bool:
    """Is the pinned Parakeet snapshot fully downloaded? (a path check only)"""
    from speech_config import MODEL_PROFILES, pinned_snapshot_cached
    return pinned_snapshot_cached(MODEL_PROFILES["parakeet"].repo, SOTTO_HF_HOME / "hub",
                                  ("config.json", "model.safetensors"))


def _complete_parakeet_snapshot(path) -> str:
    folder = Path(path)
    if (folder / "config.json").is_file() and (folder / "model.safetensors").is_file():
        return str(folder)
    raise FileNotFoundError(f"incomplete model snapshot: {folder}")


class LocalParakeet:
    """Parakeet TDT 0.6B v3: the fastest engine, 25 European languages detected
    by the model itself. Loaded from the pinned local snapshot (never through
    a Hub id) and fed the audio array directly, so no ffmpeg is involved."""

    CHUNK_SECONDS = 120  # long hands-free captures are decoded in pieces

    def __init__(self, repo: str):
        self.path = resolve_pinned_snapshot(repo, ["config.json", "model.safetensors"],
                                            _complete_parakeet_snapshot)
        self._model = None
        self.last_language = None  # the model does not report one

    def _load(self):
        if self._model is None:
            import json
            import mlx.core as mx
            from mlx.utils import tree_flatten, tree_unflatten
            from parakeet_mlx.utils import from_config
            model = from_config(json.loads((Path(self.path) / "config.json").read_text()))
            model.load_weights(str(Path(self.path) / "model.safetensors"))
            model.update(tree_unflatten([(name, value.astype(mx.bfloat16))
                                         for name, value in tree_flatten(model.parameters())]))
            self._model = model
        return self._model

    def transcribe(self, samples, **_whisper_options):
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel
        model = self._load()
        audio = np.asarray(samples, dtype=np.float32)
        step = self.CHUNK_SECONDS * SAMPLE_RATE
        pieces = []
        for start in range(0, max(len(audio), 1), step):
            chunk = audio[start:start + step]
            if len(chunk) < SAMPLE_RATE // 10:
                continue
            mel = get_logmel(mx.array(chunk), model.preprocessor_config)
            pieces.append(model.generate(mel)[0].text.strip())
        return {"text": " ".join(piece for piece in pieces if piece), "segments": [], "language": None}


class LocalWhisper:
    """Resolve/download once during startup; inference only sees a local path.

    A cached snapshot of the pinned revision is used without any Hub request,
    so an already-downloaded model never touches the network."""

    def __init__(self, backend, repo: str):
        candidate = Path(repo).expanduser()
        if candidate.is_dir():
            self.path = str(candidate.absolute())
        else:
            self.path = resolve_pinned_snapshot(repo, ["*.json", "*.safetensors", "*.npz"],
                                                _complete_whisper_snapshot)
        self.backend = backend

    allowed_languages = None  # callable -> the codes Automatic may choose among
    speed = None              # callable -> "accurate" | "fast" (Settings)
    last_language = None

    def _features(self, model, samples, *, trim: bool):
        """Encoder features for one window. ``trim`` (Fast) encodes only the
        speech plus FAST_MARGIN_FRAMES instead of padding to 30 s — about 3x
        faster; experimental, so every result still passes the unsure check."""
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        mel = log_mel_spectrogram(np.asarray(samples, dtype=np.float32),
                                  n_mels=model.dims.n_mels, padding=N_SAMPLES)
        content = max(mel.shape[-2] - N_FRAMES, 1)
        if not trim:
            return model.encoder(pad_or_trim(mel[:content], N_FRAMES, axis=-2).astype(mx.float16)[None])
        frames = min(N_FRAMES, content + FAST_MARGIN_FRAMES)
        frames += frames % 2
        encoder = model.encoder
        x = nn.gelu(encoder.conv1(mel[:frames].astype(mx.float16)[None]))
        x = nn.gelu(encoder.conv2(x))
        x = x + encoder._positional_embedding[: x.shape[1]]
        for block in encoder.blocks:
            x, _, _ = block(x)
        return encoder.ln_post(x)

    @staticmethod
    def _language_probabilities(model, features) -> dict:
        """Whisper's language-detection math for any encoder context length."""
        import mlx.core as mx
        from mlx_whisper.tokenizer import get_tokenizer
        tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages)
        logits = model.logits(mx.array([[tokenizer.sot]]), features)[:, 0]
        mask = mx.full(logits.shape[-1], -mx.inf, dtype=mx.float32)
        mask[list(tokenizer.all_language_tokens)] = 0.0
        probabilities = np.array(mx.softmax(logits + mask, axis=-1))[0]
        return {code: float(probabilities[token]) for token, code in
                zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)}

    def _model(self):
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder
        return ModelHolder.get_model(self.path, mx.float16)  # the instance transcribe() uses

    def detect_language(self, samples, allowed) -> str:
        """Most likely language among ``allowed``, from the first 30 s."""
        model = self._model()
        probabilities = self._language_probabilities(model, self._features(model, samples, trim=False))
        return choose_language(probabilities, allowed)

    def one_pass(self, samples, allowed, *, language=None, trim=False):
        """A clip of at most 30 s: ONE encoder pass serves the language choice
        (when ``language`` is None) and a greedy decode. Returns (language,
        result or None); None hands the clip to the full pipeline, whose
        temperature fallback handles hard audio, using Whisper's thresholds."""
        from mlx_whisper.decoding import DecodingOptions, DecodingTask
        model = self._model()
        features = self._features(model, samples, trim=trim)
        if language is None:
            language = choose_language(self._language_probabilities(model, features), allowed)
        task = DecodingTask(model, DecodingOptions(language=language, without_timestamps=True, temperature=0.0))
        task._get_audio_features = lambda audio_features: audio_features  # already encoded
        result = task.run(features)[0]
        if len(result.tokens) >= task.sample_len - 1:
            return language, None  # budget exhausted: never paste a cut-off prefix
        if result.no_speech_prob > 0.6 and result.avg_logprob <= -1.0:
            return language, {"text": "", "language": language, "segments": []}  # Whisper's skip rule
        if result.compression_ratio > 2.4 or result.avg_logprob < -1.0:
            return language, None
        return language, {"text": result.text, "language": language, "segments": []}

    def transcribe(self, samples, **kwargs):
        kwargs["path_or_hf_repo"] = self.path
        self.last_language = None
        explicit = kwargs.get("language")
        allowed = () if explicit else tuple(self.allowed_languages() if self.allowed_languages else ())
        language = explicit or (allowed[0] if len(allowed) == 1 else None)
        fast = bool(self.speed and self.speed() == "fast")
        short = len(samples) <= 30 * SAMPLE_RATE and "initial_prompt" not in kwargs
        if short and (fast or (language is None and allowed)):
            chosen, result = self.one_pass(samples, allowed, language=language, trim=fast)
            if not explicit:
                self.last_language = chosen
            if result is not None:
                return result
            kwargs["language"] = chosen
        elif language is not None:
            kwargs["language"] = language
            if not explicit:
                self.last_language = language
        elif allowed:
            kwargs["language"] = self.last_language = self.detect_language(samples, allowed)
        return self.backend.transcribe(samples, **kwargs)


def trigger_down_in(flags: int, mask: int, device_bit: int) -> bool:
    """Is the trigger key down according to these event/source flags?"""
    if not flags & mask:
        return False
    return bool(flags & device_bit) if device_bit else True


def hotkey_release_held_seconds(state: dict, now: float) -> float | None:
    """Consume a hotkey key-up only after an accepted matching key-down.

    Key-up events no longer carry reliable modifier flags, so the event tap
    matches them by keycode alone.  That makes the stored key-down the only
    authority: without it, a plain/stray D key-up must not finish some other
    trigger's active recording.  ``None`` also covers a release intentionally
    consumed after a toggle press already finished the recording.
    """
    pressed_at = state.get("pressed_at")
    if pressed_at is None:
        state["skip_up"] = False
        return None
    state["pressed_at"] = None
    if state.get("skip_up"):
        state["skip_up"] = False
        return None
    state["skip_up"] = False
    return max(0.0, now - float(pressed_at))


def model_rewarm_due(last_finished: float, now: float, *, rewarming: bool,
                     queue_idle: bool, adaptive: bool) -> bool:
    """Whether a baseline Metal refresh should hide inside the next capture."""
    return (not adaptive and not rewarming and queue_idle
            and now - last_finished >= MODEL_REWARM_AFTER_S)


# USB-audio terminal types macOS reports for streams (CoreAudio also uses
# fourcc forms). A headset mic sits near the mouth, so it beats the built-in;
# a plain speaker's mic sits across the room, so the built-in beats it.
HEADSET_INPUT_TERMINALS = {0x0402, 0x0403, 0x0205,
                           int.from_bytes(b"hmic", "big")}
HEADPHONE_OUTPUT_TERMINALS = {0x0302, 0x0402,
                              int.from_bytes(b"hdph", "big")}
HEADSET_NAME_HINTS = ("airpod", "headphone", "headset", "earbud", "earpod",
                      "buds", "beats", "wh-", "wf-")


def _audio_devices() -> list[dict]:
    """Every input-capable device with the facts needed to choose one."""
    import ctypes
    ca = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
    cf = ctypes.CDLL(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")

    def fourcc(code: str) -> int:
        return int.from_bytes(code.encode("ascii"), "big")

    class _Addr(ctypes.Structure):
        _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32),
                    ("element", ctypes.c_uint32)]

    def get_data(obj_id, selector, scope, buf):
        addr = _Addr(selector, scope, 0)
        size = ctypes.c_uint32(ctypes.sizeof(buf))
        err = ca.AudioObjectGetPropertyData(
            ctypes.c_uint32(obj_id), ctypes.byref(addr), 0, None,
            ctypes.byref(size), ctypes.byref(buf))
        return err, size.value

    def name_of(dev_id) -> str:
        ref = ctypes.c_void_p(0)
        get_data(dev_id, fourcc("lnam"), fourcc("glob"), ref)
        if not ref.value:
            return ""
        try:
            buf = ctypes.create_string_buffer(256)
            cf.CFStringGetCString(ref, buf, 256, 0x08000100)
            return buf.value.decode("utf-8", "replace")
        finally:
            # kAudioObjectPropertyName hands out a +1 CFStringRef that the
            # caller owns; the idle device scan runs every 5 s, so an
            # unreleased name leaked a string per device per scan.
            cf.CFRelease(ref)

    def terminals(dev_id, scope) -> list[int]:
        arr = (ctypes.c_uint32 * 32)()
        _, size = get_data(dev_id, fourcc("stm#"), scope, arr)
        found = []
        for i in range(size // 4):
            value = ctypes.c_uint32(0)
            err, _ = get_data(arr[i], fourcc("term"), fourcc("glob"), value)
            if err == 0:
                found.append(int(value.value))
        return found

    devices = (ctypes.c_uint32 * 64)()
    _, size = get_data(1, fourcc("dev#"), fourcc("glob"), devices)
    result = []
    for i in range(size // 4):
        dev_id = int(devices[i])
        transport = ctypes.c_uint32(0)
        get_data(dev_id, fourcc("tran"), fourcc("glob"), transport)
        inputs = terminals(dev_id, fourcc("inpt"))
        if not inputs:
            continue
        result.append({
            "id": dev_id,
            "name": name_of(dev_id),
            "transport": int(transport.value),
            "input_terminals": inputs,
            "output_terminals": terminals(dev_id, fourcc("outp")),
        })
    return result


def _select_input_device() -> tuple[int | None, str]:
    """Pick the mic to record from: a worn headset (AirPods, wired earbuds)
    when one is connected — its mic is at your mouth — otherwise the built-in
    MacBook mic. Never the iPhone Continuity mic or a Bluetooth speaker's mic:
    those sit across the room and produced this project's deaf recordings."""

    def fourcc(code: str) -> int:
        return int.from_bytes(code.encode("ascii"), "big")

    try:
        devices = _audio_devices()
    except Exception:
        return None, "device query failed"

    builtin = None
    headset = None
    for device in devices:
        transport = device["transport"]
        if transport in (fourcc("ccwd"), fourcc("virt")):
            continue  # iPhone Continuity / virtual (Teams, Loopback)
        if transport == fourcc("bltn") and builtin is None:
            builtin = device
            continue
        name = device["name"].lower()
        looks_worn = (
            any(t in HEADSET_INPUT_TERMINALS for t in device["input_terminals"])
            or any(t in HEADPHONE_OUTPUT_TERMINALS
                   for t in device["output_terminals"])
            or any(hint in name for hint in HEADSET_NAME_HINTS)
        )
        if looks_worn and headset is None:
            headset = device
    if headset is not None:
        return headset["id"], f"{headset['name']} (headset mic)"
    if builtin is not None:
        return builtin["id"], "built-in mic"
    return None, "no suitable input device"


def _builtin_input_device() -> int | None:
    """The MacBook's own mic, whatever is worn. This is the failover target
    when the selected headset turns out to deliver only zeros, so it must
    NOT go through _select_input_device (which prefers the headset)."""
    try:
        devices = _audio_devices()
    except Exception:
        return None
    builtin_transport = int.from_bytes(b"bltn", "big")
    for device in devices:
        if device["transport"] == builtin_transport:
            return device["id"]
    return None


def _pin_input_to_builtin(node) -> int | None:
    """Point the engine's input unit at the chosen mic (headset if worn, else
    built-in) and return that device's id. Must run BEFORE engine.prepare();
    afterwards inputFormatForBus_(0) reflects the real hardware format
    (outputFormatForBus stays stale — tap with the input one, or the engine
    silently captures nothing / refuses to start)."""
    import ctypes
    device, description = _select_input_device()
    if device is None:
        return None
    log(f"  mic: {description}")
    try:
        toolbox = ctypes.CDLL(
            "/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")
        buf = ctypes.c_uint32(device)
        err = toolbox.AudioUnitSetProperty(
            ctypes.c_void_p(node.audioUnit().pointerAsInteger),
            2000,  # kAudioOutputUnitProperty_CurrentDevice
            0, 0, ctypes.byref(buf), 4)
        if err != 0:
            log(f"! mic pin rejected (err {err}) — using system default input")
    except Exception:
        log("! could not pin the mic — using system default input")
    # The selected device is the route we intended either way; the idle
    # device scan compares against it.
    return device


PREROLL_S = 0.35     # available only while a deliberately retained engine runs
RING_S = 5.0         # rolling pre-roll ring capacity
IDLE_STOP_S = 0.0    # production privacy boundary: stop the mic on key-up.
                     # The ASR model stays resident; AVAudioEngine does not.


class CaptureService:
    """Key-scoped capture with a process-resident ASR model.

    The daemon may stay alive and keep Whisper loaded, but the microphone does
    not: key-down starts AVAudioEngine and installs its input tap; key-up stops
    the engine and removes the tap.  The AVAudioEngine object itself is reused
    across captures because allocating a new one per press leaked CoreAudio
    threads on this machine.  Reusing an inert object does not hold the input
    device or illuminate the macOS microphone indicator.

    Buffer reading: floatChannelData()[0].as_buffer(N) takes an ELEMENT
    count, not bytes. Passing 4*n once read 4x past every buffer — real
    samples followed by stale memory — which surfaced as phantom duplicate
    delivery, glitch spikes up to 1e30, and stuttered audio that whisper
    transcribed as loops. The sampleTime dedupe below stays as a no-op
    safety net. The tap never touches ObjC-outward calls.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._ring: list[tuple[np.ndarray, float]] = []   # (block, its rate)
        self._ring_samples = 0
        self._active: list[np.ndarray] | None = None
        self._active_rates: list[float] = []   # per-block rates, parallel to _active
        self._stream = None
        self._stream_enqueue = None
        self._next_expected = None
        self._engine = None
        self._node = None
        self._starting = False
        self._start_began_at = 0.0
        self._waking = False
        self._engine_obj = None   # the one engine, reused for the process life
        self._config_observer = None
        self._engine_started_at = 0.0
        self._route_rebuilding = False
        self._zero_since = None
        self._silence_repairs = 0
        self._process_started_at = time.monotonic()
        self._restart_deferred_at = 0.0
        self.restart_callback = None
        self.on_start_failed = None   # runtime hook: the mic never came up under a capture
        self._release_thread = None
        self._last_device_scan = 0.0
        self._device_id = None
        self._last_block_at = time.monotonic()
        self._last_use = time.monotonic()
        self._closed = False
        self.idle_release_s = IDLE_STOP_S
        self.native_rate = float(SAMPLE_RATE)
        self.latest_rms = 0.0

    def _start_engine(self) -> None:
        # Starting and stopping the reusable AVAudioEngine object must never
        # overlap. A release racing the next press used to be able to stop the
        # newly started capture.
        with self._lifecycle_lock:
            started = self._start_engine_locked()
        if not started:
            self._report_start_failure()

    def _start_engine_locked(self) -> bool:
        """Build, pin, and start the capture engine. NEVER raises — an audio
        error must not kill the daemon (a stale tap format threw straight out
        of installTapOnBus and took the whole process down, which launchd then
        restarted in a loop while dictations silently captured nothing).
        Returns whether the engine is live afterwards; a False under an
        in-flight capture is surfaced by _report_start_failure.

        The format must be re-read on a FRESH engine AFTER pinning: a device
        switch (AirPods run at 24 kHz, built-in at 48 kHz) briefly leaves the
        old value in place and a mismatched tap format throws. Passing nil is
        not an option — it works on Bluetooth but fails on the built-in mic
        (both matrix-tested live). Retry until the device settles.

        Voice processing (AEC) stays OFF: on this macOS beta it corrupted
        speech into stuttered audio whisper read as loops. Music bleed is
        handled by the Silero VAD gate instead.
        """
        from AVFoundation import AVAudioEngine

        try:
            with self._lock:
                if self._closed:
                    return False
            for attempt in range(5):
                with self._lock:
                    if self._closed:
                        return False
                # ONE engine for the process lifetime. Allocating a new
                # AVAudioEngine per wake leaked its CoreAudio threads (43
                # threads / 10 audio threads observed after ~50 wake cycles,
                # burning 40% CPU and driving coreaudiod). Reuse and re-pin.
                if self._engine_obj is None:
                    self._engine_obj = AVAudioEngine.alloc().init()
                    self._observe_config_changes(self._engine_obj)
                engine = self._engine_obj
                node = engine.inputNode()
                try:
                    node.removeTapOnBus_(0)  # clear any tap from a past device
                except Exception:
                    pass
                self._device_id = _pin_input_to_builtin(node)
                if self._device_id is None:
                    log("! could not pin a mic — using system default input")
                fmt = node.inputFormatForBus_(0)
                rate = float(fmt.sampleRate())
                if rate <= 0:
                    time.sleep(0.25)
                    continue
                try:
                    node.installTapOnBus_bufferSize_format_block_(
                        0, 4096, fmt, self._tap)
                except Exception:
                    log(f"! tap rejected at {rate:.0f} Hz "
                        f"(try {attempt + 1}/5) — device still settling")
                    time.sleep(0.3)
                    continue
                self.native_rate = rate
                self._next_expected = None
                engine.prepare()
                ok, err = engine.startAndReturnError_(None)
                if not ok:
                    log(f"! mic engine failed to start: {err}")
                    try:
                        node.removeTapOnBus_(0)
                    except Exception:
                        pass
                    time.sleep(0.3)
                    continue
                with self._lock:
                    if self._closed:
                        try:
                            node.removeTapOnBus_(0)
                            engine.stop()
                        except Exception:
                            pass
                        return False
                self._engine = engine
                self._node = node
                self._last_block_at = time.monotonic()
                self._engine_started_at = time.monotonic()
                self._zero_since = None
                return True
            log("! could not start the mic after 5 tries")
            return False
        except Exception as exc:
            log(f"! mic engine error: {str(exc)[:150]}")
            return False
        finally:
            self._starting = False

    def _report_start_failure(self) -> None:
        """The engine never came up: stop pretending. Clear the waking flag
        so the overlay cannot flip to "mic live", and if a capture is waiting
        on this engine hand it to the runtime hook, which ends the gesture
        and tells the user once. Idle failures (the device scan switching
        mics) stay quiet: the next press retries and reports then."""
        with self._lock:
            self._waking = False
            pending = (self._active is not None and self._engine is None
                       and not self._closed)
        handler = self.on_start_failed
        if pending and handler is not None:
            try:
                handler()
            except Exception as exc:
                log(f"! mic failure handler failed: {str(exc)[:120]}")

    def _release_engine(self, *, only_if_idle: bool = False) -> bool:
        with self._lifecycle_lock:
            return self._release_engine_locked(only_if_idle=only_if_idle)

    def _release_engine_locked(self, *, only_if_idle: bool = False) -> bool:
        """Stop capture but KEEP the engine object — see _start_engine: a new
        AVAudioEngine per wake leaks CoreAudio threads and CPU forever.

        Every teardown step runs even when an earlier one raises: a tap
        removal that threw used to skip engine.stop(), and with the handle
        already cleared nothing ever stopped that engine — idle Sotto kept
        the microphone open. Returns True only when the engine actually
        stopped; otherwise the handle stays so the next tick retries."""
        with self._lock:
            if only_if_idle and self._active is not None:
                return False
            engine, self._engine = self._engine, None
            node, self._node = self._node, None
            self._ring.clear()
            self._ring_samples = 0
            self._starting = False  # never leave a wake blocked behind us
        if engine is None:
            return False
        try:
            if node is not None:
                node.removeTapOnBus_(0)
        except Exception as exc:
            log(f"! mic tap not removed: {str(exc)[:120]}")
        try:
            engine.stop()
        except Exception as exc:
            log(f"! mic engine not stopped: {str(exc)[:120]}")
            with self._lock:
                if not self._closed:
                    self._engine, self._node = engine, node
            return False
        return True

    def _observe_config_changes(self, engine) -> None:
        """A route change under a live engine (dock unplugged, output device
        switched, ...) can leave it 'running' while every buffer is all
        zeros — VAD 0% on every capture until a manual restart (observed live
        2026-08-12). The system announces the change; rebuild instead of
        playing dead."""
        try:
            from AVFoundation import AVAudioEngineConfigurationChangeNotification
            from Foundation import NSNotificationCenter
            self._config_observer = (
                NSNotificationCenter.defaultCenter()
                .addObserverForName_object_queue_usingBlock_(
                    AVAudioEngineConfigurationChangeNotification, engine,
                    None, lambda note: self._on_config_change()))
        except Exception as exc:
            log(f"! route-change watch unavailable: {str(exc)[:80]}")

    def _on_config_change(self) -> None:
        with self._lock:
            # Ignore while idle (the next wake re-reads the route), while a
            # start is already in flight, and during the burst of
            # notifications our own re-pinning fires right after a start.
            if (self._closed or self._engine is None or self._starting
                    or self._route_rebuilding
                    or time.monotonic() - self._engine_started_at < 2.0):
                return
            self._route_rebuilding = True
        log("! audio route changed — rebuilding mic")

        def work() -> None:
            try:
                self._release_engine()
                with self._lock:
                    self._starting = True
                    self._start_began_at = time.monotonic()
                self._start_engine()
            finally:
                self._route_rebuilding = False

        threading.Thread(target=work, daemon=True).start()

    def tick(self) -> None:
        """Poller hook, once per second: restart an unexpectedly-dead engine
        while recording; release the mic (indicator off) after idle."""
        try:
            with self._lock:
                if self._closed:
                    return
                active = self._active is not None
                idle_for = time.monotonic() - self._last_use
                engine = self._engine
                starting = self._starting
            if engine is None or starting:
                return
            stalled = time.monotonic() - self._last_block_at > 1.5
            if (self._zero_since is not None and self._silence_repairs < 2
                    and time.monotonic() - self._zero_since > 3.0):
                self._drop_dead_capture()
                self._repair_dead_route(capturing=True)
                return
            if active and (not engine.isRunning() or stalled):
                reason = "stopped by the system" if not engine.isRunning() \
                    else "stopped delivering audio"
                log(f"! mic {reason} — restarting mid-recording")
                self._release_engine()
                self._start_engine()
                return
            elif not active and time.monotonic() - self._last_device_scan > 5.0:
                # Plugging in AirPods (or unplugging them) should switch the
                # mic without a restart. Only while idle — never mid-recording —
                # and only every 5s: a full CoreAudio enumeration costs ~4ms.
                self._last_device_scan = time.monotonic()
                wanted, description = _select_input_device()
                if wanted is not None and wanted != self._device_id:
                    log(f"! input device changed — switching to {description}")
                    self._release_engine()
                    self._start_engine()
                    return
            # Negative = never, exactly as release_soon reads it. only_if_idle:
            # "idle" was decided under the lock a moment ago, and a press that
            # lands in between must keep the engine it is about to record on.
            if (not active and self.idle_release_s >= 0
                    and idle_for > self.idle_release_s):
                if self._release_engine(only_if_idle=True):
                    log("○ mic released (idle) — wakes on next press")
        except Exception as exc:
            log(f"! mic health check failed: {str(exc)[:120]}")

    def _repair_dead_route(self, *, capturing: bool) -> None:
        """Escalating repair for a capture path delivering bit-exact zeros.

        A live mic always carries a noise floor, so exact zeros mean the route
        is dead — not that the room is quiet. 2026-09-20: AirPods dropped and
        reconnected under a new CoreAudio device id, and from then on EVERY
        capture in the running process came back bit-exact zero — including
        ones pinned to the built-in mic — while ffmpeg recorded those same
        AirPods at peak 18609 and a fresh python process running this very
        class captured normally. The device was fine; this process's input
        unit was wedged.

        Stopping and starting the engine does not clear it (the 11:03 log
        rebuilt straight back onto zeros), because the AVAudioEngine object
        is reused for the process lifetime and carries the wedged unit with
        it. So the first repair throws that object away, and the second
        throws the process away.

        Each rung fires once per silence run: the counter resets only when
        real audio arrives (_tap), never on engine start, or a genuinely
        muted input would rebuild forever."""
        if self._silence_repairs == 0:
            self._silence_repairs = 1
            log("! mic delivering only silence — rebuilding the audio engine")
            if capturing:
                self._rebuild_engine(start=True)
            else:
                # Key-up runs straight off the Quartz gesture tap, and
                # CoreAudio teardown can block it long enough for macOS to
                # disable the tap (2026-09-15). Defer it exactly like
                # release_soon does. Never start an engine here either: idle
                # sotto must not hold the microphone (--idle-release 0), so
                # the next press is what builds the fresh one.
                threading.Thread(target=self._rebuild_engine,
                                 daemon=True).start()
            return
        # Consume the last rung only when a relaunch actually goes out. The
        # age guard can refuse one, and burning the rung there would leave a
        # deaf process with nothing left to try until real audio arrives —
        # which is exactly what never happens on a dead route.
        if self._restart_app():
            self._silence_repairs = 2

    def _rebuild_engine(self, start: bool = False) -> None:
        """Drop the reused AVAudioEngine, optionally starting a fresh one."""
        self._release_engine()
        with self._lifecycle_lock:
            self._forget_engine_object()
        if start:
            self._start_engine()

    def _forget_engine_object(self) -> None:
        """Drop the reused AVAudioEngine so the next start allocates a fresh
        one, taking its wedged HAL input unit with it. Reusing the object is
        the rule (a new engine per wake leaked CoreAudio threads); one
        allocation per dead route is not the per-wake churn that rule
        prevents."""
        engine, self._engine_obj = self._engine_obj, None
        observer, self._config_observer = self._config_observer, None
        if observer is not None:
            try:
                from Foundation import NSNotificationCenter
                NSNotificationCenter.defaultCenter().removeObserver_(observer)
            except Exception as exc:
                log(f"! route-change watch not removed: {str(exc)[:80]}")
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def _restart_app(self) -> bool:
        """Last rung: launchd relaunches a fresh process. A fresh process is
        the one thing observed to cure this (2026-09-20). Only for a process
        old enough that restarting cannot become a loop — a young one just
        came back from a restart, so if it is still deaf the restart is not
        the cure and the user needs to know.

        Returns whether a relaunch was actually kicked off; a refusal leaves
        the rung armed for when the process IS old enough."""
        now = time.monotonic()
        if now - self._process_started_at < SILENCE_RESTART_AFTER_S:
            # The caller retries every few seconds until then — say it once a
            # minute, not once a tick.
            if now - self._restart_deferred_at > 60.0:
                self._restart_deferred_at = now
                log("! mic still silent in a freshly started process — check "
                    "the mic in System Settings; 'Restart sotto' is in the menu")
            return False
        log("! mic still silent after a fresh audio engine — restarting sotto")
        try:
            if self.restart_callback is None:
                log("! no runtime restart handler; quit and launch Sotto again")
                return False
            return bool(self.restart_callback())
        except Exception as exc:
            log(f"! restart failed: {str(exc)[:120]}")
            return False

    def _drop_dead_capture(self) -> None:
        """Throw away an in-flight capture that is bit-exact zeros end to end.

        Those frames came from the route we are failing over from, and a
        capture that opens with digital silence is exactly what Whisper turns
        into a repetition loop. Only drop when EVERY frame so far is zero:
        audio recorded before a live route died is real speech and must
        survive the rebuild."""
        with self._lock:
            if self._active is None:
                return
            if all(not block.any() for block in self._active):
                self._active = []
                self._active_rates = []

    def _tap(self, buffer, when) -> None:
        n = int(buffer.frameLength())
        if n <= 0:
            return
        start = int(when.sampleTime())
        ptr = buffer.floatChannelData()[0]
        data = np.frombuffer(ptr.as_buffer(n), dtype=np.float32)
        if self._next_expected is not None and start < self._next_expected:
            skip = self._next_expected - start
            if skip >= n:
                return  # pure duplicate buffer
            data = data[skip:]
            start += skip
        self._next_expected = start + len(data)
        block = data.copy()
        self.latest_rms = float(np.sqrt(np.mean(block**2)))
        # Exactly-zero buffers are a dead route, not a quiet room — a live
        # mic always carries noise floor. Timed here, acted on in tick().
        if self.latest_rms == 0.0:
            if self._zero_since is None:
                self._zero_since = time.monotonic()
        else:
            self._zero_since = None
            # Real audio: the route recovered, so the repair ladder rearms.
            # This is the ONLY place the counter resets — resetting it on
            # engine start instead made a permanently silent input rebuild
            # forever.
            self._silence_repairs = 0
        self._waking = False  # audio is flowing now
        self._last_block_at = time.monotonic()
        # trust the buffer's own rate over the format read at start-up
        try:
            self.native_rate = float(buffer.format().sampleRate())
        except Exception:
            pass
        rate = self.native_rate
        with self._lock:
            if self._active is not None:
                # Each block keeps its own rate: a route change mid-recording
                # (built-in 48 kHz -> AirPods 24 kHz) rebuilds the engine
                # under a live capture, and end() must not read the earlier
                # blocks at the later device's rate.
                self._active.append(block)
                self._active_rates.append(rate)
                if self._stream is not None:
                    # Only queue references on the tap. Resampling and ASR run
                    # on the existing serialized transcription worker.
                    self._stream_enqueue(("stream-audio", self._stream, block, rate))
            else:
                self._ring.append((block, rate))
                self._ring_samples += len(block)
                limit = int(RING_S * rate)
                while self._ring_samples > limit and len(self._ring) > 1:
                    self._ring_samples -= len(self._ring.pop(0)[0])

    def begin(self, *, stream=None, enqueue=None) -> None:
        cold_start = False
        with self._lock:
            if self._closed:
                return False
            if self._stream is not None:
                self._stream.cancelled.set()
                self._stream_enqueue(("stream-close", self._stream))
            preroll: list[np.ndarray] = []
            preroll_rates: list[float] = []
            needed = int(PREROLL_S * self.native_rate)
            collected = 0
            for block, rate in reversed(self._ring):
                preroll.insert(0, block)
                preroll_rates.insert(0, rate)
                collected += len(block)
                if collected >= needed:
                    break
            self._active = preroll
            self._active_rates = preroll_rates
            self._stream, self._stream_enqueue = stream, enqueue
            if stream is not None:
                for block, rate in zip(preroll, preroll_rates):
                    enqueue(("stream-audio", stream, block, rate))
            self._last_use = time.monotonic()
            # A start that never completed must not block every future wake —
            # treat a stale "starting" flag as dead and try again.
            stale_start = (self._starting and
                           time.monotonic() - self._start_began_at > 8.0)
            if self._engine is None and (not self._starting or stale_start):
                if stale_start:
                    log("! previous mic start never finished — retrying")
                self._starting = True
                self._start_began_at = time.monotonic()
                self._waking = True
                cold_start = True
        if cold_start:  # wake from idle — off-thread, the tap must never wait
            threading.Thread(target=self._start_engine, daemon=True).start()
        return cold_start

    def is_waking(self) -> bool:
        """True until audio actually flows after a cold start (up to ~5s)."""
        return self._waking

    def is_live(self) -> bool:
        """An engine is up; False after a start that failed every try."""
        with self._lock:
            return self._engine is not None

    def end(self, *, include_stream=False):
        with self._lock:
            frames, self._active = self._active or [], None
            rates, self._active_rates = self._active_rates, []
            stream, self._stream = self._stream, None
            self._stream_enqueue = None
            self._last_use = time.monotonic()
        if not frames:
            empty = np.zeros(0, dtype=np.float32)
            return (empty, stream) if include_stream else empty
        # Blocks placed directly (tests, old callers) carry no rate: they are
        # at the current one, which is also what the caller reads next.
        rates = rates + [self.native_rate] * (len(frames) - len(rates))
        samples = join_capture_blocks(frames, rates, self.native_rate)
        # A hold shorter than the in-capture silence watch (3s) ends before
        # tick() can react, so short presses would keep landing on the same
        # wedged input unit forever (2026-09-20: 1.37s and 1.88s holds, peak
        # 0, "ощ" x223). Repair at key-up too and the NEXT press is clean.
        # One second of exact zeros is past any Bluetooth warm-up blip and
        # cannot come from a live mic.
        if (not samples.any() and self._silence_repairs < 2
                and len(samples) / max(self.native_rate, 1.0) >= 1.0):
            self._repair_dead_route(capturing=False)
        return (samples, stream) if include_stream else samples

    def abort(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.cancelled.set()
                self._stream_enqueue(("stream-close", self._stream))
            self._stream = self._stream_enqueue = None
            self._active = None
            self._active_rates = []
            self._last_use = time.monotonic()

    def is_active(self) -> bool:
        with self._lock:
            return self._active is not None

    def live_text(self) -> str:
        """The streaming engine's words so far, for display only."""
        stream = self._stream
        return stream.live_text() if stream is not None and hasattr(stream, "live_text") else ""

    def release_soon(self) -> None:
        """Release the input after key-up without blocking the gesture tap.

        Production uses zero delay. A positive value remains available for
        diagnostics; a negative value deliberately retains the engine.
        """
        if self.idle_release_s < 0:
            return

        def work() -> None:
            if self.idle_release_s:
                time.sleep(self.idle_release_s)
            if self._release_engine(only_if_idle=True):
                log("○ mic released")

        threading.Thread(target=work, daemon=True).start()

    def shutdown(self) -> None:
        """Immediately release mic resources during process shutdown."""
        self.abort()
        with self._lock:
            self._closed = True
            self._active = None
            if self._release_thread is not None:
                return  # one teardown; repeated drain passes must not pile up
            # CoreAudio teardown can block; it cannot hold up the process drain.
            self._release_thread = threading.Thread(target=self._release_engine, daemon=True)
        self._release_thread.start()

    def wait_released(self, timeout: float) -> bool:
        """Bounded wait for the shutdown teardown; never blocks on a wedged HAL."""
        thread = self._release_thread
        if thread is not None:
            thread.join(timeout)
        return thread is None or not thread.is_alive()


def join_capture_blocks(frames: list, rates: list, target_rate: float) -> np.ndarray:
    """Join tap blocks into one signal at target_rate.

    A route change mid-recording (built-in mic at 48 kHz, then AirPods at
    24 kHz) leaves blocks of both rates in one capture. Reading them all at
    the last device's rate stretched the earlier speech to twice its length
    and halved its pitch; instead every run of same-rate blocks is resampled
    on its own before the runs are concatenated."""
    runs: list[tuple[list, float]] = []
    for block, rate in zip(frames, rates):
        if runs and runs[-1][1] == rate:
            runs[-1][0].append(block)
        else:
            runs.append(([block], rate))
    pieces = []
    for blocks, rate in runs:
        samples = np.concatenate(blocks).reshape(-1)
        if rate != target_rate and rate > 0 and target_rate > 0:
            from fractions import Fraction

            from scipy.signal import resample_poly
            ratio = Fraction(int(target_rate), int(rate)).limit_denominator(1000)
            samples = resample_poly(samples, ratio.numerator,
                                    ratio.denominator).astype(np.float32)
        pieces.append(samples)
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)


def prepare_for_whisper(raw: np.ndarray, native_rate: float) -> np.ndarray:
    """Native capture -> 16 kHz normalized whisper input: polyphase resample
    (linear interp aliased audibly and cost accuracy), then robust
    percentile normalization in both directions with glitch-spike clipping."""
    if not len(raw):
        return raw
    if native_rate <= 0:
        log("! capture had no valid sample rate — dropped")
        return np.zeros(0, dtype=np.float32)
    if native_rate != SAMPLE_RATE:
        from fractions import Fraction

        from scipy.signal import resample_poly
        ratio = Fraction(SAMPLE_RATE, int(native_rate)).limit_denominator(1000)
        raw = resample_poly(raw, ratio.numerator,
                            ratio.denominator).astype(np.float32)
    scale = float(np.percentile(np.abs(raw), 99.9))
    peak = float(np.abs(raw).max())
    if peak > 2 * max(scale, 1e-6):
        log(f"! input glitch spike clipped (peak {peak:.1f}, "
            f"speech level {scale:.3f})")
    if scale > 1e-4:
        raw = raw * (0.5 / scale)
    return np.clip(raw, -1.0, 1.0)


DEAD_ROUTE_TEXT = "[dead microphone]"


def is_dead_route(samples: np.ndarray) -> bool:
    """Bit-exact silence is a dead capture path, not a quiet room.

    CaptureService._tap already treats exact-zero buffers as a dead route:
    a live mic always carries a noise floor. Quiet whispered dictation is
    non-zero (2026-08-13/14) and must still reach the ASR. Exact zeros must
    not: Whisper turns them into repetition loops (2026-09-20: 40.96s of
    24 kHz zeros -> "ощ" x223).
    """
    if samples is None or len(samples) == 0:
        return True
    return float(np.max(np.abs(samples))) == 0.0


def asr_skip_reason(samples: np.ndarray) -> str | None:
    """Why this capture must not be sent to the ASR, or None to transcribe."""
    if is_dead_route(samples):
        return "dead microphone (all-zero capture)"
    return None


def collapse_silence(samples: np.ndarray) -> np.ndarray:
    """Shorten long silent stretches in a capture. Dead air is what whisper
    hallucinates on during long dictations ("Okay. Okay. Okay."), and it
    slows inference. The threshold adapts to the recording's own noise floor,
    and up to 1s of every pause is kept so sentence rhythm survives."""
    block = int(SAMPLE_RATE * 0.1)
    n_blocks = len(samples) // block
    if n_blocks < 20:
        return samples
    body = samples[:n_blocks * block].reshape(n_blocks, block)
    rms = np.sqrt(np.mean(body**2, axis=1))
    # adapt to the noise floor, but never above half the loud level — on
    # uniform wall-to-wall speech nothing must qualify as silence
    threshold = max(0.002, min(float(np.percentile(rms, 10)) * 2.0,
                               float(np.percentile(rms, 90)) * 0.5))
    keep = np.ones(n_blocks, dtype=bool)
    run = 0
    for i, level in enumerate(rms):
        run = run + 1 if level < threshold else 0
        if run > 10:  # keep at most 1.0s of any silent stretch
            keep[i] = False
    if keep.all():
        return samples
    return np.concatenate([body[keep].reshape(-1), samples[n_blocks * block:]])


class GestureEngine:
    """Press/release edges -> hold / double-tap / discard decisions.

    Every key deadline carries an epoch token: any newer key event invalidates
    it, so a Timer callback that already fired but is waiting on the lock can
    never act on stale state. The hands-free watchdog is the exception: it
    belongs to the capture, not the key epoch, so a stop tap whose release is
    lost cannot disarm it — only ending or re-arming hands-free does.
    Callbacks run OUTSIDE the lock, one at a time, in the order their
    decisions were made: a decision queues its actions while it still holds
    the lock, and whichever thread finds nobody draining runs the queue (see
    _drain). The second press of a double-tap is classified on its RELEASE:
    held short = arm hands-free, held long = it was a deliberate push-to-talk.
    """

    def __init__(self, on_start, on_finish, on_discard, on_hands_free=None) -> None:
        self._on_start = on_start
        self._on_finish = on_finish
        self._on_discard = on_discard
        self._on_hands_free = on_hands_free
        self._lock = threading.Lock()
        self._epoch = 0
        self._hands_free_token = 0
        self._recording = False
        self._hands_free = False
        self._tap_pending = False      # a lone short tap awaits its verdict
        self._second_candidate = False
        self._pressed_at = 0.0
        self._last_tap_at = -1e9
        self._pending: list = []       # actions decided, not yet run (in order)
        self._draining = False

    def snapshot(self) -> tuple[bool, bool]:
        """(recording, hands_free) — for the lost-release resync poller."""
        with self._lock:
            return self._recording, self._hands_free

    def _queue(self, callbacks) -> None:
        """Caller holds the lock: actions join the queue in decision order."""
        self._pending.extend(callbacks)

    def _drain(self) -> None:
        """Run queued actions one at a time, in order, outside the lock.

        The thread that finds nobody draining becomes the drainer and runs
        everything queued, including what other threads add meanwhile; a
        thread that arrives while another is draining has already queued its
        actions under the lock and returns at once. So the event tap never
        waits on a timer thread's callback, and a discard decided before a
        start (the tap-expiry timer racing the next press) can never run
        after it and kill the new capture."""
        with self._lock:
            if self._draining:
                return
            self._draining = True
        while True:
            with self._lock:
                if not self._pending:
                    self._draining = False
                    return
                callback = self._pending.pop(0)
            try:
                callback()
            except Exception as exc:
                log(f"! gesture callback failed: {str(exc)[:120]}")
                with self._lock:
                    self._recording = False
                    self._hands_free = False

    def pressed(self) -> None:
        fires = []
        with self._lock:
            self._epoch += 1
            now = time.monotonic()
            candidate = self._tap_pending and now - self._last_tap_at < DOUBLE_TAP_WINDOW_S
            stale_blip = self._tap_pending and not candidate
            self._tap_pending = False
            self._second_candidate = candidate and self._recording and not self._hands_free
            if stale_blip and self._recording and not self._hands_free:
                # the discard timer ran late — drop the old blip, start fresh
                self._recording = False
                fires.append(self._on_discard)
            if not self._recording:
                self._recording = True
                fires.append(self._on_start)
            self._pressed_at = now
            self._queue(fires)
        self._drain()

    def released(self) -> None:
        fires = []
        with self._lock:
            now = time.monotonic()
            held_for = now - self._pressed_at
            if not self._recording:
                pass
            elif self._hands_free:
                self._hands_free = False
                self._recording = False
                fires.append(self._on_finish)
            elif held_for >= HOLD_THRESHOLD_S:
                self._recording = False
                fires.append(self._on_finish)
            elif self._second_candidate:
                self._hands_free = True
                self._epoch += 1
                self._arm_hands_free_watchdog()
                log("● hands-free (tap again to stop)")
                if self._on_hands_free is not None:
                    fires.append(self._on_hands_free)
            else:
                self._last_tap_at = now
                self._tap_pending = True
                self._epoch += 1
                self._schedule(DOUBLE_TAP_WINDOW_S, self._expire_tap)
            self._second_candidate = False
            self._queue(fires)
        self._drain()

    def force_start(self) -> bool:
        """Menu-driven start: arm a hands-free recording without any key
        gesture (remote desktops send synthetic keys without the physical
        device bits, so the hotkey can't work through them)."""
        fires = []
        with self._lock:
            if self._recording:
                return False
            self._epoch += 1
            self._tap_pending = False
            self._recording = True
            self._hands_free = True
            self._arm_hands_free_watchdog()
            fires.append(self._on_start)
            if self._on_hands_free is not None:
                fires.append(self._on_hands_free)
            self._queue(fires)
        self._drain()
        return True

    def force_finish(self, finish=None) -> bool:
        """Menu escape hatch: end any in-flight recording as a normal finish.

        `finish` replaces on_finish for this one decision. It may run after
        this returns (another thread is draining), so anything the caller
        wants that finish to know must travel inside it — never in state the
        caller sets around this call."""
        fires = []
        with self._lock:
            self._epoch += 1
            self._tap_pending = False
            was_recording = self._recording
            if self._recording:
                self._recording = False
                self._hands_free = False
                fires.append(finish or self._on_finish)
            self._queue(fires)
        self._drain()
        return was_recording

    def chorded(self) -> None:
        """Another key joined a held trigger (⌘-Tab, ⌥-letter): that is a
        shortcut, not dictation. Drop the push-to-talk capture; hands-free
        keeps recording because its trigger is no longer held."""
        fires = []
        with self._lock:
            if not self._recording or self._hands_free:
                return
            self._epoch += 1
            self._tap_pending = False
            self._second_candidate = False
            self._recording = False
            fires.append(self._on_discard)
            self._queue(fires)
        self._drain()

    def force_reset(self) -> None:
        """Sleep/lock/tap-disable recovery: drop any in-flight recording."""
        fires = []
        with self._lock:
            self._epoch += 1
            self._tap_pending = False
            self._second_candidate = False
            if self._recording:
                self._recording = False
                self._hands_free = False
                fires.append(self._on_discard)
            self._queue(fires)
        self._drain()

    def _schedule(self, delay: float, handler, token: int | None = None) -> None:
        timer = threading.Timer(delay, handler,
                                args=(self._epoch if token is None else token,))
        timer.daemon = True
        timer.start()

    def _arm_hands_free_watchdog(self) -> None:
        """Caller holds the lock. The watchdog is bound to THIS hands-free
        capture: pressing the key again must not disarm it (a stop tap whose
        release is lost used to leave the mic open forever), and a re-armed
        hands-free gets a fresh token so the old deadline cannot end it."""
        self._hands_free_token += 1
        self._schedule(HANDS_FREE_MAX_S, self._hands_free_timeout,
                       self._hands_free_token)

    def _expire_tap(self, epoch: int) -> None:
        fires = []
        with self._lock:
            if epoch != self._epoch or not self._recording or self._hands_free:
                return
            self._recording = False
            self._tap_pending = False
            fires.append(self._on_discard)
            self._queue(fires)
        self._drain()

    def _hands_free_timeout(self, token: int) -> None:
        fires = []
        with self._lock:
            if token != self._hands_free_token or not self._hands_free:
                return
            self._hands_free = False
            self._recording = False
            log(f"! hands-free watchdog ({HANDS_FREE_MAX_S:.0f}s) — finishing")
            fires.append(self._on_finish)
            self._queue(fires)
        self._drain()


# -- clipboard injection (main thread only) ---------------------------------

_restore_generation = 0
_pending_restore: dict | None = None  # {"own_count", "snapshot"} of the restore not yet run


_NO_SPACE_AFTER = "([{\"'“‘/-\n\t"
_NO_SPACE_BEFORE = ".,;:!?)]}\"'”’%"


def compose_insertion(text: str, spacing: str, before: str | None) -> str:
    """The exact text to insert for a dictation.

    ``before`` is the character left of the caret ("" at the start of a
    field, None when the app does not expose it). "smart" adds a leading space
    only when the text would otherwise touch the previous word, and falls back
    to "trailing" when the context is unknown; "trailing" keeps consecutive
    dictations apart ('Hey what's upfirstI'm bursting'); "none" inserts as-is."""
    if not text or spacing == "none":
        return text
    if spacing == "smart" and before is not None:
        touching = bool(before) and not before.isspace() and before not in _NO_SPACE_AFTER
        return (" " + text) if touching and text[0] not in _NO_SPACE_BEFORE else text
    return text if text[-1].isspace() else text + " "


def character_before_caret(timeout: float = 0.15) -> str | None:
    """The character before the caret via Accessibility, or None when the
    focused app does not expose it (never blocks longer than ``timeout``)."""
    try:
        import ApplicationServices as AS
        system = AS.AXUIElementCreateSystemWide()
        AS.AXUIElementSetMessagingTimeout(system, timeout)
        error, focused = AS.AXUIElementCopyAttributeValue(system, AS.kAXFocusedUIElementAttribute, None)
        if error or focused is None:
            return None
        AS.AXUIElementSetMessagingTimeout(focused, timeout)
        error, selection = AS.AXUIElementCopyAttributeValue(focused, AS.kAXSelectedTextRangeAttribute, None)
        if error or selection is None:
            return None
        ok, (location, _length) = AS.AXValueGetValue(selection, AS.kAXValueCFRangeType, None)
        if not ok or location < 0:
            return None
        if location == 0:
            return ""
        window = AS.AXValueCreate(AS.kAXValueCFRangeType, AS.CFRange(location - 1, 1))
        error, previous = AS.AXUIElementCopyParameterizedAttributeValue(
            focused, AS.kAXStringForRangeParameterizedAttribute, window, None)
        if error or previous is None:
            return None
        return str(previous)[-1:] or None
    except Exception:
        return None


def post_own_event(event) -> None:
    """Tag a synthetic key event as Sotto's own and post it into the session.

    The event tap in run() skips tagged events, so a ⌘V or ⌘Z Sotto posts can
    never be read as a chord that discards a live dictation, nor as the user
    typing (which would disarm "scratch that")."""
    Quartz.CGEventSetIntegerValueField(event, Quartz.kCGEventSourceUserData, SOTTO_EVENT_TAG)
    Quartz.CGEventPost(Quartz.kCGSessionEventTap, event)


def is_own_event(event) -> bool:
    """Did Sotto post this key event itself (see post_own_event)?"""
    return Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceUserData) == SOTTO_EVENT_TAG


def post_command_key(keycode: int) -> None:
    """Press and release ⌘+key (9 = V to paste, 6 = Z to undo) as Sotto's own events."""
    for key_down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(None, keycode, key_down)
        Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskCommand)
        post_own_event(event)


def type_text(text: str) -> None:
    """Type Unicode text with synthetic key events; the clipboard is untouched."""
    characters = list(text)
    for start in range(0, len(characters), 16):  # small chunks: apps drop long ones
        chunk = "".join(characters[start:start + 16])
        units = len(chunk.encode("utf-16-le")) // 2
        for key_down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(None, 0, key_down)
            Quartz.CGEventKeyboardSetUnicodeString(event, units, chunk)
            post_own_event(event)
        time.sleep(0.004)


_secure_input_probe = None


def secure_input_active() -> bool:
    """Is a password field (or another secure-input owner) active right now?

    PyObjC does not bridge Carbon's IsSecureEventInputEnabled, so this calls
    it directly; an unavailable probe reports False rather than blocking."""
    global _secure_input_probe
    try:
        if _secure_input_probe is None:
            import ctypes
            carbon = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Carbon.framework/Carbon")
            carbon.IsSecureEventInputEnabled.restype = ctypes.c_bool
            _secure_input_probe = carbon.IsSecureEventInputEnabled
        return bool(_secure_input_probe())
    except Exception:
        return False


def frontmost_pid() -> int | None:
    """The process a synthetic key event would reach right now (NSWorkspace)."""
    try:
        import AppKit
        app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
        return int(app.processIdentifier()) if app is not None else None
    except Exception:
        return None


def own_window_without_text_field() -> bool:
    """Is Sotto itself the active app with no text field to take a paste?

    Sotto's alerts activate it (ui._alert), and callAfter still runs during
    their modal loop, so a Cmd-V would land on an OK-only alert and vanish
    (N24). The correction editor's text view still takes dictation. Main
    thread only; an unavailable probe reports False rather than blocking."""
    try:
        import AppKit
        app = AppKit.NSApp
        if app is None or not app.isActive():
            return False
        window = app.keyWindow()
        responder = window.firstResponder() if window is not None else None
        return responder is None or not responder.isKindOfClass_(AppKit.NSText)
    except Exception:
        return False


def inject(text: str, *, insert_mode: str = "paste", spacing: str = "trailing") -> bool:
    """Insert at the cursor. "paste": full-pasteboard snapshot, synthetic
    cmd-V, then a changeCount-guarded restore so a user copy in the window
    always wins. "type": synthetic Unicode typing, clipboard untouched.

    Returns True when the text was handed to the app, False when secure input
    declined it — only a real insert may arm "scratch that"."""
    global _restore_generation, _pending_restore
    if secure_input_active():
        log("! secure input active — not inserting")
        return False

    text = compose_insertion(text, spacing, character_before_caret() if spacing == "smart" else None)
    if insert_mode == "type":
        type_text(text)
        return True

    pasteboard = NSPasteboard.generalPasteboard()
    pending = _pending_restore
    if pending is not None and pasteboard.changeCount() == pending["own_count"]:
        # The pasteboard still holds Sotto's previous dictation and its restore
        # has not run yet: carry the user's original forward instead of
        # snapshotting our own text (two pastes inside the restore window).
        snapshot = pending["snapshot"]
    else:
        snapshot = []
        for item in (pasteboard.pasteboardItems() or []):
            snapshot.append([(t, item.dataForType_(t)) for t in item.types()])
    pasteboard.clearContents()
    pasteboard.setString_forType_(text, NSPasteboardTypeString)
    pasteboard.setData_forType_(b"", TRANSIENT_PASTEBOARD_TYPE)
    own_count = pasteboard.changeCount()
    post_command_key(9)  # ⌘V

    _restore_generation += 1
    generation = _restore_generation
    _pending_restore = {"own_count": own_count, "snapshot": snapshot}

    def queue_restore() -> None:
        from PyObjCTools import AppHelper
        AppHelper.callAfter(_restore_clipboard, generation, own_count, snapshot)

    timer = threading.Timer(RESTORE_DELAY_S, queue_restore)
    timer.daemon = True
    timer.start()
    return True


def _restore_clipboard(generation: int, own_count: int, snapshot: list) -> None:
    """Runs on the main thread (AppHelper.callAfter), like inject() itself, so
    the pending-restore bookkeeping needs no lock."""
    global _pending_restore
    if generation != _restore_generation:
        return  # a newer injection owns the pasteboard now (and carried this snapshot if it was still ours)
    _pending_restore = None
    pasteboard = NSPasteboard.generalPasteboard()
    if pasteboard.changeCount() != own_count:
        return  # the user copied something meanwhile — their copy wins
    pasteboard.clearContents()
    items = []
    for entry in snapshot:
        item = NSPasteboardItem.alloc().init()
        for pb_type, data in entry:
            if data is not None:
                item.setData_forType_(data, pb_type)
        items.append(item)
    if items:
        pasteboard.writeObjects_(items)


class DeliveryQueue:
    """One FIFO for everything Sotto hands to the frontmost app — finished
    dictations and "scratch that" undos — in capture order, so a newer result
    can never overtake an older one.

    Nothing is posted while the trigger or any modifier is physically held: a
    synthetic ⌘V under a held modifier is a different shortcut in the app, and
    during a hold macOS emits phantom flag edges on the trigger keycode that
    chop the live dictation. The head of the queue waits as long as a
    recording is active (the release delivers it); with no recording, a key
    held longer than DELIVERY_WAIT_MAX_S drops the pending text — it is already
    in History — with one note. While Sotto's own alert is in front
    (own_window_front), the head waits without a bound: a Cmd-V there would
    land nowhere, and the user's next app switch or click on OK delivers it.

    Every method runs on the main thread (deliver_call / AppHelper.callAfter);
    the poll timer only bounces back there, so there is no lock.

    on_done runs exactly once for every item the queue takes — when it is
    delivered (even if the insert raises), dropped after a long hold, or
    cleared at shutdown — so run() can count it in PendingDeliveries from the
    moment deliver_call schedules it until it leaves the queue.
    """

    def __init__(self, *, insert, undo_keys, keys_held, recording, frontmost_pid, keydowns,
                 secure_input, call_after, note, log=log, shutdown_requested=lambda: False,
                 clock=time.monotonic, timer=threading.Timer, on_done=lambda: None,
                 own_window_front=lambda: False) -> None:
        self._insert = insert            # (text) -> bool: did the text reach the app?
        self._undo_keys = undo_keys      # () -> None: press the app's own ⌘Z
        self._keys_held = keys_held      # () -> bool: trigger or modifier physically down
        self._recording = recording      # () -> bool: a dictation is being captured
        self._frontmost_pid = frontmost_pid  # () -> pid of the app a key event reaches
        self._keydowns = keydowns        # () -> count of the user's real key-downs so far
        self._secure_input = secure_input
        self._call_after = call_after
        self._note = note                # (title, message) -> None: one user-visible note
        self._log = log
        self._shutdown_requested = shutdown_requested
        self._clock = clock
        self._timer = timer
        self._on_done = on_done          # () -> None: one item left the queue
        self._own_window_front = own_window_front  # () -> bool: Sotto's alert would get the Cmd-V
        self._items: list[tuple] = []
        self._poll_armed = False
        self._waiting = False
        self._blocked_since: float | None = None
        self.last_delivery: dict | None = None  # the insert "scratch that" may undo

    def paste(self, text: str, in_history: bool = True) -> None:
        """in_history is False when History could not keep this dictation
        (F16a/F16b): then no note may send the user to History for it."""
        self._items.append(("paste", text, in_history))
        self._pump()

    def undo(self) -> None:
        self._items.append(("undo",))
        self._pump()

    def _pump(self) -> None:
        while self._items:
            if self._shutdown_requested():  # checked per item: Quit can land mid-pump
                self._discard_all()  # nothing is posted once shutdown is requested
                return
            if self._keys_held():
                self._wait_or_drop()
                return
            if self._own_window_front():
                self._wait_for_own_window()
                return
            self._blocked_since = None
            self._waiting = False
            item = self._items.pop(0)
            try:
                if item[0] == "paste":
                    self._deliver(item[1], item[2])
                else:
                    self._undo()
            except Exception as exc:  # one failed insert must not strand the items behind it
                self._log(f"! {item[0]} failed ({type(exc).__name__}: {str(exc)[:160]})")
                if item[0] == "paste":
                    self._note("Could not paste the dictation",
                               "It is in History. Open History to copy the text." if item[2]
                               else NOT_IN_HISTORY_EITHER)
            finally:
                self._on_done()

    def _discard_all(self) -> None:
        """Drop every waiting item; each one still reports done."""
        items, self._items = self._items, []
        for _item in items:
            self._on_done()

    def _wait_or_drop(self) -> None:
        now = self._clock()
        if self._recording():
            self._blocked_since = None  # the user is dictating: the release will deliver
        elif self._blocked_since is None:
            self._blocked_since = now
        elif now - self._blocked_since > DELIVERY_WAIT_MAX_S:
            dropped = sum(1 for item in self._items if item[0] == "paste")
            unsaved = sum(1 for item in self._items if item[0] == "paste" and not item[2])
            self._discard_all()
            self._blocked_since = None
            self._waiting = False
            self._log(f"! a key stayed held for {DELIVERY_WAIT_MAX_S:.0f}s — {dropped} dictation(s) "
                      f"not pasted, " + (f"{unsaved} not in History" if unsaved else "kept in History"))
            if unsaved:
                self._note("Dictation not pasted",
                           f"A key was held for too long to paste it. {NOT_IN_HISTORY_EITHER}")
            else:
                self._note("Dictation kept in History",
                           "A key was held for too long to paste it. Open History to copy the text.")
            return
        if not self._waiting:
            self._waiting = True
            self._log("  paste deferred — a key is still held")
        self._arm_poll()

    def _wait_for_own_window(self) -> None:
        """Hold the queue while Sotto's own window is in front (N24); no drop."""
        self._blocked_since = None
        if not self._waiting:
            self._waiting = True
            self._log("  paste deferred — Sotto's own window is in front")
        self._arm_poll()

    def _arm_poll(self) -> None:
        if not self._poll_armed:
            self._poll_armed = True
            timer = self._timer(DELIVERY_POLL_S, self._call_after, (self._resume,))
            timer.daemon = True
            timer.start()

    def _resume(self) -> None:
        self._poll_armed = False
        self._pump()

    def _deliver(self, text: str, in_history: bool = True) -> None:
        pid, keydowns = self._frontmost_pid(), self._keydowns()  # the paste target, before ⌘V
        if self._insert(text):  # a secure-input decline inserted nothing: arm no undo
            self.last_delivery = {"at": self._clock(), "pid": pid, "keydowns": keydowns}
        else:  # N34: every other drop path tells the user, so this one does too
            self._note("Dictation not pasted",
                       "Secure input is on (a password field may have focus), so Sotto did not "
                       "paste. " + ("Open History to copy the text." if in_history else NOT_IN_HISTORY_EITHER))

    def _undo(self) -> None:
        """'scratch that': the app's own ⌘Z, only for Sotto's own recent insert —
        same app still in front, nothing typed by the user since."""
        import voice_commands
        last = self.last_delivery
        if last is None or self._clock() - last["at"] > voice_commands.SCRATCH_WINDOW_S:
            self._log("  scratch that: nothing recent to undo")
            return
        pid = self._frontmost_pid()
        if pid is None or last["pid"] is None:  # no identity to compare: None == None proves nothing
            self._log("  scratch that: the app in front is unknown — nothing undone")
            return
        if pid != last["pid"]:
            self._log("  scratch that: the dictation went to another app — nothing undone")
            return
        if self._keydowns() != last["keydowns"]:
            self._log("  scratch that: you typed since the dictation — nothing undone")
            return
        if self._secure_input():
            return
        self._undo_keys()
        self.last_delivery = None
        self._log("↶ scratch that — undid the last dictation")


# -- daemon ------------------------------------------------------------------

def overlay_learning_state(entries: list[dict], active_history_ids: set[str]) -> list[dict]:
    """Return history rows with the UI-only learning-state overlay applied."""
    return [dict(entry, learning_state="active") if entry.get("id") in active_history_ids
            else dict(entry) for entry in entries]


def _active_learning_history_ids(coordinator) -> set[str]:
    """Read the coordinator-owned active corpus without exposing its text."""
    with coordinator._lock:  # coordinator is the serialization boundary
        return {record["history_id"] for record in coordinator.learning.active()
                if isinstance(record.get("history_id"), str)}


def learning_status_text(snapshots) -> str:
    """Small headless status surface; deliberately contains no transcript text."""
    return f"{len(snapshots)} active learning item(s)"


def learning_list_text(snapshots) -> str:
    """Format active learning identities only; references remain private."""
    if not snapshots:
        return "0 active learning item(s)"
    return "\n".join(
        f"sample_id={snapshot.sample_id} history_id={snapshot.history_id}"
        for snapshot in sorted(snapshots, key=lambda snapshot: snapshot.sample_id)
    )


def remove_learning_sample(sample_id: str, *, history_id_lookup, adaptive_revoke, artifact_revoke) -> bool:
    """Content-free ordering seam for the offline admin command."""
    history_id = history_id_lookup(sample_id)
    if history_id:
        adaptive_revoke(history_id)
    return bool(artifact_revoke(sample_id))


def nonadaptive_correct(dependency_guard, coordinator, entry_id: str, text: str, *, expected_revision: int | None = None):
    """Invalidate adaptive evidence before an ordinary correction mutates it."""
    return coordinator.review_transaction(
        entry_id, expected_revision,
        lambda: dependency_guard.revoke(entry_id, reason="reference_changed_nonadaptive"),
        lambda: (coordinator.correct(entry_id, text, expected_revision=expected_revision)
                 if expected_revision is not None else coordinator.correct(entry_id, text)))


def nonadaptive_delete(dependency_guard, coordinator, entry_id: str):
    dependency_guard.revoke(entry_id, reason="history_deleted")
    return coordinator.delete(entry_id)


def nonadaptive_clear(dependency_guard, coordinator) -> None:
    dependency_guard.revoke_all(reason="history_cleared")
    coordinator.clear()


def nonadaptive_revoke_learning(dependency_guard, coordinator, entry_id: str):
    dependency_guard.revoke(entry_id, reason="learning_revoked")
    return coordinator.revoke_history(entry_id)


HISTORY_UNREADABLE_TITLE = "History could not be read"


class UnsavedHistory:
    """Stands in for LearningCoordinator while history.jsonl cannot be read (F16b).

    Dictation keeps working: ``append_live`` hands back an in-memory row so the
    text is still delivered, but nothing (row, audio, learning state) is
    written. Every other coordinator operation raises the store's
    HistoryUnreadable, exactly as the real coordinator would on its re-read,
    so Retry, Correct, Delete and Clear can never act on a partial view."""

    def __init__(self, store, learning) -> None:
        self.history, self.learning = store, learning
        self._lock = threading.RLock()

    def append_live(self, text: str, samples, duration: float, model: str, **metadata) -> dict:
        """Side effects: none; the row and its audio exist only in memory."""
        return {"id": metadata.get("entry_id") or uuid.uuid4().hex[:12], "ts": metadata.get("ts"),
                "text": text, "duration": round(duration, 2), "model": model, "saved": False}

    def __getattr__(self, name: str):
        from history import HistoryUnreadable
        raise HistoryUnreadable(f"History is unreadable and was left unchanged ({self.history.unreadable})")


def history_coordinator(store, learning_store):
    """The dictation session's coordinator for ``store``.

    A store opened with ``tolerate_unreadable=True`` over an unreadable
    history.jsonl (F16b) gets an UnsavedHistory instead of a
    LearningCoordinator, whose startup re-read would raise before the menu
    bar exists; the caller tells the user once with
    ``history_unreadable_message``. Side effects: a LearningCoordinator
    validates the active learning links against History."""
    if store.unreadable is not None:
        return UnsavedHistory(store, learning_store)
    from learning import LearningCoordinator
    return LearningCoordinator(store, learning_store)


def history_unreadable_message(store) -> str:
    # Never suggest moving or deleting the file: a launch with no index would
    # treat every kept recording as an orphan and sweep it.
    cause, remedy = (("it was written by a newer version of Sotto", "update Sotto again")
                     if store.unreadable.newer_schema else
                     ("part of it is damaged", "repair the damaged line"))
    return (f"Sotto cannot read {store.index} because {cause}. The file and its recordings "
            "were left exactly as they are. Dictation still works, but new dictations are "
            f"not saved to History until the file can be read again: {remedy}, then restart Sotto.")


def _read_only_json(path: Path) -> dict | None:
    """Read a strict regular JSON record without constructing any store."""
    try:
        mode=os.lstat(path).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode): return None
        value=json.loads(path.read_text("utf-8"))
        return value if isinstance(value,dict) else None
    except (OSError,ValueError,json.JSONDecodeError):
        return None


def _read_only_last_jsonl(path: Path) -> dict | None:
    """Return the exact final deployment record; malformed tails fail closed."""
    try:
        mode=os.lstat(path).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode): return None
        lines=[line for line in path.read_text("utf-8").splitlines() if line.strip()]
        value=json.loads(lines[-1]) if lines else None
        return value if isinstance(value,dict) else None
    except (OSError,ValueError,json.JSONDecodeError):
        return None


def adaptive_status_text(snapshots, *, base_dir: Path | None = None) -> str:
    """Read content-free adaptive/silver/deployment status without stores."""
    from history import STORE_DIR
    root=base_dir or STORE_DIR
    state = _read_only_json(root / "adaptive-learning" / "state.json")
    if state is None:
        return learning_status_text(snapshots) + "\nadaptive English: collecting development 0/20"
    captures = state.get("captures", {}) if isinstance(state, dict) else {}
    dev = sum(1 for row in captures.values() if isinstance(row, dict) and row.get("outcome") in {"corrected", "correct_as_is"} and not row.get("pool_generation"))
    current = ((state.get("champions", {}) or {}).get("current") or (state.get("champions", {}) or {}).get("baseline") or {}).get("stable_id")
    active = [g for g in (state.get("generations", {}) or {}).values() if isinstance(g, dict) and g.get("status") in {"pool_open", "pool_closed", "evaluating"}]
    if active:
        g = active[-1]; stage = "evaluating" if g.get("status") in {"pool_closed", "evaluating"} else f"reviewing pool {len(g.get('pool', []))}/33"
    else:
        stage = f"collecting development {dev}/20"
    try:
        from silver_store import read_only_status
        silver=read_only_status(root)
        deploy=_read_only_last_jsonl(root / "adaptive-learning" / "silver" / "deployment.jsonl")
        if not isinstance(deploy,dict) or deploy.get("schema") != 1 or not isinstance(deploy.get("tier"),str):
            raise ValueError("deployment metadata malformed")
        horizon={}
        if silver.get("state") != "ok":
            raise RuntimeError("silver status blocked")
        jobs=silver.get("jobs",{}); worker=silver.get("worker") or {}; canary=deploy.get("canary") or {}
        silver_text=(f" · silver={silver['accepted']} accepted/{silver['abstained']} abstained"
                     f" queued={jobs.get('pending',0)} processing={jobs.get('leased',0)} discarded={jobs.get('discarded',0)}"
                     f" · worker={worker.get('state','unknown')}"
                     f" · horizon={horizon.get('status','none')} {horizon.get('members',0)}/500"
                     f" · deployment={deploy['tier']} {canary.get('stage','')}" )
    except Exception:
        silver_text=" · silver=unavailable (fail closed)"
    return learning_status_text(snapshots) + f"\nadaptive English: {stage}" + (f" · champion={current}" if current else "") + silver_text


def export_learning_snapshot(output: str | Path) -> int:
    """Copy consented corpus artifacts and write a benchmark-compatible manifest.

    The storage guard validates active snapshots and prevents cross-process
    revocation from unlinking a source while it is being copied.  We copy source
    WAV bytes verbatim: export is never a decode/re-encode operation.
    """
    from learning import LearningStore

    destination = Path(output).expanduser()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"Export destination already exists and is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        with LearningStore.admin_export_guard() as snapshots:
            # The pre-guard check is a fast failure.  This one is the race-free
            # decision: a concurrent exporter that observed an empty directory
            # must not add artifacts after the first guarded export succeeds.
            if not destination.is_dir() or any(destination.iterdir()):
                raise ValueError(
                    f"Export destination already exists and is not empty: {destination}")
            for snapshot in snapshots:
                audio_relative = Path("audio") / f"{snapshot.sample_id}.wav"
                audio_target = destination / audio_relative
                audio_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(snapshot.inference_audio_path, audio_target)
                row = {
                    "id": snapshot.sample_id,
                    "audio": audio_relative.as_posix(),
                    "reference": snapshot.corrected_text,
                    "tags": [],
                    "required_terms": [],
                    "language": None,
                    "provenance": {
                        "history_id": snapshot.history_id,
                        "history_revision": snapshot.history_revision,
                        "inference_sha256": snapshot.inference_identity.sha256,
                        "inference_format": snapshot.inference_identity.format,
                    },
                }
                if snapshot.raw_audio_path is not None:
                    raw_relative = Path("raw") / f"{snapshot.sample_id}.wav"
                    raw_target = destination / raw_relative
                    raw_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(snapshot.raw_audio_path, raw_target)
                    row["raw"] = raw_relative.as_posix()
                    row["provenance"]["raw_sha256"] = snapshot.raw_identity.sha256
                rows.append(row)
            manifest = destination / "manifest.jsonl"
            with manifest.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        # The destination was required to be empty, so a partial export must
        # remain visible for inspection rather than deleting user-selected data.
        raise
    return len(rows)


def _transcription_kwargs(speech_config, glossary_terms: tuple[str, ...], duration: float) -> tuple[dict, dict]:
    """Build MLX options and bounded, persistable attempt metadata."""
    if speech_config.profile.backend == "nemotron":
        from nemotron_backend import metadata
        return {}, metadata()
    if speech_config.profile.backend == "parakeet":
        return {}, {"profile": "parakeet", "language": None,
                    "prompt": {"disabled": "Parakeet does not use Whisper glossary prompts"}}
    from speech_config import glossary_prompt

    prompt_decision = glossary_prompt(glossary_terms, duration)
    kwargs = {
        "path_or_hf_repo": speech_config.model_repo,
        "condition_on_previous_text": False,
        # Sotto does not consume word timings.  Asking MLX Whisper to compute
        # them nearly doubled warm latency on the user's own captures.  Output
        # safety remains in the VAD metadata, no-speech verdict, implausible
        # rate/compression guard, and conservative repetition-loop salvage.
        "word_timestamps": False,
    }
    if speech_config.language is not None:
        kwargs["language"] = speech_config.language
    if prompt_decision.prompt is not None:
        kwargs["initial_prompt"] = prompt_decision.prompt
    metadata = {
        "profile": speech_config.profile.name,
        "language": speech_config.language,
        "prompt": prompt_decision.prompt if prompt_decision.prompt is not None else {
            "disabled": prompt_decision.reason,
        },
    }
    return kwargs, metadata


def transcribe_canonical_samples(mlx_whisper, samples: np.ndarray, speech_config,
                                 glossary_terms: tuple[str, ...], *, nemotron=None) -> tuple[str, dict]:
    """Transcribe exact canonical samples once, with the resolved local config."""
    duration = len(samples) / SAMPLE_RATE
    if speech_config.profile.backend == "nemotron":
        from nemotron_backend import metadata
        if nemotron is None:
            raise RuntimeError("Nemotron runtime is required")
        return nemotron.transcribe(samples), metadata()
    kwargs, metadata = _transcription_kwargs(speech_config, glossary_terms, duration)
    result = mlx_whisper.transcribe(samples, **kwargs)
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise RuntimeError("mlx_whisper.transcribe returned no text")
    chosen = getattr(mlx_whisper, "last_language", None)
    if speech_config.language is None and isinstance(chosen, str):
        metadata["detected_language"] = chosen  # Automatic's pick among the allowed set
    return result["text"].strip(), metadata


def transcribe_prepared(mlx_whisper, prepared, speech_config, glossary_terms: tuple[str, ...]) -> tuple[str, dict]:
    """Transcribe a :class:`PreparedAudio` without changing its ASR samples."""
    return transcribe_canonical_samples(mlx_whisper, prepared.asr_samples,
                                        speech_config, glossary_terms)


def stage_adaptive_live_audio(history, prepared, capture_id: str):
    """Bind live routing to the exact private canonical WAV that will persist.

    The normal push-to-talk path must not route a mutable in-memory array and
    then write an unrelated canonical file after inference.  The capture id
    is already the durable request identity, so its owned History path is a
    safe pre-publication location.  ``append_live(..., entry_id=capture_id)``
    later verifies and republishes these same prepared PCM bytes atomically
    with history metadata.  An exception caller removes this unadopted file.
    """
    from audio_codec import read_canonical_wav, write_canonical_wav
    if not isinstance(capture_id,str) or not capture_id:
        raise ValueError("invalid live capture identity")
    path=history.audio_path(capture_id)
    write_canonical_wav(path,prepared.pcm)
    _samples,identity=read_canonical_wav(path)
    if identity != prepared.identity:
        try: path.unlink(missing_ok=True)
        except OSError: pass
        raise RuntimeError("staged live canonical identity mismatch")
    return path


def discard_staged_adaptive_live_audio(history, capture_id: str) -> None:
    """Remove only an unadopted direct owned live staging file."""
    try:
        path=history.audio_path(capture_id)
        if history.get(capture_id) is None and path.exists() and not path.is_symlink():
            path.unlink();
    except (OSError,ValueError):
        pass


def adaptive_live_transcribe_prepared(runtime, prepared, capture_id: str):
    """Primary push-to-talk adaptive call with immutable canonical authority."""
    path=stage_adaptive_live_audio(runtime.history,prepared,capture_id)
    try:
        text,metadata=runtime.live_transcribe(prepared,canonical_path=path,
                                              canonical_identity=prepared.identity,capture_id=capture_id)
        return text,metadata,path
    except Exception:
        discard_staged_adaptive_live_audio(runtime.history,capture_id)
        raise


def warm_speech_runtime(*, adaptive: bool, model: str, speech_config, mlx_whisper=None, adaptive_runtime=None) -> None:
    """Injectable warmup seam with no adaptive routing side effects."""
    silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    if adaptive:
        if adaptive_runtime is None: raise RuntimeError("adaptive runtime is required")
        adaptive_runtime.warmup_baseline(silence)
        return
    if mlx_whisper is None: raise RuntimeError("mlx_whisper is required")
    kwargs = {"path_or_hf_repo": model}
    if speech_config.language is not None: kwargs["language"] = speech_config.language
    mlx_whisper.transcribe(silence, **kwargs)

def run(trigger: str, model: str | None = None, overlay: bool = True,
        hotkey_spec: str = "ctrl-opt-d", idle_release: float = IDLE_STOP_S,
        speech_config=None, glossary_terms: tuple[str, ...] = (), adaptive: bool = False,
        engine_switching: bool = True, settings_locked: frozenset | set = frozenset()) -> None:
    """Run capture using a resolved config, without importing MLX before setup."""
    from speech_config import language_mode, resolve_speech_config, save_language_mode, with_language
    if speech_config is None:
        speech_config = resolve_speech_config(model=model)
    model = speech_config.model_repo
    use_nemotron = speech_config.profile.backend == "nemotron"
    use_parakeet = speech_config.profile.backend == "parakeet"
    if adaptive and (use_nemotron or use_parakeet):
        raise ValueError("Nemotron and Parakeet are manual engines; they cannot enable adaptive routing")
    language_lock = threading.Lock()
    language_state = {"mode": language_mode(speech_config.language)}

    def current_speech_config():
        with language_lock:
            return with_language(speech_config, language_state["mode"])

    shutdown = ShutdownBoundary()
    restart = RestartController(shutdown)
    import ApplicationServices

    # Fail fast (before the slow model load) so a launchd restart loop is cheap,
    # and let macOS open the grant prompt for whatever binary is running us.
    if app_mode():
        import ui
        ui.use_app_icon(sotto_icon_path())
    if app_mode() and not ApplicationServices.AXIsProcessTrusted():
        # Sotto.app: explain, open the right pane, and continue once granted.
        if not explain_permission(
                "Sotto needs Accessibility",
                "Click Open System Settings and turn on Sotto under Privacy & Security → "
                "Accessibility, so it can hear your dictation key and put text at the cursor. "
                "This message closes by itself once it is on.\n\nAlready on? Sotto was rebuilt: "
                "select it, remove it with −, then add it again with +.",
                "Privacy_Accessibility",
                granted=accessibility_granted_now,
                register=lambda: ApplicationServices.AXIsProcessTrustedWithOptions(
                    {ApplicationServices.kAXTrustedCheckOptionPrompt: True})):
            log("accessibility not granted — open Sotto again after turning it on")
            sys.exit(0)  # a clean exit: launchd must not loop this dialog
        # This process still holds the cached "not trusted": start a fresh one.
        log("accessibility granted — restarting")
        relaunch_app_after_exit()
        sys.exit(0)
    elif not ApplicationServices.AXIsProcessTrustedWithOptions(
            {ApplicationServices.kAXTrustedCheckOptionPrompt: True}):
        log("accessibility not granted — grant it in System Settings "
            "(the entry may appear as 'Python'), then run the command again")
        sys.exit(1)

    if not microphone_permission(request=True):
        if app_mode():
            explain_permission(
                "Sotto needs the microphone",
                "Turn on Sotto under Privacy & Security → Microphone, then open Sotto again. "
                "It listens only while you hold the dictation key.",
                "Privacy_Microphone")
            sys.exit(0)
        log("microphone not granted — System Settings → Privacy & Security → "
            "Microphone → enable your terminal/Python, then run the command again")
        sys.exit(1)

    from PyObjCTools import AppHelper
    mlx_whisper = None
    nemotron = None
    if use_nemotron:
        from nemotron_backend import NemotronRuntime
        log("warming up Nemotron English streaming ...")
        try:
            nemotron = NemotronRuntime()
            log("✓ Nemotron model ready")
        except Exception as exc:
            # An optional backend must not strand the working dictation app
            # in a launchd crash loop after a runtime/model failure.
            from speech_config import load_language_mode, save_engine_mode
            log(f"! Nemotron unavailable ({str(exc)[:120]}) — restoring Whisper")
            try:
                save_engine_mode("whisper")
            except OSError:
                log("! could not persist Whisper fallback; using it for this launch")
            speech_config = resolve_speech_config("auto", language=load_language_mode())
            model = speech_config.model_repo
            language_state["mode"] = language_mode(speech_config.language)
            use_nemotron = False
    if use_parakeet:
        log("checking local Parakeet model (first use downloads about 2.5 GB) ...")
        try:
            mlx_whisper = LocalParakeet(model)
            log(f"warming up {model} ...")
            warm_speech_runtime(adaptive=False, model=model, speech_config=speech_config, mlx_whisper=mlx_whisper)
            log("✓ Parakeet ready")
        except Exception as exc:
            # A missing or broken optional engine must never strand dictation.
            from speech_config import load_language_mode, save_engine_mode
            log(f"! Parakeet unavailable ({str(exc)[:120]}) — restoring Whisper")
            try:
                save_engine_mode("whisper")
            except OSError:
                log("! could not persist Whisper fallback; using it for this launch")
            mlx_whisper = None
            speech_config = resolve_speech_config("auto", language=load_language_mode())
            model = speech_config.model_repo
            language_state["mode"] = language_mode(speech_config.language)
            use_parakeet = False
    if not adaptive and not use_nemotron and not use_parakeet:
        import mlx_whisper as mlx_whisper_module
        log("checking local Whisper model (first launch downloads model files) ...")
        try:
            mlx_whisper = LocalWhisper(mlx_whisper_module, model)
        except ModelUnavailable as exc:
            log(f"✗ Whisper model unavailable: {exc}")
            sys.exit(1)
        import settings as automatic_settings
        from speech_config import automatic_languages
        # Automatic chooses only among the languages picked in Settings.
        mlx_whisper.allowed_languages = lambda: automatic_languages(
            automatic_settings.load(automatic_settings.SETTINGS_PATH)["languages"])
        mlx_whisper.speed = lambda: automatic_settings.load(automatic_settings.SETTINGS_PATH)["speed"]
        log(f"warming up {model} ...")
        warm_speech_runtime(adaptive=False, model=model, speech_config=speech_config, mlx_whisper=mlx_whisper)
        log("✓ model ready")

    active_engine = "nemotron" if use_nemotron else "parakeet" if use_parakeet else "whisper"

    # An unreadable history.jsonl must not kill the app before its menu bar
    # exists (F16b). The adaptive lane's receipts depend on History, so there
    # it stays fatal.
    from history import HistoryStore
    from learning import LearningStore
    store = HistoryStore(tolerate_unreadable=not adaptive)
    learning_store = LearningStore()
    coordinator = history_coordinator(store, learning_store)
    if store.unreadable is not None:
        log(f"! History unreadable ({store.unreadable}): {history_unreadable_message(store)}")
    seed_totals(store)
    # Always available, no-model dependency guard: an earlier adaptive session
    # must remain revocation-safe even when this launch is non-adaptive.
    from adaptive_learning import AdaptiveLearning
    dependency_guard = AdaptiveLearning(store.base_dir)
    adaptive_runtime = None
    if adaptive:
        from adaptive_runtime import AdaptiveRuntime
        adaptive_runtime = AdaptiveRuntime(history=store, learning=learning_store,
                                           coordinator=coordinator)
        adaptive_runtime.register_retained_history()
        adaptive_runtime.reconcile()
        # Receipt-gated exact-snapshot warmup; a missing baseline fails before
        # the daemon announces readiness and never falls back to repo/main.
        warm_speech_runtime(adaptive=True, model=model, speech_config=speech_config, adaptive_runtime=adaptive_runtime)
        log("✓ adaptive receipt-backed champion ready")

    status_ui = None
    if overlay:
        import ui
        status_ui = ui.init_app()
        status_ui.on_overlay_event = log
        status_ui.adaptive_mode = adaptive
        import settings as settings_store
        status_ui.automatic_languages = settings_store.load(settings_store.SETTINGS_PATH)["languages"] or None
        status_ui.set_engine_mode(active_engine,
                                  enabled=not adaptive and engine_switching)
        status_ui.set_language_mode(language_state["mode"],
                                    enabled=not adaptive and not use_nemotron and not use_parakeet)

    pending_deliveries = PendingDeliveries()
    ui_call, deliver_call = main_thread_dispatch(status_ui is not None, AppHelper.callAfter,
                                                 pending_deliveries)
    if status_ui is not None and store.unreadable is not None:
        ui_call(status_ui.show_error, HISTORY_UNREADABLE_TITLE, history_unreadable_message(store))

    def refresh_history() -> None:
        if status_ui is not None:
            entries = store.entries(limit=HISTORY_KEEP)
            overlaid = overlay_learning_state(entries, _active_learning_history_ids(coordinator))
            if adaptive_runtime is not None:
                states = {row["history_id"]: row["status"] for row in adaptive_runtime.review_list()}
                overlaid = [dict(row, adaptive_eligible=adaptive_runtime.review_eligible_entry(row),
                                 **({"review_state": states[row["id"]]} if row.get("id") in states else {}))
                            for row in overlaid]
                adaptive_state = adaptive_runtime.status()
                active = adaptive_state.get("active_generation")
                if active and active.get("status") in {"pool_closed", "evaluating"}:
                    label = "adaptive: evaluating"
                elif active:
                    label = f"adaptive: pool {active.get('resolved', 0)}/33"
                elif adaptive_state["development"].get("ready"):
                    label = "adaptive: collecting challenger"
                else:
                    label = f"adaptive: dev {adaptive_state['development'].get('verified_nonempty', 0)}/20"
                ui_call(status_ui.set_adaptive_stage, label)
            ui_call(status_ui.refresh_history, overlaid)

    capture = CaptureService()
    capture.restart_callback = restart.request
    capture.idle_release_s = idle_release
    capture_gate = CaptureGate()
    if status_ui is not None:
        status_ui.level_source = lambda: capture.latest_rms
        status_ui.live_text_source = (
            (lambda: live_preview(capture.live_text())) if use_nemotron else None)

    # Single-flight transcription: one prewarmed worker, strict FIFO — live
    # dictations and retries can never run the model (or the clipboard cycle)
    # concurrently, and results arrive in capture order.
    jobs: queue.Queue = queue.Queue()
    model_activity = {"last_finished": time.monotonic(), "rewarming": False}

    def _copy_text(text: str) -> None:
        if shutdown.requested():
            return
        pasteboard = NSPasteboard.generalPasteboard()
        pasteboard.clearContents()
        pasteboard.setString_forType_(text, NSPasteboardTypeString)

    # Gesture callbacks mark the capture boundary. The audio engine starts and
    # stops off-thread; the ASR model stays resident independently.
    def on_start() -> None:
        with capture_gate.starting() as allowed:
            if shutdown.requested():
                return
            if not allowed:
                log("● an update is restarting Sotto; dictation resumes in a moment")
                return
            stream = None
            if use_nemotron:
                from streaming_audio import StreamingCapture
                stream = StreamingCapture(nemotron)
            cold = capture.begin(stream=stream, enqueue=lambda job: shutdown.enqueue(jobs, job))
        now = time.monotonic()
        if model_rewarm_due(model_activity["last_finished"], now,
                            rewarming=bool(model_activity["rewarming"]),
                            queue_idle=jobs.empty(), adaptive=adaptive or use_nemotron):
            model_activity["rewarming"] = True
            if not shutdown.enqueue(jobs, ("warmup",)):
                model_activity["rewarming"] = False
        if status_ui:
            ui_call(status_ui.show_recording)
        if not cold:
            log("● recording")
            return
        # Cold start after idle: audio does not exist until the engine is
        # live (measured up to ~5s on this machine). Say so, and flip the
        # indicator to "live" the moment the first block lands.
        log("● waking the mic — speak when the orb stops blinking")
        if status_ui:
            ui_call(status_ui.show_waking)

            def announce_live() -> None:
                deadline = time.monotonic() + 10
                while capture.is_waking() and time.monotonic() < deadline:
                    time.sleep(0.05)
                # A start that failed every try also ends the wait: say
                # "live" only when an engine actually exists.
                if engine.snapshot()[0] and capture.is_live():
                    ui_call(status_ui.show_recording)
                    log("● recording (mic live)")

            threading.Thread(target=announce_live, daemon=True).start()

    def on_finish() -> None:
        # In flight from before the capture ends until its job is queued, so
        # Quit's drain never sees an idle gap between the two.
        pending_deliveries.add()
        try:
            finish_capture()
        finally:
            pending_deliveries.finish()

    def finish_capture() -> None:
        released_at = time.monotonic()  # latency is measured from the key release
        if shutdown.requested():
            shutdown.stop_capture(capture)
            return
        captured_ts = time.time()
        raw, stream = capture.end(include_stream=True)
        capture.release_soon()  # audio is in hand — free the mic now
        seconds = len(raw) / max(capture.native_rate, 1.0)
        log(f"○ {seconds:.2f}s captured")
        if seconds < 0.25:
            if stream is not None:
                stream.cancelled.set()
                shutdown.enqueue(jobs, ("stream-close", stream))
            if status_ui:
                ui_call(status_ui.hide)
            return
        if status_ui:
            ui_call(status_ui.show_transcribing)
        shutdown.enqueue(jobs, ("live", raw, capture.native_rate, captured_ts,
                                released_at, uuid.uuid4().hex, current_speech_config(), stream))

    def on_discard() -> None:
        capture.abort()
        capture.release_soon()
        if status_ui:
            ui_call(status_ui.hide)
        log("○ tap ignored")

    def on_hands_free() -> None:
        if status_ui:
            ui_call(status_ui.show_hands_free, hands_free_hint(binding["trigger"]))

    def on_mic_failed() -> None:
        """The engine never started under this capture (device busy or gone).
        End the gesture so the key-up finds nothing to finish, and say so
        once — a dead mic used to show "recording" and then lose the dictation
        without a word. Finishing (not discarding) keeps whatever a
        mid-capture rebuild had already recorded."""
        if engine.force_finish():
            log("✗ could not start the microphone — dictation cancelled")
        if status_ui:
            ui_call(status_ui.show_error, "Could not start the microphone",
                    "Sotto could not open the input device. Check that another app "
                    "is not holding it and that Sotto may use the microphone "
                    "(System Settings → Privacy & Security → Microphone), then "
                    "press the key again.")

    engine = GestureEngine(on_start, on_finish, on_discard, on_hands_free=on_hands_free)
    capture.on_start_failed = on_mic_failed
    hotkey = parse_hotkey(hotkey_spec) if hotkey_spec else None
    hotkey_state: dict = {"pressed_at": None, "skip_up": False}
    if hotkey_spec and hotkey is None:
        log(f"! unrecognized --hotkey '{hotkey_spec}' — hotkey disabled")
    # One mutable binding so Settings can move the trigger at the next press.
    binding = {"trigger": trigger, "hotkey_spec": hotkey_spec if hotkey else ""}
    binding["mask"], binding["keycode"], binding["device_bit"] = TRIGGERS[trigger]
    trigger_state = {"down": False}

    def trigger_physically_down() -> bool:
        return trigger_down_in(
            Quartz.CGEventSourceFlagsState(
                Quartz.kCGEventSourceStateCombinedSessionState),
            binding["mask"], binding["device_bit"])

    def rebind(new_trigger: str | None = None, new_hotkey_spec: str | None = None) -> None:
        """Apply a Settings change to the live listener; drops any held capture."""
        nonlocal hotkey
        if new_trigger is not None:
            binding["mask"], binding["keycode"], binding["device_bit"] = TRIGGERS[new_trigger]
            binding["trigger"] = new_trigger
            trigger_state["down"] = False
            engine.force_reset()
            log(f"● trigger is now {new_trigger}")
        if new_hotkey_spec is not None:
            parsed = parse_hotkey(new_hotkey_spec) if new_hotkey_spec else None
            hotkey = parsed
            binding["hotkey_spec"] = new_hotkey_spec if parsed else ""
            hotkey_state.update(pressed_at=None, skip_up=False)
            log(f"● alternate hotkey {'is now ' + new_hotkey_spec if parsed else 'off'}")

    def keys_held() -> bool:
        """Is the trigger or any modifier physically down right now?"""
        flags = Quartz.CGEventSourceFlagsState(Quartz.kCGEventSourceStateCombinedSessionState)
        return bool(flags & ALL_MODIFIERS) or trigger_down_in(
            flags, binding["mask"], binding["device_bit"])

    def insert_text(text: str) -> bool:
        import settings
        preferences = settings.load(settings.SETTINGS_PATH)
        return inject(text, insert_mode=preferences["insert_mode"], spacing=preferences["spacing"])

    def delivery_note(title: str, message: str) -> None:
        if status_ui is not None:
            ui_call(status_ui.show_info, title, message)

    # Real key-downs seen by the tap (Sotto's tagged events excluded): typing
    # after a paste means "scratch that" must not undo it.
    user_keydowns = {"count": 0}

    # Finished text and "scratch that" share one FIFO (see DeliveryQueue):
    # capture order is kept, and nothing is posted while a key is held.
    delivery = DeliveryQueue(
        insert=insert_text, undo_keys=lambda: post_command_key(6), keys_held=keys_held,
        recording=lambda: engine.snapshot()[0], frontmost_pid=frontmost_pid,
        keydowns=lambda: user_keydowns["count"], secure_input=secure_input_active,
        call_after=AppHelper.callAfter, note=delivery_note, shutdown_requested=shutdown.requested,
        on_done=pending_deliveries.finish, own_window_front=own_window_without_text_field)

    # deliver_call counts each of these in pending_deliveries when it schedules
    # it; the queue's on_done finishes that entry once the item is delivered,
    # dropped or cleared, so Quit and the update restart wait for the paste.
    def inject_when_clear(text: str, _attempts: int = 0, in_history: bool = True) -> None:
        """Queue a finished dictation for the cursor (worker call shape kept)."""
        delivery.paste(text, in_history)

    def undo_when_clear(_attempts: int = 0) -> None:
        """Queue a 'scratch that' behind whatever is still waiting to paste."""
        delivery.undo()

    # Built here, after inject_when_clear / undo_when_clear exist: the worker
    # holds its collaborators instead of resolving them late like the closure
    # did. Imported inside run() so transcription.py can bind sotto's helpers
    # at import time without a cycle at module load.
    from transcription import TranscriptionWorker
    worker = TranscriptionWorker(
        shutdown=shutdown, jobs=jobs, log=log, model=model, mlx_whisper=mlx_whisper,
        nemotron=nemotron, current_speech_config=current_speech_config,
        coordinator=coordinator, refresh_history=refresh_history,
        adaptive_runtime=adaptive_runtime, adaptive=adaptive, use_nemotron=use_nemotron,
        status_ui=status_ui, ui_call=ui_call, deliver_call=deliver_call,
        inject_when_clear=inject_when_clear,
        undo_when_clear=undo_when_clear, copy_text=_copy_text, record_totals=record_totals,
        model_activity=model_activity, glossary_terms=glossary_terms)
    transcription_thread = threading.Thread(target=worker.run, daemon=True)
    transcription_thread.start()

    tap_watch = {"warned_at": 0.0}

    def revive_tap(reason: str) -> None:
        """Re-arm the listener and resync the trigger edge it may have missed.

        macOS disables a session tap on callback timeout (with a notification)
        but also silently — Accessibility churn, display/session changes — and
        a deaf tap looks exactly like a working one from the outside (2026-09-15:
        3 hours of dead hotkey, process healthy, tap enabled=False)."""
        Quartz.CGEventTapEnable(tap, True)
        current = trigger_physically_down()
        if trigger_state["down"] and not current:
            trigger_state["down"] = False
            engine.released()  # the release edge happened while deaf
        else:
            trigger_state["down"] = current
        if Quartz.CGEventTapIsEnabled(tap):
            log(f"! event tap re-enabled after {reason}")
            tap_watch["warned_at"] = 0.0
            if status_ui is not None:
                ui_call(status_ui.set_tap_health, True)
        elif time.monotonic() - tap_watch["warned_at"] > 60:
            tap_watch["warned_at"] = time.monotonic()
            log("✗ event tap stays disabled — re-grant Accessibility to Sotto")
            if status_ui is not None:
                ui_call(status_ui.set_tap_health, False)

    def callback(_proxy, event_type, event, _refcon):
        if shutdown.requested():
            return event
        if event_type in (Quartz.kCGEventKeyDown, Quartz.kCGEventKeyUp,
                          Quartz.kCGEventFlagsChanged) and is_own_event(event):
            return event  # Sotto's own ⌘V / ⌘Z / typing: never a chord, never the user typing
        if event_type == Quartz.kCGEventKeyDown:
            code = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
            hotkey_key = hotkey is not None and code == hotkey[1]
            if not (hotkey_key and (Quartz.CGEventGetFlags(event) & ALL_MODIFIERS) == hotkey[0]):
                user_keydowns["count"] += 1  # the user typed: "scratch that" is off the table
            if trigger_state["down"] and not hotkey_key:
                engine.chorded()  # any other key during a trigger hold makes it a shortcut
        # Hotkey: hold it and release to stop, or tap it and tap again later.
        # Normal keys carry real keycodes AND key-up events through remote
        # desktops, so both gestures survive an AnyDesk session.
        if hotkey is not None and event_type in (Quartz.kCGEventKeyDown,
                                                 Quartz.kCGEventKeyUp):
            want_flags, want_code = hotkey
            code = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventKeycode)
            if code != want_code:
                return event
            if event_type == Quartz.kCGEventKeyDown:
                if Quartz.CGEventGetIntegerValueField(
                        event, Quartz.kCGKeyboardEventAutorepeat):
                    return event  # holding the key, not a new press
                if (Quartz.CGEventGetFlags(event) & ALL_MODIFIERS) != want_flags:
                    return event
                if trigger_state["down"]:
                    engine.chorded()  # the trigger was part of the hotkey chord
                if engine.snapshot()[0]:
                    # a tap-started session is running — this press ends it
                    hotkey_state["pressed_at"] = time.monotonic()
                    hotkey_state["skip_up"] = True
                    if engine.force_finish():
                        log("● finished via hotkey")
                elif engine.force_start():
                    hotkey_state["skip_up"] = False
                    hotkey_state["pressed_at"] = time.monotonic()
                    log("● dictation started via hotkey")
            else:  # key up — flags may already be gone, match on keycode alone
                held = hotkey_release_held_seconds(
                    hotkey_state, time.monotonic())
                if (held is not None and held >= HOLD_THRESHOLD_S
                        and engine.snapshot()[0]):
                    if engine.force_finish():
                        log(f"● finished (held {held:.1f}s)")
                # a short tap leaves the session running until the next press
            return event
        if event_type in (Quartz.kCGEventTapDisabledByTimeout,
                          Quartz.kCGEventTapDisabledByUserInput):
            revive_tap("system disable")  # never die silently
            return event
        if event_type == Quartz.kCGEventFlagsChanged:
            if Quartz.CGEventGetIntegerValueField(
                    event, Quartz.kCGKeyboardEventKeycode) == binding["keycode"]:
                down = trigger_down_in(Quartz.CGEventGetFlags(event),
                                       binding["mask"], binding["device_bit"])
                if down != trigger_state["down"]:
                    trigger_state["down"] = down
                    (engine.pressed if down else engine.released)()
            elif trigger_state["down"]:
                engine.chorded()  # another modifier joined the hold
        return event

    def resync_poller() -> None:
        """Escape hatch: if a release edge is ever lost (tap deafness, phantom
        edges, sleep), a push-to-talk recording whose key is physically up for
        two consecutive polls gets its release synthesized. Hands-free is
        exempt — its key is legitimately up while recording."""
        misses = 0
        while not shutdown.requested():
            time.sleep(1.0)
            if shutdown.requested():
                break
            capture.tick()
            if not Quartz.CGEventTapIsEnabled(tap):
                # Directly on this thread — NEVER via callAfter. Bouncing to
                # the main loop was why the 2026-09-15 watchdog never fired
                # (2026-09-16: tap dead again, zero revive log lines) and
                # CGEventTapEnable is safe from a background thread.
                revive_tap("silent disable")
            recording, hands_free = engine.snapshot()
            if recording and not hands_free and not trigger_physically_down():
                misses += 1
                if misses >= 2:
                    log("! lost release recovered — finishing recording")
                    trigger_state["down"] = False
                    engine.released()
                    misses = 0
            else:
                misses = 0

    tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionListenOnly,
        Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged)
        | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
        | Quartz.CGEventMaskBit(Quartz.kCGEventKeyUp), callback, None,
    )
    if tap is None:
        log("failed to create event tap — grant Accessibility to this terminal "
            "(System Settings → Privacy & Security → Accessibility)")
        sys.exit(1)
    source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
    Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), source,
                              Quartz.kCFRunLoopCommonModes)
    Quartz.CGEventTapEnable(tap, True)
    threading.Thread(target=resync_poller, daemon=True).start()

    restart_drain = {"deadline": None}

    def stop_runtime_on_main() -> None:
        """Main-loop teardown after the event-only signal handler fires."""
        shutdown.stop_capture(capture)
        shutdown.discard_queued(jobs)
        drained = True
        if restart.foreground_pending or nemotron is not None:
            # Let the worker finish its current job; shutdown already fences
            # every paste/commit, so nothing it produces is kept.
            transcription_thread.join(APP_DRAIN_TIMEOUT)
            drained = not transcription_thread.is_alive()
        if not drained:
            if restart_drain["deadline"] is None:
                restart_drain["deadline"] = time.monotonic() + RESTART_DRAIN_DEADLINE_S
            if time.monotonic() < restart_drain["deadline"]:
                AppHelper.callLater(0.1, stop_runtime_on_main)
                return
            if not restart.foreground_pending:
                # A wedged native call must not hang Quit forever, and freeing
                # the native engine underneath it could crash: leave directly.
                log("! transcription did not finish in time — quitting anyway")
                os._exit(0)
            # exec replaces the whole image, so a wedged native call cannot
            # follow it; waiting forever would strand the user instead.
            log("! transcription did not finish in time — restarting anyway")
        if nemotron is not None and drained:
            # Cocoa terminate exits without unwinding run()'s finally block.
            # Free native streams/model before Metal's global destructors run.
            nemotron.close()
        try:
            Quartz.CGEventTapEnable(tap, False)
        except Exception:
            pass
        if restart.foreground_pending:
            if not capture.wait_released(RESTART_RELEASE_WAIT_S):
                log("! microphone teardown still running — restarting anyway")
            try:
                restart.exec_foreground()
            except OSError as exc:
                restart.foreground_pending = False
                restart.failed(f"foreground restart failed ({str(exc)[:120]}); run the command again")
        if status_ui is not None:
            ui_call(status_ui.hide)
            try:
                import AppKit
                AppKit.NSApp.terminate_(None)
            except Exception:
                pass
        else:
            try:
                Quartz.CFRunLoopStop(Quartz.CFRunLoopGetMain())
            except Exception:
                pass

    def shutdown_watcher() -> None:
        shutdown.event.wait()
        try:
            AppHelper.callAfter(stop_runtime_on_main)
        except Exception:
            # The handler remains event-only; a headless/embedded caller still
            # gets capture and queue fencing even if AppKit is unavailable.
            shutdown.stop_capture(capture)
            shutdown.discard_queued(jobs)

    threading.Thread(target=shutdown_watcher, daemon=True).start()

    if status_ui is not None:
        # menu actions + system lifecycle hooks
        def only_while_running(callback):
            def wrapped(*args, **kwargs):
                if shutdown.requested():
                    return None
                return callback(*args, **kwargs)
            return wrapped

        def action_copy(entry_id: str) -> None:
            def work() -> None:
                try:
                    entry = store.get(entry_id)
                    if entry:
                        ui_call(_copy_text, entry["text"])
                except Exception as exc:
                    ui_call(status_ui.show_error, "Could not copy transcript", str(exc)[:160])
            threading.Thread(target=work, daemon=True).start()

        def action_retry(entry_id: str) -> None:
            shutdown.enqueue(jobs, ("retry", entry_id, time.monotonic(), uuid.uuid4().hex,
                                    current_speech_config()))

        def action_set_language(mode: str) -> None:
            if adaptive or use_nemotron or use_parakeet:
                return
            try:
                selected = save_language_mode(mode)
                with language_lock:
                    language_state["mode"] = selected
                ui_call(status_ui.set_language_mode, selected, True)
                log(f"language: {selected} (applies to the next recording)")
            except Exception as exc:
                ui_call(status_ui.show_error, "Could not change language", str(exc)[:160])

        def action_set_engine(mode: str) -> None:
            if adaptive or not engine_switching or mode == active_engine:
                return
            if dictation_in_flight(capture, jobs, pending_deliveries):
                ui_call(status_ui.show_error, "Finish dictation first",
                        "Switch engines after the current recording and transcription finish.")
                return
            try:
                if mode == "nemotron":
                    from nemotron_backend import installation
                    installation()  # verify before persisting a restart choice
                if mode == "parakeet" and not parakeet_cached():
                    raise RuntimeError("Download Parakeet first: Settings → Download Parakeet.")
                if dictation_in_flight(capture, jobs, pending_deliveries):
                    raise RuntimeError("Finish the current dictation before switching engines.")
                persist_engine_and_restart(
                    mode, restart.request,
                    on_failure=lambda message: ui_call(
                        status_ui.show_error, "Could not change engine", message[:160]))
            except Exception as exc:
                ui_call(status_ui.show_error, "Could not change engine", str(exc)[:160])

        def action_save(entry_id: str) -> None:
            def work() -> None:
                try:
                    entry = store.get(entry_id)
                    if not entry:
                        return
                    target = save_audio_copy(store.audio_path(entry_id), Path.home() / "Desktop",
                                             entry["ts"])
                    log(f"✓ saved audio to Desktop as {target.name}")
                except FileNotFoundError:
                    log("! audio missing for that entry")
                    ui_call(status_ui.show_error, "Could not save audio",
                            "The audio file is no longer available.")
                except Exception as exc:
                    ui_call(status_ui.show_error, "Could not save audio", str(exc)[:160])
            threading.Thread(target=work, daemon=True).start()

        def action_delete(entry_id: str) -> None:
            def work() -> None:
                if shutdown.requested():
                    return
                try:
                    (adaptive_runtime.delete(entry_id) if adaptive_runtime is not None else
                     nonadaptive_delete(dependency_guard, coordinator, entry_id))
                    refresh_history()
                except Exception as exc:
                    ui_call(status_ui.show_error, "Could not delete recording", str(exc)[:160])
            threading.Thread(target=work, daemon=True).start()

        def action_clear() -> None:
            def work() -> None:
                if shutdown.requested():
                    return
                try:
                    (adaptive_runtime.clear() if adaptive_runtime is not None else nonadaptive_clear(dependency_guard, coordinator))
                    refresh_history()
                except Exception as exc:
                    ui_call(status_ui.show_error, "Could not clear history", str(exc)[:160])
            threading.Thread(target=work, daemon=True).start()

        def action_correct(entry_id: str, corrected_text: str, expected_revision: int | None = None) -> None:
            def work() -> None:
                if shutdown.requested():
                    return
                try:
                    entry = store.get(entry_id)
                    before = str((entry or {}).get("text") or "")
                    eligible = adaptive_runtime is not None and adaptive_runtime.review_eligible_entry(entry)
                    if eligible:
                        adaptive_runtime.review(entry_id, "corrected", reference=corrected_text, expected_revision=expected_revision)
                    elif nonadaptive_correct(dependency_guard, coordinator, entry_id, corrected_text, expected_revision=expected_revision) is None:
                        raise ValueError("The recording no longer exists")
                    refresh_history()
                    message = ("The corrected transcript and audio are retained locally for private learning."
                               if eligible else
                               "The corrected transcript was saved. Add it to the learning set separately if wanted.")
                    import dictionary
                    suggestions = dictionary.suggest(before, corrected_text,
                                                     dictionary.load(dictionary.DICTIONARY_PATH))
                    if suggestions:
                        ui_call(status_ui.offer_dictionary_rules,
                                [(rule.heard, rule.write) for rule in suggestions], message)
                    else:
                        ui_call(status_ui.show_info, "Correction saved", message)
                except Exception as exc:
                    ui_call(status_ui.show_error, "Could not save correction", str(exc)[:160])
            threading.Thread(target=work, daemon=True).start()

        def action_add_rules(rules: list) -> None:
            """Confirmed correction suggestions become dictionary rules."""
            try:
                import dictionary
                added = dictionary.add([dictionary.Rule(heard, write) for heard, write in rules],
                                       dictionary.DICTIONARY_PATH)
                log(f"dictionary: added {added} rule(s)")
                refresh_history()
            except Exception as exc:
                ui_call(status_ui.show_error, "Could not update the dictionary", str(exc)[:160])

        def action_open_dictionary() -> None:
            try:
                import dictionary
                import subprocess
                path = dictionary.ensure_file(dictionary.DICTIONARY_PATH)
                subprocess.Popen(["/usr/bin/open", "-t", str(path)])
            except Exception as exc:
                ui_call(status_ui.show_error, "Could not open the dictionary", str(exc)[:160])

        def action_enroll(entry_id: str) -> None:
            def work() -> None:
                if shutdown.requested():
                    return
                try:
                    snapshot = coordinator.enroll(entry_id)
                    if snapshot is None:
                        raise ValueError("A current correction is required before enrollment")
                    refresh_history()
                    ui_call(status_ui.show_info, "Added to learning set",
                            "The corrected audio and transcript are now retained locally.")
                except Exception as exc:
                    ui_call(status_ui.show_error, "Could not enroll sample", str(exc)[:160])
            threading.Thread(target=work, daemon=True).start()

        def action_revoke(entry_id: str) -> None:
            def work() -> None:
                if shutdown.requested():
                    return
                try:
                    count = (int(adaptive_runtime.revoke(entry_id)) if adaptive_runtime is not None else
                             nonadaptive_revoke_learning(dependency_guard, coordinator, entry_id))
                    refresh_history()
                    ui_call(status_ui.show_info, "Removed from learning set",
                            f"Removed {count} local learning sample(s).")
                except Exception as exc:
                    ui_call(status_ui.show_error, "Could not remove learning sample", str(exc)[:160])
            threading.Thread(target=work, daemon=True).start()

        def action_correct_as_is(entry_id: str, expected_revision: int | None = None) -> None:
            def work() -> None:
                if shutdown.requested():
                    return
                try:
                    if adaptive_runtime is None: raise ValueError("Adaptive English is not enabled")
                    adaptive_runtime.review(entry_id, "correct_as_is", expected_revision=expected_revision); refresh_history()
                except Exception as exc: ui_call(status_ui.show_error, "Could not review recording", str(exc)[:160])
            threading.Thread(target=work, daemon=True).start()

        def action_no_speech(entry_id: str, expected_revision: int | None = None) -> None:
            def work() -> None:
                if shutdown.requested():
                    return
                try:
                    if adaptive_runtime is None: raise ValueError("Adaptive English is not enabled")
                    adaptive_runtime.review(entry_id, "no_speech", expected_revision=expected_revision); refresh_history()
                except Exception as exc: ui_call(status_ui.show_error, "Could not review recording", str(exc)[:160])
            threading.Thread(target=work, daemon=True).start()

        def action_skip_learning_review(entry_id: str, reason: str, expected_revision: int | None = None) -> None:
            def work() -> None:
                if shutdown.requested():
                    return
                try:
                    if adaptive_runtime is None: raise ValueError("Adaptive English is not enabled")
                    adaptive_runtime.review(entry_id, "skip", reason=reason, expected_revision=expected_revision); refresh_history()
                except Exception as exc: ui_call(status_ui.show_error, "Could not review recording", str(exc)[:160])
            threading.Thread(target=work, daemon=True).start()

        def action_finish_now() -> None:
            """Menu escape hatch — always works, no gesture required."""
            if shutdown.requested():
                return
            ended = end_capture_now(engine, capture, on_finish)
            if ended == "gesture":
                log("● finished via menu")
            elif ended == "orphan":
                log("● orphan capture finished via menu")

        def action_start_now() -> None:
            if shutdown.requested():
                return
            if engine.force_start():
                log("● dictation started via menu (hands-free)")

        def action_restart(after_failure=None) -> None:
            """Restart this foreground process or its own sealed supervisor.
            after_failure runs if the restart does not happen."""
            def failed(message: str) -> None:
                if after_failure is not None:
                    after_failure()
                ui_call(status_ui.show_error, "Could not restart Sotto", message[:160])
            try:
                if not restart.request(on_failure=failed):
                    raise RuntimeError("Sotto is already shutting down")
                log("● restart requested from the menu")
            except Exception as exc:
                failed(str(exc))

        def action_quit() -> None:
            """Quit once the dictation in progress is done, waiting at most
            QUIT_DRAIN_S. Side effects: refuses new recordings, ends the
            recording in progress (even one the gesture engine lost), waits
            until it is pasted, then requests shutdown."""
            if shutdown.requested():
                return
            capture_gate.close()
            if end_capture_now(engine, capture, on_finish):
                log("● finishing the recording before quitting")

            def work() -> None:
                if not wait_until_idle(lambda: dictation_in_flight(capture, jobs, pending_deliveries),
                                       deadline_s=QUIT_DRAIN_S):
                    log(f"! dictation still running after {QUIT_DRAIN_S:.0f}s — quitting anyway")
                shutdown.request()
            threading.Thread(target=work, daemon=True).start()

        settings_view = {"controller": None, "installing": False, "parakeet_installing": False}

        class SettingsModel:
            """What the Settings window reads and changes (main thread only)."""

            def state(self) -> dict:
                import dictionary
                import login_item
                import settings as store
                from nemotron_backend import is_installed
                from speech_config import ENGINE_CHOICES, WHISPER_LANGUAGES, automatic_languages
                preferences = store.load(store.SETTINGS_PATH)
                return {
                    "settings": preferences, "trigger": binding["trigger"],
                    "hotkey": binding["hotkey_spec"], "locked": set(settings_locked),
                    "trigger_labels": {name: TRIGGER_LABELS[name] for name in store.TRIGGER_KEYS
                                       if name in TRIGGERS},
                    "engine_labels": dict(ENGINE_CHOICES),
                    "engine": active_engine,
                    "engine_switching": not adaptive and engine_switching,
                    "nemotron_installed": is_installed(),
                    "nemotron_installing": settings_view["installing"],
                    "parakeet_installed": parakeet_cached(),
                    "parakeet_installing": settings_view["parakeet_installing"],
                    "fast_available": not use_nemotron and not adaptive,
                    "language_names": WHISPER_LANGUAGES,
                    "automatic_languages": list(automatic_languages(preferences["languages"])),
                    "login_enabled": login_item.enabled(),
                    "login_available": app_mode() and bool(os.environ.get("SOTTO_APP_EXECUTABLE")),
                    "dictionary_rules": len(dictionary.load(dictionary.DICTIONARY_PATH)),
                    "data_dir": str(store.DATA_DIR),
                }

            def change(self, key: str, value) -> str | None:
                import settings as store
                if key in settings_locked:
                    return "That setting was chosen with a command-line flag for this run."
                try:
                    if key == "engine":
                        action_set_engine(value)
                        return None
                    if key == "launch_at_login":
                        import login_item
                        executable = os.environ.get("SOTTO_APP_EXECUTABLE", "")
                        if value and not (app_mode() and executable):
                            return "Install Sotto as an app first (scripts/install-mac.sh)."
                        login_item.enable(executable) if value else login_item.disable()
                    store.save({key: value}, store.SETTINGS_PATH)
                    if key == "trigger":
                        rebind(new_trigger=value)
                    elif key == "hotkey":
                        rebind(new_hotkey_spec=value)
                    elif key == "languages":
                        status_ui.automatic_languages = list(value)
                        refresh_history()
                    log(f"settings: {key} changed")
                    return None
                except Exception as exc:
                    return str(exc)

            def action(self, name: str) -> None:
                import subprocess
                if name == "open_dictionary":
                    action_open_dictionary()
                elif name == "show_data_folder":
                    from sotto_paths import DATA_DIR
                    subprocess.Popen(["/usr/bin/open", str(DATA_DIR)])
                elif name == "install_nemotron" and not settings_view["installing"]:
                    install_nemotron_in_background()
                elif name == "install_parakeet" and not settings_view["parakeet_installing"]:
                    install_parakeet_in_background()

        def install_nemotron_in_background() -> None:
            from offline_runtime import offline_requested
            if offline_requested():
                ui_call(status_ui.show_error, "Offline mode is on",
                        "Installing Nemotron downloads about 700 MB; start Sotto without the offline flags.")
                return
            settings_view["installing"] = True

            def work() -> None:
                script = Path(__file__).resolve().parent / "scripts" / "setup_nemotron.py"
                log("● installing Nemotron (about 700 MB) ...")
                try:
                    ok, detail = run_installer([sys.executable, str(script)])
                finally:
                    settings_view["installing"] = False
                refresh_history()
                if settings_view["controller"] is not None:
                    ui_call(settings_view["controller"].refresh)
                if ok:
                    log("✓ Nemotron installed")
                    ui_call(status_ui.show_info, "Nemotron installed",
                            "Choose Speech engine → Nemotron (English streaming) to use it.")
                else:
                    log(f"! Nemotron install failed ({detail[:80]})")
                    ui_call(status_ui.show_error, "Could not install Nemotron", detail)
            threading.Thread(target=work, daemon=True).start()

        def install_parakeet_in_background() -> None:
            from offline_runtime import offline_requested
            if offline_requested():
                ui_call(status_ui.show_error, "Offline mode is on",
                        "Downloading Parakeet (2.5 GB) needs the network; start Sotto without the offline flags.")
                return
            settings_view["parakeet_installing"] = True

            def work() -> None:
                log("● downloading Parakeet (about 2.5 GB) ...")
                try:
                    ok, detail = run_installer([sys.executable, str(Path(__file__).resolve()),
                                                "setup", "--profile", "parakeet"])
                finally:
                    settings_view["parakeet_installing"] = False
                refresh_history()
                if settings_view["controller"] is not None:
                    ui_call(settings_view["controller"].refresh)
                if ok:
                    log("✓ Parakeet downloaded")
                    ui_call(status_ui.show_info, "Parakeet downloaded",
                            "Choose Speech engine → Parakeet to use it.")
                else:
                    log(f"! Parakeet download failed ({detail[:80]})")
                    ui_call(status_ui.show_error, "Could not download Parakeet", detail)
            threading.Thread(target=work, daemon=True).start()

        def action_open_settings() -> None:
            if settings_view["controller"] is None:
                from settings_window import SettingsController
                settings_view["controller"] = SettingsController.alloc().initWithModel_(SettingsModel())
            settings_view["controller"].show()

        update_state = {"running": False}

        def action_apply_update() -> None:
            """Update this checkout with the installer, then restart into it."""
            if update_state["running"]:
                return
            if dictation_in_flight(capture, jobs, pending_deliveries):
                ui_call(status_ui.show_error, "Finish dictation first",
                        "Update after the current recording and transcription finish.")
                return
            update_state["running"] = True
            log("● updating Sotto")

            def work() -> None:
                import updates
                from sotto_paths import DATA_DIR
                finished, tail, source_changed = updates.apply_mac(
                    Path(__file__).resolve().parent, sys.executable, DATA_DIR / "logs" / "update.log")
                update_state["running"] = False
                if finished:
                    log("✓ update installed — restarting once dictation is done")
                    # A restart discards any recording or transcription in progress,
                    # and the user may have dictated while the update ran. Refuse new
                    # recordings first, then drain, so none can start in between.
                    capture_gate.close()
                    if not wait_until_idle(lambda: dictation_in_flight(capture, jobs, pending_deliveries),
                                           deadline_s=UPDATE_DRAIN_DEADLINE_S):
                        log(f"! dictation still running after {UPDATE_DRAIN_DEADLINE_S:.0f}s — "
                            "restarting into the update anyway")
                    action_restart(after_failure=capture_gate.reopen)
                elif source_changed:
                    log("! update installed but its setup did not finish (see logs/update.log)")
                    ui_call(status_ui.show_error, "Update not finished",
                            f"{tail}\n\nThe new version is in place but its setup did not finish. "
                            "Run scripts/install-mac.sh --update in Terminal to finish it, then "
                            "choose Restart Sotto. Details are in logs/update.log in the data folder.")
                else:
                    log("! update failed (see logs/update.log)")
                    ui_call(status_ui.show_error, "Update failed",
                            f"{tail}\n\nThe source was not switched, so Sotto keeps running the "
                            "current version. Details are in logs/update.log in the data folder.")
            threading.Thread(target=work, daemon=True).start()

        def action_check_updates() -> None:
            """Contact GitHub only now, because the user asked."""
            def work() -> None:
                import updates
                from offline_runtime import offline_requested
                result = updates.check(Path(__file__).resolve().parent, offline=offline_requested())
                log(f"update check: {result.state}" + (f", {result.behind} behind" if result.behind else ""))

                def present() -> None:
                    title, text = updates.describe(
                        result, "run scripts/install-mac.sh --update in Terminal from the Sotto folder.")
                    if result.state == "available":
                        if status_ui.ask(title, text, "Update Now", "Later"):
                            action_apply_update()
                    elif result.state == "not-git":
                        if status_ui.ask("Download the latest release", result.detail,
                                         "Open Releases Page", "Close"):
                            import AppKit
                            AppKit.NSWorkspace.sharedWorkspace().openURL_(
                                AppKit.NSURL.URLWithString_(updates.RELEASES_URL))
                    elif result.state == "failed":
                        status_ui.show_error(title, text)
                    else:
                        status_ui.show_info(title, text)
                ui_call(present)
            threading.Thread(target=work, daemon=True).start()

        status_ui.set_history_callbacks({
            "quit": action_quit,
            "restart": only_while_running(action_restart),
            "copy": only_while_running(action_copy), "retry": only_while_running(action_retry),
            "save": only_while_running(action_save), "delete": only_while_running(action_delete),
            "clear": only_while_running(action_clear), "correct": only_while_running(action_correct),
            "enroll": only_while_running(action_enroll), "revoke": only_while_running(action_revoke),
            "correct_as_is": only_while_running(action_correct_as_is),
            "no_speech": only_while_running(action_no_speech),
            "skip_learning_review": only_while_running(action_skip_learning_review),
            "finish_now": only_while_running(action_finish_now),
            "start_now": only_while_running(action_start_now),
            "set_language": only_while_running(action_set_language),
            "set_engine": only_while_running(action_set_engine),
            "add_rules": only_while_running(action_add_rules),
            "open_dictionary": only_while_running(action_open_dictionary),
            "open_settings": only_while_running(action_open_settings),
            "check_updates": only_while_running(action_check_updates),
        })
        status_ui.on_wake = lambda: (not shutdown.requested() and
                                     (engine.force_reset(), Quartz.CGEventTapEnable(tap, True)))
        refresh_history()

    log(f"listening on {trigger} · hold to talk, double-tap for hands-free"
        + (f" · {hotkey_spec}: hold or tap (remote-friendly)" if hotkey else "")
        + f" · model: {model.split('/')[-1]} · language: {language_state['mode']} · ^C to quit")
    try:
        with shutdown_signal_handlers(shutdown):
            # Cocoa/CFRunLoop can otherwise sit in native code indefinitely;
            # CPython only delivers SIGINT/SIGTERM on a main-thread bytecode
            # boundary. A lightweight timer gives those handlers a boundary.
            def signal_tick():
                if not shutdown.requested():
                    AppHelper.callLater(0.25, signal_tick)

            AppHelper.callLater(0.25, signal_tick)
            if status_ui is not None:
                import ui
                ui.run_loop()
            else:
                Quartz.CFRunLoopRun()
    except KeyboardInterrupt:
        shutdown.request()
        log("\nshutting down")
    finally:
        shutdown.stop_capture(capture)
        shutdown.discard_queued(jobs)
        transcription_thread.join(APP_DRAIN_TIMEOUT)
        if nemotron is not None and not transcription_thread.is_alive():
            nemotron.close()


def doctor_input_probe() -> int:
    """The doctor's input check, run in a subprocess (see doctor): pin the
    same mic a key press pins, THEN read the input bus format the tap will
    use. The output bus of an unpinned engine describes the system default
    device — a USB interface at 96 kHz / 2 ch, say — while dictation records
    the pinned built-in or headset mic at its own rate."""
    from AVFoundation import AVAudioEngine
    engine = AVAudioEngine.alloc().init()
    node = engine.inputNode()
    _pin_input_to_builtin(node)   # logs "mic: <which one>" on stderr
    fmt = node.inputFormatForBus_(0)
    print(f"{fmt.sampleRate():.0f} Hz, {fmt.channelCount()} ch")
    return 0 if fmt.sampleRate() > 0 and fmt.channelCount() > 0 else 1


def doctor() -> None:
    import subprocess
    import ApplicationServices

    trusted = ApplicationServices.AXIsProcessTrusted()
    log(("✓" if trusted else "✗") + " accessibility (this terminal)")
    microphone = microphone_permission(request=True)
    log(("✓" if microphone else "✗") + " microphone permission (terminal/Python)")
    if not microphone:
        log("→ System Settings → Privacy & Security → Microphone → enable your terminal/Python")
    usable_input = False
    # Probe the input device in a subprocess: CoreAudio hard-crashes (a native
    # SIGSEGV, not a catchable exception) in sessions with no usable audio
    # context, and a diagnostic must survive the conditions it diagnoses.
    # The probe is doctor_input_probe: the capture path's own pin + input bus.
    probe = "import sotto; raise SystemExit(sotto.doctor_input_probe())"
    try:
        result = subprocess.run([sys.executable, "-c", probe],
                                capture_output=True, text=True, timeout=30,
                                cwd=str(Path(__file__).resolve().parent))
    except subprocess.TimeoutExpired:
        log("✗ input device: probe timed out (waiting on a permission prompt?)")
    else:
        if result.returncode == 0:
            usable_input = True
            mics = [line.strip()[len("mic:"):].strip()
                    for line in result.stderr.splitlines()
                    if line.strip().startswith("mic:")]
            log(f"✓ input device: {result.stdout.strip()} "
                f"({mics[0] if mics else 'AVAudioEngine'})")
        else:
            log("✗ input device: no usable audio input in this session "
                "(microphone permission missing, or headless/SSH context)")
    if not trusted:
        log("→ System Settings → Privacy & Security → Accessibility → add your terminal")
    if not (trusted and microphone and usable_input):
        sys.exit(1)


def main() -> None:
    from speech_config import SUPPORTED_LANGUAGES
    parser = argparse.ArgumentParser(description="Local push-to-talk dictation.")
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run", "doctor", "setup", "dictionary", "learning-status", "learning-list",
                                 "learning-remove", "export-learning", "learning-preflight",
                                 "learning-review-list", "learning-evaluate", "learning-rollback"])
    parser.add_argument("--trigger", default=None, choices=sorted(TRIGGERS),
                        help="dictation key for this run (default: Settings, initially right-option)")
    parser.add_argument("--profile", default=None,
                        choices=("auto", "nemotron-en", "parakeet"),
                        help="local speech profile (default: saved engine, initially auto)")
    parser.add_argument("--model", default=None,
                        help="explicit local MLX model repository override")
    parser.add_argument("--language", choices=SUPPORTED_LANGUAGES, default=None, metavar="CODE",
                        help="explicit language override; default uses the profile")
    parser.add_argument("--glossary", help="local UTF-8 text or JSON glossary")
    parser.add_argument("--adaptive", action="store_true",
                        help="enable human-grounded adaptive English runtime (forces --language en)")
    parser.add_argument("--output", help="empty directory for export-learning")
    parser.add_argument("--sample-id", help="active learning sample ID for learning-remove")
    parser.add_argument("--no-overlay", action="store_true",
                        help="headless: no pill, no menu-bar icon")
    parser.add_argument("--hotkey", default=None,
                        help="hotkey that works over remote desktop — hold it "
                             "or tap it, e.g. ctrl-opt-d or f13 ('' to disable)")
    parser.add_argument("--idle-release", type=float, default=IDLE_STOP_S,
                        help="seconds to retain the mic after key-up "
                             "(0 = immediately; negative = never)")
    args = parser.parse_args()
    if app_mode() and args.command == "run":
        from sotto_paths import DATA_DIR
        route_app_logs(DATA_DIR)

    if args.command == "doctor":
        doctor()
        return
    if args.command == "setup":
        # Download the default model now (with progress) so the first
        # dictation, or the first Sotto.app launch, is ready immediately.
        try:
            if args.profile == "parakeet":
                from speech_config import MODEL_PROFILES
                engine, path = "Parakeet", LocalParakeet(MODEL_PROFILES["parakeet"].repo).path
            else:
                engine, path = "Whisper", LocalWhisper(None, DEFAULT_MODEL).path
        except ModelUnavailable as exc:
            print(f"✗ {exc}")
            sys.exit(1)
        print(f"✓ {engine} model ready ({Path(path).name[:12]}) in {SOTTO_HF_HOME}")
        from offline_runtime import offline_requested
        import vad
        try:
            ready = vad.install(allow_download=not offline_requested())
            print(f"✓ voice detection ready (Silero VAD v6.2) in {ready.parent}")
        except Exception as exc:  # advisory: transcription works without it
            print(f"! voice detection not installed ({str(exc)[:120]}); run setup again later")
        return
    if args.command == "dictionary":
        import dictionary
        rules = dictionary.load(dictionary.DICTIONARY_PATH)
        print(f"{dictionary.DICTIONARY_PATH} · {len(rules)} rule(s)")
        for rule in rules:
            print(f"  {rule.heard} {dictionary.SEPARATOR} {rule.write}")
        return

    # Corpus administration must remain useful on an offline Mac with no MLX
    # runtime installed and must never trigger an inference model download.
    if args.command in {"learning-status", "learning-list", "learning-remove", "export-learning"}:
        from learning import LearningStore
        if args.command == "learning-status":
            print(adaptive_status_text(LearningStore.admin_snapshot()))
            return
        if args.command == "learning-list":
            print(learning_list_text(LearningStore.admin_snapshot()))
            return
        if args.command == "learning-remove":
            if not args.sample_id:
                parser.error("learning-remove requires --sample-id SAMPLE_ID")
            from adaptive_learning import AdaptiveLearning
            from history import STORE_DIR
            if not remove_learning_sample(
                    args.sample_id, history_id_lookup=LearningStore.admin_history_id,
                    adaptive_revoke=lambda ident: AdaptiveLearning(STORE_DIR).revoke(ident, reason="learning_sample_removed"),
                    artifact_revoke=LearningStore.admin_revoke):
                parser.error("Learning sample was not found or was already removed")
            print(f"Removed learning sample {args.sample_id}")
            return
        if not args.output:
            parser.error("export-learning requires --output DIRECTORY")
        try:
            count = export_learning_snapshot(args.output)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(f"Exported {count} learning item(s) to {Path(args.output).expanduser()}")
        return

    if args.command in {"learning-preflight", "learning-review-list", "learning-evaluate", "learning-rollback"}:
        from adaptive_runtime import AdaptiveRuntime
        runtime = AdaptiveRuntime()
        if args.command == "learning-preflight":
            for result in runtime.preflight():
                print(f"backend={result['backend']} stable_id={result['stable_id']} success=true")
            return
        if args.command == "learning-review-list":
            for row in runtime.review_list():
                fields = (f"history_id={row['history_id']} ts={row['captured_ts']} status={row['status']}"
                          + (f" reason={row.get('skip_reason')}" if row.get("skip_reason") else ""))
                print(fields)
            return
        if args.command == "learning-evaluate":
            decision = runtime.evaluate()
            print("decision=" + str(decision.get("outcome")) + (f" winner={decision.get('winner')}" if decision.get("winner") else ""))
            return
        result = runtime.rollback()
        print(f"rollback={result['outcome']}")
        return

    try:
        from speech_config import load_engine_mode, load_glossary, load_language_mode, resolve_speech_config
        saved_engine = load_engine_mode() if not args.adaptive and args.model is None else "whisper"
        profile = args.profile or {"nemotron": "nemotron-en", "parakeet": "parakeet"}.get(saved_engine, "auto")
        if args.adaptive and profile in ("nemotron-en", "parakeet"):
            parser.error("Nemotron and Parakeet cannot be combined with --adaptive")
        if args.adaptive and args.language not in {None, "en"}:
            parser.error("--adaptive requires English; use --language en")
        language = args.language
        if not args.adaptive and language is None and profile == "auto":
            language = load_language_mode()
        speech_config = resolve_speech_config(profile, model=args.model,
                                              language="en" if args.adaptive else language)
        source = glossary_source(args.glossary)
        glossary_terms = load_glossary(source) if source else ()
    except ValueError as exc:
        parser.error(str(exc))
    import settings
    preferences = settings.load(settings.SETTINGS_PATH)
    trigger = args.trigger or preferences["trigger"]
    if trigger not in TRIGGERS:
        trigger = settings.DEFAULTS["trigger"]
    run(trigger, overlay=not args.no_overlay,
        hotkey_spec=args.hotkey if args.hotkey is not None else preferences["hotkey"],
        settings_locked={name for name, flag in (("trigger", args.trigger), ("hotkey", args.hotkey))
                         if flag is not None},
        idle_release=args.idle_release,
        speech_config=speech_config, glossary_terms=glossary_terms, adaptive=args.adaptive,
        engine_switching=args.profile is None and args.model is None)


if __name__ == "__main__":
    main()

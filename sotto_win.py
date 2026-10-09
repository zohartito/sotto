#!/usr/bin/env python3
"""sotto on Windows — hold right-ctrl, speak, release; the words land at your cursor.

Same pipeline as sotto.py — the pure logic (GestureEngine, VAD, hallucination
guards, history/learning stores, dictionary, settings) is imported and shared —
with Windows I/O:
  * trigger   pynput low-level keyboard hook (win_hotkey)
  * capture   one fresh stream per capture on the default input, MME (win_capture)
  * ASR       faster-whisper / CTranslate2: CUDA float16, else CPU int8 (win_asr)
  * insert    paste (clipboard restored) or type (SendInput Unicode) (win_inject)
  * tray      system-tray menu, Settings and History dialogs (win_ui, --tray)

    python sotto_win.py                          # console run (tests, headless)
    python sotto_win.py --tray                   # tray app (win_launch.py for pythonw)
    python sotto_win.py --trigger left-ctrl
    python sotto_win.py --device cpu             # or SOTTO_DEVICE=cpu
    python sotto_win.py setup                    # download the pinned model + VAD now
    python sotto_win.py doctor                   # device + model + mic checks
    python sotto_win.py history                  # history-delete / history-clear

No overlay, no adaptive/silver lane, no Nemotron.  Nothing leaves the machine;
each pinned model downloads once, on its first run.  Logs hold timings and
counts, never dictated text.
"""

from __future__ import annotations

import argparse
import atexit
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid

if sys.platform != "win32":
    raise ImportError("sotto_win is Windows-only")

import dictionary
import pipeline
import progress
import settings as user_settings
import sotto
from offline_runtime import offline_requested
from sotto_paths import DATA_DIR, MODEL_CACHE_DIR
from speech_config import (SpeechConfig, automatic_languages, get_profile, load_glossary,
                           load_language_mode, normalize_language, save_language_mode,
                           with_language)
import win_asr
import win_capture
import win_hotkey
import win_inject

from history import HistoryStore
from learning import LearningCoordinator, LearningStore

INSERT_WAIT_S = 120.0  # insert once modifiers are released; after this, History only
# Restart waits this long for the dictation in flight: a long CPU dictation
# finishes well inside it, but a wedged CUDA/CT2 call must not block recovery.
RESTART_DRAIN_CAP_S = 300.0
# Quit finishes the dictation in flight, up to this long: the Mac's
# sotto.QUIT_DRAIN_S (branch fix/app-lifecycle), the owner's choice for both.
QUIT_DRAIN_S = 10.0
DRAIN_POLL_S = 0.1  # how often a drain checks whether the work in flight is done
UNREAD_PASTE = "a paste the app has not read yet"  # in_flight()'s words for it
TRANSCRIPTION_FAILED_TEXT = "[transcription failed]"  # History text when the model raised
LOG_ROTATE_BYTES = 5 * 1024 * 1024


class _ConsoleLog:
    """stderr to the console (when there is one) AND ``DATA_DIR/sotto.log``.

    sotto.log() prints to sys.stderr; teeing it here gives a durable log.
    Under pythonw there is no console (sys.stderr is None): file only.  It
    holds timings and character counts, never transcript text.

    Callers only queue the text; one writer thread does the console and file
    I/O.  Gesture callbacks log on the keyboard-hook thread, and Windows
    silently drops a low-level hook that is slow to return, so a log line must
    never wait for the disk there.  drain() (also at exit) writes what is left.
    """

    def __init__(self) -> None:
        self._console = sys.stderr
        self._file = None
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            path = DATA_DIR / "sotto.log"
            if path.is_file() and path.stat().st_size > LOG_ROTATE_BYTES:
                path.replace(path.with_name("sotto.log.1"))
            self._file = open(path, "a", encoding="utf-8", buffering=1)
        except OSError:
            pass  # console-only rather than crash on a read-only profile
        self._pending: queue.SimpleQueue = queue.SimpleQueue()
        self._draining = False
        self._writer = threading.Thread(target=self._write_pending, daemon=True,
                                        name="sotto-log")
        self._writer.start()
        atexit.register(self.drain)

    def write(self, text: str) -> int:
        if self._draining:
            self._emit(text)  # after drain(): nothing reads the queue any more
        else:
            self._pending.put(text)
        return len(text)

    def flush(self) -> None:
        pass  # the writer flushes every line it writes

    def drain(self, timeout: float = 2.0) -> None:
        """Write every queued line, then write directly from now on.

        Side effects: stops the writer thread.
        """
        if self._draining:
            return
        self._pending.put(None)
        self._writer.join(timeout)
        self._draining = True
        while True:  # a line queued behind the stop marker is written, never dropped
            try:
                text = self._pending.get_nowait()
            except queue.Empty:
                return
            if text is not None:
                self._emit(text)

    def _write_pending(self) -> None:
        while (text := self._pending.get()) is not None:
            self._emit(text)

    def _emit(self, text: str) -> None:
        if self._console is not None:
            try:
                self._console.write(text)
                self._console.flush()
            except (OSError, ValueError):
                self._console = None  # the console went away
        if self._file is not None:
            try:
                self._file.write(text)
                self._file.flush()
            except (OSError, ValueError):
                self._file = None


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def resolve_config(profile: str, model: str | None, language: str | None,
                   speed: str = "accurate") -> SpeechConfig:
    """Profile (+ speed) -> CTranslate2 repo + language, without loading any model.

    speech_config.MODEL_PROFILES name the MLX repos for the Mac; the CT2 ids
    live in win_asr (same profile names).  Speed "fast" swaps only the auto
    profile's model; an explicit --model wins.
    """
    profile_obj = get_profile(profile)
    if model is not None:
        model_repo = model.strip()
    elif profile == "auto" and speed == "fast":
        model_repo = win_asr.FAST_REPO
    else:
        model_repo = win_asr.repo_for_profile(profile)
    if not model_repo:
        raise ValueError("Model repository must not be empty")
    if language is None and profile == "auto":
        language = load_language_mode()
    # No explicit override: the profile's own language default wins
    # (same rule as speech_config.resolve_speech_config).
    resolved = profile_obj.language if language is None else normalize_language(language)
    return SpeechConfig(profile_obj, model_repo, resolved)


# -- per-dictation delivery ----------------------------------------------------

@dataclass(frozen=True)
class DeliveryPrefs:
    """The settings one dictation uses, read when the worker starts it."""
    insert_mode: str = "paste"
    spacing: str = "smart"
    languages: tuple[str, ...] = field(default_factory=automatic_languages)


SCRATCH = object()  # a delivery that undoes the last dictation instead of inserting
VK_Z = 0x5A


def delivery_prefs(saved: dict) -> DeliveryPrefs:
    return DeliveryPrefs(saved.get("insert_mode", "paste"), saved.get("spacing", "smart"),
                         automatic_languages(saved.get("languages")))


def screen_transcript(text: str, seconds: float, speech_fraction: float,
                      preprocessing: dict) -> tuple[str, str | None]:
    """Output guards before anything is inserted: (text, reason it is kept back or None).

    Empty output becomes the no-speech marker, a clean prefix before a
    repetition loop is salvaged, anything else implausible is suspect.
    """
    no_speech = sotto.reads_as_no_speech(text, speech_fraction)
    if no_speech == "empty transcript":
        text = "[no speech detected]"
    hallucinated = sotto.looks_hallucinated(text, seconds)
    if hallucinated and not no_speech:
        # A loop that starts partway through must not cost the user the real
        # dictation in front of it.
        salvaged = sotto.salvage_repetition_loop(text, seconds)
        if salvaged is not None:
            text, trim_receipt = salvaged
            preprocessing["repetition_trimmed"] = trim_receipt
            log(f"  cut a {trim_receipt['repeats']}x repetition loop "
                f"({trim_receipt['dropped_chars']} chars) — keeping the clean prefix")
            hallucinated = None
    return text, hallucinated or no_speech


def move_detected_language(attempt_metadata: dict, preprocessing: dict) -> None:
    """Automatic's pick lives in preprocessing, exactly like the Mac's rows."""
    detected = attempt_metadata.pop("detected_language", None)
    if detected:
        preprocessing["detected_language"] = detected


# -- process -------------------------------------------------------------------

def instance_name() -> str:
    """One dictation app per Windows session: two hooks on one key would
    insert every dictation twice, whatever their data folders.  Tests give
    each run its own name through SOTTO_INSTANCE_KEY."""
    key = os.environ.get("SOTTO_INSTANCE_KEY", "").strip() or "dictation"
    return "Local\\sotto-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def single_instance() -> int | None:
    """The session-wide named mutex, or None when another Sotto holds it."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
    handle = kernel32.CreateMutexW(None, False, instance_name())
    if handle and ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        release_instance(handle)
        return None
    return handle


def release_instance(handle: int | None) -> None:
    if handle:
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle(handle)


def restart_command() -> list[str]:
    """The same launch again, without a console window.  A relative script
    path is made absolute: the restart runs in the repository folder, not in
    the folder Sotto was started from."""
    import win_startup
    script = sys.argv[0]
    if script and not Path(script).is_absolute():
        script = str(Path(script).absolute())
    return [str(win_startup.gui_python()), script, *sys.argv[1:]]


def drain_until_idle(busy, *, deadline_s: float, clock=time.monotonic,
                     sleep=time.sleep) -> bool:
    """Poll until busy() is False (True) or deadline_s has passed (False).

    Side effects: sleeps on the calling thread.
    """
    give_up = clock() + deadline_s
    while busy():
        if clock() >= give_up:
            return False
        sleep(DRAIN_POLL_S)
    return True


def spawn_restart() -> None:
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(restart_command(), close_fds=True, creationflags=flags,
                     cwd=str(Path(__file__).resolve().parent))


def setup(profile: str, model: str | None, speed: str = "accurate") -> int:
    """Download the pinned model and the pinned Silero VAD now (the Mac's
    ``sotto.py setup``), so the first launch is ready.  Offline: nothing is
    fetched."""
    repo = resolve_config(profile, model, "auto", speed).model_repo
    try:
        path = win_asr.resolve_model_dir(repo, cache_dir=MODEL_CACHE_DIR,
                                         offline=offline_requested(), log=log)
    except win_asr.SetupError as exc:
        log(f"✗ {exc}")
        return 1
    log(f"✓ speech model ready ({repo.split('/')[-1]}) in {path}")
    import vad
    try:
        ready = vad.install(allow_download=not offline_requested())
        log(f"✓ voice detection ready (Silero VAD v6.2) in {ready.parent}")
    except Exception as exc:  # advisory: transcription works without it
        log(f"! voice detection not installed ({str(exc)[:120]}); run setup again later")
    return 0


def doctor(profile: str, model: str | None, device: str, speed: str = "accurate") -> None:
    """Content-free checks: no model load, no download, no microphone open."""
    ok = True
    log(f"  data: {DATA_DIR}")
    log(f"  model cache: {MODEL_CACHE_DIR}")
    count = win_asr.cuda_device_count()
    plan = win_asr.device_plan(device, count)
    if count > 0:
        log(f"✓ CUDA: {count} device(s) visible to CTranslate2")
    else:
        log("! CUDA: no device visible to CTranslate2 — ASR uses the CPU")
    log(f"  ASR device ({device}): tries "
        + " then ".join(f"{name} ({compute})" for name, compute in plan)
        + "; the first startup transcription decides")
    if device == "cuda" and count == 0:
        ok = False
        log("✗ --device cuda / SOTTO_DEVICE=cuda requested, but no CUDA device is visible")
    try:
        repo = resolve_config(profile, model, "auto", speed).model_repo
        if Path(repo).expanduser().is_dir():
            if win_asr.complete_local(Path(repo).expanduser()):
                log(f"✓ model: local directory {repo}")
            else:
                ok = False
                log(f"✗ model: {repo} lacks one of {', '.join(win_asr.MODEL_FILES)}")
        else:
            pinned_repo, revision = win_asr.pinned(repo)
            snapshot = win_asr.snapshot_dir(MODEL_CACHE_DIR, pinned_repo, revision)
            if win_asr.complete(snapshot, win_asr.required_files(pinned_repo)):
                log(f"✓ model: {pinned_repo}@{revision[:12]} cached")
            elif offline_requested():
                ok = False
                log(f"✗ model: {pinned_repo}@{revision[:12]} not cached and offline "
                    "mode is on — run once online to download it")
            else:
                log(f"! model: {pinned_repo}@{revision[:12]} not downloaded yet — "
                    "the first run downloads it (network needed once)")
    except ValueError as exc:
        ok = False
        log(f"✗ model: {exc}")
    try:
        import sounddevice as sd
        default_in = sd.default.device[0]
        if default_in is None:
            ok = False
            log("✗ microphone: no default input device")
        else:
            info = sd.query_devices(default_in, "input")
            log(f"✓ microphone: {info['name']} "
                f"({int(info['default_samplerate'])} Hz default)")
    except Exception as exc:
        ok = False
        log(f"✗ microphone: {str(exc)[:120]}")
    try:
        import vad
        if (vad.MODEL_PATH.is_file()
                and hashlib.sha256(vad.MODEL_PATH.read_bytes()).hexdigest() == vad.MODEL_SHA256):
            log(f"✓ VAD model: {vad.MODEL_PATH} (Silero v6.2)")
        else:
            log(f"! VAD model missing: run `sotto_win.py setup` (it installs {vad.MODEL_PATH}); "
                "transcription still works, but silence can type a stock phrase")
    except Exception as exc:
        log(f"! VAD: {str(exc)[:120]}")
    try:
        ctypes.windll.user32.SendInput
        log("✓ SendInput available (note: Windows drops synthetic input to "
            "elevated/admin windows)")
    except Exception as exc:
        ok = False
        log(f"✗ SendInput: {str(exc)[:120]}")
    try:
        import tkinter  # noqa: F401
        log("✓ tkinter available (Settings and Correct dialogs)")
    except Exception:
        log("! tkinter missing: the tray menu works, but Settings/Correct dialogs need "
            "Python's 'tcl/tk and IDLE' option (python.org installer → Modify)")
    if not ok:
        sys.exit(1)
    log("✓ ready")


@contextmanager
def console_signal_handlers(stop):
    """Ctrl-C (SIGINT), Ctrl-Break (SIGBREAK) and SIGTERM call ``stop``; the
    previous handlers come back afterwards.  Only the main thread installs
    them (an embedded caller keeps its own).

    Closing the console window also raises SIGBREAK, but Windows ends the
    process as soon as the C runtime's console handler returns, before any
    Python handler runs (measured: gone within 0.01 s), so a console close
    never reaches ``stop`` or the teardown.
    """
    previous: dict[int, object] = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGBREAK):
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, stop)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def history_command(command: str, *, limit: int, entry_id: str | None,
                    yes: bool) -> int:
    """List, delete or clear History through the shared nonadaptive paths."""
    from adaptive_learning import AdaptiveLearning
    store = HistoryStore()
    coordinator = LearningCoordinator(store, LearningStore())
    # The same always-on revocation guard the Mac menu passes.
    dependency_guard = AdaptiveLearning(store.base_dir)
    if command == "history":
        # Redirected output would otherwise use the ANSI code page and fail
        # on non-Latin text; a real console already writes UTF-16 either way.
        sys.stdout.reconfigure(encoding="utf-8")
        entries = store.entries(limit=limit)
        for entry in entries:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.get("ts", 0)))
            print(f"{entry['id']}  {stamp}  {float(entry.get('duration') or 0):5.1f}s  "
                  f"{entry.get('text', '')}")
        log(f"{len(entries)} of at most {store.keep} retained entries · {store.base_dir}")
        return 0
    if command == "history-delete":
        if not entry_id:
            log("history-delete requires --id ID (see `sotto_win.py history`)")
            return 2
        if not sotto.nonadaptive_delete(dependency_guard, coordinator, entry_id):
            log(f"no history entry {entry_id}")
            return 1
        log(f"deleted {entry_id} and its audio")
        return 0
    if not yes:
        log("history-clear deletes every transcript and recording; "
            "run again with --yes to confirm")
        return 2
    sotto.nonadaptive_clear(dependency_guard, coordinator)
    log("history cleared")
    return 0


class Controller:
    """Tray/Settings actions on the running app (win_ui calls these off its menu thread)."""

    # The drains' time source; tests drive them on a virtual clock.
    clock = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)

    def __init__(self, **parts) -> None:
        self.__dict__.update(parts)
        self.restart_requested = False
        # Restart, Update and Quit close the capture gate under this lock, so
        # one of them never reopens a gate another has closed.
        self._lifecycle_lock = threading.Lock()

    # dictation
    def start_now(self) -> str | None:
        """Hands-free from the tray; refused (with a message) while a
        Restart, Update or Quit drains.  A start that races the gate closing
        is refused by on_start, which ends the gesture again."""
        if self.shutdown.requested():
            return None
        closed = self.lifecycle["closed"]
        if closed:
            log(f"○ {closed} — dictation not started")
            return f"Sotto is {closed}; dictation was not started."
        if self.engine.force_start() and self.recording():
            log("● dictation started from the tray (hands-free)")
        return None

    def finish_now(self) -> None:
        """Finish from the menu.  The menu took the focus away from the app
        being dictated into, so this dictation is copied, not inserted."""
        if self.shutdown.requested():
            return
        ended = self._end_recording()
        if ended == "gesture":
            log("● finished from the tray (the text is copied)")
        elif ended == "orphan":
            log("● orphan capture finished from the tray")

    def _end_recording(self) -> str | None:
        """End the recording in progress so it is transcribed, not dropped.
        A tray action took the focus, so its text is copied, not inserted.
        The finish may run later on whichever thread is draining gesture
        actions, so "copy" travels with it.

        Returns "gesture", "orphan" (the capture outlived the gesture engine's
        state and was ended directly) or None when nothing was recording.
        """
        if self.engine.force_finish(finish=lambda: self.on_finish(copy_only=True)):
            return "gesture"
        if self.capture.is_active():
            self.on_finish(copy_only=True)  # gesture engine desynced; end the capture anyway
            return "orphan"
        return None

    def recording(self) -> bool:
        return self.engine.snapshot()[0]

    def in_flight(self) -> list[str]:
        """The captures, transcriptions and insertions still in flight, in words."""
        work = []
        if self.capture.is_active() or self.finishing():
            work.append("a recording")
        if self.jobs.unfinished_tasks:
            work.append(f"{self.jobs.unfinished_tasks} transcription job(s)")
        if self.deliveries.unfinished_tasks:
            work.append(f"{self.deliveries.unfinished_tasks} insertion(s)")
        if win_inject.paste_settling():
            work.append(UNREAD_PASTE)
        return work

    def busy(self) -> bool:
        """A capture, a transcription or an insertion is still in flight."""
        return bool(self.in_flight())

    # history
    def entries(self, limit: int = 10) -> list[dict]:
        return self.store.entries(limit=limit)

    def copy(self, entry_id: str) -> None:
        entry = self.store.get(entry_id)
        if entry:
            win_inject.copy_text(entry["text"])

    def retry(self, entry_id: str) -> str:
        """Transcribe a History entry again; returns the tray's message.
        Refused while a Restart, Update or Quit drains: new work would keep
        it waiting.  Checked and queued under _lifecycle_lock, so a drain
        that closes the gate afterwards sees this Retry as in flight."""
        with self._lifecycle_lock:
            closed = self.lifecycle["closed"]
            if closed or self.shutdown.requested():
                return f"Not retried: Sotto is {closed or 'stopping'}."
            self.shutdown.enqueue(self.jobs, ("retry", entry_id, time.monotonic(),
                                              uuid.uuid4().hex, self.current_speech_config()))
        return "Retrying — the new text is copied when ready."

    def delete(self, entry_id: str) -> None:
        if not self.shutdown.requested():
            sotto.nonadaptive_delete(self.dependency_guard, self.coordinator, entry_id)

    def clear(self) -> None:
        if not self.shutdown.requested():
            sotto.nonadaptive_clear(self.dependency_guard, self.coordinator)
            log("history cleared from the tray")

    def correct(self, entry_id: str, before: str, after: str, revision: int | None) -> list:
        """Save a correction; returns dictionary rules it suggests (never added here)."""
        if self.shutdown.requested():
            raise RuntimeError("Sotto is shutting down")
        if sotto.nonadaptive_correct(self.dependency_guard, self.coordinator, entry_id, after,
                                     expected_revision=revision) is None:
            raise ValueError("The recording changed or no longer exists; open Correct again.")
        log("✓ correction saved")
        return dictionary.suggest(before, after, dictionary.load(dictionary.DICTIONARY_PATH))

    def add_rules(self, rules: list) -> int:
        added = dictionary.add(rules, dictionary.DICTIONARY_PATH)
        log(f"dictionary: added {added} rule(s)")
        return added

    # preferences
    def language(self) -> str:
        with self.language_lock:
            return self.language_state["mode"]

    def set_language(self, mode: str) -> None:
        selected = save_language_mode(mode)
        with self.language_lock:
            self.language_state["mode"] = selected
        log(f"language: {selected} (applies to the next recording)")

    def saved_languages(self) -> list[str]:
        return user_settings.load()["languages"]

    def set_speed(self, speed: str) -> str:
        user_settings.save({"speed": speed})
        if speed == self.speed:
            return "Speed unchanged."
        return self.restart()

    def settings(self) -> dict:
        import win_startup
        saved = user_settings.load()
        saved["launch_at_login"] = win_startup.enabled()
        return saved

    def save_settings(self, updates: dict) -> list[str]:
        """Persist; apply what can change live; say what needs a restart."""
        import win_startup
        notes = []
        # The registry value is the truth (Task Manager can remove it too);
        # touch it only when the choice actually changes.
        if "launch_at_login" in updates and bool(updates["launch_at_login"]) != win_startup.enabled():
            win_startup.set_enabled(bool(updates["launch_at_login"]),
                                    win_startup.launch_command(DATA_DIR, MODEL_CACHE_DIR))
        saved = user_settings.save(updates)
        trigger = win_hotkey.supported(saved["trigger"])
        if self.trigger_locked and trigger != self.hook.trigger:
            notes.append("The --trigger option wins for this run; the new key applies after restart.")
        elif trigger != self.hook.trigger:
            self.hook.set_trigger(trigger)
            log(f"trigger: {trigger}")
            notes.append(f"Trigger is now {win_hotkey.label(trigger)}.")
        if saved["speed"] != self.speed:
            notes.append("Speed applies after restart.")
        notes.append("Languages, insertion and spacing apply to the next dictation.")
        return notes

    def progress(self) -> list[str]:
        entries = self.store.entries(limit=self.store.keep)
        rules = len(dictionary.load(dictionary.DICTIONARY_PATH))
        return progress.lines(progress.summarize(entries, rules=rules),
                              progress.load_totals(sotto.totals_path()))

    def open_dictionary(self) -> None:
        path = dictionary.ensure_file(dictionary.DICTIONARY_PATH)
        subprocess.Popen(["notepad.exe", str(path)], close_fds=True)

    def show_data_folder(self) -> None:
        os.startfile(DATA_DIR)

    # lifecycle
    def restart(self) -> str:
        """Restart once nothing is in flight.  A dictation being recorded,
        transcribed or inserted finishes first, however slowly the model runs
        (a long dictation on the CPU can take well over 20 s), up to
        RESTART_DRAIN_CAP_S: past that the model call is taken to be wedged,
        and the restart goes ahead and logs what it abandoned.  A Quit chosen
        meanwhile takes over (and waits at most QUIT_DRAIN_S).

        Side effects: refuses new captures from now on; requests shutdown once idle.
        """
        with self._lifecycle_lock:
            if self.shutdown.requested() or self.lifecycle["closed"] == "quitting":
                return "Sotto is already stopping."
            if self.restart_requested:
                return "Restarting…"
            if self.lifecycle["closed"]:
                return "Sotto is updating; it restarts when that is done."
            self.restart_requested = True
            self._close_gate("restarting")  # on_start takes no new capture from now on
        if not self.busy():
            log("● restart requested from the tray")
            self.shutdown.request()
            return "Restarting…"

        def when_idle() -> None:
            idle = drain_until_idle(lambda: self.busy() and not self.shutdown.requested(),
                                    deadline_s=RESTART_DRAIN_CAP_S,
                                    clock=self.clock, sleep=self.sleep)
            if not self.restart_requested:
                return  # Quit won
            if idle:
                log("● restarting after the current dictation")
            else:
                log(f"! restarting anyway after {RESTART_DRAIN_CAP_S / 60:.0f} minutes; "
                    f"abandoned: {', '.join(self.in_flight())}")
            self.shutdown.request()

        threading.Thread(target=when_idle, daemon=True).start()
        return "Restarting after the current dictation…"

    def check_updates(self):
        """Compare this checkout with GitHub; only when the user asks."""
        import updates
        from offline_runtime import offline_requested
        return updates.check(Path(__file__).resolve().parent, offline=offline_requested())

    def update_and_restart(self) -> str:
        """Hand the update to scripts/update-windows.ps1: Windows keeps a
        running Python's files in use, so it waits for Sotto to quit, updates,
        and starts Sotto again.

        Side effects: closes the capture gate before the busy check, so no
        recording can start between the check and the quit; reopens it when
        the update is refused or the updater cannot be started.
        """
        with self._lifecycle_lock:
            if self.shutdown.requested() or self.lifecycle["closed"]:
                return "Sotto is already restarting or quitting."
            self._close_gate("updating")
        try:
            work = self.in_flight()
            if work:
                self._reopen_gate("updating")
                if work == [UNREAD_PASTE]:
                    # A Ctrl+V the app never reads holds Update until the
                    # clipboard is put back, at most RENDER_WAIT_S.
                    return ("The last paste has not been read by its app yet; "
                            f"try again in {win_inject.RENDER_WAIT_S:.0f} seconds.")
                return "Finish the current dictation first, then update."
            root = Path(__file__).resolve().parent
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-ExecutionPolicy",
                              "Bypass", "-File", str(root / "scripts" / "update-windows.ps1"),
                              "-DataDir", str(DATA_DIR), "-VenvDir", sys.prefix],
                             close_fds=True, creationflags=flags, cwd=str(root))
        except BaseException:
            self._reopen_gate("updating")
            raise
        log("● updating Sotto; it restarts when the update is done")
        self.quit()
        return "Updating — Sotto restarts when it is done."

    def quit(self) -> None:
        """Quit once the dictation in flight is done: recorded, transcribed
        and inserted (the app has read its paste), waiting at most
        QUIT_DRAIN_S — the Mac's Quit.  Returns at once: the tray calls this
        on its menu thread, so the drain runs on a thread of its own.

        Side effects: refuses new recordings from now on; ends a recording in
        progress (even one the gesture engine lost) as the tray's Finish does;
        requests shutdown once idle or when the wait runs out.
        """
        with self._lifecycle_lock:
            self.restart_requested = False  # Quit wins over a pending restart
            if self.shutdown.requested() or self.lifecycle["closed"] == "quitting":
                return
            self._close_gate("quitting")

        def finish_then_stop() -> None:
            # Whatever fails here, Quit stops Sotto: the gate is already
            # closed as "quitting", so a later Quit would return at once.
            try:
                try:
                    ended = self._end_recording()
                except Exception as exc:
                    ended = None
                    log(f"! could not finish the recording before quitting: {str(exc)[:160]}")
                if ended:
                    log(f"● finishing the {'orphan capture' if ended == 'orphan' else 'recording'} "
                        "before quitting (the text is copied)")
                if not drain_until_idle(lambda: self.busy() and not self.shutdown.requested(),
                                        deadline_s=QUIT_DRAIN_S, clock=self.clock,
                                        sleep=self.sleep):
                    log(f"! quitting anyway after {QUIT_DRAIN_S:.0f}s; "
                        f"abandoned: {', '.join(self.in_flight())}")
            except Exception as exc:
                log(f"! quitting without waiting: {str(exc)[:160]}")
            finally:
                self.shutdown.request()

        threading.Thread(target=finish_then_stop, daemon=True, name="sotto-quit").start()

    def console_stop(self, *_signal) -> None:
        """Ctrl-C, Ctrl-Break or SIGTERM in a console run: the first one
        quits like the tray's Quit, finishing the dictation in flight (up to
        QUIT_DRAIN_S); another one while that drain runs stops at once.
        Never a restart.  Runs as a signal handler on the main thread, which
        never holds _lifecycle_lock."""
        with self._lifecycle_lock:
            draining = self.lifecycle["closed"] == "quitting"
        if draining or self.shutdown.requested():
            self.stop_now()
        else:
            self.quit()  # also cancels a restart waiting for the dictation

    def stop_now(self) -> None:
        """Stop without waiting for the work in flight, and never restart."""
        with self._lifecycle_lock:
            self.restart_requested = False
            self.shutdown.request()

    def _close_gate(self, reason: str) -> None:
        """Refuse new recordings; on_start logs ``reason``.  Caller holds _lifecycle_lock."""
        self.lifecycle["closed"] = reason  # set first: whoever sees the gate closed reads it
        self.capture_gate.close()

    def _reopen_gate(self, reason: str) -> None:
        """Reopen the gate, only if ``reason`` still holds it (a Quit may have taken over)."""
        with self._lifecycle_lock:
            if self.lifecycle["closed"] == reason:
                self.capture_gate.reopen()
                self.lifecycle["closed"] = None


def run(trigger: str, profile: str, model: str | None, language: str | None,
        glossary: str | None, idle_release: float, device: str = "auto", *,
        speed: str = "accurate", tray: bool = False, trigger_locked: bool = False) -> bool:
    """Dictate until shutdown; returns True when a restart was requested."""
    speech_config = resolve_config(profile, model, language, speed)
    model_repo = speech_config.model_repo
    glossary_terms = load_glossary(glossary) if glossary else ()

    language_lock = threading.Lock()
    language_state = {"mode": "auto" if speech_config.language is None else speech_config.language}

    def current_speech_config() -> SpeechConfig:
        with language_lock:
            return with_language(speech_config, language_state["mode"])

    shutdown = sotto.ShutdownBoundary()
    ui = None
    if tray:
        import win_ui
        ui = win_ui.TrayApp(quit=shutdown.request, log=log)
        ui.start()

    def status(phase: str) -> None:
        if ui is not None:
            ui.set_phase(phase)

    # Fail fast on the slow path: resolve (first run: download) and warm the
    # pinned model before arming the hotkey, so a broken setup is a startup
    # error, not a lost dictation.  The warmup also picks CUDA or CPU.
    try:
        status("Preparing the speech model…")
        model_dir = win_asr.resolve_model_dir(model_repo, cache_dir=MODEL_CACHE_DIR,
                                              offline=offline_requested(), log=log)
        whisper = win_asr.LocalCT2Whisper(model_dir, device=device, log=log)
        log(f"warming up {model_repo.split('/')[-1]} from {model_dir} ...")
        status("Warming up the speech model…")
        started = time.monotonic()
        sotto.warm_speech_runtime(
            adaptive=False, model=model_repo, speech_config=speech_config,
            mlx_whisper=win_asr.LanguageRestricted(
                whisper, delivery_prefs(user_settings.load()).languages))
        log(f"✓ model ready on {whisper.device} ({whisper.compute_type}) "
            f"in {time.monotonic() - started:.1f}s")
    except BaseException:
        if ui is not None:
            ui.stop()
        raise
    if shutdown.requested():  # Quit from the tray during startup
        if ui is not None:
            ui.stop()
        log("stopped before listening")
        return False

    # An unreadable history.jsonl must not stop the app at every login (F16b):
    # dictation keeps working, History is left untouched, the user is told once.
    store = HistoryStore(tolerate_unreadable=True)
    coordinator = sotto.history_coordinator(store, LearningStore())
    if store.unreadable is not None:
        message = sotto.history_unreadable_message(store)
        log(f"! History unreadable ({store.unreadable}): {message}")
        if ui is not None:  # a toast truncates this long a message; the box blocks only its thread
            threading.Thread(target=win_ui.message_box, args=(sotto.HISTORY_UNREADABLE_TITLE, message),
                             kwargs={"error": True}, daemon=True).start()
    sotto.seed_totals(store)
    from adaptive_learning import AdaptiveLearning
    # The same always-on revocation guard the Mac menu passes.
    dependency_guard = AdaptiveLearning(store.base_dir)

    capture = win_capture.WinCapture()
    capture.idle_release_s = idle_release  # no-op: fresh stream per capture

    # Single-flight transcription: one worker, strict FIFO — same contract as
    # the Mac's run(): live dictations never run the model concurrently and
    # results arrive in capture order.
    jobs: queue.Queue = queue.Queue()
    # One insertion thread, strict FIFO: deferred inserts can never overtake
    # or interleave with each other.
    deliveries: queue.Queue = queue.Queue()
    model_activity = {"last_finished": time.monotonic(), "rewarming": False}
    vad_warnings: set[str] = set()
    # Restart, Update and Quit close the gate (the reason goes in lifecycle);
    # on_start decides and begins a capture under it, so a closer either sees
    # that capture as in flight or the capture is refused.
    capture_gate = sotto.CaptureGate()
    lifecycle: dict = {"closed": None}
    # Captures between capture.end() and their enqueue still count as busy.
    finishing_lock = threading.Lock()
    finishing_count = [0]

    def finishing() -> bool:
        with finishing_lock:
            return finishing_count[0] > 0

    def ui_state(phase: str | None = None) -> None:
        # win_ui applies icon changes on its own thread: this may be the
        # keyboard-hook thread, which must never wait.
        if ui is None:
            return
        if phase is None:
            phase = ("recording" if engine.snapshot()[0] else
                     "transcribing" if jobs.unfinished_tasks else "idle")
        ui.set_state(phase)

    def live_job(job) -> None:
        (_, raw, native_rate, captured_ts, queued_at, _capture_id, job_config, released_at,
         copy_only) = job
        # Settings apply per dictation, read here on the worker (never on the
        # keyboard-hook thread).
        prefs = delivery_prefs(user_settings.load())
        started = time.monotonic()
        samples = sotto.prepare_for_whisper(raw, native_rate)
        seconds = len(samples) / sotto.SAMPLE_RATE
        if seconds < 0.2:
            log("○ sub-0.2s capture dropped (key-tap artifact)")
            return
        skip = sotto.asr_skip_reason(samples)
        if skip:
            # Exact zeros never reach Whisper — a dead capture path, not a
            # quiet room (same guard as the Mac).
            from audio_codec import prepare_canonical
            prepared = prepare_canonical(samples)
            if shutdown.requested():
                return
            text = sotto.DEAD_ROUTE_TEXT
            _, attempt_metadata = sotto._transcription_kwargs(job_config, glossary_terms, seconds)
            attempt_metadata.update({
                "vad": {"available": False, "speech_fraction": 0.0, "span_count": 0},
                "preprocessing": {"resampled_normalized": True, "vad_trimmed": False,
                                  "silence_collapsed": False, "outcome": "suspect"},
                "latency": {"queue_wait_seconds": round(time.monotonic() - queued_at, 4),
                            "asr_seconds": 0.0},
            })
            log(f"! not pasted ({skip})")
            try:
                sotto.finalize_primary_live_delivery(
                    append=lambda: coordinator.append_live(
                        text, prepared, seconds, model_repo, ts=captured_ts, raw_samples=raw,
                        raw_sample_rate=native_rate, provenance="live_suspect",
                        adaptive=False, **attempt_metadata),
                    adaptive_runtime=None, appended_publication=None, shutdown=shutdown,
                    inject=None)
            except Exception as exc:
                history_append_failed(exc)  # held: nothing to deliver
                return
            log(f"→ 0.00s · {len(text)} chars")
            return
        # VAD is advisory only: it trims dead air from long captures and
        # feeds the output-side no-speech verdict.  It must never block
        # transcription (2026-08-13/14).
        vad_available = True
        speech_fraction = 1.0
        speech_spans: list = []
        try:
            import vad
            speech_fraction, speech_spans = vad.analyze(samples)
        except Exception as exc:
            vad_available = False
            reason = str(exc)[:80]
            if reason not in vad_warnings:  # once per cause, not every dictation
                vad_warnings.add(reason)
                log(f"○ advisory VAD unavailable ({reason}); transcribing without trimming")
        vad_metadata = {"available": vad_available,
                        "speech_fraction": round(float(speech_fraction), 4),
                        "span_count": len(speech_spans)}
        preprocessing = {"resampled_normalized": True, "vad_trimmed": False,
                         "silence_collapsed": False}
        if speech_spans and seconds > 10:
            before = seconds
            samples = vad.trim_to_speech(samples, speech_spans)
            trimmed = before - len(samples) / sotto.SAMPLE_RATE
            preprocessing["vad_trimmed"] = trimmed > 0
            if trimmed > 1:
                log(f"  trimmed {trimmed:.0f}s of non-speech (VAD)")
        if len(samples) / sotto.SAMPLE_RATE > sotto.LONG_CAPTURE_S:
            before = len(samples)
            samples = sotto.collapse_silence(samples)
            preprocessing["silence_collapsed"] = len(samples) < before
            if len(samples) < before:
                log(f"  trimmed {(before - len(samples)) / sotto.SAMPLE_RATE:.0f}s of dead air")
        from audio_codec import prepare_canonical
        prepared = prepare_canonical(samples)
        if shutdown.requested():
            return
        restricted = win_asr.LanguageRestricted(whisper, prefs.languages)
        try:
            text, attempt_metadata = sotto.transcribe_prepared(
                restricted, prepared, job_config, glossary_terms)
        except Exception as exc:
            keep_failed_capture(exc, prepared, raw, native_rate, captured_ts, job_config,
                                vad_metadata, preprocessing, queued_at, started)
            return
        # A backend may ignore cancellation: its result is never pasted or
        # persisted after shutdown.
        if shutdown.requested():
            return
        elapsed = time.monotonic() - started
        model_activity["last_finished"] = time.monotonic()
        move_detected_language(attempt_metadata, preprocessing)
        attempt_metadata.update({
            "vad": vad_metadata, "preprocessing": preprocessing,
            "latency": {"queue_wait_seconds": round(started - queued_at, 4),
                        "asr_seconds": round(elapsed, 4)}})
        prepared_seconds = len(prepared.asr_samples) / sotto.SAMPLE_RATE
        text, reason = screen_transcript(text, prepared_seconds, speech_fraction, preprocessing)
        voice_action = None
        if not reason:
            # After every guard, never on text kept back (the Mac's rule).
            text = sotto.apply_personal_dictionary(text, preprocessing)
            text, voice_action = sotto.apply_voice_cleanup(
                text, preprocessing, preprocessing.get("detected_language") or job_config.language)
        # Key release until the text is ready to insert (the Mac's measure,
        # read by the shared progress summary).
        attempt_metadata["latency"]["release_to_text_seconds"] = round(
            time.monotonic() - released_at, 4)
        if shutdown.requested():
            return
        if reason:
            log(f"! not pasted ({reason}) — kept in history")
            preprocessing["outcome"] = "suspect"
        deliver = None if reason or not (text or voice_action) else lambda: deliveries.put(
            (SCRATCH if voice_action == "scratch" else text, prefs, copy_only, time.monotonic()))
        try:
            appended = sotto.finalize_primary_live_delivery(
                append=lambda: coordinator.append_live(
                    text, prepared, prepared_seconds, model_repo, ts=captured_ts,
                    raw_samples=raw, raw_sample_rate=native_rate,
                    provenance="live_suspect" if reason else "live", adaptive=False,
                    **attempt_metadata),
                adaptive_runtime=None, appended_publication=None, shutdown=shutdown,
                inject=deliver)
        except Exception as exc:
            appended = None
            history_append_failed(exc, deliver)
        if appended is not None and text and not reason and voice_action is None:
            sotto.record_totals(text, len(raw) / max(native_rate, 1.0))
        log(f"→ {attempt_metadata['latency']['release_to_text_seconds']:.2f}s after release "
            f"(speech model {elapsed:.2f}s) · {len(text)} chars")

    def keep_failed_capture(exc, prepared, raw, native_rate, captured_ts, job_config,
                            vad_metadata, preprocessing, queued_at, started) -> None:
        """The speech model raised (CUDA out of memory, a driver error): the
        recording must not be lost.  It is kept as a suspect History row that
        Retry can transcribe again, and the tray says so — unless History
        could not take the row (an append that raised, or History unreadable
        since startup), when the tray says the recording is lost instead.

        Side effects: appends one History row with its audio; a tray notification.
        """
        model_activity["last_finished"] = time.monotonic()
        log(f"! transcription failed: {str(exc)[:160]} — the recording is kept in History")
        if shutdown.requested():
            return
        prepared_seconds = len(prepared.asr_samples) / sotto.SAMPLE_RATE
        _, attempt_metadata = sotto._transcription_kwargs(job_config, glossary_terms,
                                                         prepared_seconds)
        preprocessing.update({"outcome": "suspect", "transcription_error": type(exc).__name__})
        attempt_metadata.update({
            "vad": vad_metadata, "preprocessing": preprocessing,
            "latency": {"queue_wait_seconds": round(started - queued_at, 4),
                        "asr_seconds": round(time.monotonic() - started, 4)}})
        try:
            row = sotto.finalize_primary_live_delivery(
                append=lambda: coordinator.append_live(
                    TRANSCRIPTION_FAILED_TEXT, prepared, prepared_seconds, model_repo,
                    ts=captured_ts, raw_samples=raw, raw_sample_rate=native_rate,
                    provenance="live_suspect", adaptive=False, **attempt_metadata),
                adaptive_runtime=None, appended_publication=None, shutdown=shutdown,
                inject=None)
        except Exception as append_exc:
            log(f"! failed capture not saved ({type(append_exc).__name__}: "
                f"{str(append_exc)[:160]})")
            row = None
        # UnsavedHistory (History unreadable) hands back a row it never wrote.
        saved = row is not None and row.get("saved", True)
        if ui is not None:
            ui.notify("Transcription failed. The recording is in History: "
                      "choose Retry there to transcribe it again." if saved else
                      "Transcription failed, and the recording could not be saved to "
                      "History either; please dictate again.")

    def history_append_failed(exc: Exception, deliver=None) -> None:
        """History could not take the live row (the Mac's F16a): disk full,
        a locked or damaged store.  The dictation still reaches the user,
        who is told once that it was not saved.

        Side effects: runs ``deliver`` (the insertion, copy or "scratch that"
        a saved row would have queued); one tray notification; a log line.
        Nothing once shutdown is requested.
        """
        log(f"! History append failed ({type(exc).__name__}: {str(exc)[:160]})"
            + (" — delivering the text anyway" if deliver is not None else ""))
        if shutdown.requested():
            return
        if deliver is not None:
            deliver()
        if ui is not None:
            ui.notify("History could not be saved. "
                      + ("The text was still delivered, but it is not in History."
                         if deliver is not None else
                         "The held-back text could not be kept in History."))

    def retry_job(job) -> None:
        _, entry_id, queued_at, _capture_id, job_config = job
        prefs = delivery_prefs(user_settings.load())
        snapshot = coordinator.snapshot_for_retry(entry_id)
        if snapshot is None:
            log("↻ retry skipped — entry deleted")
            return
        prepared_seconds = len(snapshot.samples) / sotto.SAMPLE_RATE
        # Retry uses the immutable canonical decode from its snapshot; it
        # must never re-trim or re-collapse persisted inference input.
        if sotto.asr_skip_reason(snapshot.samples):
            _, attempt_metadata = sotto._transcription_kwargs(job_config, glossary_terms,
                                                             prepared_seconds)
            attempt_metadata.update({
                "vad": {"available": None, "speech_fraction": None, "span_count": None},
                "preprocessing": {"retry_canonical": True, "outcome": "suspect"},
                "latency": {"queue_wait_seconds": round(time.monotonic() - queued_at, 4),
                            "asr_seconds": 0.0}})
            coordinator.commit_retry(entry_id, sotto.DEAD_ROUTE_TEXT, snapshot.expected_revision,
                                     model=model_repo, provenance="retry", **attempt_metadata)
            log("↻ retry skipped — dead microphone (all-zero capture)")
            return
        started = time.monotonic()
        if shutdown.requested():
            return
        restricted = win_asr.LanguageRestricted(whisper, prefs.languages)
        text, attempt_metadata = sotto.transcribe_canonical_samples(
            restricted, snapshot.samples, job_config, glossary_terms)
        model_activity["last_finished"] = time.monotonic()
        if shutdown.requested():
            return
        preprocessing: dict = {"retry_canonical": True}
        move_detected_language(attempt_metadata, preprocessing)
        # The same guards and cleanup as live dictation (pipeline.screen, like
        # the Mac's Retry).  No VAD verdict for a retry (speech_fraction 1.0),
        # and Retry never undoes anything: a "scratch that" result keeps its words.
        screened = pipeline.screen(
            text, seconds=prepared_seconds, speech_fraction=1.0,
            language=preprocessing.get("detected_language") or job_config.language, log=log)
        preprocessing.update(screened.receipts)
        text = (preprocessing.pop("voice")["asr_text"] if screened.voice_action == "scratch"
                else screened.text)
        reason = screened.held
        if reason:
            preprocessing["outcome"] = "suspect"
        attempt_metadata.update({
            "vad": {"available": None, "speech_fraction": None, "span_count": None},
            "preprocessing": preprocessing,
            "latency": {"queue_wait_seconds": round(started - queued_at, 4),
                        "asr_seconds": round(time.monotonic() - started, 4)}})
        if shutdown.requested():
            return
        committed = coordinator.commit_retry(entry_id, text, snapshot.expected_revision,
                                             model=model_repo, provenance="retry",
                                             **attempt_metadata)
        if committed is None:
            log("↻ retry discarded — the entry changed meanwhile")
        elif reason:
            log(f"↻ retried · kept in history ({reason})")
        elif not shutdown.requested():
            try:
                win_inject.copy_text(text)  # the Mac copies a retry; it never pastes it
                log(f"↻ retried · {len(text)} chars copied")
            except OSError as exc:
                log(f"↻ retried · {len(text)} chars in History (not copied: {str(exc)[:80]})")

    def transcribe_worker() -> None:
        while not shutdown.requested():
            try:
                job = jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if shutdown.requested():
                    continue
                if job[0] == "warmup":
                    try:
                        sotto.warm_speech_runtime(
                            adaptive=False, model=model_repo,
                            speech_config=current_speech_config(),
                            mlx_whisper=win_asr.LanguageRestricted(
                                whisper, delivery_prefs(user_settings.load()).languages))
                        log("✓ model refreshed after idle")
                    finally:
                        model_activity["last_finished"] = time.monotonic()
                        model_activity["rewarming"] = False
                elif job[0] == "live":
                    live_job(job)
                elif job[0] == "retry":
                    retry_job(job)
                else:
                    raise RuntimeError("unknown transcription job")
            except Exception as exc:
                log(f"! transcription failed: {str(exc)[:160]}")
                if ui is not None and job[0] in ("live", "retry"):
                    ui.notify("A dictation could not be transcribed; see sotto.log in the data folder.")
            finally:
                jobs.task_done()
                ui_state()

    transcription_thread = threading.Thread(target=transcribe_worker, daemon=True)

    # Gesture callbacks mark the capture boundary, exactly as on the Mac.
    def on_start() -> None:
        if shutdown.requested():
            return
        with capture_gate.starting() as may_start:
            if may_start:
                cold = capture.begin()
        if not may_start:
            # A restart, update or Quit is draining: a new capture would
            # keep it waiting and then be cut off.  The mic stays closed, and
            # the gesture ends too (the Mac's on_mic_failed does the same), or
            # the engine would believe it is recording: the tray would offer
            # "Finish dictation" and a hands-free start would stay armed.
            log(f"○ {lifecycle['closed']} — this press is ignored")
            engine.force_finish()
            return
        ui_state("recording")
        now = time.monotonic()
        # CPU int8 needs no rewarm, and one decode there costs about as much
        # as a dictation, which would sit in the queue ahead of this capture.
        if whisper.device != "cpu" and sotto.model_rewarm_due(
                model_activity["last_finished"], now,
                rewarming=bool(model_activity["rewarming"]),
                queue_idle=jobs.empty(), adaptive=False):
            model_activity["rewarming"] = True
            if not shutdown.enqueue(jobs, ("warmup",)):
                model_activity["rewarming"] = False
        if not cold:
            log("● recording")
            return
        # Cold start: the stream opens off-thread and no audio exists until
        # the first block lands.
        log("● waking the mic — speak in a moment")

        def announce_live() -> None:
            deadline = time.monotonic() + 10
            while capture.is_waking() and time.monotonic() < deadline:
                time.sleep(0.05)
            if engine.snapshot()[0]:
                log("● recording (mic live)")

        threading.Thread(target=announce_live, daemon=True).start()

    def on_finish(copy_only: bool = False) -> None:
        released_at = time.monotonic()
        if shutdown.requested():
            shutdown.stop_capture(capture)
            return
        with finishing_lock:
            finishing_count[0] += 1
        try:
            captured_ts = time.time()
            raw = capture.end()
            capture.release_soon()  # the stream is already closed; kept for parity
            seconds = len(raw) / max(capture.native_rate, 1.0)
            log(f"○ {seconds:.2f}s captured")
            if seconds >= 0.25:
                shutdown.enqueue(jobs, ("live", raw, capture.native_rate, captured_ts,
                                        time.monotonic(), uuid.uuid4().hex,
                                        current_speech_config(), released_at, copy_only))
        finally:
            with finishing_lock:
                finishing_count[0] -= 1
        ui_state()

    def on_discard() -> None:
        capture.abort()
        capture.release_soon()
        log("○ tap ignored")
        ui_state()

    engine = sotto.GestureEngine(on_start, on_finish, on_discard)
    hook = win_hotkey.TriggerHook(engine, trigger=trigger)
    hook.start()
    hook_lock = threading.Lock()  # the resync poller's revive vs the teardown's stop

    def insert_one(text: str, prefs: DeliveryPrefs, ready_at: float) -> None:
        """Never insert while a modifier key is held — synthetic keystrokes or
        Ctrl+V during a hold race the release edge and could combine with it.
        A text not inserted within INSERT_WAIT_S of being ready stays in History
        instead; the wait counts from ready_at, so a queued text never gets a
        fresh wait after the ones ahead of it used theirs up."""
        deadline = ready_at + INSERT_WAIT_S
        deferred = False
        while hook.modifiers_held:
            if shutdown.requested():
                return
            if time.monotonic() >= deadline:
                log("! not inserted (a modifier key held for 2 minutes) — kept in history")
                return
            if not deferred:
                log("  insert deferred — a modifier key is still held")
                deferred = True
            time.sleep(0.05)
        if shutdown.requested():
            return
        if text is SCRATCH:
            undo_last_dictation()
            return
        try:
            # The insert's target and the user's key-downs so far, read
            # before it: "scratch that" may undo only this insert, there.
            target, keydowns = win_inject.foreground_identity(), hook.keydowns
            if win_inject.deliver(text, mode=prefs.insert_mode, spacing=prefs.spacing, log=log):
                last_delivery["insert"] = {"at": time.monotonic(), "target": target,
                                           "keydowns": keydowns}
        except OSError as exc:
            log(f"! not inserted ({str(exc)[:120]}) — kept in history")

    last_delivery: dict = {"insert": None}  # the insert "scratch that" may undo

    def undo_last_dictation() -> None:
        """'scratch that': the app's own Ctrl+Z, only for Sotto's own insert
        within the last minute, with the same window still in front and
        nothing typed by the user since (the Mac's DeliveryQueue._undo)."""
        import voice_commands
        last = last_delivery["insert"]
        if last is None or time.monotonic() - last["at"] > voice_commands.SCRATCH_WINDOW_S:
            log("  scratch that: nothing recent to undo")
            return
        target = win_inject.foreground_identity()
        if target is None or last["target"] is None:  # None == None proves nothing
            log("  scratch that: the window in front is unknown — nothing undone")
            return
        if target != last["target"]:
            log("  scratch that: the dictation went to another window — nothing undone")
            return
        if hook.keydowns != last["keydowns"]:
            log("  scratch that: you typed since the dictation — nothing undone")
            return
        win_inject.send_inputs(win_inject.chord_inputs(win_inject.VK_CONTROL, VK_Z))
        last_delivery["insert"] = None
        log("↶ scratch that — undid the last dictation")

    def copy_instead(text: str) -> None:
        """Finished from the tray: the menu took the focus, so copy instead."""
        try:
            win_inject.copy_text(text)
            log(f"→ copied ({len(text)} chars): finished from the tray")
            if ui is not None:
                ui.notify("Dictation copied — paste it with Ctrl+V.")
        except OSError as exc:
            log(f"! not copied ({str(exc)[:120]}) — kept in history")

    def delivery_worker() -> None:
        while not shutdown.requested():
            try:
                text, prefs, copy_only, ready_at = deliveries.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if copy_only and text is SCRATCH:
                    pass  # finished from the tray: there is no insertion to undo
                else:
                    copy_instead(text) if copy_only else insert_one(text, prefs, ready_at)
            except Exception as exc:
                log(f"! not inserted ({str(exc)[:120]}) — kept in history")
            finally:
                deliveries.task_done()

    def resync_poller() -> None:
        """Escape hatch: if a release edge is ever lost (listener death,
        sleep), a push-to-talk recording whose key is physically up for two
        consecutive polls gets its release synthesized. Hands-free is exempt —
        its key is legitimately up while recording."""
        misses = 0
        while not shutdown.event.wait(1.0):
            capture.tick()
            if not hook.alive():
                # The teardown requests shutdown before it stops the hook under
                # this lock, so a hook it stopped is never started again.
                with hook_lock:
                    if not shutdown.requested():
                        log("! keyboard listener died — restarting it")
                        hook.start()
            recording, hands_free = engine.snapshot()
            if recording and not hands_free and not hook.physically_down:
                misses += 1
                if misses >= 2:
                    log("! lost release recovered — finishing recording")
                    engine.released()
                    misses = 0
            else:
                misses = 0

    controller = Controller(
        shutdown=shutdown, engine=engine, capture=capture, jobs=jobs, deliveries=deliveries,
        store=store, coordinator=coordinator, dependency_guard=dependency_guard, hook=hook,
        on_finish=on_finish, current_speech_config=current_speech_config,
        language_lock=language_lock, language_state=language_state, speed=speed,
        trigger_locked=trigger_locked, model_repo=model_repo, whisper=whisper,
        lifecycle=lifecycle, finishing=finishing, capture_gate=capture_gate)

    transcription_thread.start()
    delivery_thread = threading.Thread(target=delivery_worker, daemon=True)
    delivery_thread.start()
    threading.Thread(target=resync_poller, daemon=True).start()

    log(f"listening on {hook.trigger} · hold to talk, double-tap for hands-free "
        f"· model: {model_repo.split('/')[-1]} · language: {language_state['mode']} "
        + ("· tray menu for options" if tray else "· ^C to quit"))
    if ui is not None:
        ui.attach(controller)
    # The handlers stay installed through teardown, so a Ctrl-C during the
    # teardown only re-requests shutdown instead of interrupting it.
    with console_signal_handlers(controller.console_stop):
        try:
            # Short waits keep the main thread returning to the interpreter,
            # which is where Windows delivers Ctrl-C and Ctrl-Break.
            while not shutdown.event.wait(0.2):
                pass
        except KeyboardInterrupt:
            # Only without the handlers above (run() off the main thread);
            # the teardown below starts at once, so there is no drain.
            controller.stop_now()
        finally:
            log("shutting down")
            shutdown.stop_capture(capture)
            shutdown.discard_queued(jobs)
            shutdown.discard_queued(deliveries)  # their text is already in History
            with hook_lock:
                hook.stop()
            transcription_thread.join(sotto.APP_DRAIN_TIMEOUT)
            if transcription_thread.is_alive():
                log("  an in-flight transcription was abandoned (never pasted or saved)")
            # An insertion in progress finishes (under the paste lock) before
            # the clipboard is put back.
            delivery_thread.join(3.0)
            try:
                win_inject.flush_clipboard()  # never leave dictated text on the clipboard
            except Exception:
                pass
            if ui is not None:
                ui.stop()
            log("✓ stopped")
    return controller.restart_requested


def _gui_process() -> None:
    """pythonw has no console: give print() a sink and keep libraries quiet."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:  # crisp Tk dialogs and menus on high-DPI screens
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Local push-to-talk dictation (Windows).")
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run", "setup", "doctor", "history", "history-delete",
                                 "history-clear"])
    parser.add_argument("--tray", action="store_true",
                        help="run as a system-tray app with menus and Settings")
    parser.add_argument("--trigger", default=None, choices=tuple(win_hotkey.TRIGGERS),
                        help="trigger key for this run (default: Settings, else right-ctrl)")
    parser.add_argument("--profile", default="auto",
                        choices=("auto",),
                        help="speech profile (default: auto)")
    parser.add_argument("--model", default=None,
                        help="local CTranslate2 model directory, or org/name@<commit>")
    parser.add_argument("--device", choices=win_asr.DEVICE_CHOICES, default=None,
                        help="ASR device: auto (CUDA, else CPU), cuda or cpu; "
                             "default: SOTTO_DEVICE, else auto")
    parser.add_argument("--language", default=None,
                        help="auto or a language code (en, he, …); default: the saved choice")
    parser.add_argument("--glossary", help="local UTF-8 text or JSON glossary")
    parser.add_argument("--idle-release", type=float, default=0.0,
                        help="no effect on Windows (fresh stream per capture); "
                             "kept for CLI parity with the Mac")
    parser.add_argument("--limit", type=int, default=20,
                        help="history: number of newest entries to list")
    parser.add_argument("--id", dest="entry_id", help="history-delete: entry id")
    parser.add_argument("--yes", action="store_true", help="history-clear: confirm")
    args = parser.parse_args(argv)
    try:
        device = win_asr.requested_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    saved = user_settings.load()

    if args.command == "doctor":
        doctor(args.profile, args.model, device, saved["speed"])
        return
    if args.command == "setup":
        try:
            raise SystemExit(setup(args.profile, args.model, saved["speed"]))
        except ValueError as exc:
            parser.error(str(exc))
    if args.command.startswith("history"):
        raise SystemExit(history_command(args.command, limit=args.limit,
                                         entry_id=args.entry_id, yes=args.yes))
    if args.tray:
        _gui_process()
    sys.stderr = _ConsoleLog()
    trigger = args.trigger or win_hotkey.supported(saved["trigger"])

    def alert(message: str) -> None:
        if args.tray:
            import win_ui
            win_ui.message_box("Sotto", message, error=True)

    instance = single_instance()
    if instance is None:
        log("✗ Sotto is already running")
        alert("Sotto is already running (see the tray icon).")
        raise SystemExit(0 if args.tray else 1)
    restart = False
    try:
        restart = run(trigger, args.profile, args.model, args.language, args.glossary,
                      args.idle_release, device, speed=saved["speed"], tray=args.tray,
                      trigger_locked=args.trigger is not None)
    except win_asr.SetupError as exc:
        log(f"✗ {exc}")
        alert(str(exc))
        raise SystemExit(1)
    except ValueError as exc:
        alert(str(exc))
        parser.error(str(exc))
    except KeyboardInterrupt:
        # Ctrl-C before "listening" (download or warmup): nothing is captured
        # or queued yet, so stopping here is a clean exit.  An interrupted
        # download resumes on the next run.
        log("stopped before listening")
    finally:
        release_instance(instance)
    if restart:
        spawn_restart()


if __name__ == "__main__":
    main()

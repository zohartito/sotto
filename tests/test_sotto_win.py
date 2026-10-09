"""sotto_win: config resolution, delivery wiring, tray actions, CLI, shutdown.

No model load, no microphone, no real hotkey, no keystrokes: ``run`` is driven
with fakes, subprocess checks stop before the model loads, and every one of
them runs against its own temporary state (settings, dictionary, History).
Each test process uses its own single-instance name, so a Sotto running on the
same desktop never collides with the tests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

if sys.platform == "win32":
    os.environ.setdefault("SOTTO_INSTANCE_KEY", f"sotto-tests-{os.getpid()}")
    import dictionary
    import settings as user_settings
    import sotto
    import sotto_win
    import win_asr

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == "win32", "Windows-only entry point")
class ResolveConfigTest(unittest.TestCase):
    def _resolve(self, profile, model, language, speed="accurate"):
        # The persisted language-mode file must not leak into the test.
        with mock.patch.object(sotto_win, "load_language_mode", return_value="auto"):
            return sotto_win.resolve_config(profile, model, language, speed)

    def test_auto_profile_maps_to_ct2_repo(self):
        config = self._resolve("auto", None, None)
        self.assertEqual(config.model_repo, win_asr.PROFILES["auto"])
        self.assertIsNone(config.language)

    def test_model_override_wins(self):
        config = self._resolve("auto", "org/some-ct2-repo@" + "a" * 40, None, "fast")
        self.assertEqual(config.model_repo, "org/some-ct2-repo@" + "a" * 40)

    def test_language_override_accepts_any_whisper_language(self):
        config = self._resolve("auto", None, "auto")
        self.assertIsNone(config.language)
        self.assertEqual(self._resolve("auto", None, "fr").language, "fr")  # any Whisper language
        with self.assertRaises(ValueError):
            self._resolve("auto", None, "klingon")

    def test_fast_speed_swaps_only_the_auto_profile_model(self):
        self.assertEqual(self._resolve("auto", None, None, "fast").model_repo, win_asr.FAST_REPO)

    def test_blank_model_rejected(self):
        with self.assertRaises(ValueError):
            self._resolve("auto", "   ", None)

    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):
            self._resolve("nope", None, None)


@unittest.skipUnless(sys.platform == "win32", "Windows-only entry point")
class DeliveryLogicTest(unittest.TestCase):
    def test_prefs_come_from_settings_with_the_shared_automatic_default(self):
        prefs = sotto_win.delivery_prefs(user_settings.DEFAULTS)
        self.assertEqual((prefs.insert_mode, prefs.spacing, prefs.languages),
                         ("paste", "smart", ("en",)))
        prefs = sotto_win.delivery_prefs(dict(user_settings.DEFAULTS, insert_mode="type",
                                              spacing="none", languages=["fr", "xx", "de"]))
        self.assertEqual((prefs.insert_mode, prefs.spacing, prefs.languages), ("type", "none", ("fr", "de")))

    def test_windows_spacing_matches_the_shared_rule_with_unknown_context(self):
        import win_inject
        for text in ("Hi", "Hi ", "", "Γεια", "end.\n"):
            for spacing in ("smart", "trailing", "none"):
                self.assertEqual(win_inject.spaced(text, spacing),
                                 sotto.compose_insertion(text, spacing, None), (text, spacing))

    def test_automatic_uses_the_macs_language_choice_rule(self):
        cases = [({"en": 0.6, "pt": 0.3, "fr": 0.1}, ("en", "pt")),
                 ({"fr": 0.5, "pt": 0.3, "en": 0.2}, ("en", "pt")),
                 ({"es": 0.86, "en": 0.11, "pt": 0.01}, ("en", "pt")),
                 ({"es": 0.49, "en": 0.11}, ("en", "pt")),
                 ({"es": 0.6, "en": 0.25}, ("en", "pt")),
                 ({"de": 0.4, "en": 0.3, "pt": 0.3}, ("pt", "en"))]
        for probabilities, allowed in cases:
            self.assertEqual(win_asr.choose_language(probabilities, allowed),
                             sotto.choose_language(probabilities, allowed), (probabilities, allowed))

    def test_detected_language_moves_into_preprocessing_like_the_mac(self):
        metadata, preprocessing = {"detected_language": "pt", "profile": "auto"}, {}
        sotto_win.move_detected_language(metadata, preprocessing)
        self.assertEqual((metadata, preprocessing), ({"profile": "auto"}, {"detected_language": "pt"}))

    def test_screen_keeps_back_empty_and_looping_output_and_salvages_a_prefix(self):
        self.assertEqual(sotto_win.screen_transcript("", 1.0, 1.0, {}),
                         ("[no speech detected]", "empty transcript"))
        text, reason = sotto_win.screen_transcript("loop " * 60, 1.0, 1.0, {})
        self.assertIsNotNone(reason)
        prefix = "This part of the dictation is real and should be kept intact."
        preprocessing: dict = {}
        with mock.patch.object(sotto_win, "log"):
            text, reason = sotto_win.screen_transcript(prefix + " difference" * 40, 12.0, 1.0,
                                                       preprocessing)
        self.assertEqual((text, reason), (prefix, None))
        self.assertIn("repetition_trimmed", preprocessing)


@unittest.skipUnless(sys.platform == "win32", "Windows-only entry point")
class SetupCommandTest(unittest.TestCase):
    def test_setup_fetches_the_pinned_model_and_vad_and_never_downloads_offline(self):
        import vad
        for offline in (False, True):
            with mock.patch.object(sotto_win, "offline_requested", return_value=offline), \
                    mock.patch.object(sotto_win, "load_language_mode", return_value="auto"), \
                    mock.patch.object(sotto_win.win_asr, "resolve_model_dir",
                                      return_value=Path("C:/models/small")) as resolve, \
                    mock.patch.object(vad, "install", return_value=Path("C:/data/models/x.onnx")) as install, \
                    mock.patch.object(sotto_win, "log") as log:
                self.assertEqual(sotto_win.setup("auto", None, "fast"), 0)
            self.assertEqual(resolve.call_args.args[0], win_asr.FAST_REPO)
            self.assertEqual(resolve.call_args.kwargs["offline"], offline)
            install.assert_called_once_with(allow_download=not offline)
            log.assert_any_call("✓ voice detection ready (Silero VAD v6.2) in C:\\data\\models")

    def test_a_vad_failure_is_advisory(self):
        import vad
        with mock.patch.object(sotto_win.win_asr, "resolve_model_dir", return_value=Path("C:/m")), \
                mock.patch.object(sotto_win, "load_language_mode", return_value="auto"), \
                mock.patch.object(vad, "install", side_effect=RuntimeError("no network")), \
                mock.patch.object(sotto_win, "log") as log:
            self.assertEqual(sotto_win.setup("auto", None), 0)
        self.assertTrue(any("voice detection not installed" in call.args[0] for call in log.call_args_list))


@unittest.skipUnless(sys.platform == "win32", "Windows-only entry point")
class ProcessTest(unittest.TestCase):
    def test_one_dictation_app_per_session(self):
        with mock.patch.dict(os.environ, {"SOTTO_INSTANCE_KEY": f"one-per-session-{os.getpid()}"}):
            first = sotto_win.single_instance()
            self.assertIsNotNone(first)
            try:
                self.assertIsNone(sotto_win.single_instance())
            finally:
                sotto_win.release_instance(first)
            again = sotto_win.single_instance()
            self.assertIsNotNone(again)
            sotto_win.release_instance(again)
        with mock.patch.dict(os.environ, {"SOTTO_INSTANCE_KEY": ""}):
            self.assertEqual(sotto_win.instance_name(), sotto_win.instance_name())
            self.assertTrue(sotto_win.instance_name().startswith("Local\\sotto-"))

    def test_restart_uses_the_same_arguments_without_a_console(self):
        with mock.patch.object(sys, "argv", ["C:/sotto/win_launch.py", "--data-dir", "D:/x"]):
            command = sotto_win.restart_command()
        pythonw = Path(sys.executable).with_name("pythonw.exe")
        self.assertEqual(Path(command[0]), pythonw if pythonw.is_file() else Path(sys.executable))
        self.assertEqual(command[1:], ["C:/sotto/win_launch.py", "--data-dir", "D:/x"])

    def test_restart_finds_a_script_started_by_a_relative_path(self):
        # [F36] `python sotto-copy\sotto_win.py --tray` from another folder: the
        # restart runs in the repository folder, so a relative path must not reach it.
        with tempfile.TemporaryDirectory(prefix="sotto-win-cwd-") as folder, \
                mock.patch.object(sys, "argv", ["sotto-copy\\sotto_win.py", "--tray"]):
            previous = os.getcwd()
            os.chdir(folder)
            try:
                command = sotto_win.restart_command()
            finally:
                os.chdir(previous)
        self.assertTrue(Path(command[1]).is_absolute(), command)
        self.assertEqual(Path(command[1]).resolve(), (Path(folder) / "sotto-copy" / "sotto_win.py").resolve())
        self.assertEqual(command[2:], ["--tray"])


@unittest.skipUnless(sys.platform == "win32", "Windows-only entry point")
class RestartTest(unittest.TestCase):
    def controller(self, busy):
        import queue
        jobs, deliveries = queue.Queue(), queue.Queue()
        if busy:
            jobs.put("live")
        capture = mock.Mock()
        capture.is_active.return_value = False
        return sotto_win.Controller(shutdown=sotto.ShutdownBoundary(), capture=capture,
                                    jobs=jobs, deliveries=deliveries, finishing=lambda: False,
                                    lifecycle={"restarting": False}), jobs

    def test_restart_waits_for_in_flight_work_and_quit_cancels_it(self):
        controller, jobs = self.controller(busy=True)
        with mock.patch.object(sotto_win, "log"):
            self.assertIn("after the current dictation", controller.restart())
            time.sleep(0.3)
            self.assertFalse(controller.shutdown.requested(), "a queued dictation is never dropped")
            jobs.get_nowait()
            jobs.task_done()
            deadline = time.monotonic() + 5
            while not controller.shutdown.requested() and time.monotonic() < deadline:
                time.sleep(0.05)
        self.assertTrue(controller.shutdown.requested())
        self.assertTrue(controller.restart_requested)
        idle, _jobs = self.controller(busy=False)
        with mock.patch.object(sotto_win, "log"):
            self.assertEqual(idle.restart(), "Restarting…")
        self.assertTrue(idle.shutdown.requested())
        waiting, _jobs = self.controller(busy=True)
        with mock.patch.object(sotto_win, "log"):
            waiting.restart()
            waiting.quit()
        self.assertTrue(waiting.shutdown.requested())
        self.assertFalse(waiting.restart_requested, "Quit wins over a pending restart")

    def test_restart_and_speed_change_never_abandon_a_slow_dictation(self):
        # [F30] A long dictation on the CPU outlasts the old 20 s drain deadline;
        # Restart and a Speed change must still wait for it (deadline shrunk here).
        for start in ("restart", "speed"):
            controller, jobs = self.controller(busy=True)
            controller.speed = "accurate"
            with mock.patch.object(sotto, "RESTART_DRAIN_DEADLINE_S", 0.2), \
                    mock.patch.object(sotto_win.user_settings, "save"), \
                    mock.patch.object(sotto_win, "log"):
                message = controller.restart() if start == "restart" else controller.set_speed("fast")
                self.assertIn("after the current dictation", message)
                time.sleep(0.8)
                self.assertFalse(controller.shutdown.requested(),
                                 f"{start}: a dictation still transcribing is never abandoned")
                jobs.get_nowait()
                jobs.task_done()
                deadline = time.monotonic() + 5
                while not controller.shutdown.requested() and time.monotonic() < deadline:
                    time.sleep(0.05)
            self.assertTrue(controller.shutdown.requested(), f"{start}: restarts once idle")
            self.assertTrue(controller.restart_requested)

    @staticmethod
    def virtual_clock(controller, on_sleep=lambda now: None) -> dict:
        """Drive the controller's drains on a virtual clock: every poll's
        sleep advances it at once, then runs ``on_sleep(now)``."""
        clock = {"now": 0.0}

        def sleep(seconds):
            clock["now"] += seconds
            on_sleep(clock["now"])

        controller.clock, controller.sleep = (lambda: clock["now"]), sleep
        return clock

    def test_restart_waits_minutes_for_a_slow_dictation(self):
        # A long dictation on the CPU can take minutes: it still finishes first.
        controller, jobs = self.controller(busy=True)
        slow_s = 4 * 60.0
        stopped_while_busy = []

        def on_sleep(now):
            stopped_while_busy.append(controller.shutdown.requested())
            if now >= slow_s and jobs.unfinished_tasks:
                jobs.get_nowait()
                jobs.task_done()

        clock, logs = self.virtual_clock(controller, on_sleep), []
        with mock.patch.object(sotto_win, "log", logs.append):
            self.assertIn("after the current dictation", controller.restart())
            self.assertTrue(controller.shutdown.event.wait(5), "restarts once the dictation is done")
        self.assertTrue(stopped_while_busy, "the drain polls on the controller's clock")
        self.assertFalse(any(stopped_while_busy), "a slow dictation is never abandoned")
        self.assertGreaterEqual(clock["now"], slow_s)
        self.assertIn("● restarting after the current dictation", logs)
        self.assertFalse([line for line in logs if "abandoned" in line], logs)

    def test_a_wedged_dictation_cannot_hold_a_restart_forever(self):
        # [F30 follow-up] A hung CUDA/CT2 call left Restart refusing every
        # capture until the user quit: past the cap, the restart goes ahead.
        controller, jobs = self.controller(busy=True)
        clock, logs = self.virtual_clock(controller), []
        with mock.patch.object(sotto_win, "log", logs.append):
            controller.restart()
            self.assertTrue(controller.shutdown.event.wait(5), "a wedged dictation still restarts")
        self.assertTrue(controller.restart_requested)
        self.assertEqual(jobs.unfinished_tasks, 1, "the dictation never finished")
        self.assertAlmostEqual(clock["now"], sotto_win.RESTART_DRAIN_CAP_S, delta=1.0)
        self.assertGreaterEqual(sotto_win.RESTART_DRAIN_CAP_S, 2 * 60, "minutes, not seconds")
        abandoned = [line for line in logs if "abandoned" in line]
        self.assertEqual(len(abandoned), 1, logs)
        self.assertIn("1 transcription job(s)", abandoned[0])


@unittest.skipUnless(sys.platform == "win32", "Windows-only entry point")
class ConsoleCloseHandlerTest(unittest.TestCase):
    def test_ctrl_break_requests_shutdown_and_handler_is_restored(self):
        previous = signal.getsignal(signal.SIGBREAK)
        boundary = sotto.ShutdownBoundary()
        with sotto_win.console_close_handler(boundary):
            handler = signal.getsignal(signal.SIGBREAK)
            handler(signal.SIGBREAK, None)
        self.assertTrue(boundary.requested())
        self.assertEqual(signal.getsignal(signal.SIGBREAK), previous)


@unittest.skipUnless(sys.platform == "win32", "Windows-only entry point")
class MainExitTest(unittest.TestCase):
    def test_ctrl_c_before_listening_is_a_clean_exit(self):
        # Download/warmup run before the event-only handlers are installed.
        with mock.patch.object(sotto_win, "run", side_effect=KeyboardInterrupt), \
                mock.patch.object(sotto_win, "_ConsoleLog", lambda: sys.stderr), \
                mock.patch.object(sotto_win, "log") as log:
            sotto_win.main([])  # returns normally: exit code 0
        log.assert_called_with("stopped before listening")

    def test_settings_trigger_and_speed_reach_run_and_the_flag_wins(self):
        saved = dict(user_settings.DEFAULTS, trigger="left-shift", speed="fast")
        for argv, trigger, locked in (([], "left-shift", False),
                                      (["--trigger", "left-ctrl"], "left-ctrl", True)):
            with mock.patch.object(sotto_win, "run", return_value=False) as run, \
                    mock.patch.object(sotto_win.user_settings, "load", return_value=saved), \
                    mock.patch.object(sotto_win, "_ConsoleLog", lambda: sys.stderr):
                sotto_win.main(argv)
            args, kwargs = run.call_args
            self.assertEqual((args[0], kwargs["speed"], kwargs["trigger_locked"], kwargs["tray"]),
                             (trigger, "fast", locked, False))

    def test_a_second_copy_exits_without_running(self):
        held = sotto_win.single_instance()
        try:
            with mock.patch.object(sotto_win, "run") as run, \
                    mock.patch.object(sotto_win, "_ConsoleLog", lambda: sys.stderr), \
                    mock.patch.object(sotto_win, "log") as log:
                with self.assertRaises(SystemExit) as ctx:
                    sotto_win.main([])
            self.assertEqual(ctx.exception.code, 1)
            run.assert_not_called()
            log.assert_any_call("✗ Sotto is already running")
        finally:
            sotto_win.release_instance(held)


def _app_fakes(rate, captures, replies, hooks, whisper_calls):
    import numpy as np

    class FakeWhisper:
        def __init__(self, model_dir, *, device, log):
            self.device, self.compute_type = "cpu", "int8"

        def transcribe(self, samples, **kwargs):
            whisper_calls.append((np.array(samples), kwargs))
            text = replies.pop(0) if len(whisper_calls) > 1 else ""
            if isinstance(text, Exception):
                raise text  # the model failed (CUDA out of memory, a driver error)
            result = {"text": text}
            if kwargs.get("language") is None and kwargs.get("allowed_languages"):
                result["language"] = tuple(kwargs["allowed_languages"])[0]
            return result

    class FakeCapture:
        native_rate, idle_release_s = rate, 0.0
        opened = closed = shutdowns = 0
        active = False

        def begin(self):
            FakeCapture.opened += 1
            FakeCapture.active = True
            return False

        def end(self):
            FakeCapture.closed += 1
            FakeCapture.active = False
            return captures.pop(0)

        def abort(self): FakeCapture.active = False
        def release_soon(self): pass
        def tick(self): pass
        def is_waking(self): return False
        def is_active(self): return FakeCapture.active

        def shutdown(self):
            FakeCapture.shutdowns += 1

    class FakeHook:
        def __init__(self, engine, *, trigger):
            self.engine, self.trigger = engine, trigger
            self.physically_down = self.modifiers_held = self.stopped = False
            hooks.append(self)

        def start(self): pass
        def alive(self): return True

        def set_trigger(self, trigger):
            self.trigger = trigger

        def stop(self):
            self.stopped = True

    return FakeWhisper, FakeCapture, FakeHook


@unittest.skipUnless(sys.platform == "win32", "Windows-only entry point")
class RunPipelineTest(unittest.TestCase):
    """``sotto_win.run`` end to end with fake key, microphone, model and injector."""

    def test_only_validated_text_is_delivered_with_dictionary_and_settings(self):
        import numpy as np
        from history import HistoryStore
        from learning import LearningStore
        import vad

        rate = 16_000
        voiced = (np.sin(np.arange(rate) / 3) * 0.2).astype(np.float32)
        # dead route (exact zeros), real text, empty output, a wall-to-wall loop
        captures = [np.zeros(rate, dtype=np.float32), voiced, voiced.copy(), voiced.copy()]
        replies = ["hello world", "", "loop " * 60]
        delivered, hooks, boundaries, logs, delivered_while_held, rows_at_delivery = [], [], [], [], [], []
        whisper_calls: list = []
        FakeWhisper, FakeCapture, FakeHook = _app_fakes(rate, captures, replies, hooks, whisper_calls)
        saved = dict(user_settings.DEFAULTS, insert_mode="type", spacing="none", languages=["pt", "en"])

        class RecordingBoundary(sotto.ShutdownBoundary):
            def __init__(self):
                super().__init__()
                boundaries.append(self)

        with tempfile.TemporaryDirectory(prefix="sotto-win-run-") as temporary:
            data = Path(temporary)
            rules_file = data / "dictionary.txt"
            # "loop" must never be rewritten: suspect rows are not delivered.
            rules_file.write_text("hello world => Hello, World\nloop => hoop\n", encoding="utf-8")

            def drive():
                deadline = time.monotonic() + 60
                while not hooks and time.monotonic() < deadline:
                    time.sleep(0.02)
                hook = hooks[0]
                for index in range(4):
                    hook.physically_down = hook.modifiers_held = True
                    hook.engine.pressed()
                    time.sleep(0.45)
                    hook.physically_down = False
                    hook.modifiers_held = index == 1  # keep a modifier held past the valid result
                    hook.engine.released()
                    # One capture at a time: wait for its row (and, for the
                    # valid one, its delivery) before the next press.
                    while len(HistoryStore(data).entries(10)) <= index and time.monotonic() < deadline:
                        time.sleep(0.02)
                    if index == 1:
                        time.sleep(0.5)
                        delivered_while_held.extend(delivered)
                        hook.modifiers_held = False
                        while not delivered and time.monotonic() < deadline:
                            time.sleep(0.02)
                boundaries[0].request()

            def fake_deliver(text, *, mode, spacing, log):
                rows_at_delivery.append([row["text"] for row in HistoryStore(data).entries(10)])
                delivered.append((text, mode, spacing))
                return mode

            with mock.patch.object(sotto_win.win_asr, "resolve_model_dir", return_value=data), \
                    mock.patch.object(sotto_win.win_asr, "LocalCT2Whisper", FakeWhisper), \
                    mock.patch.object(sotto_win.win_capture, "WinCapture", FakeCapture), \
                    mock.patch.object(sotto_win.win_hotkey, "TriggerHook", FakeHook), \
                    mock.patch.object(sotto_win.win_inject, "deliver", fake_deliver), \
                    mock.patch.object(sotto_win, "HistoryStore", lambda **kwargs: HistoryStore(data, **kwargs)), \
                    mock.patch.object(sotto_win, "LearningStore", lambda: LearningStore(data)), \
                    mock.patch.object(sotto_win, "load_language_mode", return_value="auto"), \
                    mock.patch.object(sotto_win.user_settings, "load", return_value=saved), \
                    mock.patch.object(dictionary, "DICTIONARY_PATH", rules_file), \
                    mock.patch.object(sotto_win, "log", logs.append), \
                    mock.patch.object(sotto, "ShutdownBoundary", RecordingBoundary), \
                    mock.patch.object(vad, "MODEL_PATH", data / "no-vad.onnx"):
                driver = threading.Thread(target=drive, daemon=True)
                driver.start()
                self.assertFalse(sotto_win.run("right-ctrl", "auto", None, None, None, 0.0, "cpu"))
                driver.join(5)

            entries = HistoryStore(data).entries(10)[::-1]
        self.assertEqual([entry["text"] for entry in entries],
                         [sotto.DEAD_ROUTE_TEXT, "Hello, World", "[no speech detected]",
                          ("loop " * 60).strip()])
        self.assertEqual(delivered, [("Hello, World", "type", "none")])
        self.assertIn("Hello, World", rows_at_delivery[0], "History is written before insertion")
        self.assertEqual(delivered_while_held, [], "nothing is inserted while a modifier is held")
        self.assertIn("  insert deferred — a modifier key is still held", logs)
        valid = entries[1]
        self.assertEqual(valid["preprocessing"]["dictionary"], {
            "rules": [{"heard": "hello world", "write": "Hello, World", "count": 1}],
            "asr_text": "hello world"})
        self.assertEqual(valid["preprocessing"]["detected_language"], "pt")
        latency = valid["latency"]["release_to_text_seconds"]
        self.assertGreaterEqual(latency, 0.0)
        self.assertLess(latency, 30.0)
        self.assertTrue(any(line.startswith(f"→ {latency:.2f}s after release (speech model ")
                            and line.endswith("· 12 chars") for line in logs), logs)
        for suspect in (entries[2], entries[3]):
            self.assertNotIn("dictionary", suspect["preprocessing"])
            self.assertIn("release_to_text_seconds", suspect["latency"])  # the Mac's measure
        self.assertNotIn("release_to_text_seconds", entries[0]["latency"])  # dead mic: no ASR
        # warmup + the three voiced captures; the all-zero capture never reached the model
        self.assertEqual(len(whisper_calls), 4)
        self.assertTrue(all(np.any(samples) for samples, _ in whisper_calls[1:]))
        self.assertTrue(all(kwargs["allowed_languages"] == ("pt", "en") for _, kwargs in whisper_calls))
        self.assertEqual((FakeCapture.opened, FakeCapture.closed), (4, 4))
        self.assertGreaterEqual(FakeCapture.shutdowns, 1)
        self.assertTrue(hooks[0].stopped)
        self.assertIn("✓ stopped", logs)
        self.assertEqual(sum(line.startswith("! not pasted") for line in logs), 3)
        self.assertEqual(sum("advisory VAD unavailable" in line for line in logs), 1)
        self.assertFalse([line for line in logs if "hello" in line.lower() or "loop loop" in line],
                         "logs never contain dictated text")

    def test_an_unreadable_history_does_not_stop_dictation(self):
        """F16b: a damaged history.jsonl used to raise before the hotkey was
        armed, at every login. Now dictation works and the file is untouched."""
        import numpy as np
        from history import HistoryStore
        from learning import LearningStore
        import vad

        rate = 16_000
        voiced = (np.sin(np.arange(rate) / 3) * 0.2).astype(np.float32)
        delivered, hooks, boundaries, logs, whisper_calls = [], [], [], [], []
        FakeWhisper, FakeCapture, FakeHook = _app_fakes(rate, [voiced], ["hello world"], hooks, whisper_calls)

        class RecordingBoundary(sotto.ShutdownBoundary):
            def __init__(self):
                super().__init__()
                boundaries.append(self)

        with tempfile.TemporaryDirectory(prefix="sotto-win-unreadable-") as temporary:
            data = Path(temporary)
            index = data / "history.jsonl"
            index.write_text('{"id": "abc123", "text": "kept"}\n{"id": "def456", "text": "cut of', encoding="utf-8")
            before = index.read_bytes()

            def drive():
                deadline = time.monotonic() + 60
                while not hooks and time.monotonic() < deadline:
                    time.sleep(0.02)
                hook = hooks[0]
                hook.physically_down = hook.modifiers_held = True
                hook.engine.pressed()
                time.sleep(0.45)
                hook.physically_down = hook.modifiers_held = False
                hook.engine.released()
                while not delivered and time.monotonic() < deadline:
                    time.sleep(0.02)
                boundaries[0].request()

            with mock.patch.object(sotto_win.win_asr, "resolve_model_dir", return_value=data), \
                    mock.patch.object(sotto_win.win_asr, "LocalCT2Whisper", FakeWhisper), \
                    mock.patch.object(sotto_win.win_capture, "WinCapture", FakeCapture), \
                    mock.patch.object(sotto_win.win_hotkey, "TriggerHook", FakeHook), \
                    mock.patch.object(sotto_win.win_inject, "deliver",
                                      lambda text, **kwargs: delivered.append(text) or kwargs["mode"]), \
                    mock.patch.object(sotto_win, "HistoryStore", lambda **kwargs: HistoryStore(data, **kwargs)), \
                    mock.patch.object(sotto_win, "LearningStore", lambda: LearningStore(data)), \
                    mock.patch.object(sotto_win, "load_language_mode", return_value="auto"), \
                    mock.patch.object(sotto_win.user_settings, "load", return_value=dict(user_settings.DEFAULTS)), \
                    mock.patch.object(dictionary, "DICTIONARY_PATH", data / "dictionary.txt"), \
                    mock.patch.object(sotto_win, "log", logs.append), \
                    mock.patch.object(sotto, "ShutdownBoundary", RecordingBoundary), \
                    mock.patch.object(vad, "MODEL_PATH", data / "no-vad.onnx"):
                driver = threading.Thread(target=drive, daemon=True)
                driver.start()
                self.assertFalse(sotto_win.run("right-ctrl", "auto", None, None, None, 0.0, "cpu"))
                driver.join(5)
            self.assertEqual(index.read_bytes(), before, "History is left exactly as it was")
        self.assertEqual(len(delivered), 1)
        self.assertIn("hello world", delivered[0].lower())
        self.assertEqual(sum(line.startswith("! History unreadable") for line in logs), 1, logs)


@unittest.skipUnless(sys.platform == "win32", "Windows-only entry point")
class ControllerTest(unittest.TestCase):
    """Tray actions against a running ``run`` (fakes for key, mic, model, clipboard)."""

    def test_retry_correct_suggest_settings_language_progress_and_clear(self):
        import numpy as np
        from history import HistoryStore
        from learning import LearningStore
        import vad

        rate = 16_000
        voiced = (np.sin(np.arange(rate) / 3) * 0.2).astype(np.float32)
        captures = [voiced]
        replies = ["whisper flow works", "whisper flow works in tel aviv"]
        hooks, boundaries, controllers, logs, copied, delivered = [], [], [], [], [], []
        whisper_calls: list = []
        FakeWhisper, FakeCapture, FakeHook = _app_fakes(rate, captures, replies, hooks, whisper_calls)
        results: dict = {}

        class RecordingBoundary(sotto.ShutdownBoundary):
            def __init__(self):
                super().__init__()
                boundaries.append(self)

        class RecordingController(sotto_win.Controller):
            def __init__(self, **parts):
                super().__init__(**parts)
                controllers.append(self)

        with tempfile.TemporaryDirectory(prefix="sotto-win-ctl-") as temporary:
            data = Path(temporary)
            rules_file = data / "dictionary.txt"
            dictionary.add(dictionary.parse("whisper flow => Wispr Flow"), rules_file)
            settings_file = data / "settings.json"
            language_file = data / "language-mode"
            real_save_language = sotto_win.save_language_mode
            real_load_settings, real_save_settings = user_settings.load, user_settings.save

            def wait_for(condition, what):
                deadline = time.monotonic() + 30
                while not condition():
                    if time.monotonic() > deadline:
                        raise AssertionError(f"timed out waiting for {what}")
                    time.sleep(0.02)

            def drive():
                try:
                    wait_for(lambda: controllers and hooks, "listening")
                    controller, hook = controllers[0], hooks[0]
                    hook.engine.pressed()
                    time.sleep(0.45)
                    hook.engine.released()
                    wait_for(lambda: delivered, "the dictation")
                    entry = HistoryStore(data).entries(1)[0]
                    # Retry: dictionary applies, the text is copied, never pasted.
                    controller.retry(entry["id"])
                    wait_for(lambda: copied, "the retry")
                    entry = results["after_retry"] = HistoryStore(data).get(entry["id"])
                    # Correct: saved through the shared path, rules only suggested.
                    suggestions = controller.correct(entry["id"], entry["text"],
                                                     "Wispr Flow works in Jaffa", entry["revision"])
                    results["suggestions"] = [(rule.heard, rule.write) for rule in suggestions]
                    results["rules_before_confirm"] = len(dictionary.load(rules_file))
                    with self.assertRaises(ValueError):  # stale revision
                        controller.correct(entry["id"], entry["text"], "x", entry["revision"])
                    results["added"] = controller.add_rules(suggestions)
                    # Settings: trigger applies live, speed needs a restart.
                    results["notes"] = controller.save_settings(
                        {"trigger": "left-shift", "speed": "fast", "launch_at_login": True})
                    results["trigger"] = hook.trigger
                    controller.set_language("fr")
                    results["language"] = (controller.language(), language_file.read_text("utf-8").strip())
                    results["progress"] = controller.progress()
                    controller.clear()
                    results["after_clear"] = HistoryStore(data).entries(10)
                finally:
                    boundaries[0].request()

            with mock.patch.object(sotto_win.win_asr, "resolve_model_dir", return_value=data), \
                    mock.patch.object(sotto_win.win_asr, "LocalCT2Whisper", FakeWhisper), \
                    mock.patch.object(sotto_win.win_capture, "WinCapture", FakeCapture), \
                    mock.patch.object(sotto_win.win_hotkey, "TriggerHook", FakeHook), \
                    mock.patch.object(sotto_win.win_inject, "deliver",
                                      lambda text, **kwargs: delivered.append(text)), \
                    mock.patch.object(sotto_win.win_inject, "copy_text", copied.append), \
                    mock.patch.object(sotto_win, "HistoryStore", lambda **kwargs: HistoryStore(data, **kwargs)), \
                    mock.patch.object(sotto_win, "LearningStore", lambda: LearningStore(data)), \
                    mock.patch.object(sotto_win, "Controller", RecordingController), \
                    mock.patch.object(sotto_win, "load_language_mode", return_value="auto"), \
                    mock.patch.object(sotto_win, "save_language_mode",
                                      lambda mode: real_save_language(mode, language_file)), \
                    mock.patch.object(sotto_win.user_settings, "load",  # save() calls load(path)
                                      lambda path=settings_file: real_load_settings(path)), \
                    mock.patch.object(sotto_win.user_settings, "save",
                                      lambda updates: real_save_settings(updates, settings_file)), \
                    mock.patch("win_startup.set_enabled") as set_enabled, \
                    mock.patch("win_startup.enabled", return_value=False), \
                    mock.patch.object(dictionary, "DICTIONARY_PATH", rules_file), \
                    mock.patch.object(sotto_win, "log", logs.append), \
                    mock.patch.object(sotto, "ShutdownBoundary", RecordingBoundary), \
                    mock.patch.object(vad, "MODEL_PATH", data / "no-vad.onnx"):
                driver = threading.Thread(target=drive, daemon=True)
                driver.start()
                sotto_win.run("right-ctrl", "auto", None, None, None, 0.0, "cpu")
                driver.join(10)
            saved = real_load_settings(settings_file)

        self.assertEqual(delivered, ["Wispr Flow works"])  # the retry never pastes
        retried = results["after_retry"]
        self.assertEqual(copied, ["Wispr Flow works in tel aviv"])
        self.assertEqual(retried["text"], "Wispr Flow works in tel aviv")
        self.assertEqual(retried["attempts"][-1]["preprocessing"]["dictionary"]["asr_text"],
                         "whisper flow works in tel aviv")
        self.assertEqual(results["suggestions"], [("tel aviv", "Jaffa")])
        self.assertEqual(results["rules_before_confirm"], 1, "no rule without confirmation")
        self.assertEqual(results["added"], 1)
        self.assertEqual(results["trigger"], "left-shift")
        self.assertIn("Speed applies after restart.", results["notes"])
        set_enabled.assert_called_once()
        self.assertTrue(set_enabled.call_args.args[0])
        self.assertIn("win_launch.py", set_enabled.call_args.args[1])
        self.assertEqual((saved["trigger"], saved["speed"], saved["launch_at_login"]),
                         ("left-shift", "fast", True))
        self.assertEqual(results["language"], ("fr", "fr"))
        # The shared progress summary (the Mac's wording), from local History.
        self.assertTrue(results["progress"][0].startswith("All time: "), results["progress"])
        self.assertTrue(results["progress"][1].startswith("Last 7 days: 1 dictation · ready "),
                        results["progress"])
        self.assertIn("Dictionary: 2 rules · fixed 1 word this period", results["progress"])
        self.assertIn("Corrections saved: 1 · in learning set: 0", results["progress"])
        self.assertEqual(results["after_clear"], [])
        self.assertFalse([line for line in logs if any(word in line.lower()
                                                       for word in ("flow", "aviv", "jaffa"))],
                         "logs never contain dictated text or rules")


@unittest.skipUnless(sys.platform == "win32", "Windows-only entry point")
class LifecycleTest(unittest.TestCase):
    """Tray finish copies; a restart drains the dictation in flight and takes no new one."""

    def test_menu_finish_copies_and_restart_drains_without_new_captures(self):
        import numpy as np
        from history import HistoryStore
        from learning import LearningStore
        import vad

        rate = 16_000
        voiced = (np.sin(np.arange(rate) / 3) * 0.2).astype(np.float32)
        captures = [voiced, voiced.copy()]
        replies = ["finished from the menu", "dictated with the key"]
        hooks, boundaries, controllers, logs, copied, delivered = [], [], [], [], [], []
        whisper_calls: list = []
        FakeWhisper, FakeCapture, FakeHook = _app_fakes(rate, captures, replies, hooks, whisper_calls)
        results: dict = {}

        class RecordingBoundary(sotto.ShutdownBoundary):
            def __init__(self):
                super().__init__()
                boundaries.append(self)

        class RecordingController(sotto_win.Controller):
            def __init__(self, **parts):
                super().__init__(**parts)
                controllers.append(self)

        def wait_for(condition, what):
            deadline = time.monotonic() + 30
            while not condition():
                if time.monotonic() > deadline:
                    raise AssertionError(f"timed out waiting for {what}")
                time.sleep(0.02)

        with tempfile.TemporaryDirectory(prefix="sotto-win-life-") as temporary:
            data = Path(temporary)

            def drive():
                try:
                    wait_for(lambda: controllers and hooks, "listening")
                    controller, hook = controllers[0], hooks[0]
                    controller.start_now()
                    time.sleep(0.3)
                    controller.finish_now()
                    wait_for(lambda: copied, "the menu-finished dictation")
                    hook.modifiers_held = True  # keeps the next insertion (and the drain) waiting
                    hook.engine.pressed()
                    time.sleep(0.45)
                    results["restart"] = controller.restart()
                    hook.engine.released()
                    wait_for(lambda: any("insert deferred" in line for line in logs), "the deferred insert")
                    hook.engine.pressed()  # during the drain: refused, the mic stays closed
                    hook.engine.released()
                    time.sleep(0.6)
                    results["stopped_while_busy"] = boundaries[0].requested()
                    results["opened"] = FakeCapture.opened
                    hook.modifiers_held = False
                except BaseException as exc:
                    results["error"] = exc
                    boundaries[0].request()

            with mock.patch.object(sotto_win.win_asr, "resolve_model_dir", return_value=data), \
                    mock.patch.object(sotto_win.win_asr, "LocalCT2Whisper", FakeWhisper), \
                    mock.patch.object(sotto_win.win_capture, "WinCapture", FakeCapture), \
                    mock.patch.object(sotto_win.win_hotkey, "TriggerHook", FakeHook), \
                    mock.patch.object(sotto_win.win_inject, "deliver",
                                      lambda text, **kwargs: delivered.append(text)), \
                    mock.patch.object(sotto_win.win_inject, "copy_text", copied.append), \
                    mock.patch.object(sotto_win, "HistoryStore", lambda **kwargs: HistoryStore(data, **kwargs)), \
                    mock.patch.object(sotto_win, "LearningStore", lambda: LearningStore(data)), \
                    mock.patch.object(sotto_win, "Controller", RecordingController), \
                    mock.patch.object(sotto_win, "load_language_mode", return_value="auto"), \
                    mock.patch.object(sotto_win.user_settings, "load", return_value=dict(user_settings.DEFAULTS)), \
                    mock.patch.object(dictionary, "DICTIONARY_PATH", data / "dictionary.txt"), \
                    mock.patch.object(sotto_win, "log", logs.append), \
                    mock.patch.object(sotto, "ShutdownBoundary", RecordingBoundary), \
                    mock.patch.object(vad, "MODEL_PATH", data / "no-vad.onnx"):
                driver = threading.Thread(target=drive, daemon=True)
                driver.start()
                restarted = sotto_win.run("right-ctrl", "auto", None, None, None, 0.0, "cpu")
                driver.join(10)
            texts = [entry["text"] for entry in HistoryStore(data).entries(10)][::-1]

        self.assertNotIn("error", results, results)
        self.assertEqual(copied, ["finished from the menu"])  # copied, never inserted
        self.assertEqual(delivered, ["dictated with the key"])  # drained before the restart
        self.assertEqual(texts, ["finished from the menu", "dictated with the key"])
        self.assertIn("after the current dictation", results["restart"])
        self.assertFalse(results["stopped_while_busy"], "a restart never drops a dictation in flight")
        self.assertEqual(results["opened"], 2, "no new capture during the drain")
        self.assertIn("○ restarting — this press is ignored", logs)
        self.assertTrue(restarted)

    def test_tray_finish_queued_behind_a_callback_in_flight_is_still_copied(self):
        """Gesture actions run in decision order on whichever thread is
        draining (F6). A tray finish decided while the key thread is still
        inside on_start runs later, on that thread — it must still copy."""
        import numpy as np
        from history import HistoryStore
        from learning import LearningStore
        import vad

        rate = 16_000
        voiced = (np.sin(np.arange(rate) / 3) * 0.2).astype(np.float32)
        hooks, boundaries, controllers, logs, copied, delivered = [], [], [], [], [], []
        whisper_calls: list = []
        FakeWhisper, FakeCapture, FakeHook = _app_fakes(
            rate, [voiced], ["finished from the tray"], hooks, whisper_calls)
        begin_entered, resume_begin = threading.Event(), threading.Event()
        self.addCleanup(resume_begin.set)
        results: dict = {}

        class SlowStartCapture(FakeCapture):
            def begin(self):
                cold = super().begin()
                begin_entered.set()
                resume_begin.wait(10)  # the mic is slow to open: on_start is in flight
                return cold

        class RecordingBoundary(sotto.ShutdownBoundary):
            def __init__(self):
                super().__init__()
                boundaries.append(self)

        class RecordingController(sotto_win.Controller):
            def __init__(self, **parts):
                super().__init__(**parts)
                controllers.append(self)

        def wait_for(condition, what):
            deadline = time.monotonic() + 30
            while not condition():
                if time.monotonic() > deadline:
                    raise AssertionError(f"timed out waiting for {what}")
                time.sleep(0.02)

        with tempfile.TemporaryDirectory(prefix="sotto-win-tray-") as temporary:
            data = Path(temporary)

            def drive():
                try:
                    wait_for(lambda: controllers and hooks, "listening")
                    controller, hook = controllers[0], hooks[0]
                    hook.physically_down = True
                    key_thread = threading.Thread(target=hook.engine.pressed, daemon=True)
                    key_thread.start()  # the key thread becomes the drainer ...
                    if not begin_entered.wait(10):
                        raise AssertionError("on_start never reached the capture")
                    controller.finish_now()  # ... so the tray's finish queues behind it
                    resume_begin.set()
                    key_thread.join(10)
                    hook.physically_down = False
                    wait_for(lambda: copied or delivered, "the tray-finished dictation")
                except BaseException as exc:
                    results["error"] = exc
                finally:
                    boundaries[0].request()

            with mock.patch.object(sotto_win.win_asr, "resolve_model_dir", return_value=data), \
                    mock.patch.object(sotto_win.win_asr, "LocalCT2Whisper", FakeWhisper), \
                    mock.patch.object(sotto_win.win_capture, "WinCapture", SlowStartCapture), \
                    mock.patch.object(sotto_win.win_hotkey, "TriggerHook", FakeHook), \
                    mock.patch.object(sotto_win.win_inject, "deliver",
                                      lambda text, **kwargs: delivered.append(text)), \
                    mock.patch.object(sotto_win.win_inject, "copy_text", copied.append), \
                    mock.patch.object(sotto_win, "HistoryStore", lambda **kwargs: HistoryStore(data, **kwargs)), \
                    mock.patch.object(sotto_win, "LearningStore", lambda: LearningStore(data)), \
                    mock.patch.object(sotto_win, "Controller", RecordingController), \
                    mock.patch.object(sotto_win, "load_language_mode", return_value="auto"), \
                    mock.patch.object(sotto_win.user_settings, "load", return_value=dict(user_settings.DEFAULTS)), \
                    mock.patch.object(dictionary, "DICTIONARY_PATH", data / "dictionary.txt"), \
                    mock.patch.object(sotto_win, "log", logs.append), \
                    mock.patch.object(sotto, "ShutdownBoundary", RecordingBoundary), \
                    mock.patch.object(vad, "MODEL_PATH", data / "no-vad.onnx"):
                driver = threading.Thread(target=drive, daemon=True)
                driver.start()
                self.assertFalse(sotto_win.run("right-ctrl", "auto", None, None, None, 0.0, "cpu"))
                driver.join(10)

        self.assertNotIn("error", results, results)
        self.assertEqual(delivered, [], "a tray-finished dictation was inserted into the focused app")
        self.assertEqual(copied, ["finished from the tray"])


@unittest.skipUnless(sys.platform == "win32", "Windows-only entry point")
class IsolatedCliTest(unittest.TestCase):
    """Real ``sotto_win.py`` processes against a private temporary state root."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="sotto-win-cli-")
        root = Path(self._tmp.name)
        self.data = root / "data"
        self.env = dict(os.environ, SOTTO_DATA_DIR=str(self.data), SOTTO_HF_HOME=str(root / "hf"),
                        SOTTO_INSTANCE_KEY=f"cli-{root.name}")
        for name in ("SOTTO_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "SOTTO_DEVICE",
                     "PYTHONIOENCODING", "PYTHONUTF8"):
            self.env.pop(name, None)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, *args, extra_env=None, input_code=None, script="sotto_win.py"):
        env = dict(self.env, **(extra_env or {}))
        command = [sys.executable, "-c", input_code] if input_code else [sys.executable, script, *args]
        return subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=120)

    def test_offline_missing_model_prints_a_message_not_a_traceback(self):
        result = self._run("run", extra_env={"SOTTO_OFFLINE": "1"})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn("is not in", result.stderr)
        self.assertIn("offline mode is on", result.stderr)
        self.assertNotIn("listening", result.stderr)
        self.assertFalse((Path(self.env["SOTTO_HF_HOME"]) / "hub").exists())

    def test_offline_setup_with_an_empty_cache_downloads_nothing(self):
        result = self._run("setup", extra_env={"SOTTO_OFFLINE": "1"})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("offline mode is on", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((Path(self.env["SOTTO_HF_HOME"]) / "hub").exists())
        self.assertFalse((self.data / "models").exists())

    def test_unpinned_model_is_a_usage_error(self):
        result = self._run("run", "--model", "org/unpinned", extra_env={"SOTTO_OFFLINE": "1"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("not pinned", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_launcher_applies_the_data_and_model_folders(self):
        other = Path(self._tmp.name) / "launched"
        result = self._run("--data-dir", str(other), "doctor", script="win_launch.py",
                           extra_env={"SOTTO_OFFLINE": "1"})
        self.assertIn(f"data: {other}", result.stderr)
        self.assertIn(f"model cache: {other / 'huggingface'}", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_history_list_delete_and_clear_remove_rows_and_audio(self):
        seed = (
            "import json, numpy as np\n"
            "from audio_codec import prepare_canonical\n"
            "from history import HistoryStore\n"
            "from learning import LearningCoordinator, LearningStore\n"
            "store = HistoryStore(); coordinator = LearningCoordinator(store, LearningStore())\n"
            "ids = []\n"
            "for i in range(3):\n"
            "    samples = (np.sin(np.arange(16000) / 7) * 0.1).astype(np.float32)\n"
            "    text = '\\u3053\\u3093\\u306b\\u3061\\u306f' if i == 2 else f'entry {i}'\n"
            "    row = coordinator.append_live(text, prepare_canonical(samples), 1.0, 'm', ts=1.0 + i,\n"
            "        raw_samples=samples, raw_sample_rate=16000, provenance='live', adaptive=False)\n"
            "    ids.append(row['id'])\n"
            "print(json.dumps(ids))\n")
        seeded = self._run(input_code=seed)
        self.assertEqual(seeded.returncode, 0, seeded.stderr)
        ids = json.loads(seeded.stdout)
        audio = sorted(p.name for p in (self.data / "audio").iterdir())
        self.assertEqual(len(audio), 3)

        listed = self._run("history")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertEqual([line.split()[0] for line in listed.stdout.splitlines()], ids[::-1])
        self.assertIn("こんにちは", listed.stdout)  # Japanese, UTF-8 even through a pipe

        deleted = self._run("history-delete", "--id", ids[0])
        self.assertEqual(deleted.returncode, 0, deleted.stderr)
        self.assertEqual(self._run("history-delete", "--id", ids[0]).returncode, 1)
        self.assertEqual(len(list((self.data / "audio").iterdir())), 2)
        self.assertEqual(len(list((self.data / "audio-raw").iterdir())), 2)

        refused = self._run("history-clear")
        self.assertEqual(refused.returncode, 2)
        self.assertEqual(len(list((self.data / "audio").iterdir())), 2)
        cleared = self._run("history-clear", "--yes")
        self.assertEqual(cleared.returncode, 0, cleared.stderr)
        self.assertEqual(list((self.data / "audio").iterdir()), [])
        self.assertEqual(list((self.data / "audio-raw").iterdir()), [])
        self.assertEqual(self._run("history").stdout, "")


def _patched_run(stack, data, fakes, *, logs, boundaries, controllers, settings=None):
    """Enter the patches every ``run()`` test needs into ``stack``: fake model,
    microphone and hook, History in ``data``, and recorded logs, shutdown
    boundaries and controllers."""
    from history import HistoryStore
    from learning import LearningStore
    import vad

    FakeWhisper, FakeCapture, FakeHook = fakes

    class RecordingBoundary(sotto.ShutdownBoundary):
        def __init__(self):
            super().__init__()
            boundaries.append(self)

    class RecordingController(sotto_win.Controller):
        def __init__(self, **parts):
            super().__init__(**parts)
            controllers.append(self)

    for patch in (
            mock.patch.object(sotto_win.win_asr, "resolve_model_dir", return_value=data),
            mock.patch.object(sotto_win.win_asr, "LocalCT2Whisper", FakeWhisper),
            mock.patch.object(sotto_win.win_capture, "WinCapture", FakeCapture),
            mock.patch.object(sotto_win.win_hotkey, "TriggerHook", FakeHook),
            mock.patch.object(sotto_win, "HistoryStore", lambda **kwargs: HistoryStore(data, **kwargs)),
            mock.patch.object(sotto_win, "LearningStore", lambda: LearningStore(data)),
            mock.patch.object(sotto_win, "Controller", RecordingController),
            mock.patch.object(sotto_win, "load_language_mode", return_value="auto"),
            mock.patch.object(sotto_win.user_settings, "load",
                              return_value=dict(settings or user_settings.DEFAULTS)),
            mock.patch.object(dictionary, "DICTIONARY_PATH", data / "dictionary.txt"),
            mock.patch.object(sotto_win, "log", logs.append),
            mock.patch.object(sotto, "ShutdownBoundary", RecordingBoundary),
            mock.patch.object(vad, "MODEL_PATH", data / "no-vad.onnx")):
        stack.enter_context(patch)


def _wait_for(condition, what, timeout=30.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.02)


@unittest.skipUnless(sys.platform == "win32", "Windows-only entry point")
class FailureAndTeardownTest(unittest.TestCase):
    """A failing model, a stopping app and a queue of held insertions."""

    def test_a_failed_transcription_keeps_its_recording_for_retry_and_says_so(self):
        # [F4] A runtime model error (CUDA out of memory) used to lose the
        # dictation: no History row, no Retry, no message.
        import contextlib
        import numpy as np
        from history import HistoryStore

        rate = 16_000
        voiced = (np.sin(np.arange(rate) / 3) * 0.2).astype(np.float32)
        replies = [RuntimeError("CUDA failed with error out of memory"), "recovered words"]
        hooks, boundaries, controllers, logs, copied, delivered, notes = [], [], [], [], [], [], []
        whisper_calls: list = []
        fakes = _app_fakes(rate, [voiced], replies, hooks, whisper_calls)
        results: dict = {}

        class FakeTray:
            def __init__(self, *, quit, log):
                pass

            def start(self): pass
            def stop(self): pass
            def set_phase(self, phase): pass
            def set_state(self, state): pass
            def attach(self, controller): pass

            def notify(self, text):
                notes.append(text)

        with tempfile.TemporaryDirectory(prefix="sotto-win-fail-") as temporary:
            data = Path(temporary)

            def drive():
                try:
                    _wait_for(lambda: controllers and hooks, "listening")
                    controller, hook = controllers[0], hooks[0]
                    hook.engine.pressed()
                    time.sleep(0.45)
                    hook.engine.released()
                    _wait_for(lambda: HistoryStore(data).entries(1), "the History row")
                    _wait_for(lambda: not controller.busy(), "the worker")
                    entry = results["failed"] = HistoryStore(data).entries(1)[0]
                    results["notes_after_failure"] = list(notes)
                    controller.retry(entry["id"])
                    _wait_for(lambda: copied, "the retry")
                    results["retried"] = HistoryStore(data).get(entry["id"])
                except BaseException as exc:
                    results["error"] = exc
                finally:
                    boundaries[0].request()

            with contextlib.ExitStack() as stack:
                _patched_run(stack, data, fakes, logs=logs, boundaries=boundaries, controllers=controllers)
                stack.enter_context(mock.patch("win_ui.TrayApp", FakeTray))
                stack.enter_context(mock.patch.object(
                    sotto_win.win_inject, "deliver", lambda text, **kwargs: delivered.append(text)))
                stack.enter_context(mock.patch.object(sotto_win.win_inject, "copy_text", copied.append))
                driver = threading.Thread(target=drive, daemon=True)
                driver.start()
                sotto_win.run("right-ctrl", "auto", None, None, None, 0.0, "cpu", tray=True)
                driver.join(10)

        self.assertNotIn("error", results, results)
        failed = results["failed"]
        self.assertEqual(failed["text"], "[transcription failed]")
        self.assertEqual(failed["attempts"][-1]["provenance"], "live_suspect")
        self.assertEqual(failed["attempts"][-1]["preprocessing"]["outcome"], "suspect")
        self.assertEqual(delivered, [], "a failed transcription inserts nothing")
        self.assertTrue(any("Retry" in note for note in results["notes_after_failure"]), notes)
        self.assertTrue(any(line.startswith("! transcription failed: CUDA failed") for line in logs), logs)
        # Retry recovers the dictation from the kept audio.
        self.assertEqual(results["retried"]["text"], "recovered words")
        self.assertEqual(copied, ["recovered words"])
        self.assertEqual(len(whisper_calls[1][0]), len(whisper_calls[2][0]))
        self.assertTrue(np.allclose(whisper_calls[1][0], whisper_calls[2][0], atol=1e-4),
                        "Retry transcribes the audio the failed attempt had")

    def test_the_keyboard_hook_is_never_started_again_after_shutdown(self):
        # [F35] The resync poller slept through the shutdown request and then
        # revived the stopped hook behind the teardown.
        import contextlib

        hooks, boundaries, controllers, logs = [], [], [], []
        FakeWhisper, FakeCapture, FakeHook = _app_fakes(16_000, [], [], hooks, [])

        class CountingHook(FakeHook):
            def __init__(self, engine, *, trigger):
                super().__init__(engine, trigger=trigger)
                self.starts = 0

            def start(self):
                self.starts += 1

            def alive(self):
                return not self.stopped

        with tempfile.TemporaryDirectory(prefix="sotto-win-stop-") as temporary:
            data = Path(temporary)

            def drive():
                _wait_for(lambda: controllers and hooks, "listening")
                time.sleep(1.5)  # the poller is mid-sleep, as it nearly always is
                controllers[0].quit()

            with contextlib.ExitStack() as stack:
                _patched_run(stack, data, (FakeWhisper, FakeCapture, CountingHook), logs=logs,
                             boundaries=boundaries, controllers=controllers)
                driver = threading.Thread(target=drive, daemon=True)
                driver.start()
                sotto_win.run("right-ctrl", "auto", None, None, None, 0.0, "cpu")
                driver.join(10)
                time.sleep(1.5)  # longer than one poll after the hook stopped

        self.assertTrue(hooks[0].stopped)
        self.assertEqual(hooks[0].starts, 1, logs)
        self.assertNotIn("! keyboard listener died — restarting it", logs)

    def test_a_shutdown_during_the_poll_never_revives_the_stopped_hook(self):
        # [F35] The poller woke from its wait, found the listener dead, and
        # started it again although teardown had stopped it in between.
        import contextlib

        hooks, boundaries, controllers, logs = [], [], [], []
        FakeWhisper, FakeCapture, FakeHook = _app_fakes(16_000, [], [], hooks, [])
        polled = threading.Event()

        class DyingHook(FakeHook):
            """The listener dies while Quit arrives: the shutdown and the
            teardown's stop both land between the poller's wait and its start."""

            def __init__(self, engine, *, trigger):
                super().__init__(engine, trigger=trigger)
                self.starts, self.checks = 0, 0
                self.stop_event = threading.Event()
                snapshot = engine.snapshot

                def snapshot_after_the_revive_check():
                    if self.checks:
                        polled.set()
                    return snapshot()

                engine.snapshot = snapshot_after_the_revive_check

            def start(self):
                self.starts += 1

            def stop(self):
                super().stop()
                self.stop_event.set()

            def alive(self):
                self.checks += 1
                boundaries[0].request()
                self.stop_event.wait(10)
                return False

        with tempfile.TemporaryDirectory(prefix="sotto-win-poll-") as temporary:
            data = Path(temporary)
            with contextlib.ExitStack() as stack:
                _patched_run(stack, data, (FakeWhisper, FakeCapture, DyingHook), logs=logs,
                             boundaries=boundaries, controllers=controllers)
                sotto_win.run("right-ctrl", "auto", None, None, None, 0.0, "cpu")
                self.assertTrue(polled.wait(10), "the poller finished the poll it was in")

        self.assertTrue(hooks[0].stopped)
        self.assertEqual(hooks[0].checks, 1)
        self.assertEqual(hooks[0].starts, 1, logs)
        self.assertNotIn("! keyboard listener died — restarting it", logs)

    def test_a_queued_insertion_never_outlives_its_own_wait(self):
        # [F46] Each queued text got a fresh modifier wait after the ones ahead
        # of it used theirs up, so dictation N could land N waits late.
        import contextlib
        import numpy as np
        from history import HistoryStore

        rate = 16_000
        voiced = (np.sin(np.arange(rate) / 3) * 0.2).astype(np.float32)
        hooks, boundaries, controllers, logs, delivered = [], [], [], [], []
        fakes = _app_fakes(rate, [voiced, voiced.copy()], ["first words", "second words"], hooks, [])
        ready: list[float] = []
        wait_s = 2.0

        with tempfile.TemporaryDirectory(prefix="sotto-win-held-") as temporary:
            data = Path(temporary)

            def drive():
                try:
                    _wait_for(lambda: controllers and hooks, "listening")
                    hook = hooks[0]
                    hook.modifiers_held = True  # e.g. a stuck Shift: nothing may be inserted
                    for index in range(2):
                        hook.engine.pressed()
                        time.sleep(0.45)
                        hook.engine.released()
                        _wait_for(lambda index=index: len(HistoryStore(data).entries(10)) > index, "the History row")
                        ready.append(time.monotonic())
                    # Past the second text's own wait (with a margin), but well
                    # before a second full wait after the first one expired.
                    time.sleep(max(0.0, ready[1] + wait_s + 0.4 - time.monotonic()))
                    hook.modifiers_held = False
                    time.sleep(0.6)
                finally:
                    boundaries[0].request()

            with contextlib.ExitStack() as stack:
                _patched_run(stack, data, fakes, logs=logs, boundaries=boundaries, controllers=controllers)
                stack.enter_context(mock.patch.object(sotto_win, "INSERT_WAIT_S", wait_s))
                stack.enter_context(mock.patch.object(
                    sotto_win.win_inject, "deliver", lambda text, **kwargs: delivered.append(text)))
                driver = threading.Thread(target=drive, daemon=True)
                driver.start()
                sotto_win.run("right-ctrl", "auto", None, None, None, 0.0, "cpu")
                driver.join(10)
            texts = [entry["text"] for entry in HistoryStore(data).entries(10)][::-1]

        self.assertLess(ready[1] - ready[0], wait_s - 0.5, "the test needs both texts ready close together")
        self.assertEqual(delivered, [], "no text is inserted after its own wait ran out")
        self.assertEqual(sum(line.startswith("! not inserted (a modifier key held") for line in logs), 2, logs)
        self.assertEqual(texts, ["first words", "second words"], "both stay in History")


@unittest.skipUnless(sys.platform == "win32", "Windows-only entry point")
class ConsoleLogTest(unittest.TestCase):
    def test_a_log_line_never_waits_for_the_disk(self):
        # [F32] Gesture callbacks log on the keyboard-hook thread; a slow disk
        # or console must not hold that thread (Windows drops slow hooks).
        written: list[str] = []

        class SlowFile:
            def write(self, text):
                time.sleep(0.3)
                written.append(text)

            def flush(self):
                pass

        with tempfile.TemporaryDirectory(prefix="sotto-win-log-") as temporary, \
                mock.patch.object(sotto_win, "DATA_DIR", Path(temporary)), \
                mock.patch.object(sys, "stderr", None):
            sink = sotto_win._ConsoleLog()
            sink._file.close()
            sink._file = SlowFile()
            started = time.monotonic()
            print("● recording", file=sink, flush=True)
            print("○ 0.98s captured", file=sink, flush=True)
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 0.1, "logging waited for the file")
            sink.drain(timeout=5)
        self.assertEqual("".join(written), "● recording\n○ 0.98s captured\n")
        print("after drain", file=sink)  # written directly once the writer has stopped
        self.assertTrue("".join(written).endswith("after drain\n"), written)


if __name__ == "__main__":
    unittest.main()

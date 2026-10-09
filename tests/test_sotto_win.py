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


if __name__ == "__main__":
    unittest.main()

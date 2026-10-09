"""Exercise the real Sotto queue/history/delivery path without OS side effects."""
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import history
import learning
import sotto
from speech_config import resolve_speech_config
from tests.test_nemotron import FakeRuntime


SETTLE_S = 1.5  # long enough for a transcription the fences failed to stop to land


@unittest.skipUnless(sys.platform == 'darwin', 'Nemotron pipeline drives the AppKit runtime; macOS-only')
class PipelineTests(unittest.TestCase):
    def _run_one_dictation(self, directory: Path, make_store, make_learning=None, drive=None):
        """sotto.run with one streamed dictation, then Quit; History comes from
        ``make_store(**kwargs)``, the learning store from ``make_learning(**kwargs)``
        (default: one opened on ``directory``). ``drive(app)`` replaces the
        dictation-then-Quit loop; ``app`` holds the menu ``callbacks``, the
        ``captures``, ``boundaries`` and gesture ``engines`` run() built, the
        ``status`` UI mock, the ``delivered`` event set by a paste, and
        ``feed()``, which streams the test audio into the capture. Returns
        (status UI mock, pasted, captures, runtime)."""
        callbacks, captures, pasted = {}, [], []
        boundaries, engines = [], []
        delivered = threading.Event()
        runtime = FakeRuntime()
        runtime.close = MagicMock()
        status = MagicMock()
        status.set_history_callbacks.side_effect = callbacks.update
        audio = np.sin(np.arange(32000, dtype=np.float32) / 37) * .002

        class Capture(sotto.CaptureService):
            def __init__(self):
                super().__init__()
                captures.append(self)

            def _start_engine(self):
                self._starting = False

            def tick(self):
                pass

        class Pointer:
            def __init__(self, samples):
                self.samples = samples

            def as_buffer(self, n):
                return memoryview(self.samples)

        class Buffer:
            def __init__(self, samples):
                self.samples = samples

            def frameLength(self):
                return len(self.samples)

            def floatChannelData(self):
                return [Pointer(self.samples)]

            def format(self):
                return self

            def sampleRate(self):
                return 16000

        class RecordingBoundary(sotto.ShutdownBoundary):
            def __init__(self):
                super().__init__()
                boundaries.append(self)

        class RecordingEngine(sotto.GestureEngine):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                engines.append(self)

        def inject(text, **_options):
            pasted.append(text)
            delivered.set()

        def feed():
            for start in range(0, len(audio), 4096):
                when = MagicMock()
                when.sampleTime.return_value = start
                captures[0]._tap(Buffer(audio[start:start + 4096]), when)

        def run_loop():
            if drive is not None:
                drive({"callbacks": callbacks, "captures": captures, "boundaries": boundaries,
                       "engines": engines, "status": status, "delivered": delivered, "feed": feed})
                return
            callbacks["start_now"]()
            feed()
            self.assertEqual(pasted, [])
            callbacks["finish_now"]()
            self.assertTrue(delivered.wait(5))
            callbacks["quit"]()

        with ExitStack() as stack:
            if make_learning is None:
                learning_store = learning.LearningStore(directory)

                def make_learning(**_kwargs):
                    return learning_store
            replacements = {
                "sotto.CaptureService": Capture,
                "sotto.ShutdownBoundary": RecordingBoundary,
                "sotto.GestureEngine": RecordingEngine,
                "sotto.inject": inject,
                "nemotron_backend.NemotronRuntime": lambda: runtime,
                "history.HistoryStore": make_store,
                "learning.LearningStore": make_learning,
                "ui.init_app": lambda: status,
                "ui.run_loop": run_loop,
                "ApplicationServices.AXIsProcessTrustedWithOptions": lambda _: True,
                "Quartz.CGEventTapCreate": lambda *a: object(),
                "Quartz.CFMachPortCreateRunLoopSource": lambda *a: object(),
                "Quartz.CFRunLoopAddSource": lambda *a: None,
                "Quartz.CGEventTapEnable": lambda *a: None,
                "Quartz.CGEventTapIsEnabled": lambda *a: True,
                "Quartz.CGEventSourceFlagsState": lambda *a: 0,
                "PyObjCTools.AppHelper.callAfter": lambda fn, *a: fn(*a),
                "AppKit.NSApp": MagicMock(),
                "vad.analyze": lambda samples: (0.0, []),
            }
            for name, replacement in replacements.items():
                stack.enter_context(patch(name, replacement))
            sotto.run("right-option", speech_config=resolve_speech_config("nemotron-en"))
            # Let the event-only shutdown watchers observe their fence before
            # removing mocked OS boundaries. No real event tap or mic exists.
            time.sleep(.15)
        return status, pasted, captures, runtime

    def test_streaming_capture_reaches_history_and_delivery_once_then_shuts_down(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            store = history.HistoryStore(Path(directory))
            status, pasted, captures, runtime = self._run_one_dictation(Path(directory), lambda **kwargs: store)
            rows = store.entries()
            self.assertEqual(len(rows), 1)
            self.assertEqual(pasted, ["Complete final transcript."])
            self.assertEqual(rows[0]["text"], pasted[0])
            self.assertEqual(rows[0]["profile"], "nemotron-en")
            self.assertEqual(rows[0]["language"], "en")
            self.assertEqual(rows[0]["preprocessing"]["streaming"], True)
            self.assertFalse(captures[0].is_active())
            self.assertIsNone(captures[0]._engine)
            runtime.close.assert_called()

    def test_after_a_shutdown_request_nothing_is_transcribed_saved_or_pasted(self):
        """Ctrl-C / SIGTERM (the boundary's request): the key release of the
        capture in progress queues no transcription, Retry runs no model, the
        microphone is released and run() returns. Replaces F29b's source-text
        checks of run()."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            store = history.HistoryStore(Path(directory))
            kept = store.append("kept words", np.zeros(1600, np.float32), .1, "model")

            def drive(app):
                app["callbacks"]["start_now"]()
                app["feed"]()
                self.assertTrue(app["captures"][0].is_active())
                app["boundaries"][0].request()
                app["engines"][0]._on_finish()  # the key release after the request
                app["callbacks"]["retry"](kept["id"])
                # A normal release pastes well inside this (the test above).
                self.assertFalse(app["delivered"].wait(SETTLE_S), "nothing is pasted")

            started = time.monotonic()
            _status, pasted, captures, runtime = self._run_one_dictation(
                Path(directory), lambda **kwargs: store, drive=drive)
            self.assertLess(time.monotonic() - started, sotto.APP_DRAIN_TIMEOUT + 2)
            self.assertEqual(pasted, [])
            self.assertEqual(runtime.retries, [])
            self.assertEqual([(row["id"], row["revision"]) for row in store.entries()], [(kept["id"], 0)])
            self.assertFalse(captures[0].is_active())

    def test_remove_from_learning_set_revokes_what_is_enrolled_on_disk_now(self):
        """The menu's "Remove from learning set" acts on the learning set as it
        is on disk, not as run() loaded it at launch: a sample enrolled after
        launch (by another process) is removed and reported. Replaces F29b's
        source-text check of run()."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            # run() sees the patched names; this test also needs the real ones.
            real_store, real_learning = history.HistoryStore, learning.LearningStore
            store = real_store(root)
            row = store.append("kept words", np.zeros(1600, np.float32), .1, "model")
            reported = threading.Event()

            def drive(app):
                other = learning.LearningCoordinator(real_store(root), real_learning(root))
                other.correct(row["id"], "human words")
                self.assertIsNotNone(other.enroll(row["id"]))
                app["status"].show_info.side_effect = lambda *_args: reported.set()
                app["callbacks"]["revoke"](row["id"])
                self.assertTrue(reported.wait(5))

            status, _pasted, _captures, _runtime = self._run_one_dictation(
                root, lambda **kwargs: store,
                lambda **kwargs: real_learning(root, **kwargs), drive=drive)
            self.assertIn(("Removed from learning set", "Removed 1 local learning sample(s)."),
                          [call.args for call in status.show_info.call_args_list])
            self.assertEqual(learning.LearningStore(root).active(), [])

    def test_copy_and_save_audio_read_history_on_a_daemon_worker(self):
        """Copy and Save audio read History and touch files on a daemon worker,
        never on the main (UI) thread, and copy/save that row. Replaces F29b's
        source-text check of run()."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            store = history.HistoryStore(root)
            row = store.append("copy these words", np.zeros(1600, np.float32), .1, "model")
            armed, copied, saved = threading.Event(), threading.Event(), []
            reads, real_get = [], store.get

            def get(entry_id):
                if armed.is_set():
                    worker = threading.current_thread()
                    reads.append((worker is threading.main_thread(), worker.daemon))
                return real_get(entry_id)

            def save_audio_copy(source, folder, timestamp):
                worker = threading.current_thread()
                saved.append((source, folder, timestamp, worker is threading.main_thread(), worker.daemon))
                return root / "copy.wav"

            store.get = get
            pasteboard = MagicMock()
            pasteboard.generalPasteboard.return_value.setString_forType_.side_effect = (
                lambda *_args: copied.set())

            def drive(app):
                armed.set()
                app["callbacks"]["copy"](row["id"])
                app["callbacks"]["save"](row["id"])
                self.assertTrue(copied.wait(5))
                deadline = time.monotonic() + 5
                while not saved and time.monotonic() < deadline:
                    time.sleep(.01)

            with patch("sotto.NSPasteboard", pasteboard), patch("sotto.save_audio_copy", save_audio_copy):
                self._run_one_dictation(root, lambda **kwargs: store, drive=drive)
            board = pasteboard.generalPasteboard.return_value
            self.assertEqual(board.setString_forType_.call_args.args[0], "copy these words")
            self.assertEqual(saved, [(store.audio_path(row["id"]), Path.home() / "Desktop", row["ts"], False, True)])
            self.assertEqual(reads, [(False, True), (False, True)])

    def test_an_unreadable_history_still_launches_pastes_and_alerts_once(self):
        """F16b: a damaged history.jsonl used to raise before the menu bar
        existed, so the LaunchAgent relaunched into the same crash forever."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            index = Path(directory) / "history.jsonl"
            index.write_text('{"id": "abc123", "text": "kept"}\n{"id": "def456", "text": "cut of', encoding="utf-8")
            before = index.read_bytes()
            real_store = history.HistoryStore  # run() sees the patched name
            status, pasted, _captures, _runtime = self._run_one_dictation(
                Path(directory), lambda **kwargs: real_store(Path(directory), **kwargs))
            self.assertEqual(pasted, ["Complete final transcript."])
            self.assertEqual(index.read_bytes(), before)
            alerts = [call.args for call in status.show_error.call_args_list]
            self.assertEqual([title for title, _ in alerts], [sotto.HISTORY_UNREADABLE_TITLE])
            self.assertIn(str(index), alerts[0][1])

    def test_a_damaged_learning_set_still_launches_pastes_and_alerts_once(self):
        """N21: a damaged learning.jsonl used to make LearningStore() raise
        before the menu bar existed, so the LaunchAgent relaunched into the
        same crash forever."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            damaged = Path(directory) / "learning" / "learning.jsonl"
            damaged.parent.mkdir()
            damaged.write_text('{"sample_id": "cut of', encoding="utf-8")
            before = damaged.read_bytes()
            real_store, real_learning = history.HistoryStore, learning.LearningStore
            status, pasted, _captures, _runtime = self._run_one_dictation(
                Path(directory), lambda **kwargs: real_store(Path(directory), **kwargs),
                lambda **kwargs: real_learning(Path(directory), **kwargs))
            self.assertEqual(pasted, ["Complete final transcript."])
            self.assertEqual(damaged.read_bytes(), before)
            alerts = [call.args for call in status.show_error.call_args_list]
            self.assertEqual([title for title, _ in alerts], [sotto.LEARNING_UNREADABLE_TITLE])
            self.assertIn(str(damaged), alerts[0][1])

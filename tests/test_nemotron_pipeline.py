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


@unittest.skipUnless(sys.platform == 'darwin', 'Nemotron pipeline drives the AppKit runtime; macOS-only')
class PipelineTests(unittest.TestCase):
    def _run_one_dictation(self, directory: Path, make_store):
        """sotto.run with one streamed dictation, then Quit; History comes from
        ``make_store(**kwargs)``. Returns (status UI mock, pasted, captures, runtime)."""
        callbacks, captures, pasted = {}, [], []
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

        def inject(text, **_options):
            pasted.append(text)
            delivered.set()

        def run_loop():
            callbacks["start_now"]()
            for start in range(0, len(audio), 4096):
                when = MagicMock()
                when.sampleTime.return_value = start
                captures[0]._tap(Buffer(audio[start:start + 4096]), when)
            self.assertEqual(pasted, [])
            callbacks["finish_now"]()
            self.assertTrue(delivered.wait(5))
            callbacks["quit"]()

        with ExitStack() as stack:
            learning_store = learning.LearningStore(directory)
            replacements = {
                "sotto.CaptureService": Capture,
                "sotto.inject": inject,
                "nemotron_backend.NemotronRuntime": lambda: runtime,
                "history.HistoryStore": make_store,
                "learning.LearningStore": lambda: learning_store,
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

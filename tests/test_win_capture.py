"""win_capture: stream lifecycle with a fake sounddevice factory.

The fake factory captures the callback so tests can feed synthetic blocks,
and a gated factory reproduces the slow-open race (end() winning before the
stream exists — the mic must never be left open).
"""
from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
import unittest

import numpy as np

if sys.platform == "win32":
    import win_capture

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == "win32", "Windows-only module")
class WinCaptureTest(unittest.TestCase):
    class FakeStream:
        def __init__(self) -> None:
            self.started = False
            self.closed = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    class Factory:
        def __init__(self) -> None:
            self.streams: list[WinCaptureTest.FakeStream] = []
            self.callback = None

        def __call__(self, **kwargs) -> WinCaptureTest.FakeStream:
            self.callback = kwargs["callback"]
            self.samplerate = kwargs["samplerate"]
            self.dtype = kwargs["dtype"]
            stream = WinCaptureTest.FakeStream()
            self.streams.append(stream)
            return stream

    def setUp(self) -> None:
        self.factory = WinCaptureTest.Factory()
        self.capture = win_capture.WinCapture(stream_factory=self.factory)

    def _wait_stream_open(self) -> WinCaptureTest.FakeStream:
        for _ in range(500):  # up to 5 s
            if self.factory.streams:
                return self.factory.streams[0]
            time.sleep(0.01)
        self.fail("stream never opened")

    def _wait_closed(self, stream) -> None:
        for _ in range(500):  # up to 5 s: streams close on their own thread
            if stream.closed:
                return
            time.sleep(0.01)
        self.fail("stream was never closed")

    def _feed(self, values: list[float]) -> None:
        block = np.array([[v] for v in values], dtype=np.float32)
        self.factory.callback(block, len(block), None, None)

    def test_begin_end_roundtrip(self) -> None:
        self.assertTrue(self.capture.begin())
        stream = self._wait_stream_open()
        self.assertTrue(stream.started)
        self._feed([0.1, 0.2])
        self.assertFalse(self.capture.is_waking())
        samples = self.capture.end()
        self._wait_closed(stream)
        self.assertFalse(self.capture.is_active())
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(list(samples), [0.1, 0.2])
        self.assertAlmostEqual(self.capture.latest_rms,
                               float(np.sqrt(np.mean(np.array([0.1, 0.2]) ** 2))))

    def test_end_before_open_closes_late_stream(self) -> None:
        # Slow-open race: end() wins before the factory returns; the stream
        # that arrives afterwards must be closed, not left capturing.
        gate = threading.Event()
        real_factory = self.factory

        class GatedFactory:
            def __call__(self, **kwargs):
                real_factory.callback = kwargs["callback"]
                gate.wait(5)
                stream = WinCaptureTest.FakeStream()
                real_factory.streams.append(stream)
                return stream

        capture = win_capture.WinCapture(stream_factory=GatedFactory())
        capture.begin()
        samples = capture.end()  # wins the race; no stream exists yet
        self.assertEqual(len(samples), 0)
        gate.set()
        for _ in range(500):  # up to 5 s
            if real_factory.streams and real_factory.streams[0].closed:
                return
            time.sleep(0.01)
        self.fail("late stream was not closed")

    def test_stale_slow_open_never_attaches_to_a_newer_capture(self) -> None:
        # Tap -> discard -> press again while the first (Bluetooth-slow) open
        # is still pending: the late stream must be closed, never adopted,
        # and its callback must not feed the new capture.
        first_gate = threading.Event()
        streams: list[WinCaptureTest.FakeStream] = []
        callbacks: list = []

        class SlowFirstFactory:
            def __call__(self, **kwargs):
                callbacks.append(kwargs["callback"])
                if len(callbacks) == 1:
                    first_gate.wait(5)
                stream = WinCaptureTest.FakeStream()
                streams.append(stream)
                return stream

        def wait_for(condition) -> None:
            for _ in range(500):  # up to 5 s
                if condition():
                    return
                time.sleep(0.01)
            self.fail("condition never became true")

        capture = win_capture.WinCapture(stream_factory=SlowFirstFactory())
        self.assertTrue(capture.begin())
        wait_for(lambda: len(callbacks) == 1)
        capture.abort()
        self.assertTrue(capture.begin())
        wait_for(lambda: len(streams) == 1)  # the new capture's fast open
        first_gate.set()
        wait_for(lambda: len(streams) == 2 and streams[1].closed)
        callbacks[0](np.array([[0.9]], dtype=np.float32), 1, None, None)
        callbacks[1](np.array([[0.1]], dtype=np.float32), 1, None, None)
        self.assertEqual(list(capture.end()), [np.float32(0.1)])
        wait_for(lambda: all(stream.closed for stream in streams))

    def test_a_slow_stream_teardown_never_holds_up_end_or_abort(self) -> None:
        # [F32] end()/abort() run on the keyboard-hook thread; a Bluetooth or
        # driver stop() that takes 0.4 s must happen elsewhere.
        class SlowStream(WinCaptureTest.FakeStream):
            def stop(self) -> None:
                time.sleep(0.4)

        streams: list = []

        def factory(**kwargs):
            streams.append(SlowStream())
            factory.callback = kwargs["callback"]
            return streams[-1]

        capture = win_capture.WinCapture(stream_factory=factory)
        for finish in (capture.end, capture.abort):
            self.assertTrue(capture.begin())
            for _ in range(500):  # up to 5 s; the second open waits for the first close
                if streams and capture._stream is streams[-1]:
                    break
                time.sleep(0.01)
            else:
                self.fail("stream never opened")
            stream = streams[-1]
            factory.callback(np.array([[0.3]], dtype=np.float32), 1, None, None)
            started = time.monotonic()
            result = finish()
            self.assertLess(time.monotonic() - started, 0.1, f"{finish.__name__} waited for stop()")
            if finish == capture.end:
                self.assertEqual(list(result), [np.float32(0.3)])
            self.assertFalse(capture.is_active())
            self._wait_closed(stream)
        capture.shutdown()

    def test_no_second_stream_opens_while_the_last_one_is_still_closing(self) -> None:
        # [PR15 review] A driver whose stop() outlasts the close wait used to
        # let join(timeout) return and the next open run anyway: two streams on
        # the mic.  The start must fail cleanly instead, and work once closed.
        release = threading.Event()

        class StuckStream(WinCaptureTest.FakeStream):
            def stop(self) -> None:
                release.wait(10)

        streams: list = []
        logs: list[str] = []

        def factory(**kwargs):
            streams.append(StuckStream() if not streams else WinCaptureTest.FakeStream())
            return streams[-1]

        capture = win_capture.WinCapture(stream_factory=factory, log=logs.append)
        capture._close_wait_s = 0.2
        try:
            self.assertTrue(capture.begin())
            for _ in range(500):  # up to 5 s
                if capture._stream is not None:
                    break
                time.sleep(0.01)
            capture.end()  # the stuck stop() now holds the closing thread
            self.assertTrue(capture.begin())
            for _ in range(500):  # up to 5 s: the start fails, or (the defect) a 2nd stream opens
                if not capture.is_active() or len(streams) > 1:
                    break
                time.sleep(0.01)
            self.assertEqual(len(streams), 1, "a second stream opened while the first was closing")
            self.assertFalse(capture.is_active())
            self.assertFalse(capture.is_waking())
            self.assertTrue(any("still closing" in line for line in logs), logs)
        finally:
            release.set()
        self._wait_closed(streams[0])
        self.assertTrue(capture.begin())  # the closed mic opens normally again
        for _ in range(500):  # up to 5 s
            if capture._stream is not None:
                break
            time.sleep(0.01)
        self.assertEqual(len(streams), 2)
        capture.end()
        self._wait_closed(streams[1])

    def test_abort_discards_frames(self) -> None:
        self.capture.begin()
        self._wait_stream_open()
        self._feed([0.5, 0.5])
        self.capture.abort()
        self.assertEqual(len(self.capture.end()), 0)
        self.assertFalse(self.capture.is_active())

    def test_double_begin_is_rejected(self) -> None:
        self.assertTrue(self.capture.begin())
        self._wait_stream_open()
        self.assertFalse(self.capture.begin())
        self.capture.end()

    def test_shutdown_closes_and_rejects_begin(self) -> None:
        self.capture.begin()
        self._wait_stream_open()
        self.capture.shutdown()
        self.assertTrue(self.factory.streams[0].closed)
        self.assertFalse(self.capture.begin())


if __name__ == "__main__":
    unittest.main()


class HostApiDocsTest(unittest.TestCase):
    """[F39] The stream opens PortAudio's default input, which is MME on
    Windows; the docs used to call it WASAPI."""

    def test_the_docs_name_the_host_api_the_stream_really_uses(self) -> None:
        source = (ROOT / "win_capture.py").read_text(encoding="utf-8")
        opened = source.split("self._stream_factory(", 1)[1].split(")", 1)[0]
        self.assertNotIn("device", opened, "no device or host API is chosen: PortAudio's default")
        self.assertIn("MME", source.split('"""', 2)[1])
        for name in ("win_capture.py", "sotto_win.py", "AGENTS.md", "docs/windows-alpha.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotRegex(text, r"WASAPI (stream|capture)", name)

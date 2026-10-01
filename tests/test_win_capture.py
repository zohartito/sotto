"""win_capture: stream lifecycle with a fake sounddevice factory.

The fake factory captures the callback so tests can feed synthetic blocks,
and a gated factory reproduces the slow-open race (end() winning before the
stream exists — the mic must never be left open).
"""
from __future__ import annotations

import sys
import threading
import time
import unittest

import numpy as np

if sys.platform == "win32":
    import win_capture


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
        self.assertTrue(stream.closed)
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
        self.assertTrue(all(stream.closed for stream in streams))

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

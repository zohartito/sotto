"""Windows capture: one fresh input stream per capture (sounddevice/PortAudio).

The stream opens PortAudio's default input device, which on Windows belongs to
the MME host API (Windows resamples to the 16 kHz Sotto asks for).  No host
API is chosen explicitly.

The privacy boundary matches the Mac's ``--idle-release 0``: the microphone
is open only while the trigger key is held — ``begin()`` opens the stream,
``end()`` ends it and returns the accumulated 16 kHz float32 samples.

Unlike the Mac's process-lifetime AVAudioEngine (preroll ring, dead-route
repair via engine rebuild + launchd restart), Windows v1 opens a fresh
stream per capture, so a wedged device cannot persist across captures.  The
worker-side exact-zero guard (sotto.asr_skip_reason) still quarantines a
dead route that occurs mid-capture.

Stopping a stream can take a while (Bluetooth, slow drivers), and end() and
abort() are called on the keyboard-hook thread, which must return at once: the
finished stream is stopped and closed on its own thread, and the next open
(on its own thread too) waits for that.  If that close is still running after
CLOSE_WAIT_S, the new capture fails like a failed open rather than put a
second stream on the microphone.
"""
from __future__ import annotations

import sys
import threading
import time

import numpy as np

if sys.platform != "win32":
    raise ImportError("win_capture is Windows-only")

import sounddevice as sd

SAMPLE_RATE = 16_000
CLOSE_WAIT_S = 2.0  # how long an open waits for the previous stream to close


def _log(msg: str) -> None:
    import sys as _sys

    print(msg, file=_sys.stderr, flush=True)


class WinCapture:
    """begin/end/abort surface mirroring what sotto.run() expects of CaptureService."""

    def __init__(self, stream_factory=None, log=_log) -> None:
        self.native_rate = SAMPLE_RATE
        self.idle_release_s = 0.0
        self.latest_rms = 0.0
        self._stream_factory = stream_factory or sd.InputStream
        self._log = log
        self._stream = None
        self._frames: list[np.ndarray] = []
        self._active = False
        self._closed = False
        self._waking = False
        # Each begin() owns one generation; a slow open from an earlier,
        # already-ended capture must never attach to (or feed) a newer one.
        self._generation = 0
        self._closing: list[threading.Thread] = []
        self._close_wait_s = CLOSE_WAIT_S
        self._lock = threading.Lock()

    def begin(self) -> bool:
        """Open the stream off-thread; returns True (always a cold start)."""
        with self._lock:
            if self._closed or self._active:
                return False
            self._active = True
            self._frames = []
            self._waking = True
            self._generation += 1
            generation = self._generation
        threading.Thread(target=self._open_stream, args=(generation,),
                         daemon=True).start()
        return True

    def _current(self, generation: int) -> bool:
        return self._active and self._generation == generation

    def _close_later(self, stream) -> None:
        """Stop and close a finished stream off the caller's thread.

        Side effects: starts a closing thread; the next open waits for it.
        """
        def close() -> None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                self._log(f"! mic close failed: {str(exc)[:120]}")

        with self._lock:
            self._closing = [thread for thread in self._closing if thread.is_alive()]
            thread = threading.Thread(target=close, daemon=True)
            self._closing.append(thread)
            thread.start()

    def _wait_closed(self, timeout: float) -> bool:
        """Wait up to ``timeout`` in all for streams still closing; True once none is."""
        with self._lock:
            closing = list(self._closing)
        deadline = time.monotonic() + timeout
        for thread in closing:
            thread.join(max(0.0, deadline - time.monotonic()))
        return not any(thread.is_alive() for thread in closing)

    def _fail_start(self, generation: int, reason: str) -> None:
        with self._lock:
            if self._current(generation):
                self._active = False
                self._waking = False
        self._log(f"! mic open failed: {reason[:120]}")

    def _open_stream(self, generation: int) -> None:
        # Runs on the thread begin() started, never on the keyboard hook.
        if not self._wait_closed(self._close_wait_s):
            # Two streams must never hold the mic: this press fails instead.
            self._fail_start(generation, "the previous stream is still closing (slow audio driver); "
                                         "press again in a moment")
            return

        def _callback(indata, frame_count, time_info, status) -> None:
            block = indata[:, 0].copy()
            with self._lock:
                if not self._current(generation):
                    return
                self._frames.append(block)
                if len(block):
                    self.latest_rms = float(np.sqrt(np.mean(block ** 2)))
                self._waking = False

        stream = None
        try:
            stream = self._stream_factory(
                samplerate=self.native_rate, channels=1, dtype="float32",
                callback=_callback,
            )
            stream.start()
        except Exception as exc:
            self._fail_start(generation, str(exc))
            if stream is not None:
                stream.close()
            return
        with self._lock:
            keep = self._current(generation) and self._stream is None
            if keep:
                self._stream = stream
        if not keep:
            # This capture already ended (or a newer one began) — never
            # leave the mic open.
            stream.stop()
            stream.close()

    def is_waking(self) -> bool:
        """True until the first block actually lands after a cold start."""
        with self._lock:
            return self._waking

    def end(self) -> np.ndarray:
        with self._lock:
            frames, self._frames = self._frames or [], []
            self._active = False
            self._waking = False
            stream, self._stream = self._stream, None
        if stream is not None:
            self._close_later(stream)
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames).reshape(-1)

    def abort(self) -> None:
        with self._lock:
            self._frames = []
            self._active = False
            self._waking = False
            stream, self._stream = self._stream, None
        if stream is not None:
            self._close_later(stream)

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def release_soon(self) -> None:
        # The stream is already closed by end()/abort(); nothing to defer.
        pass

    def tick(self) -> None:
        # No process-lifetime engine to babysit on Windows v1.
        pass

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            self._active = False
            self._waking = False
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self._wait_closed(self._close_wait_s)  # the mic is closed when Sotto stops

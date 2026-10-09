"""Incremental 16 kHz canonical PCM for held-key Nemotron captures.

Audio callbacks only enqueue blocks. All methods below run on the serialized
transcription worker; cancellation is the sole cross-thread operation.
"""
from __future__ import annotations

from fractions import Fraction
import threading

import numpy as np

from audio_codec import PreparedAudio, decode_canonical, prepare_canonical

HEADROOM = 0.5           # loud end of the signal (99.9th percentile) lands here
MAX_GAIN = 10000.0
OPENING_SAMPLES = 8960   # ~0.56 s at 16 kHz sets the first boost


def _boost_for(samples) -> float:
    """The gain that puts these samples' loud end at HEADROOM."""
    loud_end = float(np.percentile(np.abs(samples), 99.9))
    return min(MAX_GAIN, HEADROOM / max(loud_end, 1e-8))


class StreamingResampler:
    """Scipy polyphase resampling with retained FIR context and no edge seams."""
    def __init__(self, rate: float):
        if not 8000 <= rate <= 96000 or int(rate) != rate:
            raise ValueError("Unsupported capture sample rate")
        ratio = Fraction(16000, int(rate))
        self.up, self.down = ratio.numerator, ratio.denominator
        self.halo = 12 * max(self.up, self.down) // self.up + 2
        self.buffer = np.zeros(0, np.float32)
        self.start = self.total = self.emitted = 0

    def push(self, samples, *, final=False):
        from scipy.signal import resample_poly
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not np.isfinite(block).all():
            raise ValueError("Non-finite microphone audio")
        self.buffer = np.concatenate((self.buffer, block))
        self.total += len(block)
        if not len(self.buffer):
            return np.zeros(0, np.float32)
        limit = ((self.total * self.up + self.down - 1) // self.down if final
                 else max(0, (self.total - self.halo) * self.up // self.down))
        offset = self.start * self.up // self.down
        converted = (self.buffer if self.up == self.down else
                     resample_poly(self.buffer, self.up, self.down))
        output = converted[self.emitted - offset:limit - offset].copy()
        self.emitted = limit
        keep_from = max(0, (self.emitted * self.down // self.up - self.halo)
                        // self.down * self.down)
        self.buffer = self.buffer[keep_from - self.start:]
        self.start = keep_from
        return output


class StreamingCapture:
    def __init__(self, runtime):
        self.runtime = runtime
        self.cancelled = threading.Event()
        self.resampler = None
        self.rate = None
        self.stream = None
        self.pcm: list[bytes] = []
        self.pending = np.zeros(0, np.float32)
        self.gain = None
        self.error = None
        self.input_error = None
        self.fallback = False
        self.closed = False
        self.prepared = None  # the exact canonical audio, kept even if finish() raises

    def feed(self, samples, rate):
        if self.cancelled.is_set() or self.closed or self.input_error is not None:
            return
        try:
            if rate != self.rate:
                if self.resampler is not None:
                    self._audio(self.resampler.push([], final=True))
                self.resampler = StreamingResampler(rate)
                self.rate = rate
            self._audio(self.resampler.push(samples))
        except Exception as exc:
            # A malformed/missing chunk must never become an apparently
            # successful partial transcript on release.
            self.input_error = exc
            self._close_stream()
            raise

    def _audio(self, samples, *, final=False):
        if self.gain is None:
            self.pending = np.concatenate((self.pending, samples))
            # Bluetooth warm-up zeros are not evidence of a quiet voice.
            # Wait for actual nonzero PCM, with no energy or VAD threshold.
            if not self.pending.any():
                return
            if len(self.pending) < OPENING_SAMPLES and not final:
                return
            self.gain = _boost_for(self.pending)
            samples, self.pending = self.pending, np.zeros(0, np.float32)
        if not len(samples):
            return
        # The boost only ever goes down. A block louder than the opening (speech
        # after a quiet lead-in) lowers it before it is applied, so a level set
        # from room noise never clips the speech that follows.
        self.gain = min(self.gain, _boost_for(samples))
        canonical = prepare_canonical(np.clip(samples * self.gain, -1.0, 1.0))
        self.pcm.append(canonical.pcm)
        if self.error is None:
            try:
                if self.stream is None:
                    self.stream = self.runtime.stream()
                self.stream.push(canonical.asr_samples)
            except Exception as exc:
                self.error = exc
                self._close_stream()

    def live_text(self) -> str:
        """Words so far (completed results plus the current interim), for the
        overlay only. Never pasted or logged; finish() decides the text."""
        stream = self.stream
        if stream is None or self.closed:
            return ""
        return " ".join([*list(stream.finals), stream.partial]).strip()

    def finish(self) -> tuple[str, PreparedAudio]:
        if self.closed or self.cancelled.is_set():
            raise RuntimeError("Streaming capture was cancelled")
        try:
            if self.input_error is not None:
                raise self.input_error
            if self.resampler is not None:
                self._audio(self.resampler.push([], final=True), final=True)
            else:
                self._audio([], final=True)
            pcm = b"".join(self.pcm)
            samples, identity = decode_canonical(pcm)
            samples.setflags(write=False)
            prepared = self.prepared = PreparedAudio(pcm, samples, identity)
            if self.cancelled.is_set():
                raise RuntimeError("Streaming capture was cancelled")
            if not len(samples) or not samples.any():
                return "", prepared
            try:
                if self.error is not None:
                    raise self.error
                text = self.stream.finish()
            except Exception:
                # Never deliver partial text after a failed stream. Retry the
                # exact retained canonical input once on the same worker.
                self._close_stream()
                self.fallback = True
                text = self.runtime.transcribe(samples)
            return text, prepared
        finally:
            self.close()

    def _close_stream(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def close(self):
        self._close_stream()
        self.closed = True
        self.pcm.clear()
        self.pending = np.zeros(0, np.float32)
        self.resampler = None

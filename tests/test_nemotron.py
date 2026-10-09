from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import resample_poly

import nemotron_backend
import sotto
from speech_config import (load_engine_mode, resolve_speech_config,
                           save_engine_mode, with_language)
from streaming_audio import StreamingCapture, StreamingResampler


class FakeStream:
    def __init__(self, *, fail=False):
        self.blocks = []
        self.closed = False
        self.fail = fail

    def push(self, samples):
        self.blocks.append(samples.copy())
        if self.fail:
            raise RuntimeError("decoder interrupted")

    def finish(self):
        return "Complete final transcript."

    def close(self):
        self.closed = True


class FakeRuntime:
    def __init__(self, *, fail=False):
        self.streams = []
        self.retries = []
        self.fail = fail

    def stream(self):
        stream = FakeStream(fail=self.fail)
        self.streams.append(stream)
        return stream

    def transcribe(self, samples):
        self.retries.append(samples.copy())
        return "Recovered complete transcript."


class NemotronTests(unittest.TestCase):
    def test_profile_preserves_whisper_default_and_enforces_english(self):
        self.assertEqual(resolve_speech_config().profile.backend, "whisper")
        config = resolve_speech_config("nemotron-en")
        self.assertEqual((config.profile.backend, config.language), ("nemotron", "en"))
        for language in ("pt", "auto"):
            with self.assertRaises(ValueError):
                resolve_speech_config("nemotron-en", language=language)
            with self.assertRaises(ValueError):
                with_language(config, language)
        with self.assertRaises(ValueError):
            resolve_speech_config("nemotron-en", model="unsealed/model")

    def test_engine_preference_is_private_and_independent_of_language(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engine"
            self.assertEqual(load_engine_mode(path), "whisper")
            save_engine_mode("nemotron", path)
            self.assertEqual(load_engine_mode(path), "nemotron")
            if sys.platform != "win32":  # Windows chmod only toggles read-only
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            path.write_text("invalid")
            self.assertEqual(load_engine_mode(path), "whisper")
            with self.assertRaises(ValueError):
                save_engine_mode("invalid", path)

    def test_cli_explicit_profile_and_adaptive_do_not_inherit_engine_preference(self):
        for args, expected in (([], "nemotron-en"), (["--profile", "auto"], "auto"),
                               (["--adaptive"], "auto"), (["--model", "custom"], "auto")):
            with self.subTest(args=args), patch("sys.argv", ["sotto.py", *args]), \
                    patch("speech_config.load_engine_mode", return_value="nemotron"), \
                    patch("speech_config.load_language_mode", return_value="pt"), \
                    patch.object(sotto, "run") as run:
                sotto.main()
                self.assertEqual(run.call_args.kwargs["speech_config"].profile.name, expected)

    def test_streaming_resampling_matches_whole_recording_across_buffer_edges(self):
        rng = np.random.default_rng(7)
        for rate in (16000, 24000, 44100, 48000, 96000):
            with self.subTest(rate=rate):
                audio = rng.normal(0, 0.03, rate * 2 + 17).astype(np.float32)
                resampler = StreamingResampler(rate)
                pieces = []
                cursor = 0
                while cursor < len(audio):
                    size = int(rng.integers(1, 5000))
                    pieces.append(resampler.push(audio[cursor:cursor + size]))
                    cursor += size
                pieces.append(resampler.push([], final=True))
                actual = np.concatenate(pieces)
                expected = resample_poly(audio, resampler.up, resampler.down)
                np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=1e-6)
                self.assertLess(len(resampler.buffer), 1000)

    def test_asr_and_history_share_bit_exact_streaming_pcm(self):
        runtime = FakeRuntime()
        capture = StreamingCapture(runtime)
        audio = np.sin(np.arange(48000, dtype=np.float32) / 37) * .002
        for start in range(0, len(audio), 4096):
            capture.feed(audio[start:start + 4096], 48000)
        self.assertTrue(runtime.streams, "ASR must run before release")
        text, prepared = capture.finish()
        self.assertEqual(text, "Complete final transcript.")
        self.assertEqual(len(prepared.asr_samples), 16000)
        heard = np.concatenate(runtime.streams[0].blocks)
        np.testing.assert_array_equal(heard, prepared.asr_samples)
        self.assertEqual(hashlib.sha256(prepared.pcm).hexdigest(), prepared.identity.sha256)
        self.assertTrue(runtime.streams[0].closed)

    def test_speech_after_a_quiet_lead_in_is_not_clipped(self):
        # The boost used to be set once from the first ~0.56 s. Room noise there
        # made it huge, and the speech after it was clipped for the decoder and
        # in History alike.
        rng = np.random.default_rng(0)
        rate = 48000
        noise = rng.normal(0, 0.002, int(0.8 * rate)).astype(np.float32)
        seconds = np.arange(3 * rate) / rate
        speech = (0.25 * np.sin(2 * np.pi * 140 * seconds)).astype(np.float32)
        audio = np.concatenate([noise, speech])
        runtime = FakeRuntime()
        capture = StreamingCapture(runtime)
        for start in range(0, len(audio), 4096):
            capture.feed(audio[start:start + 4096], rate)
        _, prepared = capture.finish()
        spoken = prepared.asr_samples[len(noise) // 3:]
        self.assertLess(np.mean(np.abs(spoken) >= 0.999), 0.01)
        np.testing.assert_array_equal(np.concatenate(runtime.streams[0].blocks), prepared.asr_samples)

    def test_zero_capture_never_creates_decoder_but_quiet_audio_does(self):
        runtime = FakeRuntime()
        capture = StreamingCapture(runtime)
        capture.feed(np.zeros(48000, np.float32), 48000)
        self.assertEqual(capture.finish()[0], "")
        self.assertFalse(runtime.streams)
        capture = StreamingCapture(runtime)
        capture.feed(np.full(48000, 1e-7, np.float32), 48000)
        capture.finish()
        self.assertEqual(len(runtime.streams), 1)

    def test_stream_failure_retries_whole_canonical_audio_without_partial_delivery(self):
        runtime = FakeRuntime(fail=True)
        capture = StreamingCapture(runtime)
        audio = np.ones(32000, np.float32) * .02
        for start in range(0, len(audio), 4096):
            capture.feed(audio[start:start + 4096], 16000)
        text, prepared = capture.finish()
        self.assertEqual(text, "Recovered complete transcript.")
        self.assertTrue(capture.fallback)
        self.assertEqual(len(runtime.retries), 1)
        np.testing.assert_array_equal(runtime.retries[0], prepared.asr_samples)
        self.assertEqual(len(prepared.asr_samples), len(audio))

    def test_cancellation_and_capture_abort_drop_queued_audio(self):
        runtime = FakeRuntime()
        stream = StreamingCapture(runtime)
        queued = []
        service = sotto.CaptureService()
        with patch.object(service, "_start_engine"):
            service.begin(stream=stream, enqueue=queued.append)
        service.abort()
        self.assertTrue(stream.cancelled.is_set())
        self.assertFalse(service.is_active())
        stream.feed(np.ones(16000, np.float32), 16000)
        self.assertFalse(runtime.streams)
        self.assertEqual(queued, [("stream-close", stream)])
        with self.assertRaises(RuntimeError):
            stream.finish()
        stream.close()

    def test_rate_change_and_back_to_back_captures_do_not_mix_decoder_state(self):
        runtime = FakeRuntime()
        first = StreamingCapture(runtime)
        first.feed(np.ones(24000, np.float32) * .01, 24000)
        first.feed(np.ones(48000, np.float32) * .02, 48000)
        _, prepared = first.finish()
        self.assertEqual(len(prepared.asr_samples), 32000)
        second = StreamingCapture(runtime)
        second.feed(np.ones(16000, np.float32) * .03, 16000)
        second.finish()
        self.assertEqual(len(runtime.streams), 2)
        self.assertTrue(all(stream.closed for stream in runtime.streams))

    def test_invalid_audio_latches_failure_instead_of_delivering_a_prefix(self):
        runtime = FakeRuntime()
        capture = StreamingCapture(runtime)
        capture.feed(np.ones(16000, np.float32) * .02, 16000)
        with self.assertRaises(ValueError):
            capture.feed(np.array([np.nan], np.float32), 16000)
        capture.feed(np.ones(16000, np.float32) * .02, 16000)
        with self.assertRaises(ValueError):
            capture.finish()
        self.assertTrue(runtime.streams[0].closed)
        self.assertFalse(runtime.retries)

    def test_changed_native_library_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "1/lib/libtest.dylib"
            library.parent.mkdir(parents=True)
            library.write_bytes(b"expected library")
            (root / "1/model.gguf").write_bytes(b"model")
            spec = root / "spec.json"
            spec.write_text(json.dumps({"version": "1", "model_file": "model.gguf",
                                       "model_sha256": hashlib.sha256(b"model").hexdigest(),
                                       "libraries": {"libtest.dylib":
                                                     hashlib.sha256(library.read_bytes()).hexdigest()}}))
            with patch.object(nemotron_backend, "SPEC_PATH", spec), \
                    patch.object(nemotron_backend, "INSTALL_ROOT", root):
                nemotron_backend.installation()
                library.write_bytes(b"changed library")
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    nemotron_backend.installation()


if __name__ == "__main__":
    unittest.main()

"""Parakeet as a third engine: pinned, cache-first, array input, chunked."""
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

import sotto
import speech_config
from speech_config import MODEL_PROFILES, MODEL_REVISIONS, resolve_speech_config, with_language

REPO = MODEL_PROFILES["parakeet"].repo


class ParakeetConfigTests(unittest.TestCase):
    def test_engine_profile_pin_and_language_policy(self):
        self.assertIn("parakeet", speech_config.ENGINE_CHOICES)
        self.assertRegex(MODEL_REVISIONS[REPO], r"^[0-9a-f]{40}$")
        config = resolve_speech_config("parakeet", language="pt")
        self.assertEqual((config.profile.backend, config.language), ("parakeet", None))
        self.assertIsNone(with_language(config, "en").language)
        with self.assertRaises(ValueError):
            resolve_speech_config("parakeet", model="someone/else")
        kwargs, metadata = sotto._transcription_kwargs(config, ("Sotto",), 3.0)
        self.assertEqual(kwargs, {})
        self.assertIn("disabled", metadata["prompt"])

    def test_cache_check_is_a_pure_path_check_of_the_pinned_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            hub = Path(temporary)
            self.assertFalse(speech_config.pinned_snapshot_cached(REPO, hub, ("config.json",)))
            folder = hub / f"models--{REPO.replace('/', '--')}" / "snapshots" / MODEL_REVISIONS[REPO]
            folder.mkdir(parents=True)
            (folder / "config.json").write_text("{}")
            self.assertTrue(speech_config.pinned_snapshot_cached(REPO, hub, ("config.json",)))
            self.assertFalse(speech_config.pinned_snapshot_cached(REPO, hub, ("config.json", "model.safetensors")))
            self.assertFalse(speech_config.pinned_snapshot_cached("unpinned/repo", hub))


@unittest.skipUnless(sys.platform == "darwin", "Parakeet runs on MLX (macOS only)")
class LocalParakeetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.snapshot = Path(temporary.name)
        (self.snapshot / "config.json").write_text("{}")
        (self.snapshot / "model.safetensors").write_bytes(b"weights")

    def local(self, offline=False):
        with patch("offline_runtime.offline_requested", return_value=offline), \
             patch("huggingface_hub.snapshot_download", return_value=str(self.snapshot)) as snapshot:
            local = sotto.LocalParakeet(REPO)
        self.assertEqual(snapshot.call_args.kwargs["revision"], MODEL_REVISIONS[REPO])
        self.assertTrue(snapshot.call_args.kwargs["local_files_only"])
        return local

    def test_cached_pinned_snapshot_resolves_without_download(self):
        self.assertEqual(self.local().path, str(self.snapshot))

    def test_array_input_is_decoded_in_chunks_and_whisper_options_are_ignored(self):
        local = self.local()
        seen = []
        model = types.SimpleNamespace(
            preprocessor_config=object(),
            generate=lambda mel: [types.SimpleNamespace(text=f" part{len(seen)} ")])
        with patch.object(local, "_load", return_value=model), \
             patch("parakeet_mlx.audio.get_logmel", side_effect=lambda audio, config: seen.append(audio.shape[0])):
            samples = np.zeros(int((local.CHUNK_SECONDS * 2 + 5) * sotto.SAMPLE_RATE), dtype=np.float32)
            result = local.transcribe(samples, path_or_hf_repo="ignored", language="en")
        self.assertEqual(len(seen), 3)                       # 120 s + 120 s + 5 s
        self.assertEqual(result["text"], "part1 part2 part3")
        self.assertIsNone(result["language"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest

from speech_config import (
    GLOSSARY_MAX_CAPTURE_SECONDS,
    MAX_GLOSSARY_CHARACTERS,
    MAX_GLOSSARY_TERMS,
    glossary_prompt,
    language_mode,
    load_glossary,
    load_language_mode,
    resolve_speech_config,
    save_language_mode,
    with_language,
)


class SpeechConfigTests(unittest.TestCase):
    def test_built_in_profiles_favour_no_language_besides_english(self) -> None:
        import speech_config
        self.assertEqual(resolve_speech_config("auto").model_repo, "mlx-community/whisper-large-v3-turbo")
        self.assertIsNone(resolve_speech_config("auto").language)
        self.assertEqual(set(speech_config.MODEL_PROFILES), {"auto", "nemotron-en", "parakeet"})
        self.assertEqual(speech_config.automatic_languages(), ("en",))

    def test_model_and_language_overrides_win(self) -> None:
        config = resolve_speech_config("auto", model="local/model", language="en")
        self.assertEqual(config.model_repo, "local/model")
        self.assertEqual(config.language, "en")
        self.assertIsNone(resolve_speech_config("auto", language="auto").language)

    def test_invalid_profile_or_language_fails(self) -> None:
        with self.assertRaises(ValueError):
            resolve_speech_config("not-a-profile")
        with self.assertRaises(ValueError):
            resolve_speech_config(language="xx")
        # Every Whisper language is selectable; Automatic chooses among Settings' set.
        self.assertEqual(resolve_speech_config(language="fr").language, "fr")

    def test_language_modes_change_only_the_decode_hint(self) -> None:
        config = resolve_speech_config("auto", language="auto")
        hinted = with_language(config, "pt")
        self.assertEqual(language_mode(config.language), "auto")
        self.assertEqual(hinted.language, "pt")
        self.assertEqual(hinted.model_repo, config.model_repo)
        self.assertEqual(with_language(config, "en").language, "en")

    def test_language_preference_is_private_atomic_and_fails_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private" / "language-mode"
            self.assertEqual(load_language_mode(path), "auto")
            self.assertEqual(save_language_mode("pt", path), "pt")
            self.assertEqual(load_language_mode(path), "pt")
            if sys.platform == "win32":
                # Windows chmod only supports the read-only bit, so the
                # 0o600 tightening is a no-op; assert what still matters.
                self.assertTrue(path.is_file())
                self.assertTrue(os.access(path, os.R_OK))
            else:
                self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            path.write_text("not-a-language\n", encoding="utf-8")
            self.assertEqual(load_language_mode(path, default="en"), "en")

    def test_text_glossary_deduplicates_and_skips_comments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terms.txt"
            path.write_text("# comment\n Sotto \n\n東京\nSotto\n", encoding="utf-8")
            self.assertEqual(load_glossary(path), ("Sotto", "東京"))

    def test_json_glossary_and_bounds_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terms.json"
            path.write_text(json.dumps([f"term-{index}" for index in range(300)]), encoding="utf-8")
            terms = load_glossary(path)
            self.assertEqual(terms[0], "term-0")
            self.assertLessEqual(len(terms), MAX_GLOSSARY_TERMS)
            self.assertLessEqual(len(", ".join(terms)), MAX_GLOSSARY_CHARACTERS)

    def test_long_capture_disables_prompt_with_reason(self) -> None:
        prompt = glossary_prompt(("Sotto", "الدار البيضاء"), GLOSSARY_MAX_CAPTURE_SECONDS + 0.01)
        self.assertIsNone(prompt.prompt)
        self.assertIn("exceeds", prompt.reason or "")
        enabled = glossary_prompt(("Sotto",), 30)
        self.assertEqual(enabled.prompt, "Speech context: Sotto.")
        self.assertIsNone(enabled.reason)

    def test_benchmark_dry_run_does_not_import_mlx_whisper(self) -> None:
        # Importing benchmark is safe; its MLX import is intentionally inside
        # the non-dry-run transcription function.
        from benchmark import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clip.wav").touch()
            manifest = root / "manifest.jsonl"
            manifest.write_text('{"audio":"clip.wav","reference":"hello"}\n', encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main([str(manifest), "--dry-run"]), 0)
            self.assertIn('"dry_run": true', output.getvalue())
            self.assertNotIn("mlx_whisper", sys.modules)


if __name__ == "__main__":
    unittest.main()

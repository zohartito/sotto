"""The shared path from a finished transcript to what gets delivered."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dictionary
import pipeline
import settings
from pipeline import NO_SPEECH_TEXT, screen

LOOP = "difference " * 40


class TemporaryUserFiles:
    """Point the dictionary and settings at a temp folder for each test."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.dictionary = Path(folder.name) / "dictionary.txt"
        for target, value in ((dictionary, self.dictionary),
                              (settings, Path(folder.name) / "settings.json")):
            name = "DICTIONARY_PATH" if target is dictionary else "SETTINGS_PATH"
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)


class ScreenTests(TemporaryUserFiles, unittest.TestCase):
    def screen(self, text, *, seconds=3.0, speech_fraction=0.8, language="en", **options):
        return screen(text, seconds=seconds, speech_fraction=speech_fraction, language=language, **options)

    def test_empty_output_becomes_the_no_speech_marker_and_is_held(self):
        result = self.screen("   ")
        self.assertEqual((result.text, result.held), (NO_SPEECH_TEXT, "empty transcript"))

    def test_a_stock_phrase_from_a_speechless_capture_is_held(self):
        self.assertIn("tiny transcript", self.screen("Thank you.", speech_fraction=0.0).held)

    def test_whispered_dictation_wins_over_a_zero_vad_score(self):
        whisper = "I am whispering this whole paragraph because the room is quiet and people are asleep."
        self.assertIsNone(self.screen(whisper, seconds=8.0, speech_fraction=0.0).held)

    def test_an_impossible_rate_is_held(self):
        self.assertIn("chars/sec", self.screen("word " * 100, seconds=1.0).held)

    def test_a_loop_after_real_dictation_keeps_the_clean_prefix(self):
        prefix = "That is part of the plan we agreed on last week."
        result = self.screen(f"{prefix} {LOOP}", seconds=12.0)
        self.assertIsNone(result.held)
        self.assertEqual(result.text, prefix)
        self.assertEqual(result.receipts["repetition_trimmed"]["repeats"], 40)

    def test_dictionary_then_english_cleanup_with_receipts(self):
        self.dictionary.write_text("soto => Sotto\n", encoding="utf-8")
        result = self.screen("Um, soto works.")
        self.assertEqual(result.text, "Sotto works.")
        self.assertEqual(result.receipts["dictionary"]["asr_text"], "Um, soto works.")
        self.assertEqual(result.receipts["voice"]["asr_text"], "Um, Sotto works.")

    def test_scratch_that_is_an_action_not_text(self):
        self.assertEqual(self.screen("Scratch that.").voice_action, "scratch")

    def test_unknown_or_other_languages_keep_every_word(self):
        for language in (None, "pt"):
            self.assertEqual(self.screen("Eu comprei um carro.", language=language).text, "Eu comprei um carro.")

    def test_held_text_and_clean_false_skip_the_dictionary(self):
        self.dictionary.write_text("thank you => THANKS\n", encoding="utf-8")
        self.assertEqual(self.screen("Thank you.", speech_fraction=0.0).text, "Thank you.")
        self.assertEqual(self.screen("Thank you so much for this.", clean=False).text, "Thank you so much for this.")

    def test_log_callback_hears_salvage_and_dictionary(self):
        self.dictionary.write_text("plan => Plan\n", encoding="utf-8")
        heard = []
        self.screen(f"That is part of the plan we agreed on last week. {LOOP}", seconds=12.0, log=heard.append)
        self.assertTrue(any("repetition loop" in line for line in heard))
        self.assertTrue(any("dictionary: 1 replacement" in line for line in heard))


def legacy_inline_screen(text, seconds, speech_fraction, language):
    """The live worker's guard block in sotto.run() as of b3abda9, copied
    verbatim (variable names kept) so screen() can be checked against it."""
    import sotto
    preprocessing = {}
    no_speech = sotto.reads_as_no_speech(text, speech_fraction)
    if no_speech == "empty transcript":
        text = "[no speech detected]"
    hallucinated = sotto.looks_hallucinated(text, seconds)
    if hallucinated and not no_speech:
        salvaged = sotto.salvage_repetition_loop(text, seconds)
        if salvaged is not None:
            text, trim_receipt = salvaged
            preprocessing["repetition_trimmed"] = trim_receipt
            hallucinated = None
    reason = hallucinated or no_speech
    voice_action = None
    if not reason:
        text = sotto.apply_personal_dictionary(text, preprocessing)
        text, voice_action = sotto.apply_voice_cleanup(text, preprocessing, language)
    return text, reason, voice_action, preprocessing


class CharacterizationTests(TemporaryUserFiles, unittest.TestCase):
    """screen() gives exactly what the inline worker code gave, input by input."""

    def test_same_result_as_the_inline_worker_code(self):
        self.dictionary.write_text("soto => Sotto\nopen ai => OpenAI\n", encoding="utf-8")
        texts = ["", "  ", "you", "Thank you.", "Um, soto said open ai ships it.", "Scratch that.",
                 "New line", "Eu comprei um carro e um livro.", "word " * 100,
                 f"That is part of the plan we agreed on last week. {LOOP}", LOOP,
                 "A normal sentence of dictation, nothing odd about it at all."]
        compared = 0
        for text in texts:
            for seconds in (0.5, 3.0, 12.0):
                for speech_fraction in (0.0, 0.05, 0.8):
                    for language in ("en", "pt", None):
                        new = screen(text, seconds=seconds, speech_fraction=speech_fraction, language=language)
                        old = legacy_inline_screen(text, seconds, speech_fraction, language)
                        self.assertEqual((new.text, new.held, new.voice_action, new.receipts), old,
                                         (text[:40], seconds, speech_fraction, language))
                        compared += 1
        self.assertEqual(compared, 324)


class MovedFunctionsTests(unittest.TestCase):
    """The guards moved here from sotto.py; sotto still exposes the same names."""

    def test_sotto_reexports_the_pipeline_functions(self):
        import sotto
        for name in ("looks_hallucinated", "salvage_repetition_loop", "reads_as_no_speech"):
            self.assertIs(getattr(sotto, name), getattr(pipeline, name))


if __name__ == "__main__":
    unittest.main()

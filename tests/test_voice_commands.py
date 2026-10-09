"""English voice commands and filler cleanup, applied after the dictionary."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import settings
import voice_commands
from voice_commands import clean, is_english


class CleanTests(unittest.TestCase):
    def test_fillers_go_with_the_commas_that_set_them_off(self):
        self.assertEqual(clean("Um, I think so.")[0], "I think so.")
        self.assertEqual(clean("So, um, the plan is fine.")[0], "So the plan is fine.")
        self.assertEqual(clean("I was, uh, thinking about it.")[0], "I was thinking about it.")
        self.assertEqual(clean("Well, um.")[0], "Well.")
        self.assertEqual(clean("Um.")[0], "")

    def test_words_that_merely_contain_a_filler_stay(self):
        text = "The uh-huh moment, umbrella, hummus and Hmm."
        self.assertEqual(clean(text), (text, None))

    def test_line_breaks_only_when_said_as_their_own_phrase(self):
        self.assertEqual(clean("Hello. New line. How are you?")[0], "Hello.\nHow are you?")
        self.assertEqual(clean("Dear Sam, new paragraph. Thanks for the notes.")[0],
                         "Dear Sam,\n\nThanks for the notes.")
        self.assertEqual(clean("Hello, new line, how are you?")[0], "Hello,\nHow are you?")
        self.assertEqual(clean("We launched a new line of products.")[0], "We launched a new line of products.")

    def test_scratch_that_alone_is_an_action_and_otherwise_just_words(self):
        self.assertEqual(clean("Scratch that."), ("", "scratch"))
        self.assertEqual(clean("Please scratch that line.")[1], None)

    def test_each_part_can_be_switched_off(self):
        self.assertEqual(clean("Um, new line.", fillers=False, commands=True)[0], "Um,\n")
        self.assertEqual(clean("Scratch that.", commands=False), ("Scratch that.", None))
        self.assertEqual(clean("Um, hi.", fillers=False, commands=False), ("Um, hi.", None))

    def test_english_only(self):
        self.assertTrue(is_english("en"))
        self.assertTrue(is_english("EN"))
        self.assertFalse(is_english("pt"))                 # "um" is a Portuguese word
        # Parakeet reports no language and speaks 25 of them, so an unknown
        # language is never treated as English, whatever the Settings list says.
        self.assertFalse(is_english(None))
        self.assertFalse(is_english(""))

    def test_portuguese_parakeet_text_keeps_every_word(self):
        import sotto
        text = "Eu comprei um carro e um livro."
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(settings, "SETTINGS_PATH", Path(folder) / "settings.json"):
            self.assertEqual(sotto.apply_voice_cleanup(text, {}, None), (text, None))


class PipelineTests(unittest.TestCase):
    def test_cleanup_reads_settings_and_records_what_the_model_heard(self):
        import settings
        import sotto
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "settings.json"
            with patch.object(settings, "SETTINGS_PATH", path):
                receipt = {}
                self.assertEqual(sotto.apply_voice_cleanup("Um, hello. New line.", receipt, "en"),
                                 ("Hello.\n", None))
                self.assertEqual(receipt["voice"]["asr_text"], "Um, hello. New line.")
                untouched = {}
                self.assertEqual(sotto.apply_voice_cleanup("Um, olá.", untouched, "pt"), ("Um, olá.", None))
                self.assertEqual(untouched, {})
                settings.save({"remove_fillers": False, "voice_commands": False}, path)
                self.assertEqual(sotto.apply_voice_cleanup("Scratch that.", {}, "en"), ("Scratch that.", None))
                settings.save({"voice_commands": True}, path)
                scratch = {}
                self.assertEqual(sotto.apply_voice_cleanup("Scratch that.", scratch, "en"), ("", "scratch"))
                self.assertEqual(scratch["voice"]["action"], "scratch")

    def test_defaults_are_on_and_only_booleans_are_kept(self):
        import settings
        self.assertTrue(settings.DEFAULTS["voice_commands"] and settings.DEFAULTS["remove_fillers"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "settings.json"
            with self.assertRaises(ValueError):
                settings.save({"voice_commands": "yes"}, path)
            saved = settings.save({"remove_fillers": False}, path)
            self.assertEqual((saved["voice_commands"], saved["remove_fillers"]), (True, False))
        self.assertEqual(voice_commands.SCRATCH_WINDOW_S, 60.0)


if __name__ == "__main__":
    unittest.main()

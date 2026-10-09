"""How dictated text is inserted: spacing decisions, typing and paste dispatch."""
import sys
import unittest
from unittest.mock import MagicMock, patch

import sotto


class ComposeInsertionTests(unittest.TestCase):
    def test_smart_adds_a_leading_space_only_when_words_would_touch(self):
        compose = sotto.compose_insertion
        self.assertEqual(compose("world.", "smart", "o"), " world.")
        self.assertEqual(compose("world.", "smart", " "), "world.")
        self.assertEqual(compose("world.", "smart", ""), "world.")       # start of the field
        self.assertEqual(compose("world.", "smart", "("), "world.")
        self.assertEqual(compose("world.", "smart", "\n"), "world.")
        self.assertEqual(compose(", then", "smart", "o"), ", then")     # punctuation attaches
        self.assertEqual(compose("Γεια", "smart", "a"), " Γεια")

    def test_unknown_context_and_other_modes(self):
        compose = sotto.compose_insertion
        self.assertEqual(compose("Hi", "smart", None), "Hi ")            # falls back to trailing
        self.assertEqual(compose("Hi", "trailing", "x"), "Hi ")
        self.assertEqual(compose("Hi ", "trailing", None), "Hi ")
        self.assertEqual(compose("Hi", "none", "x"), "Hi")
        self.assertEqual(compose("", "smart", "x"), "")


class TypingTests(unittest.TestCase):
    def test_typing_sends_small_utf16_aware_chunks_and_never_touches_the_clipboard(self):
        posted = []
        with patch.object(sotto.Quartz, "CGEventCreateKeyboardEvent", side_effect=lambda *a: MagicMock()), \
             patch.object(sotto.Quartz, "CGEventKeyboardSetUnicodeString",
                          side_effect=lambda event, units, chunk: posted.append((units, chunk))), \
             patch.object(sotto.Quartz, "CGEventSetIntegerValueField"), \
             patch.object(sotto.Quartz, "CGEventPost"), patch.object(sotto.time, "sleep"), \
             patch.object(sotto, "NSPasteboard") as pasteboard, \
             patch.object(sotto, "secure_input_active", return_value=False):
            sotto.inject("Hello 😀 world, this is typed", insert_mode="type", spacing="none")
        chunks = [chunk for _units, chunk in posted[::2]]  # key-down and key-up carry the same chunk
        self.assertEqual("".join(chunks), "Hello 😀 world, this is typed")
        self.assertTrue(all(len(chunk) <= 16 for chunk in chunks))
        self.assertEqual(posted[0][0], len(chunks[0].encode("utf-16-le")) // 2)  # emoji = 2 units
        pasteboard.generalPasteboard.assert_not_called()

    def test_secure_input_blocks_both_modes(self):
        with patch.object(sotto, "secure_input_active", return_value=True), \
             patch.object(sotto, "type_text") as typed, patch.object(sotto, "NSPasteboard") as pasteboard:
            sotto.inject("secret", insert_mode="type")
            sotto.inject("secret", insert_mode="paste")
        typed.assert_not_called()
        pasteboard.generalPasteboard.assert_not_called()

    @unittest.skipUnless(sys.platform == "darwin", "macOS Accessibility (ApplicationServices) caret lookup")
    def test_caret_lookup_failure_is_unknown_not_an_error(self):
        with patch("ApplicationServices.AXUIElementCreateSystemWide", side_effect=RuntimeError("no AX")):
            self.assertIsNone(sotto.character_before_caret())



class SecureInputProbeTests(unittest.TestCase):
    def test_probe_calls_carbon_and_fails_open_only_when_unavailable(self):
        self.assertIsInstance(sotto.secure_input_active(), bool)
        with patch.object(sotto, "_secure_input_probe", lambda: True):
            self.assertTrue(sotto.secure_input_active())
        with patch.object(sotto, "_secure_input_probe", side_effect=OSError("gone")):
            self.assertFalse(sotto.secure_input_active())


if __name__ == "__main__":
    unittest.main()

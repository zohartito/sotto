"""Rescuing the clean prefix of a transcript that ends in a whisper loop.

Whisper can transcribe real dictation correctly and only then degenerate
into a repeated word or phrase. The whole-transcript hallucination guard
correctly refuses to paste the result, which used to cost the user every
word they actually said. Reference capture from this machine's history
(id 57419be8, 2026-08-15): 31.04s, 403 chars of real dictation followed by
"difference" x223 and a "time." coda — 2862 chars at 92 chars/sec.
"""

import unittest

from sotto import (LOOP_MIN_REPEATS, looks_hallucinated, salvage_repetition_loop)

# the exact transcript prefix whisper produced before it started looping
OBSERVED_PREFIX = (
    "On the Mac. If the Mac doesn't have an application that the PC has, "
    "maybe we can download it. Vice versa. If the PC doesn't have an "
    "application the Mac has, we can download it. Know what the best MCP "
    "plugin is for each. As well as how many there are for each. And what "
    "they are. And eventually the goal is to build our own. If we need "
    "Rabbit 2027 and we have 2026, that's part of the plan. To update it."
)
OBSERVED_TEXT = OBSERVED_PREFIX + " " + ("difference " * 223) + "time."
OBSERVED_SECONDS = 31.04


class SalvageRepetitionLoopTest(unittest.TestCase):
    def test_observed_capture_is_condemned_as_a_whole(self):
        # the premise: without salvage the user loses all 403 real chars
        self.assertIsNotNone(looks_hallucinated(OBSERVED_TEXT, OBSERVED_SECONDS))

    def test_observed_capture_salvages_its_real_prefix(self):
        salvaged = salvage_repetition_loop(OBSERVED_TEXT, OBSERVED_SECONDS)
        self.assertIsNotNone(salvaged)
        prefix, receipt = salvaged
        self.assertEqual(prefix, OBSERVED_PREFIX)
        self.assertEqual(receipt["repeats"], 223)
        self.assertEqual(receipt["unit"], "difference")
        self.assertEqual(receipt["dropped_chars"], len(OBSERVED_TEXT) - len(OBSERVED_PREFIX))

    def test_salvaged_prefix_passes_the_guard_that_blocked_the_whole(self):
        prefix, _ = salvage_repetition_loop(OBSERVED_TEXT, OBSERVED_SECONDS)
        self.assertIsNone(looks_hallucinated(prefix, OBSERVED_SECONDS))

    def test_trailing_coda_after_the_loop_is_dropped_too(self):
        # "time." arrives after the loop; it is not dictation the user gave
        prefix, _ = salvage_repetition_loop(OBSERVED_TEXT, OBSERVED_SECONDS)
        self.assertNotIn("time.", prefix[-16:])

    def test_wall_to_wall_garbage_has_nothing_to_salvage(self):
        # the music capture that transcribed as 'Woo' x200 — no real prefix
        self.assertIsNone(salvage_repetition_loop("Woo " * 200, 20.0))

    def test_loop_of_a_repeated_phrase_is_caught(self):
        text = OBSERVED_PREFIX + " " + ("thank you very much. " * 30)
        salvaged = salvage_repetition_loop(text, OBSERVED_SECONDS)
        self.assertIsNotNone(salvaged)
        self.assertEqual(salvaged[0], OBSERVED_PREFIX)

    def test_punctuation_and_case_do_not_hide_a_loop(self):
        # whisper's documented dead-air loop on this repo's own long captures
        text = OBSERVED_PREFIX + " " + ("Okay. okay, OKAY! " * 12)
        salvaged = salvage_repetition_loop(text, OBSERVED_SECONDS)
        self.assertIsNotNone(salvaged)
        self.assertEqual(salvaged[0], OBSERVED_PREFIX)

    def test_clean_dictation_is_never_trimmed(self):
        self.assertIsNone(salvage_repetition_loop(OBSERVED_PREFIX, OBSERVED_SECONDS))

    def test_natural_speech_repetition_survives(self):
        # people do repeat themselves; the threshold sits well above it
        emphatic = OBSERVED_PREFIX + " no no no no, that is not what I meant."
        self.assertIsNone(salvage_repetition_loop(emphatic, OBSERVED_SECONDS))
        self.assertGreaterEqual(LOOP_MIN_REPEATS, 5)

    def test_a_short_prefix_is_not_worth_pasting(self):
        # "Hey." then a loop is a failed capture, not a rescued dictation
        self.assertIsNone(salvage_repetition_loop("Hey. " + "uh " * 50, 12.0))


if __name__ == "__main__":
    unittest.main()

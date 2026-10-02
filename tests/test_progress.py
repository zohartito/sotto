"""Progress summary: trends from local History, never transcript text."""
import unittest

import progress

NOW = 1_800_000_000.0
DAY = progress.DAY


def row(days_ago, latency=None, corrected=False, replaced=0, learned=False, text="secret words"):
    entry = {"ts": NOW - days_ago * DAY, "text": text}
    if latency is not None:
        entry["latency"] = {"release_to_text_seconds": latency}
    if corrected:
        entry["correction"] = {"text": "fixed", "outcome": "corrected"}
    if replaced:
        entry["preprocessing"] = {"dictionary": {"rules": [{"heard": "a", "write": "b", "count": replaced}]}}
    if learned:
        entry["learning_state"] = "active"
    return entry


class ProgressTests(unittest.TestCase):
    def test_current_period_trends_against_the_previous_one(self):
        entries = [row(1, 0.4, replaced=2), row(2, 0.6, corrected=True, learned=True), row(3, 0.5),
                   row(9, 0.9, corrected=True), row(10, 1.1, corrected=True), row(30, 5.0)]
        summary = progress.summarize(entries, rules=4, now=NOW)
        self.assertEqual(summary["current"]["dictations"], 3)
        self.assertAlmostEqual(summary["current"]["median_latency"], 0.5)
        self.assertAlmostEqual(summary["previous"]["median_latency"], 1.0)
        self.assertAlmostEqual(summary["current"]["corrected_share"], 1 / 3)
        self.assertEqual(summary["previous"]["corrected_share"], 1.0)
        self.assertEqual(summary["current"]["replacements"], 2)
        self.assertEqual((summary["corrections"], summary["learning_samples"]), (3, 1))
        text = "\n".join(progress.lines(summary))
        self.assertIn("3 dictations", text)
        self.assertIn("ready 0.50s after release (median), was 1.00s", text)
        self.assertIn("33% of dictations (was 100%)", text)
        self.assertIn("4 rules · fixed 2 words", text)
        self.assertNotIn("secret", text)

    def test_accepting_a_transcript_as_correct_is_not_a_correction(self):
        accepted = row(1, 0.3)
        accepted["correction"] = {"text": "same", "outcome": "correct_as_is"}
        summary = progress.summarize([accepted, row(2, 0.3, corrected=True)], rules=0, now=NOW)
        self.assertEqual(summary["corrections"], 1)
        self.assertAlmostEqual(summary["current"]["corrected_share"], 0.5)

    def test_empty_history_and_rows_without_timing(self):
        summary = progress.summarize([row(1), {"text": "no timestamp"}], rules=1, now=NOW)
        text = progress.lines(summary)
        self.assertEqual(text[0], "Last 7 days: 1 dictation")
        self.assertEqual(len(progress.lines(progress.summarize([], rules=0, now=NOW))), 3)


class LifetimeTotalsTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / progress.TOTALS_NAME

    def test_seeded_once_from_inserted_dictations_only(self):
        import os
        entries = [{"text": "one two three", "duration": 2.0},
                   {"text": "held back", "duration": 9.0, "preprocessing": {"outcome": "suspect"}},
                   {"text": "also held", "duration": 9.0, "attempts": [{"provenance": "live_suspect"}]},
                   {"text": "   ", "duration": 1.0}]
        self.assertEqual(progress.ensure_totals(self.path, entries),
                         {"dictations": 1, "words": 3, "seconds": 2.0})
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
        self.assertEqual(progress.ensure_totals(self.path, entries * 5)["dictations"], 1)  # only once
        progress.record(self.path, text="four more words here", seconds=3.0)
        self.assertEqual(progress.load_totals(self.path), {"dictations": 2, "words": 7, "seconds": 5.0})
        self.assertNotIn("one two", self.path.read_text(encoding="utf-8"))  # counts, never text

    def test_time_saved_against_typing_and_the_menu_line(self):
        totals = {"dictations": 300, "words": 12_400, "seconds": 3_600.0}
        self.assertAlmostEqual(progress.saved_minutes(totals), 12_400 / 40 - 60)
        self.assertEqual(progress.totals_line(totals),
                         "All time: 12,400 words · about 4.2 hours saved vs typing at 40 wpm")
        self.assertEqual(progress.totals_line({"dictations": 2, "words": 30, "seconds": 10.0}),
                         "All time: 30 words · about 1 minute saved vs typing at 40 wpm")
        self.assertIsNone(progress.totals_line({"dictations": 0, "words": 0, "seconds": 0.0}))
        self.assertEqual(progress.saved_minutes({"dictations": 1, "words": 2, "seconds": 600.0}), 0.0)
        lines = progress.lines(progress.summarize([], rules=0, now=NOW), totals)
        self.assertTrue(lines[0].startswith("All time: 12,400 words"))
        self.assertEqual(progress.load_totals(self.path), {"dictations": 0, "words": 0, "seconds": 0.0})


if __name__ == "__main__":
    unittest.main()

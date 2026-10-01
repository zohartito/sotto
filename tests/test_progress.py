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


if __name__ == "__main__":
    unittest.main()

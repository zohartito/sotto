from __future__ import annotations

import unittest

from evaluation import (
    PairedPromotionRow,
    aggregate_results,
    character_error_rate,
    evaluate_sample,
    numeric_token_accuracy,
    required_term_accuracy,
    word_error_rate,
    holm_fixed_family,
    paired_bootstrap_statistics,
)


class EvaluationTests(unittest.TestCase):
    def test_english_and_hebrew_word_and_character_error_rate(self) -> None:
        self.assertEqual(word_error_rate("Hello, world!", "hello world"), 0)
        self.assertEqual(character_error_rate("Hello", "hello"), 0)
        self.assertEqual(word_error_rate("שלום עולם", "שלום עלם"), 0.5)
        self.assertGreater(character_error_rate("שלום", "שלו"), 0)

    def test_terms_and_numbers(self) -> None:
        accuracy, matched, total = required_term_accuracy(
            "Sotto met in תל אביב", ["Sotto", "תל אביב", "missing"]
        )
        self.assertEqual((accuracy, matched, total), (2 / 3, 2, 3))
        numeric, matched_numbers, total_numbers = numeric_token_accuracy("יש 12 תפוחים ו-3 אגסים", "יש 12 תפוחים")
        self.assertEqual((numeric, matched_numbers, total_numbers), (0.5, 1, 2))

    def test_empty_reference_is_explicitly_unscored(self) -> None:
        row = evaluate_sample("", "unexpected output")
        self.assertIsNone(row.wer)
        self.assertIsNone(row.cer)
        self.assertTrue(row.hallucination_proxy)

    def test_slices_have_counts_and_low_count_flags(self) -> None:
        rows = [
            evaluate_sample("hello", "hello", sample_id="1", language="en", tags=("short",)),
            evaluate_sample("שלום", "שלום", sample_id="2", language="he", tags=("short", "hebrew")),
        ]
        report = aggregate_results(rows)
        self.assertEqual(report["overall"]["count"], 2)
        self.assertTrue(report["overall"]["low_count"])
        self.assertEqual(report["by_language"]["he"]["count"], 1)
        self.assertTrue(report["by_tag"]["short"]["directional"])

    def test_exact_paired_bootstrap_and_fixed_holm_family(self) -> None:
        rows = [
            PairedPromotionRow("a", 2, 0, 2, 2, 0, 2),
            PairedPromotionRow("b", 1, 0, 2, 1, 0, 2),
        ]
        first = paired_bootstrap_statistics(rows, "generation", replicates=100)
        second = paired_bootstrap_statistics(rows, "generation", replicates=100)
        self.assertEqual(first, second)
        self.assertLess(first["upper_ci"], 0)
        self.assertEqual(holm_fixed_family([("b", .03), ("a", .01), ("c", .04)]), {"a": True, "b": False, "c": False})

    def test_paired_bootstrap_rejects_empty_reference_rows(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_statistics([PairedPromotionRow("empty", 0, 0, 0)], "generation", replicates=10)


if __name__ == "__main__":
    unittest.main()

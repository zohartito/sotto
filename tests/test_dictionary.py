"""Personal dictionary rules: parsing, rewriting, suggestions and storage."""
from pathlib import Path
import os
import stat
import tempfile
import unittest
import unittest.mock

import dictionary
from dictionary import Rule


class ParseTests(unittest.TestCase):
    def test_comments_blank_malformed_and_duplicate_lines_are_skipped(self):
        rules = dictionary.parse(
            "# comment\n\nwhisper flow => Wispr Flow\nno separator here\n => empty\n"
            "  sotto   =>   Sotto  \nWHISPER FLOW => ignored duplicate\n")
        self.assertEqual(rules, [Rule("whisper flow", "Wispr Flow"), Rule("sotto", "Sotto")])

    def test_overlong_sides_and_rule_cap(self):
        long_side = "x" * (dictionary.MAX_SIDE_CHARS + 1)
        self.assertEqual(dictionary.parse(f"{long_side} => y\na => {long_side}\n"), [])
        many = "\n".join(f"w{index} => W{index}" for index in range(dictionary.MAX_RULES + 10))
        self.assertEqual(len(dictionary.parse(many)), dictionary.MAX_RULES)


class ApplyTests(unittest.TestCase):
    rules = [Rule("whisper flow", "Wispr Flow"), Rule("open ai", "OpenAI"),
             Rule("open ai codex", "OpenAI Codex"), Rule("gonna", "going to"), Rule("привет", "Privet")]

    def test_case_insensitive_whole_word_and_whitespace_tolerant(self):
        text, receipt = dictionary.apply("I tried whisper  FLOW today, whisper flowing is not it.", self.rules)
        self.assertEqual(text, "I tried Wispr Flow today, whisper flowing is not it.")
        self.assertEqual(receipt, [{"heard": "whisper flow", "write": "Wispr Flow", "count": 1}])

    def test_longest_rule_wins_and_counts_every_use(self):
        text, receipt = dictionary.apply("open ai codex and open ai", self.rules)
        self.assertEqual(text, "OpenAI Codex and OpenAI")
        self.assertEqual({item["heard"]: item["count"] for item in receipt}, {"open ai": 1, "open ai codex": 1})

    def test_sentence_start_capitalisation_is_kept_for_lowercase_rules(self):
        self.assertEqual(dictionary.apply("Gonna ship it. gonna test it.", self.rules)[0],
                         "Going to ship it. going to test it.")

    def test_matching_and_duplicates_use_the_same_case_rule(self):
        rules = dictionary.parse("Straße => street\nSTRASSE => avenue\n")
        self.assertEqual(len(rules), 2)                     # distinct under the same comparison
        # a capital stays a capital
        self.assertEqual(dictionary.apply("die STRAßE und STRASSE", rules)[0], "die Street und Avenue")

    def test_non_latin_words_and_no_rules(self):
        self.assertEqual(dictionary.apply("я сказал привет всем", self.rules)[0], "я сказал Privet всем")
        self.assertEqual(dictionary.apply("unchanged", []), ("unchanged", []))
        self.assertEqual(dictionary.apply("", self.rules), ("", []))


class SuggestTests(unittest.TestCase):
    def test_substitutions_become_rules_but_sentence_case_and_common_words_do_not(self):
        suggestions = dictionary.suggest(
            "the whisper flow app on my iphone. Then the cat",
            "The Wispr Flow app on my iPhone. Then a cat")
        self.assertEqual(suggestions, [Rule("whisper flow", "Wispr Flow"), Rule("iphone", "iPhone")])

    def test_symbols_that_belong_to_the_term_survive(self):
        self.assertEqual(dictionary.suggest("I use see plus plus daily.", "I use C++ daily."),
                         [Rule("see plus plus", "C++")])
        self.assertEqual(dictionary.suggest("write it in c sharp", "write it in C#"),
                         [Rule("c sharp", "C#")])

    def test_existing_rules_insertions_and_limits(self):
        existing = [Rule("iphone", "iPhone")]
        self.assertEqual(dictionary.suggest("my iphone", "my iPhone", existing), [])
        self.assertEqual(dictionary.suggest("hello world", "hello brave new world"), [])
        before = " ".join(f"w{index}" for index in range(20))
        after = " ".join(f"W{index}x" if index % 2 else f"w{index}" for index in range(20))
        self.assertEqual(len(dictionary.suggest(before, after)), dictionary.MAX_SUGGESTIONS)


class StorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "nested" / "dictionary.txt"

    def test_add_creates_a_private_file_with_instructions_and_skips_known_rules(self):
        self.assertEqual(dictionary.add([Rule("sotto", "Sotto"), Rule("x => y", "bad")], self.path), 1)
        self.assertEqual(dictionary.add([Rule("SOTTO", "again"), Rule("iphone", "iPhone")], self.path), 1)
        text = self.path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# Sotto dictionary"))
        self.assertEqual(dictionary.parse(text), [Rule("sotto", "Sotto"), Rule("iphone", "iPhone")])
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["dictionary.txt"])

    def test_a_heard_side_the_file_would_read_as_a_comment_is_refused(self):
        """N31: "#sotto => Sotto" is a comment line to parse(), so the rule was
        written, reported as added, and never applied."""
        self.assertEqual(dictionary.suggest("tag #sotto here", "tag Sotto here"), [])
        dictionary.add([Rule("sotto", "Sotto")], self.path)
        before = self.path.read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r"#sotto.*comment"):
            dictionary.add([Rule("iphone", "iPhone"), Rule("  #sotto", "Sotto")], self.path)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before, "nothing is written")

    def test_load_rereads_only_after_a_change_and_missing_file_is_empty(self):
        self.assertEqual(dictionary.load(self.path), [])
        dictionary.ensure_file(self.path)
        self.assertEqual(dictionary.load(self.path), [])
        dictionary.add([Rule("sotto", "Sotto")], self.path)
        self.assertEqual(dictionary.load(self.path), [Rule("sotto", "Sotto")])
        self.path.write_text(self.path.read_text(encoding="utf-8") + "mac os => macOS\n", encoding="utf-8")
        self.assertEqual(dictionary.load(self.path), [Rule("sotto", "Sotto"), Rule("mac os", "macOS")])


class DeliveryTests(unittest.TestCase):
    """The app applies rules to deliverable text and keeps the raw output."""

    def test_rules_apply_with_a_receipt_and_a_broken_dictionary_never_costs_text(self):
        import sotto
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dictionary.txt"
            dictionary.add([Rule("whisper flow", "Wispr Flow")], path)
            preprocessing = {}
            with unittest.mock.patch.object(dictionary, "DICTIONARY_PATH", path):
                self.assertEqual(sotto.apply_personal_dictionary("try whisper flow", preprocessing),
                                 "try Wispr Flow")
                self.assertEqual(preprocessing["dictionary"]["asr_text"], "try whisper flow")
                untouched = {}
                self.assertEqual(sotto.apply_personal_dictionary("nothing here", untouched), "nothing here")
                self.assertEqual(untouched, {})
            with unittest.mock.patch.object(dictionary, "apply", side_effect=RuntimeError("boom")):
                self.assertEqual(sotto.apply_personal_dictionary("keep me", {}), "keep me")


if __name__ == "__main__":
    unittest.main()

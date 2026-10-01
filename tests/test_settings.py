"""Shared settings store: defaults, validation and private atomic saves."""
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

import settings


class SettingsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "state" / "settings.json"

    def test_missing_or_corrupt_file_gives_defaults(self):
        self.assertEqual(settings.load(self.path), settings.DEFAULTS)
        self.path.parent.mkdir()
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(settings.load(self.path), settings.DEFAULTS)

    def test_save_validates_persists_privately_and_merges(self):
        result = settings.save({"trigger": "right-cmd", "languages": ["EN", "pt", "en", "x1"]}, self.path)
        self.assertEqual(result["trigger"], "right-cmd")
        self.assertEqual(result["languages"], ["en", "pt"])
        settings.save({"insert_mode": "type"}, self.path)
        loaded = settings.load(self.path)
        self.assertEqual((loaded["trigger"], loaded["insert_mode"]), ("right-cmd", "type"))
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["settings.json"])

    def test_invalid_or_unknown_values_are_rejected_and_ignored_on_load(self):
        with self.assertRaises(ValueError):
            settings.save({"trigger": "caps-lock"}, self.path)
        with self.assertRaises(ValueError):
            settings.save({"volume": 11}, self.path)
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text(json.dumps({"spacing": "weird", "speed": "fast", "extra": 1}), encoding="utf-8")
        loaded = settings.load(self.path)
        self.assertEqual((loaded["spacing"], loaded["speed"]), (settings.DEFAULTS["spacing"], "fast"))
        self.assertNotIn("extra", loaded)


if __name__ == "__main__":
    unittest.main()

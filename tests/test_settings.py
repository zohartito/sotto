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

    def test_save_never_resets_settings_it_could_not_read(self):
        """N23: save() started from load(), which falls back to the defaults
        when the file cannot be read, so one change reset every other setting."""
        from unittest import mock
        settings.save({"trigger": "right-cmd", "insert_mode": "type"}, self.path)
        kept = self.path.read_bytes()
        for name, damage in {"damaged": lambda: self.path.write_text('{"trigger": "right-cmd", "ins',
                                                                     encoding="utf-8"),
                             "not an object": lambda: self.path.write_text("[]", encoding="utf-8"),
                             "locked": lambda: None}.items():
            with self.subTest(name):
                self.path.write_bytes(kept)
                damage()
                before = self.path.read_bytes()

                def locked(path, *args, real_read=Path.read_text, is_locked=name == "locked", **kwargs):
                    if is_locked and Path(path) == self.path:
                        raise PermissionError(13, "The process cannot access the file", str(path))
                    return real_read(path, *args, **kwargs)

                with mock.patch.object(Path, "read_text", locked):
                    with self.assertRaisesRegex(ValueError, "could not be read"):
                        settings.save({"speed": "fast"}, self.path)
                self.assertEqual(self.path.read_bytes(), before, "the file is left as it was")
                self.assertEqual(sorted(p.name for p in self.path.parent.iterdir()), ["settings.json"])
        self.path.unlink()
        self.assertEqual(settings.save({"speed": "fast"}, self.path)["speed"], "fast")  # missing: defaults


class IgnoredValueReportTests(unittest.TestCase):
    """An ignored setting is reported once in the log, never silently."""

    def test_an_invalid_value_is_reported_once_per_setting(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(json.dumps({"speed": "warp", "spacing": "smart"}), encoding="utf-8")
            settings._reported.clear()
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                self.assertEqual(settings.load(path)["speed"], settings.DEFAULTS["speed"])
                settings.load(path)
            lines = errors.getvalue().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertIn("speed", lines[0])
            self.assertNotIn("warp", lines[0])   # values are not logged

    def test_an_unreadable_file_is_reported(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text("{not json", encoding="utf-8")
            settings._reported.clear()
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                settings.load(path)
            self.assertIn("could not be read", errors.getvalue())


if __name__ == "__main__":
    unittest.main()

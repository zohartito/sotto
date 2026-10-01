"""win_startup / win_launch: the per-user login entry and the pythonw launcher.

Registry tests use a throwaway key under HKCU\\Software\\sotto-tests, never the
real Run key, and remove it afterwards.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
import uuid

if sys.platform == "win32":
    import winreg
    import win_launch
    import win_startup


@unittest.skipUnless(sys.platform == "win32", "Windows-only module")
class RunKeyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = rf"Software\sotto-tests\{uuid.uuid4().hex}"
        self.key = self.parent + r"\Run"

    def tearDown(self) -> None:
        for key in (self.key, self.parent, r"Software\sotto-tests"):
            try:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key)
            except OSError:
                pass  # absent, or another test's key still lives there

    def test_enable_and_disable_add_and_remove_exactly_one_value(self) -> None:
        command = win_startup.launch_command(Path(r"C:\data dir"), Path(r"C:\data dir\huggingface"),
                                             python=Path(r"C:\py\pythonw.exe"),
                                             launcher=Path(r"C:\sotto\win_launch.py"))
        self.assertFalse(win_startup.enabled(key=self.key))
        win_startup.set_enabled(True, command, key=self.key)
        self.assertEqual(win_startup.current(key=self.key), command)
        self.assertTrue(win_startup.enabled(key=self.key))
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.key) as handle:
            self.assertEqual(winreg.QueryInfoKey(handle)[1], 1)  # one value, nothing else
        win_startup.set_enabled(False, command, key=self.key)
        self.assertFalse(win_startup.enabled(key=self.key))
        win_startup.set_enabled(False, command, key=self.key)  # already off: fine

    def test_command_quotes_paths_and_fits_the_run_limit(self) -> None:
        command = win_startup.launch_command(Path(r"C:\data dir"), Path(r"D:\models"),
                                             python=Path(r"C:\py\pythonw.exe"),
                                             launcher=Path(r"C:\sotto\win_launch.py"))
        self.assertEqual(command, r'C:\py\pythonw.exe C:\sotto\win_launch.py --data-dir "C:\data dir" '
                                  r"--hf-home D:\models")
        with self.assertRaises(ValueError):
            win_startup.set_enabled(True, "x" * 300, key=self.key)
        self.assertFalse(win_startup.enabled(key=self.key))

    def test_gui_python_prefers_pythonw_next_to_the_interpreter(self) -> None:
        python = win_startup.gui_python(sys.executable)
        expected = Path(sys.executable).with_name("pythonw.exe")
        self.assertEqual(python, expected if expected.is_file() else Path(sys.executable))
        self.assertEqual(win_startup.LAUNCHER.name, "win_launch.py")
        self.assertTrue(win_startup.LAUNCHER.is_file())


@unittest.skipUnless(sys.platform == "win32", "Windows-only module")
class LauncherTest(unittest.TestCase):
    def test_paths_are_split_from_the_app_arguments(self) -> None:
        self.assertEqual(win_launch.split_paths(["--data-dir", "C:/d", "--trigger", "left-ctrl"]),
                         ("C:/d", None, ["--trigger", "left-ctrl"]))
        self.assertEqual(win_launch.split_paths(["--hf-home", "D:/m", "--data-dir", "C:/d"]),
                         ("C:/d", "D:/m", []))
        with self.assertRaises(SystemExit):
            win_launch.split_paths(["--data-dir"])


if __name__ == "__main__":
    unittest.main()

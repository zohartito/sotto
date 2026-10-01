"""scripts/install-windows.ps1: checks, the -WhatIf plan and a scoped uninstall.

Every run points the venv, data folder, Start Menu folder and login key at a
temporary location; no packages are downloaded and nothing real is touched.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "install-windows.ps1"


def _powershell() -> str | None:
    return shutil.which("powershell.exe") if sys.platform == "win32" else None


@unittest.skipUnless(_powershell(), "Windows PowerShell is Windows-only")
class InstallerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="sotto-installer-")
        root = Path(self._tmp.name)
        self.venv, self.data, self.menu = root / "venv", root / "data", root / "menu"
        self.run_key = rf"HKCU:\Software\sotto-tests\{uuid.uuid4().hex}"

    def tearDown(self) -> None:
        subprocess.run([_powershell(), "-NoProfile", "-Command",
                        f"Remove-Item -LiteralPath '{self.run_key}' -Recurse -ErrorAction SilentlyContinue"],
                       capture_output=True, timeout=60)
        self._tmp.cleanup()

    def ps(self, *args: str, script: Path = SCRIPT) -> subprocess.CompletedProcess:
        command = [_powershell(), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                   "-File", str(script), "-VenvDir", str(self.venv), "-DataDir", str(self.data),
                   "-ShortcutDir", str(self.menu), "-RunKey", self.run_key, *args]
        return subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=300)

    def psc(self, code: str) -> str:
        result = subprocess.run([_powershell(), "-NoProfile", "-Command", code], capture_output=True,
                                text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_whatif_checks_and_plans_every_step_without_changing_anything(self) -> None:
        result = self.ps("-WhatIf", "-Cpu", "-Python", sys.executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        out = result.stdout
        self.assertIn("OK Windows x64", out)
        self.assertIn(f"OK Python {sys.version_info.major}.{sys.version_info.minor}", out)
        self.assertIn("OK dependency set: cpu (-Cpu)", out)
        for step in ("Create Python environment", "Install pinned packages from requirements-alpha-windows.txt",
                     "Create the alpha data folder", "Download the pinned speech model",
                     'Create Start Menu shortcut "Sotto"'):
            self.assertIn(step, out)
        skipped = self.ps("-WhatIf", "-Cpu", "-SkipSetup", "-Python", sys.executable)
        self.assertEqual(skipped.returncode, 0, skipped.stdout + skipped.stderr)
        self.assertNotIn("Download the pinned speech model", skipped.stdout)
        self.assertEqual(sorted(os.listdir(self._tmp.name)), [], "WhatIf created something")

    def test_bad_choices_and_missing_python_fail_with_instructions(self) -> None:
        both = self.ps("-WhatIf", "-Cpu", "-Cuda", "-Python", sys.executable)
        self.assertEqual(both.returncode, 1)
        self.assertIn("at most one of -Cpu and -Cuda", both.stdout)
        missing = self.ps("-WhatIf", "-Cpu", "-Python", str(Path(os.environ["SystemRoot"]) / "System32" / "where.exe"))
        self.assertEqual(missing.returncode, 1)
        self.assertIn("Python 3.13 was not found", missing.stdout)
        self.assertIn("python.org", missing.stdout)
        self.assertEqual(sorted(os.listdir(self._tmp.name)), [])

    def test_an_existing_folder_is_never_adopted_as_the_environment(self) -> None:
        self.venv.mkdir()
        (self.venv / "notes.txt").write_text("mine", encoding="utf-8")
        result = self.ps("-Cpu", "-Python", sys.executable)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("already exists but is not a Windows Python environment", result.stdout)
        self.assertEqual(sorted(p.name for p in self.venv.iterdir()), ["notes.txt"])
        self.assertFalse(self.data.exists() or self.menu.exists())

    def test_uninstall_refuses_while_sotto_runs_from_the_environment(self) -> None:
        scripts = self.venv / "Scripts"
        scripts.mkdir(parents=True)
        (self.venv / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
        (self.venv / "sotto-install.json").write_text(json.dumps({"schema": 1, "created_venv": True}),
                                                      encoding="utf-8")
        stand_in = scripts / "ping.exe"  # any program running from the venv, like pythonw
        shutil.copy(Path(os.environ["SystemRoot"]) / "System32" / "PING.EXE", stand_in)
        running = subprocess.Popen([str(stand_in), "-n", "60", "127.0.0.1"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            result = self.ps("-Uninstall")
        finally:
            running.kill()
            running.wait(10)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("still running", result.stdout)
        self.assertTrue(stand_in.exists() and (self.venv / "pyvenv.cfg").exists(), "nothing removed")

    def test_uninstall_removes_only_what_the_manifest_and_this_launcher_own(self) -> None:
        self.venv.mkdir()
        (self.venv / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
        self.data.mkdir()
        (self.data / "history.jsonl").write_text("", encoding="utf-8")
        self.menu.mkdir()
        launcher = ROOT / "win_launch.py"
        (self.venv / "sotto-install.json").write_text(json.dumps({
            "schema": 1, "created_venv": True, "created_data": True, "data_dir": str(self.data)}),
            encoding="utf-8")
        make = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{path}'); "
                "$s.TargetPath = '{target}'; $s.Arguments = '{args}'; $s.Save()")
        ours, other = self.menu / "Sotto.lnk", self.menu / "Other.lnk"
        self.psc(make.format(path=ours, target=sys.executable, args=f'"{launcher}" --data-dir "{self.data}"'))
        self.psc(make.format(path=other, target=sys.executable, args="elsewhere.py"))
        self.psc(f"New-Item -Path '{self.run_key}' -Force | Out-Null; "
                 f"New-ItemProperty -Path '{self.run_key}' -Name Sotto -Value '\"py\" \"{launcher}\"' | Out-Null")

        dry = self.ps("-Uninstall", "-WhatIf")
        self.assertEqual(dry.returncode, 0, dry.stdout + dry.stderr)
        self.assertTrue(ours.exists() and self.venv.exists() and self.data.exists())

        result = self.ps("-Uninstall")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(ours.exists())
        self.assertTrue(other.exists(), "a shortcut to something else is never removed")
        self.assertFalse(self.venv.exists())
        self.assertTrue((self.data / "history.jsonl").exists(), "data stays without -RemoveData")
        self.assertIn("add -RemoveData", result.stdout)
        self.assertEqual(self.psc(f"(Get-ItemProperty -Path '{self.run_key}').PSObject.Properties.Name "
                                  "-contains 'Sotto'"), "False")

        # The manifest went with the venv: -RemoveData now refuses to guess.
        kept = self.ps("-Uninstall", "-RemoveData")
        self.assertEqual(kept.returncode, 0, kept.stdout + kept.stderr)
        self.assertTrue(self.data.exists())
        self.venv.mkdir()
        (self.venv / "sotto-install.json").write_text(json.dumps({
            "schema": 1, "created_venv": False, "created_data": True, "data_dir": str(self.data)}),
            encoding="utf-8")
        removed = self.ps("-Uninstall", "-RemoveData")
        self.assertEqual(removed.returncode, 0, removed.stdout + removed.stderr)
        self.assertFalse(self.data.exists())
        self.assertTrue(self.venv.exists(), "a venv the script did not create stays")


if __name__ == "__main__":
    unittest.main()

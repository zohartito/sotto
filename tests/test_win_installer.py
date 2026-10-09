"""scripts/install-windows.ps1 and update-windows.ps1: checks, the -WhatIf
plan, a scoped uninstall, and an update that never strands new source on old
packages.

Every run points the venv, data folder, Start Menu folder and login key at a
temporary location; no packages are downloaded and nothing real is touched.
The update tests run copies of both scripts whose Start Menu folder is
rewritten to a temporary one (they refuse to run if that rewrite fails).
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
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

    def shortcut(self, arguments: str) -> Path:
        """A Start Menu "Sotto" entry with these arguments, as another copy's installer makes it."""
        self.menu.mkdir(exist_ok=True)
        path = self.menu / "Sotto.lnk"
        self.psc(f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{path}'); "
                 f"$s.TargetPath = '{sys.executable}'; $s.Arguments = '{arguments}'; $s.Save()")
        return path

    def shortcut_arguments(self, path: Path) -> str:
        return self.psc(f"(New-Object -ComObject WScript.Shell).CreateShortcut('{path}').Arguments")

    @contextmanager
    def running_from_venv(self):
        """A program running from the environment, like Sotto's pythonw."""
        scripts = self.venv / "Scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        stand_in = scripts / "ping.exe"
        shutil.copy(Path(os.environ["SystemRoot"]) / "System32" / "PING.EXE", stand_in)
        running = subprocess.Popen([str(stand_in), "-n", "60", "127.0.0.1"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            yield
        finally:
            running.kill()
            running.wait(10)

    def test_a_shortcut_to_another_copy_is_never_replaced(self) -> None:
        # [F11] Re-running an installer used to repoint another copy's entry
        # (and its data folder) at this copy without a word.
        other = Path(self._tmp.name) / "other-copy"
        other.mkdir()
        (other / "win_launch.py").write_text("", encoding="utf-8")
        theirs = f'"{other / "win_launch.py"}" --data-dir "C:\\other-data"'
        link = self.shortcut(theirs)
        result = self.ps("-WhatIf", "-Cpu", "-Python", sys.executable)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("already starts another copy of Sotto", result.stdout)
        self.assertNotIn("Install pinned packages", result.stdout, "refused before planning any change")
        self.assertEqual(self.shortcut_arguments(link), theirs)
        # An entry whose copy was deleted is replaced; one for this copy is kept up to date.
        for arguments in (f'"{Path(self._tmp.name) / "gone" / "win_launch.py"}" --data-dir "C:\\x"',
                          f'"{ROOT / "win_launch.py"}" --data-dir "{self.data}"'):
            self.shortcut(arguments)
            planned = self.ps("-WhatIf", "-Cpu", "-Python", sys.executable)
            self.assertEqual(planned.returncode, 0, planned.stdout + planned.stderr)
            self.assertIn('Create Start Menu shortcut "Sotto"', planned.stdout)

    def test_install_refuses_while_sotto_runs_from_the_environment(self) -> None:
        # [F33] Installing under a running Sotto replaced its packages in use.
        with self.running_from_venv():
            result = self.ps("-WhatIf", "-Cpu", "-Python", sys.executable)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("still running", result.stdout)
        self.assertNotIn("Install pinned packages", result.stdout)

    def test_a_rerun_keeps_the_dependency_set_recorded_at_install(self) -> None:
        # [F34] Every update re-detected the GPU, so a deliberate -Cpu install
        # became CUDA (and the other way round). Both are checked, so one of them
        # differs from what this PC would detect.
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(self.venv)], check=True,
                       capture_output=True, timeout=300)
        for recorded, requirements in (("cpu", "requirements-alpha-windows.txt"),
                                       ("cuda", "requirements-alpha-windows-cuda.txt")):
            (self.venv / "sotto-install.json").write_text(json.dumps({
                "schema": 1, "created_venv": True, "created_data": False, "data_dir": str(self.data),
                "flavor": recorded}), encoding="utf-8")
            result = self.ps("-WhatIf", "-SkipSetup", "-Python", sys.executable)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(f"OK dependency set: {recorded} (recorded at install", result.stdout)
            self.assertIn(f"Install pinned packages from {requirements}", result.stdout)
        switched = self.ps("-WhatIf", "-SkipSetup", "-Cpu", "-Python", sys.executable)
        self.assertIn("OK dependency set: cpu (-Cpu)", switched.stdout, "an explicit choice still wins")

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


def _wheel(folder: Path, name: str, version: str, requires=()) -> Path:
    """A minimal pure-Python wheel, so pip can install and replace packages offline."""
    import zipfile
    dist = f"{name}-{version}.dist-info"
    files = {
        f"{name}/__init__.py": f"VERSION = '{version}'\n",
        f"{dist}/METADATA": (f"Metadata-Version: 2.1\nName: {name.replace('_', '-')}\nVersion: {version}\n"
                             + "".join(f"Requires-Dist: {requirement}\n" for requirement in requires)),
        f"{dist}/WHEEL": "Wheel-Version: 1.0\nGenerator: sotto-tests\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    path = folder / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for member, text in files.items():
            archive.writestr(member, text)
        archive.writestr(f"{dist}/RECORD", "".join(f"{member},,\n" for member in files) + f"{dist}/RECORD,,\n")
    return path


@unittest.skipUnless(_powershell() and shutil.which("git"), "Windows PowerShell and git")
class UpdateScriptTest(unittest.TestCase):
    """update-windows.ps1 on a private copy: a local bare 'GitHub', a venv and
    a Start Menu folder in a temporary directory, pip offline (PIP_NO_INDEX)."""

    REQUIREMENTS = {
        "requirements-alpha-windows.txt": "-c constraints-alpha-windows.txt\n",
        "requirements-alpha-windows-cuda.txt":
            "-r requirements-alpha-windows.txt\n-c constraints-alpha-windows-cuda.txt\n",
        "constraints-alpha-windows.txt": "# none\n",
        "constraints-alpha-windows-cuda.txt": "# none\n",
    }
    # The stand-in launcher only records that it was started.
    LAUNCHER = ("import pathlib, sys\n"
                "data = pathlib.Path(sys.argv[sys.argv.index('--data-dir') + 1])\n"
                "(data / 'launched.txt').write_text('started', encoding='utf-8')\n")

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="sotto-update-")
        base = Path(self._tmp.name)
        self.copy, self.remote, self.upstream = base / "copy", base / "remote.git", base / "upstream"
        self.venv, self.data, self.menu = base / "venv", base / "data", base / "menu"
        self.copy.mkdir()
        (self.copy / "scripts").mkdir()
        for name in ("update-windows.ps1", "install-windows.ps1"):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            text = text.replace("[Environment]::GetFolderPath('Programs')", f"'{self.menu}'")
            if "GetFolderPath" in text or f"'{self.menu}'" not in text:
                self.fail(f"refusing to run {name}: its Start Menu folder could not be redirected")
            (self.copy / "scripts" / name).write_text(text, encoding="utf-8")
        for name, text in self.REQUIREMENTS.items():
            (self.copy / name).write_text(text, encoding="utf-8")
        (self.copy / "win_launch.py").write_text(self.LAUNCHER, encoding="utf-8")
        self.git(base, "init", "--quiet", "--bare", "-b", "main", str(self.remote))
        self.git(self.copy, "init", "--quiet", "-b", "main")
        self.git(self.copy, "add", "-A")
        self.git(self.copy, "commit", "--quiet", "-m", "first")
        self.git(self.copy, "remote", "add", "origin", str(self.remote))
        self.git(self.copy, "push", "--quiet", "-u", "origin", "main")
        self.git(base, "clone", "--quiet", str(self.remote), str(self.upstream))
        subprocess.run([sys.executable, "-m", "venv", str(self.venv)], check=True, capture_output=True,
                       timeout=600)
        (self.venv / "sotto-install.json").write_text(json.dumps({
            "schema": 1, "created_venv": True, "created_data": False, "data_dir": str(self.data),
            "flavor": "cpu"}), encoding="utf-8")
        self.menu.mkdir()
        launcher = self.copy / "win_launch.py"
        subprocess.run([_powershell(), "-NoProfile", "-Command",
                        f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{self.menu / 'Sotto.lnk'}'); "
                        f"$s.TargetPath = '{self.venv / 'Scripts' / 'pythonw.exe'}'; "
                        f"$s.Arguments = '\"{launcher}\" --data-dir \"{self.data}\"'; $s.Save()"],
                       check=True, capture_output=True, timeout=60)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def git(self, cwd: Path, *args: str) -> str:
        return subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
                              cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    def publish(self, name: str, text: str) -> str:
        (self.upstream / name).write_text(text, encoding="utf-8")
        self.git(self.upstream, "add", name)
        self.git(self.upstream, "commit", "--quiet", "-m", f"change {name}")
        self.git(self.upstream, "push", "--quiet", "origin", "HEAD:main")
        return self.git(self.upstream, "rev-parse", "HEAD")

    def update(self, **extra_env: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1", **extra_env)
        result = subprocess.run([_powershell(), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                                 "-File", str(self.copy / "scripts" / "update-windows.ps1"),
                                 "-DataDir", str(self.data), "-VenvDir", str(self.venv), "-WaitSeconds", "5"],
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                env=env, timeout=600)
        deadline = time.monotonic() + 20  # Sotto is started detached
        while not (self.data / "launched.txt").exists() and time.monotonic() < deadline:
            time.sleep(0.2)
        time.sleep(1.0)  # let the stand-in exit before its venv is deleted
        return result

    def status(self) -> str:
        return (self.data / "update-status.txt").read_text(encoding="utf-8-sig").strip()

    def log_bytes(self) -> bytes:
        return (self.data / "update.log").read_bytes()

    def test_a_failed_package_install_leaves_this_copy_as_it_was(self) -> None:
        # [F2w] The update used to pull first: new source on old packages, and
        # Sotto started on it as if nothing happened.
        before = self.git(self.copy, "rev-parse", "HEAD")
        self.publish("requirements-alpha-windows.txt",
                     self.REQUIREMENTS["requirements-alpha-windows.txt"] + "sotto-test-missing-package==1.0\n")
        result = self.update()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(self.git(self.copy, "rev-parse", "HEAD"), before, "the source was not switched")
        self.assertNotIn("sotto-test-missing-package", (self.copy / "requirements-alpha-windows.txt").read_text())
        status = self.status()
        self.assertTrue(status.startswith("failed "), status)
        self.assertIn("previous packages are back", status)
        self.assertIn(b"restoring the previous packages", self.log_bytes())
        self.assertNotIn(b"\x00", self.log_bytes(), "update.log stays UTF-8 (no UTF-16 appended)")
        self.assertTrue((self.data / "launched.txt").exists(), "the unchanged copy starts again to report it")

    def test_a_package_set_that_breaks_the_environment_is_rolled_back(self) -> None:
        # [F2w] pip cannot undo an install: here the update's pin upgrades a
        # package another installed one cannot use (pip check fails after
        # pip already upgraded it). The previously installed versions come back.
        wheels = Path(self._tmp.name) / "wheels"
        wheels.mkdir()
        for name, version, requires in (("sotto_test_core", "1.0", ()), ("sotto_test_core", "2.0", ()),
                                        ("sotto_test_app", "1.0", ("sotto-test-core<2",))):
            _wheel(wheels, name, version, requires)
        python = self.venv / "Scripts" / "python.exe"
        subprocess.run([str(python), "-m", "pip", "install", "--quiet", "--no-index", "--find-links",
                        str(wheels), "sotto-test-app==1.0"], check=True, capture_output=True, timeout=300)
        before = self.git(self.copy, "rev-parse", "HEAD")
        self.publish("requirements-alpha-windows.txt",
                     self.REQUIREMENTS["requirements-alpha-windows.txt"] + "sotto-test-core==2.0\n")
        result = self.update(PIP_FIND_LINKS=str(wheels))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        log = self.log_bytes().decode("utf-8", "replace")
        self.assertIn("Successfully installed sotto-test-core-2.0", log, "pip did upgrade it first")
        installed = subprocess.run([str(python), "-m", "pip", "freeze"], capture_output=True, text=True,
                                   timeout=120).stdout.split()
        self.assertEqual(sorted(installed), ["sotto-test-app==1.0", "sotto-test-core==1.0"], log)
        self.assertEqual(self.git(self.copy, "rev-parse", "HEAD"), before)
        self.assertIn("previous packages are back", self.status())

    def test_an_update_installs_packages_then_switches_and_keeps_the_recorded_set(self) -> None:
        # [F2w][F34] Packages first, then the source; the CPU set recorded at
        # install stays CPU even on a PC with an NVIDIA GPU.
        upstream = self.publish("CHANGES.txt", "new version\n")
        result = self.update()
        self.assertEqual(result.returncode, 0,
                         result.stdout + result.stderr + self.log_bytes().decode("utf-8", "replace"))
        self.assertEqual(self.git(self.copy, "rev-parse", "HEAD"), upstream)
        status = self.status()
        self.assertTrue(status.startswith("ok ") and upstream.startswith(status[3:]), status)
        record = json.loads((self.venv / "sotto-install.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(record["flavor"], "cpu")
        log = self.log_bytes()
        self.assertNotIn(b"\x00", log)
        self.assertLess(log.index(b"== packages"), log.index(b"== source"), "packages before the source")
        self.assertTrue((self.data / "launched.txt").exists())


if __name__ == "__main__":
    unittest.main()

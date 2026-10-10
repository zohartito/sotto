"""One-line installers: download (or update) a git copy, then hand off to the platform installer."""
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
HAVE_CLANG = sys.platform == "darwin" and subprocess.run(
    ["/usr/bin/xcrun", "--find", "clang"], capture_output=True).returncode == 0

# Stands in for Python 3.12 and the venv in install-mac.sh: answers its checks;
# with PIP_FAILS=1 every package install from a requirements file fails.
FAKE_VENV_PYTHON = """#!/bin/bash
case "$1 $2" in
    "-c import platform"*) echo "${FAKE_ARCH:-arm64}"; exit 0 ;;
    "-c import sys"*) echo 3.12; exit 0 ;;
esac
if [ "$1 $2 $3" = "-m pip install" ]; then
    for arg in "$@"; do [ "$arg" = -r ] && [ "${PIP_FAILS:-0}" = 1 ] && exit 1; done
fi
exit 0
"""


def bare_copy_of_this_checkout(folder: Path) -> Path:
    """A 'GitHub' whose main branch is the commit checked out here. CI checks
    out a detached, shallow commit, so push it rather than clone the branches."""
    remote = folder / "remote.git"
    subprocess.run(["git", "init", "--quiet", "--bare", str(remote)], check=True)
    subprocess.run(["git", "-C", str(remote), "config", "receive.shallowUpdate", "true"], check=True)
    subprocess.run(["git", "-C", str(ROOT), "push", "--quiet", str(remote), "HEAD:refs/heads/main"], check=True)
    subprocess.run(["git", "-C", str(remote), "symbolic-ref", "HEAD", "refs/heads/main"], check=True)
    return remote


def long_and_short(path: Path) -> tuple[str, str]:
    """An existing Windows path's long and 8.3 short names (GitHub's runner
    gives TEMP as C:\\Users\\RUNNER~1\\...); equal where the volume keeps none."""
    import ctypes
    names = []
    for convert in (ctypes.windll.kernel32.GetLongPathNameW, ctypes.windll.kernel32.GetShortPathNameW):
        buffer = ctypes.create_unicode_buffer(32768)
        names.append(buffer.value if convert(str(path), buffer, len(buffer)) else str(path))
    return names[0], names[1]


def windows_get_script(folder: Path) -> tuple[Path, Path]:
    """A copy of get.ps1 whose Start Menu folder is a temporary one, so a
    test never reads or starts the real "Sotto" entry."""
    menu = folder / "menu"
    menu.mkdir()
    text = (ROOT / "scripts" / "get.ps1").read_text(encoding="utf-8")
    text = text.replace("[Environment]::GetFolderPath('Programs')", f"'{menu}'")
    if "GetFolderPath" in text or f"'{menu}'" not in text:
        raise AssertionError("refusing to run get.ps1: its Start Menu folder could not be redirected")
    script = folder / "get.ps1"
    script.write_text(text, encoding="utf-8")
    return script, menu


class ScriptShapeTests(unittest.TestCase):
    def test_mac_script_parses_runs_from_main_and_is_executable(self):
        script = ROOT / "scripts" / "get.sh"
        if sys.platform != "win32":
            self.assertTrue(os.access(script, os.X_OK))
            self.assertEqual(subprocess.run(["/bin/bash", "-n", str(script)]).returncode, 0)
        text = script.read_text(encoding="utf-8")
        self.assertTrue(text.rstrip().endswith('main "$@"'), "a partial download must not run anything")
        self.assertIn("raw.githubusercontent.com/zohartito/sotto/main/scripts/get.sh", text)

    def test_windows_script_never_exits_the_users_shell(self):
        text = (ROOT / "scripts" / "get.ps1").read_text(encoding="utf-8")
        code = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
        for line in code:
            if re.search(r"(?i)\bexit\b", line):
                # [N41] Only a run as a file exits, with the failure's code.
                self.assertIn("${function:Install-Sotto}.File", line,
                              "iex runs in the user's session; exit would close it")
        self.assertEqual([line for line in code if line.startswith("Install-Sotto")], ["Install-Sotto"],
                         "one call, after every definition")


@unittest.skipUnless(sys.platform == "darwin" and (ROOT / ".git").exists() and shutil.which("git"),
                     "macOS git checkout")
class MacInstallerTests(unittest.TestCase):
    def run_piped(self, env):
        # Exactly like `curl … | bash`: the script arrives on stdin.
        return subprocess.run(["/bin/bash"], input=(ROOT / "scripts" / "get.sh").read_text(encoding="utf-8"),
                              env=env, capture_output=True, text=True, timeout=120)

    def test_downloads_then_updates_and_refuses_a_foreign_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            dest = folder / "sotto"
            env = dict(os.environ, SOTTO_REPO=str(bare_copy_of_this_checkout(folder)), SOTTO_SOURCE=str(dest),
                       SOTTO_PYTHON=sys.executable, SOTTO_GET_DRY_RUN="1")
            first = self.run_piped(env)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn("Downloading Sotto", first.stdout)
            self.assertIn(f"would run {dest}/scripts/install-mac.sh", first.stdout)
            self.assertTrue((dest / "scripts" / "install-mac.sh").is_file())
            again = self.run_piped(env)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertIn("Updating Sotto", again.stdout)
            foreign = folder / "notes"
            foreign.mkdir()
            refused = self.run_piped(dict(env, SOTTO_SOURCE=str(foreign)))
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("is not a copy of Sotto", refused.stderr)

    def test_a_git_copy_of_another_project_is_never_pulled_or_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            other_remote = folder / "other.git"
            subprocess.run(["git", "init", "--quiet", "--bare", str(other_remote)], check=True)
            dest = folder / "sotto"
            subprocess.run(["git", "clone", "--quiet", str(other_remote), str(dest)], check=True,
                           capture_output=True)
            env = dict(os.environ, SOTTO_REPO=str(bare_copy_of_this_checkout(folder)), SOTTO_SOURCE=str(dest),
                       SOTTO_PYTHON=sys.executable, SOTTO_GET_DRY_RUN="1")
            refused = self.run_piped(env)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("not Sotto", refused.stderr)
            self.assertNotIn("would run", refused.stdout)

    def git(self, cwd: Path, *args: str) -> str:
        return subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
                              cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    def installed_copy(self, folder: Path) -> tuple[dict, Path, str]:
        """A copy downloaded by get.sh, with a stand-in venv, and a newer version on 'GitHub'."""
        dest = folder / "sotto"
        remote = bare_copy_of_this_checkout(folder)
        env = dict(os.environ, SOTTO_REPO=str(remote), SOTTO_SOURCE=str(dest), SOTTO_PYTHON=sys.executable,
                   SOTTO_GET_DRY_RUN="1")
        first = self.run_piped(env)
        self.assertEqual(first.returncode, 0, first.stderr)
        upstream = folder / "upstream"
        self.git(folder, "clone", "--quiet", str(remote), str(upstream))
        (upstream / "CHANGES.txt").write_text("new version\n", encoding="utf-8")
        self.git(upstream, "add", "CHANGES.txt")
        self.git(upstream, "commit", "--quiet", "-m", "new version")
        self.git(upstream, "push", "--quiet", "origin", "HEAD:main")
        return env, dest, self.git(dest, "rev-parse", "HEAD")

    @unittest.skipUnless(HAVE_CLANG, "install-mac.sh needs Apple's command line tools")
    def test_an_update_whose_packages_fail_leaves_the_source_where_it_was(self):
        # [N5] A re-run fast-forwarded the source first, then installed packages
        # with no rollback, leaving new source on old packages.
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            env, dest, before = self.installed_copy(folder)
            fake = dest / "venv-alpha" / "bin" / "python"
            fake.parent.mkdir(parents=True)
            fake.write_text(FAKE_VENV_PYTHON)
            fake.chmod(0o755)
            bin_dir = folder / "bin"
            bin_dir.mkdir()
            opener = bin_dir / "open"  # never start a real Sotto
            opener.write_text(f'#!/bin/bash\necho "$@" >> "{folder / "opened.txt"}"\n')
            opener.chmod(0o755)
            env = dict(env, SOTTO_PYTHON=str(fake), PIP_FAILS="1", SOTTO_DATA_DIR=str(folder / "data"),
                       PATH=f"{bin_dir}:{os.environ['PATH']}")
            env.pop("SOTTO_GET_DRY_RUN")
            failed = self.run_piped(env)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("source was not switched", failed.stderr)
            self.assertEqual(self.git(dest, "rev-parse", "HEAD"), before)
            self.assertFalse((folder / "opened.txt").exists())

    def test_an_update_refuses_while_sotto_runs_from_the_copy(self):
        # [N26] get.sh pulled and reinstalled under a running Sotto.
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            env, dest, before = self.installed_copy(folder)
            running = subprocess.Popen(["/bin/bash", "-c", "sleep 60; true", str(dest.resolve() / "sotto.py")])
            try:
                time.sleep(0.2)
                refused = self.run_piped(env)
            finally:
                running.kill()
                running.wait()
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn(f"Sotto is running from {dest} (process {running.pid})", refused.stderr)
            self.assertNotIn("would run", refused.stdout)
            self.assertEqual(self.git(dest, "rev-parse", "HEAD"), before)

    @unittest.skipUnless(any(Path(p).is_file() for p in ("/opt/homebrew/bin/python3.12", "/usr/local/bin/python3.12")),
                         "needs a native Python 3.12 in /opt/homebrew or /usr/local")
    def test_an_intel_python_first_on_path_is_skipped_for_a_native_one(self):
        # [N30] Discovery took the first python3.12 on PATH even when it was x86_64.
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            bin_dir = folder / "bin"
            bin_dir.mkdir()
            intel = bin_dir / "python3.12"
            intel.write_text(FAKE_VENV_PYTHON)
            intel.chmod(0o755)
            env = dict(os.environ, SOTTO_REPO=str(bare_copy_of_this_checkout(folder)),
                       SOTTO_SOURCE=str(folder / "sotto"), SOTTO_GET_DRY_RUN="1", FAKE_ARCH="x86_64",
                       PATH=f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin")
            env.pop("SOTTO_PYTHON", None)
            result = self.run_piped(env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("would run", result.stdout)
            self.assertNotIn(str(intel), result.stdout)

    def test_the_same_repository_matches_in_https_and_ssh_form(self):
        script = (ROOT / "scripts" / "get.sh").read_text(encoding="utf-8")
        check = script + '\nsame_repo "git@github.com:zohartito/sotto.git" "https://github.com/zohartito/sotto.git" ' \
                         '&& same_repo "ssh://git@github.com/zohartito/sotto.git" ' \
                         '"https://github.com/zohartito/sotto.git" ' \
                         '&& same_repo "https://github.com/zohartito/sotto/" ' \
                         '"https://github.com/zohartito/sotto.git" ' \
                         '&& ! same_repo "https://github.com/someone/sotto.git" "https://github.com/zohartito/sotto.git"'
        result = subprocess.run(["/bin/bash", "-c", check.replace('\nmain "$@"', '')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(sys.platform == "win32" and (ROOT / ".git").exists() and shutil.which("git"),
                     "Windows git checkout")
class WindowsInstallerTests(unittest.TestCase):
    def test_the_same_repository_matches_in_https_and_both_ssh_forms(self):
        script = (ROOT / "scripts" / "get.ps1").read_text(encoding="utf-8")
        function = re.search(r"(?ms)^function Test-SameRepo.*?^}", script).group(0)
        checks = ("@((Test-SameRepo 'git@github.com:zohartito/sotto.git' 'https://github.com/zohartito/sotto.git'),"
                  " (Test-SameRepo 'ssh://git@github.com/zohartito/sotto.git' 'https://github.com/zohartito/sotto.git'),"
                  " (Test-SameRepo 'https://github.com/zohartito/sotto/' 'https://github.com/zohartito/sotto.git'),"
                  " -not (Test-SameRepo 'https://github.com/someone/sotto.git' 'https://github.com/zohartito/sotto.git')"
                  ") -join ','")
        result = subprocess.run(["powershell", "-NoProfile", "-Command", f"{function}\n{checks}"],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.stdout.strip(), "True,True,True,True", result.stderr)

    def test_downloads_then_updates_without_running_the_installer(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            dest = folder / "sotto"
            env = dict(os.environ, SOTTO_REPO=str(bare_copy_of_this_checkout(folder)), SOTTO_SOURCE=str(dest),
                       SOTTO_GET_DRY_RUN="1")
            command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                       str(windows_get_script(folder)[0])]
            first = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
            self.assertIn("would run", first.stdout, first.stderr)
            self.assertTrue((dest / "scripts" / "install-windows.ps1").is_file())
            again = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
            self.assertIn("Updating Sotto", again.stdout, again.stderr)

    def test_a_git_copy_of_another_project_is_never_pulled_or_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            other_remote = folder / "other.git"
            subprocess.run(["git", "init", "--quiet", "--bare", str(other_remote)], check=True)
            dest = folder / "sotto"
            subprocess.run(["git", "clone", "--quiet", str(other_remote), str(dest)], check=True,
                           capture_output=True)
            env = dict(os.environ, SOTTO_REPO=str(bare_copy_of_this_checkout(folder)), SOTTO_SOURCE=str(dest),
                       SOTTO_GET_DRY_RUN="1")
            command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                       str(windows_get_script(folder)[0])]
            refused = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
            self.assertIn("not Sotto", refused.stdout, refused.stderr)
            self.assertNotIn("would run", refused.stdout)

    def test_an_update_refuses_while_sotto_runs_from_the_copy(self):
        # [F33] get.ps1 pulled and reinstalled under a running Sotto.
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            dest = folder / "sotto"
            env = dict(os.environ, SOTTO_REPO=str(bare_copy_of_this_checkout(folder)), SOTTO_SOURCE=str(dest),
                       SOTTO_GET_DRY_RUN="1")
            command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                       str(windows_get_script(folder)[0])]
            first = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
            self.assertIn("would run", first.stdout, first.stderr)
            scripts = dest / "venv-alpha" / "Scripts"  # gitignored, like the installer's venv
            scripts.mkdir(parents=True)
            stand_in = scripts / "ping.exe"  # any program running from the copy, like pythonw
            shutil.copy(Path(os.environ["SystemRoot"]) / "System32" / "PING.EXE", stand_in)
            # Started by one name of the folder (long or 8.3 short) and named by
            # the other, it is the same copy: GitHub's runner (short TEMP) went ahead.
            long_program, short_program = long_and_short(stand_in)
            long_dest, short_dest = long_and_short(dest)
            for program, source in ((long_program, long_dest), (short_program, long_dest),
                                    (long_program, short_dest)):
                with self.subTest(program=program, source=source):
                    running = subprocess.Popen([program, "-n", "60", "127.0.0.1"],
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    try:
                        refused = subprocess.run(command, env=dict(env, SOTTO_SOURCE=source),
                                                 capture_output=True, text=True, timeout=120)
                    finally:
                        running.kill()
                        running.wait(10)
                    self.assertIn("Sotto is running from", refused.stdout, refused.stderr)
                    self.assertNotIn("Updating Sotto", refused.stdout)
                    self.assertNotIn("would run", refused.stdout)


    def test_an_update_its_installer_would_refuse_leaves_the_copy_unchanged(self):
        # [PR15 review] get.ps1 pulled first; the installer's [F11] refusal
        # (the Start Menu entry starts another copy) then left this copy's new
        # source on its old packages.
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            dest = folder / "sotto"
            remote = bare_copy_of_this_checkout(folder)
            script, menu = windows_get_script(folder)
            env = dict(os.environ, SOTTO_REPO=str(remote), SOTTO_SOURCE=str(dest), SOTTO_GET_DRY_RUN="1")
            command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
            first = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
            self.assertIn("would run", first.stdout, first.stderr)

            def git(cwd: Path, *args: str) -> str:
                return subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                                       *args], cwd=cwd, check=True, capture_output=True,
                                      text=True).stdout.strip()

            upstream = folder / "upstream"
            git(folder, "clone", "--quiet", str(remote), str(upstream))
            (upstream / "CHANGES.txt").write_text("new version\n", encoding="utf-8")
            git(upstream, "add", "CHANGES.txt")
            git(upstream, "commit", "--quiet", "-m", "new version")
            git(upstream, "push", "--quiet", "origin", "HEAD:main")
            newer = git(upstream, "rev-parse", "HEAD")
            before = git(dest, "rev-parse", "HEAD")
            other = folder / "other-copy"
            other.mkdir()
            (other / "win_launch.py").write_text("", encoding="utf-8")
            link = menu / "Sotto.lnk"
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{link}'); "
                            f"$s.TargetPath = '{sys.executable}'; "
                            f"$s.Arguments = '\"{other / 'win_launch.py'}\" --data-dir \"C:\\other-data\"'; "
                            "$s.Save()"], check=True, capture_output=True, timeout=60)
            refused = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(git(dest, "rev-parse", "HEAD"), before, "the source did not move")
            self.assertIn("already starts another copy of Sotto", refused.stdout, refused.stderr)
            self.assertNotIn("would run", refused.stdout)
            link.unlink()  # without the other copy's entry the update goes ahead
            again = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
            self.assertIn("would run", again.stdout, again.stderr)
            self.assertEqual(git(dest, "rev-parse", "HEAD"), newer)

    def test_an_installed_copy_whose_update_packages_fail_keeps_its_source(self):
        # [N5] get.ps1 fast-forwarded an installed copy before any package was
        # installed, and nothing put the old source back when they failed.
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            dest = folder / "sotto"
            remote = bare_copy_of_this_checkout(folder)
            script, _menu = windows_get_script(folder)
            # update-windows.ps1 logs to %LOCALAPPDATA%\sotto-alpha: a temporary one here.
            env = dict(os.environ, SOTTO_REPO=str(remote), SOTTO_SOURCE=str(dest), SOTTO_GET_DRY_RUN="1",
                       LOCALAPPDATA=str(folder / "localappdata"), PIP_NO_INDEX="1",
                       PIP_DISABLE_PIP_VERSION_CHECK="1")
            command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
            first = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
            self.assertIn("would run", first.stdout, first.stderr)
            subprocess.run([sys.executable, "-m", "venv", str(dest / "venv-alpha")], check=True,
                           capture_output=True, timeout=600)

            def git(cwd: Path, *args: str) -> str:
                return subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                                       *args], cwd=cwd, check=True, capture_output=True,
                                      text=True).stdout.strip()

            upstream = folder / "upstream"
            git(folder, "clone", "--quiet", str(remote), str(upstream))
            requirements = upstream / "requirements-alpha-windows.txt"
            requirements.write_text(requirements.read_text(encoding="utf-8") + "sotto-test-missing-package==1.0\n",
                                    encoding="utf-8")
            git(upstream, "commit", "--quiet", "-am", "a package that cannot be installed")
            git(upstream, "push", "--quiet", "origin", "HEAD:main")
            before = git(dest, "rev-parse", "HEAD")
            failed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
            self.assertEqual(git(dest, "rev-parse", "HEAD"), before, failed.stdout + failed.stderr)
            self.assertIn("previous packages are back", failed.stdout, failed.stderr)
            self.assertNotIn("would run", failed.stdout)
            self.assertEqual(failed.returncode, 1, "[N41] a failure exits nonzero when run as a file")

    def test_a_failure_sets_the_exit_code_without_closing_a_pasted_shell(self):
        # [N41] Every failure was a bare return: exit code 0 as a file, and
        # $LASTEXITCODE left as whatever ran last when pasted through iex.
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            foreign = folder / "notes"
            foreign.mkdir()
            script, _menu = windows_get_script(folder)
            env = dict(os.environ, SOTTO_REPO=str(folder / "remote.git"), SOTTO_SOURCE=str(foreign),
                       SOTTO_GET_DRY_RUN="1")
            as_file = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                                     env=env, capture_output=True, text=True, timeout=120)
            self.assertIn("is not a copy of Sotto", as_file.stdout, as_file.stderr)
            self.assertEqual(as_file.returncode, 1)
            pasted = subprocess.run(["powershell", "-NoProfile", "-Command",
                                     f"cmd /c exit 0; Get-Content -Raw -LiteralPath '{script}' | iex; "
                                     "\"still here $LASTEXITCODE\""],
                                    env=env, capture_output=True, text=True, timeout=120)
            self.assertIn("is not a copy of Sotto", pasted.stdout, pasted.stderr)
            self.assertIn("still here 1", pasted.stdout, pasted.stderr)


if __name__ == "__main__":
    unittest.main()

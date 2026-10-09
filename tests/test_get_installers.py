"""One-line installers: download (or update) a git copy, then hand off to the platform installer."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def bare_copy_of_this_checkout(folder: Path) -> Path:
    """A 'GitHub' whose main branch is the commit checked out here. CI checks
    out a detached, shallow commit, so push it rather than clone the branches."""
    remote = folder / "remote.git"
    subprocess.run(["git", "init", "--quiet", "--bare", str(remote)], check=True)
    subprocess.run(["git", "-C", str(remote), "config", "receive.shallowUpdate", "true"], check=True)
    subprocess.run(["git", "-C", str(ROOT), "push", "--quiet", str(remote), "HEAD:refs/heads/main"], check=True)
    subprocess.run(["git", "-C", str(remote), "symbolic-ref", "HEAD", "refs/heads/main"], check=True)
    return remote


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
        self.assertNotRegex(text, r"(?im)^\s*exit\b", "iex runs in the user's session; exit would close it")
        self.assertTrue(text.rstrip().endswith("Install-Sotto"))


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
        import re
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
                       str(ROOT / "scripts" / "get.ps1")]
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
                       str(ROOT / "scripts" / "get.ps1")]
            refused = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
            self.assertIn("not Sotto", refused.stdout, refused.stderr)
            self.assertNotIn("would run", refused.stdout)


if __name__ == "__main__":
    unittest.main()

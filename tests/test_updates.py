"""Check for Updates: only on request, only a clean checkout strictly behind GitHub."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import updates


def completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


class FakeGit:
    """Answers the git commands updates.check runs, by subcommand."""

    def __init__(self, counts="0 0", status="", fetch_code=0, upstream_code=0, subjects="Fix a thing\n"):
        self.answers = {"rev-parse": completed("abc1234\n"), "fetch": completed(returncode=fetch_code,
                        stderr="fatal: unable to access github.com"),
                        "rev-list": completed(counts + "\n"), "status": completed(status), "log": completed(subjects)}
        self.upstream_code = upstream_code
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        subcommand = argv[3]
        if subcommand == "rev-parse" and "@{u}" in argv:
            return completed("origin/main\n", returncode=self.upstream_code)
        return self.answers[subcommand]


class CheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / ".git").mkdir()

    def check(self, fake, offline=False):
        return updates.check(self.root, offline=offline, run=fake, which=lambda name: "/usr/bin/git")

    def test_offline_and_non_git_copies_never_touch_the_network(self):
        fake = FakeGit()
        self.assertEqual(self.check(fake, offline=True).state, "offline")
        shutil.rmtree(self.root / ".git")
        self.assertEqual(self.check(fake).state, "not-git")
        self.assertEqual(fake.calls, [])

    def test_states_from_git(self):
        self.assertEqual(self.check(FakeGit(counts="0 0")).state, "current")
        available = self.check(FakeGit(counts="0 3", subjects="Add A\nFix B\n"))
        self.assertEqual((available.state, available.behind, available.changes), ("available", 3, ("Add A", "Fix B")))
        self.assertEqual(self.check(FakeGit(counts="1 2")).state, "diverged")
        self.assertEqual(self.check(FakeGit(counts="0 2", status=" M sotto.py\n")).state, "local-changes")
        failed = self.check(FakeGit(fetch_code=1))
        self.assertEqual(failed.state, "failed")
        self.assertIn("unable to access", failed.detail)
        self.assertEqual(self.check(FakeGit(upstream_code=128)).state, "failed")
        self.assertEqual(updates.check(self.root, offline=False, run=FakeGit(), which=lambda name: None).state,
                         "failed")

    def test_a_failed_git_inspection_is_never_reported_as_up_to_date(self):
        broken_count = FakeGit()
        broken_count.answers["rev-list"] = completed(returncode=128, stderr="fatal: bad revision")
        self.assertEqual(self.check(broken_count).state, "failed")
        self.assertEqual(self.check(FakeGit(counts="garbage")).state, "failed")
        broken_status = FakeGit(counts="0 2")
        broken_status.answers["status"] = completed(returncode=128, stderr="fatal: index file corrupt")
        self.assertEqual(self.check(broken_status).state, "failed")

    def test_descriptions_name_the_changes_and_the_manual_path(self):
        title, text = updates.describe(updates.UpdateCheck("available", behind=7, changes=("Add A",)), "x")
        self.assertEqual(title, "7 updates available")
        self.assertIn("• Add A", text)
        self.assertIn("• …", text)
        title, text = updates.describe(updates.UpdateCheck("local-changes", "Has edits."), "run THIS")
        self.assertIn("run THIS", text)
        self.assertEqual(updates.describe(updates.UpdateCheck("current", version="abc1234"), "x")[0],
                         "Sotto is up to date")


@unittest.skipUnless(shutil.which("git"), "needs git")
class RealGitTests(unittest.TestCase):
    """A bare 'GitHub' remote, a publisher clone and a user clone."""

    def git(self, cwd, *args):
        return subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                               "-c", "init.defaultBranch=main", *args],
                              cwd=cwd, check=True, capture_output=True, text=True)

    def test_behind_then_local_changes_then_current(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            remote, publisher, user = base / "remote.git", base / "publisher", base / "user"
            self.git(base, "init", "--bare", str(remote))
            self.git(base, "clone", str(remote), str(publisher))
            (publisher / "README.md").write_text("one\n")
            self.git(publisher, "add", "README.md")
            self.git(publisher, "commit", "-m", "First")
            self.git(publisher, "push", "origin", "HEAD:main")
            self.git(base, "clone", "--branch", "main", str(remote), str(user))
            self.assertEqual(updates.check(user, offline=False).state, "current")

            (publisher / "README.md").write_text("two\n")
            self.git(publisher, "commit", "-am", "Faster startup")
            self.git(publisher, "push", "origin", "HEAD:main")
            result = updates.check(user, offline=False)
            self.assertEqual((result.state, result.behind, result.changes), ("available", 1, ("Faster startup",)))

            (user / "README.md").write_text("my edit\n")
            self.assertEqual(updates.check(user, offline=False).state, "local-changes")
            self.git(user, "checkout", "README.md")
            self.git(user, "pull", "--ff-only")
            self.assertEqual(updates.check(user, offline=False).state, "current")


class ApplyMacTests(unittest.TestCase):
    """The installer's update path, and whether it switched the source."""

    def apply(self, heads, installer_code, output):
        heads = iter(heads)
        installer_calls = []

        def fake(argv, **kwargs):
            if argv[0] == "git":
                return completed(next(heads) + "\n")
            installer_calls.append(argv)
            kwargs["stdout"].write(output)
            return completed(returncode=installer_code)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = updates.apply_mac(root, "/venv/bin/python", root / "logs" / "update.log", run=fake)
        self.assertEqual(installer_calls, [["/bin/bash", str(root / "scripts" / "install-mac.sh"), "--update",
                                            "--python", "/venv/bin/python"]])
        return result

    def test_a_finished_update_reports_its_last_line_and_the_new_source(self):
        self.assertEqual(self.apply(["aaa", "bbb"], 0, "== Sotto.app\nDone.\n"), (True, "Done.", True))

    def test_a_failure_before_the_switch_leaves_the_source_alone(self):
        self.assertEqual(self.apply(["aaa", "aaa"], 1, "✗ Installing packages failed.\n"),
                         (False, "✗ Installing packages failed.", False))

    def test_a_failure_after_the_switch_says_the_source_changed(self):
        self.assertEqual(self.apply(["aaa", "bbb"], 1, "✗ model download failed\n"),
                         (False, "✗ model download failed", True))


@unittest.skipUnless(sys.platform == "win32", "Windows tray updater")
class WindowsUpdateTests(unittest.TestCase):
    def test_git_never_opens_a_console_window(self):
        # [F37] The tray runs under pythonw; each git call used to flash a console.
        kwargs_seen = []
        fake = FakeGit(counts="0 2")

        def run(argv, **kwargs):
            kwargs_seen.append(kwargs)
            return fake(argv, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / ".git").mkdir()
            self.assertEqual(updates.check(Path(temporary), offline=False, run=run,
                                           which=lambda name: "git").state, "available")
        self.assertEqual(len(kwargs_seen), 6)
        for kwargs in kwargs_seen:
            self.assertTrue(kwargs.get("creationflags", 0) & subprocess.CREATE_NO_WINDOW, kwargs)

    def test_hand_off_waits_for_quiet_and_reports_once(self):
        import queue
        import sotto
        import sotto_win
        import win_ui
        capture = mock.Mock()
        capture.is_active.return_value = False
        engine = mock.Mock()
        engine.force_finish.return_value = False  # nothing is recording

        def lifecycle_parts():
            return dict(engine=engine, on_finish=mock.Mock(), finish_from_menu=threading.Event(),
                        capture_gate=sotto.CaptureGate(), lifecycle={"closed": None})

        busy_jobs = queue.Queue()
        busy_jobs.put("live")
        busy = sotto_win.Controller(shutdown=sotto.ShutdownBoundary(), capture=capture, jobs=busy_jobs,
                                    deliveries=queue.Queue(), finishing=lambda: False,
                                    **lifecycle_parts())
        with mock.patch("subprocess.Popen") as spawn:
            self.assertIn("Finish the current dictation", busy.update_and_restart())
        spawn.assert_not_called()
        idle = sotto_win.Controller(shutdown=sotto.ShutdownBoundary(), capture=capture, jobs=queue.Queue(),
                                    deliveries=queue.Queue(), finishing=lambda: False,
                                    **lifecycle_parts())
        with mock.patch("subprocess.Popen") as spawn, mock.patch.object(sotto_win, "log"):
            idle.update_and_restart()
            idle.shutdown.event.wait(5)  # the quit drains on its own thread
        argv = spawn.call_args.args[0]
        self.assertTrue(argv[0].lower().startswith("powershell"))
        self.assertTrue(argv[argv.index("-File") + 1].endswith("update-windows.ps1"))
        self.assertTrue(idle.shutdown.requested())
        with tempfile.TemporaryDirectory() as temporary:
            status = Path(temporary) / "update-status.txt"
            status.write_text("﻿ok abc1234\n", encoding="utf-8")
            self.assertEqual(win_ui.update_report(temporary), "Sotto updated (abc1234).")
            self.assertIsNone(win_ui.update_report(temporary))
            status.write_text("failed Sotto did not quit, so nothing was changed.", encoding="utf-8")
            self.assertEqual(win_ui.update_report(temporary),
                             "Update failed: Sotto did not quit, so nothing was changed.")


if __name__ == "__main__":
    unittest.main()

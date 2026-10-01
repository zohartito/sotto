"""Sotto.app builder, launcher behaviour and the login item — all in temp dirs."""
import importlib.util
import os
from pathlib import Path
import plistlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock

import login_item

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("install_app", ROOT / "scripts/install_app.py")
install_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(install_app)
HAVE_CLANG = sys.platform == "darwin" and subprocess.run(
    ["/usr/bin/xcrun", "--find", "clang"], capture_output=True).returncode == 0


class LauncherSourceTests(unittest.TestCase):
    def test_paths_are_escaped_and_must_be_absolute(self):
        self.assertEqual(install_app.c_string('a "b"\\c é'), '"a \\042b\\042\\134c \\303\\251"')
        with self.assertRaises(ValueError):
            install_app.launcher_source(python=Path("python"), script=Path("/s"), data_dir=Path("/d"),
                                        hf_home=Path("/h"), app_executable=Path("/e"))

    def test_info_plist_is_a_menu_bar_app_with_a_microphone_reason(self):
        info = plistlib.loads(install_app.info_plist("0.1.0-alpha"))
        self.assertEqual(info["CFBundleIdentifier"], "org.sotto.alpha")
        self.assertTrue(info["LSUIElement"])
        self.assertEqual(info["LSMinimumSystemVersion"], "14.0")
        self.assertIn("never leaves", info["NSMicrophoneUsageDescription"])


@unittest.skipUnless(HAVE_CLANG, "needs macOS with Apple's command line tools")
class LauncherBehaviourTests(unittest.TestCase):
    """Compile the real launcher around a recorder shell script (never the app)."""

    def launcher(self, body: str):
        temporary = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, temporary)
        record = temporary / "record.txt"
        script = temporary / "child.sh"
        script.write_text(f'printf "%s\\n" "$@" "$SOTTO_DATA_DIR" "$SOTTO_LAUNCHER" > "{record}"\n{body}\n')
        source = install_app.launcher_source(
            python=Path("/bin/sh"), script=script, data_dir=temporary / "data dir",
            hf_home=temporary / "hf", app_executable=temporary / "Sotto")
        binary = install_app.compile_launcher(source, temporary / "Sotto")
        return binary, record, temporary

    def test_environment_arguments_and_exit_code_pass_through(self):
        binary, record, temporary = self.launcher("exit 3")
        result = subprocess.run([str(binary)], timeout=20)
        self.assertEqual(result.returncode, 3)
        self.assertEqual(record.read_text().splitlines(),
                         ["run", "--idle-release", "0", str(temporary / "data dir"), "app"])

    def test_sigterm_is_forwarded_to_the_python_child(self):
        binary, record, _temporary = self.launcher("sleep 30")
        process = subprocess.Popen([str(binary)])
        deadline = time.monotonic() + 10
        while not record.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=10), 128 + signal.SIGTERM)

    def test_build_makes_a_signed_bundle_and_an_identical_rebuild_keeps_it(self):
        temporary = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, temporary)
        options = dict(applications=temporary / "Applications", data_dir=temporary / "data",
                       hf_home=temporary / "data" / "huggingface", python=Path(sys.executable).absolute())
        app = install_app.build(**options)
        self.assertTrue((app / "Contents/MacOS/Sotto").is_file())
        self.assertTrue((app / "Contents/Resources/Sotto.icns").is_file())
        verify = subprocess.run(["/usr/bin/codesign", "--verify", str(app)], capture_output=True)
        self.assertEqual(verify.returncode, 0, verify.stderr)
        before = (app / "Contents/MacOS/Sotto").stat().st_mtime_ns
        install_app.build(**options)
        self.assertEqual((app / "Contents/MacOS/Sotto").stat().st_mtime_ns, before)


class InstallLocationTests(unittest.TestCase):
    def make_app(self, folder: Path, bundle_id: str) -> Path:
        contents = folder / "Sotto.app" / "Contents"
        contents.mkdir(parents=True)
        (contents / "Info.plist").write_bytes(plistlib.dumps({"CFBundleIdentifier": bundle_id}))
        return contents.parent

    def test_system_applications_when_writable_otherwise_personal(self):
        home = Path("/Users/someone")
        with unittest.mock.patch.object(install_app.os, "access", return_value=True):
            self.assertEqual(install_app.default_applications(home), Path("/Applications"))
        with unittest.mock.patch.object(install_app.os, "access", return_value=False):
            self.assertEqual(install_app.default_applications(home), home / "Applications")

    def test_only_our_bundle_is_removed_and_the_kept_copy_survives(self):
        with tempfile.TemporaryDirectory() as temporary:
            system, personal, foreign = (Path(temporary) / name for name in ("system", "personal", "foreign"))
            kept = self.make_app(system, install_app.BUNDLE_ID)
            old = self.make_app(personal, install_app.BUNDLE_ID)
            other = self.make_app(foreign, "com.someone.else.sotto")
            removed = install_app.remove_copies([system, personal, foreign, Path(temporary) / "missing"], keep=kept)
            self.assertEqual(removed, [old])
            self.assertTrue(kept.exists())
            self.assertFalse(old.exists())
            self.assertTrue(other.exists())
            self.assertFalse(install_app.ours(other))

    @unittest.skipUnless(sys.platform == "darwin", "Sotto.app is macOS only")
    def test_dialogs_use_the_icon_of_the_app_that_launched_python(self):
        import sotto
        with tempfile.TemporaryDirectory() as temporary:
            contents = Path(temporary) / "Sotto.app" / "Contents"
            (contents / "Resources").mkdir(parents=True)
            executable = contents / "MacOS" / "Sotto"
            with unittest.mock.patch.dict(os.environ, {"SOTTO_APP_EXECUTABLE": str(executable)}):
                self.assertIsNone(sotto.sotto_icon_path())
                (contents / "Resources" / "Sotto.icns").write_bytes(b"icns")
                self.assertEqual(sotto.sotto_icon_path(), contents / "Resources" / "Sotto.icns")
            with unittest.mock.patch.dict(os.environ, {"SOTTO_APP_EXECUTABLE": " "}):
                self.assertIsNone(sotto.sotto_icon_path())


@unittest.skipUnless(sys.platform == "darwin", "Sotto.app is macOS only")
class AccessibilityProbeTests(unittest.TestCase):
    def test_each_check_asks_a_fresh_interpreter(self):
        import sotto
        results = iter([1, 0])
        with unittest.mock.patch("subprocess.run", side_effect=lambda argv, **kw: unittest.mock.Mock(
                returncode=next(results))) as run:
            self.assertFalse(sotto.accessibility_granted_now())
            self.assertTrue(sotto.accessibility_granted_now())
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.args[0][:4], [sys.executable, "-I", "-S", "-c"])
        with unittest.mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("python", 5)):
            self.assertFalse(sotto.accessibility_granted_now())

    def test_probe_runs_and_answers_with_an_exit_code(self):
        import sotto
        result = subprocess.run([sys.executable, "-I", "-S", "-c", sotto.ACCESSIBILITY_PROBE])
        self.assertIn(result.returncode, (0, 1))


@unittest.skipUnless(sys.platform == "darwin", "macOS LaunchAgent login item; Windows: tests/test_win_startup.py")
class LoginItemTests(unittest.TestCase):
    def test_enable_writes_a_relaunch_on_crash_agent_and_disable_only_unloads_on_uninstall(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            path = login_item.enable("/Applications/Sotto.app/Contents/MacOS/Sotto", home)
            agent = plistlib.loads(path.read_bytes())
            self.assertEqual(agent["Label"], "org.sotto.alpha")
            self.assertEqual(agent["KeepAlive"], {"SuccessfulExit": False})
            self.assertTrue(agent["RunAtLoad"])
            self.assertTrue(login_item.enabled(home))
            calls = []
            login_item.disable(home, run=lambda *a, **k: calls.append(a[0]))
            self.assertEqual(calls, [])
            self.assertFalse(login_item.enabled(home))
            login_item.enable("/x/Sotto", home)
            login_item.disable(home, unload=True, run=lambda *a, **k: calls.append(a[0]))
            self.assertEqual(calls, [["/bin/launchctl", "bootout", f"gui/{os.getuid()}/org.sotto.alpha"]])
        with self.assertRaises(ValueError):
            login_item.agent("relative/Sotto")


@unittest.skipUnless(sys.platform == "darwin", "macOS bash installer; Windows: tests/test_win_installer.py")
class InstallerScriptTests(unittest.TestCase):
    def test_shell_installer_parses_and_documents_its_options(self):
        script = ROOT / "scripts/install-mac.sh"
        self.assertEqual(subprocess.run(["/bin/bash", "-n", str(script)]).returncode, 0)
        self.assertTrue(os.access(script, os.X_OK))
        usage = subprocess.run(["/bin/bash", str(script), "--help"], capture_output=True, text=True)
        self.assertIn("--uninstall", usage.stdout)


if __name__ == "__main__":
    unittest.main()

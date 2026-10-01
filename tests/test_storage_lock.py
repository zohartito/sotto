"""Cross-platform advisory lock: reentrancy, thread blocking, process blocking.

The process-blocking test is the one that distinguishes a real OS lock from a
no-op: a child process holds the store lock and the parent must wait for it.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from storage_lock import advisory_lock, ensure_private_directory, ensure_private_file

REPO_ROOT = Path(__file__).resolve().parent.parent


class StorageLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="sotto-lock-test-")
        self.root = Path(self._tmp.name) / "store"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_acquire_creates_private_store(self) -> None:
        with advisory_lock(self.root):
            self.assertTrue((self.root / ".sotto.lock").exists())
        info = self.root.stat()
        self.assertTrue(info.st_mode & 0o700 == 0o700 or sys.platform == "win32")

    def test_reentrant_nested(self) -> None:
        with advisory_lock(self.root):
            with advisory_lock(self.root):
                with advisory_lock(self.root):
                    pass

    def test_thread_blocking(self) -> None:
        acquired = threading.Event()
        release = threading.Event()

        def hold() -> None:
            with advisory_lock(self.root):
                acquired.set()
                release.wait(timeout=10)

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        self.assertTrue(acquired.wait(timeout=5))

        entered = {"value": False}

        def try_acquire() -> None:
            with advisory_lock(self.root):
                entered["value"] = True

        waiter = threading.Thread(target=try_acquire, daemon=True)
        waiter.start()
        time.sleep(0.3)
        self.assertFalse(entered["value"],
                         "second thread must block while the first holds")
        release.set()
        waiter.join(timeout=5)
        self.assertFalse(waiter.is_alive(), "waiter must acquire after release")
        self.assertTrue(entered["value"])

    def test_process_blocking(self) -> None:
        marker = Path(self._tmp.name) / "child-locked"
        child_script = f"""
import sys, time
sys.path.insert(0, {str(REPO_ROOT)!r})
from pathlib import Path
from storage_lock import advisory_lock
with advisory_lock({str(self.root)!r}):
    Path({str(marker)!r}).write_text("locked")
    time.sleep(4.0)
"""
        child = subprocess.Popen([sys.executable, "-c", child_script])
        try:
            deadline = time.monotonic() + 15
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(marker.exists(), "child never acquired the lock")

            entered = {"value": False}

            def try_acquire() -> None:
                with advisory_lock(self.root):
                    entered["value"] = True

            waiter = threading.Thread(target=try_acquire, daemon=True)
            waiter.start()
            time.sleep(0.5)
            self.assertFalse(entered["value"],
                             "parent must block while the child holds")
            waiter.join(timeout=15)
            self.assertTrue(entered["value"],
                            "parent must acquire after the child exits")
        finally:
            child.wait(timeout=20)

    def test_ensure_private_helpers(self) -> None:
        directory = ensure_private_directory(self.root / "nested")
        self.assertTrue(directory.is_dir())
        target = directory / "meta.json"
        target.write_text("{}")
        self.assertEqual(ensure_private_file(target), target)


if __name__ == "__main__":
    unittest.main()

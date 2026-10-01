"""Reentrant, process-wide advisory locking for Sotto's filesystem store."""
from __future__ import annotations

import os
import stat
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

if sys.platform == "win32":
    import msvcrt

    def lock_exclusive(fd: int) -> None:
        """Blocking exclusive lock: msvcrt byte-range lock on byte 0.

        ``LK_NBLCK`` raises ``OSError`` on contention, so retry to match
        ``flock(LOCK_EX)``'s blocking semantics.
        """
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(0.01)

    def unlock_exclusive(fd: int) -> None:
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def lock_exclusive(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def unlock_exclusive(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)

_REGISTRY_LOCK = threading.Lock()
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_LOCAL = threading.local()


def ensure_private_directory(path: Path | str) -> Path:
    """Create/tighten a Sotto private directory independent of umask."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if stat.S_ISLNK(os.lstat(directory).st_mode):
        raise RuntimeError(f"refusing symlinked private directory: {directory}")
    os.chmod(directory, 0o700)
    return directory


def ensure_private_file(path: Path | str) -> Path:
    """Tighten an existing private metadata/lock file."""
    target = Path(path)
    if target.exists() and not stat.S_ISLNK(os.lstat(target).st_mode):
        os.chmod(target, 0o600)
    elif target.is_symlink():
        raise RuntimeError(f"refusing symlinked private file: {target}")
    return target


def _thread_states() -> dict[str, tuple[int, int]]:
    states = getattr(_LOCAL, "states", None)
    if states is None:
        states = {}
        _LOCAL.states = states
    return states


@contextmanager
def advisory_lock(base_dir: Path | str):
    """Hold the store lock, reentrantly, across threads and POSIX processes."""
    root = Path(base_dir).resolve()
    key = str(root)
    with _REGISTRY_LOCK:
        thread_lock = _THREAD_LOCKS.setdefault(key, threading.RLock())
    thread_lock.acquire()
    states = _thread_states()
    count, fd = states.get(key, (0, -1))
    entered = False
    try:
        if count == 0:
            ensure_private_directory(root)
            lock_path = root / ".sotto.lock"
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            os.chmod(lock_path, 0o600)
            try:
                lock_exclusive(fd)
            except Exception:
                os.close(fd)
                raise
        states[key] = (count + 1, fd)
        entered = True
        yield
    finally:
        if entered:
            remaining, current_fd = states.get(key, (0, fd))
            if remaining <= 1:
                states.pop(key, None)
                if current_fd >= 0:
                    try:
                        unlock_exclusive(current_fd)
                    finally:
                        os.close(current_fd)
            else:
                states[key] = (remaining - 1, current_fd)
        thread_lock.release()

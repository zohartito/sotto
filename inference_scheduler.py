"""Crash-safe local priority scheduling for live and evaluation inference.

Lock ordering is intentionally simple: mutate the registry under the *separate*
inference-state lock, release it, then try the OS inference lease.  No storage
lock is used here and callers must never hold one while waiting for this lease.
"""
from __future__ import annotations

import fcntl
import ctypes
import ctypes.util
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from history import _fsync_dir
from storage_lock import ensure_private_directory, ensure_private_file


def default_boot_session_id() -> str | None:
    """Return a real boot-session token without invoking a shell.

    Darwin exposes ``kern.boottime`` through libc; Linux test/development hosts
    expose a boot UUID.  If neither is available we return ``None`` and never
    reclaim a live PID merely because its unverified boot token differs.
    """
    if os.uname().sysname == "Darwin":
        try:
            libc = ctypes.CDLL(None)
            value = (ctypes.c_long * 2)()
            size = ctypes.c_size_t(ctypes.sizeof(value))
            if libc.sysctlbyname(b"kern.boottime", ctypes.byref(value), ctypes.byref(size), None, 0) == 0:
                return f"darwin:{value[0]}:{value[1]}"
        except (AttributeError, OSError):
            return None
    try:
        return "linux:" + Path("/proc/sys/kernel/random/boot_id").read_text("ascii").strip()
    except OSError:
        return None
    return None


def default_process_start_token(pid: int) -> str | None:
    """Return an OS process-start token, or ``None`` when it cannot be proven."""
    if os.uname().sysname != "Darwin":
        try:
            return "linux:" + (Path("/proc") / str(pid) / "stat").read_text("ascii").split()[21]
        except (OSError, IndexError):
            return None
    # libproc's proc_bsdinfo contains a stable start timeval.  Define only the
    # prefix needed by the documented structure; failure remains fail-closed.
    try:
        class ProcBSDInfo(ctypes.Structure):
            _fields_ = [("flags", ctypes.c_uint32), ("status", ctypes.c_uint32), ("xstatus", ctypes.c_uint32), ("pbi_pid", ctypes.c_uint32), ("ppid", ctypes.c_uint32), ("uid", ctypes.c_uint32), ("gid", ctypes.c_uint32), ("ruid", ctypes.c_uint32), ("rgid", ctypes.c_uint32), ("svuid", ctypes.c_uint32), ("svgid", ctypes.c_uint32), ("rfu", ctypes.c_uint32 * 4), ("comm", ctypes.c_char * 17), ("name", ctypes.c_char * 33), ("nfiles", ctypes.c_uint32), ("pgid", ctypes.c_uint32), ("pjobc", ctypes.c_uint32), ("tdev", ctypes.c_uint32), ("tpgid", ctypes.c_uint32), ("nice", ctypes.c_int32), ("start_sec", ctypes.c_uint64), ("start_usec", ctypes.c_uint64)]
        library = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
        info = ProcBSDInfo()
        if library.proc_pidinfo(pid, 13, 0, ctypes.byref(info), ctypes.sizeof(info)) >= ctypes.sizeof(info):
            return f"darwin:{info.start_sec}:{info.start_usec}"
    except (AttributeError, OSError):
        return None
    return None


class _FileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
    @contextmanager
    def held(self) -> Iterator[None]:
        ensure_private_directory(self.path.parent)
        with self._lock:
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
            os.chmod(self.path, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)


class InferenceScheduler:
    """Owner registry plus a process-wide inference lease.

    Use :meth:`live_request` around each live inference.  Evaluators must use
    :meth:`evaluator_lease` around *one* challenger/champion sample at a time,
    releasing it between samples; this is how pending live work gets priority.
    The state lock and lease are deliberately distinct from storage locking.
    """
    heartbeat_seconds = 2.0
    stale_seconds = 30.0

    def __init__(self, base_dir: Path | str, *, pid: int | None = None,
                 pid_alive: Callable[[int], bool] | None = None,
                 start_token: Callable[[int], str | None] | None = None,
                 boot_id: Callable[[], str] | None = None,
                 wall_clock: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        root = Path(base_dir)
        self.root = root / "inference-scheduler"
        self.state_path = self.root / "state.json"
        self._state_lock = _FileLock(self.root / ".inference-state.lock")
        self._lease_path = self.root / ".inference-lease.lock"
        self.pid = os.getpid() if pid is None else pid
        self._pid_alive = pid_alive or self._default_pid_alive
        self._start_token = start_token or default_process_start_token
        self._boot_id = boot_id or default_boot_session_id
        self._wall = wall_clock
        self._mono = monotonic
        self.owner_id = uuid.uuid4().hex
        self._lease_fd: int | None = None
        self._ensure()

    @staticmethod
    def _default_pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _ensure(self) -> None:
        ensure_private_directory(self.root.parent)
        ensure_private_directory(self.root)
        ensure_private_file(self.state_path)
        ensure_private_file(self._lease_path)
        if not self.state_path.exists():
            with self._state_lock.held():
                if not self.state_path.exists(): self._write({"revision": 0, "owners": {}})

    def _read(self) -> dict[str, Any]:
        try:
            state = json.loads(self.state_path.read_text("utf-8"))
            if not isinstance(state, dict) or not isinstance(state.get("owners"), dict): raise ValueError
            return state
        except (OSError, ValueError, json.JSONDecodeError):
            # Corrupt scheduler state fails closed: an evaluator cannot run.
            return {"revision": 0, "owners": {}, "corrupt": True}

    def _write(self, state: dict[str, Any]) -> None:
        temp = self.state_path.with_name(f".{self.state_path.name}.{uuid.uuid4().hex}.tmp")
        payload = json.dumps(state, sort_keys=True, separators=(",", ":"))
        with open(temp, "w", encoding="utf-8") as out:
            out.write(payload); out.flush(); os.fsync(out.fileno())
        os.chmod(temp, 0o600)
        os.replace(temp, self.state_path); os.chmod(self.state_path, 0o600); _fsync_dir(self.root)

    def _owner(self, role: str, pending: int = 0) -> dict[str, Any]:
        return {"uuid": self.owner_id, "pid": self.pid, "start_token": self._start_token(self.pid),
                "boot_id": self._boot_id(), "role": role, "pending_live": pending,
                "wall_heartbeat": self._wall(), "monotonic_heartbeat": self._mono()}

    def _dead(self, row: dict[str, Any]) -> bool:
        current_boot = self._boot_id()
        if current_boot is not None and row.get("boot_id") != current_boot:
            return True
        pid = row.get("pid")
        if not isinstance(pid, int) or not self._pid_alive(pid):
            return True
        current_start = self._start_token(pid)
        return current_start is not None and row.get("start_token") != current_start

    def _reconcile_locked(self, state: dict[str, Any]) -> list[str]:
        now = self._wall(); reclaimed: list[str] = []
        for ident, row in list(state["owners"].items()):
            stale = not isinstance(row, dict) or now - float(row.get("wall_heartbeat", 0)) > self.stale_seconds
            if stale and (not isinstance(row, dict) or self._dead(row)):
                state["owners"].pop(ident, None); reclaimed.append(ident)
        return reclaimed

    def _suspect_evaluator_locked(self, state: dict[str, Any]) -> bool:
        return any(row.get("role") == "evaluator" and self._wall() - float(row.get("wall_heartbeat", 0)) > self.stale_seconds
                   for row in state["owners"].values() if isinstance(row, dict))

    def reconcile(self) -> list[str]:
        with self._state_lock.held():
            state = self._read()
            if state.get("corrupt"): return []
            reclaimed = self._reconcile_locked(state)
            if reclaimed:
                state["revision"] += 1; self._write(state)
            return reclaimed

    def _update(self, fn: Callable[[dict[str, Any]], Any]) -> Any:
        with self._state_lock.held():
            state = self._read()
            if state.get("corrupt"): raise RuntimeError("corrupt inference scheduler state")
            self._reconcile_locked(state)
            result = fn(state)
            state["revision"] = int(state.get("revision", 0)) + 1
            self._write(state)
            return result

    def heartbeat(self, role: str = "live") -> None:
        def update(state: dict[str, Any]) -> None:
            previous = state["owners"].get(self.owner_id, {})
            state["owners"][self.owner_id] = self._owner(role, int(previous.get("pending_live", 0)))
        self._update(update)

    @contextmanager
    def heartbeat_pump(self, role: str) -> Iterator[None]:
        """Keep an owner fresh during lease waits and long in-flight calls."""
        stop = threading.Event()

        def pump() -> None:
            while not stop.wait(self.heartbeat_seconds):
                try:
                    self.heartbeat(role)
                except RuntimeError:
                    return

        thread = threading.Thread(target=pump, name="sotto-inference-heartbeat", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=self.heartbeat_seconds + .1)

    def status(self) -> dict[str, Any]:
        with self._state_lock.held():
            state = self._read()
            if state.get("corrupt"): return {"status": "corrupt", "live_pending": None, "owners": 0}
            self._reconcile_locked(state)
            owners = list(state["owners"].values())
            return {"status": "ok", "live_pending": sum(int(row.get("pending_live", 0)) for row in owners),
                    "owners": len(owners), "evaluator_suspect": any(
                        row.get("role") == "evaluator" and self._wall() - float(row.get("wall_heartbeat", 0)) > self.stale_seconds
                        for row in owners)}

    def live_pending(self) -> bool:
        """Return whether live work is registered, failing closed on doubt.

        This is deliberately only a registry read: it never touches the
        inference lease, so an evaluator can poll it while it owns that lease
        and yield promptly for a foreground request.
        """
        try:
            with self._state_lock.held():
                state = self._read()
                if state.get("corrupt"):
                    return True
                reclaimed = self._reconcile_locked(state)
                if reclaimed:
                    state["revision"] = int(state.get("revision", 0)) + 1
                    self._write(state)
                for row in state["owners"].values():
                    if not isinstance(row, dict):
                        return True
                    pending = row.get("pending_live", 0)
                    if type(pending) is not int or pending < 0:
                        return True
                    if pending:
                        return True
                return False
        except (OSError, ValueError, TypeError, KeyError):
            return True

    @contextmanager
    def live_request(self) -> Iterator[None]:
        """Register pending live work before attempting the shared lease."""
        def plus(state: dict[str, Any]) -> None:
            old = state["owners"].get(self.owner_id, {})
            state["owners"][self.owner_id] = self._owner("live", int(old.get("pending_live", 0)) + 1)
        self._update(plus)
        fd: int | None = None
        try:
            with self.heartbeat_pump("live"):
                # Registration happened before this wait, so any evaluator
                # that releases between samples observes live work and yields.
                while fd is None:
                    fd = self._try_lease()
                    if fd is None:
                        time.sleep(.01)
                yield
        finally:
            if fd is not None: self._release_fd(fd)
            def minus(state: dict[str, Any]) -> None:
                old = state["owners"].get(self.owner_id, {})
                state["owners"][self.owner_id] = self._owner("live", max(0, int(old.get("pending_live", 0)) - 1))
            self._update(minus)

    def _try_lease(self) -> int | None:
        ensure_private_directory(self.root)
        fd = os.open(self._lease_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.chmod(self._lease_path, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            os.close(fd); return None

    @staticmethod
    def _release_fd(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)

    @contextmanager
    def evaluator_lease(self) -> Iterator[bool]:
        """Acquire only after a live-pending check and an atomic final recheck."""
        self.reconcile()
        blocked = False
        with self._state_lock.held():
            state = self._read()
            if state.get("corrupt") or self._suspect_evaluator_locked(state) or sum(int(r.get("pending_live", 0)) for r in state["owners"].values()):
                blocked = True
        if blocked:
            yield False; return
        fd = self._try_lease()
        if fd is None:
            yield False; return
        granted = False
        try:
            with self._state_lock.held():
                state = self._read()
                self._reconcile_locked(state)
                if not state.get("corrupt") and not self._suspect_evaluator_locked(state) and not sum(int(r.get("pending_live", 0)) for r in state["owners"].values()):
                    state["owners"][self.owner_id] = self._owner("evaluator")
                    state["revision"] = int(state.get("revision", 0)) + 1; self._write(state); granted = True
            if granted:
                with self.heartbeat_pump("evaluator"):
                    yield True
            else:
                yield False
        finally:
            if granted:
                self._update(lambda state: state["owners"].pop(self.owner_id, None))
            self._release_fd(fd)

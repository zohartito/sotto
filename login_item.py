"""Start Sotto.app at login with a per-user LaunchAgent (macOS).

Only the installed Sotto.app launcher is registered, so the running process
keeps the app's Microphone/Accessibility identity. launchd relaunches it after
a crash but not after Quit. Enabling never starts a second copy now: the
agent is only picked up at the next login. The label is distinct from the
maintainer's sealed LaunchAgent.
"""
from __future__ import annotations

import os
from pathlib import Path
import plistlib
import subprocess
import tempfile

LABEL = "org.sotto.alpha"


def plist_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def agent(executable: str) -> dict:
    if not os.path.isabs(executable):
        raise ValueError("the Sotto.app launcher path must be absolute")
    return {"Label": LABEL, "ProgramArguments": [executable], "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False}, "ThrottleInterval": 30,
            "ProcessType": "Interactive", "LimitLoadToSessionType": "Aqua"}


def enabled(home: Path | None = None) -> bool:
    return plist_path(home).is_file()


def enable(executable: str, home: Path | None = None) -> Path:
    path = plist_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{LABEL}-")
    try:
        with os.fdopen(fd, "wb") as handle:
            plistlib.dump(agent(executable), handle)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def disable(home: Path | None = None, *, unload: bool = False, run=subprocess.run) -> None:
    """Remove the agent so the next login does not start Sotto.

    Unticking the setting must not stop the Sotto that is running right now
    (it may be this very job), so only an uninstall also unloads it."""
    path = plist_path(home)
    if unload:
        run(["/bin/launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"], capture_output=True)
    path.unlink(missing_ok=True)

"""Launch at login: one per-user ``HKCU\\...\\Run`` value, fully reversible.

No service, no scheduled task, no admin rights.  The value starts the tray
app through ``win_launch.py`` with ``pythonw`` (no console window) and the
same data and model folders as the running app.  Task Manager's Startup apps
page lists it and can disable it there too.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import winreg

if sys.platform != "win32":
    raise ImportError("win_startup is Windows-only")

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "Sotto"
LAUNCHER = Path(__file__).resolve().with_name("win_launch.py")
MAX_COMMAND_CHARS = 260  # Run values beyond this are not reliably started


def gui_python(executable: str | Path = sys.executable) -> Path:
    """pythonw.exe next to the running interpreter (no console window)."""
    executable = Path(executable)
    candidate = executable.with_name("pythonw.exe")
    return candidate if candidate.is_file() else executable


def launch_command(data_dir: Path, hf_home: Path, *, python: Path | None = None,
                   launcher: Path = LAUNCHER) -> str:
    python = gui_python() if python is None else Path(python)
    return subprocess.list2cmdline([str(python), str(launcher), "--data-dir", str(data_dir),
                                    "--hf-home", str(hf_home)])


def current(*, key: str = RUN_KEY, name: str = VALUE_NAME) -> str | None:
    """The registered command, or None when launch at login is off."""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as handle:
            value, kind = winreg.QueryValueEx(handle, name)
    except FileNotFoundError:
        return None
    return value if kind in (winreg.REG_SZ, winreg.REG_EXPAND_SZ) else None


def enabled(*, key: str = RUN_KEY, name: str = VALUE_NAME) -> bool:
    return current(key=key, name=name) is not None


def set_enabled(on: bool, command: str, *, key: str = RUN_KEY, name: str = VALUE_NAME) -> None:
    """Add or remove exactly this one value; removing an absent value is fine."""
    if on:
        if len(command) > MAX_COMMAND_CHARS:
            raise ValueError("the install path is too long for a login entry "
                             f"({len(command)} > {MAX_COMMAND_CHARS} characters)")
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, key, 0, winreg.KEY_SET_VALUE) as handle:
            winreg.SetValueEx(handle, name, 0, winreg.REG_SZ, command)
        return
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key, 0, winreg.KEY_SET_VALUE) as handle:
            winreg.DeleteValue(handle, name)
    except FileNotFoundError:
        pass

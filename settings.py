"""User settings shared by the macOS and Windows apps: one private JSON file.

Command-line flags still override these for a single run. The existing
language-mode and engine-mode files stay the source of truth for those two
choices; this file holds everything the Settings window adds.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

from sotto_paths import DATA_DIR

SETTINGS_PATH = DATA_DIR / "settings.json"
TRIGGER_KEYS = ("fn", "right-option", "left-option", "right-cmd", "left-cmd",
                "right-ctrl", "left-ctrl", "right-shift", "left-shift")
INSERT_MODES = ("paste", "type")
SPACING_MODES = ("smart", "trailing", "none")
SPEEDS = ("accurate", "fast")
MAX_LANGUAGES = 12
DEFAULTS = {
    # macOS keeps the documented right-Option default; Windows the right Ctrl
    # the Windows alpha has always used.
    "trigger": "right-ctrl" if sys.platform == "win32" else "right-option",
    "hotkey": "" if sys.platform == "win32" else "ctrl-opt-d",
    "languages": [],          # Automatic chooses among these; empty = English
    "insert_mode": "paste",
    "spacing": "smart",
    "speed": "accurate",
    "launch_at_login": False,
}


def _valid(key: str, value):
    if key == "trigger":
        return value if value in TRIGGER_KEYS else None
    if key == "hotkey":
        return value if isinstance(value, str) and len(value) <= 40 else None
    if key == "languages":
        if not isinstance(value, list):
            return None
        codes = []
        for code in value:
            if isinstance(code, str) and 2 <= len(code) <= 3 and code.isalpha() and code.lower() not in codes:
                codes.append(code.lower())
        return codes[:MAX_LANGUAGES]
    if key == "insert_mode":
        return value if value in INSERT_MODES else None
    if key == "spacing":
        return value if value in SPACING_MODES else None
    if key == "speed":
        return value if value in SPEEDS else None
    if key == "launch_at_login":
        return value if isinstance(value, bool) else None
    return None


def load(path: Path | str = SETTINGS_PATH) -> dict:
    """Defaults overlaid with every valid saved value; bad values are ignored."""
    settings = {key: (list(value) if isinstance(value, list) else value) for key, value in DEFAULTS.items()}
    try:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings
    if isinstance(saved, dict):
        for key in DEFAULTS:
            if key in saved:
                value = _valid(key, saved[key])
                if value is not None:
                    settings[key] = value
    return settings


def save(updates: dict, path: Path | str = SETTINGS_PATH) -> dict:
    """Validate and persist changes atomically (owner-only); returns the result."""
    unknown = sorted(set(updates) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown setting: {unknown[0]}")
    settings = load(path)
    for key, value in updates.items():
        checked = _valid(key, value)
        if checked is None:
            raise ValueError(f"Invalid value for {key}")
        settings[key] = checked
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".settings-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(settings, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return settings

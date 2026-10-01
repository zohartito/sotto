"""Local storage locations; overrides keep alpha/test state separate."""
import os
from pathlib import Path
import sys


def _override(name: str):
    """A non-blank override path; a blank value counts as unset, never as cwd."""
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().absolute() if value else None


_data_override = _override("SOTTO_DATA_DIR")
if sys.platform == "win32":
    # Windows keeps the pre-alpha location: history, preferences and the model
    # cache all live under %APPDATA%\sotto unless overridden.
    DATA_DIR = _data_override or (
        _override("APPDATA") or Path.home() / "AppData" / "Roaming") / "sotto"
    MODEL_CACHE_DIR = _override("SOTTO_HF_HOME") or DATA_DIR / "huggingface"
else:
    DATA_DIR = _data_override or Path.home() / "Library/Application Support/sotto"
    # Preserve the existing cache location on case-sensitive volumes too.
    MODEL_CACHE_DIR = _override("SOTTO_HF_HOME") or (
        DATA_DIR / "huggingface" if _data_override
        else Path.home() / "Library/Application Support/Sotto/huggingface")

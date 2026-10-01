"""Start Menu and login entry point for the Windows tray app.

    pythonw win_launch.py --data-dir DIR [--hf-home DIR] [sotto_win options]

A shortcut or Run value cannot set environment variables, so this sets
SOTTO_DATA_DIR (and SOTTO_HF_HOME, default DIR\\huggingface) before any Sotto
module resolves its paths, then starts ``sotto_win`` with ``--tray``.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

if sys.platform != "win32":
    raise ImportError("win_launch is Windows-only")


def split_paths(argv: list[str]) -> tuple[str | None, str | None, list[str]]:
    """(data dir, model cache dir, remaining arguments)."""
    data_dir = hf_home = None
    rest: list[str] = []
    items = iter(argv)
    for item in items:
        if item in ("--data-dir", "--hf-home"):
            value = next(items, None)
            if not value:
                raise SystemExit(f"{item} needs a folder")
            if item == "--data-dir":
                data_dir = value
            else:
                hf_home = value
        else:
            rest.append(item)
    return data_dir, hf_home, rest


def main(argv: list[str] | None = None) -> None:
    data_dir, hf_home, rest = split_paths(list(sys.argv[1:] if argv is None else argv))
    if data_dir:
        os.environ["SOTTO_DATA_DIR"] = data_dir
        os.environ["SOTTO_HF_HOME"] = hf_home or str(Path(data_dir) / "huggingface")
    elif hf_home:
        os.environ["SOTTO_HF_HOME"] = hf_home
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sotto_win  # after the environment: sotto_paths reads it at import
    sotto_win.main(["--tray", *rest])


if __name__ == "__main__":
    main()

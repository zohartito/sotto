#!/usr/bin/env python3
"""Pinned Granite teacher adapter; no decoder output is logged."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import wave
from pathlib import Path


def pcm16_mono_16k(path: Path):
    import numpy as np
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16_000 or wav.getcomptype() != "NONE":
            raise ValueError("canonical_wav_required")
        raw = wav.readframes(wav.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def _import_root_tree(root: str) -> dict:
    """Exact no-symlink tree, byte-hashed; must match the probe child format."""
    info = os.lstat(root)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("unsafe_import_root")
    rows = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs.sort(); files.sort(); base = os.path.relpath(directory, root)
        for name in dirs + files:
            path = os.path.join(directory, name); entry = os.lstat(path)
            rel = name if base == "." else base + "/" + name
            if stat.S_ISLNK(entry.st_mode) or not (stat.S_ISDIR(entry.st_mode) or stat.S_ISREG(entry.st_mode)):
                raise ValueError("unsafe_import_root")
            row = {"path": rel, "type": "dir" if stat.S_ISDIR(entry.st_mode) else "file",
                   "marker": [entry.st_ino, entry.st_size, entry.st_mtime_ns, entry.st_ctime_ns]}
            if stat.S_ISREG(entry.st_mode):
                digest = hashlib.sha256()
                with open(path, "rb") as handle:
                    for block in iter(lambda: handle.read(1048576), b""): digest.update(block)
                row["sha256"] = digest.hexdigest()
            rows.append(row)
    return {"root": root, "marker": [info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns], "entries": rows}


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--audio", required=True); parser.add_argument("--snapshot", required=True)
    parser.add_argument("--identity", required=True); parser.add_argument("--decode", required=True)
    args = parser.parse_args()
    # ``-I -S`` starts this process stdlib-only.  The runner pipes the
    # receipt-frozen import-root trees on stdin; this child re-verifies every
    # byte of those trees BEFORE inserting any root, so a root changed after
    # the runner's validation cannot execute here.
    try:
        expected = json.load(sys.stdin).get("expected_import_roots")
        if (not isinstance(expected, list) or not expected or
                any(not isinstance(row, dict) or not os.path.isabs(str(row.get("root", ""))) for row in expected) or
                [_import_root_tree(str(row["root"])) for row in expected] != expected):
            raise ValueError("import_root")
    except Exception:
        sys.stderr.write("teacher_failed\n"); return 3
    for row in reversed(expected):
        sys.path.insert(0, str(row["root"]))
    try:
        decode = json.loads(args.decode)
        expected = {"prompt": "default_transcription", "temperature": 0, "top_p": 1.0, "top_k": 0, "max_tokens": 4096, "prefill_step_size": 2048}
        if decode != expected: raise ValueError("decode")
        snapshot = Path(args.snapshot)
        if not snapshot.is_absolute() or not snapshot.is_dir(): raise ValueError("snapshot")
        from mlx_audio.stt import load
        model = load(str(snapshot), strict=True)
        # Default transcription prompt with greedy temperature 0.  Supplying
        # numpy avoids mlx-audio's file reader and its optional codecs.
        answer = model.generate(pcm16_mono_16k(Path(args.audio)), temperature=0.0, top_p=1.0, top_k=0,
                                max_tokens=4096, prefill_step_size=2048, prompt=None, verbose=False, stream=False)
        text = getattr(answer, "text", None)
        if not isinstance(text, str): raise ValueError("result")
        sys.stdout.write(text.replace("\n", " ").replace("\r", " ").strip() + "\n")
        return 0
    except Exception:
        sys.stderr.write("teacher_failed\n")
        return 2


if __name__ == "__main__": raise SystemExit(main())

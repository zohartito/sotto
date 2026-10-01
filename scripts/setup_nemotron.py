#!/usr/bin/env python3
"""Explicit, pinned provisioning for Sotto's optional macOS Nemotron engine.

Run with venv/bin/python scripts/setup_nemotron.py. No runtime auto-downloads.
"""
from pathlib import Path
import hashlib
import io
import json
import os
import platform
import tarfile
import tempfile
import urllib.request
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sotto_paths import DATA_DIR


def install() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("This optional Sotto runtime is for Apple Silicon Macs.")
    spec = json.loads((Path(__file__).resolve().parents[1] / "config/nemotron-runtime.json").read_text())
    root = DATA_DIR / "nemotron" / spec["version"]
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    archive = urllib.request.urlopen(spec["archive_url"], timeout=60).read()
    if hashlib.sha256(archive).hexdigest() != spec["archive_sha256"]:
        raise RuntimeError("Nemotron runtime archive checksum mismatch")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for name, expected in spec["libraries"].items():
            data = tar.extractfile("nemo-speech/lib/" + name).read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError("Nemotron library checksum mismatch: " + name)
            target = root / "lib" / name
            target.parent.mkdir(exist_ok=True, mode=0o700)
            atomic_write(target, data)
        # Preserve upstream and third-party notices alongside the runtime.
        prefix = "nemo-speech/share/licenses/"
        for member in tar.getmembers():
            if member.isfile() and member.name.startswith(prefix):
                relative = Path(member.name.removeprefix(prefix))
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("Unsafe license archive member")
                target = root / "licenses" / relative
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                atomic_write(target, tar.extractfile(member).read())
    model = root / spec["model_file"]
    if not model.is_file() or digest(model) != spec["model_sha256"]:
        url = ("https://huggingface.co/" + spec["model_repo"] + "/resolve/"
               + spec["model_revision"] + "/" + spec["model_file"])
        print("Downloading pinned Nemotron English model (700 MB)...", flush=True)
        fd, name = tempfile.mkstemp(prefix=".model-", dir=root)
        temp = Path(name)
        try:
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=120) as incoming:
                while block := incoming.read(1024 * 1024):
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            if temp.stat().st_size != spec["model_size"] or digest(temp) != spec["model_sha256"]:
                raise RuntimeError("Nemotron model checksum mismatch")
            os.replace(temp, model)
        finally:
            temp.unlink(missing_ok=True)
    print(f"Verified Nemotron {spec['version']} and model: {root}")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_write(path, data):
    fd, name = tempfile.mkstemp(prefix=".install-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


if __name__ == "__main__":
    install()

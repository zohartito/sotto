"""Versioned identity for the complete local adaptive-runtime behavior boundary.

This deliberately hashes source bytes rather than an SCM revision: preflight
receipts must fail closed in a dirty or locally modified checkout, and the same
manifest is consumable later by sealed-release tooling.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


RUNTIME_SOURCE_SCHEMA = 1
# Ordered on purpose.  This is the local behavior surface that can capture
# audio, execute inference/teachers, score/promote/route, or build/run the
# calibration lane.  Launchd targets are included because they choose what
# source a worker actually starts.
RUNTIME_SOURCE_FILES = (
    "requirements.txt",
    "runtime_dependency_policy.json",
    "config/sotto-glossary.txt",
    "config/nemotron-runtime.json",
    "runtime_source_manifest.py",
    "sealed_release.py",
    "sealed_release_bootstrap.py",
    "offline_backup.py",
    "storage_lock.py",
    "offline_runtime.py",
    "sotto_paths.py",
    "dictionary.py",
    "settings.py",
    "settings_window.py",
    "progress.py",
    "login_item.py",
    "launchd_templates.py",
    "audio_codec.py",
    "comparator_spool.py",
    "vad.py",
    "speech_config.py",
    "nemotron_backend.py",
    "streaming_audio.py",
    "history.py",
    "learning.py",
    "inference_scheduler.py",
    "evaluation.py",
    "adaptive_learning.py",
    "silver_store.py",
    "silver_experiment.py",
    "deployment_controller.py",
    "speech_backends.py",
    "adaptive_runtime.py",
    "adaptive_worker.py",
    "calibration_manifest_builder.py",
    "calibration_worker.py",
    "teacher_consensus.py",
    "teacher_backends.py",
    "teacher_provision.py",
    "teachers/qwen3_adapter.py",
    "teachers/granite_adapter.py",
    "ui.py",
    "sotto.py",
    "launchd/com.zohartito.sotto.adaptive-silver.plist",
    "launchd/com.zohartito.sotto.app.plist",
)


def _root(root: Path | str | None) -> Path:
    value = Path(root) if root is not None else Path(__file__).resolve().parent
    try:
        info = os.lstat(value)
    except OSError as exc:
        raise RuntimeError("runtime source root is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError("runtime source root is unsafe")
    return value


def _safe_relative(value: Any) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError("runtime source allowlist is unsafe")
    path = Path(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError("runtime source allowlist is unsafe")
    return path


def runtime_source_manifest(root: Path | str | None = None) -> dict[str, Any]:
    """Return strict ordered per-file source identities and their aggregate."""
    base = _root(root)
    files = tuple(RUNTIME_SOURCE_FILES)
    if not files or len(files) != len(set(files)):
        raise RuntimeError("runtime source allowlist is ambiguous")
    rows: list[dict[str, str]] = []
    for value in files:
        relative = _safe_relative(value)
        target = base / relative
        try:
            info = os.lstat(target)
        except OSError as exc:
            raise RuntimeError("runtime source is missing") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("runtime source is unsafe")
        digest = hashlib.sha256()
        try:
            with target.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise RuntimeError("runtime source is unavailable") from exc
        rows.append({"path": value, "sha256": digest.hexdigest()})
    body = {"schema": RUNTIME_SOURCE_SCHEMA, "files": rows}
    aggregate = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {**body, "digest": aggregate}


def runtime_source_digest(root: Path | str | None = None) -> str:
    return str(runtime_source_manifest(root)["digest"])

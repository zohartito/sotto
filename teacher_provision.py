"""Create/verify the private immutable receipt consumed by the silver worker."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from storage_lock import ensure_private_directory
from history import _fsync_dir
from teacher_backends import GRANITE, QWEN, receipt_path, runtime_identity, sha256_file, snapshot_digest, teacher_snapshot
from teacher_consensus import policy_hash


def _receipt(family, interpreter: Path, snapshot: Path, adapter: Path) -> dict:
    import teacher_consensus
    # Capture runtime identity once so the package map and interpreter identity
    # cannot describe two different probe moments in one receipt.
    identity = runtime_identity(interpreter, family.critical_packages)
    checked_snapshot = teacher_snapshot(family, snapshot)
    return {"family": family.stable_id, "repo": family.repo, "revision": family.revision,
            "interpreter": str(interpreter), "interpreter_identity": identity,
            "snapshot_path": str(checked_snapshot.path), "snapshot_digest": snapshot_digest(family, checked_snapshot.path),
            "package_versions": identity["packages"], "adapter": str(adapter),
            "adapter_hash": sha256_file(adapter), "decode": family.decode,
            "canonicalizer_hash": policy_hash(), "canonicalizer_source_hash": sha256_file(Path(teacher_consensus.__file__))}


def provision(base: Path, qwen_python: Path, qwen_snapshot: Path, granite_python: Path, granite_snapshot: Path) -> Path:
    root = Path(__file__).resolve().parent
    rows = {QWEN.stable_id: _receipt(QWEN, qwen_python, qwen_snapshot, root / "teachers" / "qwen3_adapter.py"),
            GRANITE.stable_id: _receipt(GRANITE, granite_python, granite_snapshot, root / "teachers" / "granite_adapter.py")}
    path = receipt_path(base); ensure_private_directory(path.parent)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        json.dump(rows, out, sort_keys=True, separators=(",", ":")); out.flush(); os.fsync(out.fileno())
    os.chmod(temp, 0o600); os.replace(temp, path); os.chmod(path, 0o600); _fsync_dir(path.parent)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Provision verified offline Sotto teacher receipts")
    p.add_argument("--base-dir", required=True); p.add_argument("--qwen-python", required=True); p.add_argument("--qwen-snapshot", required=True)
    p.add_argument("--granite-python", required=True); p.add_argument("--granite-snapshot", required=True)
    args = p.parse_args(argv)
    provision(Path(args.base_dir), Path(args.qwen_python), Path(args.qwen_snapshot), Path(args.granite_python), Path(args.granite_snapshot))
    return 0

if __name__ == "__main__": raise SystemExit(main())

"""Offline, nonconstructing backup and anti-rollback restore tooling.

This module deliberately does not import or instantiate any Sotto store.  It
works on explicit fixture/user roots only; callers must never pass a live root
without arranging their own operational downtime.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from history import _fsync_dir

BACKUP_SCHEMA = 1


def _hash(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def _json(value: Any) -> bytes: return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _private_dir(path: Path, *, create: bool = False) -> None:
    if create:
        path.mkdir(parents=True, mode=0o700, exist_ok=False)
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        raise RuntimeError("unsafe backup directory")


def _regular(path: Path, *, mode: int = 0o600) -> os.stat_result:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != mode:
        raise RuntimeError("unsafe backup artifact")
    return info


def _read_regular_bytes(path: Path, *, mode: int = 0o600) -> tuple[bytes, tuple[int, int, int, int, str]]:
    """Read and identify one final component without ever following a link.

    The descriptor, not a prior ``lstat`` path observation, is the authority
    for bytes copied into a backup or compared during restore.
    """
    flags=os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd=os.open(path,flags)
    except OSError as exc:
        raise RuntimeError("unsafe backup artifact") from exc
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != mode:
            raise RuntimeError("unsafe backup artifact")
        chunks=[]; digest=hashlib.sha256()
        while True:
            block=os.read(fd,1024*1024)
            if not block: break
            chunks.append(block); digest.update(block)
        data=b"".join(chunks)
        if len(data) != info.st_size: raise RuntimeError("backup artifact changed while read")
        return data,(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,digest.hexdigest())
    finally:
        os.close(fd)


def _relative(path: Path) -> str:
    value = path.as_posix()
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError("unsafe backup path")
    return value


@contextmanager
def _quiescent_locks(root: Path) -> Iterator[None]:
    """Acquire the finite authority lock chain without creating source files."""
    _private_dir(root)
    adaptive=root / "adaptive-learning"
    paths=[root / ".sotto.lock"]
    if (adaptive / "state.json").exists(): paths.append(adaptive / ".adaptive-state.lock")
    if (adaptive / ".evaluator-process.lock").exists(): paths.append(adaptive / ".evaluator-process.lock")
    fds=[]
    try:
        for path in paths:
            _regular(path)
            fd=os.open(path,os.O_RDWR | getattr(os,"O_NOFOLLOW",0))
            try:
                fcntl.flock(fd,fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd); raise RuntimeError("backup authority lock is busy")
            fds.append(fd)
        yield
    finally:
        for fd in reversed(fds):
            fcntl.flock(fd,fcntl.LOCK_UN); os.close(fd)


def _validate_json(path: Path) -> None:
    try:
        raw,_ = _read_regular_bytes(path)
        if path.suffix == ".json":
            if not isinstance(json.loads(raw), (dict, list)): raise RuntimeError("invalid JSON artifact")
        elif path.suffix == ".jsonl":
            for line in raw.splitlines():
                if line and not isinstance(json.loads(line), dict): raise RuntimeError("invalid JSONL artifact")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("invalid JSON artifact") from exc


def _sqlite_check(path: Path, *, immutable: bool = False) -> None:
    """Integrity-check a store; ``immutable`` opens without creating WAL/SHM.

    A plain read-only connection on a WAL-mode database still materialises
    ``-wal``/``-shm`` side files.  Inside a verified bundle those would be
    rejected as extra artifacts by the next verification, so bundle and stage
    copies are inspected immutably; the live store keeps a normal read-only
    open because its side files legitimately exist and are excluded anyway.
    """
    _regular(path)
    db = sqlite3.connect(f"file:{path}?mode=ro&immutable=1" if immutable else f"file:{path}?mode=ro", uri=True)
    try:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok": raise RuntimeError("SQLite integrity failure")
        if db.execute("PRAGMA foreign_key_check").fetchone() is not None: raise RuntimeError("SQLite foreign key failure")
        if db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() is None: raise RuntimeError("SQLite schema missing")
    finally: db.close()


def _walk_source(root: Path) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        current = Path(directory)
        _private_dir(current)
        for name in sorted(list(dirs)):
            child = current / name
            if stat.S_ISLNK(os.lstat(child).st_mode): raise RuntimeError("source symlink refused")
            rel = _relative(child.relative_to(root))
            if rel in {"huggingface", "teacher-runtimes", "models", "releases", "inference-scheduler",
                       "learning/.staging", "adaptive-learning/tmp"}:
                dirs.remove(name)
                continue
            if not _allowed_state_directory(_relative(child.relative_to(root))):
                raise RuntimeError("unexpected mutable-state directory")
        for name in sorted(files):
            source = current / name
            rel = _relative(source.relative_to(root))
            if rel in {".sotto.lock", "adaptive-learning/.adaptive-state.lock", "adaptive-learning/.evaluator-process.lock"} or rel.endswith("silver.sqlite3-wal") or rel.endswith("silver.sqlite3-shm"):
                continue
            _regular(source)
            if not _allowed_state_path(rel):
                raise RuntimeError("unexpected mutable-state artifact")
            if source.suffix in {".json", ".jsonl"}: _validate_json(source)
            result.append((rel, source))
    sqlite = root / "adaptive-learning" / "silver" / "silver.sqlite3"
    if sqlite.exists(): _sqlite_check(sqlite)
    return result


def _allowed_state_path(relative: str) -> bool:
    """Finite personal-state allowlist; caches, models, locks and temps stay out."""
    path=Path(relative); parts=path.parts
    if relative in {"history.jsonl", "learning/learning.jsonl", "learning/pending-gold.jsonl", "adaptive-learning/state.json"}:
        return True
    if len(parts)==2 and parts[0] in {"audio","audio-raw"} and path.suffix==".wav": return True
    if len(parts)==3 and parts[0]=="learning" and parts[1] in {"audio","raw"} and path.suffix==".wav": return True
    if len(parts)==3 and parts[:2]==("adaptive-learning","preflight") and path.suffix==".json": return True
    if parts[:3]==("adaptive-learning","silver","evidence") and path.suffix==".wav": return True
    if parts[:4]==("adaptive-learning","silver","evidence","comparators") and path.suffix==".wav": return True
    return relative in {"adaptive-learning/silver/silver.sqlite3", "adaptive-learning/silver/deployment.jsonl", "adaptive-learning/silver/provisional_silver.json", "adaptive-learning/silver/experiments.jsonl", "adaptive-learning/silver/teacher-receipts.json"}


def _allowed_state_directory(relative: str) -> bool:
    return relative in {"audio", "audio-raw", "learning", "learning/audio", "learning/raw",
                        "adaptive-learning", "adaptive-learning/preflight", "adaptive-learning/silver",
                        "adaptive-learning/silver/evidence", "adaptive-learning/silver/evidence/comparators"}


def _copy_private(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    parent = destination.parent
    while parent.name and parent.exists():
        if parent.name.startswith(".") and parent.name.endswith(".tmp"):
            break
        os.chmod(parent, 0o700)
        if parent == parent.parent:
            break
        parent = parent.parent
    data,signature = _read_regular_bytes(source)
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(data); out.flush(); os.fsync(out.fileno())
    return signature[-1]


def _write_private_bytes(destination: Path, data: bytes) -> str:
    destination.parent.mkdir(parents=True,mode=0o700,exist_ok=True)
    os.chmod(destination.parent,0o700)
    fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"wb") as out: out.write(data); out.flush(); os.fsync(out.fileno())
    return _hash(data)


def backup(source_root: Path | str, output: Path | str) -> Path:
    """Atomically create a verified private bundle from an explicit source root."""
    source, target = Path(source_root), Path(output)
    if target.exists() or target.is_symlink() or not target.parent.is_dir(): raise RuntimeError("backup target must be fresh")
    _private_dir(source); _private_dir(target.parent)
    stage = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    _private_dir(stage, create=True)
    try:
        with _quiescent_locks(source):
            # Lock order is fixed above; the SQLite write barrier remains held
            # through every included file, its WAL-inclusive serialize image,
            # and final fd signatures.
            sql = source / "adaptive-learning" / "silver" / "silver.sqlite3"
            snapshot = sqlite3.connect(sql,timeout=0.2) if sql.exists() else None
            try:
                if snapshot: snapshot.execute("BEGIN IMMEDIATE")
                files = _walk_source(source)
                before={rel:_read_regular_bytes(path)[1] for rel,path in files}
                for rel, path in files:
                    dest = stage / "data" / rel
                    if path == sql:
                        if snapshot is None: raise RuntimeError("SQLite snapshot unavailable")
                        _write_private_bytes(dest,snapshot.serialize())
                    else: _copy_private(path, dest)
                for rel,path in files:
                    if _read_regular_bytes(path)[1] != before[rel]: raise RuntimeError("source changed during backup")
            finally:
                if snapshot:
                    if snapshot.in_transaction: snapshot.rollback()
                    snapshot.close()
        entries=[{"path":"data","type":"dir","mode":0o700}]
        for path in sorted((stage / "data").rglob("*")):
            rel=_relative(path.relative_to(stage))
            if path.is_dir(): _private_dir(path); entries.append({"path":rel,"type":"dir","mode":0o700})
            else:
                data,signature=_read_regular_bytes(path); entries.append({"path":rel,"type":"file","mode":0o600,"sha256":signature[-1],"size":len(data)})
        body={"schema":BACKUP_SCHEMA,"entries":entries}; body["digest"]=_hash(_json(body))
        manifest=stage / "backup.json"; fd=os.open(manifest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"wb") as out: out.write(_json(body)); out.flush(); os.fsync(out.fileno())
        final=target.parent / body["digest"]
        if final.exists() or final.is_symlink(): raise RuntimeError("content-addressed backup already exists")
        _fsync_dir(stage); os.rename(stage,final); _fsync_dir(target.parent)
        return final
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True); raise


def verify(bundle: Path | str) -> dict[str, Any]:
    root=Path(bundle); _private_dir(root); manifest=root / "backup.json"; _regular(manifest)
    raw,_=_read_regular_bytes(manifest); value=json.loads(raw)
    if not isinstance(value,dict) or set(value)!={"schema","entries","digest"} or value.get("schema") != BACKUP_SCHEMA: raise RuntimeError("backup manifest malformed")
    body={"schema":value["schema"],"entries":value["entries"]}
    if value["digest"] != _hash(_json(body)) or not isinstance(value["entries"],list): raise RuntimeError("backup digest mismatch")
    if root.name != value["digest"]: raise RuntimeError("backup directory identity mismatch")
    seen=set()
    for entry in value["entries"]:
        if not isinstance(entry,dict) or not isinstance(entry.get("path"),str) or entry["path"] in seen: raise RuntimeError("backup entry malformed")
        seen.add(entry["path"]); target=root / _relative(Path(entry["path"]))
        if entry.get("type")=="dir":
            if set(entry)!={"path","type","mode"} or entry.get("mode") != 0o700: raise RuntimeError("backup directory schema")
            _private_dir(target)
        elif entry.get("type")=="file":
            if set(entry)!={"path","type","mode","sha256","size"} or entry.get("mode") != 0o600 or not isinstance(entry.get("size"),int) or entry["size"] < 0 or not isinstance(entry.get("sha256"),str) or len(entry["sha256"]) != 64: raise RuntimeError("backup file schema")
            data,signature=_read_regular_bytes(target)
            if entry.get("sha256") != signature[-1] or entry.get("size") != len(data): raise RuntimeError("backup file mismatch")
        else: raise RuntimeError("backup entry type")
    actual={_relative(p.relative_to(root)) for p in root.rglob("*")}
    if actual != seen | {"backup.json"}: raise RuntimeError("backup extra artifact")
    _sqlite_check(root / "data" / "adaptive-learning" / "silver" / "silver.sqlite3", immutable=True)
    return value


def _force_shadow(root: Path) -> None:
    silver=root / "adaptive-learning" / "silver"; dbpath=silver / "silver.sqlite3"; _sqlite_check(dbpath, immutable=True)
    db=sqlite3.connect(dbpath)
    try:
        db.execute("PRAGMA foreign_keys=ON"); db.execute("BEGIN IMMEDIATE")
        epoch=int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])+1
        db.execute("UPDATE meta SET value=? WHERE key='clear_epoch'",(str(epoch),))
        db.execute("INSERT INTO epochs(clear_epoch,reason,ts) VALUES(?,?,strftime('%s','now'))",(epoch,"offline_restore"))
        # A valid backup may carry an older store whose lazy migrations (for
        # example ``comparator_intents``) have not run yet; there is nothing
        # to purge from a volatile table that does not exist yet.
        for table in ("runtime_observations","runtime_pairs","route_assignments","comparator_intents"):
            if db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?",(table,)).fetchone() is not None:
                db.execute(f"DELETE FROM {table}")
        # Prepared delivery adoption is an internal comparator outbox.  It
        # cannot survive forced-shadow restore after its exact intent/spool is
        # removed, or runtime recovery would remain permanently fenced.
        db.execute("DELETE FROM meta WHERE key LIKE 'comparator_adoption:%'")
        db.execute("INSERT INTO cohort_authorization(cohort_id,disposition,updated_ts) SELECT cohort_id,'consumed',strftime('%s','now') FROM cohorts WHERE status IN ('evaluated_passed','evaluated_failed') ON CONFLICT(cohort_id) DO UPDATE SET disposition='consumed',updated_ts=excluded.updated_ts")
        db.commit()
    except BaseException: db.rollback(); raise
    finally: db.close()
    comparators=silver / "evidence" / "comparators"
    if comparators.exists():
        info=os.lstat(comparators)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode): raise RuntimeError("unsafe comparator spool")
        for item in sorted(comparators.rglob("*"),reverse=True):
            detail=os.lstat(item)
            if stat.S_ISLNK(detail.st_mode) or not stat.S_ISREG(detail.st_mode): raise RuntimeError("unsafe comparator spool")
            item.unlink()
        comparators.rmdir()
    deploy=silver / "deployment.jsonl"; _regular(deploy); deploy_bytes,_=_read_regular_bytes(deploy); rows=[line for line in deploy_bytes.decode("utf-8").splitlines() if line]
    last=json.loads(rows[-1]) if rows else {}; revision=int(last.get("revision",0))+1; generation=int(last.get("route_generation",0))+1
    state={"schema":1,"revision":revision,"tier":"shadow_only","current":None,"lkg":None,"cohort_id":None,"canary":None,"calibration":None,"route_generation":generation,"audit":[{"event":"offline_restore_shadow"}]}
    payload=(json.dumps(state,sort_keys=True)+"\n").encode(); temp=deploy.with_name(".deployment.restore.tmp")
    fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"wb") as out: out.write(payload); out.flush(); os.fsync(out.fileno())
    os.replace(temp,deploy); (silver / "provisional_silver.json").unlink(missing_ok=True); _fsync_dir(silver)


def restore(bundle: Path | str, target: Path | str) -> Path:
    manifest=verify(bundle); source=Path(bundle); target=Path(target)
    if target.exists() or target.is_symlink() or not target.parent.is_dir(): raise RuntimeError("restore target must be fresh")
    _private_dir(target.parent); stage=target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"; _private_dir(stage,create=True)
    try:
        entries=manifest["entries"]
        for entry in entries:
            if entry["type"] == "dir":
                dest=stage / _relative(Path(entry["path"])); dest.mkdir(parents=True,mode=0o700,exist_ok=False); os.chmod(dest,0o700)
        for entry in entries:
            if entry["type"] != "file": continue
            rel=_relative(Path(entry["path"])); original=source / rel
            _regular(original)
            dest=stage / rel
            digest=_copy_private(original,dest)
            if digest != entry["sha256"] or dest.stat().st_size != entry["size"]: raise RuntimeError("backup changed during restore")
        restored=stage / "data"; _force_shadow(restored); _fsync_dir(restored); os.rename(restored,target); shutil.rmtree(stage,ignore_errors=True); _fsync_dir(target.parent); return target
    except BaseException: shutil.rmtree(stage,ignore_errors=True); raise


def main() -> None:
    parser=argparse.ArgumentParser(description="offline Sotto backup/restore")
    parser.add_argument("command",choices=("backup","verify","restore")); parser.add_argument("--source"); parser.add_argument("--bundle",required=True); parser.add_argument("--target")
    args=parser.parse_args()
    if args.command=="backup":
        if not args.source: parser.error("backup requires --source")
        print(backup(args.source,args.bundle)); return
    if args.command=="verify": verify(args.bundle); print("verified"); return
    if not args.target: parser.error("restore requires --target")
    print(restore(args.bundle,args.target))

if __name__ == "__main__": main()

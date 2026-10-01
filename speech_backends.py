"""Local, lazy speech backends used by adaptive English selection.

This module contains no eager ML imports and never talks to a service.  The
preflight receipt is deliberately more specific than a repository name: a
moving Hugging Face ref cannot be used by an adaptive generation.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import subprocess
import sys
import tempfile
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from audio_codec import AudioIdentity, PreparedAudio, prepare_canonical, read_canonical_wav, write_canonical_wav
from evaluation import PAIRED_EVALUATOR_ID
from history import _fsync_dir
from offline_runtime import offline_requested, offline_subprocess_env
from runtime_source_manifest import RUNTIME_SOURCE_SCHEMA, runtime_source_digest
from storage_lock import ensure_private_directory, ensure_private_file

WHISPER_ID = "whisper-large-v3-turbo-en"
WHISPER_GLOSSARY_ID = "whisper-large-v3-turbo-en-glossary"
PARAKEET_ID = "parakeet-tdt-0.6b-v2-en"
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
PARAKEET_REPO = "mlx-community/parakeet-tdt-0.6b-v2"
DECODE_SETTINGS = {"language": "en", "condition_on_previous_text": False,
                   "hallucination_silence_threshold": 2.0, "word_timestamps": True}


def evaluator_hash() -> str:
    """Compatibility name for the complete versioned runtime source contract."""
    return runtime_source_digest()


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("mlx-whisper", "parakeet-mlx", "numpy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _bound_regular_binary(configured: Path | str, *, label: str) -> tuple[Path, Path, os.stat_result]:
    """Resolve one configured absolute executable to a strict final target."""
    raw=Path(configured)
    if not raw.is_absolute():
        raise RuntimeError(f"{label} path is not absolute")
    try:
        resolved=raw.resolve(strict=True); info=resolved.lstat()
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable") from exc
    # A configured symlink (venv Python/Homebrew ffmpeg) is acceptable only
    # because both its configured and resolved targets are bound below.
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or not os.access(resolved,os.X_OK):
        raise RuntimeError(f"{label} target is unsafe")
    return raw,resolved,info


def _sha256_regular(path: Path) -> str:
    try: info=path.lstat()
    except OSError as exc: raise RuntimeError("runtime binary unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError("runtime binary is unsafe")
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def _ffmpeg_runtime_identity(*, ffmpeg_path: Path | str | None = None,
                             run: Callable[..., Any] = subprocess.run) -> dict[str, str]:
    configured=Path(ffmpeg_path or os.environ.get("SOTTO_FFMPEG","/opt/homebrew/bin/ffmpeg"))
    raw,resolved,_info=_bound_regular_binary(configured,label="ffmpeg")
    try:
        completed=run([str(resolved),"-version"],shell=False,check=False,text=True,capture_output=True,
                      env=offline_subprocess_env(),timeout=10)
        stdout=completed.stdout
    except (OSError,subprocess.SubprocessError) as exc:
        raise RuntimeError("ffmpeg version unavailable") from exc
    if getattr(completed,"returncode",1) != 0 or not isinstance(stdout,str) or len(stdout)>4096:
        raise RuntimeError("ffmpeg version unavailable")
    first=stdout.splitlines()[0] if stdout.splitlines() else ""
    if not first.startswith("ffmpeg version ") or len(first)>512:
        raise RuntimeError("ffmpeg version malformed")
    return {"configured_path":str(raw),"resolved_path":str(resolved),"sha256":_sha256_regular(resolved),
            "version_first_line":first,"version_sha256":hashlib.sha256(first.encode("utf-8")).hexdigest()}


def _candidate_runtime_identity(stable_id: str, *, python_executable: Path | str | None = None,
                                ffmpeg_path: Path | str | None = None,
                                package_versions: dict[str,str|None] | None = None,
                                run: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Strict, content-free local runtime identity for one candidate family."""
    if stable_id not in {WHISPER_ID,WHISPER_GLOSSARY_ID,PARAKEET_ID}:
        raise ValueError("unknown adaptive candidate")
    raw_python,resolved_python,_info=_bound_regular_binary(python_executable or sys.executable,label="python")
    all_packages=package_versions if package_versions is not None else _package_versions()
    backend_packages=("mlx-whisper","numpy") if stable_id.startswith("whisper") else ("parakeet-mlx","numpy")
    if not isinstance(all_packages,dict) or set(all_packages) != {"mlx-whisper","parakeet-mlx","numpy"} or any(value is not None and not isinstance(value,str) for value in all_packages.values()):
        raise RuntimeError("candidate package identity malformed")
    return {"schema":1,"python":{"configured_path":str(raw_python),"resolved_path":str(resolved_python),
            "sha256":_sha256_regular(resolved_python),"version":sys.version,
            "version_info":[sys.version_info.major,sys.version_info.minor,sys.version_info.micro,sys.version_info.releaselevel,sys.version_info.serial],
            "implementation":platform.python_implementation(),"implementation_name":sys.implementation.name,
            "cache_tag":str(sys.implementation.cache_tag or ""),"unicode_version":unicodedata.unidata_version},
            "packages":{name:all_packages[name] for name in backend_packages},
            "ffmpeg":_ffmpeg_runtime_identity(ffmpeg_path=ffmpeg_path,run=run)}


def _candidate_runtime_marker(stable_id: str) -> tuple[Any, ...]:
    """Cheap identity-cache key; any executable/package drift forces rebind."""
    raw_python,resolved_python,python_info=_bound_regular_binary(sys.executable,label="python")
    raw_ffmpeg,resolved_ffmpeg,ffmpeg_info=_bound_regular_binary(os.environ.get("SOTTO_FFMPEG","/opt/homebrew/bin/ffmpeg"),label="ffmpeg")
    packages=_package_versions(); backend=("mlx-whisper","numpy") if stable_id.startswith("whisper") else ("parakeet-mlx","numpy")
    return (stable_id,str(raw_python),str(resolved_python),python_info.st_dev,python_info.st_ino,python_info.st_size,python_info.st_mtime_ns,python_info.st_ctime_ns,
            str(raw_ffmpeg),str(resolved_ffmpeg),ffmpeg_info.st_dev,ffmpeg_info.st_ino,ffmpeg_info.st_size,ffmpeg_info.st_mtime_ns,ffmpeg_info.st_ctime_ns,
            sys.version,sys.implementation.cache_tag,unicodedata.unidata_version,tuple((name,packages[name]) for name in backend))


def _repo_cache_name(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


def _safe_revision(revision: str) -> str:
    if (not isinstance(revision,str) or not revision or revision in {".","..","main","refs/main"} or
            "/" in revision or "\\" in revision or Path(revision).name != revision):
        raise RuntimeError("adaptive manifest lacks immutable snapshot revision")
    return revision


def _safe_repo(repo: str) -> str:
    parts=repo.split("/") if isinstance(repo,str) else []
    if len(parts) != 2 or any(not part or part in {".",".."} or "\\" in part for part in parts):
        raise RuntimeError("adaptive manifest repository is unsafe")
    return repo


def _strict_directory(path: Path, *, label: str) -> None:
    try: info=path.lstat()
    except OSError as exc: raise FileNotFoundError(f"{label} missing") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"unsafe {label}")


def _confined_blob_target(path: Path, blobs_resolved: Path) -> Path:
    """One HF dedupe link: its target must be a real file in this model's blobs.

    The Hub cache stores snapshot files as symlinks into the sibling ``blobs``
    directory.  Accept exactly that layout and nothing else: the resolved
    target stays inside this model's blobs, is a regular file, and no path
    component below blobs is itself a symlink that could change later.
    """
    try:
        link=os.readlink(path)
        target=(path.parent/link).resolve(strict=True) if not os.path.isabs(link) else Path(link).resolve(strict=True)
        target.relative_to(blobs_resolved)
        info=os.lstat(target)
    except (OSError,ValueError) as exc:
        raise RuntimeError("adaptive snapshot link escapes model blobs") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError("adaptive snapshot link escapes model blobs")
    cursor=blobs_resolved
    for component in target.relative_to(blobs_resolved).parts:
        cursor=cursor/component
        try: mode=os.lstat(cursor).st_mode
        except OSError as exc: raise RuntimeError("adaptive snapshot link escapes model blobs") from exc
        if stat.S_ISLNK(mode): raise RuntimeError("adaptive snapshot link escapes model blobs")
    return target


def _snapshot_blobs_resolved(snapshot: Path) -> Path:
    blobs=snapshot.parent.parent/"blobs"
    _strict_directory(blobs,label="adaptive snapshot blobs")
    try: return blobs.resolve(strict=True)
    except OSError as exc: raise RuntimeError("unsafe adaptive snapshot") from exc


def _strict_snapshot(snapshot: Path) -> None:
    _strict_directory(snapshot,label="adaptive snapshot")
    blobs_resolved=None
    try:
        for path in snapshot.rglob("*"):
            info=path.lstat()
            if stat.S_ISLNK(info.st_mode):
                if blobs_resolved is None: blobs_resolved=_snapshot_blobs_resolved(snapshot)
                _confined_blob_target(path,blobs_resolved); continue
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise RuntimeError("unsafe adaptive snapshot entry")
    except OSError as exc:
        raise RuntimeError("unsafe adaptive snapshot") from exc


def _snapshot_root(repo: str, hf_home: Path | str | None = None) -> tuple[Path,Path]:
    home=Path(hf_home or os.environ.get("SOTTO_HF_HOME") or os.environ.get("HF_HOME") or Path.home()/".cache"/"huggingface")
    if not home.is_absolute(): raise RuntimeError("HF cache root is not absolute")
    _safe_repo(repo); _strict_directory(home,label="HF cache root")
    hub=home/"hub"; model=hub/_repo_cache_name(repo); snapshots=model/"snapshots"
    for path,label in ((hub,"HF hub"),(model,"HF repository cache"),(snapshots,"HF snapshots")):
        _strict_directory(path,label=label)
    return model,snapshots


def resolve_snapshot(repo: str, hf_home: Path | str | None = None, revision: str | None = None) -> tuple[str, Path]:
    """Resolve an immutable cached snapshot; never silently accept refs/main."""
    root,snapshots=_snapshot_root(repo,hf_home)
    if revision:
        revision=_safe_revision(revision); path=snapshots/revision
        _strict_snapshot(path)
        return revision, path
    ref = root / "refs" / "main"
    try:
        if ref.exists() and (ref.is_symlink() or not stat.S_ISREG(ref.lstat().st_mode)): raise RuntimeError("unsafe snapshot ref")
        resolved = ref.read_text("utf-8").strip() if ref.exists() else ""
    except OSError as exc:
        raise RuntimeError("unsafe snapshot ref") from exc
    if not resolved:
        choices=[]
        for path in snapshots.iterdir():
            try: info=path.lstat()
            except OSError: continue
            if stat.S_ISLNK(info.st_mode): raise RuntimeError("unsafe snapshot entry")
            if stat.S_ISDIR(info.st_mode): choices.append(path.name)
        if len(choices) == 1:
            resolved = choices[0]
        else:
            raise RuntimeError(f"exact snapshot revision is ambiguous for {repo}; preflight a pinned snapshot")
    resolved=_safe_revision(resolved); path=snapshots/resolved; _strict_snapshot(path)
    return resolved, path


def _safe_snapshot_digest(snapshot: Path) -> str:
    """Exact local snapshot identity, including contents (never user audio)."""
    try:
        root=snapshot.lstat()
    except OSError as exc:
        raise RuntimeError("unsafe adaptive snapshot") from exc
    if stat.S_ISLNK(root.st_mode) or not stat.S_ISDIR(root.st_mode):
        raise RuntimeError("unsafe adaptive snapshot")
    rows: list[str] = []
    blobs_resolved=None
    for path in sorted(snapshot.rglob("*")):
        info=path.lstat()
        if stat.S_ISLNK(info.st_mode):
            # Bind an HF blob-dedupe link by its link text plus the exact
            # confined target bytes; anything else fails closed.
            if blobs_resolved is None: blobs_resolved=_snapshot_blobs_resolved(snapshot)
            target=_confined_blob_target(path,blobs_resolved)
            target_info=os.lstat(target)
            file_hash = hashlib.sha256()
            with target.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    file_hash.update(block)
            rows.append(f"L\0{path.relative_to(snapshot).as_posix()}\0{os.readlink(path)}\0{target_info.st_size}\0{target_info.st_mtime_ns}\0{file_hash.hexdigest()}")
        elif stat.S_ISREG(info.st_mode):
            file_hash = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    file_hash.update(block)
            rows.append(f"{path.relative_to(snapshot).as_posix()}\0{info.st_size}\0{info.st_mtime_ns}\0{file_hash.hexdigest()}")
        elif not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("unsafe adaptive snapshot entry")
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def candidate_manifest(stable_id: str, *, glossary_terms: tuple[str, ...] = (), revision: str,
                       package_versions: dict[str, str | None] | None = None,
                       runtime_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    if stable_id not in {WHISPER_ID, WHISPER_GLOSSARY_ID, PARAKEET_ID}:
        raise ValueError("unknown adaptive candidate")
    backend = "whisper" if stable_id.startswith("whisper") else "parakeet"
    repo = WHISPER_REPO if backend == "whisper" else PARAKEET_REPO
    glossary = tuple(glossary_terms) if stable_id == WHISPER_GLOSSARY_ID else ()
    if stable_id == WHISPER_GLOSSARY_ID and not glossary:
        raise ValueError("glossary candidate requires nonempty terms")
    glossary_identity = hashlib.sha256("\n".join(glossary).encode("utf-8")).hexdigest() if glossary else None
    result={"stable_id": stable_id, "backend": backend, "repo": repo, "revision": revision,
            "package_versions": package_versions or _package_versions(), "decode_settings": dict(DECODE_SETTINGS),
            "language": "en", "glossary_hash": glossary_identity, "glossary_identity": glossary_identity,
            "evaluator_id": PAIRED_EVALUATOR_ID, "evaluator_hash": evaluator_hash(),
            "runtime_source_schema": RUNTIME_SOURCE_SCHEMA, "runtime_source_digest": runtime_source_digest(),
            "glossary_terms": glossary}
    if runtime_identity is not None: result["runtime_identity"]=runtime_identity
    return result


class PreflightReceipts:
    """Atomic private receipts.  A failed preflight never replaces one."""
    def __init__(self, base_dir: Path | str) -> None:
        self.root = Path(base_dir) / "adaptive-learning" / "preflight"
        ensure_private_directory(self.root.parent)
        ensure_private_directory(self.root)
        for receipt in self.root.glob("*.json"):
            ensure_private_file(receipt)
        # Cache is deliberately keyed by a strong lstat manifest (device,
        # inode, size, mtime *and* ctime for every file).  It avoids a full
        # multi-GiB tree rehash on every live request, but any ordinary
        # replacement/tamper invalidates the key and forces the receipt's
        # byte-for-byte digest again.  The remaining same-inode/metadata
        # adversary is outside the local immutable-artifact threat model;
        # receipt snapshots are private, exact pinned artifacts and every
        # process restart starts with an empty cache.
        self._validation_cache: dict[tuple[str, tuple[str, ...]], bool] = {}
        self._runtime_cache: dict[str, tuple[tuple[Any,...],dict[str,Any]]] = {}

    def path(self, stable_id: str) -> Path:
        return self.root / f"{stable_id}.json"

    def load(self, stable_id: str) -> dict[str, Any] | None:
        try:
            value = json.loads(self.path(stable_id).read_text("utf-8"))
            return value if isinstance(value, dict) and value.get("success") is True else None
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _current_runtime_identity(self, stable_id: str) -> dict[str, Any]:
        marker=_candidate_runtime_marker(stable_id); cached=self._runtime_cache.get(stable_id)
        if cached is not None and cached[0] == marker:
            return cached[1]
        identity=_candidate_runtime_identity(stable_id)
        self._runtime_cache[stable_id]=(marker,identity)
        return identity

    def valid_manifest(self, manifest: dict[str, Any]) -> bool:
        receipt = self.load(str(manifest["stable_id"]))
        try:
            if not receipt or not _receipt_manifest_identity_valid(receipt,manifest,runtime_identity=self._current_runtime_identity(str(manifest["stable_id"]))):
                return False
        except (RuntimeError,ValueError,OSError,KeyError):
            return False
        stored = Path(str(receipt.get("snapshot_path", "")))
        try:
            root_stat=stored.lstat()
            if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode): return False
            # The cheap change signature must cover HF blob links too: the
            # link text and the confined target's own markers, so a retargeted
            # link or changed blob bytes forces a full digest recheck.
            signature_rows=[]; blobs_resolved=None
            for path in sorted(stored.rglob("*")):
                info=path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    if blobs_resolved is None: blobs_resolved=_snapshot_blobs_resolved(stored)
                    target=_confined_blob_target(path,blobs_resolved); target_info=os.lstat(target)
                    signature_rows.append(f"L\0{path.relative_to(stored)}\0{os.readlink(path)}\0{info.st_dev}\0{info.st_ino}\0{target}\0{target_info.st_dev}\0{target_info.st_ino}\0{target_info.st_size}\0{target_info.st_mtime_ns}\0{target_info.st_ctime_ns}")
                elif stat.S_ISREG(info.st_mode):
                    signature_rows.append(f"{path.relative_to(stored)}\0{info.st_dev}\0{info.st_ino}\0{info.st_size}\0{info.st_mtime_ns}\0{info.st_ctime_ns}")
            signature=tuple(signature_rows)
        except (OSError, RuntimeError): return False
        key=(str(stored),signature)
        cached=self._validation_cache.get(key)
        if cached is not None:
            return cached
        try:
            valid=_safe_snapshot_digest(stored) == receipt.get("snapshot_metadata_digest")
        except (OSError, RuntimeError):
            valid=False
        # The key includes receipt path plus every observed artifact fact; a
        # changed receipt snapshot naturally cannot reuse this decision.
        self._validation_cache[key]=valid
        return valid

    def save(self, stable_id: str, receipt: dict[str, Any]) -> None:
        target = self.path(stable_id)
        temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        with open(temp, "w", encoding="utf-8") as out:
            json.dump(receipt, out, sort_keys=True, separators=(",", ":")); out.flush(); os.fsync(out.fileno())
        os.chmod(temp, 0o600); os.replace(temp, target); os.chmod(target, 0o600); _fsync_dir(self.root)


def _public_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    # These fields locate local transport/cache artifacts.  They are verified
    # separately by the receipt and must not make an otherwise identical
    # immutable candidate manifest fail a round trip.
    identity_keys = {"stable_id", "backend", "repo", "revision", "package_versions",
                     "decode_settings", "language", "glossary_hash", "glossary_identity",
                     "evaluator_id", "evaluator_hash","runtime_identity", "runtime_source_schema", "runtime_source_digest"}
    return {key: manifest[key] for key in identity_keys if key in manifest}


def _receipt_manifest_identity_valid(receipt: dict[str, Any], manifest: dict[str, Any], *,
                                     runtime_identity: dict[str, Any]) -> bool:
    """Shared schema-3 receipt identity predicate (no filesystem mutation)."""
    if not receipt or receipt.get("manifest") != _public_manifest(manifest): return False
    if receipt.get("schema") != 3 or receipt.get("runtime_source_schema") != RUNTIME_SOURCE_SCHEMA:
        return False
    try:
        if receipt.get("runtime_source_digest") != runtime_source_digest(): return False
    except RuntimeError:
        return False
    if receipt.get("evaluator_id") != PAIRED_EVALUATOR_ID or receipt.get("evaluator_hash") != evaluator_hash(): return False
    if receipt.get("package_versions") != _package_versions(): return False
    runtime=receipt.get("runtime_identity")
    if not isinstance(runtime,dict) or manifest.get("runtime_identity") != runtime or runtime != runtime_identity:
        return False
    stored=Path(str(receipt.get("snapshot_path", "")))
    if not (stored.is_dir() and stored.name == manifest.get("revision") and receipt.get("snapshot_revision") == manifest.get("revision")):
        return False
    try:
        revision,expected=resolve_snapshot(str(manifest.get("repo","")),revision=str(manifest["revision"]))
        return revision == manifest["revision"] and stored == expected
    except (OSError,RuntimeError,ValueError,FileNotFoundError,KeyError):
        return False


def _read_receipt_read_only(path: Path) -> dict[str, Any] | None:
    """Read a single non-symlink receipt without creating/chmodding anything."""
    flags=os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)
    try:
        before=os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode): return None
        fd=os.open(path,flags)
    except OSError:
        return None
    try:
        opened=os.fstat(fd)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_dev != before.st_dev or opened.st_ino != before.st_ino or opened.st_size > 1024*1024): return None
        raw=b""
        while True:
            block=os.read(fd,65536)
            if not block: break
            raw += block
            if len(raw)>1024*1024: return None
        final=os.fstat(fd)
        if (final.st_dev,final.st_ino,final.st_size,final.st_mtime_ns,final.st_ctime_ns) != (opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns,opened.st_ctime_ns): return None
        value=json.loads(raw.decode("utf-8"))
        return value if isinstance(value,dict) else None
    except (OSError,UnicodeDecodeError,ValueError,json.JSONDecodeError):
        return None
    finally:
        os.close(fd)


def validate_preflight_receipt_read_only(base_dir: Path | str, stable_id: str) -> dict[str, Any] | None:
    """Execution-grade candidate receipt validation for the no-write status path."""
    if stable_id not in {WHISPER_ID,WHISPER_GLOSSARY_ID,PARAKEET_ID}: return None
    receipt=_read_receipt_read_only(Path(base_dir)/"adaptive-learning"/"preflight"/f"{stable_id}.json")
    if not isinstance(receipt,dict) or receipt.get("success") is not True or not isinstance(receipt.get("manifest"),dict): return None
    manifest=dict(receipt["manifest"])
    manifest["snapshot_path"]=receipt.get("snapshot_path")
    manifest["snapshot_digest"]=receipt.get("snapshot_metadata_digest")
    try:
        runtime=_candidate_runtime_identity(stable_id)
        if not _receipt_manifest_identity_valid(receipt,manifest,runtime_identity=runtime): return None
        return receipt if _safe_snapshot_digest(Path(str(receipt["snapshot_path"]))) == receipt.get("snapshot_metadata_digest") else None
    except (OSError,RuntimeError,ValueError,KeyError):
        return None


class LocalBackendManager:
    """One lazy model object per backend, with test injection seams."""
    def __init__(self, base_dir: Path | str, *, whisper: Any | None = None,
                 parakeet_loader: Callable[[str], Any] | None = None,
                 snapshot_resolver: Callable[[dict[str, Any]], Path] | None = None) -> None:
        self.base_dir = Path(base_dir)
        self._whisper = whisper
        self._parakeet = None
        self._parakeet_path: str | None = None
        self._parakeet_loader = parakeet_loader
        self._snapshot_resolver = snapshot_resolver
        self.temp_dir = self.base_dir / "adaptive-learning" / "tmp"
        ensure_private_directory(self.base_dir)
        ensure_private_directory(self.temp_dir.parent)

    def _whisper_module(self) -> Any:
        if self._whisper is None:
            import mlx_whisper
            self._whisper = mlx_whisper
        return self._whisper

    def _parakeet_model(self, repo: str, snapshot_digest: str | None = None) -> Any:
        cache_key=f"{repo}\0{snapshot_digest or ''}"
        if self._parakeet is None or self._parakeet_path != cache_key:
            if self._parakeet_loader is not None:
                self._parakeet = self._parakeet_loader(repo)
            else:
                import parakeet_mlx  # deliberately lazy: optional challenger
                self._parakeet = parakeet_mlx.from_pretrained(repo)
            self._parakeet_path = cache_key
        return self._parakeet

    @staticmethod
    def _ensure_ffmpeg(expected: dict[str, Any] | None = None) -> None:
        """Parakeet shells out by name; launchd's default PATH lacks brew."""
        current=_ffmpeg_runtime_identity()
        bound=expected.get("ffmpeg") if isinstance(expected,dict) and "ffmpeg" in expected else expected
        if bound is not None and bound != current:
            raise RuntimeError("ffmpeg runtime identity changed")
        parent=str(Path(current["resolved_path"]).parent)
        current=os.environ.get("PATH", "")
        if parent not in current.split(":"):
            os.environ["PATH"]=parent + ":" + current

    def _exact_snapshot(self, manifest: dict[str, Any]) -> Path:
        if self._snapshot_resolver is not None:
            return self._snapshot_resolver(manifest)
        revision = _safe_revision(manifest.get("revision"))
        expected_repo=WHISPER_REPO if manifest.get("backend") == "whisper" else PARAKEET_REPO if manifest.get("backend") == "parakeet" else None
        if expected_repo is None or manifest.get("repo") != expected_repo:
            raise RuntimeError("adaptive manifest repository is unsafe")
        _resolved,expected=resolve_snapshot(expected_repo,revision=revision)
        recorded = manifest.get("snapshot_path")
        if isinstance(recorded, str):
            path = Path(recorded)
            if not path.is_absolute() or path != expected:
                raise RuntimeError("adaptive snapshot path is not receipt-bound")
        return expected

    @staticmethod
    def _identity_matches(actual: AudioIdentity, expected: AudioIdentity | dict[str, Any] | None) -> bool:
        if isinstance(expected, AudioIdentity):
            return actual == expected
        return isinstance(expected, dict) and actual.as_dict() == {key: expected.get(key) for key in actual.as_dict()}

    def transcribe(self, manifest: dict[str, Any], samples: np.ndarray | PreparedAudio, *,
                   canonical_path: Path | str | None = None,
                   canonical_identity: AudioIdentity | dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        if manifest.get("language") != "en":
            raise ValueError("adaptive backends require forced English")
        if canonical_path is not None:
            if canonical_identity is None:
                raise ValueError("canonical path requires an expected identity")
            path = Path(canonical_path)
            canonical_samples, actual_identity = read_canonical_wav(path)
            if not self._identity_matches(actual_identity, canonical_identity):
                raise ValueError("canonical path identity mismatch")
            exact_path, exact_pcm = path, None
        elif isinstance(samples, PreparedAudio):
            canonical_samples, actual_identity = samples.asr_samples, samples.identity
            exact_path, exact_pcm = None, samples.pcm
        else:
            prepared = prepare_canonical(samples)
            canonical_samples, actual_identity = prepared.asr_samples, prepared.identity
            exact_path, exact_pcm = None, prepared.pcm
        snapshot = self._exact_snapshot(manifest)
        if manifest.get("backend") == "whisper":
            kwargs = dict(manifest.get("decode_settings") or DECODE_SETTINGS)
            kwargs["path_or_hf_repo"] = str(snapshot)
            terms = tuple(manifest.get("glossary_terms") or ())
            if terms and len(canonical_samples) / 16_000 <= 30:
                kwargs["initial_prompt"] = "Glossary terms: " + ", ".join(terms)
            result = self._whisper_module().transcribe(canonical_samples, **kwargs)
            text = result.get("text") if isinstance(result, dict) else None
        elif manifest.get("backend") == "parakeet":
            self._ensure_ffmpeg(manifest.get("runtime_identity"))
            if exact_path is not None:
                result = self._parakeet_model(str(snapshot),manifest.get("snapshot_digest")).transcribe(str(exact_path))
                text = getattr(result, "text", None)
            else:
                ensure_private_directory(self.temp_dir)
                fd, raw_path = tempfile.mkstemp(prefix="asr-", suffix=".wav", dir=self.temp_dir)
                path = Path(raw_path)
                try:
                    os.fchmod(fd, 0o600); os.close(fd)
                    # PreparedAudio supplies the original canonical bytes;
                    # never round-trip its /32768 float decode through an
                    # asymmetric PCM encoder.
                    write_canonical_wav(path, exact_pcm or b"")
                    result = self._parakeet_model(str(snapshot),manifest.get("snapshot_digest")).transcribe(str(path))
                    text = getattr(result, "text", None)
                finally:
                    try: os.close(fd)
                    except OSError: pass
                    path.unlink(missing_ok=True)
        else:
            raise ValueError("unsupported adaptive backend")
        if not isinstance(text, str):
            raise RuntimeError("local backend returned no text")
        return text.strip(), {"stable_id": manifest["stable_id"], "candidate_id": manifest["stable_id"], "backend": manifest["backend"],
                              "repo": manifest["repo"], "revision": manifest["revision"],
                              "language": "en"}

    def preflight(self, stable_id: str, warm_samples: np.ndarray, *, hf_home: Path | str | None = None,
                  glossary_terms: tuple[str, ...] = ()) -> dict[str, Any]:
        repo = WHISPER_REPO if stable_id.startswith("whisper") else PARAKEET_REPO
        try:
            revision, snapshot = resolve_snapshot(repo, hf_home)
        except FileNotFoundError:
            if offline_requested():
                raise RuntimeError(f"offline preflight cannot download missing snapshot for {repo}")
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise RuntimeError("huggingface_hub is required for online preflight download") from exc
            downloaded = Path(snapshot_download(repo_id=repo, cache_dir=str(Path(hf_home or os.environ.get("SOTTO_HF_HOME") or os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")))
            revision = _safe_revision(downloaded.name)
            _resolved,snapshot=resolve_snapshot(repo,hf_home,revision)
        runtime_identity=_candidate_runtime_identity(stable_id)
        manifest = candidate_manifest(stable_id, glossary_terms=glossary_terms, revision=revision,
                                      runtime_identity=runtime_identity)
        manifest["snapshot_path"] = str(snapshot.resolve())
        # The actual inference is the deliberate warmup.  Its result is never
        # retained or printed by preflight.
        self.transcribe(manifest, warm_samples)
        return {"schema": 3, "success": True, "stable_id": stable_id, "backend": manifest["backend"],
                "manifest": _public_manifest(manifest), "snapshot_revision": revision, "snapshot_path": str(snapshot.resolve()),
                "snapshot_metadata_digest": _safe_snapshot_digest(snapshot), "package_versions": _package_versions(),
                "runtime_identity":runtime_identity,
                "runtime_source_schema": RUNTIME_SOURCE_SCHEMA, "runtime_source_digest": runtime_source_digest(),
                "python": platform.python_version(), "evaluator_id": PAIRED_EVALUATOR_ID,
                "evaluator_hash": evaluator_hash(), "warmup": "history_or_synthetic", "ts": time.time()}

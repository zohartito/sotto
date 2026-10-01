"""Durable local transcription history.

``HistoryStore`` keeps the old read and mutation APIs for the currently shipped
runtime.  New runtime code should use :class:`learning.LearningCoordinator` for
all mutations: it serializes history and separately-consented learning data.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
import stat
from pathlib import Path
from typing import Any

import numpy as np

from audio_codec import (
    AudioIdentity,
    PreparedAudio,
    decode_canonical,
    encode_canonical,
    encode_pcm16,
    read_canonical_wav,
    write_canonical_wav,
    write_pcm16_wav,
)
from storage_lock import advisory_lock, ensure_private_directory, ensure_private_file

from sotto_paths import DATA_DIR

STORE_DIR = DATA_DIR
AUDIO_DIR = STORE_DIR / "audio"
INDEX = STORE_DIR / "history.jsonl"
KEEP = 200
SCHEMA_VERSION = 2


def _fsync_dir(path: Path) -> None:
    """Durably record a rename/unlink where the platform supports it."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_private_directory(path.parent)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, 0o600)
        os.replace(temp, path)
        os.chmod(path, 0o600)
        _fsync_dir(path.parent)
    except Exception:
        temp.unlink(missing_ok=True)
        _fsync_dir(path.parent)
        raise


def _tighten_owned_audio(path: Path, directory: Path) -> bool:
    """Tighten only a regular file directly owned under one audio root."""
    try:
        # Compare lexical parents before lstat so malformed metadata (or a
        # symlink) cannot make startup traverse outside the store.
        if path.parent != directory:
            return False
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return False
        os.chmod(path, 0o600)
        return True
    except OSError:
        return False


def _identity_dict(identity: AudioIdentity, relative_path: str) -> dict[str, Any]:
    return {"path": relative_path, **identity.as_dict()}


class HistoryStore:
    """Versioned JSONL history plus exact canonical inference WAVs.

    ``base_dir`` is intentionally injectable for tests and embedded callers.
    """
    def __init__(self, base_dir: Path | str | None = None, keep: int = KEEP) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else STORE_DIR
        self.audio_dir = self.base_dir / "audio"
        self.raw_audio_dir = self.base_dir / "audio-raw"
        # Silver processing may outlive ordinary dictation retention.  Its
        # spool contains only canonical 16 kHz evidence keyed by content hash;
        # it never carries a History row, transcript, prompt, raw recording,
        # session metadata, or attempt record.
        self.silver_evidence_dir = self.base_dir / "adaptive-learning" / "silver" / "evidence"
        self.index = self.base_dir / "history.jsonl"
        self.keep = keep
        self._lock = threading.RLock()
        with advisory_lock(self.base_dir):
            ensure_private_directory(self.base_dir)
            ensure_private_directory(self.audio_dir)
            ensure_private_directory(self.raw_audio_dir)
            ensure_private_directory(self.silver_evidence_dir)
            ensure_private_file(self.index)
            self._entries: list[dict[str, Any]] = self._load()
            self._tighten_retained_audio_locked()
            self._sweep_orphans()

    @staticmethod
    def _owned_audio_root(directory: Path) -> bool:
        """Whether ``directory`` is an application-owned, non-symlink root.

        Audio metadata is untrusted durable input.  In particular, never use
        ``resolve()`` here: resolving a metadata-controlled final component
        would turn a retained-path repair or cleanup into a write outside the
        Sotto store.
        """
        try:
            info = os.lstat(directory)
            if sys.platform == "win32":
                # No uid on Windows: the store lives in the user's private
                # %APPDATA%, so dir + non-symlink is the whole ownership guard.
                return stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
            return stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_uid == os.getuid()
        except OSError:
            return False

    def _owned_audio_name(self, name: Any, directory: Path, *, require_existing: bool = False) -> Path | None:
        """Return one direct ``.wav`` child, without resolving or following it."""
        if (not isinstance(name, str) or not name or "/" in name or "\\" in name or
                name in {".", ".."} or not name.endswith(".wav") or not self._owned_audio_root(directory)):
            return None
        path = directory / name
        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError:
            return None if require_existing else path
        except OSError:
            return None
        # A final symlink is never a retained artifact.  Do not unlink it via
        # normal metadata cleanup; orphan sweeping handles only its own link.
        return path if stat.S_ISREG(mode) and not stat.S_ISLNK(mode) else None

    def _tighten_retained_audio_locked(self) -> None:
        for entry in self._entries:
            for kind, directory in (("inference", self.audio_dir), ("raw", self.raw_audio_dir)):
                path = self._entry_audio_path(entry, kind)
                if path is not None:
                    _tighten_owned_audio(path, directory)
            if entry.get("silver_spooled") is True:
                path = self._entry_audio_path(entry, "inference")
                if path is not None:
                    _tighten_owned_audio(path, self.silver_evidence_dir)

    def _owned_audio_path(self, info: dict[str, Any] | None, directory: Path) -> Path | None:
        """Validate a persisted path as one direct child of its designated root."""
        if not isinstance(info, dict) or not isinstance(info.get("path"), str):
            return None
        try:
            root_parts = directory.relative_to(self.base_dir).parts
        except ValueError:
            return None
        value = info["path"]
        # ``Path`` treats a backslash as an ordinary POSIX character, so
        # reject it explicitly as well as absolute/traversal spellings.
        if value.startswith("/") or "\\" in value:
            return None
        parts = tuple(value.split("/"))
        if (parts[:len(root_parts)] != root_parts or len(parts) != len(root_parts) + 1 or
                any(part in {"", ".", ".."} for part in parts)):
            return None
        return self._owned_audio_name(parts[-1], directory)

    def _load(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self.index.exists():
            return rows
        seen: set[str] = set()
        for line_number,line in enumerate(self.index.read_text(encoding="utf-8").splitlines(),start=1):
            if not line.strip():
                continue
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError("history metadata is malformed; refusing to sweep artifacts") from exc
            if not isinstance(source, dict) or not isinstance(source.get("id"), str) or not source["id"]:
                raise RuntimeError("history metadata is malformed; refusing to sweep artifacts")
            if source["id"] in seen:
                raise RuntimeError("history metadata has duplicate ids; refusing to sweep artifacts")
            seen.add(source["id"])
            try:
                rows.append(self._normalize(source))
            except (TypeError, ValueError, KeyError) as exc:
                raise RuntimeError("history metadata is malformed; refusing to sweep artifacts") from exc
        return rows

    def _reload_locked(self) -> None:
        """Refresh metadata after acquiring the cross-process transaction lock."""
        self._entries = self._load()

    @staticmethod
    def _normalize(source: dict[str, Any]) -> dict[str, Any]:
        """Synthesize v2 in memory; callers persist it only after mutation."""
        if source.get("schema_version") == SCHEMA_VERSION:
            result = dict(source)
            result.setdefault("hypothesis", result.get("text", ""))
            result.setdefault("revision", 0)
            result.setdefault("correction", None)
            result.setdefault("attempts", [])
            result.setdefault("attempts_omitted", 0)
            result.setdefault("audio", {})
            return result
        # Only the pre-versioned row layout is a known legacy schema.  A row
        # claiming a newer version could carry semantics we do not understand;
        # treating it as old data would silently discard those protections.
        version = source.get("schema_version")
        if version not in {None, 1}:
            raise RuntimeError("unknown future history schema; refusing to operate")
        text, model, duration = source.get("text", ""), source.get("model"), source.get("duration")
        return {
            "schema_version": SCHEMA_VERSION, "id": source["id"], "ts": source.get("ts"),
            "hypothesis": text, "text": text, "duration": duration, "model": model,
            "revision": 0, "correction": None, "attempts_omitted": 0,
            "attempts": [{"kind": "legacy", "text": text, "model": model, "duration": duration,
                          "ts": source.get("ts"), "language": None, "profile": None,
                          "prompt": None, "vad": None, "latency": None, "preprocessing": None,
                          "provenance": "legacy_unknown"}],
            "audio": {"inference": {"path": f"audio/{source['id']}.wav", "sha256": None,
                                       "sample_count": None, "sample_rate": None, "format": None}, "raw": None},
        }

    def append(self, text: str, samples: np.ndarray, duration: float, model: str,
               ts: float | None = None, **metadata: Any) -> dict[str, Any]:
        """Compatibility wrapper. New callers should use LearningCoordinator.append_live."""
        with advisory_lock(self.base_dir), self._lock:
            self._reload_locked()
            return self._append_locked(text, samples, duration, model, ts=ts, **metadata)

    def _append_locked(self, text: str, samples: np.ndarray | PreparedAudio, duration: float, model: str,
                       ts: float | None = None, entry_id: str | None = None, **metadata: Any) -> dict[str, Any]:
        entry_id = entry_id or uuid.uuid4().hex[:12]
        inference_target = self._owned_audio_name(f"{entry_id}.wav", self.audio_dir)
        raw_target = self._owned_audio_name(f"{entry_id}.wav", self.raw_audio_dir)
        if inference_target is None or raw_target is None:
            raise ValueError("invalid history audio id")
        prepared = metadata.pop("prepared_audio", None)
        if prepared is None and isinstance(samples, PreparedAudio):
            prepared = samples
        if prepared is not None:
            if not isinstance(prepared, PreparedAudio):
                raise TypeError("prepared_audio must be PreparedAudio")
            pcm, identity = prepared.pcm, prepared.identity
            if decode_canonical(pcm)[1] != identity:
                raise ValueError("prepared canonical PCM identity mismatch")
        else:
            pcm, identity = encode_canonical(samples)
        relative = f"audio/{entry_id}.wav"
        write_canonical_wav(inference_target, pcm)
        raw_audio = None
        raw_samples, raw_rate = metadata.get("raw_samples"), metadata.get("raw_sample_rate")
        if raw_samples is not None and raw_rate is not None:
            raw_pcm, raw_identity = encode_pcm16(raw_samples, int(raw_rate))
            raw_relative = f"audio-raw/{entry_id}.wav"
            write_pcm16_wav(raw_target, raw_pcm, int(raw_rate))
            raw_audio = _identity_dict(raw_identity, raw_relative)
        now = time.time() if ts is None else ts
        candidate = metadata.get("candidate") if isinstance(metadata.get("candidate"), dict) else None
        attempt = {"kind": "initial", "text": text, "model": model, "duration": round(duration, 2),
                   "ts": now, "language": metadata.get("language"), "profile": metadata.get("profile"),
                   "prompt": metadata.get("prompt"), "vad": metadata.get("vad"),
                   "latency": metadata.get("latency"), "preprocessing": metadata.get("preprocessing"),
                   "provenance": metadata.get("provenance", "live"), "candidate": candidate}
        entry: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "id": entry_id, "ts": now,
            "hypothesis": text, "text": text, "duration": round(duration, 2), "model": model,
            "revision": 0, "correction": None, "attempts": [attempt], "attempts_omitted": 0,
            "audio": {"inference": _identity_dict(identity, relative), "raw": raw_audio},
            "language": metadata.get("language"), "profile": metadata.get("profile"),
            "prompt": metadata.get("prompt"), "vad": metadata.get("vad"), "latency": metadata.get("latency"),
            "preprocessing": metadata.get("preprocessing"), "adaptive": bool(metadata.get("adaptive", False)),
            "candidate": candidate}
        # The marker is content-free and is written in this same durable JSONL
        # append as the canonical WAV identity.  A silver worker can enqueue
        # it idempotently after any crash, while retention keeps the one audio
        # copy until the marker is explicitly released.
        if (entry["adaptive"] and entry["language"] == "en" and
                entry["audio"]["inference"].get("format") == "pcm_s16le_mono_16000"):
            entry["silver_enqueue"] = {"history_revision": 0, "state": "pending"}
        self._entries.append(entry)
        self._save_locked()
        self._prune_locked()
        return dict(entry)

    def entries(self, limit: int = 15) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in reversed(self._entries[-limit:])]

    def get(self, entry_id: str) -> dict[str, Any] | None:
        with advisory_lock(self.base_dir), self._lock:
            self._reload_locked()
            entry = self._find_locked(entry_id)
            return dict(entry) if entry else None

    def silver_markers(self) -> list[dict[str, Any]]:
        """Return opaque, revision-keyed silver outbox markers for recovery."""
        with advisory_lock(self.base_dir), self._lock:
            self._reload_locked()
            changed = False
            for entry in self._entries:
                audio = entry.get("audio", {}).get("inference", {})
                eligible = (entry.get("schema_version") == SCHEMA_VERSION and entry.get("adaptive") is True and
                            entry.get("language") == "en" and entry.get("correction") is None and
                            audio.get("format") == "pcm_s16le_mono_16000" and isinstance(audio.get("sha256"), str))
                marker = entry.get("silver_enqueue")
                # Schema-v2 adaptive captures written before the silver lane
                # are a known migration.  Synthesize the marker in the same
                # durable history rewrite *before* a worker can enqueue it.
                if eligible and not isinstance(marker, dict):
                    entry["silver_enqueue"] = {"history_revision": entry.get("revision", 0), "state": "pending"}
                    changed = True
            if changed:
                self._save_locked()
            return [dict(entry) for entry in self._entries if isinstance(entry.get("silver_enqueue"), dict)
                    and entry["silver_enqueue"].get("history_revision") == entry.get("revision")]

    def set_silver_marker(self, entry_id: str, revision: int, state: str, *, job_key: str | None = None) -> bool:
        """CAS-update a content-free marker; never changes transcript fields."""
        if state not in {"pending", "queued", "held", "released"}:
            raise ValueError("invalid silver marker state")
        with advisory_lock(self.base_dir), self._lock:
            self._reload_locked()
            entry = self._find_locked(entry_id)
            marker = entry.get("silver_enqueue") if entry else None
            if not isinstance(marker, dict) or entry.get("revision") != revision or marker.get("history_revision") != revision:
                return False
            marker["state"] = state
            if job_key is not None:
                marker["job_key"] = job_key
            # A spooled row has already been removed from ordinary history.
            # Once no Silver consumer owns it, erase the final personal audio
            # and the otherwise content-free bridge row immediately.
            if state == "released" and entry.get("silver_spooled") is True:
                self._unlink_entry_audio_locked(entry)
                self._entries = [row for row in self._entries if row is not entry]
            self._save_locked()
            return True

    def _find_locked(self, entry_id: str) -> dict[str, Any] | None:
        return next((e for e in self._entries if e["id"] == entry_id), None)

    def update_text(self, entry_id: str, text: str) -> None:
        """Compatibility wrapper routed through the silver mutation fence."""
        from learning import LearningCoordinator
        from adaptive_learning import AdaptiveLearning
        entry=self.get(entry_id)
        if entry is not None and entry.get("correction") is None:
            AdaptiveLearning(self.base_dir).pre_mutate_reference(entry_id,reason="compatibility_history_update")
            LearningCoordinator(self,base_dir=self.base_dir).commit_retry(entry_id,text,int(entry["revision"]),model=entry.get("model"))

    def delete(self, entry_id: str) -> None:
        """Compatibility wrapper routed through learning/silver revocation."""
        from learning import LearningCoordinator
        from adaptive_learning import AdaptiveLearning
        AdaptiveLearning(self.base_dir).revoke(entry_id,reason="compatibility_history_delete")
        LearningCoordinator(self,base_dir=self.base_dir).delete(entry_id)

    def _delete_locked(self, entry_id: str) -> bool:
        entry = self._find_locked(entry_id)
        if entry is None:
            return False
        self._entries = [e for e in self._entries if e["id"] != entry_id]
        self._save_locked()
        self._unlink_entry_audio_locked(entry)
        return True

    def clear(self) -> None:
        """Compatibility clear shares the personal epoch/scrub fence."""
        from learning import LearningCoordinator
        from adaptive_learning import AdaptiveLearning
        AdaptiveLearning(self.base_dir).revoke_all(reason="compatibility_history_clear")
        LearningCoordinator(self,base_dir=self.base_dir).clear()

    def _entry_audio_path(self, entry: dict[str, Any] | None, kind: str, *, require_existing: bool = False) -> Path | None:
        if kind not in {"inference", "raw"}:
            return None
        if entry is None:
            return None
        audio = entry.get("audio", {})
        if not isinstance(audio, dict):
            return None
        root = self.raw_audio_dir if kind == "raw" else (self.silver_evidence_dir if entry.get("silver_spooled") is True else self.audio_dir)
        info = audio.get(kind)
        path = self._owned_audio_path(info, root)
        expected = entry.get("id")
        if kind == "inference" and entry.get("silver_spooled") is True and isinstance(info, dict):
            expected = info.get("sha256")
        if not isinstance(expected, str) or path is None or path.name != f"{expected}.wav":
            return None
        if path is None or (require_existing and self._owned_audio_name(path.name, root, require_existing=True) is None):
            return None
        return path

    def audio_path(self, entry_id: str) -> Path:
        """Return a safe inference target; retained metadata cannot escape its root."""
        with self._lock:
            entry = self._find_locked(entry_id)
            if entry is not None:
                path = self._entry_audio_path(entry, "inference")
            else:
                path = self._owned_audio_name(f"{entry_id}.wav", self.audio_dir)
            if path is None:
                raise ValueError("unsafe history inference audio path")
            return path

    def raw_audio_path(self, entry_id: str) -> Path | None:
        """Return one safe retained raw WAV, if that entry still owns one."""
        with self._lock:
            return self._entry_audio_path(self._find_locked(entry_id), "raw")

    def load_audio(self, entry_id: str) -> np.ndarray:
        with self._lock:
            path = self._entry_audio_path(self._find_locked(entry_id), "inference", require_existing=True)
        if path is None:
            raise ValueError("history inference audio is unavailable")
        samples, _ = read_canonical_wav(path)
        return samples

    def load_audio_with_identity(self, entry_id: str) -> tuple[np.ndarray, AudioIdentity]:
        with self._lock:
            path = self._entry_audio_path(self._find_locked(entry_id), "inference", require_existing=True)
        if path is None:
            raise ValueError("history inference audio is unavailable")
        return read_canonical_wav(path)

    def _commit_retry_locked(self, entry_id: str, expected_revision: int, text: str,
                             model: str | None = None, **metadata: Any) -> dict[str, Any] | None:
        entry = self._find_locked(entry_id)
        if entry is None or entry.get("revision") != expected_revision or entry.get("correction") is not None:
            return None
        candidate = metadata.get("candidate") if isinstance(metadata.get("candidate"), dict) else entry.get("candidate")
        attempt = {"kind": "retry", "text": text, "model": model, "duration": entry.get("duration"),
                   "ts": metadata.get("ts", time.time()), "language": metadata.get("language"),
                   "profile": metadata.get("profile"), "prompt": metadata.get("prompt"), "vad": metadata.get("vad"),
                   "latency": metadata.get("latency"), "preprocessing": metadata.get("preprocessing"),
                   "provenance": metadata.get("provenance", "retry"), "candidate": candidate}
        attempts = entry.setdefault("attempts", [])
        attempts.append(attempt)
        if len(attempts) > 20:
            # initial attempt is immutable; evict the oldest non-initial attempt.
            attempts.pop(1 if attempts and attempts[0].get("kind") in {"initial", "legacy"} else 0)
            entry["attempts_omitted"] = int(entry.get("attempts_omitted", 0)) + 1
        entry["text"] = text
        entry["model"] = model or entry.get("model")
        entry["candidate"] = candidate
        entry["revision"] += 1
        # A retry changes the candidate-visible history revision.  Any old
        # silver label is invalidated by the coordinator before this mutation;
        # this fresh marker is the durable request for a new teacher pass.
        audio = entry.get("audio", {}).get("inference", {})
        if (entry.get("adaptive") is True and entry.get("language") == "en" and
                audio.get("format") == "pcm_s16le_mono_16000" and isinstance(audio.get("sha256"), str)):
            entry["silver_enqueue"] = {"history_revision": entry["revision"], "state": "pending"}
        self._save_locked()
        return dict(entry)

    def _save_locked(self) -> None:
        _atomic_jsonl(self.index, self._entries)

    def _unlink_entry_audio_locked(self, entry: dict[str, Any]) -> None:
        for name, directory in (("inference", self.audio_dir), ("raw", self.raw_audio_dir)):
            path = self._entry_audio_path(entry, name)
            if path is not None and self._owned_audio_name(path.name, directory, require_existing=True) is not None:
                path.unlink(missing_ok=True)
        if entry.get("silver_spooled") is True:
            path=self._entry_audio_path(entry, "inference")
            if path is not None and self._owned_audio_name(path.name, self.silver_evidence_dir, require_existing=True) is not None:
                path.unlink(missing_ok=True)
        _fsync_dir(self.audio_dir)
        _fsync_dir(self.raw_audio_dir)
        _fsync_dir(self.silver_evidence_dir)

    def _spool_silver_entry_locked(self, entry: dict[str, Any]) -> bool:
        """Move one overflowed canonical WAV into the minimal Silver spool."""
        if entry.get("silver_spooled") is True:
            return True
        marker=entry.get("silver_enqueue")
        audio=entry.get("audio",{}).get("inference")
        if not isinstance(marker,dict) or not isinstance(audio,dict):
            return False
        digest=audio.get("sha256")
        source=self._entry_audio_path(entry,"inference")
        if not isinstance(digest,str) or len(digest) != 64 or source is None:
            return False
        try:
            if self._owned_audio_name(source.name, self.audio_dir, require_existing=True) is None:
                return False
            _samples,identity=read_canonical_wav(source)
        except (OSError,ValueError,EOFError):
            return False
        if identity.sha256 != digest or identity.format != "pcm_s16le_mono_16000":
            return False
        destination=self._owned_audio_name(f"{digest}.wav", self.silver_evidence_dir)
        if destination is None:
            return False
        try:
            existing=self._owned_audio_name(destination.name, self.silver_evidence_dir, require_existing=True)
            if existing is not None:
                _samples,existing=read_canonical_wav(destination)
                if existing.sha256 != digest:
                    return False
                source.unlink(missing_ok=True)
            elif os.path.lexists(destination):
                # A dangling or final symlink is not an owned spool target.
                return False
            else:
                os.replace(source,destination)
                os.chmod(destination,0o600)
                _fsync_dir(self.silver_evidence_dir)
        except (OSError,ValueError,EOFError):
            return False
        raw=self._entry_audio_path(entry,"raw")
        if raw is not None and self._owned_audio_name(raw.name, self.raw_audio_dir, require_existing=True) is not None:
            raw.unlink(missing_ok=True)
        relative=str(destination.relative_to(self.base_dir))
        minimal={"schema_version":SCHEMA_VERSION,"id":entry["id"],"ts":entry.get("ts"),
                 "revision":entry.get("revision"),"correction":None,
                 "audio":{"inference":_identity_dict(identity,relative),"raw":None},
                 "adaptive":True,"language":"en","silver_enqueue":dict(marker),"silver_spooled":True}
        entry.clear(); entry.update(minimal)
        _fsync_dir(self.raw_audio_dir)
        return True

    def _prune_locked(self) -> list[dict[str, Any]]:
        if len(self._entries) <= self.keep:
            return []
        prefix, retained = self._entries[:-self.keep], self._entries[-self.keep:]
        # A content-free adaptive review hand-off still needs this row's
        # canonical audio/reference to converge after a crash.  Keep it even
        # outside normal retention until the coordinator marks it complete.
        pinned = [entry for entry in prefix
                  if isinstance(entry.get("adaptive_review"), dict)
                  and not entry["adaptive_review"].get("completed", False)]
        pinned.extend(entry for entry in prefix if isinstance(entry.get("silver_enqueue"), dict)
                      and entry["silver_enqueue"].get("state") != "released" and entry not in pinned)
        # Do not let the Silver outbox pin a full personal History row beyond
        # ordinary retention.  Move the exact canonical evidence into its
        # private spool and retain only opaque identity facts needed to finish
        # or revoke the already-authorized job.
        for entry in pinned:
            if isinstance(entry.get("silver_enqueue"),dict):
                self._spool_silver_entry_locked(entry)
        overflow = [entry for entry in prefix if entry not in pinned]
        self._entries = pinned + retained
        self._save_locked()
        for entry in overflow:
            self._unlink_entry_audio_locked(entry)
        return overflow

    def _sweep_orphans(self) -> None:
        with self._lock:
            known = set()
            for entry in self._entries:
                for kind, directory in (("inference", self.audio_dir), ("raw", self.raw_audio_dir)):
                    path = self._entry_audio_path(entry, kind)
                    if path is not None:
                        known.add(path)
                if entry.get("silver_spooled") is True:
                    path=self._entry_audio_path(entry,"inference")
                    if path is not None: known.add(path)
            for directory in (self.audio_dir, self.raw_audio_dir, self.silver_evidence_dir):
                if not self._owned_audio_root(directory):
                    # Refuse to treat an untrusted root as empty: otherwise a
                    # startup repair could sweep someone else's directory.
                    raise RuntimeError("history audio root is unsafe; refusing artifact sweep")
                for path in directory.iterdir():
                    try:
                        mode = os.lstat(path).st_mode
                    except OSError:
                        continue
                    if stat.S_ISLNK(mode):
                        # A direct orphan link is safe to unlink (unlink does
                        # not follow it); never recurse into its target.
                        if path not in known:
                            path.unlink(missing_ok=True)
                        continue
                    if not stat.S_ISREG(mode):
                        continue
                    if path.suffix == ".tmp" or path not in known:
                        path.unlink(missing_ok=True)
            for temp in self.base_dir.glob(".history.jsonl.*.tmp"):
                temp.unlink(missing_ok=True)
            _fsync_dir(self.audio_dir)
            _fsync_dir(self.raw_audio_dir)
            _fsync_dir(self.silver_evidence_dir)

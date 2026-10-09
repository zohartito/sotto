"""Explicit-consent learning corpus and the mutation coordinator.

No transcript reaches this store merely because a model produced it.  A caller
must first create a human correction and then explicitly enroll that correction.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Iterator

from audio_codec import AudioIdentity, PreparedAudio, read_canonical_wav, read_pcm16_wav
from comparator_spool import ComparatorSpool
from history import HistoryStore, STORE_DIR, _atomic_jsonl, _fsync_dir
from storage_lock import advisory_lock, ensure_private_directory, ensure_private_file

LEARNING_SCHEMA_VERSION = 1
_PROCESS_LOCK = threading.RLock()


def scrub_comparator_spools(base_dir: Path | str) -> bool:
    """Remove only direct private comparator WAVs, without following links.

    This is part of Clear's exact outbox completion, not best-effort ordinary
    retention.  A malformed artifact is an authority fence: leave the clear
    marker present so recovery can be inspected rather than acknowledging a
    potentially external path.
    """
    try:
        return ComparatorSpool(base_dir).scrub(None)
    except (FileNotFoundError,OSError,RuntimeError,ValueError):
        return False


def silver_lane_exists(base_dir: Path | str) -> bool:
    """Whether a silver ledger may hold evidence that History must revoke.

    ``HistoryStore`` always creates an empty ``adaptive-learning/silver/evidence``
    and nothing else there.  Everything else under ``silver`` belongs to the
    lane: the SQLite ledger (with its WAL/SHM) is created by ``SilverStore``
    before any job, label, tombstone, cohort or comparator row can exist;
    deployment/experiment state is written only by controllers that construct
    a ``SilverStore`` first; and spooled evidence audio and comparator WAVs
    are written only for adaptive captures, whose runtime builds the ledger
    at startup.  So a silver directory that is missing, or holds nothing but
    an empty ``evidence`` directory, has never had a ledger.  Anything else --
    an unknown or non-empty entry, a non-directory or symlinked path, an
    unreadable parent -- counts as present so revocation stays fail-closed.
    """
    silver = Path(base_dir) / "adaptive-learning" / "silver"
    evidence = silver / "evidence"
    try:
        if not stat.S_ISDIR(os.lstat(silver).st_mode):
            return True
        names = os.listdir(silver)
        if any(name != "evidence" for name in names):
            return True
        if not names:
            return False
        return not stat.S_ISDIR(os.lstat(evidence).st_mode) or bool(os.listdir(evidence))
    except FileNotFoundError:
        return False
    except OSError:
        return True


def adaptive_review_eligible_entry(entry: dict[str, Any] | None) -> bool:
    """Whether a retained history row may enter the adaptive review protocol."""
    if not isinstance(entry, dict):
        return False
    audio = entry.get("audio", {}).get("inference", {})
    return (entry.get("schema_version") == 2 and entry.get("adaptive") is True and entry.get("language") == "en" and
            audio.get("format") == "pcm_s16le_mono_16000" and isinstance(audio.get("path"), str) and
            isinstance(audio.get("sha256"), str))


@dataclass(frozen=True)
class RetrySnapshot:
    entry_id: str
    expected_revision: int
    inference_audio_path: Path
    inference_identity: AudioIdentity
    samples: Any | None = None


@dataclass(frozen=True)
class LearningSnapshot:
    sample_id: str
    history_id: str
    history_revision: int
    corrected_text: str
    inference_audio_path: Path
    inference_identity: AudioIdentity
    raw_audio_path: Path | None = None
    raw_identity: AudioIdentity | None = None


def _copy_fsync(source: Path, destination: Path) -> None:
    ensure_private_directory(destination.parent)
    with open(source, "rb") as src, open(destination, "wb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    os.chmod(destination, 0o600)


def _owned_artifact_path(directory: Path, sample_id: Any) -> Path | None:
    """Return only a direct, store-owned WAV path for a persisted sample id."""
    if not isinstance(sample_id, str):
        return None
    path = directory / f"{sample_id}.wav"
    return path if path.parent == directory else None


def _tighten_owned_artifact(path: Path | None, directory: Path) -> bool:
    """Tighten a regular direct child without resolving or following symlinks."""
    if path is None or path.parent != directory:
        return False
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return False
        os.chmod(path, 0o600)
        return True
    except OSError:
        return False


def _is_owned_regular_artifact(path: Path | None, directory: Path) -> bool:
    """Validate a direct regular artifact without changing its metadata."""
    if path is None or path.parent != directory:
        return False
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def _unlink_owned_artifact(path: Path | None) -> None:
    """Unlink a direct store child; ``unlink`` never follows a final symlink."""
    if path is not None:
        path.unlink(missing_ok=True)


def _identity_matches(actual: AudioIdentity, recorded: dict[str, Any] | None) -> bool:
    """Strictly compare every persisted identity field when one is recorded.

    Pre-v2 history rows have ``sha256: null`` and cannot be upgraded safely, so
    they retain the existing compatibility path until a user produces new audio.
    """
    if not recorded or recorded.get("sha256") is None:
        return True
    return actual.as_dict() == {name: recorded.get(name) for name in actual.as_dict()}


def _snapshot_from_record(store: "LearningStore", record: dict[str, Any]) -> LearningSnapshot:
    return _snapshot_from_paths(store.audio_dir, store.raw_dir, record)


def _snapshot_from_paths(audio_dir: Path, raw_dir: Path, record: dict[str, Any]) -> LearningSnapshot:
    raw_record = record.get("raw_audio")
    raw_identity = AudioIdentity(**raw_record) if raw_record else None
    sample_id = record["sample_id"]
    return LearningSnapshot(
        sample_id,
        record["history_id"],
        record["history_revision"],
        record["corrected_text"],
        audio_dir / f"{sample_id}.wav",
        AudioIdentity(**record["inference_audio"]),
        raw_dir / f"{sample_id}.wav" if raw_identity is not None else None,
        raw_identity,
    )


def _read_records(index: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not index.exists():
        return records
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("learning metadata is malformed; refusing artifact recovery") from exc
        if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str) or not row["sample_id"]:
            raise RuntimeError("learning metadata is malformed; refusing artifact recovery")
        if row["sample_id"] in records:
            raise RuntimeError("learning metadata has duplicate ids; refusing artifact recovery")
        records[row["sample_id"]] = row
    return records


def _artifacts_valid_at(audio_dir: Path, raw_dir: Path, record: dict[str, Any]) -> bool:
    sample_id = record.get("sample_id")
    if record.get("status") != "active" or not isinstance(sample_id, str):
        return False
    try:
        inference_path = _owned_artifact_path(audio_dir, sample_id)
        if not _is_owned_regular_artifact(inference_path, audio_dir):
            return False
        _, identity = read_canonical_wav(inference_path)
        if identity.as_dict() != record.get("inference_audio"):
            return False
        raw_record = record.get("raw_audio")
        if raw_record is not None:
            raw_path = _owned_artifact_path(raw_dir, sample_id)
            if not _is_owned_regular_artifact(raw_path, raw_dir):
                return False
            _, raw_identity = read_pcm16_wav(raw_path)
            return raw_identity.as_dict() == raw_record
        return not record.get("raw_present")
    except (FileNotFoundError, ValueError, OSError, EOFError):
        return False


class LearningStore:
    """Atomic metadata and copied artifacts for separately enrolled samples."""
    def __init__(self, base_dir: Path | str | None = None) -> None:
        parent = Path(base_dir) if base_dir is not None else STORE_DIR
        self.store_root = parent
        self.base_dir = parent / "learning"
        self.audio_dir = self.base_dir / "audio"
        self.raw_dir = self.base_dir / "raw"
        self.staging_dir = self.base_dir / ".staging"
        self.index = self.base_dir / "learning.jsonl"
        self.pending_index = self.base_dir / "pending-gold.jsonl"
        with advisory_lock(self.store_root):
            for directory in (self.base_dir, self.audio_dir, self.raw_dir, self.staging_dir):
                ensure_private_directory(directory)
            ensure_private_file(self.index); ensure_private_file(self.pending_index)
            self._records = _read_records(self.index)
            self._pending_gold = _read_records(self.pending_index)
            self.recover_filesystem()

    @classmethod
    def admin_snapshot(cls, base_dir: Path | str | None = None) -> list[LearningSnapshot]:
        """Read valid active records without recovery, mutation, or history loading."""
        parent = Path(base_dir) if base_dir is not None else STORE_DIR
        with _PROCESS_LOCK, advisory_lock(parent):
            return cls._admin_snapshot_locked(parent)

    @classmethod
    @contextmanager
    def admin_export_guard(cls, base_dir: Path | str | None = None) -> Iterator[list[LearningSnapshot]]:
        """Keep a validated read-only snapshot stable while the caller copies files.

        Cross-process export callers must copy the yielded artifacts before
        leaving this context.  Unlike ``admin_snapshot``, the lock remains held
        for the complete copy operation.
        """
        parent = Path(base_dir) if base_dir is not None else STORE_DIR
        with _PROCESS_LOCK, advisory_lock(parent):
            yield cls._admin_snapshot_locked(parent)

    @classmethod
    def _admin_snapshot_locked(cls, parent: Path) -> list[LearningSnapshot]:
        base = parent / "learning"
        records = _read_records(base / "learning.jsonl")
        return [
            _snapshot_from_paths(base / "audio", base / "raw", record)
            for record in records.values()
            if _artifacts_valid_at(base / "audio", base / "raw", record)
        ]

    @classmethod
    def admin_revoke(cls, sample_id: str, base_dir: Path | str | None = None) -> bool:
        """Cross-process targeted revoke without startup recovery or history loading."""
        parent = Path(base_dir) if base_dir is not None else STORE_DIR
        with _PROCESS_LOCK, advisory_lock(parent):
            store = cls.__new__(cls)
            store.store_root = parent
            store.base_dir = parent / "learning"
            store.audio_dir = store.base_dir / "audio"
            store.raw_dir = store.base_dir / "raw"
            store.staging_dir = store.base_dir / ".staging"
            store.index = store.base_dir / "learning.jsonl"
            store._records = _read_records(store.index)
            return store._revoke_loaded_locked(sample_id)

    @classmethod
    def admin_history_id(cls, sample_id: str, base_dir: Path | str | None = None) -> str | None:
        """Content-free sample-to-history lookup for dependency revocation."""
        parent = Path(base_dir) if base_dir is not None else STORE_DIR
        with _PROCESS_LOCK, advisory_lock(parent):
            row = _read_records(parent / "learning" / "learning.jsonl").get(sample_id)
            history_id = row.get("history_id") if isinstance(row, dict) else None
            return history_id if isinstance(history_id, str) else None

    def _save(self) -> None:
        _atomic_jsonl(self.index, list(self._records.values()))

    def _save_pending(self) -> None:
        _atomic_jsonl(self.pending_index, list(self._pending_gold.values()))

    def mark_pending_gold(self, history_id: str, code: str) -> None:
        """Durable, content-free retry marker when automatic enrollment fails."""
        with advisory_lock(self.store_root):
            self._pending_gold[history_id] = {"sample_id": history_id, "history_id": history_id,
                                              "status": "pending_gold", "code": code, "ts": time.time()}
            self._save_pending()

    def clear_pending_gold(self, history_id: str) -> None:
        with advisory_lock(self.store_root):
            if history_id in self._pending_gold:
                self._pending_gold.pop(history_id); self._save_pending()

    def pending_gold(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._pending_gold.values()]

    def _reload_locked(self) -> None:
        self._records = _read_records(self.index)
        self._pending_gold = _read_records(self.pending_index)

    def active(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._records.values() if row.get("status") == "active"]

    def active_for_history(self, history_id: str) -> list[dict[str, Any]]:
        return [row for row in self._records.values()
                if row.get("status") == "active" and row.get("history_id") == history_id]

    def artifacts_valid(self, record: dict[str, Any]) -> bool:
        """Verify every artifact required by an active record from its bytes."""
        return _artifacts_valid_at(self.audio_dir, self.raw_dir, record)

    def stage_copy(self, sample_id: str, inference_source: Path, raw_source: Path | None = None,
                   *, inference_identity: AudioIdentity | None = None,
                   raw_identity: AudioIdentity | None = None) -> Path:
        """Create fsynced copies. This deliberately does not take coordinator lock."""
        with advisory_lock(self.store_root):
            stage = self.staging_dir / f"{sample_id}.{uuid.uuid4().hex}"
            stage.mkdir(mode=0o700)
            os.chmod(stage, 0o700)
            try:
                _copy_fsync(inference_source, stage / "inference.wav")
                _, staged_inference = read_canonical_wav(stage / "inference.wav")
                if inference_identity is not None and staged_inference != inference_identity:
                    raise ValueError("staged inference identity mismatch")
                if raw_source is not None:
                    _copy_fsync(raw_source, stage / "raw.wav")
                    _, staged_raw = read_pcm16_wav(stage / "raw.wav")
                    if raw_identity is None or staged_raw != raw_identity:
                        raise ValueError("staged raw identity mismatch")
                elif raw_identity is not None:
                    raise ValueError("raw identity provided without raw source")
                _fsync_dir(stage)
                _fsync_dir(self.staging_dir)
                return stage
            except Exception:
                shutil.rmtree(stage, ignore_errors=True)
                _fsync_dir(self.staging_dir)
                raise

    def publish_staged(self, sample_id: str, stage: Path, record: dict[str, Any]) -> dict[str, Any]:
        """Move verified artifacts first; atomically publish metadata last."""
        with advisory_lock(self.store_root):
            self._reload_locked()
            final_audio = self.audio_dir / f"{sample_id}.wav"
            final_raw = self.raw_dir / f"{sample_id}.wav"
            moved: list[Path] = []
            previous = self._records.get(sample_id)
            saved = False
            try:
                os.replace(stage / "inference.wav", final_audio); moved.append(final_audio)
                if (stage / "raw.wav").exists():
                    os.replace(stage / "raw.wav", final_raw); moved.append(final_raw)
                os.chmod(final_audio, 0o600)
                if final_raw.exists(): os.chmod(final_raw, 0o600)
                _fsync_dir(self.audio_dir); _fsync_dir(self.raw_dir)
                # Read back after rename, before metadata can make this active.
                _, actual = read_canonical_wav(final_audio)
                if actual.as_dict() != record["inference_audio"]:
                    raise ValueError("learning inference copy identity mismatch")
                raw_record = record.get("raw_audio")
                if raw_record is not None:
                    if not final_raw.exists():
                        raise ValueError("learning raw copy missing")
                    _, actual_raw = read_pcm16_wav(final_raw)
                    if actual_raw.as_dict() != raw_record:
                        raise ValueError("learning raw copy identity mismatch")
                elif final_raw.exists():
                    raise ValueError("unexpected learning raw copy")
                self._records[sample_id] = record
                self._save()  # active record is always the final publication step
                saved = True
                shutil.rmtree(stage, ignore_errors=True)
                _fsync_dir(self.staging_dir)
                return dict(record)
            except Exception:
                if not saved:
                    if previous is None:
                        self._records.pop(sample_id, None)
                    else:
                        self._records[sample_id] = previous
                for path in moved:
                    path.unlink(missing_ok=True)
                shutil.rmtree(stage, ignore_errors=True)
                _fsync_dir(self.audio_dir); _fsync_dir(self.raw_dir); _fsync_dir(self.staging_dir)
                raise

    def revoke(self, sample_id: str) -> bool:
        with advisory_lock(self.store_root):
            self._reload_locked()
            return self._revoke_loaded_locked(sample_id)

    def _revoke_loaded_locked(self, sample_id: str) -> bool:
        """Revoke a record already loaded while the advisory lock is held."""
        record = self._records.get(sample_id)
        if record is None or record.get("status") == "revoked":
            return False
        revoking = {"schema_version": LEARNING_SCHEMA_VERSION, "sample_id": sample_id,
            "history_id": record.get("history_id"), "status": "revoking", "ts": time.time()}
        self._records[sample_id] = revoking
        try:
            self._save()
        except Exception:
            self._records[sample_id] = record
            raise
        try:
            inference = _owned_artifact_path(self.audio_dir, sample_id)
            raw = _owned_artifact_path(self.raw_dir, sample_id)
            _tighten_owned_artifact(inference, self.audio_dir)
            _tighten_owned_artifact(raw, self.raw_dir)
            _unlink_owned_artifact(inference)
            _unlink_owned_artifact(raw)
            _fsync_dir(self.audio_dir); _fsync_dir(self.raw_dir)
        except Exception:
            # The fsynced revoking tombstone makes startup finish this safely.
            raise
        self._records[sample_id] = {"schema_version": LEARNING_SCHEMA_VERSION, "sample_id": sample_id,
            "history_id": record.get("history_id"), "status": "revoked", "ts": time.time()}
        try:
            self._save()
        except Exception:
            # Disk already contains the revoking tombstone; preserve it in memory.
            self._records[sample_id] = revoking
            raise
        return True

    def recover_filesystem(self) -> None:
        """Finish interrupted revocations and remove unpublished/orphan artifacts."""
        with advisory_lock(self.store_root):
            self._reload_locked()
            for stage in list(self.staging_dir.iterdir()):
                try:
                    mode = os.lstat(stage).st_mode
                except OSError:
                    continue
                if stat.S_ISDIR(mode):
                    os.chmod(stage, 0o700)
                    shutil.rmtree(stage, ignore_errors=True)
                else:
                    stage.unlink(missing_ok=True)
            for temp in self.base_dir.glob(".learning.jsonl.*.tmp"):
                temp.unlink(missing_ok=True)
            changed = False
            for sample_id, record in list(self._records.items()):
                status = record.get("status")
                inference = _owned_artifact_path(self.audio_dir, sample_id)
                raw = _owned_artifact_path(self.raw_dir, sample_id)
                if status == "revoking":
                    _tighten_owned_artifact(inference, self.audio_dir)
                    _tighten_owned_artifact(raw, self.raw_dir)
                    _unlink_owned_artifact(inference); _unlink_owned_artifact(raw)
                    self._records[sample_id] = {"schema_version": LEARNING_SCHEMA_VERSION, "sample_id": sample_id,
                        "history_id": record.get("history_id"), "status": "revoked", "ts": time.time()}
                    changed = True
                elif status == "active":
                    inference_private = _tighten_owned_artifact(inference, self.audio_dir)
                    raw_private = (record.get("raw_audio") is None or
                                   _tighten_owned_artifact(raw, self.raw_dir))
                    valid = inference_private and raw_private and self.artifacts_valid(record)
                    if valid and record.get("raw_audio") is None and raw is not None and raw.exists():
                        # An artifact without a recorded identity is not consent provenance.
                        _unlink_owned_artifact(raw)
                    if not valid:
                        _unlink_owned_artifact(inference); _unlink_owned_artifact(raw)
                        self._records[sample_id] = {"schema_version": LEARNING_SCHEMA_VERSION, "sample_id": sample_id,
                            "history_id": record.get("history_id"), "status": "revoked", "ts": time.time()}
                        changed = True
            active_ids = {sample_id for sample_id, row in self._records.items() if row.get("status") == "active"}
            for directory in (self.audio_dir, self.raw_dir):
                for artifact in directory.glob("*.wav"):
                    if artifact.stem not in active_ids:
                        artifact.unlink(missing_ok=True)
            _fsync_dir(self.audio_dir); _fsync_dir(self.raw_dir); _fsync_dir(self.staging_dir)
            if changed: self._save()


class LearningCoordinator:
    """The sole intended mutation surface for history and learning state."""
    def __init__(self, history: HistoryStore | None = None, learning: LearningStore | None = None,
                 base_dir: Path | str | None = None) -> None:
        self._lock = _PROCESS_LOCK
        self.history = history or HistoryStore(base_dir)
        self.learning = learning or LearningStore(base_dir)
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            self._validate_active_history_links_locked()

    def _reload_current_locked(self) -> None:
        self.history._reload_locked()
        self.learning._reload_locked()

    def _revoke_silver_before_history_mutation(self, entry_id: str, history_revision: int | None = None,
                                               operation: str = "correct") -> tuple[set[str], dict[str, Any] | None]:
        """Fence machine evidence before a human correction/delete changes it.

        Imported lazily so the established human-only startup path does not
        acquire any worker/model dependency.  Failure is deliberately fatal:
        an unknown silver schema must not permit stale pseudo-label evidence.

        A data folder that has never had a silver ledger (the plain alpha,
        Windows) has no evidence to fence, so nothing is created for it.
        Callers hold the storage advisory lock, which the worker also holds
        from marker validation through enqueue, and the worker constructs its
        ledger before that; any enqueue therefore precedes this check.
        """
        if not silver_lane_exists(self.history.base_dir):
            return set(), None
        from silver_store import SilverStore
        from deployment_controller import DeploymentController
        dependencies,intent=SilverStore(self.history.base_dir).revoke_history_with_intent(entry_id,history_revision=history_revision,
                                                                                             reason="human_or_history_mutation",operation=operation)
        DeploymentController(self.history.base_dir).invalidate_for_cohorts(dependencies, "human_or_history_mutation")
        return dependencies,intent

    def _validate_active_history_links_locked(self) -> None:
        for record in list(self.learning.active()):
            entry = self.history._find_locked(record["history_id"])
            # Retention has already removed this history row only after the
            # independently copied learning artifacts were durably enrolled.
            # Delete and clear explicitly revoke before removal, so absence at
            # startup is ordinary retention rather than withdrawn consent.
            if entry is None:
                continue
            correction = entry and entry.get("correction")
            valid = (correction is not None
                     and entry.get("revision") == record.get("history_revision")
                     and correction.get("text") == record.get("corrected_text"))
            if valid:
                try:
                    _, identity = self.history.load_audio_with_identity(record["history_id"])
                    valid = _identity_matches(identity, entry.get("audio", {}).get("inference"))
                    valid = valid and identity.as_dict() == record.get("inference_audio")
                    raw_record = record.get("raw_audio")
                    raw_info = entry.get("audio", {}).get("raw")
                    if raw_record is not None:
                        if not raw_info or not raw_info.get("path"):
                            valid = False
                        else:
                            raw_path = self.history.raw_audio_path(record["history_id"])
                            if raw_path is None:
                                raise ValueError("history raw audio path is unsafe")
                            _, raw_identity = read_pcm16_wav(raw_path)
                            valid = valid and _identity_matches(raw_identity, raw_info)
                            valid = valid and raw_identity.as_dict() == raw_record
                except (FileNotFoundError, ValueError, OSError, EOFError):
                    valid = False
            if not valid:
                self.learning.revoke(record["sample_id"])

    def append_live(self, text: str, samples: Any | PreparedAudio, duration: float, model: str,
                    **metadata: Any) -> dict[str, Any]:
        """Persist a prepared live attempt without a second canonical encoding.

        Passing an ndarray remains supported; callers that need the ASR and
        persisted bytes to match exactly pass ``PreparedAudio`` from
        :func:`audio_codec.prepare_canonical` (either as ``samples`` or the
        ``prepared_audio`` keyword).
        """
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            return self.history._append_locked(text, samples, duration, model, **metadata)

    def review_transaction(self, entry_id: str, expected_revision: int | None, before_mutation, mutation):
        """Validate UI revision then run dependency invalidation under storage locks.

        Lock order is storage/coordinator -> adaptive.  Callers must never
        hold an AdaptiveLearning lock before entering this seam.
        """
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            entry = self.history._find_locked(entry_id)
            if entry is None or (expected_revision is not None and entry.get("revision") != expected_revision):
                return None
            before_mutation()
            return mutation()

    def withdraw_review(self, entry_id: str, *, expected_revision: int | None = None) -> dict[str, Any] | None:
        """Compatibility wrapper for a private skip outbox operation."""
        return self.commit_adaptive_review(entry_id, "skip", skip_reason="private",
                                           expected_revision=expected_revision)

    def commit_adaptive_review(self, entry_id: str, outcome: str, *, reference: str = "",
                               skip_reason: str | None = None,
                               expected_revision: int | None = None,
                               before_mutation=None) -> dict[str, Any] | None:
        """Commit a human review and its content-free adaptive outbox record.

        The transcript/reference remains only in the normal reviewed History
        entry.  The adjacent outbox record is revision-keyed metadata, so an
        interrupted adaptive/learning half can be recovered after restart.
        """
        allowed = {"corrected", "correct_as_is", "no_speech", "skip"}
        if outcome not in allowed:
            raise ValueError("invalid review outcome")
        if outcome == "skip" and skip_reason not in {"private", "corrupt", "wrong_capture"}:
            raise ValueError("invalid skip reason")
        if outcome == "corrected" and not isinstance(reference, str):
            raise TypeError("reference must be text")
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            entry = self.history._find_locked(entry_id)
            if entry is None or (expected_revision is not None and entry.get("revision") != expected_revision):
                return None
            if not adaptive_review_eligible_entry(entry):
                raise ValueError("history row is not eligible for adaptive review")
            # Match the worker's advisory-held scan/enqueue fence.  Doing
            # this after acquiring the visible old revision prevents a worker
            # paused between scan and enqueue from resurrecting it.
            self._revoke_silver_before_history_mutation(entry_id,history_revision=int(entry["revision"]))
            # Dependency invalidation is deliberately first, under the same
            # storage/coordinator locks.  Thus a failed/partial invalidation
            # cannot withdraw learning consent or alter history.  The caller
            # must acquire adaptive only from this storage-first seam.
            if before_mutation is not None:
                before_mutation()
            # Silver revocation completed before this lock was acquired.  Mark
            # the visible old revision releasable before bumping it, otherwise
            # revision-keyed marker enumeration hides a retained audio pin.
            marker=entry.get("silver_enqueue")
            if isinstance(marker,dict) and marker.get("history_revision") == entry.get("revision"):
                marker["state"]="released"
            entry["revision"] += 1
            if outcome == "corrected":
                text = reference
            elif outcome == "correct_as_is":
                text = str(entry.get("text", ""))
            elif outcome == "no_speech":
                text = ""
            else:
                text = ""
            if outcome == "skip":
                # Withdrawing consent also prevents a later pending-enrollment
                # retry from publishing the superseded gold.
                entry["correction"] = None
            else:
                entry["text"] = text
                entry["correction"] = {"text": text, "outcome": outcome, "ts": time.time(),
                                       "base_attempt": len(entry.get("attempts", [])) - 1}
            entry["adaptive_review"] = {"revision": entry["revision"], "outcome": outcome,
                                        "skip_reason": skip_reason if outcome == "skip" else None,
                                        "ts": time.time()}
            # Make the recovery record durable before any artifact operation,
            # whose tombstone/delete path can itself fail after invalidation.
            self.history._save_locked()
            for record in list(self.learning.active_for_history(entry_id)):
                self.learning.revoke(record["sample_id"])
            self.learning.clear_pending_gold(entry_id)
            return dict(entry)

    def adaptive_review_metadata(self, entry_id: str) -> dict[str, Any] | None:
        """Return only revision/outcome metadata needed to resume an outbox op."""
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            entry = self.history._find_locked(entry_id)
            operation = entry.get("adaptive_review") if entry else None
            if not isinstance(operation, dict) or operation.get("revision") != entry.get("revision"):
                return None
            return {"revision": operation["revision"], "outcome": operation.get("outcome"),
                    "skip_reason": operation.get("skip_reason")}

    def adaptive_review_entries(self) -> list[str]:
        """List opaque history IDs with current revision-keyed review outbox records."""
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            return [entry["id"] for entry in self.history._entries
                    if isinstance(entry.get("adaptive_review"), dict)
                    and entry["adaptive_review"].get("revision") == entry.get("revision")
                    and not entry["adaptive_review"].get("completed", False)]

    def mark_adaptive_review_complete(self, entry_id: str, revision: int, outcome: str) -> bool:
        """Close an outbox record without changing its history revision."""
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            entry = self.history._find_locked(entry_id)
            operation = entry.get("adaptive_review") if entry else None
            if (not isinstance(operation, dict) or entry.get("revision") != revision or
                    operation.get("revision") != revision or operation.get("outcome") != outcome):
                return False
            if not operation.get("completed", False):
                operation["completed"] = True; operation["completed_ts"] = time.time()
                self.history._save_locked()
            return True

    def reconcile_adaptive_review_artifacts(self, entry_id: str) -> int:
        """Finish post-outbox revocation of superseded learning artifacts."""
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            entry = self.history._find_locked(entry_id)
            operation = entry.get("adaptive_review") if entry else None
            if not isinstance(operation, dict) or operation.get("revision") != entry.get("revision"):
                return 0
            count = 0
            for record in list(self.learning.active_for_history(entry_id)):
                if operation.get("outcome") == "skip" or record.get("history_revision") != entry.get("revision"):
                    if self.learning.revoke(record["sample_id"]):
                        count += 1
            if operation.get("outcome") == "skip":
                self.learning.clear_pending_gold(entry_id)
            return count

    def load_adaptive_review(self, entry_id: str) -> dict[str, Any] | None:
        """Load one current outbox operation and verify its canonical audio identity.

        This private coordinator seam is intentionally the only outbox reader
        that can expose a reference to the adaptive runtime.  The outbox itself
        contains no text.
        """
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            entry = self.history._find_locked(entry_id)
            operation = entry.get("adaptive_review") if entry else None
            if not isinstance(operation, dict) or operation.get("revision") != entry.get("revision"):
                return None
            if not adaptive_review_eligible_entry(entry):
                raise RuntimeError("adaptive review outbox row is not eligible")
            _, identity = read_canonical_wav(self.history.audio_path(entry_id))
            if not _identity_matches(identity, entry.get("audio", {}).get("inference", {})):
                raise ValueError("history inference audio identity mismatch")
            return dict(entry)

    def snapshot_for_retry(self, entry_id: str) -> RetrySnapshot | None:
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            entry = self.history._find_locked(entry_id)
            if entry is None: return None
            path = self.history.audio_path(entry_id)
            samples, identity = read_canonical_wav(path)
            expected = entry.get("audio", {}).get("inference", {})
            if not _identity_matches(identity, expected):
                raise ValueError("history inference audio identity mismatch")
            return RetrySnapshot(entry_id, entry["revision"], path, identity, samples)

    def commit_retry(self, entry_id: str, text: str, expected_revision: int, **metadata: Any) -> dict[str, Any] | None:
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            entry=self.history._find_locked(entry_id)
            if entry is None or entry.get("revision") != expected_revision:
                return None
            # This mutation's comparator delivery link names the *next*
            # revision. Preserve only that exact future link while fencing
            # all evidence for the revision being replaced.
            self._revoke_silver_before_history_mutation(entry_id,history_revision=int(entry["revision"]),operation="retry")
            return self.history._commit_retry_locked(entry_id, expected_revision, text, **metadata)

    def correct(self, entry_id: str, corrected_text: str, *, ts: float | None = None,
                base_attempt: int | None = None, expected_revision: int | None = None,
                auto_enroll: bool = False) -> dict[str, Any] | None:
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            entry = self.history._find_locked(entry_id)
            if entry is None or (expected_revision is not None and entry.get("revision") != expected_revision): return None
            self._revoke_silver_before_history_mutation(entry_id,history_revision=int(entry["revision"]),operation="delete")
            # A later correction invalidates its prior explicit consent first.
            for record in list(self.learning.active_for_history(entry_id)):
                self.learning.revoke(record["sample_id"])
            marker=entry.get("silver_enqueue")
            if isinstance(marker,dict) and marker.get("history_revision") == entry.get("revision"):
                marker["state"]="released"
            entry["revision"] += 1
            entry["text"] = corrected_text
            entry["correction"] = {"text": corrected_text, "outcome": "corrected", "ts": time.time() if ts is None else ts,
                                   "base_attempt": len(entry.get("attempts", [])) - 1 if base_attempt is None else base_attempt}
            self.history._save_locked()
            result = dict(entry)
        if auto_enroll:
            self._auto_enroll(entry_id)
        return result

    def correct_as_is(self, entry_id: str, *, expected_revision: int | None = None,
                      auto_enroll: bool = False) -> dict[str, Any] | None:
        """Explicitly accept the currently displayed revision as human-verified gold."""
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked(); entry = self.history._find_locked(entry_id)
            if entry is None or (expected_revision is not None and entry.get("revision") != expected_revision): return None
            self._revoke_silver_before_history_mutation(entry_id,history_revision=int(entry["revision"]))
            for record in list(self.learning.active_for_history(entry_id)): self.learning.revoke(record["sample_id"])
            # Certify exactly the text the user saw in the history menu; the
            # immutable hypothesis remains available for comparison/glossary.
            marker=entry.get("silver_enqueue")
            if isinstance(marker,dict) and marker.get("history_revision") == entry.get("revision"):
                marker["state"]="released"
            entry["revision"] += 1; entry["text"] = entry.get("text", "")
            entry["correction"] = {"text": entry["text"], "outcome": "correct_as_is", "ts": time.time(), "base_attempt": len(entry.get("attempts", [])) - 1}
            self.history._save_locked(); result = dict(entry)
        if auto_enroll: self._auto_enroll(entry_id)
        return result

    def no_speech(self, entry_id: str, *, expected_revision: int | None = None,
                  auto_enroll: bool = False) -> dict[str, Any] | None:
        """Explicit human no-speech annotation.  It is consented gold with empty text."""
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked(); entry = self.history._find_locked(entry_id)
            if entry is None or (expected_revision is not None and entry.get("revision") != expected_revision): return None
            self._revoke_silver_before_history_mutation(entry_id,history_revision=int(entry["revision"]))
            for record in list(self.learning.active_for_history(entry_id)): self.learning.revoke(record["sample_id"])
            marker=entry.get("silver_enqueue")
            if isinstance(marker,dict) and marker.get("history_revision") == entry.get("revision"):
                marker["state"]="released"
            entry["revision"] += 1; entry["text"] = ""
            entry["correction"] = {"text": "", "outcome": "no_speech", "ts": time.time(), "base_attempt": len(entry.get("attempts", [])) - 1}
            self.history._save_locked(); result = dict(entry)
        if auto_enroll: self._auto_enroll(entry_id)
        return result

    def _auto_enroll(self, entry_id: str) -> LearningSnapshot | None:
        try:
            snapshot = self.enroll(entry_id)
            if snapshot is None:
                self.learning.mark_pending_gold(entry_id, "enrollment_conflict")
            else:
                self.learning.clear_pending_gold(entry_id)
            return snapshot
        except Exception as exc:
            self.learning.mark_pending_gold(entry_id, type(exc).__name__)
            return None

    def retry_pending_gold(self) -> int:
        """Retry durable automatic-enrollment failures without losing consent."""
        completed = 0
        for row in self.learning.pending_gold():
            if self._auto_enroll(row["history_id"]) is not None: completed += 1
        return completed

    def enroll(self, entry_id: str) -> LearningSnapshot | None:
        """Copy outside lock, then CAS validate and publish active metadata last."""
        sample_id = uuid.uuid4().hex
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            entry = self.history._find_locked(entry_id)
            if entry is None or entry.get("correction") is None: return None
            revision = entry["revision"]
            corrected_text = entry["correction"]["text"]
            existing = next((record for record in self.learning.active_for_history(entry_id)
                             if record.get("history_revision") == revision), None)
            if existing is not None:
                return _snapshot_from_record(self.learning, existing)
            source = self.history.audio_path(entry_id)
            _, identity = read_canonical_wav(source)
            source_expected = entry.get("audio", {}).get("inference", {})
            if not _identity_matches(identity, source_expected):
                raise ValueError("source audio identity mismatch")
            raw_info = entry.get("audio", {}).get("raw")
            raw_source = self.history.raw_audio_path(entry_id) if raw_info else None
            if raw_info is not None and raw_source is None:
                raise ValueError("history raw audio path is unsafe")
            raw_identity = None
            if raw_source is not None:
                _, raw_identity = read_pcm16_wav(raw_source)
                if not _identity_matches(raw_identity, raw_info):
                    raise ValueError("source raw audio identity mismatch")
        stage = self.learning.stage_copy(sample_id, source, raw_source,
                                         inference_identity=identity, raw_identity=raw_identity)
        try:
            with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
                self._reload_current_locked()
                current = self.history._find_locked(entry_id)
                now_identity = read_canonical_wav(source)[1]
                now_raw_identity = read_pcm16_wav(raw_source)[1] if raw_source is not None else None
                existing = next((record for record in self.learning.active_for_history(entry_id)
                                 if record.get("history_revision") == revision), None)
                if existing is not None:
                    shutil.rmtree(stage, ignore_errors=True); _fsync_dir(self.learning.staging_dir)
                    return _snapshot_from_record(self.learning, existing)
                if (current is None or current.get("revision") != revision or current.get("correction") is None
                        or current["correction"].get("text") != corrected_text or now_identity != identity
                        or now_raw_identity != raw_identity
                        or not _identity_matches(now_identity, current.get("audio", {}).get("inference"))
                        or (now_raw_identity is not None
                            and not _identity_matches(now_raw_identity, current.get("audio", {}).get("raw")))):
                    shutil.rmtree(stage, ignore_errors=True); _fsync_dir(self.learning.staging_dir)
                    return None
                record = {"schema_version": LEARNING_SCHEMA_VERSION, "sample_id": sample_id, "status": "active",
                    "created_ts": time.time(), "history_id": entry_id, "history_revision": revision,
                    "corrected_text": corrected_text, "inference_audio": identity.as_dict(),
                    "raw_audio": raw_identity.as_dict() if raw_identity is not None else None,
                    "raw_present": raw_identity is not None}
                self.learning.publish_staged(sample_id, stage, record)
                return _snapshot_from_record(self.learning, record)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True); _fsync_dir(self.learning.staging_dir)
            raise

    def revoke(self, sample_id: str) -> bool:
        with self._lock, advisory_lock(self.history.base_dir):
            self.learning._reload_locked()
            return self.learning.revoke(sample_id)

    def revoke_history(self, entry_id: str) -> int:
        """Revoke every currently active sample for a history entry."""
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            entry=self.history._find_locked(entry_id)
            # This public coordinator seam must fence adaptive generations as
            # well as durable-learning samples even when no runtime wrapper
            # is involved.  Both operations are idempotent for absent/already
            # revoked captures.
            from adaptive_learning import AdaptiveLearning
            adaptive=AdaptiveLearning(self.history.base_dir)
            adaptive.pre_mutate_reference(entry_id,reason="history_revoked")
            adaptive.revoke(entry_id,reason="history_revoked")
            if entry is not None:
                self._revoke_silver_before_history_mutation(entry_id,history_revision=int(entry["revision"]),operation="revoke")
                marker=entry.get("silver_enqueue")
                if isinstance(marker,dict) and marker.get("history_revision") == entry.get("revision"):
                    marker["state"]="released"
                    self.history._save_locked()
            count = 0
            for record in list(self.learning.active_for_history(entry_id)):
                if self.learning.revoke(record["sample_id"]):
                    count += 1
            return count

    def finish_pending_revoke_intent(self, intent: dict[str, Any]) -> None:
        """Idempotently finish one content-free schema-2 revoke target."""
        from silver_store import _valid_revoke_intent
        if not _valid_revoke_intent(intent):
            raise ValueError("invalid revoke intent")
        history_id=str(intent["history_id"]); old_revision=intent["history_revision"]
        operation=str(intent["operation"])
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            from adaptive_learning import AdaptiveLearning
            adaptive=AdaptiveLearning(self.history.base_dir)
            adaptive.pre_mutate_reference(history_id,reason="revoke_recovery")
            adaptive.revoke(history_id,reason="revoke_recovery")
            entry=self.history._find_locked(history_id)
            marker_changed=False
            if entry is not None:
                marker=entry.get("silver_enqueue")
                if isinstance(marker,dict) and marker.get("history_revision") == old_revision and marker.get("state") != "released":
                    marker["state"]="released"; marker_changed=True
            for record in list(self.learning.active_for_history(history_id)):
                self.learning.revoke(record["sample_id"])
            self.learning.clear_pending_gold(history_id)
            if operation == "delete" and entry is not None and old_revision is not None and entry.get("revision") == old_revision:
                self.history._delete_locked(history_id)
                return
            if marker_changed:
                self.history._save_locked()

    def delete(self, entry_id: str) -> bool:
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            entry=self.history._find_locked(entry_id)
            if entry is None:
                return False
            # LearningCoordinator is a public mutation surface too.  Fence a
            # pending/admitted adaptive generation even when callers bypass
            # AdaptiveRuntime/HistoryStore convenience wrappers.
            from adaptive_learning import AdaptiveLearning
            AdaptiveLearning(self.history.base_dir).revoke(entry_id, reason="history_deleted")
            # Keep the same advisory fence the worker holds across marker
            # validation/enqueue.  Revocation therefore cannot observe an
            # empty job set and then allow a paused worker to enqueue before
            # history deletion commits.
            self._revoke_silver_before_history_mutation(entry_id,history_revision=int(entry["revision"]),operation="delete")
            # Do not remove history if any durable-learning revocation fails.
            for record in list(self.learning.active_for_history(entry_id)):
                self.learning.revoke(record["sample_id"])
            self.learning.clear_pending_gold(entry_id)
            return self.history._delete_locked(entry_id)

    def clear(self) -> None:
        # Epoch/deployment invalidation happens outside history locks, avoiding
        # cross-store lock inversion with a worker result commit.  A folder
        # that has never had a silver ledger has no epoch to fence.
        lane=silver_lane_exists(self.history.base_dir)
        if lane:
            from silver_store import SilverStore
            from deployment_controller import DeploymentController
            store=SilverStore(self.history.base_dir)
            store.clear(reason="history_clear")
            scrub_intent=store.scrub_pending()
            DeploymentController(self.history.base_dir).invalidate("history_clear")
        from adaptive_learning import AdaptiveLearning
        # Clear is stronger than ordinary revocation.  Publish an empty
        # personal adaptive namespace rather than retaining revoked capture
        # ids, references, manifests or runtime audit data in state.json.
        AdaptiveLearning(self.history.base_dir).clear_personal_state()
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            # Without an epoch fence, a ledger created since the check above
            # (a worker enqueues only under this advisory lock) must stop the
            # wipe; a retried Clear then fences it.
            if not lane and silver_lane_exists(self.history.base_dir):
                raise RuntimeError("silver ledger appeared during clear; retry")
            # Epoch fencing occurs before removing any history/audio so a
            # stale teacher result cannot publish into a newly cleared corpus.
            for record in list(self.learning.active()):
                self.learning.revoke(record["sample_id"])
            for pending in list(self.learning.pending_gold()):
                self.learning.clear_pending_gold(pending["history_id"])
            captured = list(self.history._entries)
            self.history._entries = []
            self.history._save_locked()
            for entry in captured: self.history._unlink_entry_audio_locked(entry)
        # The SQLite epoch fence is already committed.  Complete the durable
        # outbox synchronously on the ordinary path; recovery performs these
        # same idempotent erasures if this process dies between steps.
        if lane:
            from silver_experiment import SilverExperiment
            SilverExperiment(self.history.base_dir).clear_personal_state()
            DeploymentController(self.history.base_dir).clear_personal_state()
            # SQLite has already removed comparator intent/pair authority.  Its
            # private WAVs must be scrubbed before the exact clear token is
            # acknowledged; unsafe artifacts deliberately retain the hard fence.
            # (Without a lane the evidence directory is empty: no WAV exists.)
            if not scrub_comparator_spools(self.history.base_dir):
                raise RuntimeError("comparator scrub is blocked")
            if not store.complete_scrub(scrub_intent):
                raise RuntimeError("clear scrub acknowledgement changed")

    def finish_pending_clear(self) -> None:
        """Complete a SilverStore-fenced clear after a crash without new epoch."""
        from adaptive_learning import AdaptiveLearning
        AdaptiveLearning(self.history.base_dir).clear_personal_state()
        with self._lock, advisory_lock(self.history.base_dir), self.history._lock:
            self._reload_current_locked()
            for record in list(self.learning.active()): self.learning.revoke(record["sample_id"])
            for pending in list(self.learning.pending_gold()): self.learning.clear_pending_gold(pending["history_id"])
            captured=list(self.history._entries); self.history._entries=[]; self.history._save_locked()
            for entry in captured: self.history._unlink_entry_audio_locked(entry)
            # A crash can occur after the empty index save but before the
            # per-entry unlink loop.  With authoritative history now empty,
            # every regular WAV in owned history/learning artifact directories
            # is orphaned clear residue and may be safely removed.
            for directory in (self.history.audio_dir,self.history.raw_audio_dir,self.learning.audio_dir,self.learning.raw_dir):
                for path in directory.glob("*.wav"):
                    if path.is_file() and not path.is_symlink(): path.unlink(missing_ok=True)
                _fsync_dir(directory)

    def learning_status(self) -> dict[str, int]:
        with self._lock: return {"active": len(self.learning.active()), "pending_gold": len(self.learning.pending_gold())}

    def export_learning_snapshot(self) -> list[LearningSnapshot]:
        """In-process export snapshot; use ``LearningStore.admin_export_guard`` across processes."""
        with self._lock, advisory_lock(self.history.base_dir):
            self.learning._reload_locked()
            result = []
            for row in list(self.learning.active()):
                if not self.learning.artifacts_valid(row):
                    self.learning.revoke(row["sample_id"])
                    continue
                result.append(_snapshot_from_record(self.learning, row))
            return result

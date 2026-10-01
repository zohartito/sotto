"""Integration layer for Sotto's human-grounded adaptive English runtime.

The statistical protocol remains in :mod:`adaptive_learning`; this module
connects it to durable history, local model receipts, and the priority lease.
It intentionally has no AppKit or eager model imports.
"""
from __future__ import annotations

import os
import hashlib
import json
import stat
import threading
import time
import uuid
import fcntl
import io
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from adaptive_learning import AdaptiveLearning
from audio_codec import PreparedAudio, read_canonical_wav, prepare_canonical, write_canonical_wav
from comparator_spool import ComparatorSpool
from history import HistoryStore
from inference_scheduler import InferenceScheduler
from learning import LearningCoordinator, LearningStore, adaptive_review_eligible_entry
from deployment_controller import DeploymentController
from silver_store import SilverStore
from silver_experiment import SilverExperiment
from speech_backends import (LocalBackendManager, PARAKEET_ID, PreflightReceipts,
                             WHISPER_GLOSSARY_ID, WHISPER_ID, candidate_manifest,
                             evaluator_hash)
from speech_config import _clean_terms
from storage_lock import advisory_lock
from evaluation import tokenize


class AdaptiveRuntime:
    """Durable orchestration, designed to be injectable in no-model tests."""
    def __init__(self, base_dir: Path | str | None = None, *, history: HistoryStore | None = None,
                 learning: LearningStore | None = None, coordinator: LearningCoordinator | None = None,
                 adaptive: AdaptiveLearning | None = None, scheduler: InferenceScheduler | None = None,
                 backends: LocalBackendManager | None = None) -> None:
        root = Path(base_dir) if base_dir is not None else None
        self.history = history or HistoryStore(root)
        base = self.history.base_dir
        self.learning = learning or LearningStore(base)
        self.coordinator = coordinator or LearningCoordinator(self.history, self.learning)
        self.adaptive = adaptive or AdaptiveLearning(base)
        self.scheduler = scheduler or InferenceScheduler(base)
        self.backends = backends or LocalBackendManager(base)
        self.receipts = PreflightReceipts(base)
        self.silver = SilverStore(base)
        self.deployment = DeploymentController(base, silver=self.silver)
        self.comparator_spool = ComparatorSpool(base)
        self._orphan_sweep_lock = threading.Lock()
        self._orphan_sweep_timer: threading.Timer | None = None
        self._orphan_sweep_rearm = False

    def schedule_orphan_sweep(self, *, is_shutdown=lambda: False, _delay: float | None = None) -> None:
        """One coalesced deferred recovery pass just past the lease window.

        A live-path failure can strand a prepared comparator intent/spool that
        the store's expiry sweep may only discard after its lease window; this
        guarantees a reconcile fires past that window within this session even
        if no further capture or restart ever runs.  At most one timer is
        alive at a time; a request arriving while one is pending re-arms a
        single fresh full-window timer when it fires, so an orphan created
        mid-window still gets a sweep past its OWN expiry.
        """
        with self._orphan_sweep_lock:
            existing = self._orphan_sweep_timer
            if existing is not None and existing.is_alive():
                self._orphan_sweep_rearm = True
                return
            self._start_orphan_sweep_timer(is_shutdown, _delay)

    def _start_orphan_sweep_timer(self, is_shutdown, _delay: float | None) -> None:
        """Caller holds ``_orphan_sweep_lock``."""
        delay = float(getattr(self.silver, "lease_seconds", 3600)) + 60.0 if _delay is None else _delay
        def _sweep() -> None:
            if not is_shutdown():
                try:
                    self.reconcile(retry_pending=False)
                except Exception:
                    pass
            with self._orphan_sweep_lock:
                rearm = self._orphan_sweep_rearm
                self._orphan_sweep_rearm = False
                if rearm and not is_shutdown():
                    self._start_orphan_sweep_timer(is_shutdown, _delay)
        timer = threading.Timer(delay, _sweep)
        timer.daemon = True
        timer.start()
        self._orphan_sweep_timer = timer

    def _prepare_comparator_spool(self, samples, sample_rate: int) -> tuple[bytes, str, int]:
        if sample_rate != 16_000:
            raise ValueError("invalid comparator sample rate")
        prepared=samples if isinstance(samples,PreparedAudio) else prepare_canonical(samples); buffer=io.BytesIO()
        with wave.open(buffer,"wb") as handle:
            handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16_000); handle.setcomptype("NONE","not compressed"); handle.writeframes(prepared.pcm)
        return buffer.getvalue(),prepared.identity.sha256,prepared.identity.sample_count

    def _live_audio_authority(self, samples, canonical_path, canonical_identity) -> tuple[bytes, str, int] | None:
        """Freeze exactly the live PCM only when memory and owned WAV agree."""
        try:
            prepared=samples if isinstance(samples,PreparedAudio) else prepare_canonical(samples); identity=prepared.identity
            expected=(canonical_identity.as_dict() if hasattr(canonical_identity,"as_dict") else canonical_identity)
            if not isinstance(expected,dict) or identity.as_dict() != expected:
                return None
            raw=Path(canonical_path)
            allowed={self.history.audio_dir,self.history.silver_evidence_dir}
            if raw.parent not in allowed or raw.is_symlink() or stat.S_ISLNK(os.lstat(raw).st_mode) or not stat.S_ISREG(os.lstat(raw).st_mode):
                return None
            if any(parent.is_symlink() or stat.S_ISLNK(os.lstat(parent).st_mode) for parent in (raw.parent,)):
                return None
            _disk_samples,disk_identity=read_canonical_wav(raw)
            if disk_identity != identity:
                return None
            buffer=io.BytesIO()
            with wave.open(buffer,"wb") as handle:
                handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16_000); handle.setcomptype("NONE","not compressed"); handle.writeframes(prepared.pcm)
            return buffer.getvalue(),identity.sha256,identity.sample_count
        except (OSError, ValueError, TypeError):
            return None

    def _same_live_audio_authority(self, token: tuple[bytes, str, int] | None, samples, canonical_path, canonical_identity) -> bool:
        """Re-derive the live authority tuple; any mutation is nonauthority."""
        if (not isinstance(token,tuple) or len(token) != 3 or not isinstance(token[0],bytes) or
                not isinstance(token[1],str) or isinstance(token[2],bool) or not isinstance(token[2],int)):
            return False
        return self._live_audio_authority(samples,canonical_path,canonical_identity) == token

    @staticmethod
    def _comparator_capture_id(session_id: str, audio_sha256: str, sample_count: int, deployment_revision: int,
                               caller_capture_id: str | None = None) -> str:
        if (not isinstance(session_id,str) or not session_id or not isinstance(audio_sha256,str) or len(audio_sha256)!=64 or
                any(ch not in '0123456789abcdef' for ch in audio_sha256) or isinstance(sample_count,bool) or not isinstance(sample_count,int) or sample_count < 0 or
                isinstance(deployment_revision,bool) or not isinstance(deployment_revision,int) or deployment_revision < 0 or
                (caller_capture_id is not None and (not isinstance(caller_capture_id,str) or not caller_capture_id))):
            raise ValueError('invalid comparator capture identity')
        caller=caller_capture_id or ''
        return hashlib.sha256(f'sotto-comparator-v2\0{deployment_revision}\0{caller}\0{session_id}\0{audio_sha256}\0{sample_count}'.encode()).hexdigest()

    def _write_comparator_spool(self, wav_bytes: bytes, capture_id: str, expected_sha: str) -> tuple[Path, str]:
        if not isinstance(capture_id,str) or not capture_id or not isinstance(wav_bytes,bytes) or not isinstance(expected_sha,str):
            raise ValueError("invalid comparator spool identity")
        name=f"{capture_id}.wav"
        self.comparator_spool.write(name,wav_bytes,expected_sha)
        return self.comparator_spool.root / name,expected_sha

    def _delete_comparator_spool(self, capture_id: str, expected_sha: str) -> bool:
        if not isinstance(capture_id,str) or not capture_id or not isinstance(expected_sha,str):
            return False
        try:
            return self.comparator_spool.unlink(f"{capture_id}.wav",expected_sha)
        except (FileNotFoundError,OSError,RuntimeError,ValueError):
            return False

    def _cleanup_non_durable_comparator_spool(self, *, deployment_revision: int,
                                               capture_id: str, spool: Path,
                                               audio_sha256: str) -> str:
        """Unlink only while SQLite excludes a new exact outbox owner."""
        with self.silver.comparator_spool_cleanup_fence(
                deployment_revision=deployment_revision,capture_id=capture_id,
                spool_name=spool.name,audio_sha256=audio_sha256) as state:
            if state == "unlink":
                self._delete_comparator_spool(capture_id,audio_sha256)
            return state

    def acknowledge_comparator_publication(self, metadata: dict[str,Any]) -> bool:
        value=metadata.get("comparator_publication") if isinstance(metadata,dict) else None
        if not isinstance(value,dict): return True
        try:
            exact={key:value[key] for key in ("deployment_revision","capture_id","spool_name","audio_sha256","candidate")}
            adoptions=[item for item in self.silver.comparator_publication_adoptions()
                        if item["deployment_revision"] == exact["deployment_revision"] and item["capture_id"] == exact["capture_id"]]
            if len(adoptions) > 1:
                return False
            if adoptions:
                adoption=adoptions[0]
                # Match the coordinator's mutation order so a correction or
                # revoke and this final promotion cannot pass each other.
                with self.coordinator._lock, advisory_lock(self.history.base_dir), self.history._lock:
                    self.history._reload_locked()
                    entry=self.history._find_locked(adoption["history_id"])
                    audio=entry.get("audio",{}).get("inference",{}) if isinstance(entry,dict) else None
                    if (not isinstance(entry,dict) or entry.get("revision") != adoption["history_revision"]
                            or not isinstance(audio,dict) or audio.get("sha256") != adoption["audio_sha256"]):
                        candidate=exact["candidate"]
                        self.silver.discard_comparator_intent(capture_id=exact["capture_id"],
                            deployment_revision=exact["deployment_revision"],spool_name=exact["spool_name"],
                            audio_sha256=exact["audio_sha256"],candidate=candidate)
                        return False
                    # Hold the history/advisory fence through the SQLite CAS.
                    return bool(self.silver.acknowledge_adopted_comparator_publication(**exact))
            adopted=self.silver.acknowledge_adopted_comparator_publication(**exact)
            if adopted is not None:
                return adopted
            return self.silver.acknowledge_comparator_intent(**exact)
        except (KeyError,TypeError,ValueError,RuntimeError):
            return False

    def adopt_comparator_publication(self, metadata: dict[str,Any], *, history_id: str,
                                     history_revision: int) -> bool:
        """Write the content-free delivery link before History mutates.

        History deliberately never receives this internal comparator token.
        The Silver meta outbox lets restart reconcile only a subsequently
        committed exact History revision into worker-visible delivery.
        """
        value=metadata.get("comparator_publication") if isinstance(metadata,dict) else None
        if not isinstance(value,dict): return True
        try:
            return self.silver.adopt_comparator_publication(
                **{key:value[key] for key in ("deployment_revision","capture_id","spool_name","audio_sha256","candidate")},
                history_id=history_id,history_revision=history_revision)
        except (KeyError,TypeError,ValueError):
            return False

    def recover_comparator_publications(self) -> int:
        """Reconcile durable delivery links after a crash between commit and ack."""
        recovered=0
        for value in self.silver.comparator_publication_adoptions():
            entry=self.history.get(value["history_id"])
            audio=entry.get("audio",{}).get("inference",{}) if isinstance(entry,dict) else None
            candidate=self.silver.comparator_candidate(value["deployment_revision"],value["capture_id"])
            if candidate is None:
                raise RuntimeError("comparator adoption intent is unavailable")
            if (not isinstance(entry,dict) or entry.get("revision") != value["history_revision"]
                    or not isinstance(audio,dict) or audio.get("sha256") != value["audio_sha256"]):
                # The durable adoption preceded a History commit which never
                # materialized (or was corrected).  It is not delivery.
                if not self.silver.discard_comparator_intent(capture_id=value["capture_id"],
                    deployment_revision=value["deployment_revision"],spool_name=value["spool_name"],
                    audio_sha256=value["audio_sha256"],candidate=candidate):
                    raise RuntimeError("comparator unadopted publication discard failed")
                continue
            if not self.acknowledge_comparator_publication({"comparator_publication":{
                    "deployment_revision":value["deployment_revision"],"capture_id":value["capture_id"],
                    "spool_name":value["spool_name"],"audio_sha256":value["audio_sha256"],
                    "candidate":candidate}}):
                raise RuntimeError("comparator adoption acknowledgement failed")
            recovered+=1
        # Prepared intents whose owner died before terminalization are swept
        # by the store's lease-expiry recovery.  The headless worker normally
        # runs it; when only the app is installed this runtime is the sole
        # caller, so run it here and unlink the owned terminal spools it
        # leaves, keeping no orphaned private audio behind a failed
        # publication.  Expiry timing stays the store's own safety margin.
        recovered+=self.silver.recover()
        for spool_name,audio_sha256 in self.silver.terminal_comparator_spools().items():
            self._delete_comparator_spool(spool_name[:-len(".wav")],audio_sha256)
        return recovered

    def cancel_comparator_publication(self, metadata: dict[str,Any]) -> bool:
        value=metadata.get("comparator_publication") if isinstance(metadata,dict) else None
        if not isinstance(value,dict): return True
        try:
            if not self.silver.discard_comparator_intent(**{key:value[key] for key in ("deployment_revision","capture_id","spool_name","audio_sha256","candidate")}): return False
            self._delete_comparator_spool(str(value["capture_id"]),str(value["audio_sha256"])); return True
        except (KeyError,TypeError,ValueError):
            return False

    def fail_comparator_publication(self, metadata: dict[str,Any]) -> bool:
        value=metadata.get("comparator_publication") if isinstance(metadata,dict) else None
        return self.deployment.invalidate_comparator_publication(value,"comparator_publication_failed") if isinstance(value,dict) else False

    def comparator_publication_recovery_pending(self, metadata: dict[str,Any]) -> bool:
        """Whether a committed delivery is correctly waiting behind scrub."""
        value=metadata.get("comparator_publication") if isinstance(metadata,dict) else None
        if not isinstance(value,dict) or self.silver.scrub_pending() is None:
            return False
        try:
            return any(item["deployment_revision"] == value["deployment_revision"] and
                       item["capture_id"] == value["capture_id"]
                       for item in self.silver.comparator_publication_adoptions())
        except (KeyError,TypeError,ValueError,RuntimeError):
            return False

    def _capture_from_history(self, entry: dict[str, Any], *, adaptive: bool) -> dict[str, Any] | None:
        audio = entry.get("audio", {}).get("inference", {})
        digest = audio.get("sha256")
        if not isinstance(entry.get("id"), str):
            return None
        # HistoryStore is the sole authority for metadata-controlled audio
        # paths.  In particular, do not join a persisted relative value here.
        try:
            audio_path = self.history.audio_path(entry["id"])
        except ValueError:
            return None
        if not isinstance(digest, str):
            try:
                _, identity = read_canonical_wav(audio_path)
            except (OSError, ValueError, EOFError):
                return None
            digest = identity.sha256
            audio = {"path": str(audio_path.relative_to(self.history.base_dir)), **identity.as_dict()}
        return {"history_id": entry["id"], "captured_ts": entry.get("ts", 0), "duration": entry.get("duration", 0),
                "audio_digest": digest, "audio_path": str(audio_path), "audio_identity": dict(audio),
                "schema_version": entry.get("schema_version", 0), "canonical_16k_pcm": audio.get("format") == "pcm_s16le_mono_16000",
                "adaptive": adaptive, "language": "en" if adaptive else entry.get("language")}

    @staticmethod
    def review_eligible_entry(entry: dict[str, Any] | None) -> bool:
        """Content-free UI/runtime gate for adaptive human actions."""
        return adaptive_review_eligible_entry(entry)

    def register_retained_history(self) -> int:
        """Register metadata only; old rows stay reviewable but ineligible."""
        # A durable review hand-off wins over ordinary retained-history
        # registration: it restores/revokes its adaptive row before any new
        # generation can observe stale gold after a restart.
        self.recover_review_outbox()
        count = 0
        for entry in self.history.entries(limit=10_000):
            # VAD-gated explicit no-speech captures have no ASR candidate
            # provenance, but are still forced-English adaptive canonical
            # captures and valid human-review evidence after restart.
            is_new = (entry.get("schema_version") == 2 and entry.get("language") == "en"
                      and entry.get("adaptive") is True)
            capture = self._capture_from_history(entry, adaptive=is_new)
            if capture:
                self.adaptive.register_capture(capture); count += 1
        return count

    def register_live(self, entry: dict[str, Any]) -> None:
        """Register only the exact still-live history revision.

        History append and adaptive registration are separate durable files,
        so this is an explicit recoverable fence rather than an optimistic
        best-effort callback.  Clear/revoke commits its SQLite scrub marker
        before taking the same advisory/history lock; a delayed callback then
        observes the fence and cannot recreate a capture after erasure.
        """
        if entry.get("schema_version") != 2 or entry.get("language") != "en" or entry.get("adaptive") is not True:
            return
        ident=entry.get("id"); revision=entry.get("revision")
        if not isinstance(ident,str) or isinstance(revision,bool) or not isinstance(revision,int):
            return
        with advisory_lock(self.history.base_dir), self.history._lock:
            self.history._reload_locked()
            # Any exact scrub marker is a hard cross-store authority fence.
            # Do not manufacture a new adaptive record while recovery owns
            # deletion/revocation of the matching personal capture.
            if self.silver.scrub_pending() is not None:
                return
            visible=self.history._find_locked(ident)
            if visible is None or visible.get("revision") != revision or visible.get("correction") is not None:
                return
            capture = self._capture_from_history(visible, adaptive=True)
            if capture is None or not capture["canonical_16k_pcm"]:
                return
            self.adaptive.register_capture(capture)
        self.reconcile()

    def review_list(self) -> list[dict[str, Any]]:
        active = self.adaptive.active_generation()
        # Pool members are always shown; otherwise this is the salted ordinary
        # queue.  The core returns no text/reference fields.
        return self.adaptive.review_list(active["id"] if active and active["status"] == "pool_open" else None)

    def _enroll(self, history_id: str, _reference: str) -> None:
        try:
            if self.coordinator.enroll(history_id) is None:
                raise RuntimeError("enrollment_conflict")
            self.learning.clear_pending_gold(history_id)
        except Exception as exc:
            self.learning.mark_pending_gold(history_id, type(exc).__name__)
            raise

    def recover_review_outbox(self, history_ids: list[str] | None = None, *, retry_enrollment: bool = True) -> int:
        """Finish revision-keyed, content-free review hand-offs after restart.

        References are reloaded only from the reviewed history row after its
        canonical audio identity and revision have been checked.  Any failure
        is propagated so callers fail closed rather than advance a generation
        from a mixed history/adaptive state.
        """
        identifiers = history_ids if history_ids is not None else self.coordinator.adaptive_review_entries()
        completed = 0
        for history_id in identifiers:
            entry = self.coordinator.load_adaptive_review(history_id)
            if entry is None:
                continue
            operation = entry["adaptive_review"]
            outcome, revision = operation.get("outcome"), operation.get("revision")
            if outcome not in {"corrected", "correct_as_is", "no_speech", "skip"} or not isinstance(revision, int):
                raise RuntimeError("adaptive review outbox is malformed")
            # The durable outbox precedes artifact tombstones.  If the original
            # post-outbox revoke crashed, finish it before adaptive state can
            # again treat the old reviewed material as usable.
            self.coordinator.reconcile_adaptive_review_artifacts(history_id)
            state = self.adaptive.review_operation_state(history_id, revision, outcome)
            # Completed operations stay in the durable outbox as their replay
            # key.  Do not even refresh capture metadata for a no-op action.
            if state["complete"]:
                if state["pending_gold"]:
                    if not retry_enrollment:
                        raise RuntimeError("adaptive review enrollment is pending")
                    correction = entry.get("correction")
                    if not isinstance(correction, dict) or not isinstance(correction.get("text"), str):
                        raise RuntimeError("reviewed history reference is unavailable")
                    self._enroll(history_id, correction["text"])
                    if not self.adaptive.complete_review_enrollment(history_id, revision, outcome):
                        raise RuntimeError("adaptive review enrollment revision conflict")
                if not self.coordinator.mark_adaptive_review_complete(history_id, revision, outcome):
                    raise RuntimeError("adaptive review outbox revision conflict")
                continue
            capture = self._capture_from_history(entry, adaptive=entry.get("adaptive") is True)
            if capture is None or capture.get("adaptive") is not True:
                raise RuntimeError("adaptive review outbox capture is unavailable")
            self.adaptive.register_capture(capture)
            # This also invalidates a dependency that was mistakenly made
            # during an earlier partial process before the outbox resumed.
            self.adaptive.pre_mutate_reference(history_id, reason="review_outbox_recovery")
            if outcome in {"corrected", "correct_as_is", "no_speech"}:
                correction = entry.get("correction")
                if not isinstance(correction, dict) or correction.get("outcome") != outcome:
                    raise RuntimeError("reviewed history disposition is unavailable")
                reference = correction.get("text")
                if not isinstance(reference, str):
                    raise RuntimeError("reviewed history reference is malformed")
            else:
                reference = ""
            self.adaptive.resolve_review(history_id, outcome, reference=reference,
                                         skip_reason=operation.get("skip_reason"),
                                         review_revision=revision, auto_enroll=self._enroll)
            state = self.adaptive.review_operation_state(history_id, revision, outcome)
            if not state["pending_gold"]:
                if not self.coordinator.mark_adaptive_review_complete(history_id, revision, outcome):
                    raise RuntimeError("adaptive review outbox revision conflict")
            completed += 1
        return completed

    def _review_recovery_gate(self, *, retry_pending: bool) -> None:
        """Converge every cross-store review before evidence can advance."""
        self.recover_review_outbox(retry_enrollment=retry_pending)
        if retry_pending:
            self.retry_pending_gold()
        if self.coordinator.adaptive_review_entries() or self.adaptive.pending_review_enrollments():
            raise RuntimeError("adaptive review recovery is incomplete")

    def review(self, history_id: str, outcome: str, *, reference: str = "", reason: str | None = None,
               expected_revision: int | None = None) -> None:
        """Apply an explicit human action and automatically enroll its gold."""
        if outcome not in {"corrected", "correct_as_is", "no_speech", "skip"}:
            raise ValueError("invalid review outcome")
        visible = self.history.get(history_id)
        if visible is None or (expected_revision is not None and visible.get("revision") != expected_revision):
            raise KeyError(history_id)
        if not self.review_eligible_entry(visible):
            raise ValueError("history row is not eligible for adaptive review")
        prior = self.coordinator.adaptive_review_metadata(history_id)
        # A refreshed menu action for an already committed non-textual review
        # completes its existing outbox record rather than opening a second one.
        if not (outcome in {"skip", "no_speech", "correct_as_is"} and prior and
                prior.get("revision") == visible.get("revision") and prior.get("outcome") == outcome):
            # The coordinator's mutation fence revokes the silver label first.
            # It then releases the old revision marker while still visible;
            # releasing before durable revocation would permit a crash to prune
            # audio that an outstanding teacher job still owns.
            row = self.coordinator.commit_adaptive_review(history_id, outcome, reference=reference,
                                                          skip_reason=str(reason) if outcome == "skip" else None,
                                                          expected_revision=expected_revision,
                                                          before_mutation=lambda: self.adaptive.pre_mutate_reference(history_id))
            if row is None:
                raise KeyError(history_id)
        self.recover_review_outbox([history_id], retry_enrollment=False)
        self.reconcile(retry_pending=False)

    def revoke(self, history_id: str, *, reason: str = "revoked") -> bool:
        # The coordinator owns the single SQLite/history fence.  In
        # particular, do not create a legacy revoke marker here or reread an
        # outbox token after it returns: a concurrent Clear may have replaced
        # that marker and only the worker may acknowledge its exact version.
        self.adaptive.pre_mutate_reference(history_id, reason=reason)
        result = self.adaptive.revoke(history_id, reason=reason)
        self.coordinator.revoke_history(history_id)
        self.reconcile()
        return result

    def retry_pending_gold(self) -> int:
        """Retry the single durable enrollment operation without rereviewing."""
        # Adaptive state is the source of truth for a review outcome; it calls
        # exactly one idempotent coordinator.enroll per pending history id.
        completed = self.adaptive.retry_pending_gold(self._enroll)
        return completed

    def delete(self, history_id: str) -> bool:
        self.adaptive.revoke(history_id, reason="history_deleted")
        # Coordinator.delete is the authoritative history mutation hook; it
        # fences silver/deployment exactly once before deleting audio.
        result=self.coordinator.delete(history_id)
        return result

    def clear(self) -> None:
        self.adaptive.revoke_all(reason="history_cleared")
        # Coordinator.clear owns the epoch transition, cross-store scrub, and
        # its exact outbox acknowledgement.  A second read/ack here could
        # erase a newer clear/revoke marker that raced after it completed.
        self.coordinator.clear()

    def development_glossary(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Mine only development gold, comparing it to immutable hypotheses."""
        active = self.adaptive.active_generation()
        excluded_pool = set()
        if active and active["status"] in {"pool_open", "pool_closed", "evaluating"}:
            # development_references already excludes currently assigned pool;
            # this local marker documents that no promotion reference is read.
            excluded_pool.add(active["id"])
        samples: dict[str, Counter[str]] = {}
        for history_id in self.adaptive.development_reference_ids():
            entry = self.history.get(history_id)
            if not entry or not isinstance(entry.get("correction"), dict):
                continue
            reference = entry["correction"].get("text")
            hypothesis = entry.get("hypothesis")
            if not isinstance(reference, str) or not isinstance(hypothesis, str):
                continue
            ref = tuple(__import__("evaluation").tokenize(reference)); hyp = set(__import__("evaluation").tokenize(hypothesis))
            values: Counter[str] = Counter()
            for n in (1, 2, 3):
                for pos in range(len(ref) - n + 1):
                    phrase = " ".join(ref[pos:pos + n])
                    if all(token not in hyp for token in ref[pos:pos + n]): values[phrase] += 1
            samples[history_id] = values
        occurrences: Counter[str] = Counter()
        dependencies: dict[str, set[str]] = defaultdict(set)
        for ident, values in samples.items():
            for phrase, number in values.items():
                occurrences[phrase] += number; dependencies[phrase].add(ident)
        ordered = [phrase for phrase, number in sorted(occurrences.items(), key=lambda row: (-row[1], row[0]))
                   if len(dependencies[phrase]) >= 2]
        terms = _clean_terms(ordered)
        ids = tuple(sorted({ident for term in terms for ident in dependencies[term]}))
        return terms, ids

    def _receipt_manifest(self, stable_id: str, *, glossary_terms: tuple[str, ...] = ()) -> dict[str, Any] | None:
        receipt = self.receipts.load(stable_id)
        if not receipt or not isinstance(receipt.get("manifest"), dict): return None
        manifest = dict(receipt["manifest"])
        if isinstance(receipt.get("snapshot_metadata_digest"),str): manifest["snapshot_digest"] = receipt["snapshot_metadata_digest"]
        if glossary_terms: manifest["glossary_terms"] = glossary_terms
        try:
            expected = candidate_manifest(stable_id, glossary_terms=glossary_terms, revision=manifest["revision"],
                                          package_versions=manifest.get("package_versions"),
                                          runtime_identity=manifest.get("runtime_identity"))
        except (KeyError, ValueError): return None
        if not self.receipts.valid_manifest(expected): return None
        expected["snapshot_path"] = receipt.get("snapshot_path")
        # Bind every newly constructed frozen/nonbaseline manifest to the
        # exact receipt content digest.  ``candidate_manifest`` intentionally
        # does not know local receipt state, so carry it across explicitly.
        expected["snapshot_digest"] = receipt.get("snapshot_metadata_digest")
        return expected

    @staticmethod
    def _authority_digest(manifest: dict[str, Any] | None) -> str:
        return hashlib.sha256(json.dumps(manifest if isinstance(manifest,dict) else {"none":True},sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()

    def _human_authority_token(self) -> dict[str, Any]:
        """In-memory full-manifest snapshot; never durable human authority."""
        manifest=self.adaptive.champion_manifest()
        value=dict(manifest) if isinstance(manifest,dict) else None
        return {"manifest":value,"digest":self._authority_digest(value)}

    def _validate_human_authority(self, token: dict[str, Any] | None) -> bool:
        try:
            if not isinstance(token,dict): return False
            current=self.adaptive.champion_manifest(); current=dict(current) if isinstance(current,dict) else None
            if token.get("digest") != self._authority_digest(current) or token.get("manifest") != current: return False
            if current is None: return True
            if current.get("stable_id") == WHISPER_ID:
                receipt=self._receipt_manifest(WHISPER_ID)
                return receipt is not None and self._authority_digest(receipt) == self._authority_digest(current)
            return self._validated_persisted_manifest(current,require_snapshot_digest=True) is not None
        except Exception:
            return False

    def _transcribe_current_human_or_baseline(self, samples, *, canonical_path=None, canonical_identity=None):
        token=self._human_authority_token(); chosen=token["manifest"]
        manifest=(self._validated_persisted_manifest(chosen,require_snapshot_digest=True)
                  if chosen and chosen.get("stable_id") != WHISPER_ID else self._receipt_manifest(WHISPER_ID))
        if manifest is None: raise RuntimeError("adaptive baseline preflight receipt is invalid")
        text,meta=self.backends.transcribe(manifest,samples,canonical_path=canonical_path,canonical_identity=canonical_identity)
        if not self._validate_human_authority(token): raise RuntimeError("human authority changed during inference")
        if self._validated_persisted_manifest(manifest,require_snapshot_digest=True) is None:
            raise RuntimeError("inference receipt changed")
        metadata=dict(meta) if isinstance(meta,dict) else {}
        metadata["authority_fallback"]=True
        return text,metadata

    def _authority_safe_live_fallback(self, samples, *, canonical_path, canonical_identity,
                                      live_audio_token, route_token, session_id: str,
                                      enqueue_failed: bool = False,
                                      initial_human_token: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        """Return only a freshly authorized human/baseline result.

        A comparator enqueue failure is not permission to return a baseline
        result transcribed while the capture, receipt, or human authority was
        changing.  The underlying helper snapshots the *current* human
        transport around inference; this wrapper additionally binds the
        canonical capture and reports a stale routed token without returning
        its candidate output.
        """
        last: Exception | None = None
        for _attempt in range(2):
            if live_audio_token is not None and not self._same_live_audio_authority(
                    live_audio_token,samples,canonical_path,canonical_identity):
                last=RuntimeError("live audio authority changed")
                continue
            try:
                text,metadata=self._transcribe_current_human_or_baseline(
                    samples,canonical_path=canonical_path,canonical_identity=canonical_identity)
            except Exception as exc:
                last=exc
                continue
            if live_audio_token is not None and not self._same_live_audio_authority(
                    live_audio_token,samples,canonical_path,canonical_identity):
                last=RuntimeError("live audio authority changed")
                continue
            # A route token only authorizes the discarded candidate, never a
            # fallback.  Still recheck it so callers cannot mistake this
            # result for a valid canary observation.
            metadata=dict(metadata) if isinstance(metadata,dict) else {}
            metadata.update({"fallback":True,"session_id":session_id})
            if enqueue_failed: metadata["comparator_enqueue_failed"]=True
            if initial_human_token is not None and not self._validate_human_authority(initial_human_token):
                metadata["human_authority_changed"]=True
            if route_token is not None and not self.deployment.validate_route_token(route_token):
                metadata["route_token_stale"]=True
            return text,metadata
        raise RuntimeError("live fallback authority changed") from last

    def _validated_persisted_manifest(self, manifest: dict[str, Any], *, require_snapshot_digest: bool = False) -> dict[str, Any] | None:
        """Validate a frozen manifest without reconstructing private glossary terms."""
        stable_id = manifest.get("stable_id")
        if not isinstance(stable_id, str): return None
        receipt = self.receipts.load(stable_id)
        if not receipt or not self.receipts.valid_manifest(manifest): return None
        frozen_digest=manifest.get("snapshot_digest")
        current_digest=receipt.get("snapshot_metadata_digest")
        # Legacy human manifests predate the digest field; they remain valid
        # only on this compatibility path.  Every new silver horizon persists
        # it and must match the currently validated receipt exactly.
        if (require_snapshot_digest and frozen_digest is None) or (frozen_digest is not None and (not isinstance(frozen_digest,str) or frozen_digest != current_digest)):
            return None
        terms = manifest.get("glossary_terms", ())
        if manifest.get("stable_id") == WHISPER_GLOSSARY_ID:
            if (not isinstance(terms, (tuple, list)) or not terms
                    or not all(isinstance(term, str) for term in terms)): return None
            import hashlib
            identity = hashlib.sha256("\n".join(terms).encode("utf-8")).hexdigest()
            if identity != manifest.get("glossary_hash") or identity != manifest.get("glossary_identity"): return None
        elif terms:
            return None
        result = dict(manifest); result["snapshot_path"] = receipt.get("snapshot_path")
        return result

    def warmup_baseline(self, samples) -> None:
        """Warm only the receipt-backed Whisper transport path.

        Startup warmup must never look at routing, sessions, AdaptiveLearning,
        or the pair ledger: synthetic silence is not an observation.  The
        ordinary live scheduler still serializes this with foreground work.
        """
        manifest = self._receipt_manifest(WHISPER_ID)
        if manifest is None:
            raise RuntimeError("adaptive baseline preflight receipt is invalid")
        with self.scheduler.live_request():
            self.backends.transcribe(manifest, samples)

    def _validated_deployment(self) -> dict[str, Any]:
        """Walk current/prior deployment state fail-closed to a valid receipt."""
        for _ in range(4):
            manifest = self.adaptive.champion_manifest()
            if manifest is None: break
            validated = self._validated_persisted_manifest(manifest,require_snapshot_digest=manifest.get("stable_id") != WHISPER_ID)
            if validated is not None: return validated
            if manifest.get("stable_id") == WHISPER_ID: break
            self.adaptive.invalidate_current_if(manifest, reason="receipt_invalid")
        baseline = self._receipt_manifest(WHISPER_ID)
        if baseline is None: raise RuntimeError("adaptive baseline preflight receipt is invalid")
        return baseline

    def reconcile(self, *, retry_pending: bool = True) -> dict[str, Any]:
        """Advance collection state without ever running a model."""
        try:
            self.recover_comparator_publications()
        except Exception as exc:
            return {"state":"comparator_publication_recovery_blocked","reason":type(exc).__name__}
        # No generation state is observed or advanced until every durable
        # cross-store human review has converged.  This is intentionally before
        # baseline publication and legacy capture registration at startup.
        try:
            self._review_recovery_gate(retry_pending=retry_pending)
        except Exception as exc:
            return {"state": "review_recovery_blocked", "reason": type(exc).__name__}
        self.adaptive.recover_partial_publications()
        baseline = self._receipt_manifest(WHISPER_ID)
        if baseline:
            self.adaptive.ensure_baseline(baseline)
        active = self.adaptive.active_generation()
        if active:
            if active["status"] in {"pool_closed", "evaluating"}:
                # The headless worker owns evaluation recovery.  Never detach
                # a subprocess from a scheduler lease.
                return {"state": "evaluating", "generation": active["id"]}
            if active["status"] == "pool_open":
                pool = self.adaptive.pool_status(active["id"])
                if pool["selected"] == 33 and pool["resolved"] == 33:
                    if self.adaptive.close_pool(active["id"]):
                        return {"state": "evaluating", "generation": active["id"]}
            return {"state": active["status"], "generation": active["id"]}
        readiness = self.adaptive.development_ready()
        if not readiness["ready"]:
            return {"state": "collecting", **readiness}
        if not baseline:
            return {"state": "collecting", "reason": "baseline_preflight_missing", **readiness}
        try: incumbent = self._validated_deployment()
        except RuntimeError:
            return {"state": "collecting", "reason": "baseline_preflight_missing", **readiness}
        terms, dependencies = self.development_glossary()
        candidates: list[dict[str, Any]] = []
        parakeet = self._receipt_manifest(PARAKEET_ID)
        if parakeet: candidates.append(parakeet)
        if terms:
            glossary = self._receipt_manifest(WHISPER_GLOSSARY_ID, glossary_terms=terms)
            if glossary:
                glossary["development_dependencies"] = list(dependencies); candidates.append(glossary)
        def same_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
            keys = ("backend", "repo", "revision", "package_versions", "decode_settings", "language", "glossary_hash", "glossary_identity", "evaluator_id", "evaluator_hash")
            return all(left.get(key) == right.get(key) for key in keys)
        candidates = [candidate for candidate in candidates if not same_identity(candidate, incumbent)]
        if not candidates:
            return {"state": "collecting", "reason": "challenger_preflight_missing", **readiness}
        try:
            ident = self.adaptive.publish_generation(champion=incumbent, candidates=candidates,
                                                      evaluator_hash=evaluator_hash())
        except RuntimeError as exc:
            # A concurrent reconciler may have published the same generation,
            # or the one-look identity may have already been consumed.
            return {"state": "collecting", "reason": str(exc), **readiness}
        return {"state": "pool_open", "generation": ident}

    def live_transcribe(self, samples, *, duration: float | None = None,
                        canonical_path: Path | str | None = None, canonical_identity=None,
                        capture_id: str | None = None) -> tuple[str, dict[str, Any]]:
        """Route a durable session to silver only after valid deployment.

        A human-gold champion remains authoritative.  Any receipt/schema/
        identity concern uses the receipt-backed Whisper baseline instead.
        """
        session_id=self.silver.active_session_id()
        if capture_id is not None and (not isinstance(capture_id,str) or not capture_id):
            raise ValueError("capture_id must be a nonempty opaque string")
        capture_id=capture_id or uuid.uuid4().hex
        human_token=self._human_authority_token()
        human=self.adaptive.champion_manifest()
        manifest=None; force_baseline=False
        silver_selected=False; route_token=None
        if human and human.get("stable_id") != WHISPER_ID:
            manifest=self._validated_persisted_manifest(human,
                                                        require_snapshot_digest=True)
            # Human champion precedence is absolute.  If it is invalid, use
            # the trusted baseline; do not silently substitute silver.
            if manifest is None:
                manifest=self._receipt_manifest(WHISPER_ID)
                force_baseline=True
        # A human Whisper baseline is not a human promotion; it may still be
        # replaced by a valid silver canary.  Only an invalid nonbaseline
        # human champion forces the baseline.
        if manifest is None and not force_baseline:
            routed=self.deployment.route(session_id)
            if routed is not None:
                route_token=routed.pop("_deployment_token",None)
                manifest=self._validated_persisted_manifest(routed,require_snapshot_digest=True); silver_selected=manifest is not None
                if manifest is None:
                    self.deployment.invalidate_candidate(routed,"routed_receipt_invalid")
        if manifest is None:
            manifest=self._receipt_manifest(WHISPER_ID)
        if manifest is None: raise RuntimeError("adaptive baseline preflight receipt is invalid")
        if silver_selected and not self.deployment.validate_route_token(route_token):
            manifest=self._receipt_manifest(WHISPER_ID); silver_selected=False
            if manifest is None: raise RuntimeError("adaptive baseline preflight receipt is invalid")
        live_audio_token=None; audio_authority_fallback=False
        if silver_selected:
            live_audio_token=self._live_audio_authority(samples,canonical_path,canonical_identity)
            if live_audio_token is None:
                # The candidate never observes an unbound/mutable capture.
                manifest=self._receipt_manifest(WHISPER_ID); silver_selected=False; audio_authority_fallback=True
                if manifest is None: raise RuntimeError("adaptive baseline preflight receipt is invalid")
        with self.scheduler.live_request():
            started=time.monotonic()
            try:
                text, metadata = self.backends.transcribe(manifest, samples, canonical_path=canonical_path,
                                                          canonical_identity=canonical_identity)
                candidate_elapsed=time.monotonic()-started
                human_or_receipt_changed=(not self._validate_human_authority(human_token) or
                                          self._validated_persisted_manifest(manifest,require_snapshot_digest=True) is None)
                audio_changed=(silver_selected and not self._same_live_audio_authority(
                    live_audio_token,samples,canonical_path,canonical_identity))
                if human_or_receipt_changed or audio_changed:
                    if silver_selected:
                        text,metadata=self._authority_safe_live_fallback(
                            samples,canonical_path=canonical_path,canonical_identity=canonical_identity,
                            live_audio_token=live_audio_token,route_token=route_token,session_id=session_id,
                            initial_human_token=human_token)
                        if human_or_receipt_changed: metadata["human_authority_changed"]=True
                        if audio_changed: metadata["live_audio_authority_changed"]=True
                        return text,metadata
                    text,metadata=self._transcribe_current_human_or_baseline(samples,canonical_path=canonical_path,canonical_identity=canonical_identity)
                    if human_or_receipt_changed: metadata["human_authority_changed"]=True
                    if audio_changed: metadata["live_audio_authority_changed"]=True
                    metadata["fallback"]=True; metadata["session_id"]=session_id
                    return text,metadata
            except Exception as exc:
                candidate_elapsed=time.monotonic()-started
                result = self.adaptive.record_runtime_failure(manifest["stable_id"], type(exc).__name__, expected_manifest=manifest)
                candidate_evidence={"success":False,"fallback":True,"latency":candidate_elapsed,"coverage_ok":False,"hallucination_ok":False,"identity_valid":not isinstance(exc,ValueError)}
                if silver_selected:
                    fallback_started=time.monotonic()
                    text,metadata=self._authority_safe_live_fallback(
                        samples,canonical_path=canonical_path,canonical_identity=canonical_identity,
                        live_audio_token=live_audio_token,route_token=route_token,session_id=session_id,
                        initial_human_token=human_token)
                    metadata["runtime_failure_streak"]=result["streak"]
                    # A failure pair is authority only when the exact
                    # same-audio fallback remained current through its return.
                    if (not metadata.get("human_authority_changed") and not metadata.get("live_audio_authority_changed") and
                            not metadata.get("route_token_stale") and self.deployment.validate_route_token(route_token) and
                            self._same_live_audio_authority(live_audio_token,samples,canonical_path,canonical_identity)):
                        begun=self.deployment.begin_runtime_pair(session_id=session_id,capture_id=capture_id,candidate_arm=manifest["stable_id"],candidate=candidate_evidence,route_token=route_token)
                        if begun in {"inserted","replayed"}:
                            coverage,hallucination=self._runtime_safety(text,metadata,duration,samples)
                            pair=self.deployment.complete_runtime_pair(session_id=session_id,capture_id=capture_id,candidate_arm=manifest["stable_id"],route_token=route_token,candidate=candidate_evidence,
                                incumbent={"success":True,"fallback":False,"latency":time.monotonic()-fallback_started,"coverage_ok":coverage,"hallucination_ok":hallucination,"identity_valid":True})
                            metadata["route_token_stale"]=pair in {"stale","rolled_back"}
                    return text,metadata
                begun=(self.deployment.begin_runtime_pair(session_id=session_id,capture_id=capture_id,candidate_arm=manifest["stable_id"],candidate=candidate_evidence,route_token=route_token)
                       if silver_selected else "stale")
                baseline = self._receipt_manifest(WHISPER_ID)
                baseline_error=None; baseline_started=time.monotonic(); text=""; metadata={}
                try:
                    if baseline is None: raise RuntimeError("baseline receipt unavailable")
                    text, metadata = self.backends.transcribe(baseline, samples, canonical_path=canonical_path,
                                                              canonical_identity=canonical_identity)
                except Exception as fallback_exc:
                    baseline_error=fallback_exc
                metadata["fallback"] = True; metadata["runtime_failure_streak"] = result["streak"]
                if begun in {"inserted","replayed"}:
                    coverage,hallucination=self._runtime_safety(text,metadata,duration,samples) if baseline_error is None else (False,False)
                    pair_result=self.deployment.complete_runtime_pair(session_id=session_id,capture_id=capture_id,candidate_arm=manifest["stable_id"],route_token=route_token,candidate=candidate_evidence,
                        incumbent={"success":baseline_error is None,"fallback":baseline_error is not None,"latency":time.monotonic()-baseline_started,"coverage_ok":coverage,"hallucination_ok":hallucination,"identity_valid":baseline is not None and not isinstance(baseline_error,ValueError)})
                    metadata["route_token_stale"] = pair_result in {"stale","rolled_back"}
                if baseline_error is not None: raise baseline_error
                if (not self._validate_human_authority(human_token) or
                        self._validated_persisted_manifest(baseline,require_snapshot_digest=True) is None):
                    text,metadata=self._transcribe_current_human_or_baseline(samples,canonical_path=canonical_path,canonical_identity=canonical_identity)
                    metadata["human_authority_changed"]=True; metadata["fallback"]=True; metadata["session_id"]=session_id
                    return text,metadata
                metadata["session_id"]=session_id
                return text, metadata
            if silver_selected and (not self.deployment.validate_route_token(route_token) or
                                    self._validated_persisted_manifest(manifest,require_snapshot_digest=True) is None):
                return self._authority_safe_live_fallback(
                    samples,canonical_path=canonical_path,canonical_identity=canonical_identity,
                    live_audio_token=live_audio_token,route_token=route_token,session_id=session_id,
                    initial_human_token=human_token)
            human_changed=silver_selected and not self._validate_human_authority(human_token)
            receipt_changed=silver_selected and self._validated_persisted_manifest(manifest,require_snapshot_digest=True) is None
            audio_changed=silver_selected and not self._same_live_audio_authority(live_audio_token,samples,canonical_path,canonical_identity)
            if human_changed or receipt_changed or audio_changed:
                if silver_selected:
                    text,metadata=self._authority_safe_live_fallback(
                        samples,canonical_path=canonical_path,canonical_identity=canonical_identity,
                        live_audio_token=live_audio_token,route_token=route_token,session_id=session_id,
                        initial_human_token=human_token)
                    if human_changed or receipt_changed: metadata["human_authority_changed"]=True
                    if audio_changed: metadata["live_audio_authority_changed"]=True
                    return text,metadata
                text,metadata=self._transcribe_current_human_or_baseline(samples,canonical_path=canonical_path,canonical_identity=canonical_identity)
                metadata.update({"fallback":True,"session_id":session_id})
                if human_changed or receipt_changed: metadata["human_authority_changed"]=True
                if audio_changed: metadata["live_audio_authority_changed"]=True
                return text,metadata
            self.adaptive.record_runtime_success(manifest["stable_id"], expected_manifest=manifest)
            if silver_selected:
                coverage,hallucination=self._runtime_safety(text,metadata,duration,samples)
                candidate_evidence={"success":True,"fallback":False,"latency":candidate_elapsed,"coverage_ok":coverage,"hallucination_ok":hallucination,"identity_valid":True}
                if (not self._validate_human_authority(human_token) or
                        self._validated_persisted_manifest(manifest,require_snapshot_digest=True) is None or
                        not self._same_live_audio_authority(live_audio_token,samples,canonical_path,canonical_identity)):
                    text,metadata=self._authority_safe_live_fallback(
                        samples,canonical_path=canonical_path,canonical_identity=canonical_identity,
                        live_audio_token=live_audio_token,route_token=route_token,session_id=session_id,
                        initial_human_token=human_token)
                    metadata["live_audio_authority_changed"]=True
                    return text,metadata
                wav_bytes,audio_sha,sample_count=live_audio_token
                deployment_revision=int(route_token["deployment_revision"])
                comparator_id=self._comparator_capture_id(session_id,audio_sha,sample_count,deployment_revision,capture_id)
                spool=None
                durable_intent=False
                enqueue_exception=False
                try:
                    spool,_=self._write_comparator_spool(wav_bytes,comparator_id,audio_sha)
                    intent=self.deployment.enqueue_comparator_intent(session_id=session_id,capture_id=comparator_id,
                        candidate_arm=manifest["stable_id"],candidate=candidate_evidence,audio_sha256=audio_sha,
                        spool_name=spool.name,route_token=route_token)
                    durable_intent=intent in {"inserted","replayed"}
                except Exception:
                    enqueue_exception=True
                    intent="stale"
                if enqueue_exception:
                    # A post-commit acknowledgement loss returns fallback to
                    # the user.  Its candidate-success row must therefore be
                    # terminal before that fallback is exposed; otherwise a
                    # later worker could count an undelivered candidate toward
                    # rollout.  A normal duplicate returns ``False`` rather
                    # than raising and follows the preserving branch below.
                    try:
                        terminal=(self.silver.discard_comparator_intent(capture_id=comparator_id,
                            deployment_revision=deployment_revision,spool_name=spool.name,
                            audio_sha256=audio_sha,candidate=candidate_evidence) if spool is not None else False)
                    except Exception as exc:
                        raise RuntimeError("comparator intent recovery failed") from exc
                    if terminal:
                        self._delete_comparator_spool(comparator_id,audio_sha)
                    elif spool is not None:
                        # Conflict/uncertainty is not this invocation's row.
                        # Preserve its bytes rather than deleting another
                        # foreground call's live outbox authority.
                        self._cleanup_non_durable_comparator_spool(deployment_revision=deployment_revision,
                            capture_id=comparator_id,spool=spool,audio_sha256=audio_sha)
                    return self._authority_safe_live_fallback(
                        samples,canonical_path=canonical_path,canonical_identity=canonical_identity,
                        live_audio_token=live_audio_token,route_token=route_token,session_id=session_id,
                        enqueue_failed=True,initial_human_token=human_token)
                if durable_intent:
                    # Enqueue is durable, but it is not the foreground return
                    # authority.  Recheck every mutable authority immediately
                    # before exposing the candidate.
                    try:
                        human_changed=not self._validate_human_authority(human_token)
                        receipt_changed=self._validated_persisted_manifest(manifest,require_snapshot_digest=True) is None
                        audio_changed=not self._same_live_audio_authority(live_audio_token,samples,canonical_path,canonical_identity)
                        route_changed=not self.deployment.validate_route_token(route_token)
                        if not (human_changed or receipt_changed or audio_changed or route_changed):
                            metadata["comparator_pending"]=True; metadata["session_id"]=session_id
                            metadata["comparator_publication"]={"deployment_revision":deployment_revision,
                                "capture_id":comparator_id,"spool_name":spool.name,
                                "audio_sha256":audio_sha,"candidate":candidate_evidence,
                                "cohort_id":route_token["cohort_id"],"epoch":route_token["epoch"],
                                "route_generation":route_token["route_generation"],
                                "candidate_arm":manifest["stable_id"],"candidate_hash":route_token["candidate_hash"]}
                            return text,metadata
                    except Exception:
                        human_changed=receipt_changed=audio_changed=route_changed=True
                    # A durable outbox item must be terminal before removing
                    # the bytes it authorizes.  If another owner raced us, do
                    # not erase its spool or return a potentially stale text.
                    if not self.silver.discard_comparator_intent(capture_id=comparator_id,deployment_revision=deployment_revision):
                        raise RuntimeError("comparator intent terminalization failed")
                    self._delete_comparator_spool(comparator_id,audio_sha)
                    text,metadata=self._authority_safe_live_fallback(
                        samples,canonical_path=canonical_path,canonical_identity=canonical_identity,
                        live_audio_token=live_audio_token,route_token=route_token,session_id=session_id,
                        initial_human_token=human_token)
                    if human_changed or receipt_changed: metadata["human_authority_changed"]=True
                    if audio_changed: metadata["live_audio_authority_changed"]=True
                    return text,metadata
                # A rejected replay may refer to a durable pending comparator
                # with the same exact spool.  Delete only when the store proves
                # that no live exact owner remains; uncertainty is fail-closed.
                if spool is not None:
                    self._cleanup_non_durable_comparator_spool(deployment_revision=deployment_revision,
                        capture_id=comparator_id,spool=spool,audio_sha256=audio_sha)
                return self._authority_safe_live_fallback(
                    samples,canonical_path=canonical_path,canonical_identity=canonical_identity,
                    live_audio_token=live_audio_token,route_token=route_token,session_id=session_id,
                    enqueue_failed=True,initial_human_token=human_token)
            if (not self._validate_human_authority(human_token) or
                    self._validated_persisted_manifest(manifest,require_snapshot_digest=True) is None):
                if silver_selected:
                    self.deployment.invalidate_candidate(manifest,"post_inference_authority_changed")
                text,metadata=self._transcribe_current_human_or_baseline(samples,canonical_path=canonical_path,canonical_identity=canonical_identity)
                metadata["human_authority_changed"]=True; metadata["fallback"]=True; metadata["session_id"]=session_id
                return text,metadata
            if audio_authority_fallback:
                metadata["fallback"]=True; metadata["live_audio_authority_changed"]=True
            metadata["session_id"]=session_id
            return text, metadata

    @staticmethod
    def _runtime_safety(text: str, metadata: dict[str, Any] | Any, duration: float | None, samples: Any = None) -> tuple[bool, bool]:
        """Use explicit app outcomes when present; otherwise content metrics.

        Local backend metadata intentionally does not invent application safety
        fields.  A bounded token rate and repetition check are genuine output
        evidence, unlike ``bool(text)``.
        """
        words=tokenize(text if isinstance(text,str) else "")
        if duration is None:
            identity=getattr(samples,"identity",None)
            count=getattr(identity,"sample_count",None)
            rate=getattr(identity,"sample_rate",None)
            if not isinstance(count,int):
                values=getattr(samples,"asr_samples",samples)
                try: count=len(values)
                except TypeError: count=0
            if not isinstance(rate,int) or rate <= 0: rate=16000
            duration=float(count)/rate
        seconds=max(float(duration or 0), 0.25)
        coverage=bool(words) and len(words)/seconds <= 12.0
        repeated=max((words.count(word) for word in set(words)), default=0)
        hallucination=coverage and repeated <= max(3, len(words)//2)
        if isinstance(metadata,dict):
            if isinstance(metadata.get("coverage_ok"),bool): coverage=metadata["coverage_ok"]
            if isinstance(metadata.get("hallucination_ok"),bool): hallucination=metadata["hallucination_ok"]
        return coverage,hallucination

    def evaluate(self) -> dict[str, Any]:
        # CLI/direct evaluation must use the same zero-pending gate as normal
        # reconciliation; callers cannot bypass it by invoking evaluate alone.
        self._review_recovery_gate(retry_pending=True)
        lock_path = self.history.base_dir / "adaptive-learning" / ".evaluator-process.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc: raise RuntimeError("adaptive evaluator is already running") from exc
            active = self.adaptive.active_generation()
            if not active or active["status"] not in {"pool_closed", "evaluating"}:
                raise RuntimeError("no closed adaptive generation to evaluate")
            # Freeze-time manifests must still have exact, verified receipts;
            # abort before reading audio or importing a model when they do not.
            frozen = self.adaptive.frozen_manifests(active["id"])
            normalized = {item["stable_id"]: self._validated_persisted_manifest(item,require_snapshot_digest=item.get("stable_id") != WHISPER_ID) for item in frozen.get("all", [])}
            if (frozen.get("evaluator_hash") != evaluator_hash() or any(item is None for item in normalized.values())):
                return self.adaptive.technical_abort(active["id"], "receipt_or_evaluator_invalid")
            reclaimed = self.scheduler.reconcile()
            self.adaptive.recover_interrupted_attempts(reclaimed)
            def callback(manifest: dict[str, Any], capture: dict[str, Any]):
                while True:
                    with self.scheduler.evaluator_lease() as granted:
                        if granted:
                            audio_path=Path(capture["audio_path"])
                            samples, identity = read_canonical_wav(audio_path)
                            expected = capture.get("audio_identity")
                            if (identity.sha256 != capture.get("audio_digest") or not isinstance(expected, dict)
                                    or identity.as_dict() != {key: expected.get(key) for key in identity.as_dict()}):
                                raise RuntimeError("reviewed_audio_identity_mismatch")
                            # Freeze the receipt-resolved transport object, not
                            # just the stable alias, across the model call.
                            require_digest=manifest.get("stable_id") != WHISPER_ID
                            transport=dict(normalized[manifest["stable_id"]])
                            semantic_hash=self.silver.manifest_hash(transport)
                            snapshot_digest=transport.get("snapshot_digest")
                            pre_audio_identity=identity.as_dict()
                            started = time.monotonic()
                            if isinstance(self.backends, LocalBackendManager):
                                text, _metadata = self.backends.transcribe(
                                    transport, samples, canonical_path=audio_path, canonical_identity=identity)
                            else:
                                # Injectable test/evaluation backends receive
                                # the already identity-validated canonical decode.
                                text, _metadata = self.backends.transcribe(transport, samples)
                            try:
                                _post_samples, post_identity=read_canonical_wav(audio_path)
                            except (OSError, ValueError, EOFError) as exc:
                                raise RuntimeError("human_evaluation_authority_changed") from exc
                            refreshed=self._validated_persisted_manifest(
                                manifest, require_snapshot_digest=require_digest)
                            if (post_identity.as_dict() != pre_audio_identity or
                                    post_identity.sha256 != capture.get("audio_digest") or refreshed is None or
                                    self.silver.manifest_hash(refreshed) != semantic_hash or
                                    refreshed.get("snapshot_digest") != snapshot_digest or refreshed != transport):
                                raise RuntimeError("human_evaluation_authority_changed")
                            return text, time.monotonic() - started
                    time.sleep(.05)
            return self.adaptive.evaluate(active["id"], callback, evaluator_owner=self.scheduler.owner_id)
        finally:
            try: fcntl.flock(fd, fcntl.LOCK_UN)
            finally: os.close(fd)

    def status(self) -> dict[str, Any]:
        state = self.adaptive.status(); state["development"] = self.adaptive.development_ready()
        state["active_generation"] = self.adaptive.active_generation()
        state["preflight"] = {ident: bool((receipt := self.receipts.load(ident)) and isinstance(receipt.get("manifest"), dict)
                                      and self.receipts.valid_manifest(dict(receipt["manifest"])))
                             for ident in (WHISPER_ID, WHISPER_GLOSSARY_ID, PARAKEET_ID)}
        return state

    def preflight(self) -> list[dict[str, Any]]:
        """The only deliberately model-loading admin operation."""
        warm = None
        for entry in self.history.entries(limit=10_000):
            capture = self._capture_from_history(entry, adaptive=False)
            if capture:
                try: warm, _ = read_canonical_wav(Path(capture["audio_path"])); break
                except (OSError, ValueError): pass
        synthetic = warm is None
        if synthetic:
            import numpy as np
            warm = np.zeros(16_000 // 2, dtype=np.float32)
        terms, _ = self.development_glossary()
        ids = [WHISPER_ID, PARAKEET_ID] + ([WHISPER_GLOSSARY_ID] if terms else [])
        results = []
        for ident in ids:
            receipt = self.backends.preflight(ident, warm, hf_home=os.environ.get("SOTTO_HF_HOME"), glossary_terms=terms)
            receipt["warmup"] = "synthetic_non_speech" if synthetic else "retained_history_audio"
            self.receipts.save(ident, receipt); results.append({"stable_id": ident, "backend": receipt["backend"], "success": True})
        self.reconcile()
        return results

    def rollback(self) -> dict[str, Any]:
        return self.adaptive.rollback(reason="manual")

    def spawn_evaluator(self, script: Path) -> None:
        """Retired compatibility seam: evaluation is worker-owned and synchronous."""
        return None

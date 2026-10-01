"""Headless, supervised autonomous-silver worker.

This module deliberately imports neither Sotto's UI nor any ML package.  It
can be run once by launchd or retained as a small service; expensive model work
is performed without HistoryStore, SilverStore, adaptive, or scheduler locks.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import signal
import sqlite3
import stat
import threading
import time
import wave
from pathlib import Path
from typing import Any

from audio_codec import PreparedAudio, decode_canonical, read_canonical_wav
from comparator_spool import ComparatorSpool
from evaluation import evaluate_sample, tokenize
from history import HistoryStore
from learning import LearningCoordinator
from inference_scheduler import InferenceScheduler
from silver_experiment import SilverExperiment
from silver_store import SilverStore
from deployment_controller import DeploymentController
from speech_backends import (LocalBackendManager, PARAKEET_ID, PreflightReceipts,
                             WHISPER_GLOSSARY_ID, WHISPER_ID, validate_preflight_receipt_read_only)
from teacher_backends import (OfflineTeacherRunner, REQUIRED_TEACHERS, TeacherPreempted, family_hash,
                              load_receipts, receipt_identity, validate_disjoint_lineage,
                              validate_teacher_receipts_read_only)
from teacher_consensus import exact_unanimous, policy_hash
from storage_lock import advisory_lock


class AdaptiveWorker:
    DRAIN_TIMEOUT = 1.0
    def __init__(self, base_dir: Path | str, *, history: HistoryStore | None = None,
                 silver: SilverStore | None = None, scheduler: InferenceScheduler | None = None,
                 teachers: Any | None = None, receipts: dict[str, dict[str, Any]] | None = None,
                 backends: Any | None = None,
                 sleep=time.sleep) -> None:
        self.base_dir = Path(base_dir)
        self.history = history or HistoryStore(self.base_dir)
        self.silver = silver or SilverStore(self.base_dir)
        self.scheduler = scheduler or InferenceScheduler(self.base_dir)
        self.teachers = teachers or OfflineTeacherRunner(receipts if receipts is not None else load_receipts(self.base_dir))
        self.backends = backends or LocalBackendManager(self.base_dir)
        self.preflight = PreflightReceipts(self.base_dir)
        self.comparator_spool = ComparatorSpool(self.base_dir)
        self.sleep = sleep
        self.stop = threading.Event()
        self._comparator_scrub_blocked = False

    def _receipt_hash(self) -> str:
        validator = getattr(self.teachers, "validate", None)
        if not callable(validator):
            raise RuntimeError("teacher receipt validator unavailable")
        return str(validator())

    def _scrub_comparator_artifacts(self, *, targets: dict[str,str] | None = None) -> bool:
        """Two-pass FD-relative Clear/terminal cleanup for owned comparator WAVs."""
        return self.comparator_spool.scrub(targets)

    def recover_markers(self) -> int:
        """Idempotently enqueue every durable history marker before inference."""
        count = 0
        self._comparator_scrub_blocked = False
        pending=self.silver.scrub_pending()
        if pending:
            # Cross-file artifacts contain only opaque cohort/deployment
            # identities, but must be scrubbed after an interrupted Clear.
            # A clear marker is an outbox fence: never scan surviving history
            # into the new epoch until the authoritative clear caller has
            # durably removed its rows/audio and acknowledged completion.
            if pending.get("kind") == "clear":
                # The initiating process may have died after committing the
                # SQLite epoch fence.  Finish the history/learning side before
                # acknowledging: scanning surviving rows here would create
                # fresh-epoch evidence from pre-clear captures.
                LearningCoordinator(self.history, base_dir=self.base_dir).finish_pending_clear()
            if pending.get("kind") == "clear":
                # Private comparator audio is not History evidence and must
                # disappear before the exact clear marker can be acknowledged.
                if not self._scrub_comparator_artifacts():
                    return 0
                DeploymentController(self.base_dir,silver=self.silver).clear_personal_state()
                SilverExperiment(self.base_dir,silver=self.silver).clear_personal_state()
            else:
                if pending.get("kind") == "revoke" and pending.get("schema") == 2:
                    # Schema-2 revoke batches are cross-store outbox work:
                    # finish each opaque target before touching mirrors or
                    # acknowledging the exact batch version.  A malformed or
                    # interrupted target remains pending for a later retry.
                    from silver_store import _valid_revoke_batch
                    if not _valid_revoke_batch(pending):
                        return 0
                    coordinator=LearningCoordinator(self.history,base_dir=self.base_dir)
                    try:
                        for intent in pending["intents"]:
                            coordinator.finish_pending_revoke_intent(intent)
                    except (RuntimeError, ValueError, OSError):
                        return 0
                cohorts={str(value) for value in pending.get("cohorts",[]) if isinstance(value,str)}
                DeploymentController(self.base_dir,silver=self.silver).invalidate_for_cohorts(cohorts,"revoke_scrub")
                SilverExperiment(self.base_dir,silver=self.silver).clear_cohorts(cohorts)
            if not self.silver.complete_scrub(pending):
                # A concurrent clear/revoke replaced the outbox while this
                # worker was scrubbing.  Do not scan; next recovery observes
                # and completes the newer operation.
                return 0
            if pending.get("kind") == "clear": return 0
        receipt = self._receipt_hash()
        # Lease expiry/exhaustion terminalizes outbox rows in SQLite first;
        # only then remove their exact owned private spool files.
        self.silver.recover()
        try:
            terminal_spools=self.silver.terminal_comparator_spools()
        except RuntimeError:
            self._comparator_scrub_blocked=True; self.silver.set_worker_status("scrub_blocked")
            return 0
        if not self._scrub_comparator_artifacts(targets=terminal_spools):
            self._comparator_scrub_blocked=True; self.silver.set_worker_status("scrub_blocked")
            return 0
        # The boundary is committed before this scan enqueues any teacher work.
        # Thus old, already-shadowed history can never become promotion data.
        try:
            self._ensure_horizon(receipt)
        except RuntimeError:
            # Teacher shadow labeling remains useful while local live receipts
            # are not preflighted.  It never retroactively creates a cohort.
            pass
        for snapshot in self.history.silver_markers():
            # Hold the shared history/advisory fence across revalidation,
            # enqueue and marker transition.  Clear independently verifies
            # the expected SQLite epoch, so a delayed worker can never add a
            # pre-clear capture to the new epoch.
            with advisory_lock(self.base_dir), self.history._lock:
                self.history._reload_locked()
                entry=self.history._find_locked(str(snapshot.get("id","")))
                if entry is None or entry.get("revision") != snapshot.get("revision"):
                    continue
                audio = entry.get("audio", {}).get("inference", {})
                marker = entry.get("silver_enqueue", {})
                if (entry.get("adaptive") is not True or entry.get("language") != "en" or
                        not isinstance(audio.get("sha256"), str) or marker.get("state") == "released"):
                    continue
                epoch=self.silver.epoch()
                key = self.silver.enqueue(history_id=entry["id"], history_revision=int(entry["revision"]),
                                          audio_sha256=audio["sha256"], teacher_family_hash=family_hash(),
                                          consensus_policy_hash=policy_hash(), receipt_hash=receipt,
                                          provenance_hash=hashlib.sha256((entry["id"] + audio["sha256"]).encode()).hexdigest(),
                                          expected_epoch=epoch)
                if key is None: continue
                marker["state"]="queued"; marker["job_key"]=key
                self.history._save_locked()
                count += 1
        self.silver.recover()
        try:
            terminal_spools=self.silver.terminal_comparator_spools()
        except RuntimeError:
            self._comparator_scrub_blocked=True; self.silver.set_worker_status("scrub_blocked")
            return count
        if not self._scrub_comparator_artifacts(targets=terminal_spools):
            self._comparator_scrub_blocked=True; self.silver.set_worker_status("scrub_blocked")
            return count
        self._reconcile_markers()
        return count

    def _reconcile_markers(self) -> int:
        """Close the crash gap after a label commit but before marker update."""
        changed=0
        for entry in self.history.silver_markers():
            marker=entry.get("silver_enqueue", {})
            if marker.get("state") == "released":
                continue
            # A pending marker has not yet been durably enqueued.  It is an
            # outbox promise, not evidence that audio may be pruned; a live
            # append interleaving with recovery must leave it for the next
            # scan rather than silently releasing it forever.
            if marker.get("state") == "pending":
                continue
            history_id=entry.get("id"); revision=entry.get("revision")
            if not isinstance(history_id,str) or not isinstance(revision,int):
                continue
            # Only an active prospective member needs audio retention.  A
            # discarded/revoked job, invalidated horizon, or terminal cohort
            # must converge to release on every restart.
            if not self.silver.should_hold_marker(history_id, revision):
                self.history.set_silver_marker(history_id, revision, "released", job_key=marker.get("job_key"))
                changed += 1
        return changed

    def _receipt_manifest(self, stable_id: str) -> dict[str, Any] | None:
        receipt=self.preflight.load(stable_id)
        if not receipt or not isinstance(receipt.get("manifest"),dict): return None
        manifest=dict(receipt["manifest"]); manifest["snapshot_path"]=receipt.get("snapshot_path")
        if isinstance(receipt.get("snapshot_metadata_digest"),str): manifest["snapshot_digest"]=receipt["snapshot_metadata_digest"]
        if manifest.get("language") != "en" or not self.preflight.valid_manifest(manifest): return None
        return manifest

    def _ensure_horizon(self, receipt_hash: str) -> dict[str, Any]:
        champion=self._receipt_manifest(WHISPER_ID)
        if champion is None: raise RuntimeError("receipt-backed Whisper champion unavailable")
        candidates=[]
        for stable_id in (PARAKEET_ID, WHISPER_GLOSSARY_ID):
            candidate=self._receipt_manifest(stable_id)
            # A glossary arm is usable only when its frozen private terms were
            # deliberately persisted in the receipt-generated manifest.
            if candidate is None:
                continue
            if stable_id == WHISPER_GLOSSARY_ID and not candidate.get("glossary_terms"):
                continue
            if candidate and validate_disjoint_lineage(candidate): candidates.append(candidate)
        if not candidates: raise RuntimeError("no disjoint receipt-backed silver challenger")
        return self.silver.ensure_prospective_horizon(start_ts=self.silver.clock(), champion=champion,
                                                      candidates=candidates, receipt_hash=receipt_hash)

    def _fresh_entry(self, job: dict[str, Any]) -> Path | None:
        """Revalidate the full history identity before every irreversible step."""
        if int(job["clear_epoch"]) != self.silver.epoch():
            return None
        entry = self.history.get(str(job["history_id"]))
        if not entry or int(entry.get("revision", -1)) != int(job["history_revision"]):
            return None
        audio = entry.get("audio", {}).get("inference", {})
        if audio.get("sha256") != job["audio_sha256"] or entry.get("correction") is not None:
            return None  # HUMAN_VERIFIED takes precedence over silver.
        marker = entry.get("silver_enqueue", {})
        if marker.get("history_revision") != int(job["history_revision"]):
            return None
        try:
            path = self.history.audio_path(str(job["history_id"]))
        except ValueError:
            return None
        try:
            _, actual = read_canonical_wav(path)
        except (OSError, ValueError, EOFError):
            return None
        return path if actual.sha256 == job["audio_sha256"] else None

    def _comparator_audio(self, intent: dict[str,Any]):
        try:
            name=intent.get('spool_name'); digest=intent.get('audio_sha256')
            if not isinstance(name,str) or not isinstance(digest,str): return None
            wav_bytes=self.comparator_spool.read(name,digest)
            with wave.open(io.BytesIO(wav_bytes),"rb") as handle:
                if (handle.getnchannels(),handle.getsampwidth(),handle.getframerate(),handle.getcomptype()) != (1,2,16_000,"NONE"):
                    return None
                pcm=handle.readframes(handle.getnframes())
            samples,identity=decode_canonical(pcm)
            if identity.sha256 != digest: return None
            samples.setflags(write=False)
            return name,wav_bytes,PreparedAudio(pcm,samples,identity)
        except (OSError,RuntimeError,ValueError,EOFError,wave.Error): return None

    def _transcribe_comparator(self, audio):
        name,wav_bytes,prepared=audio; identity=prepared.identity; baseline=self._receipt_manifest(WHISPER_ID)
        if baseline is None: return None
        started=time.monotonic(); text,metadata=self.backends.transcribe(baseline,prepared,canonical_identity=identity); latency=time.monotonic()-started
        try:
            if self.comparator_spool.read(name,identity.sha256) != wav_bytes: return None
        except (FileNotFoundError,OSError,RuntimeError,ValueError): return None
        post=self._receipt_manifest(WHISPER_ID)
        if post is None or self.silver.manifest_hash(post) != self.silver.manifest_hash(baseline) or post.get('snapshot_digest') != baseline.get('snapshot_digest'): return None
        from adaptive_runtime import AdaptiveRuntime
        coverage,hallucination=AdaptiveRuntime._runtime_safety(text,metadata,identity.sample_count/16_000,prepared.asr_samples)
        return {'success':True,'fallback':False,'latency':latency,'coverage_ok':coverage,'hallucination_ok':hallucination,'identity_valid':True}

    def _delete_comparator_audio(self, intent: dict[str,Any]) -> bool:
        try:
            name=intent.get('spool_name'); digest=intent.get('audio_sha256')
            if not isinstance(name,str) or not isinstance(digest,str): return False
            return self.comparator_spool.unlink(name,digest)
        except (FileNotFoundError,OSError,RuntimeError,ValueError):
            return False

    def process_comparator_one(self) -> bool:
        if self.stop.is_set(): return False
        intent=self.silver.claim_comparator_intent(self.scheduler.owner_id)
        if intent is None: return False
        audio=self._comparator_audio(intent)
        delete=lambda: self._delete_comparator_audio(intent)
        if audio is None:
            self.silver.finish_comparator_intent(intent,complete=False)
            if self.silver.comparator_intent_terminal(intent): delete()
            return True
        try:
            with self.scheduler.evaluator_lease() as granted:
                if not granted: self.silver.release_comparator_intent(intent,refund=True); return True
                evidence=self._transcribe_comparator(audio)
            if self.stop.is_set(): self.silver.release_comparator_intent(intent,refund=True); return True
            if evidence is None:
                self.silver.finish_comparator_intent(intent,complete=False)
                if self.silver.comparator_intent_terminal(intent): delete()
                return True
            result=DeploymentController(self.base_dir,silver=self.silver).finish_comparator_pair(intent,evidence)
            if result != 'stale':
                if self.silver.comparator_intent_terminal(intent): delete()
                return True
            self.silver.finish_comparator_intent(intent,complete=False)
            if self.silver.comparator_intent_terminal(intent): delete()
            return True
        except Exception:
            self.silver.release_comparator_intent(intent)
            if self.silver.comparator_intent_terminal(intent): delete()
            return True

    @staticmethod
    def _old_capture_shadow_statistics(entry: dict[str, Any], reference: str) -> dict[str, int] | None:
        """Compare only the immutable capture hypothesis, retaining no text."""
        hypothesis=entry.get("hypothesis")
        if not isinstance(hypothesis,str):
            return None
        metric=evaluate_sample(reference,hypothesis)
        return {"word_distance":metric.word_distance,"reference_words":metric.reference_words,
                "character_distance":metric.character_distance,"reference_characters":metric.reference_characters}

    def process_one(self) -> bool:
        if self.stop.is_set(): return False
        job = self.silver.claim(self.scheduler.owner_id)
        if job is None:
            return False
        path = self._fresh_entry(job)
        if path is None:
            self.silver.finish(job, outcome="discarded", code="stale_history")
            return True
        entry = self.history.get(str(job["history_id"])) or {}
        # Admission is deliberately before even the first teacher invocation.
        self.silver.admit_prospective_member(job, captured_ts=float(entry.get("ts", job.get("created_ts", 0))),
                                             session_id=(entry.get("candidate") or {}).get("session_id"))
        results: dict[str, str | None] = {}
        try:
            for teacher in REQUIRED_TEACHERS:
                if self.stop.is_set():
                    self.silver.finish(job, outcome="retry", code="live_preempted")
                    return True
                if not self.silver.renew(job):
                    return True
                # Lease exactly one teacher/sample; never hold persistent
                # storage locks while waiting or performing inference.
                with self.scheduler.evaluator_lease() as granted:
                    if not granted:
                        self.silver.finish(job, outcome="retry", code="scheduler_busy")
                        return True
                    path = self._fresh_entry(job)
                    if path is None:
                        self.silver.finish(job, outcome="discarded", code="stale_history")
                        return True
                    results[teacher.stable_id] = self.teachers.transcribe(
                        teacher, path, str(job["job_key"]), cancel=lambda: self.stop.is_set() or self.scheduler.live_pending())
        except TeacherPreempted:
            # Foreground work interrupts only this teacher sample.  The job
            # remains pending with its canonical evidence/marker intact and
            # the refunded attempt can be retried on a later worker pass.
            self.silver.finish(job, outcome="retry", code="live_preempted")
            return True
        except Exception as exc:
            # No teacher output or exception message may be emitted; only its type.
            self.silver.finish(job, outcome="retry", code=type(exc).__name__)
            return True
        if self._fresh_entry(job) is None or self._receipt_hash() != job["receipt_hash"]:
            self.silver.finish(job, outcome="discarded", code="stale_identity")
            return True
        decision = exact_unanimous(results, [x.stable_id for x in REQUIRED_TEACHERS])
        # One last freshness check is intentionally adjacent to label acceptance.
        if self._fresh_entry(job) is None:
            self.silver.finish(job, outcome="discarded", code="stale_history")
            return True
        shadow_audit=None
        if decision.state == "accepted" and isinstance(decision.reference,str):
            # `hypothesis` is the immutable capture-time transcription.  The
            # mutable display/correction text is intentionally never used for
            # this non-authority comparison or persisted in the audit table.
            shadow_audit=self._old_capture_shadow_statistics(entry,decision.reference)
        finished=self.silver.finish(job, outcome=decision.state, reference=decision.reference,
                                    vote_digest=decision.vote_digest, code=decision.reason,
                                    shadow_audit=shadow_audit,
                                    captured_ts=float(entry.get("ts",job.get("created_ts",0))))
        if not finished:
            return True
        # A prospective member keeps the canonical audio until every frozen
        # arm has a terminal attempt.  Nonmembers remain ordinary shadow work.
        held=self.silver.is_prospective_member(str(job["history_id"]),int(job["history_revision"]))
        self.history.set_silver_marker(str(job["history_id"]), int(job["history_revision"]), "held" if held else "released", job_key=str(job["job_key"]))
        return True

    def process_candidate_one(self) -> bool:
        if self.stop.is_set(): return False
        attempt=self.silver.claim_candidate_attempt(self.scheduler.owner_id)
        if attempt is None: return False
        horizon=self.silver.prospective_status()
        if not horizon or horizon.get("cohort_id") != attempt["cohort_id"] or horizon.get("clear_epoch") != self.silver.epoch():
            self.silver.finish_candidate_attempt(attempt, code="stale_horizon"); return True
        manifests={str(x.get("stable_id")):x for x in [horizon["champion"],*horizon["candidates"]]}
        manifest=manifests.get(attempt["arm_id"])
        entry=self.history.get(str(attempt["history_id"]))
        receipt_manifest=self._receipt_manifest(str(attempt["arm_id"]))
        if (not manifest or not entry or int(entry.get("revision",-1)) != int(attempt["history_revision"]) or
                not self.preflight.valid_manifest(manifest) or receipt_manifest is None or
                self.silver.manifest_hash(receipt_manifest) != self.silver.manifest_hash(manifest)):
            self.silver.finish_candidate_attempt(attempt, code="stale_history"); return True
        audio=entry.get("audio",{}).get("inference",{})
        try:
            path=self.history.audio_path(str(attempt["history_id"]))
        except ValueError:
            self.silver.finish_candidate_attempt(attempt, code="stale_history"); return True
        try:
            samples,identity=read_canonical_wav(path)
            if identity.sha256 != attempt["audio_sha256"] or audio.get("sha256") != identity.sha256:
                raise ValueError("audio_identity")
            reference=self.silver.candidate_reference(attempt)
            if reference is None: raise ValueError("reference_identity")
            with self.scheduler.evaluator_lease() as granted:
                if not granted:
                    self.silver.finish_candidate_attempt(attempt, code="scheduler_busy", retry=True); return True
                started=time.monotonic()
                # The frozen manifest establishes semantic identity, while
                # the receipt-resolved manifest supplies the currently
                # validated transport path.  Never execute a stale persisted
                # snapshot path after a same-revision receipt rotation.
                text,_=self.backends.transcribe(receipt_manifest,samples,canonical_path=path,canonical_identity=identity)
                latency=time.monotonic()-started
            if self.stop.is_set():
                self.silver.finish_candidate_attempt(attempt,code="live_preempted",retry=True); return True
            current_horizon=self.silver.prospective_status()
            current_entry=self.history.get(str(attempt["history_id"]))
            current_receipt=self._receipt_manifest(str(attempt["arm_id"]))
            try:
                _post_samples,post_identity=read_canonical_wav(path)
            except (OSError,ValueError,EOFError):
                post_identity=None
            if (self.silver.epoch()!=attempt["clear_epoch"] or not current_horizon or
                    current_horizon.get("cohort_id") != attempt["cohort_id"] or current_horizon.get("clear_epoch") != attempt["clear_epoch"] or
                    not current_entry or int(current_entry.get("revision",-1)) != int(attempt["history_revision"]) or
                    current_entry.get("audio",{}).get("inference",{}).get("sha256") != attempt["audio_sha256"] or
                    post_identity is None or post_identity.sha256 != attempt["audio_sha256"] or
                    current_receipt is None or not self.preflight.valid_manifest(manifest) or
                    self.silver.manifest_hash(current_receipt) != self.silver.manifest_hash(manifest) or
                    current_receipt.get("snapshot_digest") != manifest.get("snapshot_digest")):
                self.silver.finish_candidate_attempt(attempt,code="stale_identity"); return True
            metric=evaluate_sample(reference,text)
            # Predeclared content-free runaway-output proxy; the ordinary
            # nonempty-reference evaluator proxy cannot detect this class.
            hallucination=(not tokenize(text) or len(tokenize(text)) > max(12,4*len(tokenize(reference))))
            self.silver.finish_candidate_attempt(attempt,stats={"word_distance":metric.word_distance,"reference_words":metric.reference_words,
                "character_distance":metric.character_distance,"reference_characters":metric.reference_characters,
                "hallucination":hallucination,"latency":latency,"cluster_hash":attempt["cluster_hash"]})
        except Exception as exc:
            self.silver.finish_candidate_attempt(attempt,code=type(exc).__name__,retry=True)
        return True

    def reconcile_silver(self) -> dict[str, Any]:
        """Advance closed horizon, all synchronously; no detached evaluator."""
        if self.silver.scrub_pending() is not None:
            return {"state":"scrub_blocked"}
        cohort=self.silver.close_and_freeze_prospective()
        while not self.stop.is_set() and self.process_candidate_one():
            pass
        horizon=self.silver.prospective_status()
        target: tuple[str,str] | None = None
        if horizon and horizon.get("cohort_id"):
            target=(str(horizon["cohort_id"]),str(horizon["identity_hash"]))
        # An evaluation may have committed just before an authorization crash;
        # terminal cohorts remain independently discoverable even after a new
        # collecting horizon has opened on restart.
        # A passed terminal cohort is a restart-recovery fallback only while
        # deployment is still shadow.  A newly opened collecting horizon must
        # not hide an evaluation->authorization crash, but once a rollout is
        # active an older terminal must not mask its fresh post-canary pool.
        if target is None:
            try:
                controller=DeploymentController(self.base_dir,silver=self.silver)
                deployment_state=controller._read()
                deployment_tier=deployment_state.get("tier")
                if deployment_tier not in {"shadow_only","provisional_silver","canary","deployed_silver"}:
                    return {"state":"deployment_state_blocked"}
            except (OSError, RuntimeError, ValueError):
                # A corrupt deployment document is never evidence that an old
                # terminal may be republished.
                return {"state":"deployment_state_blocked"}
            # Finalize a published JSON rollout before deciding whether any
            # shadow terminal is replayable.  This is deliberately outside
            # the shadow branch: a crash after JSON publication but before
            # the SQLite phase transition is otherwise stranded forever.
            if deployment_tier in {"provisional_silver","canary","deployed_silver"}:
                for cohort_id in self.silver.terminal_cohort_ids():
                    if self.silver.authorization_disposition(cohort_id) == "authorizing":
                        if deployment_state.get("cohort_id") == cohort_id:
                            self.silver.mark_cohort_authorized(cohort_id)
                        else:
                            self.silver.consume_cohort_authorization(cohort_id)
            if deployment_tier == "shadow_only":
                for cohort_id in self.silver.terminal_cohort_ids():
                    cohort=self.silver.cohort_status(cohort_id) or {}
                    outcome=cohort.get("evaluation_outcome")
                    frozen=self.silver.horizon_for_cohort(cohort_id)
                    disposition=self.silver.authorization_disposition(cohort_id)
                    # ``authorizing`` is a two-store intent.  Only its exact
                    # claimed pre-publication controller state may be retried.
                    # A published state finalizes it; a changed/reset/malformed
                    # state is conservatively consumed rather than resurrected.
                    if disposition == "authorizing":
                        intent=self.silver.authorization_intent(cohort_id) or {}
                        if (int(deployment_state.get("revision",-1)) == intent.get("expected_revision") and
                              controller._state_digest(deployment_state) == intent.get("expected_state_digest")):
                            self.silver.release_cohort_authorization_claim(cohort_id)
                            disposition=None
                        else:
                            self.silver.consume_cohort_authorization(cohort_id)
                    if (disposition is None and cohort.get("status") == "evaluated_passed" and isinstance(outcome,dict) and
                            isinstance(outcome.get("winner"),str) and outcome.get("winner") and frozen):
                        target=(cohort_id,str(frozen["identity_hash"])); break
        if target is not None:
            cohort_id, identity_hash=target
            try:
                result=SilverExperiment(self.base_dir,silver=self.silver).evaluate_persisted(cohort_id,identity_hash=identity_hash) or {"state":"evaluating"}
            except RuntimeError as exc:
                # A corrupted/invalidated terminal cohort is not retried in a
                # tight loop; it remains fail-closed and its marker fence is
                # reconciled below.
                result={"state":"invalid", "reason":type(exc).__name__}
            if result.get("winner"):
                result["authorized"] = DeploymentController(self.base_dir, silver=self.silver).authorize_provisional(cohort_id)
            for history_id,revision in self.silver.releasable_cohort_members(cohort_id):
                self.history.set_silver_marker(history_id,revision,"released")
            self._reconcile_markers()
            return result
        return {"state":"collecting"}

    def run_once(self) -> dict[str, Any]:
        recovered = self.recover_markers()
        if self._comparator_scrub_blocked:
            return {"recovered": recovered, "processed": 0, "comparators_processed": 0,
                    "human": {"state":"scrub_blocked"}, "silver": {"state":"scrub_blocked"},
                    "status": self.silver.status()}
        if self.stop.is_set():
            return {"recovered": recovered, "processed": 0, "comparators_processed": 0, "human": {"state":"stopping"},
                    "silver": {"state":"stopping"}, "status": self.silver.status()}
        # Any unresolved (including malformed) scrub marker is an authority
        # fence.  Do not scan, claim, evaluate, or route around it.
        if self.silver.scrub_pending() is not None:
            self.silver.set_worker_status("scrub_blocked")
            return {"recovered": recovered, "processed": 0, "comparators_processed": 0, "human": {"state":"scrub_blocked"},
                    "silver": {"state":"scrub_blocked"}, "status": self.silver.status()}
        # A comparator is bounded to one claim per cycle.  A busy foreground
        # lease releases it back to pending, so looping here would only churn
        # the same intent instead of yielding to the next scheduled pass.
        comparators_processed = int(not self.stop.is_set() and self.process_comparator_one())
        processed = 0
        while not self.stop.is_set() and self.process_one():
            processed += 1
        if self.stop.is_set():
            return {"recovered": recovered, "processed": processed, "comparators_processed": comparators_processed,
                    "human": {"state":"stopping"}, "silver": {"state":"stopping"}, "status": self.silver.status()}
        # Preserve the established human lane, but reconcile it in-process
        # rather than starting a subprocess that cannot be leased/recovered.
        human_state: dict[str, Any]
        try:
            from adaptive_runtime import AdaptiveRuntime
            human=AdaptiveRuntime(self.base_dir)
            human_state=human.reconcile()
            if self.stop.is_set():
                return {"recovered": recovered, "processed": processed, "comparators_processed": comparators_processed,
                        "human": {"state":"stopping"}, "silver": {"state":"stopping"}, "status": self.silver.status()}
            if human_state.get("state") in {"review_recovery_blocked", "human_maintenance_blocked"}:
                self.silver.set_worker_status("human_maintenance_blocked", str(human_state.get("reason") or "unresolved"))
                return {"recovered": recovered, "processed": processed, "comparators_processed": comparators_processed, "human": human_state,
                        "silver": {"state":"blocked_by_human"}, "status": self.silver.status()}
            if human_state.get("state") == "evaluating":
                human.evaluate()
        except Exception as exc:
            # Do not advance silver while the human lane's durable work is
            # unresolved.  The status contains only an exception class.
            human_state={"state":"human_maintenance_blocked", "reason":type(exc).__name__}
            self.silver.set_worker_status("human_maintenance_blocked", type(exc).__name__)
            return {"recovered": recovered, "processed": processed, "comparators_processed": comparators_processed, "human": human_state,
                    "silver": {"state":"blocked_by_human"}, "status": self.silver.status()}
        if self.stop.is_set():
            return {"recovered": recovered, "processed": processed, "comparators_processed": comparators_processed,
                    "human": {"state":"stopping"}, "silver": {"state":"stopping"}, "status": self.silver.status()}
        silver_state=self.reconcile_silver()
        try:
            runtime_stage=DeploymentController(self.base_dir, silver=self.silver).reconcile_runtime()
            silver_state["runtime_stage"]=runtime_stage
        except Exception as exc:
            self.silver.set_worker_status("runtime_reconciliation_blocked",type(exc).__name__)
            return {"recovered": recovered, "processed": processed, "comparators_processed": comparators_processed, "human": human_state,
                    "silver": {"state":"runtime_reconciliation_blocked"}, "status": self.silver.status()}
        self.silver.set_worker_status("ok")
        return {"recovered": recovered, "processed": processed, "comparators_processed": comparators_processed, "human": human_state,
                "silver": silver_state, "status": self.silver.status()}

    def serve(self, interval: float = 30.0, drain_timeout: float = DRAIN_TIMEOUT) -> None:
        """Run one daemon-bounded cycle at a time; signal handlers only set stop."""
        while not self.stop.is_set():
            done=threading.Event()
            def cycle_body() -> None:
                try: self.run_once()
                finally: done.set()
            cycle=threading.Thread(target=cycle_body,daemon=True,name="sotto-worker-cycle")
            cycle.start()
            while not done.wait(.05):
                if self.stop.is_set():
                    cycle.join(max(0.0,drain_timeout))
                    return
            if self.stop.is_set(): return
            self.stop.wait(interval)


def _status_json(path: Path, *, jsonl: bool = False) -> dict[str, Any] | None:
    """Strict regular-file JSON reader for the operational, no-write path."""
    try:
        info=os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode): return None
        raw=path.read_text("utf-8")
        if jsonl:
            rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
            return rows[-1] if rows and all(isinstance(row,dict) for row in rows) else None
        value=json.loads(raw)
        return value if isinstance(value,dict) else None
    except (OSError,ValueError,json.JSONDecodeError): return None


def sealed_readiness_status(base_dir: Path | str) -> dict[str, Any]:
    """Content-free, nonconstructing status for shadow-only collection readiness."""
    base=Path(base_dir); blocked: list[str]=[]
    try:
        from runtime_source_manifest import runtime_source_digest
        source_digest=runtime_source_digest()
    except RuntimeError:
        source_digest=None; blocked.append("runtime_source")
    candidate_valid=[]
    for stable_id in (WHISPER_ID, PARAKEET_ID):
        receipt=validate_preflight_receipt_read_only(base,stable_id)
        candidate_valid.append(bool(receipt and receipt.get("runtime_source_digest") == source_digest))
    candidates={"required":2,"present":sum(candidate_valid),"current":all(candidate_valid)}
    if not candidates["current"]: blocked.append("candidate_receipts")
    teacher_rows=_status_json(base/"adaptive-learning"/"silver"/"teacher-receipts.json")
    try: teacher_hash=validate_teacher_receipts_read_only(teacher_rows) if isinstance(teacher_rows,dict) else None
    except (OSError,RuntimeError,TypeError,ValueError): teacher_hash=None
    teachers={"required":len(REQUIRED_TEACHERS),"present":len(REQUIRED_TEACHERS) if teacher_hash else 0,"lineage_valid":bool(teacher_hash)}
    if not teachers["lineage_valid"]: blocked.append("teacher_receipts")
    from silver_store import read_only_status
    silver=read_only_status(base)
    if silver.get("state") != "ok":
        blocked.append("silver_store")
        return {"state":"blocked","ready":False,"blocked":blocked,"runtime_source_digest":source_digest,
                "candidate_receipts":candidates,"teacher_receipts":teachers,"silver":silver}
    calibration={"state":"unknown","passed":False}; horizon={"state":"none","members":0}; cohorts={}
    db=None
    try:
        path=base/"adaptive-learning"/"silver"/"silver.sqlite3"; db=sqlite3.connect(f"file:{path}?mode=ro",uri=True); db.execute("PRAGMA query_only=ON")
        from calibration_manifest_builder import SOURCE_HASH, _protocol_hash, LEDGER_HASH
        from teacher_consensus import policy_hash as current_policy_hash
        from teacher_backends import family_hash as current_family_hash
        row=db.execute("SELECT holdout_key,source_hash,protocol_hash,policy_hash,receipt_hash,state,result FROM calibration_universes ORDER BY created_ts DESC LIMIT 1").fetchone()
        if row is not None:
            holdout,source_hash,protocol_hash,policy_hash,receipt_hash,state,result_raw=row
            expected_count=int(db.execute("SELECT COUNT(*) FROM calibration_expected_v2 WHERE holdout_key=?",(holdout,)).fetchone()[0])
            raw_count=int(db.execute("SELECT COUNT(*) FROM calibration_items_v2 WHERE holdout_key=?",(holdout,)).fetchone()[0])
            frozen=[tuple(item) for item in db.execute("SELECT ordinal,item_hash,cluster_hash FROM calibration_expected_v2 WHERE holdout_key=? ORDER BY ordinal",(holdout,))]
            ledger=hashlib.sha256(json.dumps(frozen,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
            rows=db.execute("SELECT e.ordinal,e.cluster_hash,i.state,i.outcome,i.reference_match FROM calibration_expected_v2 e LEFT JOIN calibration_items_v2 i ON i.holdout_key=e.holdout_key AND i.item_hash=e.item_hash WHERE e.holdout_key=? ORDER BY e.ordinal",(holdout,)).fetchall()
            terminal=(len(rows) == 500 and all(item[0] == index and item[2] == "terminal" and item[3] in {"accepted","abstained"} and ((item[3] == "accepted" and item[4] in {0,1}) or (item[3] == "abstained" and item[4] is None)) for index,item in enumerate(rows)))
            accepted=sum(item[3] == "accepted" for item in rows); clusters=len({item[1] for item in rows if item[3] == "accepted"}); errors=sum(item[3] == "accepted" and item[4] != 1 for item in rows)
            passed=(terminal and holdout == SOURCE_HASH and source_hash == SOURCE_HASH and protocol_hash == _protocol_hash() and policy_hash == current_policy_hash() and receipt_hash == teacher_hash and expected_count == raw_count == 500 and len({item[2] for item in frozen}) == 339 and ledger == LEDGER_HASH and [item[0] for item in frozen] == list(range(500)) and 5*accepted >= 2*500 and 10*accepted <= 9*500 and clusters >= 149 and errors == 0)
            try: result=json.loads(result_raw) if isinstance(result_raw,str) else None
            except (ValueError,json.JSONDecodeError): result=None
            expected_state="passed" if passed else "failed"
            passed=bool(passed and state == expected_state and isinstance(result,dict) and set(result) == {"state","accepted","clusters","reference_errors","total"} and result == {"state":expected_state,"accepted":accepted,"clusters":clusters,"reference_errors":errors,"total":500})
            calibration={"state":"passed" if passed else "invalid","passed":passed,"accepted":accepted,"clusters":clusters,"reference_errors":errors,"terminal":len(rows)}
        row=db.execute("SELECT horizon_id,status,frozen_cohort_id FROM prospective_horizons ORDER BY created_ts DESC LIMIT 1").fetchone()
        if row is not None: horizon={"state":str(row[1]),"members":int(db.execute("SELECT COUNT(*) FROM prospective_members WHERE horizon_id=?",(row[0],)).fetchone()[0]),"frozen":row[2] is not None}
        cohorts={str(state):int(count) for state,count in db.execute("SELECT status,COUNT(*) FROM cohorts GROUP BY status")}
    except (sqlite3.Error,TypeError,ValueError,json.JSONDecodeError): blocked.append("silver_metadata")
    finally:
        if db is not None: db.close()
    if not calibration.get("passed"): blocked.append("public_calibration")
    deployment=_status_json(base/"adaptive-learning"/"silver"/"deployment.jsonl",jsonl=True)
    if not isinstance(deployment,dict) or deployment.get("schema") != 1 or deployment.get("tier") not in {"shadow_only","provisional_silver","canary","deployed_silver"}:
        deployment_view={"state":"blocked"}; blocked.append("deployment")
    else:
        deployment_view={"state":"ok","tier":deployment["tier"],"revision":deployment.get("revision"),"route_generation":deployment.get("route_generation"),"current":isinstance(deployment.get("current"),dict),"lkg":isinstance(deployment.get("lkg"),dict),"canary":isinstance(deployment.get("canary"),dict)}
    worker=silver.get("worker") if isinstance(silver.get("worker"),dict) else {"state":"unknown"}
    if silver.get("scrub_pending"): blocked.append("scrub_pending")
    if deployment_view.get("tier") != "shadow_only": blocked.append("not_shadow_only")
    ready=not blocked
    return {"state":"ready" if ready else "blocked","ready":ready,"blocked":blocked,"runtime_source_digest":source_digest,
            "candidate_receipts":candidates,"teacher_receipts":teachers,"public_calibration":calibration,"prospective_horizon":horizon,"cohorts":cohorts,"deployment":deployment_view,"worker":{"state":worker.get("state","unknown"),"quiesce":"stopping" if worker.get("state") == "stopping" else "clear"},"scrub_pending":bool(silver.get("scrub_pending"))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sotto headless autonomous-silver worker")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--status", action="store_true", help="print strict content-free sealed readiness JSON")
    args = parser.parse_args(argv)
    if args.status:
        result=sealed_readiness_status(args.base_dir)
        print(json.dumps(result,sort_keys=True,separators=(",",":")))
        return 0 if result.get("ready") is True else 1
    worker = AdaptiveWorker(args.base_dir)
    signal.signal(signal.SIGTERM, lambda *_: worker.stop.set())
    signal.signal(signal.SIGINT, lambda *_: worker.stop.set())
    if args.serve:
        worker.serve()
    else:
        worker.run_once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

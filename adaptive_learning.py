"""Private, durable protocol state for measured Sotto model selection.

This module intentionally contains no model imports.  Integrations supply a
preflighted immutable candidate manifest and a transcription callback.  Admin
methods return ids/counts/statuses only, never reference or transcript content.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from evaluation import PAIRED_EVALUATOR_ID, PairedPromotionRow, holm_fixed_family, paired_bootstrap_statistics, tokenize
from history import _fsync_dir
from storage_lock import ensure_private_directory, ensure_private_file, lock_exclusive, unlock_exclusive

POOL_SIZE = 33
REVIEW_FRACTION_NUMERATOR = 20
REVIEW_FRACTION_DENOMINATOR = 100
_OUTCOMES = {"corrected", "correct_as_is", "no_speech"}
_SKIPS = {"private", "corrupt", "wrong_capture"}
_CANDIDATE_FIELDS = {"stable_id", "backend", "repo", "revision", "package_versions", "decode_settings", "language", "glossary_hash", "glossary_identity", "evaluator_id", "evaluator_hash"}


class AdaptiveLearning:
    """Single-writer adaptive state with revision/CAS publication.

    The adaptive lock is independent from both HistoryStore's storage lock and
    InferenceScheduler's locks.  Callers prepare history/audio facts before
    entering this API; callbacks run outside its lock.

    Integration API (all status/list methods are content-free):
    ``register_capture``, ``review_list``, ``resolve_review``,
    ``retry_pending_gold``, ``development_references``,
    ``development_ready``, ``create_generation``, ``freeze_generation``,
    ``open_pool``, ``pool_status``, ``close_pool``,
    ``recover_interrupted_attempts``, ``evaluate``, ``decide``,
    ``rollback``, ``revoke``, and ``status``.  ``evaluate`` may be called
    repeatedly after process interruption; it resumes the persisted attempt
    state and accepts ``callback(candidate_manifest, capture_metadata)``.
    """
    def __init__(self, base_dir: Path | str, *, clock: Callable[[], float] = time.time) -> None:
        self.base_dir = Path(base_dir)
        self.root = self.base_dir / "adaptive-learning"
        self.path = self.root / "state.json"
        self.lock_path = self.root / ".adaptive-state.lock"
        self.clock = clock
        ensure_private_directory(self.base_dir)
        ensure_private_directory(self.root)
        ensure_private_file(self.path)
        ensure_private_file(self.lock_path)
        if not self.path.exists():
            with self._locked():
                if not self.path.exists():
                    self._write({"schema": 1, "revision": 0, "salt": os.urandom(32).hex(), "captures": {}, "generations": {}, "champions": {"baseline": None, "current": None, "prior": []}, "tombstones": [], "runtime": {"failures": {}, "audit": []}})

    @contextmanager
    def _locked(self) -> Iterator[None]:
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.chmod(self.lock_path, 0o600)
        try:
            lock_exclusive(fd); yield
        finally:
            unlock_exclusive(fd); os.close(fd)

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text("utf-8"))
            if not isinstance(value, dict) or value.get("schema") != 1 or not isinstance(value.get("captures"), dict): raise ValueError
            return value
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("adaptive state is malformed; refusing to operate") from exc

    def _write(self, value: dict[str, Any]) -> None:
        temp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":")); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temp, 0o600); os.replace(temp, self.path); os.chmod(self.path, 0o600); _fsync_dir(self.root)

    def _mutate(self, change: Callable[[dict[str, Any]], Any], expected_revision: int | None = None) -> Any:
        with self._locked():
            state = self._read()
            if expected_revision is not None and state["revision"] != expected_revision: raise RuntimeError("adaptive revision conflict")
            result = change(state)
            state["revision"] += 1; self._write(state)
            return result

    @staticmethod
    def _eligible(capture: dict[str, Any], freeze_ts: float) -> bool:
        return (capture.get("adaptive") is True and capture.get("language") == "en" and
                float(capture.get("captured_ts", -math.inf)) > freeze_ts and capture.get("schema_version") == 2 and
                capture.get("canonical_16k_pcm") is True and isinstance(capture.get("audio_digest"), str) and
                .5 <= float(capture.get("duration", 0)) <= 45 and not capture.get("revoked") and
                not capture.get("pool_generation") and not capture.get("prior_pool"))

    def register_capture(self, capture: dict[str, Any]) -> str:
        """Record only canonical capture metadata; no hypothesis is accepted."""
        history_id = capture.get("history_id") or capture.get("id")
        if not isinstance(history_id, str) or not history_id: raise ValueError("history_id required")
        required = ("captured_ts", "duration", "audio_digest")
        if any(key not in capture for key in required): raise ValueError("incomplete capture")
        def change(state: dict[str, Any]) -> str:
            old = state["captures"].get(history_id, {})
            # Correction/reference state survives metadata refreshes.
            row = {"history_id": history_id, "captured_ts": float(capture["captured_ts"]), "duration": float(capture["duration"]),
                   "audio_digest": str(capture["audio_digest"]), "schema_version": int(capture.get("schema_version", 2)),
                   "canonical_16k_pcm": bool(capture.get("canonical_16k_pcm", True)),
                   # A retained-history reconciliation is intentionally
                   # conservative.  It must never downgrade a known new live
                   # adaptive capture to a legacy/ineligible row on restart.
                   "adaptive": bool(old.get("adaptive", False) or capture.get("adaptive", False)),
                   "language": capture.get("language", "en"), "revoked": bool(old.get("revoked", False)),
                   "pool_generation": old.get("pool_generation"), "prior_pool": bool(old.get("prior_pool", False)), "outcome": old.get("outcome"), "reference": old.get("reference"),
                   "skip_reason": old.get("skip_reason"), "review_revision": old.get("review_revision"),
                   "reference_revision": int(old.get("reference_revision", 0)), "pending_gold": old.get("pending_gold")}
            # A later runtime supplies these directly from HistoryStore after
            # verifying the schema-v2 canonical WAV identity.  They are not
            # returned by status/review APIs, only to the ASR callback.
            if "audio_path" in capture: row["audio_path"] = str(capture["audio_path"])
            if "audio_identity" in capture: row["audio_identity"] = dict(capture["audio_identity"])
            state["captures"][history_id] = row
            self._assign_open_pools(state)
            return history_id
        return self._mutate(change)

    def review_selected(self, history_id: str) -> bool:
        with self._locked():
            state = self._read(); raw = bytes.fromhex(state["salt"])
            digest = hmac.new(raw, history_id.encode("utf-8"), hashlib.sha256).digest()
            return int.from_bytes(digest, "big") < (REVIEW_FRACTION_NUMERATOR * (1 << 256)) // REVIEW_FRACTION_DENOMINATOR

    def review_list(self, generation_id: str | None = None) -> list[dict[str, Any]]:
        with self._locked():
            state = self._read(); result = []; raw = bytes.fromhex(state["salt"])
            for row in state["captures"].values():
                # Revocation leaves an intentionally compact tombstone without
                # capture metadata; consumed cohorts must never re-enter the
                # ordinary salted queue on a later reconciliation.
                if (row.get("revoked") or row.get("adaptive") is not True or row.get("language") != "en" or
                        row.get("schema_version") != 2 or row.get("canonical_16k_pcm") is not True or
                        (generation_id is None and row.get("prior_pool"))):
                    continue
                digest = hmac.new(raw, row["history_id"].encode("utf-8"), hashlib.sha256).digest()
                normal = int.from_bytes(digest, "big") < (REVIEW_FRACTION_NUMERATOR * (1 << 256)) // REVIEW_FRACTION_DENOMINATOR
                selected = row.get("pool_generation") == generation_id if generation_id else (not row.get("pool_generation") and normal)
                if selected:
                    result.append({"history_id": row["history_id"], "captured_ts": row["captured_ts"], "status": row.get("outcome") or "pending", "skip_reason": row.get("skip_reason"), "pending_gold": bool(row.get("pending_gold"))})
            return sorted(result, key=lambda item: (item["captured_ts"], item["history_id"]))

    def review_disposition(self, history_id: str) -> str | None:
        """Return only a review's durable disposition, never its gold text."""
        with self._locked():
            row = self._read()["captures"].get(history_id)
            if row is None:
                return None
            if row.get("revoked"):
                return "revoked"
            return row.get("outcome") or "pending"

    def review_operation_state(self, history_id: str, revision: int, outcome: str) -> dict[str, bool]:
        """Content-free completion state for a history-backed review outbox."""
        with self._locked():
            row = self._read()["captures"].get(history_id) or {}
            complete = (not row.get("revoked") and row.get("outcome") == outcome and
                        row.get("review_revision") == revision)
            return {"complete": complete, "pending_gold": bool(row.get("pending_gold")) if complete else False}

    def complete_review_enrollment(self, history_id: str, revision: int, outcome: str) -> bool:
        """Clear a matching outbox review's enrollment retry flag by CAS."""
        def change(state: dict[str, Any]) -> bool:
            row = state["captures"].get(history_id)
            if not row or row.get("outcome") != outcome or row.get("review_revision") != revision:
                return False
            if not row.get("pending_gold"):
                return True
            row["pending_gold"] = None
            return True
        return self._mutate(change)

    def pending_review_enrollments(self) -> int:
        """Count unfinished automatic enrollments without exposing gold."""
        with self._locked():
            return sum(bool(row.get("pending_gold")) for row in self._read()["captures"].values()
                       if row.get("review_revision") is not None)

    def resolve_review(self, history_id: str, outcome: str, *, reference: str = "", skip_reason: str | None = None,
                       review_revision: int | None = None, auto_enroll: Callable[[str, str], Any] | None = None) -> None:
        if outcome not in _OUTCOMES and outcome != "skip": raise ValueError("invalid review outcome")
        if outcome == "skip" and skip_reason not in _SKIPS: raise ValueError("invalid skip reason")
        if outcome == "corrected" and not isinstance(reference, str): raise TypeError("reference must be text")
        # Gold must first be durable, even if the external enrollment action fails.
        def change(state: dict[str, Any]) -> None:
            row = state["captures"].get(history_id)
            if row is None or row.get("revoked"): raise KeyError(history_id)
            row["outcome"] = outcome; row["skip_reason"] = skip_reason
            if review_revision is not None: row["review_revision"] = review_revision
            row["reference"] = reference if outcome in {"corrected", "correct_as_is"} else ""
            row["reference_revision"] = int(row.get("reference_revision", 0)) + 1
            row["pending_gold"] = {"code": "not_attempted", "ts": self.clock()} if outcome in _OUTCOMES else None
            generation_id = row.get("pool_generation")
            if generation_id:
                generation = state["generations"].get(generation_id)
                if generation and generation.get("status") in {"pool_closed", "evaluating"}:
                    # A changed frozen reference is no longer promotion evidence.
                    generation.update({"status": "invalidated", "consumed": True, "decision": {"outcome": "reference_revision_invalidated"}})
                    for selected in generation["pool"]:
                        member = state["captures"][selected]
                        member["pool_generation"] = None; member["prior_pool"] = True
        self._mutate(change)
        if outcome in _OUTCOMES and auto_enroll is not None:
            try:
                auto_enroll(history_id, reference)
            except Exception as exc:
                self._mutate(lambda state: state["captures"][history_id].update({"pending_gold": {"code": type(exc).__name__, "ts": self.clock()}}))
            else:
                self._mutate(lambda state: state["captures"][history_id].update({"pending_gold": None}))

    def human_correction(self, history_id: str, reference: str, *, auto_enroll: Callable[[str, str], Any] | None = None) -> None:
        self.resolve_review(history_id, "corrected", reference=reference, auto_enroll=auto_enroll)

    def correct_as_is(self, history_id: str, reference: str, *, auto_enroll: Callable[[str, str], Any] | None = None) -> None:
        self.resolve_review(history_id, "correct_as_is", reference=reference, auto_enroll=auto_enroll)

    def explicit_no_speech(self, history_id: str, *, auto_enroll: Callable[[str, str], Any] | None = None) -> None:
        self.resolve_review(history_id, "no_speech", auto_enroll=auto_enroll)

    def skip_review(self, history_id: str, reason: str) -> None:
        # A replayed refreshed menu action has no work left to do.  In
        # particular, do not rewrite reference_revision or pool state.
        if self.review_disposition(history_id) == "skip":
            return
        self.resolve_review(history_id, "skip", skip_reason=reason)

    def retry_pending_gold(self, enroll: Callable[[str, str], Any]) -> int:
        with self._locked():
            state = self._read(); items = [(x["history_id"], x.get("reference", "")) for x in state["captures"].values() if x.get("pending_gold")]
        completed = 0
        for history_id, reference in items:
            try: enroll(history_id, reference)
            except Exception as exc:
                self._mutate(lambda state, ident=history_id, err=exc: state["captures"][ident].update({"pending_gold": {"code": type(err).__name__, "ts": self.clock()}}))
            else:
                completed += 1; self._mutate(lambda state, ident=history_id: state["captures"][ident].update({"pending_gold": None}))
        return completed

    def development_references(self) -> list[dict[str, str]]:
        with self._locked():
            state = self._read()
            return [{"history_id": row["history_id"], "reference": row.get("reference", "")} for row in state["captures"].values()
                    if not row.get("revoked") and not row.get("pool_generation") and row.get("outcome") in {"corrected", "correct_as_is"} and tokenize(row.get("reference", ""))]

    def development_reference_ids(self) -> list[str]:
        """Opaque development IDs for integrations that must fetch history locally.

        The older ``development_references`` API is retained for accepted-core
        compatibility; new glossary code must use this content-free form.
        """
        with self._locked():
            state = self._read()
            return sorted(row["history_id"] for row in state["captures"].values()
                          if not row.get("revoked") and not row.get("pool_generation")
                          and row.get("outcome") in {"corrected", "correct_as_is"})

    def development_ready(self) -> dict[str, Any]:
        refs = self.development_references(); words = sum(len(tokenize(item["reference"])) for item in refs)
        return {"verified_nonempty": len(refs), "words": words, "ready": len(refs) >= 20 and words >= 200}

    def create_generation(self, *, generation_id: str | None = None, champion: dict[str, Any], evaluator_hash: str,
                          freeze_ts: float | None = None) -> str:
        """Create a draft.  Call ``freeze_generation`` then ``open_pool``.

        A draft accepts no captures, which makes it impossible for a mutable
        challenger family to influence a cohort.  ``freeze_ts`` is retained
        only for deterministic tests; production callers omit it.
        """
        ident = generation_id or uuid.uuid4().hex
        if _CANDIDATE_FIELDS - set(champion) or champion.get("evaluator_id") != PAIRED_EVALUATOR_ID or champion.get("evaluator_hash") != evaluator_hash:
            raise ValueError("champion immutable evaluator identity mismatch")
        def change(state: dict[str, Any]) -> str:
            if ident in state["generations"]: raise ValueError("generation exists")
            if any(g.get("status") in {"draft", "candidates_frozen", "pool_open", "pool_closed", "evaluating"}
                   for g in state["generations"].values()):
                raise RuntimeError("an unfinished generation already exists")
            state["generations"][ident] = {
                "id": ident,
                "status": "draft",
                "requested_freeze_ts": freeze_ts,
                "freeze_ts": None,
                "pool": [],
                "champion": dict(champion),
                "champion_attempts": {},
                "evaluator_hash": evaluator_hash,
                "candidates": {},
                "family_size": None,
                "decision": None,
                "consumed": False,
            }
            return ident
        return self._mutate(change)

    def _sample_digest(self, state: dict[str, Any], history_id: str) -> str:
        return hmac.new(bytes.fromhex(state["salt"]), history_id.encode("utf-8"), hashlib.sha256).hexdigest()

    def _freeze_manifest(self, state: dict[str, Any], candidate: dict[str, Any], evaluator_hash: str) -> dict[str, Any]:
        missing = _CANDIDATE_FIELDS - set(candidate)
        if missing:
            raise ValueError("candidate lacks immutable identity: " + ",".join(sorted(missing)))
        if candidate.get("evaluator_id") != PAIRED_EVALUATOR_ID or candidate.get("evaluator_hash") != evaluator_hash:
            raise ValueError("evaluator identity mismatch")
        manifest = dict(candidate)
        dependencies = manifest.pop("development_dependencies", ())
        if not isinstance(dependencies, (list, tuple, set)) or not all(isinstance(item, str) for item in dependencies):
            raise ValueError("development_dependencies must be sample ids")
        manifest["development_dependency_digests"] = sorted({self._sample_digest(state, item) for item in dependencies})
        manifest["development_dependency_revisions"] = sorted(
            [{"sample": self._sample_digest(state, item),
              "reference_revision": int((state["captures"].get(item) or {}).get("reference_revision", 0))}
             for item in set(dependencies)], key=lambda item: item["sample"])
        return manifest

    def freeze_generation(self, generation_id: str, candidates: Iterable[dict[str, Any]], *, expected_revision: int | None = None) -> None:
        """Atomically freeze the complete challenger family before cohort open."""
        frozen_input = list(candidates)
        def change(state: dict[str, Any]) -> None:
            generation = state["generations"][generation_id]
            if generation["status"] != "draft":
                raise RuntimeError("candidate family is already immutable")
            manifests = [self._freeze_manifest(state, item, generation["evaluator_hash"]) for item in frozen_input]
            stable_ids = [str(item["stable_id"]) for item in manifests]
            if not manifests or len(set(stable_ids)) != len(stable_ids):
                raise ValueError("candidate family must be nonempty with unique stable ids")
            generation["candidates"] = {item["stable_id"]: {"manifest": item, "attempts": {}, "metrics": None} for item in manifests}
            generation["family_size"] = len(manifests)
            generation["freeze_ts"] = self.clock() if generation["requested_freeze_ts"] is None else float(generation["requested_freeze_ts"])
            generation["status"] = "candidates_frozen"
        self._mutate(change, expected_revision)

    def _comparison_identity(self, state: dict[str, Any], champion: dict[str, Any],
                             manifests: list[dict[str, Any]], evaluator_hash: str) -> str:
        """Hash the fixed comparison arms and the development input they use.

        Pool rows are deliberately excluded: a new cohort is not permission to
        repeat an unchanged model/configuration comparison after its one look.
        Relevant development revisions are already frozen into each manifest.
        """
        # Use the same semantic projection as persisted receipts.  In
        # particular, local snapshot/cache paths and runtime provenance are
        # transport details, not a new statistical comparison.
        from speech_backends import _public_manifest
        def semantic(manifest: dict[str, Any]) -> dict[str, Any]:
            result = _public_manifest(manifest)
            result["development_dependency_digests"] = sorted(manifest.get("development_dependency_digests", ()))
            result["development_dependency_revisions"] = sorted(
                [dict(item) for item in manifest.get("development_dependency_revisions", ())],
                key=lambda item: (item.get("sample", ""), item.get("reference_revision", 0)))
            return result
        payload = {"champion": semantic(champion),
                   "challengers": sorted((semantic(item) for item in manifests), key=lambda item: item["stable_id"]),
                   "evaluator": {"id": PAIRED_EVALUATOR_ID, "hash": evaluator_hash}}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def publish_generation(self, *, champion: dict[str, Any], candidates: Iterable[dict[str, Any]], evaluator_hash: str,
                           generation_id: str | None = None, freeze_ts: float | None = None) -> str:
        """CAS-publish one complete frozen comparison and open its cohort.

        Runtime callers must use this single transition; the older staged
        methods remain solely for compatibility and deterministic core tests.
        """
        ident = generation_id or uuid.uuid4().hex
        frozen_input = list(candidates)
        if _CANDIDATE_FIELDS - set(champion) or champion.get("evaluator_id") != PAIRED_EVALUATOR_ID or champion.get("evaluator_hash") != evaluator_hash:
            raise ValueError("champion immutable evaluator identity mismatch")
        def change(state: dict[str, Any]) -> str:
            if ident in state["generations"]: raise ValueError("generation exists")
            if any(g.get("status") in {"draft", "candidates_frozen", "pool_open", "pool_closed", "evaluating"}
                   for g in state["generations"].values()):
                raise RuntimeError("an unfinished generation already exists")
            manifests = [self._freeze_manifest(state, item, evaluator_hash) for item in frozen_input]
            stable_ids = [str(item["stable_id"]) for item in manifests]
            if not manifests or len(set(stable_ids)) != len(stable_ids):
                raise ValueError("candidate family must be nonempty with unique stable ids")
            comparison_identity = self._comparison_identity(state, dict(champion), manifests, evaluator_hash)
            if any(g.get("consumed") and g.get("comparison_identity") == comparison_identity
                   for g in state["generations"].values()):
                raise RuntimeError("exact comparison already consumed; frozen input must change")
            generation = {"id": ident, "status": "pool_open", "requested_freeze_ts": freeze_ts,
                          "freeze_ts": self.clock() if freeze_ts is None else float(freeze_ts), "pool": [],
                          "champion": dict(champion), "champion_attempts": {}, "evaluator_hash": evaluator_hash,
                          "candidates": {item["stable_id"]: {"manifest": item, "attempts": {}, "metrics": None} for item in manifests},
                          "family_size": len(manifests), "comparison_identity": comparison_identity,
                          "decision": None, "consumed": False}
            state["generations"][ident] = generation
            self._assign_open_pools(state)
            return ident
        return self._mutate(change)

    def recover_partial_publications(self) -> list[str]:
        """Abort legacy staged publications; never infer a mutable family after restart."""
        with self._locked():
            state = self._read()
            partial = [g for g in state["generations"].values() if g.get("status") in {"draft", "candidates_frozen"}]
            if not partial:
                return []
            for generation in partial:
                generation.update({"status": "publication_aborted", "consumed": True,
                                   "decision": {"outcome": "publication_aborted"}})
            state["revision"] += 1; self._write(state)
            return sorted(g["id"] for g in partial)

    def open_pool(self, generation_id: str, *, expected_revision: int | None = None) -> bool:
        """Open a cohort only after immutable champion/family/evaluator freeze."""
        def change(state: dict[str, Any]) -> bool:
            generation = state["generations"][generation_id]
            if generation["status"] == "pool_open":
                return True
            if generation["status"] != "candidates_frozen":
                raise RuntimeError("freeze the complete candidate family before opening a pool")
            generation["status"] = "pool_open"
            self._assign_open_pools(state)
            return True
        return self._mutate(change, expected_revision)

    def _assign_open_pools(self, state: dict[str, Any]) -> None:
        for generation in state["generations"].values():
            if generation.get("status") != "pool_open":
                continue
            # Before the first human disposition, recompute the chronological
            # first-33 set atomically.  This handles delayed persistence of an
            # earlier capture without allowing a reviewed cohort to drift.
            reviewed = any(state["captures"][ident].get("outcome") is not None for ident in generation["pool"])
            eligible = []
            for row in state["captures"].values():
                if row.get("pool_generation") not in {None, generation["id"]}:
                    continue
                candidate = dict(row)
                candidate["pool_generation"] = None
                if self._eligible(candidate, generation["freeze_ts"]):
                    eligible.append(row)
            eligible.sort(key=lambda row: (row["captured_ts"], row["history_id"]))
            if reviewed:
                selected = [state["captures"][ident] for ident in generation["pool"]]
                selected.extend(row for row in eligible if row["history_id"] not in generation["pool"])
                selected = selected[:POOL_SIZE]
                for row in selected:
                    row["pool_generation"] = generation["id"]
                generation["pool"] = [row["history_id"] for row in selected]
                continue
            selected = eligible[:POOL_SIZE]
            selected_ids = {row["history_id"] for row in selected}
            for old_id in generation["pool"]:
                if old_id not in selected_ids:
                    state["captures"][old_id]["pool_generation"] = None
            for row in selected:
                row["pool_generation"] = generation["id"]
            generation["pool"] = [row["history_id"] for row in selected]

    def pool_status(self, generation_id: str) -> dict[str, Any]:
        with self._locked():
            state = self._read(); generation = state["generations"][generation_id]
            rows = [state["captures"][ident] for ident in generation["pool"]]
            outcomes = [row.get("outcome") for row in rows]
            verified = sum(value in _OUTCOMES for value in outcomes)
            nonempty = sum(len(tokenize(row.get("reference", ""))) for row in rows if row.get("outcome") in {"corrected", "correct_as_is"})
            return {"status": generation["status"], "selected": len(rows), "resolved": sum(value is not None for value in outcomes), "verified": verified,
                    "words": nonempty, "no_speech": outcomes.count("no_speech"), "skips": outcomes.count("skip"),
                    "ready": len(rows) == POOL_SIZE and all(value is not None for value in outcomes) and verified >= 30 and nonempty >= 300 and outcomes.count("no_speech") >= 3 and outcomes.count("skip") <= 3}

    def close_pool(self, generation_id: str) -> bool:
        with self._locked():
            state = self._read()
            g = state["generations"][generation_id]
            if g["status"] == "pool_closed":
                return True
            if g["status"] != "pool_open":
                raise RuntimeError("pool is not open")
            rows = [state["captures"][i] for i in g["pool"]]
            outcomes = [row.get("outcome") for row in rows]
            # Pending selections remain open.  A single look is consumed only
            # once the fixed pool has every required human disposition.
            if len(rows) < POOL_SIZE or any(value is None for value in outcomes):
                return False
            words = sum(len(tokenize(row.get("reference", ""))) for row in rows if row.get("outcome") in {"corrected", "correct_as_is"})
            ready = len(rows) == POOL_SIZE and all(x is not None for x in outcomes) and sum(x in _OUTCOMES for x in outcomes) >= 30 and words >= 300 and outcomes.count("no_speech") >= 3 and outcomes.count("skip") <= 3
            g["status"] = "pool_closed" if ready else "invalidated"
            g["consumed"] = not ready
            for row in rows:
                row["prior_pool"] = True
            state["revision"] += 1
            self._write(state)
            return ready

    def freeze_candidate(self, generation_id: str, candidate: dict[str, Any]) -> str:
        """Rejected compatibility shim: partial family mutation is unsafe."""
        del generation_id, candidate
        raise RuntimeError("use freeze_generation with the complete candidate family before open_pool")

    @staticmethod
    def _attempt_authority_digest(generation_id: str, evaluator_owner: str | None,
                                  manifest: dict[str, Any], capture: dict[str, Any]) -> str:
        """Local callback authority binding; never persist its content-bearing input."""
        payload={"generation_id":generation_id,"evaluator":evaluator_owner,
                 "manifest":manifest,"capture":capture}
        return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()

    def _attempt(self, generation_id: str, stable_id: str | None, sample_id: str, kind: str, callback: Callable[..., Any], evaluator_owner: str | None = None) -> tuple[str, float]:
        """Persist pending/running/completed attempts; callback executes unlocked."""
        with self._locked():
            state = self._read()
            g = state["generations"][generation_id]
            if g.get("status") != "evaluating":
                raise RuntimeError("generation no longer evaluating")
            c = g["candidates"].get(stable_id) if stable_id is not None else None
            if kind == "champion":
                attempts = g["champion_attempts"].setdefault(sample_id, [])
            elif c is not None:
                attempts = c["attempts"].setdefault(sample_id, [])
            else:
                raise ValueError("candidate attempt requires a stable id")
            completed = next((item for item in attempts if item["status"] == "completed"), None)
            if completed: return completed["text"], completed["latency"]
            for prior in attempts:
                if prior.get("status") == "running":
                    # A prior process/evaluator disappeared between durable state
                    # transitions.  Preserve it as a retryable interruption.
                    prior.update({"status": "failed", "code": "interrupted"})
            if len(attempts) >= 3: raise RuntimeError("attempts exhausted")
            attempt_id=uuid.uuid4().hex
            row = {"status": "running", "ts": self.clock(), "owner": evaluator_owner, "attempt_id":attempt_id}
            attempts.append(row)
            candidate = dict(g["champion"] if kind == "champion" else c["manifest"])
            stored_capture = dict(state["captures"][sample_id])
            authority=self._attempt_authority_digest(generation_id,evaluator_owner,candidate,stored_capture)
            state["revision"] += 1
            self._write(state)
            stored_capture = state["captures"][sample_id]
            # Evaluation callbacks receive only the canonical-audio locator and
            # immutable capture facts, never a human reference/review outcome.
            capture = {key: stored_capture[key] for key in ("history_id", "captured_ts", "duration", "audio_digest", "audio_path", "audio_identity", "schema_version", "canonical_16k_pcm", "language") if key in stored_capture}
        started = time.monotonic()
        try:
            answer = callback(candidate, capture)
            text = answer[0] if isinstance(answer, tuple) else answer
            latency = float(answer[1]) if isinstance(answer, tuple) and len(answer) > 1 else time.monotonic() - started
            if not isinstance(text, str): raise TypeError("transcription callback must return text")
        except Exception as exc:
            with self._locked():
                state = self._read()
                g = state["generations"].get(generation_id)
                if g is not None:
                    attempts = (g.get("champion_attempts",{}).get(sample_id,[]) if kind == "champion"
                              else g.get("candidates",{}).get(stable_id,{}).get("attempts",{}).get(sample_id,[]))
                    exact=next((item for item in attempts if item.get("attempt_id") == attempt_id),None)
                    if exact is not None and exact.get("status") == "running":
                        authority_error=str(exc) in {"human_evaluation_authority_changed","reviewed_audio_identity_mismatch"}
                        exact.update({"status": "failed", "code": "authority_changed" if authority_error else type(exc).__name__})
                        state["revision"] += 1; self._write(state)
            raise
        with self._locked():
            state = self._read()
            g = state["generations"].get(generation_id)
            attempts = ([] if g is None else (g.get("champion_attempts",{}).get(sample_id,[]) if kind == "champion"
                        else g.get("candidates",{}).get(stable_id,{}).get("attempts",{}).get(sample_id,[])))
            exact=next((item for item in attempts if item.get("attempt_id") == attempt_id),None)
            current_manifest=(g.get("champion") if kind == "champion" else g.get("candidates",{}).get(stable_id,{}).get("manifest")) if g is not None else None
            current_capture=state.get("captures",{}).get(sample_id)
            valid=(g is not None and g.get("status") == "evaluating" and exact is not None and exact.get("status") == "running" and
                   isinstance(current_manifest,dict) and isinstance(current_capture,dict) and
                   self._attempt_authority_digest(generation_id,evaluator_owner,current_manifest,current_capture) == authority)
            if not valid:
                if exact is not None and exact.get("status") == "running":
                    exact.update({"status":"failed","code":"authority_changed"})
                    state["revision"] += 1; self._write(state)
                raise RuntimeError("human_evaluation_authority_changed")
            exact.update({"status": "completed", "text": text, "latency": latency})
            state["revision"] += 1; self._write(state)
        return text, latency

    def recover_interrupted_attempts(self, reclaimed_evaluator_owners: Iterable[str]) -> int:
        """Mark work abandoned by scheduler-reclaimed evaluator owners retryable."""
        owners = set(reclaimed_evaluator_owners)
        if not owners: return 0
        def change(state: dict[str, Any]) -> int:
            changed = 0
            for generation in state["generations"].values():
                for attempts in generation.get("champion_attempts", {}).values():
                    for attempt in attempts:
                        if attempt.get("status") == "running" and attempt.get("owner") in owners:
                            attempt.update({"status": "failed", "code": "interrupted"})
                            changed += 1
                for candidate in generation["candidates"].values():
                    for attempts in candidate["attempts"].values():
                        for attempt in attempts:
                            if attempt.get("status") == "running" and attempt.get("owner") in owners:
                                attempt.update({"status": "failed", "code": "interrupted"}); changed += 1
            return changed
        return self._mutate(change)

    def evaluate(self, generation_id: str, transcribe: Callable[..., Any], *, evaluator_owner: str | None = None) -> dict[str, Any]:
        """Resume deterministic candidate/sample evaluation; no content in return."""
        with self._locked():
            state = self._read(); g = state["generations"][generation_id]
            if g["status"] not in {"pool_closed", "evaluating"}: raise RuntimeError("generation not evaluable")
            from speech_backends import evaluator_hash
            if g.get("evaluator_hash") != evaluator_hash():
                return self._technical_abort_locked(state, g, "evaluator_hash_current_mismatch")
            if g["champion"].get("evaluator_id") != PAIRED_EVALUATOR_ID: return self._technical_abort_locked(state, g, "evaluator_mismatch")
            if g["champion"].get("evaluator_hash") != g.get("evaluator_hash") or any(c["manifest"].get("evaluator_hash") != g.get("evaluator_hash") for c in g["candidates"].values()):
                return self._technical_abort_locked(state, g, "evaluator_hash_mismatch")
            g["status"] = "evaluating"; state["revision"] += 1; self._write(state)
            # Skips resolve the fixed cohort administratively but are not
            # speech evidence: never invoke either model on their audio.
            ids = [sample_id for sample_id in g["pool"]
                   if state["captures"][sample_id].get("outcome") in _OUTCOMES]
            candidates = sorted(g["candidates"])
        # The incumbent is generation-level evidence: exactly one transcript per
        # cohort sample, then every challenger uses those same completed rows.
        for sample_id in ids:
            while True:
                try:
                    self._attempt(generation_id, None, sample_id, "champion", transcribe, evaluator_owner)
                    break
                except Exception:
                    terminal=self._evaluation_terminal_outcome(generation_id)
                    if terminal is not None: return terminal
                    if self._attempt_count(generation_id, None, "champion", sample_id) >= 3:
                        with self._locked():
                            state = self._read()
                            return self._technical_abort_locked(state, state["generations"][generation_id], "technical_abort_champion")
        for stable_id in candidates:
            for sample_id in ids:
                while True:
                    try:
                        self._attempt(generation_id, stable_id, sample_id, "candidate", transcribe, evaluator_owner)
                        break
                    except Exception:
                        terminal=self._evaluation_terminal_outcome(generation_id)
                        if terminal is not None: return terminal
                        if self._attempt_count(generation_id, stable_id, "candidate", sample_id) >= 3: break
        return self.decide(generation_id)

    def _evaluation_terminal_outcome(self, generation_id: str) -> dict[str, Any] | None:
        with self._locked():
            generation=self._read()["generations"].get(generation_id)
            if generation is None or generation.get("status") == "evaluating": return None
            decision=generation.get("decision")
            return dict(decision) if isinstance(decision,dict) else {"outcome":str(generation.get("status","invalidated"))}

    def _attempt_count(self, generation_id: str, stable_id: str | None, kind: str, sample_id: str) -> int:
        with self._locked():
            generation = self._read()["generations"][generation_id]
            attempts = generation["champion_attempts"].get(sample_id, []) if kind == "champion" else generation["candidates"][stable_id]["attempts"].get(sample_id, [])
            return len(attempts)

    def _technical_abort_locked(self, state: dict[str, Any], g: dict[str, Any], code: str) -> dict[str, Any]:
        g.update({"status": code, "consumed": True, "decision": {"outcome": code}}); state["revision"] += 1; self._write(state); return {"outcome": code}

    def decide(self, generation_id: str, *, expected_revision: int | None = None) -> dict[str, Any]:
        def change(state: dict[str, Any]) -> dict[str, Any]:
            g = state["generations"][generation_id]
            if g.get("decision") is not None: return dict(g["decision"])
            if g.get("consumed"): return {"outcome": g["status"]}
            stats: dict[str, dict[str, Any]] = {}; pvals: list[tuple[str, float]] = []
            for stable_id in sorted(g["candidates"]):
                candidate = g["candidates"][stable_id]; rows: list[PairedPromotionRow] = []; unresolved = False
                champion_hallucinations = candidate_hallucinations = 0
                champion_latencies: list[float] = []; candidate_latencies: list[float] = []
                for sample_id in g["pool"]:
                    capture = state["captures"][sample_id]; attempts = candidate["attempts"]
                    if capture.get("outcome") not in _OUTCOMES:
                        continue
                    ca = attempts.get(sample_id, [])
                    ch = g["champion_attempts"].get(sample_id, [])
                    done_c = next((x for x in ca if x["status"] == "completed"), None); done_h = next((x for x in ch if x["status"] == "completed"), None)
                    if done_h is None: return self._technical_abort_locked(state, g, "technical_abort_champion")
                    if done_c is None: unresolved = True; continue
                    ref = capture.get("reference", ""); ref_words = len(tokenize(ref));
                    champion_latencies.append(done_h["latency"]); candidate_latencies.append(done_c["latency"])
                    if not ref_words:
                        champion_hallucinations += bool(tokenize(done_h["text"])); candidate_hallucinations += bool(tokenize(done_c["text"]))
                    if ref_words:
                        # local import avoids retaining full evaluation transcripts in a status object
                        from evaluation import _edit_distance
                        ct, nt, rt = tokenize(done_h["text"]), tokenize(done_c["text"]), tokenize(ref)
                        rows.append(PairedPromotionRow(sample_id, _edit_distance(rt, ct), _edit_distance(rt, nt), len(rt),
                            _edit_distance(tuple("".join(rt)), tuple("".join(ct))), _edit_distance(tuple("".join(rt)), tuple("".join(nt))), len("".join(rt)), False, False, done_h["latency"], done_c["latency"]))
                if unresolved or not rows:
                    stats[stable_id] = {"p_value": 1.0, "unresolved": True}; pvals.append((stable_id, 1.0)); continue
                metric = paired_bootstrap_statistics(rows, generation_id); metric["unresolved"] = False
                metric["champion_hallucinations"] += champion_hallucinations; metric["candidate_hallucinations"] += candidate_hallucinations
                # The protocol's latency median includes explicit no-speech rows too.
                champion_latencies.sort(); candidate_latencies.sort()
                nlat = len(champion_latencies); mid = nlat // 2
                metric["champion_latency_median"] = champion_latencies[mid] if nlat % 2 else (champion_latencies[mid - 1] + champion_latencies[mid]) / 2
                metric["candidate_latency_median"] = candidate_latencies[mid] if nlat % 2 else (candidate_latencies[mid - 1] + candidate_latencies[mid]) / 2
                stats[stable_id] = metric; pvals.append((stable_id, metric["p_value"]))
            holm = holm_fixed_family(pvals)
            passing: list[tuple[tuple[Any, ...], str]] = []
            for stable_id, metric in stats.items():
                if metric.get("unresolved"): continue
                abs_gain = metric["champion_wer"] - metric["candidate_wer"]
                rel_gain = abs_gain / metric["champion_wer"] if metric["champion_wer"] else 0
                gates = (holm[stable_id] and metric["upper_ci"] < 0 and abs_gain >= .02 and rel_gain >= .10 and
                    (metric["candidate_cer"] - metric["champion_cer"] <= .005) and metric["win_or_tie_rate"] >= .65 and
                    metric["candidate_hallucinations"] <= metric["champion_hallucinations"] and metric["candidate_latency_median"] <= 1.5 * metric["champion_latency_median"])
                if gates: passing.append(((metric["candidate_wer"], metric["candidate_cer"], metric["candidate_latency_median"], stable_id), stable_id))
            winner = min(passing)[1] if passing else None
            result = {"outcome": "promoted" if winner else "no_promotion", "winner": winner, "candidates": {ident: {key: value for key, value in metric.items() if key not in {"seed_sha256"}} for ident, metric in stats.items()}}
            g.update({"status": result["outcome"], "decision": result, "consumed": True})
            if winner:
                prior = state["champions"].get("current")
                manifest = dict(g["candidates"][winner]["manifest"])
                manifest["generation_id"] = generation_id
                manifest["promotion_dependency_generation"] = generation_id
                manifest["promotion_dependency_digests"] = sorted(
                    self._sample_digest(state, ident) for ident in g["pool"]
                    if state["captures"][ident].get("outcome") in _OUTCOMES)
                manifest["invalid"] = False
                if prior:
                    state["champions"]["prior"] = ([prior] + state["champions"].get("prior", []))[:2]
                state["champions"]["current"] = manifest
            return result
        return self._mutate(change, expected_revision)

    def rollback(self, *, reason: str = "manual") -> dict[str, Any]:
        def change(state: dict[str, Any]) -> dict[str, Any]:
            old = state["champions"].get("current")
            if old:
                self._invalidate_deployment_identity(state, old)
                self._retire_runtime_identity(state, old)
            state["champions"]["current"] = self._newest_independent(state, excluded=old)
            return {"outcome": "rolled_back", "reason": reason, "had_current": bool(old)}
        return self._mutate(change)

    manual_rollback = rollback

    # Runtime integration -------------------------------------------------
    # These deliberately live beside promotion publication: deployment state
    # is durable and has the same single-writer/CAS boundary as a promotion.
    def ensure_baseline(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Publish the immutable last-known-good baseline once.

        A receipt-backed manifest is supplied by ``adaptive_runtime``.  A
        different baseline is refused rather than silently changing the
        comparison/control arm after data collection began.
        """
        missing = _CANDIDATE_FIELDS - set(manifest)
        if missing:
            raise ValueError("baseline lacks immutable identity: " + ",".join(sorted(missing)))
        def change(state: dict[str, Any]) -> dict[str, Any]:
            baseline = state["champions"].get("baseline")
            if baseline is None:
                state["champions"]["baseline"] = dict(manifest)
                return dict(state["champions"]["baseline"])
            if baseline.get("stable_id") != manifest.get("stable_id") or baseline.get("revision") != manifest.get("revision"):
                raise RuntimeError("baseline identity changed; run a new preflight before changing it")
            if baseline != manifest:
                # Same pinned identity but a refreshed receipt binding (for
                # example a digest-algorithm or path migration).  A stale row
                # can never match runtime authority again, so refresh it while
                # no generation is collecting against the baseline arm.  Mid-
                # collection the control arm must never change: the stored row
                # stays authoritative and per-capture authority checks keep
                # failing closed until the generation resolves.
                active = [g for g in state["generations"].values()
                          if g.get("status") in {"draft", "candidates_frozen", "pool_open", "pool_closed", "evaluating"}]
                if not active:
                    state["champions"]["baseline"] = dict(manifest)
            return dict(state["champions"]["baseline"])
        return self._mutate(change)

    def champion_manifest(self) -> dict[str, Any] | None:
        """Return the current valid champion, falling back to baseline."""
        with self._locked():
            state = self._read()
            current = state["champions"].get("current")
            chosen = current if current and not current.get("invalid") else state["champions"].get("baseline")
            return dict(chosen) if chosen else None

    def active_generation(self) -> dict[str, Any] | None:
        """Content-free summary of the newest unfinished generation."""
        with self._locked():
            state = self._read()
            active = [g for g in state["generations"].values()
                      if g.get("status") in {"draft", "candidates_frozen", "pool_open", "pool_closed", "evaluating"}]
            if not active:
                return None
            g = max(active, key=lambda row: (float(row.get("freeze_ts") or -1), row["id"]))
            if g.get("status") in {"pool_open", "pool_closed", "evaluating"}:
                rows = [state["captures"][ident] for ident in g.get("pool", ())]
                outcomes = [row.get("outcome") for row in rows]
                words = sum(len(tokenize(row.get("reference", ""))) for row in rows
                            if row.get("outcome") in {"corrected", "correct_as_is"})
                result = {"status": g["status"], "selected": len(rows),
                          "resolved": sum(value is not None for value in outcomes),
                          "verified": sum(value in _OUTCOMES for value in outcomes), "words": words,
                          "ready": False}
            else:
                result = {"status": g["status"], "selected": len(g.get("pool", ())), "resolved": 0, "ready": False}
            return {"id": g["id"], **result, "candidates": len(g.get("candidates", {}))}

    def frozen_manifests(self, generation_id: str) -> dict[str, Any]:
        """Internal immutable manifests for receipt validation, never status."""
        with self._locked():
            g = self._read()["generations"][generation_id]
            return {"champion": dict(g["champion"]),
                    "candidates": [dict(item["manifest"]) for item in g["candidates"].values()],
                    "all": [dict(g["champion"])] + [dict(item["manifest"]) for item in g["candidates"].values()],
                    "evaluator_hash": g.get("evaluator_hash")}

    def technical_abort(self, generation_id: str, code: str) -> dict[str, Any]:
        with self._locked():
            state = self._read()
            g = state["generations"][generation_id]
            g.update({"status": code, "consumed": True, "decision": {"outcome": code}})
            state["revision"] += 1
            self._write(state)
            return {"outcome": code}

    def pre_mutate_reference(self, history_id: str, *, reason: str = "reference_changed") -> int:
        """Consume any evidence depending on a reference before it mutates."""
        def change(state: dict[str, Any]) -> int:
            digest = self._sample_digest(state, history_id); changed = 0
            row = state["captures"].get(history_id, {})
            for g in state["generations"].values():
                # First disposition of a pending pool row is normal cohort
                # collection, not a reference revision.  Later edits are not.
                pool_changed = history_id in g.get("pool", ()) and row.get("outcome") is not None
                dependent = pool_changed or any(
                    digest in item["manifest"].get("development_dependency_digests", ())
                    for item in g.get("candidates", {}).values())
                if dependent and g.get("status") in {"draft", "candidates_frozen", "pool_open", "pool_closed", "evaluating"}:
                    g.update({"status": "invalidated", "consumed": True, "decision": {"outcome": reason}})
                    for ident in g.get("pool", ()):
                        row = state["captures"].get(ident)
                        if row: row["pool_generation"] = None; row["prior_pool"] = True
                    changed += 1
            # A corrected development/cohort reference also invalidates any
            # deployed decision that relied on its digest.  Do this before the
            # history mutation so a later storage failure is conservative.
            directly_dependent: list[dict[str, Any]] = []
            for manifest in [state["champions"].get("current")] + list(state["champions"].get("prior", [])):
                if not manifest:
                    continue
                dependent = (digest in manifest.get("development_dependency_digests", ()) or
                             digest in manifest.get("promotion_dependency_digests", ()))
                if not dependent:
                    continue
                manifest["invalid"] = True
                directly_dependent.append(manifest)
                source = state["generations"].get(manifest.get("promotion_dependency_generation"))
                if source is not None:
                    source["deployment_invalidated"] = True
                    if isinstance(source.get("decision"), dict): source["decision"]["deployment_invalidated"] = True
            for manifest in directly_dependent:
                self._invalidate_deployment_identity(state, manifest)
            current = state["champions"].get("current")
            if current and current.get("invalid"):
                self._invalidate_deployment_identity(state, current)
                state["champions"]["current"] = self._newest_independent(state, excluded=current)
            return changed
        return self._mutate(change)

    def record_runtime_success(self, stable_id: str, *, expected_manifest: dict[str, Any] | None = None) -> None:
        def change(state: dict[str, Any]) -> None:
            runtime = state.setdefault("runtime", {"failures": {}, "audit": []})
            failures = runtime.setdefault("failures", {})
            identity = self._deployment_identity(expected_manifest) if expected_manifest else None
            for key, row in list(failures.items()):
                if row.get("stable_id", key) == stable_id and (identity is None or row.get("identity") == self._identity_key(identity)):
                    failures.pop(key, None)
        self._mutate(change)

    def record_runtime_failure(self, stable_id: str, normalized_code: str, *, expected_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        """Persist a deployed-champion failure and roll back after three.

        Only the currently deployed non-baseline champion is eligible.  This
        prevents an old/asynchronous model request from rolling back a newer
        promotion.  The transition is atomic with the audit record.
        """
        code = "".join(ch for ch in str(normalized_code)[:80] if ch.isalnum() or ch in "_-") or "runtime_error"
        def change(state: dict[str, Any]) -> dict[str, Any]:
            runtime = state.setdefault("runtime", {"failures": {}, "audit": []})
            failures = runtime.setdefault("failures", {})
            current = state["champions"].get("current")
            baseline = state["champions"].get("baseline")
            current_identity = self._deployment_identity(current)
            expected_identity = self._deployment_identity(expected_manifest) if expected_manifest else current_identity
            if (not current or current.get("stable_id") != stable_id or current_identity != expected_identity or
                    (baseline and current_identity == self._deployment_identity(baseline))):
                return {"stable_id": stable_id, "streak": 0, "rolled_back": False}
            identity_key = self._identity_key(current_identity)
            row = failures.setdefault(identity_key, {"stable_id": stable_id, "identity": identity_key, "streak": 0, "code": None})
            row["streak"] = int(row.get("streak", 0)) + 1; row["code"] = code; row["ts"] = self.clock()
            rolled = row["streak"] >= 3
            if rolled:
                self._invalidate_deployment_identity(state, current)
                replacement = self._newest_independent(state, excluded=current)
                state["champions"]["current"] = replacement
                runtime["audit"].append({"ts": self.clock(), "event": "automatic_rollback", "from": stable_id,
                                         "to": (replacement or {}).get("stable_id"), "code": code})
                runtime["audit"] = runtime["audit"][-50:]
                self._retire_runtime_identity(state, current)
            return {"stable_id": stable_id, "streak": row["streak"], "rolled_back": rolled,
                    "current": (state["champions"].get("current") or {}).get("stable_id")}
        return self._mutate(change)

    def invalidate_current_if(self, expected: dict[str, Any], *, reason: str = "receipt_invalid") -> bool:
        """CAS-like deployment invalidation; never rolls back a newer champion."""
        def change(state: dict[str, Any]) -> bool:
            current = state["champions"].get("current")
            if not current or self._deployment_identity(current) != self._deployment_identity(expected): return False
            self._invalidate_deployment_identity(state, current)
            self._retire_runtime_identity(state, current)
            state["champions"]["current"] = self._newest_independent(state, excluded=current)
            state.setdefault("runtime", {"failures": {}, "audit": []})["audit"].append({"ts": self.clock(), "event": reason, "from": current.get("stable_id")})
            return True
        return self._mutate(change)

    def revoke(self, history_id: str, *, reason: str = "revoked") -> bool:
        def change(state: dict[str, Any]) -> bool:
            row = state["captures"].get(history_id)
            if not row or row.get("revoked"): return False
            generation_id = row.get("pool_generation")
            digest = self._sample_digest(state, history_id)
            # This belongs here rather than only in integration wrappers: all
            # callers (admin remove, clear and direct revocation) must consume
            # affected evidence before the user artifact disappears.
            for g in state["generations"].values():
                dependent = history_id in g.get("pool", ()) or any(
                    digest in item["manifest"].get("development_dependency_digests", ())
                    for item in g.get("candidates", {}).values())
                if dependent and g.get("status") in {"draft", "candidates_frozen", "pool_open", "pool_closed", "evaluating"}:
                    g.update({"status": "invalidated", "consumed": True, "decision": {"outcome": reason}})
                    for ident in g.get("pool", ()):
                        member = state["captures"].get(ident)
                        if member:
                            member["pool_generation"] = None; member["prior_pool"] = True
            row.clear()
            row.update({"history_id": history_id, "revoked": True, "prior_pool": bool(generation_id), "pool_generation": None})
            state["tombstones"].append({"sample_digest": digest, "generation_id": generation_id, "role": "pool" if generation_id else "development", "ts": self.clock(), "reason": reason})
            manifests = [state["champions"].get("current")] + list(state["champions"].get("prior", []))
            directly_dependent: list[dict[str, Any]] = []
            for manifest in manifests:
                if not manifest:
                    continue
                dependent = digest in manifest.get("development_dependency_digests", ()) or digest in manifest.get("promotion_dependency_digests", ())
                if dependent:
                    manifest["invalid"] = True
                    directly_dependent.append(manifest)
                    source = state["generations"].get(manifest.get("promotion_dependency_generation"))
                    if source is not None:
                        source["deployment_invalidated"] = True
                        if isinstance(source.get("decision"), dict):
                            source["decision"]["deployment_invalidated"] = True
            for manifest in directly_dependent:
                self._invalidate_deployment_identity(state, manifest)
            current = state["champions"].get("current")
            if current and current.get("invalid"):
                self._invalidate_deployment_identity(state, current)
                state["champions"]["current"] = self._newest_independent(state, excluded=current)
            return True
        return self._mutate(change)

    @staticmethod
    def _deployment_identity(manifest: dict[str, Any] | None) -> str | None:
        """Stable deployment identity, excluding paths and mutable bookkeeping."""
        if not manifest:
            return None
        from speech_backends import _public_manifest
        return json.dumps(_public_manifest(manifest), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _identity_key(identity: str | None) -> str:
        return hashlib.sha256((identity or "").encode("utf-8")).hexdigest()

    def _invalidate_deployment_identity(self, state: dict[str, Any], manifest: dict[str, Any] | None) -> None:
        """Invalidate the current deployment and every semantically identical prior."""
        identity = self._deployment_identity(manifest)
        if identity is None:
            return
        current = state["champions"].get("current")
        if current is not None and self._deployment_identity(current) == identity:
            current["invalid"] = True
        for prior in state["champions"].get("prior", []):
            if self._deployment_identity(prior) == identity:
                prior["invalid"] = True

    def _retire_runtime_identity(self, state: dict[str, Any], manifest: dict[str, Any] | None) -> None:
        runtime = state.setdefault("runtime", {"failures": {}, "audit": []})
        runtime.setdefault("failures", {}).pop(self._identity_key(self._deployment_identity(manifest)), None)

    def _newest_independent(self, state: dict[str, Any], *, excluded: dict[str, Any] | None = None) -> dict[str, Any] | None:
        excluded_identity = self._deployment_identity(excluded)
        for manifest in state["champions"].get("prior", []):
            if not manifest.get("invalid", False) and self._deployment_identity(manifest) != excluded_identity:
                return manifest
        baseline = state["champions"].get("baseline")
        return baseline if self._deployment_identity(baseline) != excluded_identity else None

    handle_revocation = revoke

    def capture_ids(self) -> list[str]:
        """Opaque IDs only, for conservative integration-wide revocation."""
        with self._locked():
            return sorted(self._read()["captures"])

    def revoke_all(self, *, reason: str = "cleared") -> int:
        """Revoke every retained dependency, including pruned history rows."""
        count = 0
        for history_id in self.capture_ids():
            if self.revoke(history_id, reason=reason): count += 1
        return count

    def clear_personal_state(self) -> None:
        """Erase every personal adaptive artifact in one durable publication.

        Preflight receipts and the independent Silver calibration ledger live
        outside this file.  Nothing in this state is public calibration
        evidence: captures, reviewed references, generation manifests and
        runtime audit metadata all derive from personal dictation.  Keeping a
        revocation tombstone here would itself retain a stable personal
        identifier, so Clear deliberately starts a fresh salted namespace.
        """
        def change(state: dict[str, Any]) -> None:
            state["salt"] = os.urandom(32).hex()
            state["captures"] = {}
            state["generations"] = {}
            state["champions"] = {"baseline": None, "current": None, "prior": []}
            state["tombstones"] = []
            state["runtime"] = {"failures": {}, "audit": []}
        self._mutate(change)

    def status(self) -> dict[str, Any]:
        with self._locked():
            state = self._read()
            runtime = state.get("runtime", {})
            return {"revision": state["revision"], "captures": len(state["captures"]), "pending_gold": sum(bool(x.get("pending_gold")) for x in state["captures"].values()),
                    "generations": {ident: {"status": g["status"], "pool": len(g["pool"]), "candidates": len(g["candidates"])} for ident, g in state["generations"].items()},
                    "baseline": (state["champions"].get("baseline") or {}).get("stable_id"),
                    "current_champion": (state["champions"].get("current") or state["champions"].get("baseline") or {}).get("stable_id"),
                    "runtime_failures": {row.get("stable_id", ident): {"streak": row.get("streak", 0), "code": row.get("code")}
                                         for ident, row in runtime.get("failures", {}).items()}}


# Compatibility-friendly descriptive name for later UI/CLI integration.
AdaptiveLearningStore = AdaptiveLearning

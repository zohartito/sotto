import json
import multiprocessing
import os
import stat
import shutil
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audio_codec import prepare_canonical, read_canonical_wav
from history import HistoryStore, _atomic_jsonl
from learning import LearningCoordinator, LearningStore
import sys

if sys.platform != "darwin":
    raise unittest.SkipTest("adaptive lane is macOS-only in v1 (fcntl)")

from adaptive_runtime import AdaptiveRuntime
from adaptive_learning import AdaptiveLearning
from inference_scheduler import InferenceScheduler
from speech_backends import PreflightReceipts, WHISPER_ID, candidate_manifest
from storage_lock import advisory_lock


def _hold_advisory_lock(base_dir, ready, release):
    with advisory_lock(base_dir):
        ready.set()
        release.wait(5)


def _read_admin_snapshot(base_dir, result):
    snapshots = LearningStore.admin_snapshot(base_dir)
    result.put(len(snapshots))


def _probe_advisory_lock(base_dir, acquired):
    with advisory_lock(base_dir):
        acquired.set()


class HistoryLearningTests(unittest.TestCase):
    def setUp(self):
        # Comparator spool authority rejects macOS's `/var` symlink ancestry;
        # use a lexical private workspace root for fixtures that exercise
        # Clear's comparator scrub path.
        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.tmp.name)
        self.history = HistoryStore(self.root)
        self.learning = LearningStore(self.root)
        self.coordinator = LearningCoordinator(self.history, self.learning)

    def tearDown(self):
        self.tmp.cleanup()

    def live(self, text="model text", **metadata):
        return self.coordinator.append_live(
            text, np.array([-1.0, -.2, 0, .2, 1.0], np.float32), .5, "model", **metadata)

    def test_legacy_migration_and_immutable_hypothesis(self):
        self.history.index.parent.mkdir(parents=True, exist_ok=True)
        self.history.index.write_text(json.dumps({"id": "old", "ts": 1, "text": "old words", "duration": 2, "model": "m"}) + "\n")
        h = HistoryStore(self.root)
        row = h.get("old")
        self.assertEqual(row["attempts"][0]["kind"], "legacy")
        self.assertEqual(row["attempts"][0]["provenance"], "legacy_unknown")
        self.assertEqual(json.loads(h.index.read_text())["text"], "old words")  # unmutated on disk
        c = LearningCoordinator(h, LearningStore(self.root))
        c.correct("old", "human words")
        self.assertEqual(h.get("old")["hypothesis"], "old words")
        self.assertEqual(h.get("old")["text"], "human words")

    def test_private_modes_are_tightened_under_permissive_umask_and_existing_paths(self):
        def mode(path: Path) -> int:
            return stat.S_IMODE(path.stat().st_mode)
        old_umask = os.umask(0o022)
        try:
            root = self.root / "private"; root.mkdir(mode=0o755)
            (root / "history.jsonl").write_text(""); os.chmod(root / "history.jsonl", 0o644)
            (root / "learning").mkdir(mode=0o755); (root / "adaptive-learning").mkdir(mode=0o755)
            history, learning = HistoryStore(root), LearningStore(root)
            coordinator = LearningCoordinator(history, learning)
            row = coordinator.append_live("private", prepare_canonical(np.array([.25, -.25], np.float32)), .5, "m",
                                          language="en", adaptive=True, raw_samples=np.array([-.5, 0, .5], np.float32),
                                          raw_sample_rate=22_050)
            runtime = AdaptiveRuntime(root, history=history, learning=learning, coordinator=coordinator)
            runtime.register_live(row); runtime.review(row["id"], "corrected", reference="gold", expected_revision=row["revision"])
            adaptive = AdaptiveLearning(root); adaptive.register_capture({"history_id": "mode", "captured_ts": 1,
                                                                           "duration": 1, "audio_digest": "x", "adaptive": True})
            receipts = PreflightReceipts(root); receipts.save(WHISPER_ID, {"success": True})
            scheduler = InferenceScheduler(root); scheduler.reconcile()
            sample_id = learning.active_for_history(row["id"])[0]["sample_id"]
            retained = (history.audio_path(row["id"]), root / row["audio"]["raw"]["path"])
            enrolled = (learning.audio_dir / f"{sample_id}.wav", learning.raw_dir / f"{sample_id}.wav")
            for artifact in (*retained, *enrolled):
                os.chmod(artifact, 0o644)
            # Startup repair covers legacy audio whose original creation obeyed
            # a permissive mode, for both retained and recoverable consent data.
            HistoryStore(root); LearningStore(root)
            for directory in (root, history.audio_dir, history.raw_audio_dir, learning.base_dir, learning.audio_dir,
                              learning.raw_dir, learning.staging_dir, adaptive.root, receipts.root, scheduler.root):
                self.assertEqual(mode(directory), 0o700, str(directory))
            for private_file in (history.index, history.audio_path(row["id"]), learning.index, adaptive.path,
                                 adaptive.lock_path, receipts.path(WHISPER_ID), scheduler.state_path, *retained, *enrolled):
                self.assertEqual(mode(private_file), 0o600, str(private_file))
        finally:
            os.umask(old_umask)

    def test_startup_permission_repair_does_not_follow_audio_symlinks(self):
        external = self.root / "outside.wav"
        external.write_bytes(b"must not be touched")
        os.chmod(external, 0o644)

        row = self.live("retained")
        retained = self.history.audio_path(row["id"])
        retained.unlink(); retained.symlink_to(external)
        HistoryStore(self.root)
        self.assertFalse(retained.exists())
        self.assertEqual(stat.S_IMODE(external.stat().st_mode), 0o644)

        row = self.live("learning")
        self.coordinator.correct(row["id"], "human")
        enrolled = self.coordinator.enroll(row["id"])
        enrolled.inference_audio_path.unlink(); enrolled.inference_audio_path.symlink_to(external)
        recovered = LearningStore(self.root)
        self.assertFalse(enrolled.inference_audio_path.exists())
        self.assertEqual(stat.S_IMODE(external.stat().st_mode), 0o644)
        self.assertEqual(recovered._records[enrolled.sample_id]["status"], "revoked")

    def test_history_audio_metadata_and_owned_symlinks_never_escape_roots(self):
        root = self.root / "audio-authority"
        history = HistoryStore(root, keep=10)
        outside = root / "outside.wav"
        sentinel = b"outside audio must remain untouched"
        outside.write_bytes(sentinel)

        retained = history.append("retained", np.array([-.2, .2], np.float32), .2, "m",
                                  raw_samples=np.array([-.3, .3], np.float32), raw_sample_rate=22_050)
        other = history.append("other", np.array([-.4, .4], np.float32), .2, "m",
                               raw_samples=np.array([-.5, .5], np.float32), raw_sample_rate=22_050)
        # Direct names are allowed even before their artifact exists; metadata
        # must never turn them into an arbitrary path.
        self.assertEqual(history.audio_path("future-safe"), history.audio_dir / "future-safe.wav")
        original = dict(retained["audio"]["inference"])
        for unsafe in (str(outside), "../outside.wav", "audio/nested/outside.wav", other["audio"]["inference"]["path"]):
            entry = history._find_locked(retained["id"])
            entry["audio"]["inference"] = {**original, "path": unsafe}
            with self.assertRaisesRegex(ValueError, "unsafe"):
                history.audio_path(retained["id"])
            with self.assertRaisesRegex(ValueError, "unavailable"):
                history.load_audio(retained["id"])
        history._find_locked(retained["id"])["audio"]["inference"] = original
        raw_original = dict(retained["audio"]["raw"])
        history._find_locked(retained["id"])["audio"]["raw"] = {**raw_original, "path": other["audio"]["raw"]["path"]}
        self.assertIsNone(history.raw_audio_path(retained["id"]))
        history._find_locked(retained["id"])["audio"]["raw"] = raw_original

        # Overflow moves only canonical evidence to the private silver root.
        silver_root = self.root / "silver-authority"
        silver_history = HistoryStore(silver_root, keep=1)
        silver = silver_history.append("silver", np.array([-.1, .1], np.float32), .2, "m", language="en", adaptive=True)
        silver_history.append("new", np.array([-.4, .4], np.float32), .2, "m", language="en", adaptive=True)
        spooled = silver_history.get(silver["id"])
        self.assertTrue(spooled["silver_spooled"])
        silver_original = dict(spooled["audio"]["inference"])
        silver_history._find_locked(silver["id"])["audio"]["inference"] = {**silver_original, "path": "adaptive-learning/silver/evidence/" + "0" * 64 + ".wav"}
        with self.assertRaisesRegex(ValueError, "unsafe"):
            silver_history.audio_path(silver["id"])
        silver_history._find_locked(silver["id"])["audio"]["inference"] = silver_original

        paths = [history.audio_path(retained["id"]), history.raw_audio_path(retained["id"]), silver_history.audio_path(silver["id"])]
        self.assertTrue(all(path is not None and path.exists() for path in paths))
        for path in paths:
            path.unlink()
            path.symlink_to(outside)

        # Startup's direct-child sweep may remove the owned links, but must
        # neither dereference nor chmod the external target.
        restarted = HistoryStore(root, keep=10)
        silver_restarted = HistoryStore(silver_root, keep=1)
        for path in paths[:2]:
            self.assertFalse(os.path.lexists(path), str(path))
        self.assertFalse(os.path.lexists(paths[2]), str(paths[2]))
        self.assertEqual(outside.read_bytes(), sentinel)
        self.assertFalse(restarted.raw_audio_path(retained["id"]).exists())
        with self.assertRaisesRegex(ValueError, "unavailable"):
            silver_restarted.load_audio(silver["id"])
        self.assertTrue(history.audio_path(other["id"]).exists())
        self.assertTrue(history.raw_audio_path(other["id"]).exists())

    def test_correct_as_is_certifies_current_retry_text_not_immutable_hypothesis(self):
        row = self.live("original")
        snapshot = self.coordinator.snapshot_for_retry(row["id"])
        self.coordinator.commit_retry(row["id"], "visible retry", snapshot.expected_revision)
        accepted = self.coordinator.correct_as_is(row["id"])
        self.assertEqual(accepted["correction"]["text"], "visible retry")
        self.assertEqual(accepted["hypothesis"], "original")

    def test_correct_as_is_rejects_stale_display_revision(self):
        row = self.live("original")
        stale_revision = row["revision"]
        snapshot = self.coordinator.snapshot_for_retry(row["id"])
        self.coordinator.commit_retry(row["id"], "new", snapshot.expected_revision)
        self.assertIsNone(self.coordinator.correct_as_is(row["id"], expected_revision=stale_revision))

    def test_bounded_attempts_preserve_initial(self):
        row = self.live("first")
        for i in range(25):
            row = self.coordinator.commit_retry(row["id"], str(i), row["revision"])
        row = self.history.get(row["id"])
        self.assertEqual(len(row["attempts"]), 20)
        self.assertEqual(row["attempts"][0]["text"], "first")
        self.assertEqual(row["attempts_omitted"], 6)

    def test_retry_cas_and_correction_wins(self):
        row = self.live(); snapshot = self.coordinator.snapshot_for_retry(row["id"])
        self.assertIsNotNone(self.coordinator.commit_retry(row["id"], "new", snapshot.expected_revision))
        self.assertIsNone(self.coordinator.commit_retry(row["id"], "stale", snapshot.expected_revision))
        latest = self.history.get(row["id"])
        self.coordinator.correct(row["id"], "actual")
        self.assertIsNone(self.coordinator.commit_retry(row["id"], "overwrite", latest["revision"]))
        self.assertEqual(self.history.get(row["id"])["text"], "actual")

    def test_stale_history_update_text_cannot_overwrite_newer_correction(self):
        row = self.history.append("initial", np.zeros(10, np.float32), .1, "m")
        stale = HistoryStore(self.root)
        corrected = self.coordinator.correct(row["id"], "human")
        stale.update_text(row["id"], "stale retry")
        current = self.history.get(row["id"])
        self.assertEqual(current["text"], "human")
        self.assertEqual(current["revision"], corrected["revision"])

    def test_exact_pcm_retry_round_trip_and_digest(self):
        row = self.live(); snapshot = self.coordinator.snapshot_for_retry(row["id"])
        samples, identity = self.history.load_audio_with_identity(row["id"])
        self.assertEqual(identity, snapshot.inference_identity)
        self.assertEqual(identity.sha256, row["audio"]["inference"]["sha256"])
        loaded, wav_identity = read_canonical_wav(snapshot.inference_audio_path)
        self.assertTrue(np.array_equal(samples, loaded)); self.assertEqual(wav_identity, identity)
        self.assertTrue(np.array_equal(snapshot.samples, loaded))

    def test_prepared_live_pcm_is_the_asr_and_persisted_pcm(self):
        prepared = prepare_canonical(np.array([-.99, -.1, .1, .99], np.float32))
        row = self.coordinator.append_live("model text", prepared, .5, "model")
        persisted, identity = self.history.load_audio_with_identity(row["id"])
        self.assertEqual(identity, prepared.identity)
        self.assertEqual(row["audio"]["inference"], {"path": f"audio/{row['id']}.wav", **prepared.identity.as_dict()})
        self.assertTrue(np.array_equal(persisted, prepared.asr_samples))
        self.assertTrue(np.array_equal(self.coordinator.snapshot_for_retry(row["id"]).samples, prepared.asr_samples))

    def test_correction_enrollment_revision_coupling(self):
        row = self.live()
        self.assertIsNone(self.coordinator.enroll(row["id"]))
        corrected = self.coordinator.correct(row["id"], "human")
        enrolled = self.coordinator.enroll(row["id"])
        self.assertEqual(enrolled.history_revision, corrected["revision"])
        self.coordinator.correct(row["id"], "new human")
        self.assertEqual(self.coordinator.learning_status()["active"], 0)

    def test_auto_enrollment_failure_is_pending_and_retryable(self):
        row = self.live()
        with patch.object(self.coordinator, "enroll", side_effect=OSError("full")):
            self.coordinator.correct(row["id"], "human", auto_enroll=True)
        self.assertEqual(self.coordinator.learning_status()["pending_gold"], 1)
        self.assertEqual(self.coordinator.retry_pending_gold(), 1)
        self.assertEqual(self.coordinator.learning_status()["pending_gold"], 0)

    def test_staging_cas_failure_cleanup_and_active_last(self):
        row = self.live(); self.coordinator.correct(row["id"], "human")
        original = self.learning.stage_copy
        def stage_then_mutate(*args, **kwargs):
            stage = original(*args, **kwargs)
            self.coordinator.correct(row["id"], "changed")
            return stage
        with patch.object(self.learning, "stage_copy", stage_then_mutate):
            self.assertIsNone(self.coordinator.enroll(row["id"]))
        self.assertEqual(list(self.learning.staging_dir.iterdir()), [])
        self.assertEqual(self.coordinator.learning_status()["active"], 0)
        self.coordinator.correct(row["id"], "final")
        enrolled = self.coordinator.enroll(row["id"])
        self.assertTrue(enrolled.inference_audio_path.exists())
        self.assertEqual(self.learning.active()[0]["status"], "active")

    def test_interrupted_revoke_recovery_and_invalid_active_demotion(self):
        row = self.live(); self.coordinator.correct(row["id"], "human")
        enrolled = self.coordinator.enroll(row["id"])
        record = self.learning._records[enrolled.sample_id]
        self.learning._records[enrolled.sample_id] = {"schema_version": 1, "sample_id": enrolled.sample_id,
            "history_id": record["history_id"], "status": "revoking", "ts": 0}
        self.learning._save()
        recovered = LearningStore(self.root)
        self.assertEqual(recovered._records[enrolled.sample_id]["status"], "revoked")

    def test_raw_identity_tampering_demotes_active_and_export_exposes_raw(self):
        row = self.live("with raw", raw_samples=np.array([-.5, 0, .5], np.float32), raw_sample_rate=22_050)
        self.coordinator.correct(row["id"], "human")
        enrolled = self.coordinator.enroll(row["id"])
        self.assertIsNotNone(enrolled.raw_audio_path)
        self.assertIsNotNone(enrolled.raw_identity)
        exported = self.coordinator.export_learning_snapshot()[0]
        self.assertEqual(exported.raw_identity, enrolled.raw_identity)
        with open(enrolled.raw_audio_path, "r+b") as handle:
            handle.seek(-1, 2)
            handle.write(b"\x00")
        recovered = LearningStore(self.root)
        self.assertEqual(recovered._records[enrolled.sample_id]["status"], "revoked")

    def test_duplicate_enrollment_returns_the_existing_active_snapshot(self):
        row = self.live(); self.coordinator.correct(row["id"], "human")
        original = self.learning.stage_copy
        barrier = threading.Barrier(2)

        def synchronized_stage(*args, **kwargs):
            stage = original(*args, **kwargs)
            barrier.wait(timeout=2)
            return stage

        with patch.object(self.learning, "stage_copy", synchronized_stage):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first, second = list(pool.map(lambda _: self.coordinator.enroll(row["id"]), range(2)))
        self.assertEqual(first.sample_id, second.sample_id)
        active = self.learning.active_for_history(row["id"])
        self.assertEqual(len(active), 1)
        row = self.live("other"); self.coordinator.correct(row["id"], "truth")
        enrolled = self.coordinator.enroll(row["id"])
        enrolled.inference_audio_path.write_bytes(b"broken")
        recovered = LearningStore(self.root)
        self.assertEqual(recovered._records[enrolled.sample_id]["status"], "revoked")

    def test_plain_history_actions_never_create_the_silver_lane(self):
        # Plain alpha: sotto.run always builds this guard, never a SilverStore.
        AdaptiveLearning(self.root)
        silver = self.root / "adaptive-learning" / "silver"
        row, other = self.live(), self.live("other")
        retried = self.coordinator.commit_retry(row["id"], "retry text", row["revision"])
        self.coordinator.correct(row["id"], "truth", expected_revision=retried["revision"])
        self.assertIsNotNone(self.coordinator.enroll(row["id"]))
        self.coordinator.correct_as_is(other["id"])
        self.coordinator.no_speech(other["id"])
        self.assertEqual(self.coordinator.revoke_history(row["id"]), 1)
        self.assertTrue(self.coordinator.delete(row["id"]))
        self.coordinator.clear()
        self.assertEqual(self.history.entries(99), [])
        # HistoryStore's own empty evidence spool is the only silver-path entry.
        self.assertEqual(sorted(os.listdir(silver)), ["evidence"])
        self.assertEqual(os.listdir(silver / "evidence"), [])

    def test_existing_damaged_silver_lane_still_fails_delete_closed(self):
        import sqlite3
        row = self.live()
        silver = self.root / "adaptive-learning" / "silver"
        (silver / "silver.sqlite3").write_bytes(b"not a database")
        with self.assertRaises(sqlite3.DatabaseError):
            self.coordinator.delete(row["id"])
        self.assertIsNotNone(HistoryStore(self.root).get(row["id"]))

    def test_silver_lane_is_absent_only_when_its_root_is_provably_missing(self):
        import learning
        base = self.root / "probe"
        self.assertFalse(learning.silver_lane_exists(base))
        HistoryStore(base)  # creates only adaptive-learning/silver/evidence
        self.assertFalse(learning.silver_lane_exists(base))
        parent = base / "adaptive-learning"
        silver = parent / "silver"
        for name in ("silver.sqlite3", "silver.sqlite3-wal", "deployment.jsonl", "unknown",
                     "evidence/spooled.wav"):
            (silver / name).write_bytes(b"")
            self.assertTrue(learning.silver_lane_exists(base), name)
            (silver / name).unlink()
        (silver / "evidence" / "comparators").mkdir(mode=0o700)
        self.assertTrue(learning.silver_lane_exists(base))
        (silver / "evidence" / "comparators").rmdir()
        self.assertFalse(learning.silver_lane_exists(base))
        os.chmod(parent, 0)
        try:
            self.assertTrue(learning.silver_lane_exists(base))
        finally:
            os.chmod(parent, 0o700)
        silver.rename(parent / "elsewhere")
        silver.symlink_to(parent / "elsewhere", target_is_directory=True)
        self.assertTrue(learning.silver_lane_exists(base))
        silver.unlink()
        shutil.rmtree(parent)
        parent.write_bytes(b"")
        self.assertTrue(learning.silver_lane_exists(base))

    def test_clear_refuses_when_the_silver_lane_appears_mid_clear(self):
        from silver_store import SilverStore
        row = self.live()
        original = AdaptiveLearning.clear_personal_state

        def racing(guard):
            original(guard)
            SilverStore(self.root)

        with patch.object(AdaptiveLearning, "clear_personal_state", racing):
            with self.assertRaises(RuntimeError):
                self.coordinator.clear()
        self.assertIsNotNone(HistoryStore(self.root).get(row["id"]))
        self.coordinator.clear()
        self.assertEqual(HistoryStore(self.root).entries(99), [])

    def test_delete_clear_failure_ordering(self):
        row = self.live(); self.coordinator.correct(row["id"], "truth")
        enrolled = self.coordinator.enroll(row["id"])
        with patch.object(self.learning, "revoke", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError): self.coordinator.delete(row["id"])
        self.assertIsNotNone(self.history.get(row["id"]))
        self.coordinator.delete(row["id"])
        self.assertIsNone(self.history.get(row["id"]))
        a, b = self.live("a"), self.live("b")
        self.coordinator.clear(); self.assertEqual(self.history.entries(99), [])

    def test_retention_prune_is_serialized_and_keeps_learning(self):
        h = HistoryStore(self.root / "small", keep=1); l = LearningStore(self.root / "small")
        c = LearningCoordinator(h, l)
        one = c.append_live("one", np.zeros(10, np.float32), .1, "m")
        c.correct(one["id"], "truth"); enrolled = c.enroll(one["id"])
        c.append_live("two", np.zeros(10, np.float32), .1, "m")
        self.assertIsNone(h.get(one["id"]))
        self.assertEqual(c.learning_status()["active"], 1)
        self.assertTrue(enrolled.inference_audio_path.exists())
        restarted_history = HistoryStore(self.root / "small", keep=1)
        restarted_learning = LearningStore(self.root / "small")
        restarted = LearningCoordinator(restarted_history, restarted_learning)
        self.assertEqual(restarted.learning_status()["active"], 1)
        restored = restarted.export_learning_snapshot()[0]
        self.assertEqual(restored.sample_id, enrolled.sample_id)
        self.assertTrue(restored.inference_audio_path.exists())

    def test_pruned_sample_is_visible_to_admin_and_admin_revoke_tombstones_it(self):
        base = self.root / "admin-pruned"
        history = HistoryStore(base, keep=1); learning = LearningStore(base)
        coordinator = LearningCoordinator(history, learning)
        first = coordinator.append_live("one", np.zeros(10, np.float32), .1, "m")
        coordinator.correct(first["id"], "human")
        enrolled = coordinator.enroll(first["id"])
        coordinator.append_live("two", np.zeros(10, np.float32), .1, "m")
        self.assertIsNone(history.get(first["id"]))
        self.assertEqual(LearningStore.admin_snapshot(base)[0].sample_id, enrolled.sample_id)
        self.assertTrue(LearningStore.admin_revoke(enrolled.sample_id, base))
        self.assertEqual(LearningStore.admin_snapshot(base), [])
        self.assertFalse(enrolled.inference_audio_path.exists())
        tombstone = LearningStore(base)._records[enrolled.sample_id]
        self.assertEqual(tombstone["status"], "revoked")
        self.assertEqual(set(tombstone), {"schema_version", "sample_id", "history_id", "status", "ts"})
        self.assertFalse(LearningStore.admin_revoke(enrolled.sample_id, base))

    def test_clear_revokes_active_samples_pruned_from_history(self):
        base = self.root / "clear-pruned"
        history = HistoryStore(base, keep=1); learning = LearningStore(base)
        coordinator = LearningCoordinator(history, learning)
        first = coordinator.append_live("one", np.zeros(10, np.float32), .1, "m")
        coordinator.correct(first["id"], "human")
        enrolled = coordinator.enroll(first["id"])
        coordinator.append_live("two", np.zeros(10, np.float32), .1, "m")
        self.assertIsNone(history.get(first["id"]))
        coordinator.clear()
        self.assertEqual(coordinator.learning_status()["active"], 0)
        self.assertEqual(history.entries(99), [])
        self.assertFalse(enrolled.inference_audio_path.exists())

    def test_revoke_history_reloads_and_revokes_newer_matching_enrollment(self):
        base = self.root / "stale-revoke"
        fresh_history, fresh_learning = HistoryStore(base), LearningStore(base)
        fresh = LearningCoordinator(fresh_history, fresh_learning)
        stale = LearningCoordinator(HistoryStore(base), LearningStore(base))
        row = fresh.append_live("one", np.zeros(10, np.float32), .1, "m")
        fresh.correct(row["id"], "human")
        enrolled = fresh.enroll(row["id"])
        self.assertEqual(stale.revoke_history(row["id"]), 1)
        self.assertEqual(LearningStore.admin_snapshot(base), [])
        self.assertFalse(enrolled.inference_audio_path.exists())

    def test_stale_recovery_reloads_before_orphan_cleanup(self):
        base = self.root / "stale-recovery"
        history, learning = HistoryStore(base), LearningStore(base)
        coordinator = LearningCoordinator(history, learning)
        stale = LearningStore(base)
        row = coordinator.append_live("one", np.zeros(10, np.float32), .1, "m")
        coordinator.correct(row["id"], "human")
        enrolled = coordinator.enroll(row["id"])
        stale.recover_filesystem()
        restored = LearningStore(base)
        self.assertEqual(restored.active()[0]["sample_id"], enrolled.sample_id)
        self.assertTrue(enrolled.inference_audio_path.exists())

    def test_publish_save_failure_restores_memory_and_removes_artifacts(self):
        row = self.live(); self.coordinator.correct(row["id"], "human")
        with patch.object(self.learning, "_save", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                self.coordinator.enroll(row["id"])
        self.assertEqual(self.learning.active(), [])
        self.assertEqual(list(self.learning.audio_dir.glob("*.wav")), [])
        self.assertEqual(list(self.learning.staging_dir.iterdir()), [])
        self.assertEqual(LearningStore(self.root).active(), [])

    def test_revoke_first_save_failure_restores_active_memory_and_artifacts(self):
        row = self.live(); self.coordinator.correct(row["id"], "human")
        enrolled = self.coordinator.enroll(row["id"])
        with patch.object(self.learning, "_save", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                self.learning.revoke(enrolled.sample_id)
        self.assertEqual(self.learning._records[enrolled.sample_id]["status"], "active")
        self.assertTrue(enrolled.inference_audio_path.exists())
        self.assertEqual(LearningStore(self.root).active()[0]["sample_id"], enrolled.sample_id)

    def test_revoke_final_save_failure_keeps_recoverable_revoking_tombstone(self):
        row = self.live(); self.coordinator.correct(row["id"], "human")
        enrolled = self.coordinator.enroll(row["id"])
        original = self.learning._save
        calls = 0

        def fail_second_save():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected")
            original()

        with patch.object(self.learning, "_save", side_effect=fail_second_save):
            with self.assertRaises(OSError):
                self.learning.revoke(enrolled.sample_id)
        self.assertEqual(self.learning._records[enrolled.sample_id]["status"], "revoking")
        self.assertFalse(enrolled.inference_audio_path.exists())
        recovered = LearningStore(self.root)
        self.assertEqual(recovered._records[enrolled.sample_id]["status"], "revoked")

    def test_atomic_jsonl_failure_cleans_unique_temporary(self):
        target = self.root / "history.jsonl"
        with patch("history.os.replace", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                _atomic_jsonl(target, [{"id": "safe"}])
        self.assertEqual(list(self.root.glob(".history.jsonl.*.tmp")), [])

    def test_startup_sweeps_only_known_jsonl_temporaries(self):
        known_history = self.root / ".history.jsonl.interrupted.tmp"
        unrelated_history = self.root / ".unrelated.tmp"
        known_history.write_bytes(b"x"); unrelated_history.write_bytes(b"x")
        HistoryStore(self.root)
        self.assertFalse(known_history.exists())
        self.assertTrue(unrelated_history.exists())
        learning_base = self.root / "learning"
        known_learning = learning_base / ".learning.jsonl.interrupted.tmp"
        unrelated_learning = learning_base / ".unrelated.tmp"
        known_learning.write_bytes(b"x"); unrelated_learning.write_bytes(b"x")
        LearningStore(self.root)
        self.assertFalse(known_learning.exists())
        self.assertTrue(unrelated_learning.exists())

    def test_admin_snapshot_waits_for_live_lock_and_does_not_recover_stages(self):
        row = self.live(); self.coordinator.correct(row["id"], "human")
        self.coordinator.enroll(row["id"])
        sentinel = self.learning.staging_dir / "admin-sentinel"
        sentinel.mkdir()
        context = multiprocessing.get_context("spawn")
        ready, release, result = context.Event(), context.Event(), context.Queue()
        holder = context.Process(target=_hold_advisory_lock, args=(str(self.root), ready, release))
        reader = context.Process(target=_read_admin_snapshot, args=(str(self.root), result))
        holder.start(); self.assertTrue(ready.wait(3))
        reader.start()
        time.sleep(.15)
        self.assertTrue(reader.is_alive())
        self.assertTrue(sentinel.exists())
        release.set()
        self.assertEqual(result.get(timeout=3), 1)
        holder.join(3); reader.join(3)
        self.assertEqual(holder.exitcode, 0); self.assertEqual(reader.exitcode, 0)
        self.assertTrue(sentinel.exists())

    def test_admin_export_guard_holds_lock_until_copy_finishes(self):
        row = self.live(); self.coordinator.correct(row["id"], "human")
        self.coordinator.enroll(row["id"])
        context = multiprocessing.get_context("spawn")
        acquired = context.Event()
        copied = self.root / "export-copy.wav"
        with LearningStore.admin_export_guard(self.root) as snapshots:
            probe = context.Process(target=_probe_advisory_lock, args=(str(self.root), acquired))
            probe.start()
            self.assertFalse(acquired.wait(.15))
            shutil.copyfile(snapshots[0].inference_audio_path, copied)
            self.assertTrue(copied.exists())
        self.assertTrue(acquired.wait(3))
        probe.join(3)
        self.assertEqual(probe.exitcode, 0)

    def test_malformed_or_duplicate_metadata_fences_before_artifact_sweep(self):
        row=self.live()
        wav=self.history.audio_path(row["id"])
        self.assertTrue(wav.exists())
        # A parse failure must not be treated as an empty index followed by an
        # orphan sweep that destroys the only recovery artifact.
        self.history.index.write_text('{broken\n',encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError,"malformed"):
            HistoryStore(self.root)
        self.assertTrue(wav.exists())
        # Duplicate ids are an equally ambiguous authority state.
        record={"id":"duplicate","text":"x"}
        self.history.index.write_text(json.dumps(record)+"\n"+json.dumps(record)+"\n",encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError,"duplicate"):
            HistoryStore(self.root)
        self.assertTrue(wav.exists())

    def _unreadable_index(self, root: Path, bad_line: str) -> tuple[Path, Path, bytes]:
        """One real dictation (row + WAV), then an unreadable line after it."""
        coordinator = LearningCoordinator(HistoryStore(root), LearningStore(root))
        row = coordinator.append_live("kept words", np.array([-1.0, 0, 1.0], np.float32), .5, "model")
        index = coordinator.history.index
        index.write_text(index.read_text(encoding="utf-8") + bad_line + "\n", encoding="utf-8")
        return index, coordinator.history.audio_path(row["id"]), index.read_bytes()

    def test_unreadable_history_opens_read_only_and_keeps_every_byte(self):
        """F16b: a damaged or newer-schema history.jsonl must not stop the app
        launching, and nothing may rewrite, move or sweep the user's data."""
        from history import HistoryUnreadable
        cases = {"truncated row": ('{"id": "abc123", "text": "hi"', False),
                 "newer schema": (json.dumps({"id": "abc123", "schema_version": 99, "text": "hi"}), True)}
        for name, (bad_line, newer) in cases.items():
            with self.subTest(name):
                root = self.root / name.replace(" ", "-")
                index, wav, before = self._unreadable_index(root, bad_line)
                store = HistoryStore(root, tolerate_unreadable=True)
                self.assertIsInstance(store.unreadable, HistoryUnreadable)
                self.assertEqual(store.unreadable.newer_schema, newer)
                self.assertEqual(store.entries(), [])
                # Every mutation still re-reads the index first, so all of them refuse.
                with self.assertRaises(HistoryUnreadable):
                    store.append("new", np.zeros(8, np.float32), .5, "model")
                with self.assertRaises(HistoryUnreadable):
                    LearningCoordinator(store, LearningStore(root))
                self.assertEqual(index.read_bytes(), before)
                self.assertEqual([path.name for path in store.audio_dir.iterdir()], [wav.name])
                # Without the opt-in (worker, CLI, adaptive lane) it stays fatal.
                with self.assertRaises(HistoryUnreadable):
                    HistoryStore(root)

    def test_app_coordinator_keeps_dictation_working_without_saving(self):
        """F16b: the app gets a stand-in coordinator that hands dictation back
        unsaved and refuses every History action, never touching the file."""
        import sotto
        from history import HistoryUnreadable
        root = self.root / "app"
        index, wav, before = self._unreadable_index(root, '{"id": "abc123", "text": "hi"')
        store = HistoryStore(root, tolerate_unreadable=True)
        coordinator = sotto.history_coordinator(store, LearningStore(root))
        self.assertIsInstance(coordinator, sotto.UnsavedHistory)
        row = coordinator.append_live("please call me back", prepare_canonical(np.zeros(1600, np.float32)),
                                      .1, "model", ts=5.0, provenance="live", adaptive=False)
        self.assertEqual((row["text"], row["saved"]), ("please call me back", False))
        self.assertEqual(sotto._active_learning_history_ids(coordinator), set())
        for action in (lambda: coordinator.clear(), lambda: coordinator.snapshot_for_retry(row["id"])):
            with self.assertRaises(HistoryUnreadable):
                action()
        self.assertEqual(index.read_bytes(), before)
        self.assertEqual([path.name for path in store.audio_dir.iterdir()], [wav.name])
        message = sotto.history_unreadable_message(store)
        self.assertIn(str(index), message)
        self.assertIn("not saved to History", message)
        # Once the file reads again, the next launch gets the real coordinator.
        index.write_text(before.decode("utf-8").splitlines()[0] + "\n", encoding="utf-8")
        store = HistoryStore(root, tolerate_unreadable=True)
        self.assertIsNone(store.unreadable)
        self.assertEqual([entry["text"] for entry in store.entries()], ["kept words"])
        self.assertNotIsInstance(sotto.history_coordinator(store, LearningStore(root)), sotto.UnsavedHistory)

    def test_unicode_line_separators_never_split_a_row(self):
        """N4: str.splitlines() also breaks on U+2028, U+2029 and U+0085. One
        transcript or correction containing them used to make History and
        learning.jsonl unreadable for good, including files already written
        with the character raw by an earlier version."""
        for char in (" ", " ", "\u0085"):
            with self.subTest(f"U+{ord(char):04X}"):
                root = self.root / f"separator-{ord(char):04x}"
                coordinator = LearningCoordinator(HistoryStore(root), LearningStore(root))
                row = coordinator.append_live(f"first{char}second", np.array([-1.0, 0, 1.0], np.float32),
                                              .5, "model")
                coordinator.correct(row["id"], f"fixed{char}text")
                self.assertIsNotNone(coordinator.enroll(row["id"]))
                indexes = (coordinator.history.index, coordinator.learning.index)
                for index in indexes:
                    self.assertNotIn(char, index.read_text(encoding="utf-8"), f"{index.name} is written escaped")
                    # An earlier version wrote the character raw; that file must read too.
                    escaped = json.dumps(char)[1:-1]
                    index.write_text(index.read_text(encoding="utf-8").replace(escaped, char), encoding="utf-8")
                    self.assertIn(char, index.read_text(encoding="utf-8"))
                reopened = HistoryStore(root)
                self.assertEqual(reopened.get(row["id"])["hypothesis"], f"first{char}second")
                learning = LearningStore(root)
                self.assertEqual([record["corrected_text"] for record in learning.active()], [f"fixed{char}text"])

    def _rewrite_row(self, store: HistoryStore, entry_id: str, **fields) -> None:
        rows = [json.loads(line) for line in store.index.read_text(encoding="utf-8").split("\n") if line]
        for row in rows:
            if row["id"] == entry_id:
                row.update(fields)
        store.index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_orphan_sweep_keeps_recordings_of_rows_with_unusable_audio_metadata(self):
        """F16c: a v2 row whose "audio" is null (or names no usable path) is
        still a row; startup used to sweep its recordings as orphans while
        History reported itself readable."""
        cases = {"audio null": None, "audio a list": [],
                 "inference null": {"inference": None, "raw": None},
                 "foreign path": {"inference": {"path": "audio/someone-else.wav"}, "raw": {"path": "x"}}}
        for name, audio in cases.items():
            with self.subTest(name):
                root = self.root / name.replace(" ", "-")
                store = HistoryStore(root)
                row = LearningCoordinator(store, LearningStore(root)).append_live(
                    "kept words", np.array([-1.0, 0, 1.0], np.float32), .5, "model",
                    raw_samples=np.array([-.5, .5], np.float32), raw_sample_rate=22_050)
                wav, raw = store.audio_path(row["id"]), store.raw_audio_path(row["id"])
                self.assertTrue(wav.exists() and raw.exists())
                orphan = store.audio_dir / "0123456789ab.wav"
                orphan.write_bytes(b"orphan")
                self._rewrite_row(store, row["id"], audio=audio)
                reopened = HistoryStore(root)
                self.assertIsNone(reopened.unreadable)
                self.assertEqual([entry["id"] for entry in reopened.entries()], [row["id"]])
                self.assertTrue(wav.exists(), "the row's recording survives the sweep")
                self.assertTrue(raw.exists(), "the row's raw recording survives the sweep")
                self.assertFalse(orphan.exists(), "a recording no row names is still swept")

    def test_orphan_sweep_keeps_silver_evidence_when_a_spooled_row_names_none(self):
        """F16c: a spooled row whose audio metadata is unusable could own any
        file in the Silver spool, so none of them is provably orphaned."""
        store = HistoryStore(self.root / "spooled")
        evidence = store.silver_evidence_dir / f"{'a' * 64}.wav"
        evidence.write_bytes(b"evidence")
        row = {"schema_version": 2, "id": "spooled00001", "revision": 0, "correction": None,
               "audio": None, "adaptive": True, "language": "en", "silver_spooled": True,
               "silver_enqueue": {"history_revision": 0, "state": "queued"}}
        store.index.write_text(json.dumps(row) + "\n", encoding="utf-8")
        HistoryStore(store.base_dir)
        self.assertTrue(evidence.exists())

    def test_a_failed_history_save_leaves_no_phantom_row(self):
        """N29: append_live used to add the row in memory before saving, so a
        failed save left a row in entries() that was never on disk."""
        kept = self.live("kept")
        with patch("history._atomic_jsonl", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.live("never saved")
        self.assertEqual([entry["id"] for entry in self.history.entries()], [kept["id"]])

if __name__ == "__main__":
    unittest.main()

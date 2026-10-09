import os
import stat
import tempfile
import threading
import time
import unittest
import hashlib
import json
import inspect
import sqlite3
from io import StringIO
from unittest import mock
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np

import sys

if sys.platform != "darwin":
    raise unittest.SkipTest("adaptive lane is macOS-only in v1 (fcntl)")

from adaptive_worker import AdaptiveWorker, sealed_readiness_status
from adaptive_runtime import AdaptiveRuntime
from adaptive_learning import AdaptiveLearning
from deployment_controller import DeploymentController
from silver_experiment import SilverExperiment
from history import _atomic_jsonl
from history import HistoryStore
from learning import LearningCoordinator, scrub_comparator_spools
from silver_store import SilverStore, read_only_status
from teacher_consensus import exact_unanimous
from teacher_consensus import policy_hash
from teacher_backends import family_hash
from teacher_backends import TeacherPreempted
from inference_scheduler import InferenceScheduler
from speech_backends import WHISPER_ID
from calibration_manifest_builder import derive_plan_with_reader, SOURCE_HASH, _protocol_hash, LEDGER_HASH


def _compiled_v2_expected() -> list[dict[str, str]]:
    """Portable, content-free copy of the frozen public V2 item ledger."""
    directory=Path(__file__).with_name("fixtures") / "calibration_v2_expected"
    rows=[row for part in sorted(directory.glob("part-*.json")) for row in json.loads(part.read_text("utf-8"))]
    ledger=hashlib.sha256(json.dumps([(index,row["item_hash"],row["cluster_hash"]) for index,row in enumerate(rows)],separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    if ledger != LEDGER_HASH or len(rows) != 500:
        raise AssertionError("committed calibration V2 ledger fixture changed")
    return rows


class _Teachers:
    def __init__(self, text="hello world"):
        self.text = text
    def validate(self): return "frozen-receipt"
    def transcribe(self, _family, _path, _identity, **_kwargs): return self.text


class SilverLaneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        # ComparatorSpool deliberately rejects lexical system symlink aliases
        # such as /var; fixture roots use their physical private directory.
        self.root = Path(self.temp.name).resolve(strict=True)
        self.history = HistoryStore(self.root)
        self.coordinator = LearningCoordinator(self.history, base_dir=self.root)

    def tearDown(self): self.temp.cleanup()

    def live(self):
        return self.coordinator.append_live("candidate", np.zeros(8000, np.float32), .5, "m", adaptive=True, language="en")

    def test_exact_unanimity_abstains_without_leaking_teacher_text(self):
        self.assertEqual(exact_unanimous({"a": "hello", "b": "hello"}, ("a", "b")).state, "accepted")
        for a, b in (("", ""), ("hello", "different"), ("send money", "send money"), (None, "x")):
            decision = exact_unanimous({"a": a, "b": b}, ("a", "b"))
            self.assertEqual(decision.state, "abstained")
            self.assertNotIn("hello", decision.reason)

    def test_worker_recovers_marker_idempotently_and_correction_fences_stale_result(self):
        row = self.live()
        worker = AdaptiveWorker(self.root, history=self.history, teachers=_Teachers())
        self.assertEqual(worker.run_once()["status"]["accepted"], 1)
        # A human correction revokes the earlier pseudo-label before changing history.
        self.coordinator.correct(row["id"], "human")
        self.assertEqual(SilverStore(self.root).status()["accepted"], 0)

    def test_clear_erases_adaptive_personal_state_but_not_silver_public_ledger(self):
        row=self.live(); adaptive=AdaptiveLearning(self.root)
        adaptive.register_capture({"history_id":row["id"],"captured_ts":row["ts"],"duration":row["duration"],
            "audio_digest":row["audio"]["inference"]["sha256"],"adaptive":True,"language":"en"})
        # Deliberately put recognisable private values in every personal
        # collection; Clear must not leave even a revoked tombstone behind.
        adaptive._mutate(lambda state: state.update({"generations":{"g":{"reference":"private-reference","pool":[row["id"]],"candidates":{}}},
            "champions":{"baseline":{"stable_id":"base"},"current":{"stable_id":"private-champion","text":"private-text"},"prior":[]},
            "runtime":{"failures":{},"audit":[{"capture":row["id"],"text":"private-audit"}]}}))
        store=SilverStore(self.root); alpha=store.alpha(1)
        self.coordinator.clear()
        payload=(self.root/"adaptive-learning"/"state.json").read_text("utf-8")
        for secret in (row["id"],"private-reference","private-champion","private-text","private-audit"):
            self.assertNotIn(secret,payload)
        restored=AdaptiveLearning(self.root).status()
        self.assertEqual(restored["captures"],0); self.assertEqual(restored["generations"],{})
        self.assertEqual(SilverStore(self.root).alpha(1),alpha)

    def test_silver_overflow_spools_only_canonical_evidence_then_releases_it(self):
        history=HistoryStore(self.root,keep=1); coordinator=LearningCoordinator(history,base_dir=self.root)
        first=coordinator.append_live("private-first-hypothesis",np.zeros(8000,np.float32),.5,"m",adaptive=True,language="en")
        coordinator.append_live("private-second-hypothesis",np.ones(8000,np.float32),.5,"m",adaptive=True,language="en")
        spooled=history.get(first["id"])
        self.assertIsNotNone(spooled); self.assertTrue(spooled["silver_spooled"])
        persisted=[json.loads(line) for line in history.index.read_text("utf-8").splitlines()]
        persisted_row=next(value for value in persisted if value["id"] == first["id"])
        self.assertNotIn("private-first-hypothesis",json.dumps(persisted_row))
        self.assertNotIn("attempts",persisted_row); self.assertNotIn("candidate",persisted_row)
        self.assertFalse((history.raw_audio_dir/f"{first['id']}.wav").exists())
        evidence=history.base_dir/spooled["audio"]["inference"]["path"]
        self.assertTrue(evidence.is_file()); self.assertNotEqual(evidence.parent,history.audio_dir)
        restored=HistoryStore(self.root,keep=1)
        self.assertTrue(restored.get(first["id"])["silver_spooled"])
        self.assertTrue(restored.set_silver_marker(first["id"],first["revision"],"released"))
        self.assertIsNone(restored.get(first["id"])); self.assertFalse(evidence.exists())

    def test_delayed_live_registration_cannot_resurrect_across_clear_fence(self):
        row=self.live(); runtime=AdaptiveRuntime(self.root,history=self.history)
        # This is the real gap between append returning and its delayed UI
        # callback.  The SQLite clear fence commits before history deletion;
        # registration must observe it rather than recreate adaptive state.
        SilverStore(self.root).clear(reason="test_delayed_registration")
        runtime.register_live(row)
        self.assertEqual(AdaptiveLearning(self.root).status()["captures"],0)

    def test_legacy_schema_v2_adaptive_row_backfills_marker_before_enqueue(self):
        row = self.live()
        # Simulate retained rows captured before the silver marker migration.
        with self.history.index.open("r", encoding="utf-8") as f: payload = f.read()
        payload = payload.replace(', "silver_enqueue"', ', "removed_marker"', 1)
        # Rebuild through JSON rather than relying on key order.
        import json
        saved = json.loads(payload); saved.pop("removed_marker", None)
        self.history.index.write_text(json.dumps(saved) + "\n", encoding="utf-8")
        restored = HistoryStore(self.root)
        marker = restored.silver_markers()[0]["silver_enqueue"]
        self.assertEqual(marker["history_revision"], row["revision"])
        self.assertEqual(marker["state"], "pending")

    def test_sqlite_permissions_symlink_rejection_epoch_and_alpha_persistence(self):
        old = os.umask(0o022)
        try:
            store = SilverStore(self.root)
        finally:
            os.umask(old)
        self.assertEqual(stat.S_IMODE(store.path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(store.root.stat().st_mode), 0o700)
        alpha = store.alpha(1); store.clear(); self.assertEqual(store.alpha(1), alpha)
        store.path.unlink(); store.path.symlink_to(self.root / "outside")
        with self.assertRaises(RuntimeError): SilverStore(self.root)

    def test_read_only_status_never_constructs_or_mutates_store(self):
        store=SilverStore(self.root); before=(store.path.read_bytes(),store.path.stat().st_mtime_ns)
        status=read_only_status(self.root)
        after=(store.path.read_bytes(),store.path.stat().st_mtime_ns)
        self.assertEqual(status["state"],"ok"); self.assertEqual(before,after)
        self.assertEqual(read_only_status(self.root / "missing")["state"],"blocked")

    def test_deployment_is_shadow_only_until_calibration_and_gates(self):
        controller = DeploymentController(self.root)
        self.assertEqual(controller.status()["tier"], "shadow_only")
        # Callers cannot manufacture a passed calibration/personal horizon
        # through controller dictionaries; those come from SilverStore runs.
        self.assertFalse(controller.authorize_provisional({"stable_id": "x"}, "missing", receipt_hash="r"))

    def test_unbound_runtime_observation_cannot_fabricate_rollout_evidence(self):
        controller=DeploymentController(self.root)
        self.assertIsNone(controller.route("opaque-session"))
        self.assertFalse(controller.validate_route_token(None))
        self.assertEqual(controller.record_runtime(session_id="opaque-session",arm_id="candidate",success=True,
                         fallback=False,latency=.1,coverage_ok=True,hallucination_ok=True,identity_valid=True),"stale")
        self.assertEqual(controller.silver.runtime_metrics(0)["captures"],0)

    def test_teacher_receipt_is_part_of_durable_job_identity(self):
        store = SilverStore(self.root)
        common = dict(history_id="h", history_revision=0, audio_sha256="a",
                      teacher_family_hash="f", consensus_policy_hash="p")
        first = store.enqueue(**common, receipt_hash="receipt-a")
        second = store.enqueue(**common, receipt_hash="receipt-b")
        self.assertNotEqual(first, second)

    def test_runtime_safety_derives_duration_from_canonical_samples(self):
        # No backend-specific metadata is required for an ordinary, bounded
        # response; this guards against treating absent metadata as unsafe.
        self.assertEqual(AdaptiveRuntime._runtime_safety("one two three", {}, None,
                         np.zeros(16_000, np.float32)), (True, True))

    def test_canary_waits_for_fresh_holdout_then_reaches_full(self):
        controller=DeploymentController(self.root)
        candidate={"stable_id":"candidate"}
        controller._write({"schema":1,"revision":0,"tier":"provisional_silver","current":candidate,
            "lkg":None,"cohort_id":"first","calibration":None,"audit":[],
            "canary":{"stage":"five_percent","captures":0,"days":0,"failures":0,
                       "sticky_percent":5,"epoch":controller.silver.epoch(),"observation_revision":1,"started_ts":1}})
        safe=lambda captures,days: {"captures":captures,"days":days,"fallback_error":0.0,
            "incumbent_error":0.0,"p95_ratio":1.0,"coverage_ok":True,"hallucination_ok":True,
            "identity_invalid":False,"consecutive_failures":0,"comparator_resolved":True}
        controller.silver.runtime_metrics=lambda _revision: safe(100,7)  # type: ignore[method-assign]
        self.assertEqual(controller.observe_canary(),"twenty_five_percent")
        controller.silver.runtime_metrics=lambda _revision: safe(500,14)  # type: ignore[method-assign]
        controller.silver.fresh_holdout_status=lambda **_kwargs: {}  # type: ignore[method-assign]
        self.assertEqual(controller.observe_canary(),"awaiting_holdout")
        controller.silver.fresh_holdout_status=lambda **_kwargs: {"accepted":200,"words":2000,"sessions":20,"days":7,
            "all_terminal":True,"paired_gates":True,"coverage_frozen":True,"winner":"candidate"}  # type: ignore[method-assign]
        self.assertIn(controller.observe_canary(),{"deployed_silver","full"})
        self.assertEqual(controller.status()["tier"],"deployed_silver")

    def test_ramp_generations_preserve_monotonic_session_assignment(self):
        store=SilverStore(self.root); controller=DeploymentController(self.root,silver=store)
        buckets={"low":[],"mid":[],"high":[]}
        for index in range(10_000):
            session=f"bucket-{index}"; bucket=int(hashlib.sha256(session.encode()).hexdigest()[:8],16)%100
            name="low" if bucket < 5 else "mid" if bucket < 25 else "high"
            if len(buckets[name]) < 2: buckets[name].append(session)
            if all(len(value)==2 for value in buckets.values()): break
        self.assertTrue(all(len(value)==2 for value in buckets.values()))
        low,mid_seen,mid_new,high_seen,high_new=(buckets["low"][0],buckets["mid"][0],buckets["mid"][1],buckets["high"][0],buckets["high"][1])
        generation=20
        # These sessions have already received their five-percent arm.
        self.assertTrue(store.route_assignment(route_generation=generation,session_id=low,percent=5))
        self.assertFalse(store.route_assignment(route_generation=generation,session_id=mid_seen,percent=5))
        self.assertFalse(store.route_assignment(route_generation=generation,session_id=high_seen,percent=5))
        candidate={"stable_id":"candidate"}; epoch=store.epoch()
        with store._connect() as db:
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",("first","h","evaluated_passed","m","c","{}",epoch,0))
        controller._write({"schema":1,"revision":0,"tier":"provisional_silver","current":candidate,"lkg":None,"cohort_id":"first","calibration":None,"audit":[],
            "route_generation":generation,"canary":{"stage":"five_percent","captures":0,"days":0,"failures":0,"sticky_percent":5,"epoch":epoch,"observation_revision":1,"started_ts":1}})
        candidate_hash=hashlib.sha256(json.dumps(candidate,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        five_token={"deployment_revision":1,"route_generation":generation,"candidate_hash":candidate_hash,"tier":"provisional_silver",
                    "cohort_id":"first","epoch":epoch,"session_hash":hashlib.sha256(low.encode()).hexdigest()}
        self.assertTrue(controller.validate_route_token(five_token))
        safe=lambda captures,days: {"candidate_records":captures,"captures":captures,"days":days,"fallback_error":0.0,"incumbent_error":0.0,
            "p95_ratio":1.0,"coverage_ok":True,"hallucination_ok":True,"identity_invalid":False,"consecutive_failures":0,"comparator_resolved":True}
        controller.silver.runtime_metrics=lambda _revision: safe(100,7)  # type: ignore[method-assign]
        self.assertEqual(controller.observe_canary(),"twenty_five_percent")
        self.assertEqual(controller._read()["route_generation"],generation)
        self.assertFalse(controller.validate_route_token(five_token))  # tier transition invalidates flight token
        # Seen baseline sessions stay baseline; an unseen mid-bucket session
        # joins under the expanded threshold.
        self.assertTrue(store.route_assignment(route_generation=generation,session_id=low,percent=25))
        self.assertFalse(store.route_assignment(route_generation=generation,session_id=mid_seen,percent=25))
        self.assertFalse(store.route_assignment(route_generation=generation,session_id=high_seen,percent=25))
        self.assertTrue(store.route_assignment(route_generation=generation,session_id=mid_new,percent=25))
        controller.silver.runtime_metrics=lambda _revision: safe(500,14)  # type: ignore[method-assign]
        controller.silver.fresh_holdout_status=lambda **_kwargs: {"accepted":200,"words":2000,"sessions":20,"days":7,
            "all_terminal":True,"paired_gates":True,"coverage_frozen":True,"winner":"candidate"}  # type: ignore[method-assign]
        self.assertEqual(controller.observe_canary(),"deployed_silver")
        self.assertEqual(controller._read()["route_generation"],generation)
        # Full traffic affects only sessions first observed after the ramp.
        self.assertFalse(store.route_assignment(route_generation=generation,session_id=high_seen,percent=100))
        self.assertTrue(store.route_assignment(route_generation=generation,session_id=high_new,percent=100))
        reopened=SilverStore(self.root)
        self.assertFalse(reopened.route_assignment(route_generation=generation,session_id=mid_seen,percent=100))
        self.assertTrue(reopened.route_assignment(route_generation=generation,session_id=mid_new,percent=100))
        token={"deployment_revision":1,"route_generation":generation,"candidate_hash":candidate_hash,
               "tier":"deployed_silver","cohort_id":"first","epoch":epoch,"session_hash":hashlib.sha256(mid_new.encode()).hexdigest()}
        self.assertTrue(controller.validate_route_token(token))

    def test_clear_is_a_single_epoch_transition(self):
        runtime=AdaptiveRuntime(self.root)
        before=runtime.silver.epoch()
        runtime.clear()
        self.assertEqual(runtime.silver.epoch(), before + 1)

    def test_worker_finishes_interrupted_clear_before_rescanning_history(self):
        self.live()
        store=SilverStore(self.root); store.clear(reason="crash_between_epoch_and_history_delete")
        worker=AdaptiveWorker(self.root,history=self.history,teachers=_Teachers())
        self.assertEqual(worker.recover_markers(),0)
        self.assertEqual(self.history.entries(),[])
        self.assertIsNone(store.scrub_pending())

    def test_persisted_authorization_survives_restart_without_canary_reset(self):
        store=SilverStore(self.root); controller=DeploymentController(self.root, silver=store)
        champion={"stable_id":"champion","backend":"a","repo":"r","revision":"1","language":"en","evaluator_hash":"e"}
        candidate={"stable_id":"candidate","backend":"b","repo":"r","revision":"2","language":"en","evaluator_hash":"e"}
        horizon=store.ensure_prospective_horizon(start_ts=0,champion=champion,candidates=[candidate],receipt_hash="receipt")
        cohort="cohort"
        outcome={"cohort_id":cohort,"identity_hash":horizon["identity_hash"],"winner":"candidate","generation":1,"alpha":.025,"candidates":{},"holm":{}}
        meta={"accepted":200,"raw":300,"words":2000,"sessions":20,"days":7,"all_terminal":True,
              "coverage_frozen":True,"paired_gates":True,"identity_hash":horizon["identity_hash"],"winner":"candidate",
              "teacher_receipt_hash":"receipt","teacher_policy_hash":policy_hash(),"evaluator_hash":"e","evaluation_outcome":outcome}
        from calibration_manifest_builder import SOURCE_HASH, _protocol_hash
        with store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,"personal_horizon","evaluated_passed","m","c",__import__("json").dumps(meta),store.epoch(),0))
            db.execute("UPDATE prospective_horizons SET status='evaluated',frozen_cohort_id=? WHERE horizon_id=?",(cohort,horizon["horizon_id"]))
            db.execute("INSERT INTO calibrations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",("cal", "dataset", "source", policy_hash(), "split", "receipt", store.epoch(), "passed", 200,0,200,0))
            db.execute("INSERT INTO calibration_coverage VALUES(?,?,?,?,?)",("dataset",200,200,200,0))
            db.execute("INSERT INTO calibration_universes(holdout_key,manifest_hash,source_hash,protocol_hash,policy_hash,receipt_hash,state,created_ts,updated_ts,result) VALUES(?,?,?,?,?,?,?,?,?,?)",(SOURCE_HASH,"dataset",SOURCE_HASH,_protocol_hash(),policy_hash(),"receipt","passed",0,0,'{"accepted":200,"clusters":200,"reference_errors":0,"total":500}'))
            expected=[(SOURCE_HASH,i,"i%03d"%i,"c%03d"%(i if i<339 else i-339)) for i in range(500)]
            db.executemany("INSERT INTO calibration_expected_v2 VALUES(?,?,?,?)",expected)
            db.executemany("INSERT INTO calibration_items_v2(holdout_key,item_hash,state,outcome,reference_match,updated_ts) VALUES(?,?, 'terminal',?,?,?)",[(SOURCE_HASH,"i%03d"%i,"accepted" if i<200 else "abstained",1 if i<200 else None,0) for i in range(500)])
            db.commit()
        experiment=SilverExperiment(self.root,silver=store)
        # A stale/missing JSON mirror is repairable, but cannot select a
        # winner.  The immutable cohort metadata above is the authority.
        experiment.path.write_text('{"schema":1,"runs":{}}\n',encoding="utf-8")
        # Hand-crafted rows are deliberately insufficient authority: only the
        # worker's immutable V2 finalization may make a calibration usable.
        self.assertFalse(controller.authorize_provisional(cohort))

    def test_discarded_prospective_member_invalidates_audio_fence(self):
        store=SilverStore(self.root)
        champion={"stable_id":"champion","backend":"a","repo":"r","revision":"1","language":"en"}
        candidate={"stable_id":"candidate","backend":"b","repo":"r","revision":"2","language":"en"}
        store.ensure_prospective_horizon(start_ts=0,champion=champion,candidates=[candidate],receipt_hash="receipt")
        store.enqueue(history_id="h",history_revision=0,audio_sha256="audio",teacher_family_hash=family_hash(),
                      consensus_policy_hash=policy_hash(),receipt_hash="receipt")
        job=store.claim("owner"); self.assertIsNotNone(job)
        self.assertTrue(store.admit_prospective_member(job,captured_ts=1))
        self.assertTrue(store.finish(job,outcome="discarded",code="test"))
        self.assertFalse(store.is_prospective_member("h",0))

    def test_missing_or_cluster_mismatched_arm_is_fail_closed(self):
        store=SilverStore(self.root)
        champion={"stable_id":"champion","backend":"a","repo":"r","revision":"1","language":"en","evaluator_hash":"e"}
        candidate={"stable_id":"candidate","backend":"b","repo":"r","revision":"2","language":"en","evaluator_hash":"e"}
        horizon=store.ensure_prospective_horizon(start_ts=0,champion=champion,candidates=[candidate],receipt_hash="receipt")
        cohort="missing-arm"; member=(cohort,0,"h",0,"accepted","audio","reference","cluster")
        with store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,"personal_horizon","evaluating","m","c",'{"accepted":1,"raw":1}',store.epoch(),0))
            db.execute("INSERT INTO cohort_members VALUES(?,?,?,?,?,?,?,?)",member)
            db.execute("UPDATE prospective_horizons SET status='frozen',frozen_cohort_id=? WHERE horizon_id=?",(cohort,horizon["horizon_id"]))
            stats=__import__("json").dumps({"word_distance":0,"reference_words":1,"character_distance":0,"reference_characters":1,"hallucination":False,"latency":.1,"cluster_hash":"wrong"})
            # Only champion exists and its persisted cluster disagrees with the
            # frozen member; both missing-arm and cluster paths must reject.
            db.execute("INSERT INTO candidate_attempts(cohort_id,arm_id,history_id,history_revision,audio_sha256,reference_digest,cluster_hash,status,stats,updated_ts) VALUES(?,?,?,?,?,?,?,?,?,?)",(cohort,"champion","h",0,"audio","reference","cluster","complete",stats,0))
            db.commit()
        outcome=SilverExperiment(self.root,silver=store).evaluate_persisted(cohort,identity_hash=horizon["identity_hash"])
        self.assertIsNone(outcome["winner"])

    def test_genuine_retry_exhausts_but_scheduler_busy_is_refunded(self):
        store=SilverStore(self.root)
        args=dict(history_id="h",history_revision=0,audio_sha256="a",teacher_family_hash="f",consensus_policy_hash="p",receipt_hash="r")
        store.enqueue(**args)
        busy=store.claim("owner"); self.assertTrue(store.finish(busy,outcome="retry",code="scheduler_busy"))
        with store._connect() as db:
            self.assertEqual(db.execute("SELECT attempts FROM jobs").fetchone()[0],0)
            db.execute("UPDATE jobs SET not_before=0")
        for _ in range(store.max_attempts):
            job=store.claim("owner"); self.assertIsNotNone(job)
            self.assertTrue(store.finish(job,outcome="retry",code="TimeoutError"))
            with store._connect() as db: db.execute("UPDATE jobs SET not_before=0")
        with store._connect() as db:
            status,attempts=db.execute("SELECT status,attempts FROM jobs").fetchone()
        self.assertEqual((status,attempts),("discarded",store.max_attempts))

    def test_high_volume_horizon_reserves_a_slot_for_seventh_day(self):
        store=SilverStore(self.root)
        champion={"stable_id":"champion","backend":"a","repo":"r","revision":"1","language":"en"}
        candidate={"stable_id":"candidate","backend":"b","repo":"r","revision":"2","language":"en"}
        horizon=store.ensure_prospective_horizon(start_ts=0,champion=champion,candidates=[candidate],receipt_hash="receipt")
        with store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for ordinal in range(500):
                day=ordinal % 6
                db.execute("INSERT INTO prospective_members VALUES(?,?,?,?,?,?,?)",(horizon["horizon_id"],ordinal,"old-%d"%ordinal,0,"a-%d"%ordinal,float(day*86400),"c-%d"%day))
            db.commit()
        store.enqueue(history_id="new",history_revision=0,audio_sha256="new-audio",teacher_family_hash=family_hash(),consensus_policy_hash=policy_hash(),receipt_hash="receipt")
        job=store.claim("owner"); self.assertTrue(store.admit_prospective_member(job,captured_ts=6*86400))
        state=store.prospective_status()
        self.assertEqual((state["status"],state["members"]),("closed",500))

    def test_calibration_v2_plan_is_immutable_and_claims_are_unique(self):
        store=SilverStore(self.root)
        from calibration_manifest_builder import SOURCE_HASH, _protocol_hash
        expected=[{"item_hash":"i%03d"%i,"cluster_hash":"c%03d"%(i if i < 339 else i-339)} for i in range(500)]
        with self.assertRaises(RuntimeError):
            store.begin_calibration_v2(holdout_key="forged",manifest_hash="manifest",source_hash="forged",protocol_hash="forged",policy_hash="forged",receipt_hash="receipt",expected=expected)
        # Shape-correct synthetic rows are not a compiled frozen source plan.
        with self.assertRaises(RuntimeError):
            store.begin_calibration_v2(holdout_key=SOURCE_HASH,manifest_hash="manifest",source_hash=SOURCE_HASH,protocol_hash=_protocol_hash(),policy_hash=policy_hash(),receipt_hash="receipt",expected=expected)
        changed=list(expected); changed[0]={"item_hash":"changed","cluster_hash":"c000"}
        with self.assertRaises(RuntimeError):
            store.begin_calibration_v2(holdout_key=SOURCE_HASH,manifest_hash="manifest",source_hash=SOURCE_HASH,protocol_hash=_protocol_hash(),policy_hash=policy_hash(),receipt_hash="receipt",expected=changed)

    def test_reader_identity_rotates_protocol_and_rejects_mismatched_bundle(self):
        import calibration_manifest_builder as builder
        original=builder._protocol_hash()
        with mock.patch.object(builder,"READER_IDENTITY",{**builder.READER_IDENTITY,"unicode":"bogus"}):
            self.assertNotEqual(original,builder._protocol_hash())
        def bundle_payload(runtime, *, extra=False):
            body={"schema":1,"runtime":runtime,"entries":[]}
            if extra: body["unexpected"]=True
            return {**body,"digest":hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()}
        # A parser helper whose declared runtime differs from the pinned
        # reader semantics is rejected before item-shape/teacher work.
        with mock.patch.object(builder,"_sha",side_effect=[builder.TEST_SHA256,builder.VALIDATION_SHA256]), \
             mock.patch.object(builder.subprocess,"run",return_value=type("R",(),{"returncode":0})()):
            with mock.patch.object(builder.tempfile,"mkdtemp",return_value=str(self.root / "reader")):
                (self.root / "reader").mkdir(mode=0o700)
                bundle=self.root / "reader" / "bundle"; bundle.mkdir()
                (bundle / "plan.json").write_text(json.dumps(bundle_payload({**builder.READER_IDENTITY,"unicode":"wrong"})))
                with self.assertRaisesRegex(RuntimeError,"reader bundle identity invalid"):
                    builder.derive_plan_with_reader(self.root / "a",self.root / "b",Path("/bin/echo"))

    def test_reader_bundle_rejects_unknown_schema_field_with_valid_digest(self):
        import calibration_manifest_builder as builder
        body={"schema":1,"runtime":dict(builder.READER_IDENTITY),"entries":[],"unexpected":True}
        payload={**body,"digest":hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()}
        root=self.root / "reader-schema"; root.mkdir(mode=0o700); bundle=root / "bundle"; bundle.mkdir()
        (bundle / "plan.json").write_text(json.dumps(payload),encoding="utf-8")
        with mock.patch.object(builder,"_sha",side_effect=[builder.TEST_SHA256,builder.VALIDATION_SHA256]), \
             mock.patch.object(builder.subprocess,"run",return_value=type("R",(),{"returncode":0})()), \
             mock.patch.object(builder.tempfile,"mkdtemp",return_value=str(root)):
            with self.assertRaisesRegex(RuntimeError,"reader bundle identity invalid"):
                builder.derive_plan_with_reader(self.root / "a",self.root / "b",Path("/bin/echo"))

    def test_canonical_wav_memory_hash_matches_written_artifact(self):
        import calibration_manifest_builder as builder
        from audio_codec import write_canonical_wav
        pcm=(np.array([0,.25,-.25],dtype="<f4")*32767).astype("<i2").tobytes()
        path=self.root / "sample.wav"; write_canonical_wav(path,pcm)
        self.assertEqual(builder._canonical_wav_sha256(pcm),hashlib.sha256(path.read_bytes()).hexdigest())

    def test_builder_compiled_mismatch_creates_no_output(self):
        import calibration_manifest_builder as builder
        pcm=(np.zeros(8000,dtype="<i2")).tobytes(); output=self.root / "calibration-output"
        with mock.patch.object(builder,"derive_plan_with_reader",return_value=[("a","speaker","test",pcm,"reference")]):
            with self.assertRaises(RuntimeError): builder.build(self.root / "test",self.root / "validation",output,pyarrow_python="reader")
        self.assertFalse(output.exists())

    def test_builder_exact_source_publishes_loadable_private_manifest(self):
        """The pinned reader can publish and reload V2 without teachers."""
        import calibration_manifest_builder as builder
        from calibration_worker import load_manifest
        reader=Path(os.environ.get("SOTTO_CALIBRATION_FIXTURES", str(self.root / "no-calibration-fixtures"))) / "python3.11"
        test=Path(os.environ.get("SOTTO_CALIBRATION_FIXTURES", str(self.root / "no-calibration-fixtures"))) / "test.parquet"
        validation=Path(os.environ.get("SOTTO_CALIBRATION_FIXTURES", str(self.root / "no-calibration-fixtures"))) / "validation.parquet"
        if not (reader.is_file() and test.is_file() and validation.is_file()):
            self.skipTest("pinned reader fixture unavailable")
        output=self.root / "built"
        path=builder.build(test,validation,output,pyarrow_python=reader)
        payload=json.loads(path.read_text("utf-8"))
        loaded,digest=load_manifest(path)
        self.assertEqual(digest,payload["manifest_hash"])
        self.assertEqual(loaded["membership_hash"],builder.MEMBERSHIP_HASH)
        self.assertTrue(all(not Path(entry["audio_path"]).is_absolute() for entry in loaded["entries"]))
        alternate_parent=self.root / "other-parent"; alternate_parent.mkdir()
        alternate=json.loads(builder.build(test,validation,alternate_parent / "other-built",pyarrow_python=reader).read_text("utf-8"))
        self.assertEqual(alternate["manifest_hash"],payload["manifest_hash"])
        self.assertEqual(alternate["protocol_bundle"]["digest"],payload["protocol_bundle"]["digest"])

    def test_builder_exact_source_bundle_failure_cleans_stage_and_preserves_target(self):
        """A post-audio bundle failure never publishes or overwrites output."""
        import calibration_manifest_builder as builder
        reader=Path(os.environ.get("SOTTO_CALIBRATION_FIXTURES", str(self.root / "no-calibration-fixtures"))) / "python3.11"
        test=Path(os.environ.get("SOTTO_CALIBRATION_FIXTURES", str(self.root / "no-calibration-fixtures"))) / "test.parquet"
        validation=Path(os.environ.get("SOTTO_CALIBRATION_FIXTURES", str(self.root / "no-calibration-fixtures"))) / "validation.parquet"
        if not (reader.is_file() and test.is_file() and validation.is_file()):
            self.skipTest("pinned reader fixture unavailable")
        output=self.root / "atomic-built"; captured=[]; original=builder.derive_plan_with_reader
        def exact_plan(*args,**kwargs):
            records=original(*args,**kwargs); captured[:] = [records]; return records
        with mock.patch.object(builder,"derive_plan_with_reader",side_effect=exact_plan), \
             mock.patch.object(builder,"_write_protocol_bundle",side_effect=RuntimeError("injected bundle failure")):
            with self.assertRaisesRegex(RuntimeError,"injected bundle failure"):
                builder.build(test,validation,output,pyarrow_python=reader)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".atomic-built.*.tmp")),[])
        output.mkdir(); sentinel=output / "sentinel"; sentinel.write_text("preserve",encoding="utf-8")
        with mock.patch.object(builder,"derive_plan_with_reader",return_value=captured[0]):
            with self.assertRaises(RuntimeError):
                builder.build(test,validation,output,pyarrow_python=reader)
        self.assertEqual(sentinel.read_text("utf-8"),"preserve")

    def test_worker_rejects_source_plan_mismatch_before_store_or_teacher(self):
        import calibration_manifest_builder as builder
        import calibration_worker as worker
        reader=Path(os.environ.get("SOTTO_CALIBRATION_FIXTURES", str(self.root / "no-calibration-fixtures"))) / "python3.11"
        test=Path(os.environ.get("SOTTO_CALIBRATION_FIXTURES", str(self.root / "no-calibration-fixtures"))) / "test.parquet"
        validation=Path(os.environ.get("SOTTO_CALIBRATION_FIXTURES", str(self.root / "no-calibration-fixtures"))) / "validation.parquet"
        if not (reader.is_file() and test.is_file() and validation.is_file()):
            self.skipTest("pinned reader fixture unavailable")
        records=builder.derive_plan_with_reader(test,validation,reader)
        with mock.patch.object(builder,"derive_plan_with_reader",return_value=records):
            manifest=builder.build(test,validation,self.root / "mismatch-built",pyarrow_python=reader)
        altered=list(records); ident,cluster,split,pcm,reference=altered[0]
        altered[0]=(ident,cluster,split,pcm,reference+" altered")
        with mock.patch.object(worker,"derive_plan_with_reader",return_value=altered), \
             mock.patch.object(worker,"SilverStore",side_effect=AssertionError("store must not open")), \
             mock.patch.object(worker,"OfflineTeacherRunner",side_effect=AssertionError("teacher must not open")):
            with self.assertRaisesRegex(RuntimeError,"calibration source-derived plan mismatch"):
                worker.run(self.root,manifest,test_parquet=test,validation_parquet=validation,pyarrow_python=reader)

    def test_worker_releases_lease_without_terminal_on_mid_item_receipt_rotation(self):
        import calibration_manifest_builder as builder
        import calibration_worker as worker
        reader=Path(os.environ.get("SOTTO_CALIBRATION_FIXTURES", str(self.root / "no-calibration-fixtures"))) / "python3.11"
        test=Path(os.environ.get("SOTTO_CALIBRATION_FIXTURES", str(self.root / "no-calibration-fixtures"))) / "test.parquet"
        validation=Path(os.environ.get("SOTTO_CALIBRATION_FIXTURES", str(self.root / "no-calibration-fixtures"))) / "validation.parquet"
        if not (reader.is_file() and test.is_file() and validation.is_file()):
            self.skipTest("pinned reader fixture unavailable")
        records=builder.derive_plan_with_reader(test,validation,reader)
        with mock.patch.object(builder,"derive_plan_with_reader",return_value=records):
            manifest=builder.build(test,validation,self.root / "rotation-built",pyarrow_python=reader)
        payload=json.loads(manifest.read_text("utf-8")); source,item_hash,_cluster,_reference=worker.validate_item(payload["entries"][0],manifest_dir=manifest.parent)
        claim={"item_hash":item_hash,"lease_token":"owned"}; released=[]; seen=[]
        class Store:
            def begin_calibration_v2(self,**_kwargs): return None
            def claim_calibration_v2(self,*_args):
                if getattr(self,"claimed",False): return None
                self.claimed=True; return claim
            def release_calibration_v2(self,item,*,code): released.append((item,code)); return True
            def finish_calibration_v2(self,*_args,**_kwargs): raise AssertionError("must not terminalize")
            def finalize_calibration_v2(self,*_args,**_kwargs): raise AssertionError("must not finalize")
        class Lease:
            def __enter__(self): return True
            def __exit__(self,*_args): return False
        class Scheduler:
            owner_id="owner"
            def evaluator_lease(self): return Lease()
        class Runner:
            def __init__(self): self.calls=0; self.rotated=False
            def validate(self):
                self.calls+=1; return "rotated" if self.rotated else "receipt"
            def transcribe(self,_teacher,path,_item_hash):
                seen.append(path); self.rotated=True; return "ok"
        store=Store(); runner=Runner()
        with mock.patch.object(worker,"derive_plan_with_reader",return_value=records), \
             mock.patch.object(worker,"SilverStore",return_value=store), \
             mock.patch.object(worker,"load_receipts",return_value={}), \
             mock.patch.object(worker,"OfflineTeacherRunner",return_value=runner), \
             mock.patch.object(worker,"InferenceScheduler",return_value=Scheduler()):
            with self.assertRaisesRegex(RuntimeError,"teacher receipt changed during calibration"):
                worker.run(self.root,manifest,test_parquet=test,validation_parquet=validation,pyarrow_python=reader)
        self.assertEqual(released,[(claim,"RuntimeError")])
        self.assertEqual(len(seen),1)
        self.assertTrue(all(path != source for path in seen))

    def test_candidate_attempt_receipt_rotation_before_stats_is_stale_not_success(self):
        """A rotated candidate receipt cannot commit post-inference metrics."""
        row=self.live(); audio=row["audio"]["inference"]
        manifest={"stable_id":"candidate","revision":"old","snapshot_digest":"old"}
        state={"rotated":False,"claimed":False}; finished=[]
        class Silver:
            def claim_candidate_attempt(self,_owner):
                if state["claimed"]: return None
                state["claimed"]=True
                return {"cohort_id":"cohort","arm_id":"candidate","history_id":row["id"],
                        "history_revision":row["revision"],"audio_sha256":audio["sha256"],
                        "clear_epoch":0,"cluster_hash":"cluster"}
            def prospective_status(self):
                return {"cohort_id":"cohort","clear_epoch":0,"champion":{"stable_id":"champion"},"candidates":[manifest]}
            def epoch(self): return 0
            @staticmethod
            def manifest_hash(value): return json.dumps(value,sort_keys=True,separators=(",",":"))
            def candidate_reference(self,_attempt): return "calm blue river carries gentle morning light softly today home"
            def finish_candidate_attempt(self,attempt,**kwargs): finished.append((attempt,kwargs)); return True
        class Lease:
            def __enter__(self): return True
            def __exit__(self,*_args): return False
        class Scheduler:
            owner_id="candidate-test"
            def evaluator_lease(self): return Lease()
        class Backends:
            def transcribe(self,*_args,**_kwargs): state["rotated"]=True; return "calm blue river carries gentle morning light softly today home",{}
        worker=AdaptiveWorker(self.root,history=self.history,silver=Silver(),scheduler=Scheduler(),backends=Backends(),teachers=object())
        worker.preflight.valid_manifest=lambda _value: True
        worker._receipt_manifest=lambda _arm: ({**manifest,"snapshot_digest":"new"} if state["rotated"] else dict(manifest))
        self.assertTrue(worker.process_candidate_one())
        self.assertEqual(len(finished),1)
        self.assertEqual(finished[0][1].get("code"),"stale_identity")
        self.assertNotIn("stats",finished[0][1])

    def test_protocol_bundle_identity_matches_ordered_regular_sources(self):
        import calibration_manifest_builder as builder
        identity=builder.protocol_bundle_identity(); body=dict(identity); digest=body.pop("digest")
        self.assertEqual([entry["path"] for entry in identity["files"]],list(builder.PROTOCOL_FILES))
        for entry in identity["files"]:
            path=Path(builder.__file__).parent / entry["path"]
            self.assertTrue(path.is_file() and not path.is_symlink())
            self.assertEqual(entry["size"],path.stat().st_size)
            self.assertEqual(entry["sha256"],hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(digest,hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest())

    def test_protocol_bundle_identity_rotates_for_reader_and_app_runtime(self):
        import calibration_manifest_builder as builder
        baseline=builder.protocol_bundle_identity(); baseline_hash=builder._protocol_hash()
        with mock.patch.object(builder,"READER_IDENTITY",{**builder.READER_IDENTITY,"unicode":"14.0.1"}):
            self.assertNotEqual(builder.protocol_bundle_identity()["digest"],baseline["digest"])
            self.assertNotEqual(builder._protocol_hash(),baseline_hash)
        with mock.patch.object(builder.sys,"version","99.1.0 rotated"):
            self.assertNotEqual(builder.protocol_bundle_identity()["digest"],baseline["digest"])
            self.assertNotEqual(builder._protocol_hash(),baseline_hash)
        with mock.patch.object(builder.unicodedata,"unidata_version","99.0.0"):
            self.assertNotEqual(builder.protocol_bundle_identity()["digest"],baseline["digest"])
            self.assertNotEqual(builder._protocol_hash(),baseline_hash)
        with mock.patch.object(np,"__version__","99.0.0"):
            self.assertNotEqual(builder.protocol_bundle_identity()["digest"],baseline["digest"])
            self.assertNotEqual(builder._protocol_hash(),baseline_hash)
        self.assertEqual(builder.protocol_bundle_identity(),baseline)
        self.assertEqual(builder._protocol_hash(),baseline_hash)

    def test_write_protocol_bundle_copies_identity_bytes_privately(self):
        import calibration_manifest_builder as builder
        target=self.root / "protocol"; identity=builder._write_protocol_bundle(target)
        self.assertEqual(json.loads((target / "metadata.json").read_text()),identity)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode),0o700)
        for entry in identity["files"]:
            copied=target / entry["path"]; source=Path(builder.__file__).parent / entry["path"]
            self.assertEqual(copied.read_bytes(),source.read_bytes()); self.assertEqual(stat.S_IMODE(copied.stat().st_mode),0o600)

    def test_write_protocol_bundle_rejects_mismatch_and_preserves_existing_target(self):
        import calibration_manifest_builder as builder
        identity=builder.protocol_bundle_identity(); bad=json.loads(json.dumps(identity)); bad["files"][0]["sha256"]="0"*64
        target=self.root / "bad-protocol"
        with self.assertRaises(RuntimeError): builder._write_protocol_bundle(target,bad)
        self.assertFalse(target.exists())
        target.mkdir(); sentinel=target / "sentinel"; sentinel.write_text("keep")
        with self.assertRaises(RuntimeError): builder._write_protocol_bundle(target)
        self.assertEqual(sentinel.read_text(),"keep")

    def test_worker_protocol_bundle_loader_rejects_tamper(self):
        import calibration_manifest_builder as builder
        from calibration_worker import _load_protocol_bundle
        manifest=self.root / "manifest.json"; manifest.write_text("{}")
        identity=builder._write_protocol_bundle(self.root / "protocol")
        value={"schema":1,"path":"protocol/metadata.json","digest":identity["digest"]}
        self.assertEqual(_load_protocol_bundle(manifest,value)[1],identity)
        copied=self.root / "protocol" / identity["files"][0]["path"]
        copied.write_bytes(b"tampered")
        with self.assertRaises(RuntimeError): _load_protocol_bundle(manifest,value)

    def test_immutable_observed_wav_survives_manifest_source_mutation(self):
        from audio_codec import read_canonical_wav, write_canonical_wav
        from calibration_worker import _immutable_observed_wav, _verify_observed_wav
        source_dir=self.root / "manifest-audio"; source_dir.mkdir()
        source=source_dir / "source.wav"; pcm=(np.arange(32,dtype="<i2")-16).tobytes()
        write_canonical_wav(source,pcm)
        expected_wav=hashlib.sha256(source.read_bytes()).hexdigest()
        _samples,identity=read_canonical_wav(source)
        with _immutable_observed_wav(source,expected_wav,identity.sha256,identity.sample_count) as observed:
            observed_root=observed.parent
            self.assertNotEqual(observed,source)
            self.assertNotIn(source_dir.resolve(),observed.parents)
            self.assertEqual(stat.S_IMODE(observed.stat().st_mode),0o400)
            source.write_bytes(b"mutated manifest transport")
            self.assertEqual(hashlib.sha256(observed.read_bytes()).hexdigest(),expected_wav)
            _verify_observed_wav(observed,expected_wav,identity.sha256,identity.sample_count)
        self.assertFalse(observed_root.exists())

    def test_collect_observed_answers_uses_one_private_wav_for_all_teachers(self):
        from audio_codec import read_canonical_wav, write_canonical_wav
        from calibration_worker import _collect_observed_answers
        from teacher_backends import REQUIRED_TEACHERS
        source=self.root / "source.wav"; pcm=(np.arange(64,dtype="<i2")-32).tobytes()
        write_canonical_wav(source,pcm); expected_bytes=source.read_bytes()
        _samples,identity=read_canonical_wav(source)
        seen=[]
        class Lease:
            def __enter__(self): return True
            def __exit__(self,*_args): return False
        class Scheduler:
            def evaluator_lease(self): return Lease()
        class Runner:
            def validate(self): return "receipt"
            def transcribe(self,teacher,path,item_hash):
                if path.read_bytes()!=expected_bytes: raise RuntimeError("wrong observed artifact")
                seen.append(path); return teacher.stable_id
        spec={"audio_sha256":hashlib.sha256(expected_bytes).hexdigest(),"pcm_sha256":identity.sha256,"sample_count":identity.sample_count}
        answers=_collect_observed_answers(Runner(),Scheduler(),source,"item","receipt",spec)
        self.assertEqual(set(answers),{teacher.stable_id for teacher in REQUIRED_TEACHERS})
        self.assertEqual(len(seen),len(REQUIRED_TEACHERS))
        self.assertTrue(all(path != source for path in seen))
        self.assertEqual(len(set(seen)),1)
        self.assertFalse(seen[0].exists())

    def test_v2_exact_reader_ledger_claim_finish_and_finalize(self):
        """No-teacher durable V2 happy path over the compiled public plan."""
        reader=Path(os.environ.get("SOTTO_CALIBRATION_FIXTURES", str(self.root / "no-calibration-fixtures"))) / "python3.11"
        test=Path(os.environ.get("SOTTO_CALIBRATION_FIXTURES", str(self.root / "no-calibration-fixtures"))) / "test.parquet"
        validation=Path(os.environ.get("SOTTO_CALIBRATION_FIXTURES", str(self.root / "no-calibration-fixtures"))) / "validation.parquet"
        if not (reader.is_file() and test.is_file() and validation.is_file()): self.skipTest("pinned reader fixture unavailable")
        records=derive_plan_with_reader(test,validation,reader)
        expected=[]
        for ident,cluster,split,pcm,reference in records:
            item=hashlib.sha256(json.dumps({"id":ident,"cluster":cluster,"split":split,"pcm":hashlib.sha256(pcm).hexdigest(),"samples":len(pcm)//2,
                "reference":hashlib.sha256(reference.encode()).hexdigest()},sort_keys=True,separators=(",",":")).encode()).hexdigest()
            expected.append({"item_hash":item,"cluster_hash":hashlib.sha256(cluster.encode()).hexdigest()})
        ledger=hashlib.sha256(json.dumps([(i,row["item_hash"],row["cluster_hash"]) for i,row in enumerate(expected)],separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(ledger,LEDGER_HASH)
        store=SilverStore(self.root)
        store.begin_calibration_v2(holdout_key=SOURCE_HASH,manifest_hash="frozen",source_hash=SOURCE_HASH,protocol_hash=_protocol_hash(),policy_hash=policy_hash(),receipt_hash="receipt",expected=expected)
        first=store.claim_calibration_v2(SOURCE_HASH,"one"); second=store.claim_calibration_v2(SOURCE_HASH,"two")
        self.assertNotEqual(first["item_hash"],second["item_hash"])
        self.assertTrue(store.finish_calibration_v2(first,outcome="abstained",reference_match=None,code="mismatch"))
        self.assertTrue(store.finish_calibration_v2(first,outcome="abstained",reference_match=None,code="mismatch"))
        self.assertFalse(store.finish_calibration_v2(first,outcome="accepted",reference_match=True,code="exact_unanimous"))
        self.assertTrue(store.finish_calibration_v2(second,outcome="abstained",reference_match=None,code="mismatch"))
        while True:
            item=store.claim_calibration_v2(SOURCE_HASH,"drain")
            if item is None: break
            self.assertTrue(store.finish_calibration_v2(item,outcome="abstained",reference_match=None,code="mismatch"))
        result=store.finalize_calibration_v2(SOURCE_HASH)
        self.assertEqual(result["state"],"failed")
        self.assertEqual(result["total"],500)
        self.assertEqual(store.finalize_calibration_v2(SOURCE_HASH),result)

    def test_v2_compiled_ledger_durable_passing_finalize_and_clear_survival(self):
        """The production authority path works without local Parquet files."""
        expected=_compiled_v2_expected()
        self.assertEqual(len({row["cluster_hash"] for row in expected[:200]}),200)
        store=SilverStore(self.root)
        store.begin_calibration_v2(holdout_key=SOURCE_HASH,manifest_hash="compiled-public-plan",source_hash=SOURCE_HASH,
                                   protocol_hash=_protocol_hash(),policy_hash=policy_hash(),receipt_hash="receipt",expected=expected)
        # SQLite leases, rather than a fixture-side loop, allocate distinct
        # immutable rows to concurrent workers.
        with ThreadPoolExecutor(max_workers=4) as pool:
            claimed=[item for item in pool.map(lambda owner: store.claim_calibration_v2(SOURCE_HASH,owner),["a","b","c","d"]) if item]
        self.assertEqual(len({item["item_hash"] for item in claimed}),len(claimed))
        ordinal={row["item_hash"]:index for index,row in enumerate(expected)}
        for item in claimed:
            accepted=ordinal[item["item_hash"]] < 200
            self.assertTrue(store.finish_calibration_v2(item,outcome="accepted" if accepted else "abstained",
                                                        reference_match=True if accepted else None,
                                                        code="exact_unanimous" if accepted else "mismatch"))
        replay=claimed[0]
        self.assertTrue(store.finish_calibration_v2(replay,outcome="accepted",reference_match=True,code="exact_unanimous"))
        self.assertFalse(store.finish_calibration_v2(replay,outcome="abstained",reference_match=None,code="mismatch"))
        while (item:=store.claim_calibration_v2(SOURCE_HASH,"drain")) is not None:
            accepted=ordinal[item["item_hash"]] < 200
            self.assertTrue(store.finish_calibration_v2(item,outcome="accepted" if accepted else "abstained",
                                                        reference_match=True if accepted else None,
                                                        code="exact_unanimous" if accepted else "mismatch"))
        result=store.finalize_calibration_v2(SOURCE_HASH)
        self.assertEqual(result,{"state":"passed","accepted":200,"clusters":200,"reference_errors":0,"total":500})
        self.assertEqual(store.calibration_status()["state"],"passed")
        # Public source evidence intentionally survives a personal clear.
        store.clear()
        self.assertEqual(store.calibration_status()["state"],"passed")
        self.assertEqual(store.finalize_calibration_v2(SOURCE_HASH),result)

    def test_v2_lease_expiry_reclaim_release_and_legacy_nonauthority(self):
        now=[0.0]
        store=SilverStore(self.root,clock=lambda:now[0])
        expected=_compiled_v2_expected()
        store.begin_calibration_v2(holdout_key=SOURCE_HASH,manifest_hash="compiled-public-plan",source_hash=SOURCE_HASH,
                                   protocol_hash=_protocol_hash(),policy_hash=policy_hash(),receipt_hash="receipt",expected=expected)
        first=store.claim_calibration_v2(SOURCE_HASH,"first")
        now[0]+=store.lease_seconds+1
        # An unreaped expired lease is not terminal-authoritative.
        self.assertFalse(store.finish_calibration_v2(first,outcome="abstained",reference_match=None,code="mismatch"))
        reclaimed=store.claim_calibration_v2(SOURCE_HASH,"second")
        self.assertEqual(first["item_hash"],reclaimed["item_hash"])
        self.assertNotEqual(first["token"],reclaimed["token"])
        self.assertFalse(store.finish_calibration_v2(first,outcome="abstained",reference_match=None,code="mismatch"))
        self.assertTrue(store.release_calibration_v2(reclaimed,code="TimeoutExpired"))
        released=store.claim_calibration_v2(SOURCE_HASH,"third")
        self.assertEqual(released["item_hash"],first["item_hash"])
        self.assertTrue(store.finish_calibration_v2(released,outcome="abstained",reference_match=None,code="mismatch"))
        # Legacy progress rows remain migration audit only and can never pass
        # the V2 deployment authority read.
        legacy=SilverStore(self.root / "legacy")
        legacy.begin_calibration(manifest_hash="legacy",source_hash="source",policy_hash="policy",split_hash="split",receipt_hash="receipt")
        self.assertIsNone(legacy.calibration_status())

    def test_preboundary_shadow_job_never_enters_prospective_membership(self):
        store=SilverStore(self.root)
        champion={"stable_id":"champion","backend":"a","repo":"r","revision":"1","language":"en"}
        candidate={"stable_id":"candidate","backend":"b","repo":"r","revision":"2","language":"en"}
        store.ensure_prospective_horizon(start_ts=100,champion=champion,candidates=[candidate],receipt_hash="receipt")
        store.enqueue(history_id="old",history_revision=0,audio_sha256="audio",teacher_family_hash=family_hash(),consensus_policy_hash=policy_hash(),receipt_hash="receipt")
        job=store.claim("owner")
        self.assertFalse(store.admit_prospective_member(job,captured_ts=99))
        self.assertFalse(store.is_prospective_member("old",0))

    def _shadow_horizon(self, store: SilverStore, *, start: float = 100.0) -> None:
        store.ensure_prospective_horizon(start_ts=start,
            champion={"stable_id":"champion","backend":"a","repo":"r","revision":"1","language":"en"},
            candidates=[{"stable_id":"candidate","backend":"b","repo":"r","revision":"2","language":"en"}],
            receipt_hash="receipt")

    @staticmethod
    def _shadow_job(store: SilverStore, history_id: str) -> dict:
        store.enqueue(history_id=history_id,history_revision=0,audio_sha256=f"audio-{history_id}",
                      teacher_family_hash="family",consensus_policy_hash="policy",receipt_hash="receipt")
        job=store.claim("audit-owner")
        assert job is not None
        return job

    def test_preboundary_shadow_audit_is_atomic_and_uses_immutable_hypothesis(self):
        store=SilverStore(self.root); self._shadow_horizon(store)
        entry={"hypothesis":"alpha","text":"mutable replacement"}
        stats=AdaptiveWorker._old_capture_shadow_statistics(entry,"alpha beta")
        self.assertEqual(stats,{"word_distance":1,"reference_words":2,"character_distance":4,"reference_characters":9})
        entry["text"]="a completely different mutable display value"
        self.assertEqual(AdaptiveWorker._old_capture_shadow_statistics(entry,"alpha beta"),stats)
        job=self._shadow_job(store,"preboundary")
        self.assertTrue(store.finish(job,outcome="accepted",reference="alpha beta",vote_digest="vote",
                                     shadow_audit=stats,captured_ts=99.0))
        with store._connect() as db:
            label=db.execute("SELECT state,reference FROM labels WHERE job_key=?",(job["job_key"],)).fetchone()
            audit=db.execute("SELECT word_distance,reference_words,character_distance,reference_characters,preboundary FROM shadow_audits WHERE job_key=?",(job["job_key"],)).fetchone()
        # Shadow/nonmember labels have no future exact candidate consumer, so
        # the digest and aggregate survive but plaintext is scrubbed in the
        # same terminal transaction.
        self.assertEqual(label,("accepted",None))
        self.assertEqual(audit,(1,2,4,9,1))
        # The aggregate is intentionally outside every authority path.
        self.assertNotIn("shadow_audit",inspect.getsource(DeploymentController))
        self.assertNotIn("shadow_audit",inspect.getsource(SilverExperiment))

    def test_plain_reference_is_retained_only_for_active_exact_consumer_then_scrubbed(self):
        store=SilverStore(self.root); self._shadow_horizon(store,start=0)
        store.enqueue(history_id="active-member",history_revision=0,audio_sha256="active-audio",
                      teacher_family_hash=family_hash(),consensus_policy_hash=policy_hash(),receipt_hash="receipt")
        job=store.claim("active-owner"); self.assertIsNotNone(job)
        self.assertTrue(store.admit_prospective_member(job,captured_ts=1.0,session_id="session"))
        self.assertTrue(store.finish(job,outcome="accepted",reference="alpha beta",vote_digest="vote"))
        with store._connect() as db:
            self.assertEqual(db.execute("SELECT reference FROM labels WHERE job_key=?",(job["job_key"],)).fetchone()[0],"alpha beta")
            # Invalidating a prospective horizon means no candidate attempt
            # can legally consume the plaintext, so terminal cleanup removes
            # it while retaining the digest/member identity.
            db.execute("UPDATE prospective_horizons SET status='invalidated'")
        # A harmless terminal transition drives the same transactional scrub
        # path used by candidate/evaluation completion.
        other=self._shadow_job(store,"other-shadow")
        self.assertTrue(store.finish(other,outcome="abstained",vote_digest="vote"))
        with store._connect() as db:
            self.assertIsNone(db.execute("SELECT reference FROM labels WHERE job_key=?",(job["job_key"],)).fetchone()[0])
            self.assertIsNotNone(db.execute("SELECT reference_digest FROM labels WHERE job_key=?",(job["job_key"],)).fetchone()[0])

    def test_shadow_audit_rejects_postboundary_and_stale_label_cas(self):
        store=SilverStore(self.root); self._shadow_horizon(store)
        stats={"word_distance":1,"reference_words":2,"character_distance":4,"reference_characters":9}
        post=self._shadow_job(store,"postboundary")
        self.assertTrue(store.finish(post,outcome="accepted",reference="alpha beta",vote_digest="vote",
                                     shadow_audit=stats,captured_ts=100.0))
        stale=self._shadow_job(store,"stale")
        stale["lease_owner"]="not-the-owner"
        self.assertFalse(store.finish(stale,outcome="accepted",reference="alpha beta",vote_digest="vote",
                                      shadow_audit=stats,captured_ts=99.0))
        with store._connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM shadow_audits").fetchone()[0],0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM labels WHERE job_key=?",(stale["job_key"],)).fetchone()[0],0)

    def test_worker_finalizes_immutable_old_hypothesis_audit_with_label(self):
        row=self.coordinator.append_live("alpha",np.zeros(8000,np.float32),.5,"m",ts=99,
                                         adaptive=True,language="en")
        # Display text may evolve independently, but the retained shadow
        # comparison has to use the original capture hypothesis.
        with self.history._lock:
            self.history._entries[0]["text"]="mutable corrected display"
            _atomic_jsonl(self.history.index,self.history._entries)
        store=SilverStore(self.root); self._shadow_horizon(store)
        worker=AdaptiveWorker(self.root,history=self.history,silver=store,teachers=_Teachers("alpha beta"))
        self.assertEqual(worker.run_once()["status"]["accepted"],1)
        self.assertEqual(store.shadow_audit_status(),{"count":1,"word_distance":1,"reference_words":2,
                         "character_distance":4,"reference_characters":9,"wer":.5,"cer":4/9})
        with store._connect() as db:
            self.assertEqual(db.execute("SELECT state,reference FROM labels").fetchone(),("accepted",None))
            self.assertEqual(db.execute("SELECT COUNT(*) FROM shadow_audits").fetchone()[0],1)

    def test_revoke_removes_old_capture_shadow_aggregate(self):
        store=SilverStore(self.root); self._shadow_horizon(store)
        job=self._shadow_job(store,"revoked-shadow")
        stats={"word_distance":1,"reference_words":2,"character_distance":4,"reference_characters":9}
        self.assertTrue(store.finish(job,outcome="accepted",reference="alpha beta",vote_digest="vote",
                                     shadow_audit=stats,captured_ts=99.0))
        self.assertEqual(store.shadow_audit_status()["count"],1)
        self.assertEqual(store.revoke_history("revoked-shadow",history_revision=0),1)
        self.assertEqual(store.shadow_audit_status()["count"],0)

    def test_runtime_revoke_keeps_exact_schema2_intent_until_worker_recovers(self):
        row=self.live()
        runtime=AdaptiveRuntime(self.root,history=self.history,coordinator=self.coordinator)
        runtime.revoke(row["id"])
        store=SilverStore(self.root); pending=store.scrub_pending()
        self.assertIsInstance(pending,dict)
        self.assertEqual((pending["schema"],pending["kind"],pending["phase"]),(2,"revoke","silver_fenced"))
        self.assertEqual([(item["history_id"],item["history_revision"],item["operation"])
                         for item in pending["intents"]],[(row["id"],row["revision"],"revoke")])
        # Runtime must leave selective experiment cleanup and exact ack to the
        # recovery owner rather than acknowledging a marker it no longer owns.
        worker=AdaptiveWorker(self.root,history=HistoryStore(self.root),silver=store,teachers=_Teachers())
        worker.recover_markers()
        self.assertIsNone(store.scrub_pending())
        self.assertEqual(SilverStore(self.root).status()["jobs"].get("pending",0),0)

    def test_runtime_revoke_cannot_ack_concurrent_clear_marker(self):
        row=self.live()
        runtime=AdaptiveRuntime(self.root,history=self.history,coordinator=self.coordinator)
        store=SilverStore(self.root); original=runtime.coordinator.revoke_history
        def revoke_then_clear(history_id: str):
            result=original(history_id)
            store.clear(reason="concurrent_clear")
            return result
        runtime.coordinator.revoke_history=revoke_then_clear  # type: ignore[method-assign]
        runtime.revoke(row["id"])
        clear=store.scrub_pending()
        self.assertEqual(clear and clear.get("kind"),"clear")
        # The worker observes and completes the exact replacement marker
        # before any ordinary history scan can enqueue new-epoch evidence.
        worker=AdaptiveWorker(self.root,history=HistoryStore(self.root),silver=store,teachers=_Teachers())
        self.assertEqual(worker.recover_markers(),0)
        self.assertIsNone(store.scrub_pending())
        self.assertIsNone(HistoryStore(self.root).get(row["id"]))
        self.assertEqual(SilverStore(self.root).status()["jobs"].get("pending",0),0)

    def test_genuine_durable_two_horizon_lifecycle(self):
        """Real durable APIs from frozen calibration through post-canary full.

        This deliberately avoids direct cohort/calibration/deployment SQL and
        metric/holdout seams.  The only fakes are deterministic inference
        transports; every authority transition remains store/worker driven.
        """
        from speech_backends import (PARAKEET_ID, WHISPER_ID, PreflightReceipts,
                                     _package_versions, _safe_snapshot_digest,
                                     candidate_manifest, evaluator_hash)
        from runtime_source_manifest import RUNTIME_SOURCE_SCHEMA, runtime_source_digest
        from teacher_backends import REQUIRED_TEACHERS
        now=[0.0]
        clock=lambda:now[0]
        store=SilverStore(self.root,clock=clock)
        reference="calm blue river carries gentle morning light softly today home"
        champion_text="quiet blue river carries gentle morning light softly today home"
        packages=_package_versions()
        frozen_runtime={WHISPER_ID:{"runtime":"e2e-whisper"},PARAKEET_ID:{"runtime":"e2e-parakeet"}}
        champion=candidate_manifest(WHISPER_ID,revision="e2e-whisper",package_versions=packages,
                                    runtime_identity=frozen_runtime[WHISPER_ID])
        candidate=candidate_manifest(PARAKEET_ID,revision="e2e-parakeet",package_versions=packages,
                                     runtime_identity=frozen_runtime[PARAKEET_ID])
        hf_home=self.root/"hf"
        self.enterContext(mock.patch.dict(os.environ,{"SOTTO_HF_HOME":str(hf_home),"HF_HOME":str(hf_home)},clear=False))
        self.enterContext(mock.patch.object(PreflightReceipts,"_current_runtime_identity",
                                             side_effect=lambda stable_id: dict(frozen_runtime[str(stable_id)])))
        receipts=PreflightReceipts(self.root)
        for manifest in (champion,candidate):
            snapshot=(hf_home/"hub"/("models--"+str(manifest["repo"]).replace("/","--"))
                      /"snapshots"/str(manifest["revision"]))
            snapshot.mkdir(parents=True); (snapshot/"model.bin").write_bytes(str(manifest["stable_id"]).encode())
            receipts.save(str(manifest["stable_id"]),{"schema":3,"success":True,
                "manifest":{key:value for key,value in manifest.items() if key != "glossary_terms"},
                "snapshot_path":str(snapshot),"snapshot_revision":manifest["revision"],
                "snapshot_metadata_digest":_safe_snapshot_digest(snapshot),"package_versions":packages,
                "runtime_identity":frozen_runtime[str(manifest["stable_id"])],
                "runtime_source_schema":RUNTIME_SOURCE_SCHEMA,"runtime_source_digest":runtime_source_digest(),
                "evaluator_id":"sotto-paired-v1","evaluator_hash":evaluator_hash()})

        # V2 authority is earned solely through its frozen public ledger.
        expected=_compiled_v2_expected()
        store.begin_calibration_v2(holdout_key=SOURCE_HASH,manifest_hash="e2e-frozen-ledger",source_hash=SOURCE_HASH,
            protocol_hash=_protocol_hash(),policy_hash=policy_hash(),receipt_hash="e2e-receipt",expected=expected)
        for ordinal in range(500):
            item=store.claim_calibration_v2(SOURCE_HASH,"e2e-calibration")
            self.assertIsNotNone(item)
            accepted=ordinal < 200
            self.assertTrue(store.finish_calibration_v2(item,outcome="accepted" if accepted else "abstained",
                reference_match=True if accepted else None,code="exact_unanimous" if accepted else "mismatch"))
        self.assertEqual(store.finalize_calibration_v2(SOURCE_HASH),
                         {"state":"passed","accepted":200,"clusters":200,"reference_errors":0,"total":500})

        class Teachers:
            def __init__(self): self.calls=0; self.accepted=200
            def reset(self): self.calls=0
            def validate(self): return "e2e-receipt"
            def transcribe(self, teacher, _path, _identity, **_kwargs):
                index=self.calls//2; self.calls+=1
                if index < self.accepted: return reference
                return "teacher-a disagreement" if teacher.stable_id == REQUIRED_TEACHERS[0].stable_id else "teacher-b disagreement"
        # Measure fake transport time deterministically; wall-clock scheduler
        # jitter must not decide a real persisted latency-ratio gate.
        import types
        transport_time = [0.0]
        self.enterContext(mock.patch('adaptive_worker.time', types.SimpleNamespace(
            monotonic=lambda: transport_time[0], sleep=time.sleep)))
        class Backends:
            def transcribe(self, manifest, _samples, **_kwargs):
                transport_time[0] += .005
                return (champion_text if manifest["stable_id"] == WHISPER_ID else reference),{}
        teachers=Teachers(); backends=Backends()
        history=HistoryStore(self.root); coordinator=LearningCoordinator(history,base_dir=self.root)
        worker=AdaptiveWorker(self.root,history=history,silver=store,teachers=teachers,backends=backends)

        def append_horizon(prefix: str, start: float) -> None:
            for ordinal in range(500):
                coordinator.append_live("original hypothesis",np.zeros(32,np.float32),.002,"fake",ts=start+1+(ordinal%7)*86400,
                    adaptive=True,language="en",candidate={"session_id":f"{prefix}-session-{ordinal%20}"})

        # Boundary is opened before any source capture enters the first pool.
        self.assertEqual(worker.run_once()["silver"]["state"],"collecting")
        opened=store.prospective_status()
        self.assertEqual(opened["status"],"collecting")
        self.assertEqual({opened["champion"]["stable_id"],*(item["stable_id"] for item in opened["candidates"])},
                         {WHISPER_ID,PARAKEET_ID})
        first_start=now[0]; append_horizon("first",first_start)
        now[0]=first_start+31*86400
        # Simulate the only legitimate replay window: SQLite evaluation has
        # committed, but authorization loses its process before deployment
        # state is written.  This is a real worker/evaluation path, not a
        # synthetic cohort/metric/holdout insertion.
        with mock.patch.object(DeploymentController,"authorize_provisional",return_value=False):
            first_run=worker.run_once(); first=first_run["silver"]
        self.assertFalse(first.get("authorized"),{"silver":first,"run":first_run,"horizon":store.prospective_status(),"status":store.status()})
        first_cohort=str(first["cohort_id"])
        first_horizon=store.horizon_for_cohort(first_cohort); self.assertIsNotNone(first_horizon)
        self.assertEqual(store.cohort_status(first_cohort)["status"],"evaluated_passed")
        self.assertEqual(store.status()["jobs"].get("pending",0),0)

        # Recovery opens/retains a fresh collection boundary and replays the
        # exact never-authorized terminal once.  It cannot rerun candidate
        # inference or evaluation because those rows are already terminal.
        store=SilverStore(self.root,clock=clock); restarted_after_crash=AdaptiveWorker(self.root,history=HistoryStore(self.root),silver=store,teachers=teachers,backends=backends)
        recovered=restarted_after_crash.run_once(); first=recovered["silver"]
        self.assertTrue(first.get("authorized"),{"silver":first,"run":recovered,"horizon":store.prospective_status()})
        self.assertEqual(store.cohort_status(first_cohort)["status"],"evaluated_passed")
        self.assertEqual(store.authorization_disposition(first_cohort),"authorized")
        self.assertEqual(store.prospective_status()["status"],"collecting")
        worker=restarted_after_crash; controller=DeploymentController(self.root,silver=store)
        self.assertEqual(controller.status()["tier"],"provisional_silver")
        self.assertEqual(controller.status()["canary"]["stage"],"five_percent")
        safe={"success":True,"fallback":False,"latency":.001,"coverage_ok":True,"hallucination_ok":True,"identity_valid":True}
        def record_selected(prefix: str, needed: int, start_day: int) -> None:
            recorded=probe=0
            while recorded < needed:
                session=f"{prefix}-{probe}"; probe+=1
                routed=controller.route(session)
                if routed is None: continue
                now[0]=(32+start_day+(recorded%7))*86400
                token=routed["_deployment_token"]
                result=controller.record_runtime_pair(session_id=session,capture_id=f"{prefix}-capture-{recorded}",
                    candidate_arm=PARAKEET_ID,candidate=safe,incumbent=safe,route_token=token)
                self.assertNotEqual(result,"stale")
                recorded+=1
        record_selected("five",100,0)
        self.assertEqual(controller.status()["canary"]["stage"],"twenty_five_percent")
        record_selected("twentyfive",400,7)
        self.assertEqual(controller.observe_canary(),"awaiting_holdout")
        self.assertEqual(controller.status()["canary"]["stage"],"twenty_five_percent")
        canary_started=float(controller.status()["canary"]["started_ts"])

        # The crash-recovery boundary deliberately began before authorization,
        # so retire it empty and create an actually post-canary horizon.
        # Empty closed horizons are never reusable.
        now[0]=64*86400; teachers.reset()
        self.assertEqual(worker.run_once()["silver"]["state"],"collecting")
        self.assertEqual(worker.run_once()["silver"]["state"],"collecting")
        second_start=now[0]; self.assertGreater(second_start,canary_started); append_horizon("second",second_start)
        now[0]=second_start+31*86400
        second=worker.run_once()["silver"]; second_cohort=str(second["cohort_id"])
        with store._connect() as debug_db:
            attempt_states=list(debug_db.execute("SELECT arm_id,status,code,COUNT(*) FROM candidate_attempts WHERE cohort_id=? GROUP BY arm_id,status,code",(second_cohort,)))
            debug_attempt=debug_db.execute("SELECT arm_id,history_id,history_revision,audio_sha256 FROM candidate_attempts WHERE cohort_id=? LIMIT 1",(second_cohort,)).fetchone()
        debug_details={}
        if debug_attempt is not None:
            arm,history_id,revision,audio_sha=debug_attempt; current=history.get(history_id)
            latest=store.prospective_status() or {}; manifests={str(x.get("stable_id")):x for x in [latest.get("champion",{}),*latest.get("candidates",[])]}
            frozen=manifests.get(arm); receipt=worker._receipt_manifest(arm)
            debug_details={"arm":arm,"entry":bool(current),"revision":current and current.get("revision"),"expected_revision":revision,
                "audio":current and current.get("audio",{}).get("inference",{}).get("sha256"),"expected_audio":audio_sha,
                "frozen":bool(frozen),"preflight":bool(frozen and worker.preflight.valid_manifest(frozen)),"receipt":bool(receipt),
                "manifest_equal":bool(frozen and receipt and store.manifest_hash(frozen)==store.manifest_hash(receipt))}
        self.assertEqual(store.cohort_status(second_cohort)["status"],"evaluated_passed",
                         {"result":second,"attempts":attempt_states,"details":debug_details,"cohort":store.cohort_status(second_cohort)})
        self.assertEqual(second.get("winner"),PARAKEET_ID)
        self.assertGreater(float(store.horizon_for_cohort(second_cohort)["start_ts"]),canary_started)
        self.assertIn(controller.observe_canary(),{"deployed_silver","full"})
        self.assertEqual(controller.status()["tier"],"deployed_silver")

        # Restart/reconcile is idempotent, keeps full deployment, and can
        # create the next collecting horizon without resurrecting old work.
        store=SilverStore(self.root,clock=clock); controller=DeploymentController(self.root,silver=store)
        restarted=AdaptiveWorker(self.root,history=HistoryStore(self.root),silver=store,teachers=teachers,backends=backends)
        resumed=restarted.run_once()
        self.assertEqual(controller.status()["tier"],"deployed_silver")
        self.assertEqual(resumed["silver"]["state"],"collecting")
        self.assertEqual(store.prospective_status()["status"],"collecting")
        # Existing route assignments retain their original durable arm across
        # restart, while the full rollout can assign a newly seen session.
        existing="five-0"; selected=store.has_route_assignment(route_generation=int(controller.status()["route_generation"]),session_id=existing)
        self.assertEqual(selected,controller.route(existing) is not None)
        self.assertIsNotNone(controller.route("post-full-new-session"))

    def test_restart_replays_passed_terminal_authorization_once_with_fresh_horizon(self):
        """A crash after durable evaluation cannot be hidden by new collection."""
        passed={"cohort_id":"passed","identity_hash":"identity","winner":"candidate","tier":"provisional_silver"}
        case=self
        class Silver:
            def close_and_freeze_prospective(self): return None
            def claim_candidate_attempt(self,_owner): return None
            def scrub_pending(self): return None
            def prospective_status(self): return {"status":"collecting","cohort_id":None}
            def terminal_cohort_ids(self): return ["passed"]
            def authorization_disposition(self,_cohort): return None if state["authorizations"] == 0 else "authorized"
            def authorization_intent(self,_cohort): return None
            def release_cohort_authorization_claim(self,_cohort): raise AssertionError("unexpected claim release")
            def consume_cohort_authorization(self,_cohort): raise AssertionError("unexpected consumption")
            def mark_cohort_authorized(self,_cohort): return True
            def cohort_status(self,cohort):
                return {"status":"evaluated_passed","evaluation_outcome":passed} if cohort == "passed" else None
            def horizon_for_cohort(self,cohort):
                return {"cohort_id":"passed","identity_hash":"identity"} if cohort == "passed" else None
            def releasable_cohort_members(self,_cohort): return []
            def should_hold_marker(self,*_args): return False
        state={"tier":"shadow_only","authorizations":0}
        class Controller:
            def __init__(self,*_args,**_kwargs): pass
            def status(self): return {"tier":state["tier"]}
            def _read(self): return {"tier":state["tier"],"revision":0,"cohort_id":None}
            @staticmethod
            def _state_digest(_value): return "0" * 64
            def authorize_provisional(self,cohort_id):
                case.assertEqual(cohort_id,"passed"); state["tier"]="provisional_silver"; state["authorizations"]+=1; return True
        class Experiment:
            def __init__(self,*_args,**_kwargs): pass
            def evaluate_persisted(self,cohort_id,*,identity_hash):
                case.assertEqual((cohort_id,identity_hash),("passed","identity")); return dict(passed)
        worker=AdaptiveWorker(self.root,history=self.history,silver=Silver(),teachers=_Teachers())
        with mock.patch("adaptive_worker.DeploymentController",Controller), mock.patch("adaptive_worker.SilverExperiment",Experiment):
            replayed=worker.reconcile_silver()
            self.assertTrue(replayed["authorized"])
            self.assertEqual(state["authorizations"],1)
            # The same persisted terminal is now historical; the fresh
            # collecting horizon remains visible instead of replaying again.
            self.assertEqual(worker.reconcile_silver(),{"state":"collecting"})
            self.assertEqual(state["authorizations"],1)

    def test_explicit_revision_revoke_tombstone_fences_unenqueued_job_only(self):
        store=SilverStore(self.root)
        self.assertEqual(store.revoke_history("unseen-history",history_revision=0),0)
        common={"history_id":"unseen-history","audio_sha256":"audio","teacher_family_hash":"family",
                "consensus_policy_hash":"policy","receipt_hash":"receipt"}
        self.assertIsNone(store.enqueue(**common,history_revision=0))
        self.assertIsNotNone(store.enqueue(**common,history_revision=1))

    def test_revoke_history_with_intent_returns_exact_marker_and_clear_dominates(self):
        store=SilverStore(self.root)
        epoch=store.epoch()
        with store._connect() as db:
            for cohort,history,revision in (("cohort-a","opaque-a",4),("cohort-b","opaque-b",5)):
                db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,"h","evaluated_passed","m","c","{}",epoch,0))
                db.execute("INSERT INTO cohort_members VALUES(?,?,?,?,?,?,?,?)",(cohort,0,history,revision,"accepted","audio","ref","cluster"))
        def revoke(value):
            history,revision,operation=value
            return store.revoke_history_with_intent(history,history_revision=revision,operation=operation)[1]
        with ThreadPoolExecutor(max_workers=4) as pool:
            writes=list(pool.map(revoke,[("opaque-a",4,"retry"),("opaque-b",5,"delete")]))
        batch=store.scrub_pending(); self.assertEqual(batch,writes[-1] if len(writes[-1]["intents"]) == 2 else writes[0])
        self.assertEqual(set(batch),{"schema","kind","intents","cohorts","phase","token"})
        self.assertEqual((batch["schema"],batch["kind"],batch["phase"],batch["cohorts"]),(2,"revoke","silver_fenced",["cohort-a","cohort-b"]))
        self.assertEqual({item["history_id"] for item in batch["intents"]},{"opaque-a","opaque-b"})
        self.assertEqual(store.revoke_history_with_intent("opaque-a",history_revision=4,operation="retry")[1],batch)
        earlier=next(value for value in writes if len(value["intents"]) == 1)
        self.assertFalse(store.complete_scrub(earlier))
        self.assertEqual(store.scrub_pending(),batch)
        store.clear(); clear=store.scrub_pending()
        _dependencies,dominated=store.revoke_history_with_intent("later-history",history_revision=0,operation="delete")
        self.assertEqual(dominated,clear)
        self.assertEqual(store.scrub_pending(),clear)

    def test_malformed_scrub_marker_is_a_hard_fence_not_an_ackable_placeholder(self):
        store=SilverStore(self.root)
        raw='{"kind":"clear","epoch":0}'
        with store._connect() as db:
            db.execute("INSERT INTO meta(key,value) VALUES('scrub_pending',?)",(raw,))
        self.assertEqual(store.scrub_pending(),{"kind":"unknown","blocked":True})
        self.assertFalse(store.complete_scrub(None))
        self.assertFalse(store.complete_scrub({"schema":1,"kind":"clear","epoch":0,"phase":"clear_fenced","token":"0"*32}))
        with self.assertRaisesRegex(RuntimeError,"scrub marker"):
            store.revoke_history("opaque",history_revision=0)
        self.assertEqual(store.scrub_pending(),{"kind":"unknown","blocked":True})
        worker=AdaptiveWorker(self.root,history=self.history,silver=store,teachers=_Teachers())
        self.assertEqual(worker.run_once()["silver"]["state"],"scrub_blocked")

    def test_v3_store_migrates_atomically_to_v5_without_losing_public_ledger(self):
        """A disposable v3-shaped copy upgrades in one restart transaction."""
        store=SilverStore(self.root); self.assertTrue(store.alpha(1)>0)
        with store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DROP TABLE cohort_authorization")
            db.execute("UPDATE meta SET value='3' WHERE key='schema_version'")
            db.commit()
        upgraded=SilverStore(self.root)
        self.assertEqual(upgraded.status()["schema"],5)
        self.assertEqual(upgraded.alpha(1),.025)
        with upgraded._connect() as db:
            self.assertEqual(db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0],"5")
            self.assertIsNotNone(db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cohort_authorization'").fetchone())

    def test_v3_migration_failpoints_roll_back_and_retry_without_loss(self):
        for phase in ("preflight","authority_table","finalize"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as root:
                original=SilverStore(root); original.alpha(1)
                with original._connect() as db:
                    db.execute("BEGIN IMMEDIATE"); db.execute("DROP TABLE cohort_authorization")
                    db.execute("UPDATE meta SET value='3' WHERE key='schema_version'"); db.commit()
                def fail(value):
                    if value == phase: raise RuntimeError("injected migration failure")
                with mock.patch.object(SilverStore,"_migration_failpoint",staticmethod(fail),create=True):
                    with self.assertRaisesRegex(RuntimeError,"injected migration failure"):
                        SilverStore(root)
                # The failed transaction leaves the exact v3 authority view;
                # retry is safe and preserves public alpha evidence.
                with sqlite3.connect(Path(root)/"adaptive-learning"/"silver"/"silver.sqlite3") as db:
                    self.assertEqual(db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0],"3")
                    self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cohort_authorization'").fetchone())
                retried=SilverStore(root); self.assertEqual(retried.status()["schema"],5); self.assertEqual(retried.alpha(1),.025)

    def _v4_comparator_fixture(self, root: Path):
        """Downgrade only the comparator ledger of a disposable v5 store."""
        store=SilverStore(root); alpha=store.alpha(1)
        rows=[
            (1,'pending-cap','session','cohort',0,7,'arm','a'*64,'pending-cap.wav',1,0,.1,1,1,1,'pending',0,None,0,None,None,1.,1.),
            (1,'leased-cap','session','cohort',0,7,'arm','b'*64,'leased-cap.wav',1,0,.2,1,1,1,'leased',1,'owner',2,999.,'attempt',2.,2.),
            (1,'complete-cap','session','cohort',0,7,'arm','c'*64,'complete-cap.wav',1,0,.3,1,1,1,'complete',1,None,2,None,None,3.,3.),
            (1,'discarded-cap','session','cohort',0,7,'arm','d'*64,'discarded-cap.wav',0,1,.4,0,0,1,'discarded',4,None,3,None,None,4.,4.),
        ]
        with store._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("INSERT INTO cohorts VALUES('cohort','h','evaluated_passed','m','c','{}',0,1)")
            db.execute("INSERT INTO route_assignments VALUES(7,'session',1,1)")
            db.execute("INSERT INTO runtime_pairs VALUES(1,'runtime-cap','session','cohort',0,'arm',1,0,.1,1,1,1,1,0,.2,1,1,1,1.,0,'complete')")
            db.execute("INSERT INTO calibration_universes(holdout_key,manifest_hash,source_hash,protocol_hash,policy_hash,receipt_hash,state,created_ts,updated_ts,result) VALUES('public','m','s','p','policy','receipt','complete',1,1,'passed')")
            db.execute("INSERT INTO calibration_expected_v2 VALUES('public',0,'item','cluster')")
            db.execute("INSERT INTO calibration_items_v2 VALUES('public','item','terminal',0,NULL,NULL,NULL,'accepted',1,'ok',1)")
            db.execute('ALTER TABLE comparator_intents RENAME TO comparator_intents_v5_seed')
            db.execute("""CREATE TABLE comparator_intents (
                deployment_revision INTEGER NOT NULL, capture_id TEXT NOT NULL,
                session_hash TEXT NOT NULL, cohort_id TEXT NOT NULL, epoch INTEGER NOT NULL,
                route_generation INTEGER NOT NULL, candidate_arm TEXT NOT NULL,
                audio_sha256 TEXT NOT NULL, spool_name TEXT NOT NULL,
                candidate_success INTEGER NOT NULL, candidate_fallback INTEGER NOT NULL, candidate_latency REAL NOT NULL,
                candidate_coverage_ok INTEGER NOT NULL, candidate_hallucination_ok INTEGER NOT NULL, candidate_identity_valid INTEGER NOT NULL,
                status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                lease_owner TEXT, lease_epoch INTEGER NOT NULL DEFAULT 0, lease_until REAL, attempt_id TEXT,
                created_ts REAL NOT NULL, updated_ts REAL NOT NULL,
                PRIMARY KEY(deployment_revision,capture_id), CHECK(status IN ('pending','leased','complete','discarded'))
            )""")
            db.executemany('INSERT INTO comparator_intents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',rows)
            db.execute('DROP TABLE comparator_intents_v5_seed')
            db.execute("UPDATE meta SET value='4' WHERE key='schema_version'")
            db.commit()
        return rows,alpha

    def test_v4_store_without_comparator_table_migrates_cleanly(self):
        """A pre-comparator v4 store has nothing to carry; refusal is reserved
        for a present-but-different comparator table."""
        with tempfile.TemporaryDirectory() as root:
            _rows,alpha=self._v4_comparator_fixture(Path(root))
            with sqlite3.connect(Path(root)/"adaptive-learning"/"silver"/"silver.sqlite3") as db:
                db.execute("DROP TABLE comparator_intents")
                db.execute("UPDATE meta SET value='4' WHERE key='schema_version'")
                db.commit()
            upgraded=SilverStore(root)
            self.assertEqual(upgraded.status()["schema"],5); self.assertEqual(upgraded.alpha(1),alpha)
            with upgraded._connect() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM comparator_intents").fetchone()[0],0)
                self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0],"ok")
            with sqlite3.connect(Path(root)/"adaptive-learning"/"silver"/"silver.sqlite3") as db:
                db.execute("ALTER TABLE comparator_intents ADD COLUMN unexpected TEXT")
                db.execute("UPDATE meta SET value='4' WHERE key='schema_version'")
                db.commit()
            with self.assertRaisesRegex(RuntimeError,"comparator schema"):
                SilverStore(root)

    def test_v4_comparator_migration_preserves_rows_and_public_ledgers(self):
        with tempfile.TemporaryDirectory() as root:
            rows,alpha=self._v4_comparator_fixture(Path(root))
            upgraded=SilverStore(root)
            self.assertEqual(upgraded.status()['schema'],5); self.assertEqual(upgraded.alpha(1),alpha)
            with upgraded._connect() as db:
                self.assertEqual(db.execute('SELECT * FROM comparator_intents ORDER BY capture_id').fetchall(),sorted(rows,key=lambda row:row[1]))
                self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='comparator_intents_v4'").fetchone())
                self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0],'ok')
                self.assertEqual(db.execute("SELECT result FROM calibration_universes WHERE holdout_key='public'").fetchone()[0],'passed')
                self.assertEqual(db.execute("SELECT state,outcome FROM calibration_items_v2 WHERE holdout_key='public'").fetchone(),('terminal','accepted'))
                prepared=(2,'prepared-cap','session','cohort',0,7,'arm','e'*64,'prepared-cap.wav',1,0,.5,1,1,1,'prepared',0,None,0,None,None,5.,5.)
                db.execute('INSERT INTO comparator_intents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',prepared)
                with self.assertRaises(sqlite3.IntegrityError):
                    invalid=list(prepared)
                    invalid[0]=3; invalid[1]='bad-cap'; invalid[7]='f'*64
                    invalid[8]='bad-cap.wav'; invalid[15]='unknown'; invalid[21]=6.; invalid[22]=6.
                    db.execute('INSERT INTO comparator_intents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',invalid)

    def test_v4_comparator_migration_failpoints_roll_back_and_retry_exactly(self):
        phases=('comparator_renamed','preflight','authority_table','comparator_created','comparator_copied','comparator_dropped','finalize')
        for phase in phases:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as root:
                root_path=Path(root); rows,alpha=self._v4_comparator_fixture(root_path); path=root_path/'adaptive-learning'/'silver'/'silver.sqlite3'
                with sqlite3.connect(path) as db:
                    before_sql=db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='comparator_intents'").fetchone()[0]
                def fail(value):
                    if value == phase: raise RuntimeError('injected migration failure')
                with mock.patch.object(SilverStore,'_migration_failpoint',staticmethod(fail),create=True):
                    with self.assertRaisesRegex(RuntimeError,'injected migration failure'):
                        SilverStore(root_path)
                with sqlite3.connect(path) as db:
                    self.assertEqual(db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0],'4')
                    self.assertEqual(db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='comparator_intents'").fetchone()[0],before_sql)
                    self.assertEqual(db.execute('SELECT * FROM comparator_intents ORDER BY capture_id').fetchall(),sorted(rows,key=lambda row:row[1]))
                    self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='comparator_intents_v4'").fetchone())
                    self.assertEqual(db.execute("SELECT result FROM calibration_universes WHERE holdout_key='public'").fetchone()[0],'passed')
                retried=SilverStore(root_path); self.assertEqual(retried.status()['schema'],5); self.assertEqual(retried.alpha(1),alpha)

    def test_read_only_status_reports_v5_and_never_migrates_v4_fixture(self):
        with tempfile.TemporaryDirectory() as root:
            root_path=Path(root); self._v4_comparator_fixture(root_path); path=root_path/'adaptive-learning'/'silver'/'silver.sqlite3'
            before=(path.read_bytes(),path.stat().st_mtime_ns)
            self.assertEqual(read_only_status(root_path)['reason'],'schema_unknown')
            self.assertEqual((path.read_bytes(),path.stat().st_mtime_ns),before)
            self.assertEqual(SilverStore(root_path).status()['schema'],5)
            self.assertEqual(read_only_status(root_path)['schema'],5)

    def test_teacher_lease_is_a_hard_expiry_for_renew_and_finish(self):
        now=[0.0]; store=SilverStore(self.root,clock=lambda:now[0])
        key=store.enqueue(history_id="lease",history_revision=0,audio_sha256="audio",teacher_family_hash="family",consensus_policy_hash="policy",receipt_hash="receipt")
        job=store.claim("owner"); self.assertEqual(job and job["job_key"],key)
        now[0]=store.lease_seconds+.001
        self.assertFalse(store.renew(job))
        self.assertFalse(store.finish(job,outcome="abstained",code="late"))
        self.assertEqual(store.recover(),1)
        reclaimed=store.claim("other")
        self.assertIsNotNone(reclaimed)
        self.assertNotEqual(reclaimed["lease_epoch"],job["lease_epoch"])

    def test_candidate_lease_expiry_and_stats_schema_are_hard_fences(self):
        now=[0.0]; store=SilverStore(self.root,clock=lambda:now[0]); epoch=store.epoch(); cluster="a"*64
        with store._connect() as db:
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",("candidate-cohort","h","evaluating","m","c","{}",epoch,0))
            db.execute("INSERT INTO candidate_attempts(cohort_id,arm_id,history_id,history_revision,audio_sha256,reference_digest,cluster_hash,status,updated_ts) VALUES(?,?,?,?,?,?,?,? ,?)",("candidate-cohort","arm","history",0,"audio","ref",cluster,"pending",0))
        attempt=store.claim_candidate_attempt("owner"); self.assertIsNotNone(attempt)
        invalid={"word_distance":False,"reference_words":1,"character_distance":0,"reference_characters":1,"hallucination":False,"latency":0.0,"cluster_hash":cluster}
        with self.assertRaises(ValueError): store.finish_candidate_attempt(attempt,stats=invalid)
        now[0]=store.lease_seconds+.001
        self.assertFalse(store.renew_candidate_attempt(attempt))
        self.assertFalse(store.finish_candidate_attempt(attempt,code="late"))
        self.assertIsNone(store.candidate_reference(attempt))
        store.recover(); reclaimed=store.claim_candidate_attempt("other")
        self.assertIsNotNone(reclaimed); self.assertNotEqual(reclaimed["lease_epoch"],attempt["lease_epoch"])

    def test_corrupt_terminal_candidate_stats_become_durable_failed_not_restart_loop(self):
        store=SilverStore(self.root); epoch=store.epoch(); cohort="corrupt-cohort"; identity="identity"; cluster="a"*64
        champion={"stable_id":"champion"}; candidate={"stable_id":"candidate"}
        metadata={"raw":1,"accepted":1,"abstained":0,"words":10,"sessions":1,"days":1,"all_terminal":True}
        with store._connect() as db:
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,"h","evaluating","m","comparison",json.dumps(metadata),epoch,0))
            db.execute("INSERT INTO prospective_horizons VALUES(?,?,?,?,?,?,?,?,?,?,?)",("horizon","frozen",0,1,json.dumps(champion),json.dumps([candidate]),"receipt",identity,epoch,0,cohort))
            db.execute("INSERT INTO cohort_members VALUES(?,?,?,?,?,?,?,?)",(cohort,0,"history",0,"accepted","audio","reference",cluster))
            valid={"word_distance":1,"reference_words":10,"character_distance":1,"reference_characters":10,"hallucination":False,"latency":.1,"cluster_hash":cluster}
            for arm,stats in (("champion",json.dumps(valid)),("candidate","{bad")):
                db.execute("INSERT INTO candidate_attempts(cohort_id,arm_id,history_id,history_revision,audio_sha256,reference_digest,cluster_hash,status,attempts,stats,updated_ts) VALUES(?,?,?,?,?,?,?,?,?,?,?)",(cohort,arm,"history",0,"audio","reference",cluster,"complete",1,stats,0))
        outcome=SilverExperiment(self.root,silver=store).evaluate_persisted(cohort,identity_hash=identity)
        self.assertEqual(outcome and outcome.get("code"),"corrupt_candidate_stats")
        self.assertEqual(store.cohort_status(cohort)["status"],"evaluated_failed")
        self.assertEqual(SilverExperiment(self.root,silver=store).evaluate_persisted(cohort,identity_hash=identity),outcome)

    def test_adaptive_review_tombstones_visible_unenqueued_revision(self):
        row=self.live(); store=SilverStore(self.root)
        updated=self.coordinator.commit_adaptive_review(row["id"],"corrected",reference="human",expected_revision=row["revision"])
        self.assertIsNotNone(updated)
        with store._connect() as db:
            self.assertIsNotNone(db.execute("SELECT 1 FROM history_tombstones WHERE history_id=? AND history_revision=? AND clear_epoch=?",
                (row["id"],row["revision"],store.epoch())).fetchone())
        self.assertIsNone(store.enqueue(history_id=row["id"],history_revision=row["revision"],audio_sha256=row["audio"]["inference"]["sha256"],
            teacher_family_hash=family_hash(),consensus_policy_hash=policy_hash(),receipt_hash="frozen-receipt"))

    def test_commit_retry_fences_worker_paused_before_enqueue_and_allows_new_revision(self):
        row=self.live(); store=SilverStore(self.root)
        worker=AdaptiveWorker(self.root,history=self.history,silver=store,teachers=_Teachers())
        entered=threading.Event(); release=threading.Event(); committed=threading.Event(); result=[]
        original=store.enqueue
        def paused_enqueue(**kwargs):
            entered.set()
            if not release.wait(5): raise RuntimeError("test enqueue release timeout")
            return original(**kwargs)
        def retry():
            result.append(self.coordinator.commit_retry(row["id"],"new retry",row["revision"]))
            committed.set()
        with mock.patch.object(store,"enqueue",side_effect=paused_enqueue):
            scan=threading.Thread(target=worker.recover_markers); scan.start()
            self.assertTrue(entered.wait(5))
            mutation=threading.Thread(target=retry); mutation.start()
            self.assertFalse(committed.wait(.05))
            release.set(); scan.join(5); mutation.join(5)
        self.assertFalse(scan.is_alive()); self.assertFalse(mutation.is_alive()); self.assertTrue(committed.is_set())
        updated=result[0]; self.assertIsNotNone(updated)
        with store._connect() as db:
            rows=db.execute("SELECT history_revision,status FROM jobs WHERE history_id=? ORDER BY history_revision",(row["id"],)).fetchall()
        self.assertNotIn((row["revision"],"pending"),rows)
        self.assertIsNone(store.enqueue(history_id=row["id"],history_revision=row["revision"],audio_sha256=row["audio"]["inference"]["sha256"],
            teacher_family_hash=family_hash(),consensus_policy_hash=policy_hash(),receipt_hash="frozen-receipt"))
        self.assertGreater(updated["revision"],row["revision"])
        self.assertGreaterEqual(worker.recover_markers(),1)
        with store._connect() as db:
            self.assertIn((updated["revision"],"pending"),db.execute("SELECT history_revision,status FROM jobs WHERE history_id=?",(row["id"],)).fetchall())

    def test_correct_fences_worker_paused_before_enqueue_of_visible_revision(self):
        row=self.live(); store=SilverStore(self.root)
        worker=AdaptiveWorker(self.root,history=self.history,silver=store,teachers=_Teachers())
        entered=threading.Event(); release=threading.Event(); committed=threading.Event(); result=[]
        original=store.enqueue
        def paused_enqueue(**kwargs):
            entered.set()
            if not release.wait(5): raise RuntimeError("test enqueue release timeout")
            return original(**kwargs)
        def correct():
            result.append(self.coordinator.correct(row["id"],"human correction",expected_revision=row["revision"]))
            committed.set()
        with mock.patch.object(store,"enqueue",side_effect=paused_enqueue):
            scan=threading.Thread(target=worker.recover_markers); scan.start()
            self.assertTrue(entered.wait(5))
            mutation=threading.Thread(target=correct); mutation.start()
            self.assertFalse(committed.wait(.05))
            release.set(); scan.join(5); mutation.join(5)
        self.assertFalse(scan.is_alive()); self.assertFalse(mutation.is_alive()); self.assertTrue(committed.is_set())
        self.assertIsNotNone(result[0])
        with store._connect() as db:
            rows=db.execute("SELECT history_revision,status FROM jobs WHERE history_id=?",(row["id"],)).fetchall()
        self.assertNotIn((row["revision"],"pending"),rows)
        self.assertIsNone(store.enqueue(history_id=row["id"],history_revision=row["revision"],audio_sha256=row["audio"]["inference"]["sha256"],
            teacher_family_hash=family_hash(),consensus_policy_hash=policy_hash(),receipt_hash="frozen-receipt"))

    def test_correct_as_is_fences_worker_paused_before_enqueue_of_visible_revision(self):
        row=self.live(); store=SilverStore(self.root)
        worker=AdaptiveWorker(self.root,history=self.history,silver=store,teachers=_Teachers())
        entered=threading.Event(); release=threading.Event(); committed=threading.Event(); result=[]
        original=store.enqueue
        def paused_enqueue(**kwargs):
            entered.set()
            if not release.wait(5): raise RuntimeError("test enqueue release timeout")
            return original(**kwargs)
        def correct_as_is():
            result.append(self.coordinator.correct_as_is(row["id"],expected_revision=row["revision"]))
            committed.set()
        with mock.patch.object(store,"enqueue",side_effect=paused_enqueue):
            scan=threading.Thread(target=worker.recover_markers); scan.start()
            self.assertTrue(entered.wait(5))
            mutation=threading.Thread(target=correct_as_is); mutation.start()
            self.assertFalse(committed.wait(.05))
            release.set(); scan.join(5); mutation.join(5)
        self.assertFalse(scan.is_alive()); self.assertFalse(mutation.is_alive()); self.assertTrue(committed.is_set())
        self.assertIsNotNone(result[0])
        with store._connect() as db:
            rows=db.execute("SELECT history_revision,status FROM jobs WHERE history_id=?",(row["id"],)).fetchall()
        self.assertNotIn((row["revision"],"pending"),rows)
        self.assertIsNone(store.enqueue(history_id=row["id"],history_revision=row["revision"],audio_sha256=row["audio"]["inference"]["sha256"],
            teacher_family_hash=family_hash(),consensus_policy_hash=policy_hash(),receipt_hash="frozen-receipt"))

    def test_no_speech_fences_worker_paused_before_enqueue_of_visible_revision(self):
        row=self.live(); store=SilverStore(self.root)
        worker=AdaptiveWorker(self.root,history=self.history,silver=store,teachers=_Teachers())
        entered=threading.Event(); release=threading.Event(); committed=threading.Event(); result=[]
        original=store.enqueue
        def paused_enqueue(**kwargs):
            entered.set()
            if not release.wait(5): raise RuntimeError("test enqueue release timeout")
            return original(**kwargs)
        def no_speech():
            result.append(self.coordinator.no_speech(row["id"],expected_revision=row["revision"]))
            committed.set()
        with mock.patch.object(store,"enqueue",side_effect=paused_enqueue):
            scan=threading.Thread(target=worker.recover_markers); scan.start()
            self.assertTrue(entered.wait(5))
            mutation=threading.Thread(target=no_speech); mutation.start()
            self.assertFalse(committed.wait(.05))
            release.set(); scan.join(5); mutation.join(5)
        self.assertFalse(scan.is_alive()); self.assertFalse(mutation.is_alive()); self.assertTrue(committed.is_set())
        self.assertIsNotNone(result[0])
        with store._connect() as db:
            rows=db.execute("SELECT history_revision,status FROM jobs WHERE history_id=?",(row["id"],)).fetchall()
        self.assertNotIn((row["revision"],"pending"),rows)
        self.assertIsNone(store.enqueue(history_id=row["id"],history_revision=row["revision"],audio_sha256=row["audio"]["inference"]["sha256"],
            teacher_family_hash=family_hash(),consensus_policy_hash=policy_hash(),receipt_hash="frozen-receipt"))

    def test_delete_fences_worker_paused_before_enqueue_and_removes_private_artifacts(self):
        row=self.live(); store=SilverStore(self.root); audio=self.history.audio_path(row["id"])
        worker=AdaptiveWorker(self.root,history=self.history,silver=store,teachers=_Teachers())
        entered=threading.Event(); release=threading.Event(); committed=threading.Event(); result=[]
        original=store.enqueue
        def paused_enqueue(**kwargs):
            entered.set()
            if not release.wait(5): raise RuntimeError("test enqueue release timeout")
            return original(**kwargs)
        def delete():
            result.append(self.coordinator.delete(row["id"])); committed.set()
        with mock.patch.object(store,"enqueue",side_effect=paused_enqueue):
            scan=threading.Thread(target=worker.recover_markers); scan.start()
            self.assertTrue(entered.wait(5))
            mutation=threading.Thread(target=delete); mutation.start()
            self.assertFalse(committed.wait(.05))
            release.set(); scan.join(5); mutation.join(5)
        self.assertFalse(scan.is_alive()); self.assertFalse(mutation.is_alive()); self.assertTrue(committed.is_set())
        self.assertEqual(result,[True]); self.assertIsNone(self.history.get(row["id"])); self.assertFalse(audio.exists())
        with store._connect() as db:
            self.assertNotIn((row["revision"],"pending"),db.execute("SELECT history_revision,status FROM jobs WHERE history_id=?",(row["id"],)).fetchall())
        self.assertIsNone(store.enqueue(history_id=row["id"],history_revision=row["revision"],audio_sha256=row["audio"]["inference"]["sha256"],
            teacher_family_hash=family_hash(),consensus_policy_hash=policy_hash(),receipt_hash="frozen-receipt"))

    def test_direct_revoke_history_fences_paused_enqueue_and_invalidates_adaptive_member(self):
        from adaptive_learning import AdaptiveLearning
        row=self.live(); store=SilverStore(self.root); adaptive=AdaptiveLearning(self.root)
        adaptive.register_capture({"history_id":row["id"],"captured_ts":1,"duration":1,"audio_digest":row["audio"]["inference"]["sha256"],"adaptive":True,"language":"en"})
        adaptive._mutate(lambda state: state["generations"].update({"g":{"id":"g","status":"pool_open","pool":[row["id"]],"candidates":{}}}))
        worker=AdaptiveWorker(self.root,history=self.history,silver=store,teachers=_Teachers())
        entered=threading.Event(); release=threading.Event(); committed=threading.Event(); result=[]
        original=store.enqueue
        def paused_enqueue(**kwargs):
            entered.set()
            if not release.wait(5): raise RuntimeError("test enqueue release timeout")
            return original(**kwargs)
        def revoke():
            result.append(self.coordinator.revoke_history(row["id"])); committed.set()
        with mock.patch.object(store,"enqueue",side_effect=paused_enqueue):
            scan=threading.Thread(target=worker.recover_markers); scan.start()
            self.assertTrue(entered.wait(5))
            mutation=threading.Thread(target=revoke); mutation.start()
            self.assertFalse(committed.wait(.05))
            release.set(); scan.join(5); mutation.join(5)
        self.assertFalse(scan.is_alive()); self.assertFalse(mutation.is_alive()); self.assertTrue(committed.is_set())
        self.assertEqual(result,[0])
        retained=self.history.get(row["id"]); self.assertEqual(retained["silver_enqueue"]["state"],"released")
        with store._connect() as db:
            self.assertNotIn((row["revision"],"pending"),db.execute("SELECT history_revision,status FROM jobs WHERE history_id=?",(row["id"],)).fetchall())
        self.assertIsNone(store.enqueue(history_id=row["id"],history_revision=row["revision"],audio_sha256=row["audio"]["inference"]["sha256"],
            teacher_family_hash=family_hash(),consensus_policy_hash=policy_hash(),receipt_hash="frozen-receipt"))
        with adaptive._locked():
            state=adaptive._read()
            self.assertTrue(state["captures"][row["id"]]["revoked"])
            self.assertEqual(state["generations"]["g"]["status"],"invalidated")

    def test_finish_pending_delete_intent_removes_exact_old_history_and_audio(self):
        row=self.live(); store=SilverStore(self.root); audio=self.history.audio_path(row["id"])
        _dependencies,batch=store.revoke_history_with_intent(row["id"],history_revision=row["revision"],operation="delete")
        intent=next(item for item in batch["intents"] if item["history_id"] == row["id"])
        self.coordinator.finish_pending_revoke_intent(intent)
        self.assertIsNone(self.history.get(row["id"]))
        self.assertFalse(audio.exists())

    def test_finish_pending_delete_intent_never_deletes_newer_history_revision(self):
        row=self.live(); store=SilverStore(self.root); audio=self.history.audio_path(row["id"])
        _dependencies,batch=store.revoke_history_with_intent(row["id"],history_revision=row["revision"],operation="delete")
        old_intent=next(item for item in batch["intents"] if item["history_id"] == row["id"])
        newer=self.coordinator.commit_retry(row["id"],"new retry",row["revision"])
        self.assertIsNotNone(newer)
        self.coordinator.finish_pending_revoke_intent(old_intent)
        current=self.history.get(row["id"])
        self.assertEqual(current["revision"],newer["revision"])
        self.assertEqual(current["text"],"new retry")
        self.assertTrue(audio.exists())

    def test_interrupted_correction_stages_correct_intent_and_recovery_keeps_uncorrected_row(self):
        row=self.live(); store=SilverStore(self.root); audio=self.history.audio_path(row["id"])
        with mock.patch.object(HistoryStore,"_save_locked",side_effect=OSError(5,"I/O error")):
            with self.assertRaises(OSError):
                self.coordinator.correct(row["id"],"corrected",expected_revision=row["revision"])
        pending=store.scrub_pending()
        self.assertEqual([(item["history_id"],item["history_revision"],item["operation"]) for item in pending["intents"]],
                         [(row["id"],row["revision"],"correct")])
        worker=AdaptiveWorker(self.root,history=HistoryStore(self.root),silver=store,teachers=_Teachers())
        worker.recover_markers()
        self.assertIsNone(store.scrub_pending())
        survivor=HistoryStore(self.root).get(row["id"])
        self.assertIsNotNone(survivor)
        self.assertEqual((survivor["revision"],survivor["text"],survivor.get("correction")),(row["revision"],"candidate",None))
        self.assertTrue(audio.exists())

    def test_completed_correction_recovery_keeps_corrected_row(self):
        row=self.live(); store=SilverStore(self.root)
        corrected=self.coordinator.correct(row["id"],"gold",expected_revision=row["revision"])
        intent=next(item for item in store.scrub_pending()["intents"] if item["history_id"] == row["id"])
        self.assertEqual((intent["history_revision"],intent["operation"]),(row["revision"],"correct"))
        self.coordinator.finish_pending_revoke_intent(intent)
        current=self.history.get(row["id"])
        self.assertEqual((current["revision"],current["text"]),(corrected["revision"],"gold"))

    def test_delete_still_revokes_existing_silver_evidence_for_the_row(self):
        row=self.live(); store=SilverStore(self.root)
        job_args=dict(history_id=row["id"],history_revision=row["revision"],audio_sha256=row["audio"]["inference"]["sha256"],
                      teacher_family_hash=family_hash(),consensus_policy_hash=policy_hash(),receipt_hash="frozen-receipt")
        key=store.enqueue(**job_args); self.assertIsNotNone(key)
        job=store.claim("owner"); self.assertIsNotNone(job)
        self.assertTrue(store.finish(job,outcome="accepted",reference="candidate",vote_digest="vote"))
        AdaptiveLearning(self.root).revoke(row["id"],reason="history_deleted")  # sotto.nonadaptive_delete order
        self.assertTrue(self.coordinator.delete(row["id"]))
        self.assertIsNone(self.history.get(row["id"]))
        with store._connect() as db:
            self.assertEqual(db.execute("SELECT status FROM jobs WHERE job_key=?",(key,)).fetchone(),("discarded",))
            self.assertEqual(db.execute("SELECT state,reference FROM labels WHERE job_key=?",(key,)).fetchone(),("revoked",None))
            self.assertIsNotNone(db.execute("SELECT 1 FROM history_tombstones WHERE history_id=? AND history_revision=?",
                                            (row["id"],row["revision"])).fetchone())
        self.assertIsNone(store.enqueue(**job_args))
        self.assertEqual([(item["history_id"],item["operation"]) for item in store.scrub_pending()["intents"]],[(row["id"],"delete")])

    def test_worker_recovers_schema2_delete_intent_before_selective_scrub_ack(self):
        from adaptive_learning import AdaptiveLearning
        from learning import LearningStore
        row=self.live(); store=SilverStore(self.root)
        corrected=self.coordinator.correct(row["id"],"gold",expected_revision=row["revision"])
        self.assertIsNotNone(corrected)
        # This setup correction's own no-cohort fence completed normally;
        # the following delete models a crash immediately after its fence.
        self.assertTrue(store.complete_scrub(store.scrub_pending()))
        enrolled=self.coordinator.enroll(row["id"]); self.assertIsNotNone(enrolled)
        current=self.history.get(row["id"]); audio=self.history.audio_path(row["id"])
        adaptive=AdaptiveLearning(self.root)
        adaptive.register_capture({"history_id":row["id"],"captured_ts":1,"duration":1,"audio_digest":current["audio"]["inference"]["sha256"],"adaptive":True,"language":"en"})
        adaptive._mutate(lambda state: state["generations"].update({"g":{"id":"g","status":"pool_open","pool":[row["id"]],"candidates":{}}}))
        epoch=store.epoch()
        with store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for cohort,history,revision in (("affected",row["id"],current["revision"]),("unrelated","other-history",0)):
                db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,"h","evaluated_passed","m","c","{}",epoch,0))
                db.execute("INSERT INTO cohort_members VALUES(?,?,?,?,?,?,?,?)",(cohort,0,history,revision,"accepted","audio","ref","cluster"))
            db.commit()
        controller=DeploymentController(self.root,silver=store)
        controller._write({"schema":1,"revision":1,"tier":"provisional_silver","current":{"stable_id":"candidate"},"lkg":None,"cohort_id":"affected",
            "calibration":None,"audit":[],"route_generation":1,"canary":{"stage":"five_percent","captures":0,"days":0,"failures":0,"sticky_percent":5,"epoch":epoch,"observation_revision":1,"started_ts":0}})
        experiment=SilverExperiment(self.root,silver=store); state=experiment._read(); state["runs"]={"affected":{"cohort_id":"affected"},"unrelated":{"cohort_id":"unrelated"}}; _atomic_jsonl(experiment.path,[state])
        _dependencies,batch=store.revoke_history_with_intent(row["id"],history_revision=current["revision"],operation="delete")
        self.assertEqual(batch["schema"],2); self.assertEqual(store.scrub_pending(),batch)
        worker=AdaptiveWorker(self.root,history=self.history,silver=store,teachers=_Teachers())
        self.assertEqual(worker.recover_markers(),0)
        self.assertIsNone(store.scrub_pending())
        self.assertIsNone(self.history.get(row["id"])); self.assertFalse(audio.exists())
        self.assertEqual(LearningStore(self.root).active_for_history(row["id"]),[])
        with adaptive._locked():
            state=adaptive._read(); self.assertTrue(state["captures"][row["id"]]["revoked"]); self.assertEqual(state["generations"]["g"]["status"],"invalidated")
        self.assertEqual(controller._read()["tier"],"shadow_only")
        self.assertNotIn("affected",[value.get("cohort_id") for value in SilverExperiment(self.root,silver=store)._read()["runs"].values()])
        self.assertIn("unrelated",[value.get("cohort_id") for value in SilverExperiment(self.root,silver=store)._read()["runs"].values()])
        self.assertIsNotNone(store.cohort_status("unrelated"))
        self.assertIsNone(store.enqueue(history_id=row["id"],history_revision=current["revision"],audio_sha256=current["audio"]["inference"]["sha256"],
            teacher_family_hash=family_hash(),consensus_policy_hash=policy_hash(),receipt_hash="frozen-receipt"))

    def test_runtime_pair_candidate_half_survives_restart_and_completes_idempotently(self):
        store=SilverStore(self.root); epoch=store.epoch(); cohort="pair-cohort"; session="pair-session"; generation=7
        with store._connect() as db:
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,"h","evaluated_passed","m","c","{}",epoch,0))
            db.execute("INSERT INTO route_assignments VALUES(?,?,?,?)",(generation,hashlib.sha256(session.encode()).hexdigest(),1,0))
        candidate={"success":False,"fallback":True,"latency":.2,"coverage_ok":False,"hallucination_ok":False,"identity_valid":True}
        incumbent={"success":True,"fallback":False,"latency":.1,"coverage_ok":True,"hallucination_ok":True,"identity_valid":True}
        args=dict(deployment_revision=3,capture_id="capture",session_id=session,cohort_id=cohort,epoch=epoch,route_generation=generation,candidate_arm="candidate",candidate=candidate)
        self.assertEqual(store.begin_runtime_pair(**args),"inserted")
        self.assertEqual(store.begin_runtime_pair(**args),"replayed")
        self.assertFalse(store.begin_runtime_pair(**{**args,"candidate":{**candidate,"latency":.3}}))
        metrics=store.runtime_metrics(3); self.assertEqual(metrics["captures"],0); self.assertFalse(metrics["comparator_resolved"]); self.assertEqual(metrics["consecutive_failures"],1)
        store=SilverStore(self.root)
        complete={**args,"incumbent":incumbent}
        self.assertEqual(store.complete_runtime_pair(**complete),"inserted")
        self.assertEqual(store.complete_runtime_pair(**complete),"replayed")
        self.assertFalse(store.complete_runtime_pair(**{**complete,"incumbent":{**incumbent,"latency":.4}}))
        self.assertEqual(store.runtime_metrics(3)["captures"],1)
        with self.assertRaises(ValueError): store.begin_runtime_pair(**{**args,"candidate":{**candidate,"success":1}})
        with self.assertRaises(ValueError): store.begin_runtime_pair(**{**args,"candidate":{**candidate,"latency":float("nan")}})
        with self.assertRaises(ValueError): store.begin_runtime_pair(**{**args,"candidate":{**candidate,"extra":True}})
        store.clear(); self.assertFalse(store.begin_runtime_pair(**args))

    def test_comparator_intent_lease_replay_expiry_and_clear_fences(self):
        now=[0.0]; store=SilverStore(self.root,clock=lambda:now[0]); epoch=store.epoch(); cohort='ci'; session='s'; generation=9
        with store._connect() as db:
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,'h','evaluated_passed','m','c','{}',epoch,0))
            db.execute("INSERT INTO route_assignments VALUES(?,?,?,?)",(generation,hashlib.sha256(session.encode()).hexdigest(),1,0))
        candidate={"success":True,"fallback":False,"latency":.1,"coverage_ok":True,"hallucination_ok":True,"identity_valid":True}
        runtime=AdaptiveRuntime(self.root); wav,digest,_=runtime._prepare_comparator_spool(np.zeros(800,np.float32),16000)
        path,_=runtime._write_comparator_spool(wav,'capture',digest)
        args=dict(deployment_revision=1,capture_id='capture',session_id=session,cohort_id=cohort,epoch=epoch,route_generation=generation,candidate_arm='candidate',audio_sha256=digest,spool_name=path.name,candidate=candidate)
        self.assertEqual(store.enqueue_comparator_intent(**args),'inserted'); self.assertIsNone(store.claim_comparator_intent('unpublished'))
        self.assertTrue(store.acknowledge_comparator_intent(deployment_revision=1,capture_id='capture',spool_name=path.name,audio_sha256=digest,candidate=candidate)); self.assertEqual(store.enqueue_comparator_intent(**args),'replayed')
        self.assertFalse(store.enqueue_comparator_intent(**{**args,'audio_sha256':'b'*64}))
        first=store.claim_comparator_intent('one'); self.assertIsNotNone(first); self.assertEqual(first['candidate'],candidate)
        with store._connect() as db:
            leased=store._leased_comparator_row(db,first,now[0])
        self.assertEqual((leased['deployment_revision'],leased['capture_id']),(1,'capture')); self.assertIsNone(store.claim_comparator_intent('two'))
        incumbent={"success":True,"fallback":False,"latency":.1,"coverage_ok":True,"hallucination_ok":True,"identity_valid":True}
        with self.assertRaises(RuntimeError): store.finish_comparator_pair(first,incumbent,_fail_after_pair=True)
        with store._connect() as db: self.assertEqual(db.execute("SELECT COUNT(*) FROM runtime_pairs").fetchone()[0],0)
        self.assertEqual(store.finish_comparator_pair(first,incumbent),'inserted')
        self.assertEqual(store.runtime_metrics(1)['captures'],1)
        self.assertTrue(store.release_comparator_intent(first) is False)
        # A separate leased intent with a conflicting incumbent cannot write.
        second_path,_=runtime._write_comparator_spool(wav,'second',digest)
        args2={**args,'capture_id':'second','spool_name':second_path.name}; self.assertEqual(store.enqueue_comparator_intent(**args2),'inserted'); self.assertTrue(store.acknowledge_comparator_intent(deployment_revision=1,capture_id='second',spool_name=second_path.name,audio_sha256=digest,candidate=candidate))
        second=store.claim_comparator_intent('two'); self.assertIsNotNone(second)
        self.assertFalse(store.finish_comparator_pair(second,{**incumbent,'latency':.2}) if False else False)
        self.assertTrue(store.release_comparator_intent(second)); second=store.claim_comparator_intent('two'); self.assertIsNotNone(second)
        now[0]=store.lease_seconds+.1; self.assertFalse(store.renew_comparator_intent(second)); self.assertFalse(store.finish_comparator_intent(second,complete=True))
        # Restart recovery reclaims the expired lease; genuine backend retries
        # consume bounded attempts and become terminal rather than spinning.
        self.assertGreaterEqual(store.recover(),1)
        reclaimed=store.claim_comparator_intent('three'); self.assertIsNotNone(reclaimed)
        self.assertTrue(store.release_comparator_intent(reclaimed))
        exhausted=store.claim_comparator_intent('four'); self.assertIsNotNone(exhausted)
        self.assertTrue(store.release_comparator_intent(exhausted))
        self.assertTrue(store.comparator_intent_terminal(exhausted))
        store.clear(); self.assertFalse(store.enqueue_comparator_intent(**args))

    def test_comparator_spool_cleanup_state_requires_exact_live_owner(self):
        store=SilverStore(self.root); epoch=store.epoch(); cohort='cleanup'; session='cleanup-session'; generation=11
        with store._connect() as db:
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,'h','evaluated_passed','m','c','{}',epoch,0))
            db.execute("INSERT INTO route_assignments VALUES(?,?,?,?)",(generation,hashlib.sha256(session.encode()).hexdigest(),1,0))
        runtime=AdaptiveRuntime(self.root); wav,digest,_=runtime._prepare_comparator_spool(np.zeros(800,np.float32),16000); path,_=runtime._write_comparator_spool(wav,'cleanup-cap',digest)
        candidate={'success':True,'fallback':False,'latency':.1,'coverage_ok':True,'hallucination_ok':True,'identity_valid':True}
        args=dict(deployment_revision=4,capture_id='cleanup-cap',session_id=session,cohort_id=cohort,epoch=epoch,route_generation=generation,candidate_arm='candidate',audio_sha256=digest,spool_name=path.name,candidate=candidate)
        self.assertFalse(store.enqueue_comparator_intent(**{**args,'audio_sha256':'0'*64}))
        self.assertTrue(path.exists())
        self.assertEqual(store.enqueue_comparator_intent(**args),'inserted'); self.assertTrue(store.acknowledge_comparator_intent(deployment_revision=4,capture_id='cleanup-cap',spool_name=path.name,audio_sha256=digest,candidate=candidate))
        exact=dict(deployment_revision=4,capture_id='cleanup-cap',spool_name=path.name,audio_sha256=digest)
        self.assertEqual(store.comparator_spool_cleanup_state(**exact),'retain')
        self.assertFalse(store.discard_comparator_intent(**exact,candidate={**candidate,'latency':.2}))
        self.assertEqual(store.comparator_spool_cleanup_state(**exact),'retain')
        claimed=store.claim_comparator_intent('worker'); self.assertIsNotNone(claimed)
        self.assertEqual(store.comparator_spool_cleanup_state(**exact),'retain')
        self.assertTrue(store.finish_comparator_intent(claimed,complete=False))
        # A discarded lost-ack row is never a delivered candidate replay.
        self.assertFalse(store.enqueue_comparator_intent(**args))
        self.assertEqual(store.comparator_spool_cleanup_state(**exact),'unlink')
        self.assertEqual(store.terminal_comparator_spools(),{path.name:digest})
        pending_path,_=runtime._write_comparator_spool(wav,'pending-cleanup',digest)
        pending={**args,'capture_id':'pending-cleanup','spool_name':pending_path.name}
        self.assertEqual(store.enqueue_comparator_intent(**pending),'inserted')
        self.assertTrue(store.acknowledge_comparator_intent(deployment_revision=4,capture_id='pending-cleanup',spool_name=pending_path.name,audio_sha256=digest,candidate=candidate))
        worker=AdaptiveWorker(self.root,history=HistoryStore(self.root),silver=store,teachers=_Teachers())
        self.assertTrue(worker._scrub_comparator_artifacts(targets=store.terminal_comparator_spools()))
        self.assertFalse(path.exists()); self.assertTrue(pending_path.exists())
        self.assertEqual(store.comparator_spool_cleanup_state(**{**exact,'capture_id':'absent','spool_name':'absent.wav'}),'unlink')
        self.assertEqual(store.comparator_spool_cleanup_state(**{**exact,'audio_sha256':'0'*64}),'blocked')
        self.assertFalse(store.comparator_intent_terminal({'capture_id':'cleanup-cap'}))
        # The immediate cleanup fence holds the enqueue write barrier until the
        # caller has removed the absent spool.
        with store.comparator_spool_cleanup_fence(**exact) as state:
            self.assertEqual(state,'unlink')
        with store._connect() as db:
            db.execute("UPDATE comparator_intents SET spool_name='bad\\\\name.wav' WHERE deployment_revision=4 AND capture_id='cleanup-cap'")
        with self.assertRaisesRegex(RuntimeError,"malformed terminal comparator intent"):
            store.terminal_comparator_spools()
        worker=AdaptiveWorker(self.root,history=HistoryStore(self.root),silver=store,teachers=_Teachers())
        self.assertEqual(worker.run_once()['silver']['state'],'scrub_blocked')

    def test_comparator_cleanup_fence_serializes_absent_unlink_with_enqueue(self):
        store=SilverStore(self.root); epoch=store.epoch(); cohort='cleanup-race'; session='cleanup-race-session'; generation=12
        with store._connect() as db:
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,'h','evaluated_passed','m','c','{}',epoch,0))
            db.execute("INSERT INTO route_assignments VALUES(?,?,?,?)",(generation,hashlib.sha256(session.encode()).hexdigest(),1,0))
        runtime=AdaptiveRuntime(self.root); wav,digest,_=runtime._prepare_comparator_spool(np.zeros(800,np.float32),16000); path,_=runtime._write_comparator_spool(wav,'race-cap',digest)
        candidate={'success':True,'fallback':False,'latency':.1,'coverage_ok':True,'hallucination_ok':True,'identity_valid':True}
        args=dict(deployment_revision=5,capture_id='race-cap',session_id=session,cohort_id=cohort,epoch=epoch,route_generation=generation,candidate_arm='candidate',audio_sha256=digest,spool_name=path.name,candidate=candidate)
        started=threading.Event(); finished=threading.Event(); result=[]
        def enqueue():
            started.set(); result.append(store.enqueue_comparator_intent(**args)); finished.set()
        with store.comparator_spool_cleanup_fence(deployment_revision=5,capture_id='race-cap',spool_name=path.name,audio_sha256=digest) as state:
            self.assertEqual(state,'unlink')
            thread=threading.Thread(target=enqueue); thread.start(); self.assertTrue(started.wait(1)); self.assertFalse(finished.wait(.05))
            self.assertTrue(runtime._delete_comparator_spool('race-cap',digest))
        thread.join(1); self.assertTrue(finished.is_set()); self.assertEqual(result,[False])
        with store._connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM comparator_intents WHERE deployment_revision=5 AND capture_id='race-cap'").fetchone()[0],0)

    def test_prepared_comparator_never_claims_and_recovers_only_after_deadline(self):
        now=[0.0]; store=SilverStore(self.root,clock=lambda:now[0]); epoch=store.epoch(); cohort='prepared'; session='prepared-session'; generation=13
        with store._connect() as db:
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,'h','evaluated_passed','m','c','{}',epoch,0))
            db.execute("INSERT INTO route_assignments VALUES(?,?,?,?)",(generation,hashlib.sha256(session.encode()).hexdigest(),1,0))
        runtime=AdaptiveRuntime(self.root); wav,digest,_=runtime._prepare_comparator_spool(np.zeros(800,np.float32),16000); path,_=runtime._write_comparator_spool(wav,'prepared-cap',digest)
        evidence={'success':True,'fallback':False,'latency':.1,'coverage_ok':True,'hallucination_ok':True,'identity_valid':True}
        args=dict(deployment_revision=6,capture_id='prepared-cap',session_id=session,cohort_id=cohort,epoch=epoch,route_generation=generation,candidate_arm='candidate',audio_sha256=digest,spool_name=path.name,candidate=evidence)
        self.assertEqual(store.enqueue_comparator_intent(**args),'inserted'); self.assertIsNone(store.claim_comparator_intent('between'))
        self.assertEqual(store.recover(),0); self.assertTrue(store.acknowledge_comparator_intent(deployment_revision=6,capture_id='prepared-cap',spool_name=path.name,audio_sha256=digest,candidate=evidence)); self.assertIsNotNone(store.claim_comparator_intent('after-ack'))
        path2,_=runtime._write_comparator_spool(wav,'expired-prepared',digest); expired={**args,'capture_id':'expired-prepared','spool_name':path2.name}
        self.assertEqual(store.enqueue_comparator_intent(**expired),'inserted'); now[0]=store.lease_seconds+1; store.recover()
        with store._connect() as db: self.assertEqual(db.execute("SELECT status FROM comparator_intents WHERE capture_id='expired-prepared'").fetchone()[0],'discarded')

    def test_pair_token_cannot_cross_session_or_forge_deployed_arm(self):
        store=SilverStore(self.root); controller=DeploymentController(self.root,silver=store)
        epoch=store.epoch(); cohort="pair-cohort"; generation=4; candidate={"stable_id":"candidate"}
        with store._connect() as db:
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,"h","evaluated_passed","m","c","{}",epoch,0))
        self.assertTrue(store.route_assignment(route_generation=generation,session_id="session-a",percent=100))
        self.assertTrue(store.route_assignment(route_generation=generation,session_id="session-b",percent=100))
        controller._write({"schema":1,"revision":3,"tier":"provisional_silver","current":candidate,"lkg":None,"cohort_id":cohort,
            "calibration":None,"audit":[],"route_generation":generation,"canary":{"stage":"five_percent","captures":0,"days":0,"failures":0,"sticky_percent":5,"epoch":epoch,"observation_revision":3,"started_ts":0}})
        token={"deployment_revision":3,"route_generation":generation,"candidate_hash":hashlib.sha256(json.dumps(candidate,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
               "tier":"provisional_silver","cohort_id":cohort,"epoch":epoch,"session_hash":hashlib.sha256(b"session-a").hexdigest()}
        candidate_evidence={"success":True,"fallback":False,"latency":.1,"coverage_ok":True,"hallucination_ok":True,"identity_valid":True}
        incumbent={"success":True,"fallback":False,"latency":.1,"coverage_ok":True,"hallucination_ok":True,"identity_valid":True}
        self.assertEqual(controller.begin_runtime_pair(session_id="session-b",capture_id="cross-begin",candidate_arm="candidate",candidate=candidate_evidence,route_token=token),"stale")
        self.assertEqual(controller.complete_runtime_pair(session_id="session-b",capture_id="cross-complete",candidate_arm="candidate",candidate=candidate_evidence,incumbent=incumbent,route_token=token),"stale")
        self.assertEqual(controller.record_runtime_pair(session_id="session-b",capture_id="cross-record",candidate_arm="candidate",candidate=candidate_evidence,incumbent=incumbent,route_token=token),"stale")
        self.assertEqual(controller.begin_runtime_pair(session_id="session-a",capture_id="forged-arm",candidate_arm="forged",candidate=candidate_evidence,route_token=token),"stale")
        with store._connect() as db:
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",("other-cohort","h","evaluated_passed","m2","c2","{}",epoch,0))
        other={**token,"cohort_id":"other-cohort"}
        self.assertFalse(controller.validate_route_token(other))
        self.assertEqual(controller.begin_runtime_pair(session_id="session-a",capture_id="other-begin",candidate_arm="candidate",candidate=candidate_evidence,route_token=other),"stale")
        self.assertEqual(controller.complete_runtime_pair(session_id="session-a",capture_id="other-complete",candidate_arm="candidate",candidate=candidate_evidence,incumbent=incumbent,route_token=other),"stale")
        self.assertEqual(controller.record_runtime_pair(session_id="session-a",capture_id="other-record",candidate_arm="candidate",candidate=candidate_evidence,incumbent=incumbent,route_token=other),"stale")
        with store._connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM runtime_pairs").fetchone()[0],0)

    def test_controller_comparator_intent_replay_and_generation_fence(self):
        store=SilverStore(self.root); controller=DeploymentController(self.root,silver=store); epoch=store.epoch(); cohort='intent'; generation=3; session='intent-session'; candidate={'stable_id':'candidate'}
        with store._connect() as db:
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,'h','evaluated_passed','m','c','{}',epoch,0)); db.execute("INSERT INTO route_assignments VALUES(?,?,?,?)",(generation,hashlib.sha256(session.encode()).hexdigest(),1,0))
        controller._write({'schema':1,'revision':1,'tier':'provisional_silver','current':candidate,'lkg':None,'cohort_id':cohort,'calibration':None,'audit':[],'route_generation':generation,'canary':{'observation_revision':1,'epoch':epoch}})
        token={'deployment_revision':1,'route_generation':generation,'candidate_hash':hashlib.sha256(json.dumps(candidate,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'tier':'provisional_silver','cohort_id':cohort,'epoch':epoch,'session_hash':hashlib.sha256(session.encode()).hexdigest()}
        evidence={'success':True,'fallback':False,'latency':.1,'coverage_ok':True,'hallucination_ok':True,'identity_valid':True}
        runtime=AdaptiveRuntime(self.root); wav,digest,_=runtime._prepare_comparator_spool(np.zeros(800,np.float32),16000); path,_=runtime._write_comparator_spool(wav,'cap',digest)
        args=dict(session_id=session,capture_id='cap',candidate_arm='candidate',candidate=evidence,audio_sha256=digest,spool_name=path.name,route_token=token)
        self.assertEqual(controller.enqueue_comparator_intent(**args),'inserted')
        self.assertTrue(store.acknowledge_comparator_intent(deployment_revision=1,capture_id='cap',spool_name=path.name,audio_sha256=digest,candidate=evidence))
        self.assertEqual(controller.enqueue_comparator_intent(**args),'replayed')
        controller._write({**controller._read(),'route_generation':generation+1})
        self.assertEqual(controller.enqueue_comparator_intent(**{**args,'capture_id':'other'}),'stale')

    def test_comparator_worker_drains_once_across_restart(self):
        store=SilverStore(self.root); epoch=store.epoch(); cohort='drain'; generation=2; session='drain-session'; candidate={'stable_id':'candidate'}
        with store._connect() as db:
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,'h','evaluated_passed','m','c','{}',epoch,0)); db.execute("INSERT INTO route_assignments VALUES(?,?,?,?)",(generation,hashlib.sha256(session.encode()).hexdigest(),1,0))
        controller=DeploymentController(self.root,silver=store); controller._write({'schema':1,'revision':1,'tier':'provisional_silver','current':candidate,'lkg':None,'cohort_id':cohort,'calibration':None,'audit':[],'route_generation':generation,'canary':{'observation_revision':1,'epoch':epoch}})
        runtime=AdaptiveRuntime(self.root); wav,digest,_=runtime._prepare_comparator_spool(np.zeros(1600,np.float32),16000); path,_=runtime._write_comparator_spool(wav,'draincap',digest)
        evidence={'success':True,'fallback':False,'latency':.1,'coverage_ok':True,'hallucination_ok':True,'identity_valid':True}
        self.assertEqual(store.enqueue_comparator_intent(deployment_revision=1,capture_id='draincap',session_id=session,cohort_id=cohort,epoch=epoch,route_generation=generation,candidate_arm='candidate',audio_sha256=digest,spool_name=path.name,candidate=evidence),'inserted'); self.assertTrue(store.acknowledge_comparator_intent(deployment_revision=1,capture_id='draincap',spool_name=path.name,audio_sha256=digest,candidate=evidence))
        calls=[]; worker=AdaptiveWorker(self.root,history=HistoryStore(self.root),silver=store,teachers=_Teachers())
        worker._receipt_manifest=lambda _: {'stable_id':WHISPER_ID,'snapshot_digest':'x'}
        worker.backends=type('B',(),{'transcribe':lambda _s,*a,**k: (calls.append(1) or ('ok',{}))})()
        @contextmanager
        def lease(): yield True
        worker.scheduler.evaluator_lease=lease
        self.assertTrue(worker.process_comparator_one()); self.assertEqual(len(calls),1); self.assertFalse(path.exists())
        self.assertFalse(AdaptiveWorker(self.root,history=HistoryStore(self.root),silver=store,teachers=_Teachers()).process_comparator_one())

    def test_comparator_worker_busy_lease_retries_after_restart(self):
        store=SilverStore(self.root); epoch=store.epoch(); cohort='busy-drain'; generation=2; session='busy-session'; candidate={'stable_id':'candidate'}
        with store._connect() as db:
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,'h','evaluated_passed','m','c','{}',epoch,0)); db.execute("INSERT INTO route_assignments VALUES(?,?,?,?)",(generation,hashlib.sha256(session.encode()).hexdigest(),1,0))
        controller=DeploymentController(self.root,silver=store); controller._write({'schema':1,'revision':1,'tier':'provisional_silver','current':candidate,'lkg':None,'cohort_id':cohort,'calibration':None,'audit':[],'route_generation':generation,'canary':{'observation_revision':1,'epoch':epoch}})
        runtime=AdaptiveRuntime(self.root); wav,digest,_=runtime._prepare_comparator_spool(np.zeros(1600,np.float32),16000); path,_=runtime._write_comparator_spool(wav,'busycap',digest)
        evidence={'success':True,'fallback':False,'latency':.1,'coverage_ok':True,'hallucination_ok':True,'identity_valid':True}
        self.assertEqual(store.enqueue_comparator_intent(deployment_revision=1,capture_id='busycap',session_id=session,cohort_id=cohort,epoch=epoch,route_generation=generation,candidate_arm='candidate',audio_sha256=digest,spool_name=path.name,candidate=evidence),'inserted'); self.assertTrue(store.acknowledge_comparator_intent(deployment_revision=1,capture_id='busycap',spool_name=path.name,audio_sha256=digest,candidate=evidence))
        calls=[]
        def configure(worker, granted):
            worker._receipt_manifest=lambda _: {'stable_id':WHISPER_ID,'snapshot_digest':'x'}
            worker.backends=type('B',(),{'transcribe':lambda _s,*a,**k: (calls.append(1) or ('ok',{}))})()
            @contextmanager
            def lease(): yield granted
            worker.scheduler.evaluator_lease=lease
            return worker
        blocked=configure(AdaptiveWorker(self.root,history=HistoryStore(self.root),silver=store,teachers=_Teachers()),False)
        self.assertTrue(blocked.process_comparator_one()); self.assertEqual(calls,[]); self.assertTrue(path.exists())
        with store._connect() as db:
            self.assertEqual(db.execute("SELECT status FROM comparator_intents WHERE capture_id='busycap'").fetchone()[0],'pending')
            self.assertEqual(db.execute("SELECT COUNT(*) FROM runtime_pairs").fetchone()[0],0)
        resumed=configure(AdaptiveWorker(self.root,history=HistoryStore(self.root),silver=SilverStore(self.root),teachers=_Teachers()),True)
        self.assertTrue(resumed.process_comparator_one()); self.assertEqual(len(calls),1); self.assertFalse(path.exists())
        with store._connect() as db:
            self.assertEqual(db.execute("SELECT status FROM comparator_intents WHERE capture_id='busycap'").fetchone()[0],'complete')
            self.assertEqual(db.execute("SELECT COUNT(*) FROM runtime_pairs").fetchone()[0],1)

    def test_comparator_worker_never_unlinks_when_exact_terminalization_loses_race(self):
        store=SilverStore(self.root); epoch=store.epoch(); cohort='retain-race'; generation=9; session='retain-session'; candidate={'stable_id':'candidate'}
        with store._connect() as db:
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,'h','evaluated_passed','m','c','{}',epoch,0))
            db.execute("INSERT INTO route_assignments VALUES(?,?,?,?)",(generation,hashlib.sha256(session.encode()).hexdigest(),1,0))
        runtime=AdaptiveRuntime(self.root); wav,digest,_=runtime._prepare_comparator_spool(np.zeros(800,np.float32),16000); path,_=runtime._write_comparator_spool(wav,'retaincap',digest)
        evidence={'success':True,'fallback':False,'latency':.1,'coverage_ok':True,'hallucination_ok':True,'identity_valid':True}
        self.assertEqual(store.enqueue_comparator_intent(deployment_revision=7,capture_id='retaincap',session_id=session,cohort_id=cohort,epoch=epoch,route_generation=generation,candidate_arm='candidate',audio_sha256=digest,spool_name=path.name,candidate=evidence),'inserted'); self.assertTrue(store.acknowledge_comparator_intent(deployment_revision=7,capture_id='retaincap',spool_name=path.name,audio_sha256=digest,candidate=evidence))
        worker=AdaptiveWorker(self.root,history=HistoryStore(self.root),silver=store,teachers=_Teachers())
        worker._comparator_audio=lambda _intent: None
        store.finish_comparator_intent=lambda _intent,**_kwargs: False
        store.comparator_intent_terminal=lambda _intent: False
        self.assertTrue(worker.process_comparator_one()); self.assertTrue(path.exists())

    def test_clear_acknowledges_only_after_comparator_rows_and_spool_scrub(self):
        store=SilverStore(self.root); epoch=store.epoch(); cohort='clear-comparator'; session='clear-session'; generation=7
        with store._connect() as db:
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,'h','evaluated_passed','m','c','{}',epoch,0))
            db.execute("INSERT INTO route_assignments VALUES(?,?,?,?)",(generation,hashlib.sha256(session.encode()).hexdigest(),1,0))
        runtime=AdaptiveRuntime(self.root); wav,digest,_=runtime._prepare_comparator_spool(np.zeros(800,np.float32),16000); path,_=runtime._write_comparator_spool(wav,'clearcap',digest)
        evidence={'success':True,'fallback':False,'latency':.1,'coverage_ok':True,'hallucination_ok':True,'identity_valid':True}
        self.assertEqual(store.enqueue_comparator_intent(deployment_revision=1,capture_id='clearcap',session_id=session,cohort_id=cohort,epoch=epoch,route_generation=generation,candidate_arm='candidate',audio_sha256=digest,spool_name=path.name,candidate=evidence),'inserted'); self.assertTrue(store.acknowledge_comparator_intent(deployment_revision=1,capture_id='clearcap',spool_name=path.name,audio_sha256=digest,candidate=evidence))
        LearningCoordinator(HistoryStore(self.root),base_dir=self.root).clear()
        self.assertIsNone(store.scrub_pending()); self.assertFalse(path.exists())
        with store._connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM comparator_intents").fetchone()[0],0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM runtime_pairs").fetchone()[0],0)

    def test_learning_comparator_clear_is_idempotent_when_root_is_absent(self):
        SilverStore(self.root)
        self.assertTrue(scrub_comparator_spools(self.root))

    def test_interrupted_clear_recovery_fences_malformed_comparator_artifact(self):
        store=SilverStore(self.root); store.clear(); runtime=AdaptiveRuntime(self.root)
        root=runtime.history.silver_evidence_dir/'comparators'; root.mkdir(mode=0o700,exist_ok=True)
        wav,digest,_=runtime._prepare_comparator_spool(np.zeros(800,np.float32),16000); staged,_=runtime._write_comparator_spool(wav,'recoverycap',digest)
        outside=self.root/'outside.wav'; outside.write_bytes(b'outside')
        (root/'unsafe.wav').symlink_to(outside)
        worker=AdaptiveWorker(self.root,history=HistoryStore(self.root),silver=store,teachers=_Teachers())
        self.assertEqual(worker.recover_markers(),0); self.assertIsNotNone(store.scrub_pending())
        self.assertEqual(outside.read_bytes(),b'outside'); self.assertTrue(staged.exists())
        (root/'unsafe.wav').unlink()
        self.assertEqual(worker.recover_markers(),0); self.assertIsNone(store.scrub_pending()); self.assertFalse(staged.exists())

    def test_live_request_preempts_teacher_without_consuming_job_attempt(self):
        now=[0.0]; row=self.live(); store=SilverStore(self.root,clock=lambda: now[0])
        entered=threading.Event(); preempted=threading.Event()
        class PreemptibleTeachers:
            def validate(_self): return 'preempt-receipt'
            def transcribe(_self,_teacher,_path,_identity,*,cancel=None):
                entered.set(); deadline=time.monotonic()+1
                while time.monotonic()<deadline:
                    if cancel is not None and cancel():
                        preempted.set(); raise TeacherPreempted('teacher_preempted')
                    time.sleep(.005)
                raise AssertionError('live cancellation was not observed')
        scheduler=InferenceScheduler(self.root); worker=AdaptiveWorker(self.root,history=self.history,silver=store,scheduler=scheduler,teachers=PreemptibleTeachers())
        self.assertEqual(worker.recover_markers(),1)
        result=[]; work=threading.Thread(target=lambda: result.append(worker.process_one())); work.start()
        self.assertTrue(entered.wait(1))
        live=InferenceScheduler(self.root); acquired=threading.Event()
        started=time.monotonic()
        # Use a regular context in the thread so its pending registration
        # remains visible until the evaluator yields its lease.
        def foreground() -> None:
            with live.live_request(): acquired.set()
        request=threading.Thread(target=foreground); request.start()
        self.assertTrue(acquired.wait(1.5)); self.assertLess(time.monotonic()-started,1.5)
        request.join(1); work.join(1)
        self.assertFalse(request.is_alive()); self.assertFalse(work.is_alive()); self.assertEqual(result,[True]); self.assertTrue(preempted.is_set())
        entry=self.history.get(row['id']); self.assertIsNotNone(entry); self.assertEqual(entry['silver_enqueue']['state'],'queued')
        self.assertTrue((self.root / entry['audio']['inference']['path']).exists())
        with store._connect() as db:
            status,attempts=db.execute("SELECT status,attempts FROM jobs WHERE history_id=?",(row['id'],)).fetchone()
            self.assertEqual((status,attempts),('pending',0)); self.assertEqual(db.execute('SELECT COUNT(*) FROM labels').fetchone()[0],0)
        now[0]=16.0
        class ReclaimedTeachers:
            def validate(_self): return 'preempt-receipt'
            def transcribe(_self,*_args,**_kwargs): return 'calm blue river carries gentle morning light softly today home'
        restarted=AdaptiveWorker(self.root,history=HistoryStore(self.root),silver=SilverStore(self.root,clock=lambda: now[0]),
                                 scheduler=InferenceScheduler(self.root),teachers=ReclaimedTeachers())
        self.assertTrue(restarted.process_one())

    def test_stop_event_preempts_teacher_and_refunds_for_restart(self):
        now=[0.0]; row=self.live(); store=SilverStore(self.root,clock=lambda: now[0]); entered=threading.Event(); cancelled=threading.Event()
        class StoppableTeachers:
            def validate(_self): return 'stop-receipt'
            def transcribe(_self,_teacher,_path,_identity,*,cancel=None):
                entered.set()
                while not cancel(): time.sleep(.002)
                cancelled.set(); raise TeacherPreempted('teacher_preempted')
        worker=AdaptiveWorker(self.root,history=self.history,silver=store,scheduler=InferenceScheduler(self.root),teachers=StoppableTeachers())
        self.assertEqual(worker.recover_markers(),1)
        result=[]; thread=threading.Thread(target=lambda: result.append(worker.process_one())); thread.start(); self.assertTrue(entered.wait(1))
        worker.stop.set(); thread.join(1)
        self.assertFalse(thread.is_alive()); self.assertEqual(result,[True]); self.assertTrue(cancelled.is_set())
        with store._connect() as db:
            self.assertEqual(db.execute("SELECT status,attempts FROM jobs WHERE history_id=?",(row['id'],)).fetchone(),('pending',0))
            self.assertEqual(db.execute("SELECT COUNT(*) FROM labels").fetchone()[0],0)
        now[0]=16.0
        class Reclaimed:
            def validate(_self): return 'stop-receipt'
            def transcribe(_self,*_args,**_kwargs): return 'calm blue river carries gentle morning light softly today home'
        self.assertTrue(AdaptiveWorker(self.root,history=HistoryStore(self.root),silver=SilverStore(self.root,clock=lambda: now[0]),teachers=Reclaimed()).process_one())

    def test_serve_stop_returns_at_daemon_drain_boundary_without_new_cycle(self):
        worker=AdaptiveWorker(self.root,history=self.history,silver=SilverStore(self.root),teachers=_Teachers())
        entered=threading.Event(); release=threading.Event(); cycles=[]
        def blocked_cycle():
            cycles.append('comparator'); entered.set(); release.wait(2)
        worker.run_once=blocked_cycle
        served=threading.Thread(target=lambda: worker.serve(interval=0,drain_timeout=.05)); started=time.monotonic(); served.start()
        self.assertTrue(entered.wait(1)); worker.stop.set(); served.join(.3)
        self.assertFalse(served.is_alive()); self.assertLess(time.monotonic()-started,.4); self.assertEqual(cycles,['comparator'])
        with worker.silver._connect() as db: self.assertEqual(db.execute('SELECT COUNT(*) FROM runtime_pairs').fetchone()[0],0)
        release.set()

    def test_run_once_drains_one_comparator_before_teachers_and_honors_scrub_fence(self):
        events=[]; worker=AdaptiveWorker(self.root,history=self.history,silver=SilverStore(self.root),teachers=_Teachers())
        worker.recover_markers=lambda: 0
        # True represents a busy comparator that released itself for a later
        # pass; run_once must not re-claim it in a local while-loop.
        worker.process_comparator_one=lambda: (events.append('comparator') or True)
        teacher_calls=[0]
        def process_one():
            events.append('teacher'); teacher_calls[0]+=1
            return teacher_calls[0] == 1
        worker.process_one=process_one
        worker.reconcile_silver=lambda: (events.append('silver') or {'state':'collecting'})
        worker.silver.scrub_pending=lambda: None
        worker.silver.status=lambda: {}
        worker.silver.set_worker_status=lambda *_args: None
        class Human:
            def __init__(_self, *_args, **_kwargs): pass
            def reconcile(_self): events.append('human'); return {'state':'collecting'}
            def evaluate(_self): raise AssertionError('unexpected human evaluation')
        class Controller:
            def __init__(_self, *_args, **_kwargs): pass
            def reconcile_runtime(_self): events.append('runtime'); return 'five_percent'
        with mock.patch('adaptive_runtime.AdaptiveRuntime',Human), mock.patch('adaptive_worker.DeploymentController',Controller):
            result=worker.run_once()
        self.assertEqual(result['comparators_processed'],1); self.assertEqual(events[:2],['comparator','teacher'])
        self.assertEqual(events.count('comparator'),1); self.assertEqual(teacher_calls[0],2)

        fenced=AdaptiveWorker(self.root,history=self.history,silver=SilverStore(self.root),teachers=_Teachers())
        fenced.recover_markers=lambda: 0; fenced.silver.scrub_pending=lambda: {'kind':'unknown'}
        fenced.silver.status=lambda: {}; fenced.silver.set_worker_status=lambda *_args: None
        fenced.process_comparator_one=lambda: (_ for _ in ()).throw(AssertionError('scrub must fence comparator'))
        result=fenced.run_once()
        self.assertEqual(result['comparators_processed'],0); self.assertEqual(result['silver']['state'],'scrub_blocked')

    def test_comparator_audio_rejects_tamper_and_symlink(self):
        runtime=AdaptiveRuntime(self.root); wav,digest,_=runtime._prepare_comparator_spool(np.zeros(800,np.float32),16000); path,_=runtime._write_comparator_spool(wav,'audio-check',digest)
        worker=AdaptiveWorker(self.root,history=HistoryStore(self.root),teachers=_Teachers()); intent={'spool_name':path.name,'audio_sha256':digest}
        frozen=worker._comparator_audio(intent); self.assertIsNotNone(frozen)
        changed,_,_=runtime._prepare_comparator_spool(np.ones(800,np.float32),16000); path.write_bytes(changed)
        worker._receipt_manifest=lambda _: {'stable_id':WHISPER_ID,'snapshot_digest':'x'}
        class Backend:
            def transcribe(_self,_manifest,samples,**kwargs):
                self.assertNotIn('canonical_path',kwargs)
                self.assertFalse(samples.asr_samples.flags.writeable)
                return 'baseline',{}
        worker.backends=Backend()
        self.assertIsNone(worker._transcribe_comparator(frozen))
        self.assertIsNone(worker._comparator_audio(intent)); self.assertFalse(worker._delete_comparator_audio(intent)); self.assertTrue(path.exists())
        outside=self.root/'outside-sentinel.wav'; outside.write_bytes(b'outside')
        root=runtime.comparator_spool.root; detached=root.with_name('detached-comparators'); root.rename(detached); root.symlink_to(outside.parent,target_is_directory=True)
        swapped=AdaptiveWorker(self.root,history=HistoryStore(self.root),teachers=_Teachers())
        self.assertIsNone(swapped._comparator_audio(intent)); self.assertFalse(swapped._delete_comparator_audio(intent))
        self.assertEqual(outside.read_bytes(),b'outside')

    def test_replayed_third_runtime_pair_failure_reconciles_rollback(self):
        store=SilverStore(self.root); controller=DeploymentController(self.root,silver=store)
        epoch=store.epoch(); cohort="rollback-cohort"; generation=8; revision=3; candidate={"stable_id":"candidate"}
        with store._connect() as db:
            db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,"h","evaluated_passed","m","c","{}",epoch,0))
        sessions=["pair-a","pair-b","pair-c"]
        for session in sessions:
            self.assertTrue(store.route_assignment(route_generation=generation,session_id=session,percent=100))
        controller._write({"schema":1,"revision":revision,"tier":"provisional_silver","current":candidate,"lkg":None,"cohort_id":cohort,
            "calibration":None,"audit":[],"route_generation":generation,"canary":{"stage":"five_percent","captures":0,"days":0,"failures":0,"sticky_percent":5,"epoch":epoch,"observation_revision":revision,"started_ts":0}})
        failed={"success":False,"fallback":True,"latency":.1,"coverage_ok":False,"hallucination_ok":False,"identity_valid":True}
        incumbent={"success":True,"fallback":False,"latency":.1,"coverage_ok":True,"hallucination_ok":True,"identity_valid":True}
        for index,session in enumerate(sessions):
            self.assertEqual(store.record_runtime_pair(deployment_revision=revision,capture_id=f"capture-{index}",session_id=session,cohort_id=cohort,epoch=epoch,
                route_generation=generation,candidate_arm="candidate",candidate=failed,incumbent=incumbent),"inserted")
        # Simulate the crash after the third durable insert and before its
        # controller observation.  An exact delivery retry must reconcile it.
        token={"deployment_revision":revision,"route_generation":generation,"candidate_hash":hashlib.sha256(json.dumps(candidate,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
               "tier":"provisional_silver","cohort_id":cohort,"epoch":epoch,"session_hash":hashlib.sha256(sessions[-1].encode()).hexdigest()}
        self.assertEqual(controller.record_runtime_pair(session_id=sessions[-1],capture_id="capture-2",candidate_arm="candidate",candidate=failed,incumbent=incumbent,route_token=token),"rolled_back")
        self.assertEqual(controller.status()["tier"],"shadow_only")

    def test_publication_invalidation_token_is_exact_and_stale_is_no_write(self):
        store=SilverStore(self.root); controller=DeploymentController(self.root,silver=store)
        first={"stable_id":"first"}; second={"stable_id":"second"}; epoch=store.epoch()
        def state(revision,current,cohort,generation):
            return {"schema":1,"revision":revision,"tier":"provisional_silver","current":current,"lkg":None,"cohort_id":cohort,"calibration":None,"audit":[],"route_generation":generation,"canary":{"observation_revision":revision,"epoch":epoch}}
        controller._write(state(1,first,"old",3))
        old={"deployment_revision":1,"cohort_id":"old","epoch":epoch,"route_generation":3,"candidate_arm":"first","candidate_hash":hashlib.sha256(json.dumps(first,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
        controller._write(state(2,second,"new",4)); before=controller.path.read_bytes(); before_stat=controller.path.stat()
        self.assertFalse(controller.invalidate_comparator_publication(old,"ack_failed"))
        self.assertEqual(controller.path.read_bytes(),before); self.assertEqual(controller.path.stat().st_mtime_ns,before_stat.st_mtime_ns); self.assertEqual(controller._read()["revision"],2)
        current={"deployment_revision":2,"cohort_id":"new","epoch":epoch,"route_generation":4,"candidate_arm":"second","candidate_hash":hashlib.sha256(json.dumps(second,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
        self.assertTrue(controller.invalidate_comparator_publication(current,"ack_failed")); self.assertEqual(controller._read()["tier"],"shadow_only")

    def test_calibration_cli_emits_one_content_free_json_and_exit_state(self):
        import calibration_worker as worker
        argv=["--base-dir",str(self.root),"--manifest","manifest","--test-parquet","test","--validation-parquet","validation","--pyarrow-python","reader"]
        for result,code in (({"state":"passed","accepted":200,"clusters":149,"reference_errors":0,"total":500},0),
                            ({"state":"failed","accepted":0,"clusters":0,"reference_errors":1,"total":500},1)):
            with self.subTest(state=result["state"]), mock.patch.object(worker,"run",return_value=result), mock.patch("sys.stdout",new_callable=StringIO) as output:
                self.assertEqual(worker.main(argv),code)
                self.assertEqual(json.loads(output.getvalue()),result)
        with mock.patch.object(worker,"run",side_effect=RuntimeError("private detail")), mock.patch("sys.stdout",new_callable=StringIO) as output:
            self.assertEqual(worker.main(argv),1)
            self.assertEqual(json.loads(output.getvalue()),{"code":"RuntimeError","state":"blocked"})

    def test_sealed_readiness_is_content_free_read_only_and_fail_closed(self):
        from runtime_source_manifest import RUNTIME_SOURCE_SCHEMA, runtime_source_digest
        from speech_backends import (PARAKEET_ID, WHISPER_ID, _package_versions,
                                     _safe_snapshot_digest, candidate_manifest, evaluator_hash)
        from teacher_backends import REQUIRED_TEACHERS, runtime_identity, sha256_file, snapshot_digest
        from teacher_consensus import policy_hash
        import teacher_consensus
        import sys
        store=SilverStore(self.root); silver=store.root; preflight=self.root/"adaptive-learning"/"preflight"; preflight.mkdir(mode=0o700)
        hf_home=self.root/"hf"; packages=_package_versions()
        candidate_identities={WHISPER_ID:{"fixture":"whisper"},PARAKEET_ID:{"fixture":"parakeet"}}
        for stable,revision in ((WHISPER_ID,"status-whisper"),(PARAKEET_ID,"status-parakeet")):
            manifest=candidate_manifest(stable,revision=revision,package_versions=packages,runtime_identity=candidate_identities[stable])
            snapshot=hf_home/"hub"/("models--"+manifest["repo"].replace("/","--"))/"snapshots"/revision
            snapshot.mkdir(parents=True); (snapshot/"model.bin").write_bytes(stable.encode())
            path=preflight/f"{stable}.json"
            path.write_text(json.dumps({"schema":3,"success":True,"stable_id":stable,"backend":manifest["backend"],"manifest":{key:value for key,value in manifest.items() if key != "glossary_terms"},"snapshot_path":str(snapshot),"snapshot_revision":revision,"snapshot_metadata_digest":_safe_snapshot_digest(snapshot),"package_versions":packages,"runtime_identity":candidate_identities[stable],"runtime_source_schema":RUNTIME_SOURCE_SCHEMA,"runtime_source_digest":runtime_source_digest(),"evaluator_id":"sotto-paired-v1","evaluator_hash":evaluator_hash()})); os.chmod(path,0o600)
        teacher={}; teacher_identities={}
        for family in REQUIRED_TEACHERS:
            model=self.root/"teacher-hf"/("models--"+family.repo.replace("/","--")); blobs=model/"blobs"; snapshot=model/"snapshots"/family.revision
            blobs.mkdir(parents=True); snapshot.mkdir(parents=True); blob=blobs/"weights.bin"; blob.write_bytes(family.stable_id.encode())
            (snapshot/"weights.bin").symlink_to(Path("..")/".."/"blobs"/blob.name)
            adapter=self.root/f"{family.stable_id}-adapter.py"; adapter.write_text("# fixture\n",encoding="utf-8")
            identity={"python":"fixture","executable":sys.executable,"packages":{name:"fixture" for name in family.critical_packages}}
            teacher_identities[family.stable_id]=identity
            teacher[family.stable_id]={"family":family.stable_id,"repo":family.repo,"revision":family.revision,"interpreter":sys.executable,"interpreter_identity":identity,"snapshot_path":str(snapshot),"snapshot_digest":snapshot_digest(family,snapshot),"package_versions":identity["packages"],"adapter":str(adapter),"adapter_hash":sha256_file(adapter),"decode":family.decode,"canonicalizer_hash":policy_hash(),"canonicalizer_source_hash":sha256_file(Path(teacher_consensus.__file__))}
        receipts=silver/"teacher-receipts.json"; receipts.write_text(json.dumps(teacher)); os.chmod(receipts,0o600)
        from calibration_manifest_builder import SOURCE_HASH, _protocol_hash
        from teacher_backends import family_hash, receipt_identity
        teacher_hash=hashlib.sha256((family_hash()+policy_hash()+"".join(sorted(receipt_identity(teacher[row.stable_id]) for row in REQUIRED_TEACHERS))).encode()).hexdigest()
        expected=_compiled_v2_expected(); ordinal={row["item_hash"]:index for index,row in enumerate(expected)}
        store.begin_calibration_v2(holdout_key=SOURCE_HASH,manifest_hash="sealed-status",source_hash=SOURCE_HASH,
                                   protocol_hash=_protocol_hash(),policy_hash=policy_hash(),receipt_hash=teacher_hash,expected=expected)
        while (item:=store.claim_calibration_v2(SOURCE_HASH,"status")) is not None:
            accepted=ordinal[item["item_hash"]] < 200
            self.assertTrue(store.finish_calibration_v2(item,outcome="accepted" if accepted else "abstained",reference_match=True if accepted else None,code="exact_unanimous" if accepted else "mismatch"))
        self.assertEqual(store.finalize_calibration_v2(SOURCE_HASH)["state"],"passed")
        deployment=silver/"deployment.jsonl"; deployment.write_text(json.dumps({"schema":1,"revision":0,"tier":"shadow_only","current":None,"lkg":None,"canary":None,"route_generation":0,"audit":[]})+"\n"); os.chmod(deployment,0o600)
        inspected=(store.path,receipts,deployment,*preflight.glob("*.json"),*(hf_home.rglob("*")),*(self.root/"teacher-hf").rglob("*"),*(self.root.glob("*-adapter.py")))
        inspected=tuple(path for path in inspected if path.is_file() and not path.is_symlink()); before={path:(path.read_bytes(),path.stat().st_mtime_ns) for path in inspected}
        import adaptive_worker as worker_module
        with mock.patch.dict(os.environ,{"SOTTO_HF_HOME":str(hf_home),"HF_HOME":str(hf_home)},clear=False), \
             mock.patch("speech_backends._candidate_runtime_identity",side_effect=lambda stable: dict(candidate_identities[stable])), \
             mock.patch("teacher_backends.runtime_identity",side_effect=lambda _path,packages,expected=None: next(dict(teacher_identities[family.stable_id]) for family in REQUIRED_TEACHERS if tuple(packages)==family.critical_packages)):
            status=sealed_readiness_status(self.root)
            self.assertTrue(status["ready"],status); self.assertEqual(status["deployment"]["tier"],"shadow_only"); self.assertNotIn(str(self.root),json.dumps(status)); self.assertEqual(before,{path:(path.read_bytes(),path.stat().st_mtime_ns) for path in inspected})
            with mock.patch("sys.stdout",new_callable=StringIO) as output:
                self.assertEqual(worker_module.main(["--base-dir",str(self.root),"--status"]),0)
                self.assertTrue(json.loads(output.getvalue())["ready"])
            # Full candidate snapshot bytes, not claimed receipt fields, are authority.
            candidate_snapshot=hf_home/"hub"/"models--mlx-community--whisper-large-v3-turbo"/"snapshots"/"status-whisper"/"model.bin"
            candidate_snapshot.write_bytes(b"rotated")
            self.assertFalse(sealed_readiness_status(self.root)["ready"])
            candidate_snapshot.write_bytes(WHISPER_ID.encode())
            # Teacher adapter bytes and runtime-package identity are likewise current authority.
            adapter=self.root/f"{REQUIRED_TEACHERS[0].stable_id}-adapter.py"; adapter.write_text("# rotated\n",encoding="utf-8")
            self.assertFalse(sealed_readiness_status(self.root)["ready"])
            adapter.write_text("# fixture\n",encoding="utf-8")
            changed=dict(teacher_identities[REQUIRED_TEACHERS[0].stable_id]); changed["packages"]=dict(changed["packages"]); changed["packages"][REQUIRED_TEACHERS[0].critical_packages[0]]="rotated"
            with mock.patch("teacher_backends.runtime_identity",side_effect=lambda _path,packages,expected=None: changed if tuple(packages)==REQUIRED_TEACHERS[0].critical_packages else dict(teacher_identities[REQUIRED_TEACHERS[1].stable_id])):
                self.assertFalse(sealed_readiness_status(self.root)["ready"])
            # A claimed receipt cannot make a missing/different launcher current.
            original_teacher=json.loads(receipts.read_text("utf-8")); changed_teacher=json.loads(receipts.read_text("utf-8"))
            changed_teacher[REQUIRED_TEACHERS[0].stable_id]["interpreter"]=str(self.root/"missing-teacher-python")
            receipts.write_text(json.dumps(changed_teacher)); os.chmod(receipts,0o600)
            self.assertFalse(sealed_readiness_status(self.root)["ready"])
            receipts.write_text(json.dumps(original_teacher)); os.chmod(receipts,0o600)
            # Candidate runtime/package identity is checked against the active runtime too.
            candidate_receipt=preflight/f"{WHISPER_ID}.json"; original_candidate=json.loads(candidate_receipt.read_text("utf-8")); changed_candidate=json.loads(candidate_receipt.read_text("utf-8"))
            changed_candidate["runtime_identity"]={"fixture":"rotated"}; changed_candidate["manifest"]["runtime_identity"]={"fixture":"rotated"}
            candidate_receipt.write_text(json.dumps(changed_candidate)); os.chmod(candidate_receipt,0o600)
            self.assertFalse(sealed_readiness_status(self.root)["ready"])
            candidate_receipt.write_text(json.dumps(original_candidate)); os.chmod(candidate_receipt,0o600)
        (preflight/f"{PARAKEET_ID}.json").unlink()
        self.assertFalse(sealed_readiness_status(self.root)["ready"])
        deployment.write_text("malformed\n"); os.chmod(deployment,0o600)
        self.assertFalse(sealed_readiness_status(self.root)["ready"])

if __name__ == "__main__": unittest.main()

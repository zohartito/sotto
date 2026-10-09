from __future__ import annotations

import json
import hashlib
import os
import queue
import signal
from contextlib import contextmanager
from dataclasses import replace
from io import StringIO
import inspect
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from audio_codec import prepare_canonical, read_canonical_wav
from history import HistoryStore, _atomic_jsonl
from learning import LearningCoordinator, LearningStore
if sys.platform != "darwin":
    raise unittest.SkipTest("adaptive lane is macOS-only in v1 (fcntl)")

from adaptive_runtime import AdaptiveRuntime
from deployment_controller import DeploymentController
from silver_store import SilverStore
from speech_backends import (PARAKEET_ID, WHISPER_ID, PreflightReceipts,
                             _package_versions, _safe_snapshot_digest,
                             candidate_manifest, evaluator_hash)
from runtime_source_manifest import RUNTIME_SOURCE_SCHEMA, runtime_source_digest
from calibration_manifest_builder import SOURCE_HASH, _protocol_hash, LEDGER_HASH
from teacher_consensus import policy_hash
from speech_config import resolve_speech_config
import sotto


class _FakeWhisper:
    def __init__(self) -> None:
        self.calls = []

    def transcribe(self, samples, **kwargs):
        self.calls.append((samples, kwargs))
        return {"text": " model output "}


class SpeechRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        # Comparator spool authority intentionally rejects macOS's `/var`
        # symlink ancestry, so strict live-route fixtures use a lexical
        # private workspace root just like the production-safe spool tests.
        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.tmp.name)
        self.history = HistoryStore(self.root)
        self.learning = LearningStore(self.root)
        self.coordinator = LearningCoordinator(self.history, self.learning)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _enrolled(self):
        prepared = prepare_canonical(np.array([-.75, -.1, .1, .75], np.float32))
        row = self.coordinator.append_live("model text", prepared, .5, "model",
                                          language="pt", profile="auto",
                                          raw_samples=np.array([-.5, 0, .5], np.float32),
                                          raw_sample_rate=22_050)
        self.coordinator.correct(row["id"], "human reference")
        return row, self.coordinator.enroll(row["id"])

    def test_cli_resolution_precedence_and_short_glossary(self):
        config = resolve_speech_config("auto", model="local/test", language="en")
        self.assertEqual(config.model_repo, "local/test")
        self.assertEqual(config.language, "en")
        fake = _FakeWhisper()
        prepared = prepare_canonical(np.zeros(1600, np.float32))
        text, metadata = sotto.transcribe_prepared(fake, prepared, config, ("Sotto",))
        self.assertEqual(text, "model output")
        self.assertIs(fake.calls[0][0], prepared.asr_samples)
        self.assertEqual(fake.calls[0][1]["language"], "en")
        self.assertEqual(fake.calls[0][1]["initial_prompt"], "Speech context: Sotto.")
        self.assertFalse(fake.calls[0][1]["word_timestamps"])
        self.assertNotIn("hallucination_silence_threshold", fake.calls[0][1])
        self.assertEqual(metadata["profile"], "auto")

    def test_multilingual_model_auto_detects_or_accepts_a_language_hint(self):
        prepared = prepare_canonical(np.zeros(1600, np.float32))
        automatic = _FakeWhisper()
        text, metadata = sotto.transcribe_prepared(
            automatic, prepared, resolve_speech_config("auto", language="auto"), ()
        )
        self.assertEqual(text, "model output")
        self.assertNotIn("language", automatic.calls[0][1])
        self.assertIsNone(metadata["language"])

        hinted = _FakeWhisper()
        _text, metadata = sotto.transcribe_prepared(
            hinted, prepared, resolve_speech_config("auto", language="pt"), ()
        )
        self.assertEqual(hinted.calls[0][1]["language"], "pt")
        self.assertEqual(metadata["language"], "pt")

    def test_warmup_routing_is_adaptive_receipt_only_or_generic(self):
        config = resolve_speech_config("auto", language="en")
        generic = _FakeWhisper()
        class Runtime:
            def __init__(self): self.calls = []
            def warmup_baseline(self, samples): self.calls.append(samples)
        runtime = Runtime()
        sotto.warm_speech_runtime(adaptive=False, model="repo/main", speech_config=config, mlx_whisper=generic)
        self.assertEqual(generic.calls[0][1]["path_or_hf_repo"], "repo/main")
        sotto.warm_speech_runtime(adaptive=True, model="repo/main", speech_config=config, adaptive_runtime=runtime)
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual(len(generic.calls), 1)

    def test_model_rewarm_is_single_flight_baseline_only(self):
        due = sotto.MODEL_REWARM_AFTER_S
        self.assertFalse(sotto.model_rewarm_due(10.0, 10.0 + due - 0.01,
                                                rewarming=False, queue_idle=True,
                                                adaptive=False))
        self.assertTrue(sotto.model_rewarm_due(10.0, 10.0 + due,
                                               rewarming=False, queue_idle=True,
                                               adaptive=False))
        self.assertFalse(sotto.model_rewarm_due(10.0, 10.0 + due,
                                                rewarming=True, queue_idle=True,
                                                adaptive=False))
        self.assertFalse(sotto.model_rewarm_due(10.0, 10.0 + due,
                                                rewarming=False, queue_idle=False,
                                                adaptive=False))
        self.assertFalse(sotto.model_rewarm_due(10.0, 10.0 + due,
                                                rewarming=False, queue_idle=True,
                                                adaptive=True))

    def test_capture_service_construction_does_not_start_microphone(self):
        with patch.object(sotto.CaptureService, "_start_engine") as start:
            capture = sotto.CaptureService()
        start.assert_not_called()
        self.assertIsNone(capture._engine)
        self.assertIsNone(capture._node)
        self.assertEqual(capture.idle_release_s, 0.0)

    def test_capture_release_removes_tap_and_stops_only_while_idle(self):
        class Engine:
            def __init__(self): self.stops = 0
            def stop(self): self.stops += 1

        class Node:
            def __init__(self): self.removals = []
            def removeTapOnBus_(self, bus): self.removals.append(bus)

        capture = sotto.CaptureService()
        engine, node = Engine(), Node()
        capture._engine = engine
        capture._node = node
        capture._active = [np.ones(8, dtype=np.float32)]
        self.assertFalse(capture._release_engine(only_if_idle=True))
        self.assertIs(capture._engine, engine)
        self.assertEqual((engine.stops, node.removals), (0, []))

        capture._active = None
        capture._ring = [np.ones(4, dtype=np.float32)]
        capture._ring_samples = 4
        self.assertTrue(capture._release_engine(only_if_idle=True))
        self.assertIsNone(capture._engine)
        self.assertIsNone(capture._node)
        self.assertEqual((engine.stops, node.removals), (1, [0]))
        self.assertEqual((capture._ring, capture._ring_samples), ([], 0))

    def test_cli_zero_idle_release_means_immediate_not_never(self):
        with patch.object(sys, "argv", ["sotto.py", "run", "--idle-release", "0"]), \
             patch.object(sotto, "run") as run:
            sotto.main()
        self.assertEqual(run.call_args.kwargs["idle_release"], 0.0)

    def test_shutdown_signal_discards_active_capture_and_queued_jobs(self):
        class Capture:
            def __init__(self): self.calls = []
            def abort(self): self.calls.append("abort")
            def release_soon(self): self.calls.append("release")

        boundary = sotto.ShutdownBoundary()
        jobs: queue.Queue = queue.Queue()
        jobs.put(("live", "captured")); jobs.put(("retry", "queued"))
        prior = signal.getsignal(signal.SIGTERM)
        with sotto.shutdown_signal_handlers(boundary):
            handler = signal.getsignal(signal.SIGTERM)
            self.assertIsNot(handler, prior)
            handler(signal.SIGTERM, None)  # deterministic handler invocation, no OS signal
            self.assertTrue(boundary.requested())
        self.assertIs(signal.getsignal(signal.SIGTERM), prior)
        capture = Capture()
        boundary.stop_capture(capture)
        self.assertEqual(capture.calls, ["abort", "release"])
        self.assertEqual(boundary.discard_queued(jobs), 2)
        self.assertEqual(jobs.unfinished_tasks, 0)
        self.assertFalse(boundary.enqueue(jobs, ("live", "late")))

    def test_shutdown_drops_blocked_inference_result_but_normal_commit_works(self):
        boundary = sotto.ShutdownBoundary()
        entered, release = threading.Event(), threading.Event()
        pasted: list[str] = []

        def blocked_backend() -> str:
            entered.set()
            release.wait(2)
            return "partial text"

        def worker() -> None:
            text = blocked_backend()
            sotto.commit_if_running(boundary, lambda: pasted.append(text))

        thread = threading.Thread(target=worker, daemon=True)
        started = time.monotonic()
        thread.start()
        self.assertTrue(entered.wait(1))
        boundary.request()
        release.set()
        thread.join(sotto.APP_DRAIN_TIMEOUT)
        self.assertFalse(thread.is_alive())
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual(pasted, [])

        normal = sotto.ShutdownBoundary()
        self.assertTrue(sotto.commit_if_running(normal, lambda: pasted.append("complete text")))
        self.assertEqual(pasted, ["complete text"])

    def test_primary_live_delivery_never_schedules_cursor_before_durable_authority(self):
        """Primary append/register/ack failures are all pre-cursor fences."""
        publication={"comparator_publication":{"opaque":"token"}}
        for failure in ("append","register","ack","shutdown"):
            with self.subTest(failure=failure):
                events=[]; boundary=sotto.ShutdownBoundary()
                class Runtime:
                    def register_live(_self,row):
                        events.append("register")
                        if failure == "register": raise RuntimeError("register failed")
                    def acknowledge_comparator_publication(_self,value):
                        events.append("ack")
                        if failure == "shutdown": boundary.request()
                        return failure != "ack"
                    def cancel_comparator_publication(_self,value): events.append("cancel")
                    def fail_comparator_publication(_self,value): events.append("fail")
                def append():
                    events.append("append")
                    if failure == "append": raise RuntimeError("append failed")
                    return {"id":"history"}
                if failure in {"append","register","ack"}:
                    with self.assertRaises(RuntimeError):
                        sotto.finalize_primary_live_delivery(append=append,adaptive_runtime=Runtime(),
                            appended_publication=publication,shutdown=boundary,
                            inject=lambda: events.append("inject"))
                else:
                    self.assertEqual(sotto.finalize_primary_live_delivery(append=append,adaptive_runtime=Runtime(),
                        appended_publication=publication,shutdown=boundary,
                        inject=lambda: events.append("inject")),{"id":"history"})
                self.assertNotIn("inject",events)
        events=[]; boundary=sotto.ShutdownBoundary()
        class Success:
            def register_live(_self,row): events.append("register")
            def acknowledge_comparator_publication(_self,value): events.append("ack"); return True
            def cancel_comparator_publication(_self,value): raise AssertionError("unexpected cancel")
            def fail_comparator_publication(_self,value): raise AssertionError("unexpected fail")
        self.assertEqual(sotto.finalize_primary_live_delivery(append=lambda: events.append("append") or {"id":"history"},
            adaptive_runtime=Success(),appended_publication=publication,shutdown=boundary,
            inject=lambda: events.append("inject")),{"id":"history"})
        self.assertEqual(events,["append","register","ack","inject"])

    def test_run_shutdown_boundary_fences_capture_queue_and_post_inference_sinks(self):
        source = inspect.getsource(sotto.run)
        self.assertIn("if shutdown.requested():\n            shutdown.stop_capture(capture)", source)
        self.assertIn("shutdown.enqueue(jobs, (\"live\"", source)
        self.assertIn("shutdown.enqueue(jobs, (\"retry\"", source)
        self.assertIn("transcription_thread.join(APP_DRAIN_TIMEOUT)", source)
        # The post-inference fences live in the worker (transcription.py).
        from transcription import TranscriptionWorker
        worker = inspect.getsource(TranscriptionWorker._live)
        self.assertIn("# A backend may ignore cancellation.", worker)
        self.assertIn("finalize_primary_live_delivery(", worker)

    def test_learning_remove_dependency_revoke_precedes_artifact(self):
        calls = []
        self.assertTrue(sotto.remove_learning_sample("sample", history_id_lookup=lambda _: "history",
                        adaptive_revoke=lambda ident: calls.append(("adaptive", ident)),
                        artifact_revoke=lambda ident: calls.append(("artifact", ident)) or True))
        self.assertEqual(calls, [("adaptive", "history"), ("artifact", "sample")])

    def test_nonadaptive_mutation_helpers_guard_before_coordinator(self):
        calls = []
        class Guard:
            def pre_mutate_reference(self, ident): calls.append(("pre", ident))
            def revoke(self, ident, **kwargs): calls.append(("revoke", ident))
            def revoke_all(self, **kwargs): calls.append(("all",))
        class Coordinator:
            def review_transaction(self, ident, revision, before, mutation): before(); return mutation()
            def correct(self, ident, text): calls.append(("correct", ident)); return True
            def delete(self, ident): calls.append(("delete", ident)); return True
            def clear(self): calls.append(("clear",))
            def revoke_history(self, ident): calls.append(("learning", ident)); return 1
        guard, coordinator = Guard(), Coordinator()
        sotto.nonadaptive_correct(guard, coordinator, "x", "text")
        sotto.nonadaptive_delete(guard, coordinator, "x")
        sotto.nonadaptive_clear(guard, coordinator)
        sotto.nonadaptive_revoke_learning(guard, coordinator, "x")
        self.assertEqual(calls, [("revoke", "x"), ("correct", "x"), ("revoke", "x"), ("delete", "x"),
                                 ("all",), ("clear",), ("revoke", "x"), ("learning", "x")])

    def test_correction_disclosure_is_mode_aware_without_appkit_loop(self):
        import ui
        self.assertIn("separate", ui.correction_disclosure(False))
        self.assertIn("automatically", ui.correction_disclosure(True))

    def test_language_menu_offers_automatic_english_and_an_outside_choice(self):
        import ui
        options = ui.language_menu_options("pt")
        self.assertEqual([mode for mode, _label, _selected in options], ["auto", "en", "pt"])
        self.assertEqual([mode for mode, _label, selected in options if selected], ["pt"])
        self.assertEqual(options[0][1], "Automatic (English)")

    def test_nonadaptive_menu_retains_manual_enroll_action(self):
        import ui
        self.assertTrue(ui.show_manual_enroll_action(adaptive_mode=False, corrected=True, active=False))
        self.assertFalse(ui.show_manual_enroll_action(adaptive_mode=True, corrected=True, active=False))
        self.assertFalse(ui.show_manual_enroll_action(adaptive_mode=False, corrected=True, active=True))
        self.assertFalse(ui.show_manual_enroll_action(adaptive_mode=False, corrected=False, active=False))

    def test_adaptive_review_actions_require_adaptive_mode_and_reviewable_row(self):
        import ui
        self.assertTrue(ui.show_adaptive_review_actions(adaptive_mode=True, reviewable=True))
        self.assertFalse(ui.show_adaptive_review_actions(adaptive_mode=False, reviewable=True))
        self.assertFalse(ui.show_adaptive_review_actions(adaptive_mode=True, reviewable=False))
        self.assertTrue(ui.show_manual_enroll_action(adaptive_mode=True, corrected=True, active=False,
                                                      adaptive_eligible=False))
        self.assertFalse(ui.show_manual_enroll_action(adaptive_mode=True, corrected=True, active=False,
                                                       adaptive_eligible=True))

    def test_long_capture_disables_glossary_and_exact_prepared_reaches_store(self):
        fake = _FakeWhisper()
        prepared = prepare_canonical(np.linspace(-.2, .2, 16000 * 31, dtype=np.float32))
        config = resolve_speech_config("auto")
        _, metadata = sotto.transcribe_prepared(fake, prepared, config, ("Sotto",))
        self.assertIs(fake.calls[0][0], prepared.asr_samples)
        self.assertNotIn("initial_prompt", fake.calls[0][1])
        self.assertIn("disabled", metadata["prompt"])
        row = self.coordinator.append_live("output", prepared, 31, config.model_repo)
        persisted, identity = self.history.load_audio_with_identity(row["id"])
        self.assertEqual(identity, prepared.identity)
        self.assertTrue(np.array_equal(persisted, prepared.asr_samples))

    def test_primary_adaptive_prepared_path_routes_real_silver_and_persists_comparator(self):
        """The push-to-talk seam reaches the real strict route and intent store.

        The fixture freezes schema-3 receipts at the production cache paths,
        then constructs only content-free public rollout evidence.  The live
        call itself does not mock routing, receipt validation, or intent I/O.
        """
        hf_home=self.root / "hf"; identities={}; manifests=[]
        for stable,revision in ((WHISPER_ID,"baseline-revision"),(PARAKEET_ID,"silver-revision")):
            identity={"fixture_runtime":stable,"revision":revision}
            manifest=candidate_manifest(stable,revision=revision,runtime_identity=identity)
            snapshot=hf_home / "hub" / ("models--" + manifest["repo"].replace("/","--")) / "snapshots" / revision
            snapshot.mkdir(parents=True,exist_ok=True); (snapshot / "model.bin").write_bytes(revision.encode())
            manifests.append((stable,manifest,snapshot)); identities[stable]=identity
        receipts=PreflightReceipts(self.root)
        for stable,manifest,snapshot in manifests:
            digest=_safe_snapshot_digest(snapshot)
            receipts.save(stable,{"schema":3,"success":True,"manifest":{key:value for key,value in manifest.items() if key != "glossary_terms"},
                "snapshot_path":str(snapshot),"snapshot_revision":manifest["revision"],
                "snapshot_metadata_digest":digest,"package_versions":_package_versions(),
                "runtime_identity":identities[stable],"runtime_source_schema":RUNTIME_SOURCE_SCHEMA,
                "runtime_source_digest":runtime_source_digest(),"evaluator_id":"sotto-paired-v1",
                "evaluator_hash":evaluator_hash()})
            manifest["snapshot_digest"]=digest
        baseline=next(value for stable,value,_ in manifests if stable == WHISPER_ID)
        candidate=next(value for stable,value,_ in manifests if stable == PARAKEET_ID)
        # Receipts/horizons are JSON transports.  Use their exact JSON shape
        # for the frozen deployment lineage (not Python's empty tuple alias).
        candidate=json.loads(json.dumps(candidate,sort_keys=True))
        expected=[]
        fixture=Path(__file__).with_name("fixtures") / "calibration_v2_expected"
        for part in sorted(fixture.glob("part-*.json")): expected.extend(json.loads(part.read_text("utf-8")))
        ledger=hashlib.sha256(json.dumps([(index,row["item_hash"],row["cluster_hash"]) for index,row in enumerate(expected)],separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
        self.assertEqual((len(expected),ledger),(500,LEDGER_HASH))
        receipt_hash="fixture-receipt"; store=SilverStore(self.root)
        with patch.dict(os.environ,{"SOTTO_HF_HOME":str(hf_home),"HF_HOME":str(hf_home)},clear=False), \
             patch.object(PreflightReceipts,"_current_runtime_identity",new=lambda _self,stable: dict(identities[str(stable)])):
            store.begin_calibration_v2(holdout_key=SOURCE_HASH,manifest_hash="fixture-manifest",source_hash=SOURCE_HASH,
                protocol_hash=_protocol_hash(),policy_hash=policy_hash(),receipt_hash=receipt_hash,expected=expected)
            accepted_clusters=set()
            while (item:=store.claim_calibration_v2(SOURCE_HASH,"fixture")) is not None:
                accept=len(accepted_clusters)<200 and item["cluster_hash"] not in accepted_clusters
                if accept: accepted_clusters.add(item["cluster_hash"])
                self.assertTrue(store.finish_calibration_v2(item,outcome="accepted" if accept else "abstained",
                    reference_match=True if accept else None))
            self.assertEqual(store.finalize_calibration_v2(SOURCE_HASH)["state"],"passed")
            runtime=AdaptiveRuntime(self.root,history=self.history,learning=self.learning,coordinator=self.coordinator)
            self.assertTrue(receipts.valid_manifest(baseline), "strict fixture receipt failed validation")
            runtime.adaptive.ensure_baseline(runtime._receipt_manifest(WHISPER_ID))
            session=store.active_session_id(); epoch=store.epoch(); cohort="fixture-cohort"; horizon="fixture-horizon"; identity_hash="fixture-identity"
            outcome={"cohort_id":cohort,"identity_hash":identity_hash,"winner":PARAKEET_ID}
            meta={"winner":PARAKEET_ID,"evaluation_outcome":outcome,"teacher_receipt_hash":receipt_hash,
                  "teacher_policy_hash":policy_hash()}
            with store._connect() as db:
                db.execute("INSERT INTO prospective_horizons VALUES(?,?,?,?,?,?,?,?,?,?,?)",(horizon,"evaluated",0.,1.,json.dumps(baseline),json.dumps([candidate]),receipt_hash,identity_hash,epoch,0.,cohort))
                db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,"personal_horizon","evaluated_passed","members","comparison",json.dumps(meta,sort_keys=True),epoch,0.))
            controller=runtime.deployment; controller._write({"schema":1,"revision":1,"tier":"deployed_silver","current":dict(candidate),"lkg":dict(candidate),"cohort_id":cohort,"calibration":None,"audit":[],"route_generation":41,"canary":{"stage":"full","sticky_percent":100,"epoch":epoch,"observation_revision":1,"started_ts":0.}})
            _atomic_jsonl(controller.manifest_path,[controller._signed_manifest(candidate,cohort,outcome)])
            self.assertEqual(controller._winner(cohort),(candidate,outcome))
            calibration=store.calibration_status(); cohort_state=store.cohort_status(cohort)
            signed=json.loads(controller.manifest_path.read_text("utf-8")); expected_signed=controller._signed_manifest(candidate,cohort,outcome)
            self.assertEqual(calibration["state"],"passed"); self.assertEqual(calibration["receipt_hash"],receipt_hash)
            self.assertEqual(cohort_state["clear_epoch"],epoch); self.assertEqual(cohort_state["teacher_receipt_hash"],receipt_hash)
            self.assertEqual(signed["candidate"],candidate); self.assertEqual(signed["content_hash"],expected_signed["content_hash"]); self.assertEqual(signed["signature"],expected_signed["signature"])
            self.assertEqual((controller.route(session) or {}).get("stable_id"),PARAKEET_ID)
            calls=[]
            class Backend:
                def transcribe(_self,manifest,*_args,**_kwargs):
                    calls.append(manifest["stable_id"])
                    if manifest["stable_id"] != PARAKEET_ID: raise AssertionError("synchronous baseline comparator invoked")
                    return "calm blue river",{"repo":manifest["repo"]}
            runtime.backends=Backend()
            prepared=prepare_canonical(np.zeros(16_000,np.float32))
            text,metadata,staged=sotto.adaptive_live_transcribe_prepared(runtime,prepared,"primary-real")
            self.assertEqual(text,"calm blue river"); self.assertTrue(metadata["comparator_pending"]); self.assertEqual(calls,[PARAKEET_ID])
            self.assertEqual(staged,self.history.audio_path("primary-real")); disk,identity=read_canonical_wav(staged)
            self.assertEqual(identity,prepared.identity); self.assertTrue(np.array_equal(disk,prepared.asr_samples))
            expected_capture=runtime._comparator_capture_id(session,prepared.identity.sha256,prepared.identity.sample_count,1,"primary-real")
            with store._connect() as db:
                row=db.execute("SELECT capture_id,session_hash,cohort_id,audio_sha256,spool_name,status FROM comparator_intents").fetchone()
            self.assertEqual(row[:4],(expected_capture,hashlib.sha256(session.encode()).hexdigest(),cohort,prepared.identity.sha256)); self.assertEqual(row[5],"prepared")
            spool=self.history.silver_evidence_dir / "comparators" / row[4]
            self.assertTrue(spool.is_file()); self.assertEqual(spool.stat().st_mode & 0o777,0o600)
            appended=self.coordinator.append_live(text,prepared,1.,candidate["repo"],entry_id="primary-real",language="en",adaptive=True)
            self.assertEqual(appended["id"],"primary-real"); self.assertEqual(appended["audio"]["inference"]["sha256"],prepared.identity.sha256)
            self.assertTrue(runtime.acknowledge_comparator_publication(metadata))

    def test_primary_adaptive_prepared_path_cleans_failure(self):
        runtime=AdaptiveRuntime(self.root,history=self.history,learning=self.learning,coordinator=self.coordinator)
        prepared=prepare_canonical(np.zeros(1600,np.float32))
        runtime.live_transcribe=lambda *_args,**_kwargs: (_ for _ in ()).throw(RuntimeError("route failed"))  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError,"route failed"):
            sotto.adaptive_live_transcribe_prepared(runtime,prepared,"failed-capture")
        self.assertFalse(runtime.history.audio_path("failed-capture").exists())

    def test_retry_snapshot_cas_cannot_overwrite_correction(self):
        row, _ = self._enrolled()
        snapshot = self.coordinator.snapshot_for_retry(row["id"])
        fake = _FakeWhisper()
        sotto.transcribe_canonical_samples(
            fake, snapshot.samples, resolve_speech_config(), ())
        self.assertIs(fake.calls[0][0], snapshot.samples)
        self.coordinator.correct(row["id"], "later human correction")
        self.assertIsNone(self.coordinator.commit_retry(
            row["id"], "stale model retry", snapshot.expected_revision))
        self.assertEqual(self.history.get(row["id"])["text"], "later human correction")

    def test_overlay_status_and_export_are_headless_and_manifest_compatible(self):
        row, _ = self._enrolled()
        snapshots = self.coordinator.export_learning_snapshot()
        overlaid = sotto.overlay_learning_state(
            self.history.entries(10), {row["id"]})
        self.assertEqual(overlaid[0]["learning_state"], "active")
        self.assertEqual(sotto.learning_status_text(snapshots), "1 active learning item(s)")
        sys.modules.pop("mlx_whisper", None)
        with patch.object(HistoryStore, "__init__", side_effect=AssertionError("history constructor")), \
             patch.object(LearningStore, "__init__", side_effect=AssertionError("learning constructor")), \
             patch.object(LearningCoordinator, "__init__", side_effect=AssertionError("coordinator constructor")), \
             patch.object(LearningStore, "admin_snapshot", return_value=snapshots) as admin_snapshot, \
             patch.object(sys, "argv", ["sotto.py", "learning-status"]), \
             patch("sys.stdout", new_callable=StringIO) as stdout:
            sotto.main()
        self.assertIn("1 active", stdout.getvalue())
        admin_snapshot.assert_called_once_with()
        self.assertNotIn("mlx_whisper", sys.modules)

        list_snapshots = [
            replace(snapshots[0], sample_id="z-sample", history_id="history-z",
                    corrected_text="private reference z"),
            replace(snapshots[0], sample_id="a-sample", history_id="history-a",
                    corrected_text="private reference a"),
        ]
        with patch.object(HistoryStore, "__init__", side_effect=AssertionError("history constructor")), \
             patch.object(LearningStore, "__init__", side_effect=AssertionError("learning constructor")), \
             patch.object(LearningCoordinator, "__init__", side_effect=AssertionError("coordinator constructor")), \
             patch.object(LearningStore, "admin_snapshot", return_value=list_snapshots) as admin_snapshot, \
             patch.object(sys, "argv", ["sotto.py", "learning-list"]), \
             patch("sys.stdout", new_callable=StringIO) as stdout:
            sotto.main()
        self.assertEqual(stdout.getvalue().splitlines(), [
            "sample_id=a-sample history_id=history-a",
            "sample_id=z-sample history_id=history-z",
        ])
        self.assertNotIn("private reference", stdout.getvalue())
        admin_snapshot.assert_called_once_with()

        with patch.object(HistoryStore, "__init__", side_effect=AssertionError("history constructor")), \
             patch.object(LearningStore, "__init__", side_effect=AssertionError("learning constructor")), \
             patch.object(LearningCoordinator, "__init__", side_effect=AssertionError("coordinator constructor")), \
             patch.object(LearningStore, "admin_snapshot", return_value=[]), \
             patch.object(sys, "argv", ["sotto.py", "learning-list"]), \
             patch("sys.stdout", new_callable=StringIO) as stdout:
            sotto.main()
        self.assertEqual(stdout.getvalue(), "0 active learning item(s)\n")

        with patch.object(HistoryStore, "__init__", side_effect=AssertionError("history constructor")), \
             patch.object(LearningStore, "__init__", side_effect=AssertionError("learning constructor")), \
             patch.object(LearningCoordinator, "__init__", side_effect=AssertionError("coordinator constructor")), \
             patch.object(LearningStore, "admin_revoke", return_value=True) as admin_revoke, \
             patch.object(sys, "argv", ["sotto.py", "learning-remove", "--sample-id", "a-sample"]), \
             patch("sys.stdout", new_callable=StringIO) as stdout:
            sotto.main()
        self.assertEqual(stdout.getvalue(), "Removed learning sample a-sample\n")
        admin_revoke.assert_called_once_with("a-sample")

        with patch.object(HistoryStore, "__init__", side_effect=AssertionError("history constructor")), \
             patch.object(LearningStore, "__init__", side_effect=AssertionError("learning constructor")), \
             patch.object(LearningCoordinator, "__init__", side_effect=AssertionError("coordinator constructor")), \
             patch.object(LearningStore, "admin_revoke", return_value=False) as admin_revoke, \
             patch.object(sys, "argv", ["sotto.py", "learning-remove", "--sample-id", "gone"]), \
             patch("sys.stderr", new_callable=StringIO) as stderr:
            with self.assertRaises(SystemExit) as exit_error:
                sotto.main()
        self.assertEqual(exit_error.exception.code, 2)
        self.assertIn("not found or was already removed", stderr.getvalue())
        admin_revoke.assert_called_once_with("gone")
        self.assertNotIn("mlx_whisper", sys.modules)

        output = self.root / "export"
        guard_exited = []
        @contextmanager
        def export_guard():
            yield snapshots
            self.assertTrue((output / "manifest.jsonl").is_file())
            self.assertTrue((output / "audio" / f"{snapshots[0].sample_id}.wav").is_file())
            self.assertTrue((output / "raw" / f"{snapshots[0].sample_id}.wav").is_file())
            guard_exited.append(True)
            for snapshot in snapshots:
                snapshot.inference_audio_path.unlink(missing_ok=True)
                if snapshot.raw_audio_path is not None:
                    snapshot.raw_audio_path.unlink(missing_ok=True)

        from benchmark import load_manifest
        with patch.object(HistoryStore, "__init__", side_effect=AssertionError("history constructor")), \
             patch.object(LearningStore, "__init__", side_effect=AssertionError("learning constructor")), \
             patch.object(LearningCoordinator, "__init__", side_effect=AssertionError("coordinator constructor")), \
             patch.object(LearningStore, "admin_export_guard", side_effect=export_guard) as admin_guard, \
             patch.object(sys, "argv", ["sotto.py", "export-learning", "--output", str(output)]), \
             patch("sys.stdout", new_callable=StringIO) as stdout:
            sotto.main()
        self.assertIn("Exported 1 learning item", stdout.getvalue())
        admin_guard.assert_called_once_with()
        self.assertEqual(guard_exited, [True])
        manifest_rows = load_manifest(output / "manifest.jsonl")
        self.assertEqual(manifest_rows[0].sample_id, snapshots[0].sample_id)
        self.assertEqual(manifest_rows[0].reference, "human reference")
        self.assertTrue((output / "audio" / f"{manifest_rows[0].sample_id}.wav").is_file())
        self.assertTrue((output / "raw" / f"{manifest_rows[0].sample_id}.wav").is_file())
        self.assertFalse(snapshots[0].inference_audio_path.exists())
        self.assertFalse(snapshots[0].raw_audio_path.exists())
        with self.assertRaises(ValueError):
            sotto.export_learning_snapshot(output)
        self.assertNotIn("mlx_whisper", sys.modules)

    def test_adaptive_status_reads_only_current_deployment_jsonl(self):
        root=self.root; adaptive=root/"adaptive-learning"; silver=adaptive/"silver"; silver.mkdir(parents=True,exist_ok=True)
        state=adaptive/"state.json"; deploy=silver/"deployment.jsonl"
        state.write_text(json.dumps({"captures":{},"champions":{"current":None,"baseline":{"stable_id":"base"}},"generations":{}}))
        deploy.write_text(json.dumps({"schema":1,"tier":"provisional_silver","canary":{"stage":"five_percent"}})+"\n")
        before={path:(path.read_bytes(),path.stat().st_mtime_ns) for path in (state,deploy)}
        with patch("silver_store.read_only_status",return_value={"state":"ok","accepted":2,"abstained":1,"jobs":{"pending":3},"worker":{"state":"idle"}}):
            value=sotto.adaptive_status_text([],base_dir=root)
        self.assertIn("deployment=provisional_silver",value); self.assertIn("silver=2 accepted/1 abstained",value)
        self.assertEqual(before,{path:(path.read_bytes(),path.stat().st_mtime_ns) for path in (state,deploy)})
        deploy.write_text("not-json\n")
        with patch("silver_store.read_only_status",return_value={"state":"ok"}):
            self.assertIn("unavailable (fail closed)",sotto.adaptive_status_text([],base_dir=root))
        deploy.unlink()
        self.assertIn("unavailable (fail closed)",sotto.adaptive_status_text([],base_dir=root))

    def test_ui_revoke_uses_current_disk_coordinator_operation(self):
        runtime_source = inspect.getsource(sotto.run)
        self.assertIn("nonadaptive_revoke_learning(dependency_guard, coordinator, entry_id)", runtime_source)
        self.assertNotIn("coordinator.learning.active_for_history(entry_id)", runtime_source)
        self.assertIn("Removed {count} local learning sample(s).", runtime_source)

    def test_copy_and_save_callbacks_only_start_daemon_workers(self):
        runtime_source = inspect.getsource(sotto.run)
        copy_source = runtime_source.split("def action_copy(entry_id: str) -> None:", 1)[1].split(
            "def action_retry(entry_id: str) -> None:", 1)[0]
        save_source = runtime_source.split("def action_save(entry_id: str) -> None:", 1)[1].split(
            "def action_delete(entry_id: str) -> None:", 1)[0]
        for source in (copy_source, save_source):
            worker = source.index("def work() -> None:")
            self.assertNotIn("store.get", source[:worker])
            self.assertIn("threading.Thread(target=work, daemon=True).start()", source)
            self.assertLess(worker, source.index("store.get"))
        self.assertIn("deliver_call(_copy_text, entry[\"text\"])", copy_source)  # counted (N19)
        self.assertIn('save_audio_copy(store.audio_path(entry_id), Path.home() / "Desktop",',
                      save_source)
        self.assertNotIn("shutil.copy", save_source)

    def test_concurrent_export_same_destination_has_one_clean_winner(self):
        _, _ = self._enrolled()
        snapshots = self.coordinator.export_learning_snapshot()
        output = self.root / "concurrent-export"
        guard_lock = threading.Lock()
        start = threading.Barrier(2)
        successes: list[int] = []
        failures: list[Exception] = []

        @contextmanager
        def serialized_guard():
            with guard_lock:
                yield snapshots

        def export() -> None:
            start.wait(timeout=2)
            try:
                successes.append(sotto.export_learning_snapshot(output))
            except Exception as exc:
                failures.append(exc)

        with patch.object(LearningStore, "admin_export_guard", side_effect=serialized_guard):
            first = threading.Thread(target=export)
            second = threading.Thread(target=export)
            first.start(); second.start()
            first.join(timeout=3); second.join(timeout=3)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(successes, [1])
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], ValueError)
        from benchmark import load_manifest
        rows = load_manifest(output / "manifest.jsonl")
        self.assertEqual(len(rows), 1)
        sample_id = snapshots[0].sample_id
        self.assertEqual(sorted(
            path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()),
            [f"audio/{sample_id}.wav", "manifest.jsonl", f"raw/{sample_id}.wav"],
        )


if __name__ == "__main__":
    unittest.main()

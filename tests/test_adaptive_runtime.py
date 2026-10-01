from __future__ import annotations

import tempfile
import stat
import threading
import os
import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import contextmanager

import numpy as np

import sys

if sys.platform != "darwin":
    raise unittest.SkipTest("adaptive lane is macOS-only in v1 (fcntl)")

from adaptive_learning import AdaptiveLearning
from adaptive_runtime import AdaptiveRuntime
from adaptive_worker import AdaptiveWorker
from audio_codec import encode_canonical, prepare_canonical, read_canonical_wav, write_canonical_wav
from deployment_controller import DeploymentController
from history import HistoryStore
from learning import LearningCoordinator, LearningStore
from speech_backends import WHISPER_ID, PreflightReceipts, _package_versions, _safe_snapshot_digest, candidate_manifest, evaluator_hash
from speech_backends import WHISPER_GLOSSARY_ID
from runtime_source_manifest import RUNTIME_SOURCE_SCHEMA, runtime_source_digest


@contextmanager
def strict_receipts(root: Path, manifests: list[dict]):
    """Create current schema-3 receipts at the same strict cache paths as production.

    The production identity checks are deliberately not mocked away: the
    fixture supplies a coherent frozen identity through the receipt resolver,
    just as the durable lifecycle test does.
    """
    hf_home=root / "hf"; identities: dict[str,dict] = {}; snapshots: dict[str,Path] = {}
    for manifest in manifests:
        stable=str(manifest["stable_id"])
        identity={"fixture_runtime": stable, "revision": str(manifest["revision"])}
        manifest["runtime_identity"]=identity
        snapshot=hf_home / "hub" / ("models--" + str(manifest["repo"]).replace("/","--")) / "snapshots" / str(manifest["revision"])
        snapshot.mkdir(parents=True,exist_ok=True); (snapshot / "model").write_text(str(manifest["revision"]))
        identities[stable]=identity; snapshots[stable]=snapshot
    receipts=PreflightReceipts(root)
    for manifest in manifests:
        stable=str(manifest["stable_id"]); snapshot=snapshots[stable]
        digest=_safe_snapshot_digest(snapshot)
        receipts.save(stable,{"schema":3,"success":True,
            "manifest":{key:value for key,value in manifest.items() if key != "glossary_terms"},
            "snapshot_path":str(snapshot),"snapshot_revision":manifest["revision"],
            "snapshot_metadata_digest":digest,"package_versions":_package_versions(),
            "runtime_identity":identities[stable],"runtime_source_schema":RUNTIME_SOURCE_SCHEMA,
            "runtime_source_digest":runtime_source_digest(),"evaluator_id":"sotto-paired-v1","evaluator_hash":evaluator_hash()})
        # Freeze-time nonbaseline authority includes the transport digest, but
        # it remains outside the receipt's public semantic manifest.
        manifest["snapshot_digest"]=digest
    with patch.dict(os.environ,{"SOTTO_HF_HOME":str(hf_home),"HF_HOME":str(hf_home)},clear=False), \
         patch.object(PreflightReceipts,"_current_runtime_identity",new=lambda _self,stable: dict(identities[str(stable)])):
        yield receipts,snapshots


class AdaptiveRuntimeCoreTests(unittest.TestCase):
    def test_orphan_sweep_is_coalesced_fires_reconcile_and_respects_shutdown(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            runtime=AdaptiveRuntime(directory)
            fired=threading.Event()
            with patch.object(runtime,"reconcile",side_effect=lambda **_kw: fired.set()):
                # Coalescing: while one timer is pending, scheduling only
                # flags a re-arm instead of creating a second timer.
                runtime.schedule_orphan_sweep(_delay=30.0)
                first=runtime._orphan_sweep_timer
                runtime.schedule_orphan_sweep(_delay=30.0)
                self.assertIs(runtime._orphan_sweep_timer,first)
                self.assertTrue(runtime._orphan_sweep_rearm)
                first.cancel(); first.join(5.0)
                runtime._orphan_sweep_timer=None; runtime._orphan_sweep_rearm=False
                # A fresh timer fires reconcile exactly once past its delay.
                runtime.schedule_orphan_sweep(_delay=0.01)
                self.assertTrue(fired.wait(5.0))
                runtime._orphan_sweep_timer.join(5.0)
            # A request arriving mid-window re-arms one fresh full-window
            # timer, so an orphan created during the first window still gets
            # a sweep past its own expiry.
            counts=[]
            done=threading.Event()
            with patch.object(runtime,"reconcile",side_effect=lambda **_kw: (counts.append(1),len(counts)>=2 and done.set())):
                runtime._orphan_sweep_timer=None; runtime._orphan_sweep_rearm=False
                runtime.schedule_orphan_sweep(_delay=0.05)
                runtime.schedule_orphan_sweep(_delay=0.05)
                self.assertTrue(done.wait(5.0))
                runtime._orphan_sweep_timer.join(5.0)
            self.assertEqual(len(counts),2)
            self.assertFalse(runtime._orphan_sweep_rearm)
            # Shutdown suppresses the deferred pass entirely.
            fired.clear()
            with patch.object(runtime,"reconcile",side_effect=lambda **_kw: fired.set()):
                runtime._orphan_sweep_timer=None
                runtime._orphan_sweep_rearm=False
                runtime.schedule_orphan_sweep(is_shutdown=lambda: True,_delay=0.01)
                runtime._orphan_sweep_timer.join(5.0)
                self.assertFalse(fired.is_set())

    def test_comparator_delivery_adoption_recovers_primary_and_retry_after_commit_before_ack(self):
        """History commits, not a volatile callback, decide prepared delivery."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            runtime=AdaptiveRuntime(directory); store=runtime.silver; epoch=store.epoch(); cohort="adoption"; session="adoption-session"; generation=17
            with store._connect() as db:
                db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,"h","evaluated_passed","m","c","{}",epoch,0))
                db.execute("INSERT INTO route_assignments VALUES(?,?,?,?)",(generation,hashlib.sha256(session.encode()).hexdigest(),1,0))
            candidate={"success":True,"fallback":False,"latency":.1,"coverage_ok":True,"hallucination_ok":True,"identity_valid":True}
            prepared=prepare_canonical(np.zeros(800,np.float32))

            def publication(capture: str, revision: int, history_id: str):
                wav,digest,_=runtime._prepare_comparator_spool(prepared,16_000); path,_=runtime._write_comparator_spool(wav,capture,digest)
                exact={"deployment_revision":9,"capture_id":capture,"session_id":session,"cohort_id":cohort,"epoch":epoch,
                       "route_generation":generation,"candidate_arm":"candidate","audio_sha256":digest,"spool_name":path.name,"candidate":candidate}
                self.assertEqual(store.enqueue_comparator_intent(**exact),"inserted")
                metadata={"comparator_publication":{key:exact[key] for key in ("deployment_revision","capture_id","spool_name","audio_sha256","candidate")}}
                self.assertTrue(runtime.adopt_comparator_publication(metadata,history_id=history_id,history_revision=revision))
                return metadata,digest

            # Primary append/register has committed; simulate process death
            # before the app-level prepared→pending acknowledgement.
            primary,digest=publication("primary-outbox",0,"primary-history")
            row=runtime.coordinator.append_live("primary",prepared,.05,"candidate",entry_id="primary-history",language="en",adaptive=True)
            # The process dies after registration itself, before the normal
            # outer-loop acknowledgement; suppress its ordinary reconciliation
            # callback to model that exact crash boundary.
            runtime.reconcile=lambda **_kwargs: {"state":"crashed_before_ack"}
            runtime.register_live(row)
            with store._connect() as db:
                db.execute("UPDATE comparator_intents SET created_ts=0 WHERE capture_id='primary-outbox'")
            self.assertEqual(store.recover(),0)  # adopted is never generic-expired
            restarted=AdaptiveRuntime(directory)
            self.assertEqual(restarted.recover_comparator_publications(),1)
            with restarted.silver._connect() as db:
                self.assertEqual(db.execute("SELECT status FROM comparator_intents WHERE capture_id='primary-outbox'").fetchone()[0],"pending")
            self.assertFalse(restarted.silver.comparator_publication_adoptions())

            # Retry commit increments its History revision; the exact outbox
            # link survives the same crash window and is replayed once.
            retry_base=runtime.coordinator.append_live("old",prepared,.05,"candidate",entry_id="retry-history",language="en",adaptive=True)
            retry,digest=publication("retry-outbox",retry_base["revision"] + 1,"retry-history")
            committed=runtime.coordinator.commit_retry("retry-history","retry",retry_base["revision"],model="candidate",language="en",adaptive=True)
            self.assertIsNotNone(committed)
            with store._connect() as db:
                db.execute("UPDATE comparator_intents SET created_ts=0 WHERE capture_id='retry-outbox'")
            self.assertEqual(store.recover(),0)
            # Retry's old-revision revocation remains a hard scrub fence; it
            # must clear before restart can promote the adopted successor.
            marker=store.scrub_pending(); self.assertIsNotNone(marker); self.assertTrue(store.complete_scrub(marker))
            restarted=AdaptiveRuntime(directory)
            self.assertEqual(restarted.recover_comparator_publications(),1)
            with restarted.silver._connect() as db:
                self.assertEqual(db.execute("SELECT status FROM comparator_intents WHERE capture_id='retry-outbox'").fetchone()[0],"pending")

            # An adoption staged before a History commit is unadopted, even
            # after a delayed restart; it never becomes worker claimable.
            missing,digest=publication("missing-outbox",0,"missing-history")
            with store._connect() as db:
                db.execute("UPDATE comparator_intents SET created_ts=0 WHERE capture_id='missing-outbox'")
            restarted=AdaptiveRuntime(directory)
            self.assertEqual(restarted.recover_comparator_publications(),0)
            with restarted.silver._connect() as db:
                self.assertEqual(db.execute("SELECT status FROM comparator_intents WHERE capture_id='missing-outbox'").fetchone()[0],"discarded")
                self.assertIsNone(db.execute("SELECT 1 FROM meta WHERE key LIKE 'comparator_adoption:%missing-outbox'").fetchone())
            wav,digest,_=runtime._prepare_comparator_spool(prepared,16_000); orphan_path,_=runtime._write_comparator_spool(wav,"orphan-outbox",digest)
            orphan={"deployment_revision":9,"capture_id":"orphan-outbox","session_id":session,"cohort_id":cohort,"epoch":epoch,
                    "route_generation":generation,"candidate_arm":"candidate","audio_sha256":digest,"spool_name":orphan_path.name,"candidate":candidate}
            self.assertEqual(store.enqueue_comparator_intent(**orphan),"inserted")
            with store._connect() as db:
                db.execute("UPDATE comparator_intents SET created_ts=0 WHERE capture_id='orphan-outbox'")
            store.recover()
            with store._connect() as db:
                self.assertEqual(db.execute("SELECT status FROM comparator_intents WHERE capture_id='orphan-outbox'").fetchone()[0],"discarded")

    def test_correction_or_scrub_fences_adopted_comparator_acknowledgement(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            runtime=AdaptiveRuntime(directory); store=runtime.silver; epoch=store.epoch(); cohort="adoption-fence"; session="fence-session"; generation=18
            with store._connect() as db:
                db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,"h","evaluated_passed","m","c","{}",epoch,0))
                db.execute("INSERT INTO route_assignments VALUES(?,?,?,?)",(generation,hashlib.sha256(session.encode()).hexdigest(),1,0))
            prepared=prepare_canonical(np.zeros(800,np.float32)); candidate={"success":True,"fallback":False,"latency":.1,"coverage_ok":True,"hallucination_ok":True,"identity_valid":True}
            wav,digest,_=runtime._prepare_comparator_spool(prepared,16_000); path,_=runtime._write_comparator_spool(wav,"fenced",digest)
            exact={"deployment_revision":10,"capture_id":"fenced","session_id":session,"cohort_id":cohort,"epoch":epoch,
                   "route_generation":generation,"candidate_arm":"candidate","audio_sha256":digest,"spool_name":path.name,"candidate":candidate}
            self.assertEqual(store.enqueue_comparator_intent(**exact),"inserted")
            metadata={"comparator_publication":{key:exact[key] for key in ("deployment_revision","capture_id","spool_name","audio_sha256","candidate")}}
            self.assertTrue(runtime.adopt_comparator_publication(metadata,history_id="fenced-history",history_revision=0))
            runtime.coordinator.append_live("candidate",prepared,.05,"candidate",entry_id="fenced-history",language="en",adaptive=True)
            # A correction/revoke removes the exact adoption in the same
            # Silver transaction before it updates the visible history row.
            self.assertIsNotNone(runtime.coordinator.commit_retry("fenced-history","human",0,model="human",language="en",adaptive=True))
            self.assertFalse(runtime.acknowledge_comparator_publication(metadata))
            with store._connect() as db:
                self.assertEqual(db.execute("SELECT status FROM comparator_intents WHERE capture_id='fenced'").fetchone()[0],"discarded")
            # A current adoption also cannot cross a Clear scrub fence.
            store.clear()
            self.assertFalse(runtime.acknowledge_comparator_publication(metadata))

    def test_comparator_acknowledgement_strict_state_exception_returns_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            runtime=AdaptiveRuntime(directory)
            metadata={"comparator_publication":{"deployment_revision":1,"capture_id":"capture",
                      "spool_name":"capture.wav","audio_sha256":"a" * 64,
                      "candidate":{"success":True,"fallback":False,"latency":.1,
                                   "coverage_ok":True,"hallucination_ok":True,"identity_valid":True}}}
            runtime.silver.comparator_publication_adoptions=lambda: (_ for _ in ()).throw(RuntimeError("malformed outbox"))
            self.assertFalse(runtime.acknowledge_comparator_publication(metadata))

    def test_live_audio_authority_rejects_memory_path_and_identity_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime=AdaptiveRuntime(directory); samples=np.zeros(800,np.float32); prepared=prepare_canonical(samples)
            path=runtime.history.audio_path('live-authority'); write_canonical_wav(path,prepared.pcm)
            token=runtime._live_audio_authority(samples,path,prepared.identity)
            self.assertIsNotNone(token); self.assertTrue(runtime._same_live_audio_authority(token,samples,path,prepared.identity))
            changed=samples.copy(); changed[0]=.5
            self.assertFalse(runtime._same_live_audio_authority(token,changed,path,prepared.identity))
            replacement=prepare_canonical(np.ones(800,np.float32)*.5); write_canonical_wav(path,replacement.pcm)
            self.assertFalse(runtime._same_live_audio_authority(token,samples,path,prepared.identity))
            write_canonical_wav(path,prepared.pcm)
            wrong={**prepared.identity.as_dict(),'sha256':'0'*64}
            self.assertIsNone(runtime._live_audio_authority(samples,path,wrong))

    def test_comparator_symlink_root_never_follows_or_unlinks_outside(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            runtime=AdaptiveRuntime(directory); outside=Path(directory)/"outside"; outside.mkdir()
            sentinel=outside/"capture.wav"; sentinel.write_bytes(b"outside")
            root=runtime.history.silver_evidence_dir/"comparators"; root.symlink_to(outside,target_is_directory=True)
            wav,digest,_=runtime._prepare_comparator_spool(np.zeros(800,np.float32),16000)
            with self.assertRaises((RuntimeError,OSError)):
                runtime._write_comparator_spool(wav,"capture",digest)
            runtime._delete_comparator_spool("capture",digest)
            self.assertEqual(sentinel.read_bytes(),b"outside")

    def test_live_silver_mutation_matrix_discards_candidate_without_pair_artifacts(self):
        for mutation in ('human_promotion','receipt_rotation','audio_drift'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                runtime=AdaptiveRuntime(directory); baseline=candidate_manifest(WHISPER_ID,revision='base')
                human={**candidate_manifest(WHISPER_ID,revision='human'),'stable_id':'human','snapshot_digest':'human-digest'}
                silver={**candidate_manifest(WHISPER_ID,revision='silver'),'stable_id':'silver','snapshot_digest':'silver-digest'}
                runtime.adaptive.ensure_baseline(baseline)
                samples=np.zeros(1600,np.float32); prepared=prepare_canonical(samples)
                path=runtime.history.audio_path('live-matrix'); write_canonical_wav(path,prepared.pcm)
                state={'rotated':False}; enqueues=[]; token={'deployment_revision':1,'route_generation':1,'candidate_hash':'x','tier':'provisional_silver','cohort_id':'c','epoch':0,'session_hash':'s'}
                @contextmanager
                def live_request(): yield
                runtime.scheduler.live_request=live_request
                def validate(manifest,require_snapshot_digest=False):
                    if manifest.get('stable_id') == 'silver' and state['rotated']: return None
                    return dict(manifest)
                runtime._validated_persisted_manifest=validate
                runtime._receipt_manifest=lambda ident: dict(baseline) if ident == WHISPER_ID else None
                runtime.deployment.route=lambda _session: {**silver,'_deployment_token':dict(token)}
                runtime.deployment.validate_route_token=lambda _token: True
                runtime.deployment.enqueue_comparator_intent=lambda **kwargs: enqueues.append(kwargs) or 'inserted'
                class Backend:
                    def transcribe(_self,manifest,_samples,**_kwargs):
                        ident=manifest['stable_id']
                        if ident == 'silver':
                            if mutation == 'human_promotion':
                                runtime.adaptive._mutate(lambda value: value['champions'].__setitem__('current',dict(human)))
                            elif mutation == 'receipt_rotation':
                                state['rotated']=True
                            else:
                                changed=prepare_canonical(np.ones(1600,np.float32)*.5)
                                write_canonical_wav(path,changed.pcm)
                            return 'candidate-stale',{}
                        return ('human-current' if ident == 'human' else 'baseline-current'),{}
                runtime.backends=Backend()
                if mutation == 'audio_drift':
                    with self.assertRaisesRegex(RuntimeError,'live fallback authority changed'):
                        runtime.live_transcribe(samples,canonical_path=path,canonical_identity=prepared.identity,capture_id='matrix-capture')
                else:
                    text,metadata=runtime.live_transcribe(samples,canonical_path=path,canonical_identity=prepared.identity,capture_id='matrix-capture')
                    self.assertEqual(text,'human-current' if mutation == 'human_promotion' else 'baseline-current')
                    self.assertTrue(metadata['fallback'])
                self.assertFalse(enqueues)
                if mutation == 'human_promotion':
                    self.assertEqual(runtime.adaptive.champion_manifest()['stable_id'],'human')
                else:
                    self.assertEqual(runtime.adaptive.champion_manifest()['stable_id'],WHISPER_ID)
                comparator_dir=runtime.history.silver_evidence_dir/'comparators'
                self.assertFalse(comparator_dir.exists() and any(comparator_dir.iterdir()))
                with runtime.silver._connect() as db:
                    self.assertEqual(db.execute('SELECT COUNT(*) FROM runtime_pairs').fetchone()[0],0)
                    self.assertEqual(db.execute('SELECT COUNT(*) FROM comparator_intents').fetchone()[0],0)

    def test_stable_silver_success_validates_its_inference_manifest(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            runtime=AdaptiveRuntime(directory); baseline=candidate_manifest(WHISPER_ID,revision="base")
            silver={**candidate_manifest(WHISPER_ID,revision="silver"),"stable_id":"silver"}
            runtime.adaptive.ensure_baseline(baseline)
            @contextmanager
            def live_request(): yield
            runtime.scheduler.live_request=live_request
            runtime._validated_persisted_manifest=lambda manifest,require_snapshot_digest=False: dict(manifest)
            runtime._receipt_manifest=lambda _stable_id: dict(baseline)
            runtime._validate_human_authority=lambda _token: True
            runtime.adaptive.record_runtime_success=lambda *_args,**_kwargs: None
            token={"deployment_revision":1,"route_generation":1,"candidate_hash":"x","tier":"provisional_silver","cohort_id":"c","epoch":0,"session_hash":"s"}
            runtime.deployment.route=lambda _session: {**silver,"_deployment_token":token}
            runtime.deployment.validate_route_token=lambda _value: True
            intents=[]
            runtime.deployment.enqueue_comparator_intent=lambda **kwargs: intents.append(kwargs) or "inserted"
            runtime.silver.acknowledge_comparator_intent=lambda **_kwargs: True
            class Backend:
                def transcribe(self,manifest,*_args,**_kwargs):
                    if manifest["stable_id"] != "silver": raise AssertionError("foreground comparator invoked baseline")
                    return ("silver-current",{"repo":"silver"})
            runtime.backends=Backend()
            prepared=prepare_canonical(np.zeros(8,np.float32)); path=runtime.history.audio_path("stable-silver"); write_canonical_wav(path,prepared.pcm)
            text,metadata=runtime.live_transcribe(np.zeros(8,np.float32),canonical_path=path,canonical_identity=prepared.identity)
            self.assertEqual(text,"silver-current")
            self.assertFalse(metadata.get("fallback",False))
            self.assertTrue(metadata["comparator_pending"]); self.assertEqual(len(intents),1)
            self.assertEqual(intents[0]["candidate"]["latency"],metadata.get("latency",intents[0]["candidate"]["latency"]))

    def test_human_promotion_during_silver_inference_discards_silver_output(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime=AdaptiveRuntime(directory)
            baseline=candidate_manifest(WHISPER_ID,revision="base")
            human={**candidate_manifest(WHISPER_ID,revision="human"),"stable_id":"human"}
            silver={**candidate_manifest(WHISPER_ID,revision="silver"),"stable_id":"silver"}
            runtime.adaptive.ensure_baseline(baseline)
            @contextmanager
            def live_request(): yield
            runtime.scheduler.live_request=live_request
            runtime._validated_persisted_manifest=lambda manifest,require_snapshot_digest=False: dict(manifest)
            runtime._receipt_manifest=lambda stable_id: dict(baseline) if stable_id == WHISPER_ID else None
            token={"deployment_revision":1,"route_generation":1,"candidate_hash":"x","tier":"provisional_silver","cohort_id":"c","epoch":0,"session_hash":"s"}
            runtime.deployment.route=lambda session: {**silver,"_deployment_token":token}
            runtime.deployment.validate_route_token=lambda value: True
            calls=[]; runtime.deployment.enqueue_comparator_intent=lambda **kwargs: calls.append(("enqueue",kwargs)) or "inserted"
            class Backend:
                def transcribe(self, manifest, *_args, **_kwargs):
                    if manifest["stable_id"] == "silver":
                        runtime.adaptive._mutate(lambda state: state["champions"].update({"current":human}))
                        return "silver-stale",{"repo":"silver"}
                    return "human-current",{"repo":"human"}
            runtime.backends=Backend(); prepared=prepare_canonical(np.zeros(8,np.float32)); path=runtime.history.audio_path("promotion"); write_canonical_wav(path,prepared.pcm)
            text,meta=runtime.live_transcribe(np.zeros(8,np.float32),canonical_path=path,canonical_identity=prepared.identity)
            self.assertEqual(text,"human-current"); self.assertTrue(meta["human_authority_changed"]); self.assertEqual(calls,[])

    def test_human_promotion_during_comparator_enqueue_discards_pending_silver(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            runtime=AdaptiveRuntime(directory); baseline=candidate_manifest(WHISPER_ID,revision="base")
            human={**candidate_manifest(WHISPER_ID,revision="human"),"stable_id":"human"}; silver={**candidate_manifest(WHISPER_ID,revision="silver"),"stable_id":"silver"}
            runtime.adaptive.ensure_baseline(baseline)
            @contextmanager
            def live_request(): yield
            runtime.scheduler.live_request=live_request
            runtime._validated_persisted_manifest=lambda value,require_snapshot_digest=False: dict(value)
            runtime._receipt_manifest=lambda _ident: dict(baseline)
            token={"deployment_revision":1,"route_generation":1,"candidate_hash":"x","tier":"provisional_silver","cohort_id":"c","epoch":0,"session_hash":"s"}
            runtime.deployment.route=lambda _session:{**silver,"_deployment_token":dict(token)}
            runtime.deployment.validate_route_token=lambda _token:True
            def enqueue(**_kwargs):
                runtime.adaptive._mutate(lambda state:state["champions"].update({"current":dict(human)})); return "inserted"
            runtime.deployment.enqueue_comparator_intent=enqueue
            runtime.silver.discard_comparator_intent=lambda **_kwargs: True
            class Backend:
                def transcribe(_self,manifest,*_args,**_kwargs):
                    return ("silver-stale" if manifest["stable_id"] == "silver" else "human-current"),{}
            runtime.backends=Backend(); prepared=prepare_canonical(np.zeros(800,np.float32)); path=runtime.history.audio_path("enqueue-race"); write_canonical_wav(path,prepared.pcm)
            text,metadata=runtime.live_transcribe(prepared,canonical_path=path,canonical_identity=prepared.identity,capture_id="enqueue-race")
            self.assertEqual(text,"human-current"); self.assertTrue(metadata["fallback"]); self.assertTrue(metadata["human_authority_changed"])
            comparator_dir=runtime.history.silver_evidence_dir/"comparators"
            self.assertFalse(comparator_dir.exists() and any(comparator_dir.iterdir()))
            with runtime.silver._connect() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM runtime_pairs").fetchone()[0],0)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM comparator_intents").fetchone()[0],0)

    def test_enqueue_failure_fallback_rechecks_human_receipt_and_audio_authority(self):
        for mutation in ("human","receipt","audio"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
                runtime=AdaptiveRuntime(directory); baseline=candidate_manifest(WHISPER_ID,revision="base")
                human={**candidate_manifest(WHISPER_ID,revision="human"),"stable_id":"human"}
                silver={**candidate_manifest(WHISPER_ID,revision="silver"),"stable_id":"silver"}
                runtime.adaptive.ensure_baseline(baseline); state={"rotated":False}
                @contextmanager
                def live_request(): yield
                runtime.scheduler.live_request=live_request
                runtime._receipt_manifest=lambda _ident: dict(baseline)
                def validate(value,require_snapshot_digest=False):
                    return None if state["rotated"] and value.get("stable_id") == "silver" else dict(value)
                runtime._validated_persisted_manifest=validate
                token={"deployment_revision":1,"route_generation":1,"candidate_hash":"x","tier":"provisional_silver","cohort_id":"c","epoch":0,"session_hash":"s"}
                runtime.deployment.route=lambda _session:{**silver,"_deployment_token":dict(token)}
                runtime.deployment.validate_route_token=lambda _token:True
                prepared=prepare_canonical(np.zeros(800,np.float32)); path=runtime.history.audio_path(f"enqueue-{mutation}"); write_canonical_wav(path,prepared.pcm)
                def enqueue(**_kwargs):
                    if mutation == "human": runtime.adaptive._mutate(lambda row:row["champions"].update({"current":dict(human)}))
                    elif mutation == "receipt": state["rotated"]=True
                    else: write_canonical_wav(path,prepare_canonical(np.ones(800,np.float32)*.5).pcm)
                    raise RuntimeError("enqueue lost acknowledgement")
                runtime.deployment.enqueue_comparator_intent=enqueue
                class Backend:
                    def transcribe(_self,manifest,*_args,**_kwargs):
                        if manifest["stable_id"] == "silver": return "stale silver",{}
                        return ("human current" if manifest["stable_id"] == "human" else "baseline current"),{}
                runtime.backends=Backend()
                if mutation == "audio":
                    with self.assertRaisesRegex(RuntimeError,"live fallback authority changed"):
                        runtime.live_transcribe(prepared,canonical_path=path,canonical_identity=prepared.identity,capture_id=f"enqueue-{mutation}")
                else:
                    text,metadata=runtime.live_transcribe(prepared,canonical_path=path,canonical_identity=prepared.identity,capture_id=f"enqueue-{mutation}")
                    self.assertEqual(text,"human current" if mutation == "human" else "baseline current")
                    self.assertTrue(metadata["fallback"]); self.assertTrue(metadata["comparator_enqueue_failed"])
                comparator=runtime.history.silver_evidence_dir/"comparators"
                self.assertFalse(comparator.exists() and any(comparator.iterdir()))
                with runtime.silver._connect() as db:
                    self.assertEqual(db.execute("SELECT COUNT(*) FROM comparator_intents").fetchone()[0],0)

    def test_enqueue_exception_after_durable_insert_discards_undelivered_candidate_authority(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            runtime=AdaptiveRuntime(directory); baseline=candidate_manifest(WHISPER_ID,revision="base")
            silver={**candidate_manifest(WHISPER_ID,revision="silver"),"stable_id":"silver"}; runtime.adaptive.ensure_baseline(baseline)
            @contextmanager
            def live_request(): yield
            runtime.scheduler.live_request=live_request; runtime._receipt_manifest=lambda _ident:dict(baseline)
            runtime._validated_persisted_manifest=lambda value,require_snapshot_digest=False:dict(value)
            session=runtime.silver.active_session_id(); epoch=runtime.silver.epoch(); cohort="durable-enqueue"
            with runtime.silver._connect() as db:
                db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,"h","evaluated_passed","m","c","{}",epoch,0))
                db.execute("INSERT INTO route_assignments VALUES(?,?,?,?)",(1,hashlib.sha256(session.encode()).hexdigest(),1,0))
            token={"deployment_revision":1,"route_generation":1,"candidate_hash":"x","tier":"provisional_silver","cohort_id":cohort,"epoch":epoch,"session_hash":hashlib.sha256(session.encode()).hexdigest()}
            runtime.deployment.route=lambda _session:{**silver,"_deployment_token":dict(token)}; runtime.deployment.validate_route_token=lambda _token:True
            published={}
            def enqueue(**kwargs):
                published.update(kwargs)
                self.assertEqual(runtime.silver.enqueue_comparator_intent(deployment_revision=1,capture_id=kwargs["capture_id"],session_id=kwargs["session_id"],cohort_id=cohort,epoch=epoch,route_generation=1,candidate_arm="silver",candidate=kwargs["candidate"],audio_sha256=kwargs["audio_sha256"],spool_name=kwargs["spool_name"]),"inserted")
                raise RuntimeError("ack lost after durable insert")
            runtime.deployment.enqueue_comparator_intent=enqueue
            original_delete=runtime._delete_comparator_spool; replay=[]
            def paused_delete(value,digest):
                # Exact interleaving: the row is already discarded, but its
                # bytes have not yet been unlinked.  A concurrent retry must
                # not claim candidate replay authority from that row.
                replay.append(runtime.silver.enqueue_comparator_intent(deployment_revision=1,capture_id=published["capture_id"],session_id=published["session_id"],cohort_id=cohort,epoch=epoch,route_generation=1,candidate_arm="silver",candidate=published["candidate"],audio_sha256=published["audio_sha256"],spool_name=published["spool_name"]))
                original_delete(value,digest)
            runtime._delete_comparator_spool=paused_delete
            class Backend:
                def transcribe(_self,manifest,*_args,**_kwargs): return ("silver" if manifest["stable_id"] == "silver" else "baseline"),{}
            runtime.backends=Backend(); prepared=prepare_canonical(np.zeros(800,np.float32)); path=runtime.history.audio_path("durable-enqueue"); write_canonical_wav(path,prepared.pcm)
            text,metadata=runtime.live_transcribe(prepared,canonical_path=path,canonical_identity=prepared.identity,capture_id="durable-enqueue")
            self.assertEqual(text,"baseline"); self.assertTrue(metadata["comparator_enqueue_failed"])
            with runtime.silver._connect() as db:
                status,spool=db.execute("SELECT status,spool_name FROM comparator_intents").fetchone()
            # This invocation returned baseline after the lost acknowledgement,
            # so the candidate must never later become rollout authority.
            self.assertEqual(status,"discarded"); self.assertFalse((runtime.history.silver_evidence_dir/"comparators"/spool).exists())
            self.assertEqual(replay,[False])
            worker=AdaptiveWorker(directory,history=HistoryStore(directory),silver=runtime.silver,teachers=object())
            self.assertFalse(worker.process_comparator_one())
            with runtime.silver._connect() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM runtime_pairs").fetchone()[0],0)
            self.assertEqual(runtime.silver.runtime_metrics(1)["captures"],0)

    def test_duplicate_live_capture_keeps_first_pending_comparator_spool(self):
        """A later timing mismatch must not strand the first durable outbox row."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            runtime=AdaptiveRuntime(directory); baseline=candidate_manifest(WHISPER_ID,revision="base")
            silver={**candidate_manifest(WHISPER_ID,revision="silver"),"stable_id":"silver"}; runtime.adaptive.ensure_baseline(baseline)
            @contextmanager
            def live_request(): yield
            runtime.scheduler.live_request=live_request
            runtime._receipt_manifest=lambda _ident:dict(baseline)
            runtime._validated_persisted_manifest=lambda value,require_snapshot_digest=False:dict(value)
            session=runtime.silver.active_session_id(); epoch=runtime.silver.epoch(); cohort="duplicate-capture"
            with runtime.silver._connect() as db:
                db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,"h","evaluated_passed","m","c","{}",epoch,0))
                db.execute("INSERT INTO route_assignments VALUES(?,?,?,?)",(1,hashlib.sha256(session.encode()).hexdigest(),1,0))
            token={"deployment_revision":1,"route_generation":1,"candidate_hash":"x","tier":"provisional_silver","cohort_id":cohort,"epoch":epoch,"session_hash":hashlib.sha256(session.encode()).hexdigest()}
            runtime.deployment.route=lambda _session:{**silver,"_deployment_token":dict(token)}
            runtime.deployment.validate_route_token=lambda _token:True
            enqueues=[]
            def enqueue(**kwargs):
                # The actual store sees timing as exact authority.  Model a
                # naturally different second inference latency deterministically.
                candidate=dict(kwargs["candidate"])
                if enqueues: candidate["latency"] += 1.0
                result=runtime.silver.enqueue_comparator_intent(deployment_revision=1,capture_id=kwargs["capture_id"],
                    session_id=kwargs["session_id"],cohort_id=cohort,epoch=epoch,route_generation=1,
                    candidate_arm="silver",candidate=candidate,audio_sha256=kwargs["audio_sha256"],spool_name=kwargs["spool_name"])
                enqueues.append(result); return result
            runtime.deployment.enqueue_comparator_intent=enqueue
            class Backend:
                def transcribe(_self,manifest,*_args,**_kwargs):
                    return ("candidate" if manifest["stable_id"] == "silver" else "baseline"),{}
            runtime.backends=Backend(); prepared=prepare_canonical(np.zeros(800,np.float32)); path=runtime.history.audio_path("duplicate-capture"); write_canonical_wav(path,prepared.pcm)
            first,first_meta=runtime.live_transcribe(prepared,canonical_path=path,canonical_identity=prepared.identity,capture_id="caller-capture")
            self.assertTrue(runtime.acknowledge_comparator_publication(first_meta))
            second,second_meta=runtime.live_transcribe(prepared,canonical_path=path,canonical_identity=prepared.identity,capture_id="caller-capture")
            self.assertEqual((first,second),("candidate","baseline")); self.assertTrue(first_meta["comparator_pending"]); self.assertTrue(second_meta["comparator_enqueue_failed"])
            self.assertEqual(enqueues,["inserted",False])
            with runtime.silver._connect() as db:
                revision,capture,status,spool=db.execute("SELECT deployment_revision,capture_id,status,spool_name FROM comparator_intents").fetchone()
            self.assertEqual((revision,status),(1,"pending")); spool_path=runtime.history.silver_evidence_dir/"comparators"/spool
            self.assertTrue(spool_path.exists())
            # A reconstructed worker can later complete the original exact row.
            controller=DeploymentController(directory,silver=runtime.silver)
            controller._write({"schema":1,"revision":1,"tier":"provisional_silver","current":{"stable_id":"silver"},
                "lkg":None,"cohort_id":cohort,"calibration":None,"audit":[],"route_generation":1,
                "canary":{"observation_revision":1,"epoch":epoch}})
            worker=AdaptiveWorker(directory,history=HistoryStore(directory),silver=runtime.silver,teachers=object())
            worker._receipt_manifest=lambda _ident:{"stable_id":WHISPER_ID,"snapshot_digest":"fixture"}
            worker.backends=type("B",(),{"transcribe":lambda _self,*_args,**_kwargs:("baseline",{})})()
            @contextmanager
            def evaluator_lease(): yield True
            worker.scheduler.evaluator_lease=evaluator_lease
            self.assertTrue(worker.process_comparator_one())
            self.assertFalse(spool_path.exists())
            with runtime.silver._connect() as db:
                self.assertEqual(db.execute("SELECT status FROM comparator_intents WHERE deployment_revision=? AND capture_id=?",(revision,capture)).fetchone()[0],"complete")
                self.assertEqual(db.execute("SELECT COUNT(*) FROM runtime_pairs").fetchone()[0],1)

    def test_human_replacement_during_human_inference_discards_old_output(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime=AdaptiveRuntime(directory); baseline=candidate_manifest(WHISPER_ID,revision="base")
            old={**candidate_manifest(WHISPER_ID,revision="old"),"stable_id":"old-human"}; new={**candidate_manifest(WHISPER_ID,revision="new"),"stable_id":"new-human"}
            runtime.adaptive.ensure_baseline(baseline); runtime.adaptive._mutate(lambda state: state["champions"].update({"current":old}))
            @contextmanager
            def live_request(): yield
            runtime.scheduler.live_request=live_request; runtime._validated_persisted_manifest=lambda manifest,require_snapshot_digest=False: dict(manifest); runtime._receipt_manifest=lambda stable_id: dict(baseline)
            runtime.deployment.route=lambda _session: self.fail("silver route used"); runtime.deployment.begin_runtime_pair=lambda **_: self.fail("pair used")
            class Backend:
                def transcribe(self, manifest, *_args, **_kwargs):
                    if manifest["stable_id"] == "old-human":
                        runtime.adaptive._mutate(lambda state: state["champions"].update({"current":new})); return "old-stale",{}
                    return "new-current",{}
            prepared=prepare_canonical(np.zeros(800,np.float32)); path=runtime.history.audio_path("exception-promotion"); write_canonical_wav(path,prepared.pcm)
            runtime.backends=Backend(); text,meta=runtime.live_transcribe(prepared,canonical_path=path,canonical_identity=prepared.identity)
            self.assertEqual(text,"new-current"); self.assertTrue(meta["human_authority_changed"])

    def test_human_promotion_during_exception_fallback_discards_stale_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime=AdaptiveRuntime(directory); baseline=candidate_manifest(WHISPER_ID,revision="base"); human={**candidate_manifest(WHISPER_ID,revision="human"),"stable_id":"human"}; silver={**candidate_manifest(WHISPER_ID,revision="silver"),"stable_id":"silver"}
            runtime.adaptive.ensure_baseline(baseline)
            @contextmanager
            def live_request(): yield
            runtime.scheduler.live_request=live_request; runtime._validated_persisted_manifest=lambda m,require_snapshot_digest=False: dict(m); runtime._receipt_manifest=lambda stable_id: dict(baseline)
            token={"deployment_revision":1,"route_generation":1,"candidate_hash":"x","tier":"provisional_silver","cohort_id":"c","epoch":0,"session_hash":"s"}; runtime.deployment.route=lambda _: {**silver,"_deployment_token":token}; runtime.deployment.validate_route_token=lambda _: True; runtime.deployment.begin_runtime_pair=lambda **_: "stale"
            class Backend:
                def transcribe(self, manifest, *_args, **_kwargs):
                    if manifest["stable_id"] == "silver": raise RuntimeError("backend")
                    if manifest["stable_id"] == WHISPER_ID:
                        runtime.adaptive._mutate(lambda state: state["champions"].update({"current":human})); return "baseline-stale",{}
                    return "human-current",{}
            prepared=prepare_canonical(np.zeros(800,np.float32)); path=runtime.history.audio_path("exception-receipt"); write_canonical_wav(path,prepared.pcm)
            runtime.backends=Backend(); text,meta=runtime.live_transcribe(prepared,canonical_path=path,canonical_identity=prepared.identity)
            self.assertEqual(text,"human-current"); self.assertTrue(meta["human_authority_changed"])

    def test_receipt_rotation_during_exception_fallback_discards_stale_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime=AdaptiveRuntime(directory); baseline=candidate_manifest(WHISPER_ID,revision="base")
            rotated={**candidate_manifest(WHISPER_ID,revision="rotated"),"snapshot_digest":"rotated-digest"}
            silver={**candidate_manifest(WHISPER_ID,revision="silver"),"stable_id":"silver"}; state={"rotated":False}
            runtime.adaptive.ensure_baseline(baseline)
            @contextmanager
            def live_request(): yield
            runtime.scheduler.live_request=live_request
            def validate(manifest,require_snapshot_digest=False):
                if manifest.get("stable_id") == "silver": return dict(manifest)
                if manifest.get("revision") == "base": return None if state["rotated"] else dict(manifest)
                return dict(manifest) if manifest.get("revision") == "rotated" else None
            runtime._validated_persisted_manifest=validate
            runtime._receipt_manifest=lambda _stable_id: dict(rotated if state["rotated"] else baseline)
            runtime._validate_human_authority=lambda _token: True
            runtime.adaptive.record_runtime_failure=lambda *_args,**_kwargs: {"streak":1}
            token={"deployment_revision":1,"route_generation":1,"candidate_hash":"x","tier":"provisional_silver","cohort_id":"c","epoch":0,"session_hash":"s"}
            runtime.deployment.route=lambda _session: {**silver,"_deployment_token":token}; runtime.deployment.validate_route_token=lambda _token: True
            runtime.deployment.begin_runtime_pair=lambda **_kwargs: "stale"
            class Backend:
                def transcribe(self,manifest,*_args,**_kwargs):
                    if manifest["stable_id"] == "silver": raise RuntimeError("candidate failed")
                    if manifest["revision"] == "base":
                        state["rotated"]=True; return "baseline-stale",{}
                    return "baseline-current",{}
            runtime.backends=Backend(); text,meta=runtime.live_transcribe(np.zeros(8,np.float32))
            self.assertEqual(text,"baseline-current")
            self.assertTrue(meta["human_authority_changed"])
            self.assertTrue(meta["fallback"])

    def test_audio_drift_during_exception_fallback_fails_closed_without_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime=AdaptiveRuntime(directory); baseline=candidate_manifest(WHISPER_ID,revision="base"); silver={**candidate_manifest(WHISPER_ID,revision="silver"),"stable_id":"silver"}
            runtime.adaptive.ensure_baseline(baseline)
            @contextmanager
            def live_request(): yield
            runtime.scheduler.live_request=live_request; runtime._validated_persisted_manifest=lambda m,require_snapshot_digest=False:dict(m); runtime._receipt_manifest=lambda _ident:dict(baseline)
            token={"deployment_revision":1,"route_generation":1,"candidate_hash":"x","tier":"provisional_silver","cohort_id":"c","epoch":0,"session_hash":"s"}
            runtime.deployment.route=lambda _session:{**silver,"_deployment_token":dict(token)}; runtime.deployment.validate_route_token=lambda _token:True
            runtime.deployment.begin_runtime_pair=lambda **_kwargs:self.fail("stale audio must not begin a pair")
            prepared=prepare_canonical(np.zeros(800,np.float32)); path=runtime.history.audio_path("exception-audio"); write_canonical_wav(path,prepared.pcm)
            class Backend:
                def transcribe(_self,manifest,*_args,**_kwargs):
                    if manifest["stable_id"] == "silver":
                        write_canonical_wav(path,prepare_canonical(np.ones(800,np.float32)*.5).pcm)
                        raise RuntimeError("candidate failed")
                    return "baseline stale",{}
            runtime.backends=Backend()
            with self.assertRaisesRegex(RuntimeError,"live fallback authority changed"):
                runtime.live_transcribe(prepared,canonical_path=path,canonical_identity=prepared.identity)

    def test_live_transcribe_preserves_supplied_capture_id(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            runtime=AdaptiveRuntime(directory); baseline=candidate_manifest(WHISPER_ID,revision="base"); silver={**candidate_manifest(WHISPER_ID,revision="silver"),"stable_id":"silver"}; runtime.adaptive.ensure_baseline(baseline)
            @contextmanager
            def live_request(): yield
            runtime.scheduler.live_request=live_request; runtime._validated_persisted_manifest=lambda m,require_snapshot_digest=False: dict(m); runtime._receipt_manifest=lambda _: dict(baseline)
            token={"deployment_revision":1,"route_generation":1,"candidate_hash":"x","tier":"provisional_silver","cohort_id":"c","epoch":0,"session_hash":"s"}; runtime.deployment.route=lambda _: {**silver,"_deployment_token":token}; runtime.deployment.validate_route_token=lambda _: True
            seen=[]; runtime.deployment.enqueue_comparator_intent=lambda **kw: seen.append(kw["capture_id"]) or "inserted"
            runtime.silver.acknowledge_comparator_intent=lambda **_kwargs: True
            class Backend:
                def transcribe(self, manifest, *_args, **_kwargs): return ("silver",{}) if manifest["stable_id"] == "silver" else ("base",{})
            runtime.backends=Backend(); prepared=prepare_canonical(np.zeros(8,np.float32)); path=runtime.history.audio_path("capture-id"); write_canonical_wav(path,prepared.pcm)
            runtime.live_transcribe(np.zeros(8,np.float32),canonical_path=path,canonical_identity=prepared.identity,capture_id="logical-request"); runtime.live_transcribe(np.zeros(8,np.float32),canonical_path=path,canonical_identity=prepared.identity,capture_id="logical-request"); runtime.live_transcribe(np.zeros(8,np.float32),canonical_path=path,canonical_identity=prepared.identity,capture_id="other-request")
            self.assertEqual(len(seen),3); self.assertEqual(seen[0],seen[1]); self.assertNotEqual(seen[1],seen[2])
    def test_baseline_and_three_deployed_failures_roll_back_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AdaptiveLearning(Path(directory))
            baseline = candidate_manifest(WHISPER_ID, revision="base")
            store.ensure_baseline(baseline)
            challenger = {**candidate_manifest(WHISPER_ID, revision="new"), "stable_id": "promoted"}
            store._mutate(lambda state: state["champions"].update({"current": challenger, "prior": []}))
            self.assertEqual(store.record_runtime_failure("promoted", "TimeoutError")["streak"], 1)
            store.record_runtime_failure("promoted", "TimeoutError")
            outcome = store.record_runtime_failure("promoted", "TimeoutError")
            self.assertTrue(outcome["rolled_back"])
            self.assertEqual(store.champion_manifest()["stable_id"], WHISPER_ID)

    def test_success_resets_only_its_failure_streak(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AdaptiveLearning(Path(directory))
            baseline = candidate_manifest(WHISPER_ID, revision="base")
            store.ensure_baseline(baseline)
            store._mutate(lambda state: state["champions"].update({"current": {**baseline, "stable_id": "promoted"}}))
            store.record_runtime_failure("promoted", "x")
            store.record_runtime_success("promoted")
            self.assertNotIn("promoted", store.status()["runtime_failures"])

    def test_legacy_canonical_row_is_not_adaptive_review_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            history = HistoryStore(directory)
            row = history.append("old", prepare_canonical(np.zeros(800, np.float32)), .5, "old")
            # Simulate a normalized legacy row with no persisted identity.
            raw = history._entries[0]; raw["schema_version"] = 1; raw["audio"]["inference"]["sha256"] = None
            history._save_locked()
            runtime = AdaptiveRuntime(directory)
            self.assertEqual(runtime.register_retained_history(), 1)
            with runtime.adaptive._locked():
                self.assertFalse(runtime.adaptive._read()["captures"][row["id"]]["adaptive"])
            self.assertEqual(runtime.review_list(), [])
            self.assertFalse(runtime.review_eligible_entry(history.get(row["id"])))
            with self.assertRaisesRegex(ValueError, "not eligible"):
                runtime.review(row["id"], "corrected", reference="gold", expected_revision=row["revision"])
            self.assertNotIn("adaptive_review", history.get(row["id"]))
            runtime._review_recovery_gate(retry_pending=True)

    def test_legacy_rows_keep_ordinary_correction_and_manual_enrollment(self):
        with tempfile.TemporaryDirectory() as directory:
            history, learning = HistoryStore(directory), LearningStore(directory); coordinator = LearningCoordinator(history, learning)
            row = coordinator.append_live("old", prepare_canonical(np.zeros(800, np.float32)), .5, "m", language="en", adaptive=False)
            runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator)
            import sotto
            corrected = sotto.nonadaptive_correct(runtime.adaptive, coordinator, row["id"], "human", expected_revision=row["revision"])
            self.assertEqual(corrected["correction"]["text"], "human")
            self.assertNotIn("adaptive_review", history.get(row["id"]))
            self.assertIsNotNone(coordinator.enroll(row["id"]))

    def test_coordinator_and_recovery_reject_ineligible_adaptive_outbox_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            history, learning = HistoryStore(directory), LearningStore(directory); coordinator = LearningCoordinator(history, learning)
            row = coordinator.append_live("old", prepare_canonical(np.zeros(800, np.float32)), .5, "m", language="en", adaptive=False)
            with self.assertRaisesRegex(ValueError, "not eligible"):
                coordinator.commit_adaptive_review(row["id"], "corrected", reference="gold", expected_revision=row["revision"])
            self.assertNotIn("adaptive_review", history.get(row["id"]))
            history._entries[0]["adaptive_review"] = {"revision": row["revision"], "outcome": "corrected", "ts": 0}
            history._save_locked()
            runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator)
            with self.assertRaisesRegex(RuntimeError, "not eligible"):
                runtime._review_recovery_gate(retry_pending=True)
            self.assertFalse(learning.active_for_history(row["id"]))

    def test_clear_revokes_pruned_adaptive_capture_ids_before_clear(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            runtime = AdaptiveRuntime(directory)
            runtime.adaptive.register_capture({"history_id": "pruned", "captured_ts": 1, "duration": 1,
                                               "audio_digest": "x", "adaptive": True, "language": "en"})
            runtime.clear()
            with runtime.adaptive._locked():
                # Clear intentionally erases personal identifiers rather than
                # retaining a revoked tombstone for a pruned history row.
                self.assertNotIn("pruned",runtime.adaptive._read()["captures"])

    def test_adaptive_review_enrolls_once_then_retry_once(self):
        with tempfile.TemporaryDirectory() as directory:
            history, learning = HistoryStore(directory), LearningStore(directory)
            coordinator = LearningCoordinator(history, learning)
            row = coordinator.append_live("hyp", prepare_canonical(np.zeros(800, np.float32)), .5, "m", language="en", adaptive=True,
                                          candidate={"stable_id": "x", "backend": "whisper", "repo": "r", "revision": "z"})
            runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator)
            runtime.register_live(row)
            calls = []
            original = coordinator.enroll
            def fail_once(ident):
                calls.append(ident)
                if len(calls) == 1: raise OSError("full")
                return original(ident)
            coordinator.enroll = fail_once  # type: ignore[method-assign]
            runtime.review(row["id"], "corrected", reference="gold")
            self.assertEqual(calls, [row["id"]])
            self.assertEqual(len(learning.pending_gold()), 1)
            runtime.retry_pending_gold()
            self.assertEqual(calls, [row["id"], row["id"]])
            self.assertEqual(learning.pending_gold(), [])

    def test_receipt_round_trip_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = candidate_manifest(WHISPER_ID, revision="rev")
            with strict_receipts(root,[manifest]) as (receipts,snapshots):
                snapshot=snapshots[WHISPER_ID]; receipt=receipts.load(WHISPER_ID)
                runtime = AdaptiveRuntime(root)
                self.assertIsNotNone(runtime._receipt_manifest(WHISPER_ID))
                for mutate in (lambda: ((snapshot / "model").unlink(), snapshot.rmdir()),
                               lambda: (snapshot / "model").write_text('{"changed":true}'),
                               lambda: receipt.__setitem__("package_versions", {"bad": "1"}),
                               lambda: receipt.__setitem__("evaluator_hash", "bad")):
                    # Restore a valid schema-3 receipt before each independent tamper.
                    if not snapshot.exists(): snapshot.mkdir(); (snapshot / "model").write_text("rev")
                    receipt.update({"snapshot_metadata_digest": _safe_snapshot_digest(snapshot), "package_versions": _package_versions(), "evaluator_hash": evaluator_hash()})
                    mutate(); receipts.save(WHISPER_ID, receipt)
                    self.assertIsNone(runtime._receipt_manifest(WHISPER_ID))

    def test_runtime_evaluate_33_rows_uses_float_latency_and_one_incumbent_pass(self):
        class Scheduler:
            owner_id = "fake"
            def reconcile(self): return []
            @contextmanager
            def evaluator_lease(self): yield True
        class Backend:
            def __init__(self): self.calls = []
            def transcribe(self, manifest, samples):
                self.calls.append((manifest["stable_id"], manifest.get("snapshot_path"))); return "word " * 10, {}
        with tempfile.TemporaryDirectory() as directory:
            history, learning = HistoryStore(directory), LearningStore(directory); coordinator = LearningCoordinator(history, learning)
            backend = Backend(); runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator,
                                                           scheduler=Scheduler(), backends=backend)
            from speech_backends import evaluator_hash
            champion = candidate_manifest(WHISPER_ID, revision="base")
            challenger = {**candidate_manifest(WHISPER_ID, revision="chall"), "stable_id": "challenger"}
            with strict_receipts(Path(directory),[champion,challenger]) as (_receipts,snapshots):
                canonical_paths={stable:str(path) for stable,path in snapshots.items()}
                generation = runtime.adaptive.create_generation(champion=champion, evaluator_hash=evaluator_hash(), freeze_ts=0)
                runtime.adaptive.freeze_generation(generation, [challenger]); runtime.adaptive.open_pool(generation)
                for index in range(33):
                    row = coordinator.append_live("hyp", prepare_canonical(np.zeros(16000, np.float32)), 1, "m", language="en", adaptive=True,
                                                  candidate={"stable_id": "x", "backend": "whisper", "repo": "r", "revision": "q"}, ts=index + 1)
                    runtime.register_live(row)
                    runtime.adaptive.resolve_review(row["id"], "no_speech" if index < 3 else "corrected",
                                                    reference="" if index < 3 else "word " * 10)
                self.assertTrue(runtime.adaptive.close_pool(generation))
                result = runtime.evaluate()
                self.assertNotIn(result["outcome"], {"technical_abort_champion", "receipt_or_evaluator_invalid"})
                self.assertEqual(sum(name == WHISPER_ID for name, _ in backend.calls), 33)
                self.assertEqual(sum(name == "challenger" for name, _ in backend.calls), 33)
                self.assertTrue(all(path == canonical_paths[name] for name, path in backend.calls))
                with runtime.adaptive._locked():
                    attempts = runtime.adaptive._read()["generations"][generation]["champion_attempts"]
                    self.assertTrue(all(isinstance(rows[0]["latency"], float) for rows in attempts.values()))

    def test_runtime_evaluate_rejects_replaced_reviewed_wav_before_backend(self):
        class Scheduler:
            owner_id = "fake"
            def reconcile(self): return []
            @contextmanager
            def evaluator_lease(self): yield True
        class Backend:
            def __init__(self): self.calls = 0
            def transcribe(self, manifest, samples): self.calls += 1; return "word " * 10, {}
        with tempfile.TemporaryDirectory() as directory:
            history, learning = HistoryStore(directory), LearningStore(directory); coordinator = LearningCoordinator(history, learning)
            backend = Backend(); runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator, scheduler=Scheduler(), backends=backend)
            champion = candidate_manifest(WHISPER_ID, revision="base"); challenger = {**candidate_manifest(WHISPER_ID, revision="chall"), "stable_id": "challenger"}
            with strict_receipts(Path(directory),[champion,challenger]):
                generation = runtime.adaptive.create_generation(champion=champion, evaluator_hash=evaluator_hash(), freeze_ts=0)
                runtime.adaptive.freeze_generation(generation, [challenger]); runtime.adaptive.open_pool(generation)
                rows = []
                for index in range(33):
                    row = coordinator.append_live("hyp", prepare_canonical(np.zeros(16000, np.float32)), 1, "m", language="en", adaptive=True, candidate={"stable_id":"x","backend":"whisper","repo":"r","revision":"q"}, ts=index + 1); rows.append(row)
                    runtime.register_live(row); runtime.adaptive.resolve_review(row["id"], "no_speech" if index < 3 else "corrected", reference="" if index < 3 else "word " * 10)
                self.assertTrue(runtime.adaptive.close_pool(generation))
                # Same valid WAV format, deliberately different canonical bytes.
                from audio_codec import encode_canonical, write_canonical_wav
                pcm, _ = encode_canonical(np.ones(16000, np.float32) * .5)
                write_canonical_wav(history.audio_path(rows[0]["id"]), pcm)
                result = runtime.evaluate()
                self.assertEqual(backend.calls, 0)
                self.assertIn(result["outcome"], {"technical_abort_champion", "receipt_or_evaluator_invalid"})

    def test_runtime_evaluate_discards_post_inference_audio_or_receipt_drift(self):
        class Scheduler:
            owner_id='fake'
            def reconcile(_self): return []
            @contextmanager
            def evaluator_lease(_self): yield True
        for drift in ('audio','receipt'):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as directory:
                history,learning=HistoryStore(directory),LearningStore(directory); coordinator=LearningCoordinator(history,learning)
                champion=candidate_manifest(WHISPER_ID,revision='base'); challenger={**candidate_manifest(WHISPER_ID,revision='chall'),'stable_id':'challenger'}
                with strict_receipts(Path(directory),[champion,challenger]) as (_receipts,snapshots):
                    class Backend:
                        def __init__(_self): _self.calls=0
                        def transcribe(_self,manifest,_samples):
                            _self.calls+=1
                            if _self.calls == 1:
                                if drift == 'audio':
                                    pcm,_=encode_canonical(np.ones(16000,np.float32)*.5)
                                    write_canonical_wav(history.audio_path(first['id']),pcm)
                                else:
                                    (snapshots[manifest['stable_id']]/'model').write_text('rotated')
                            return 'attractive transcript that must not persist',{}
                    backend=Backend(); runtime=AdaptiveRuntime(directory,history=history,learning=learning,coordinator=coordinator,scheduler=Scheduler(),backends=backend)
                    generation=runtime.adaptive.create_generation(champion=champion,evaluator_hash=evaluator_hash(),freeze_ts=0)
                    runtime.adaptive.freeze_generation(generation,[challenger]); runtime.adaptive.open_pool(generation)
                    rows={}
                # Make a corrected capture chronological first, so the
                # injected post-inference mutation targets its exact attempt.
                    order=[3,*[value for value in range(33) if value != 3]]
                    for ordinal,index in enumerate(order):
                        row=coordinator.append_live('hyp',prepare_canonical(np.zeros(16000,np.float32)),1,'m',language='en',adaptive=True,
                            candidate={'stable_id':'x','backend':'whisper','repo':'r','revision':'q'},ts=ordinal+1)
                        rows[index]=row; runtime.register_live(row)
                        runtime.adaptive.resolve_review(row['id'],'no_speech' if index < 3 else 'corrected',
                            reference='' if index < 3 else 'human correction remains authoritative across this exact frozen evaluation cycle today')
                    first=rows[3]
                    self.assertTrue(runtime.adaptive.close_pool(generation))
                    result=runtime.evaluate()
                    self.assertIn(result['outcome'],{'technical_abort_champion','no_promotion'})
                    with runtime.adaptive._locked():
                        state=runtime.adaptive._read(); generation_state=state['generations'][generation]
                        attempts=generation_state['champion_attempts'][first['id']]
                        self.assertTrue(attempts and all(item['status']=='failed' and item.get('code')=='authority_changed' and 'text' not in item for item in attempts))
                        self.assertEqual(state['captures'][first['id']]['reference'],'human correction remains authoritative across this exact frozen evaluation cycle today')
                        self.assertIsNone(state['champions']['current'])
                    self.assertEqual(history.get(first['id'])['text'],'hyp')

    def test_reconcile_resumes_closed_or_evaluating_but_not_pending_open(self):
        with tempfile.TemporaryDirectory() as directory:
            for status in ("pool_closed", "evaluating", "pool_open"):
                subdir = Path(directory) / status; runtime = AdaptiveRuntime(subdir); from speech_backends import evaluator_hash
                champion = candidate_manifest(WHISPER_ID, revision="base"); ident = "g-" + status
                runtime.adaptive.create_generation(generation_id=ident, champion=champion, evaluator_hash=evaluator_hash())
                runtime.adaptive._mutate(lambda state, ident=ident, status=status: state["generations"][ident].update({"status": status}))
                result = runtime.reconcile()
                # Evaluation is worker-owned; reconcile never fire-and-forgets.
                if status != "pool_open": self.assertEqual(result["state"], "evaluating")

    def test_validated_deployment_skips_invalid_current_and_prior(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = AdaptiveRuntime(directory)
            base = candidate_manifest(WHISPER_ID, revision="base")
            bad = {**candidate_manifest(WHISPER_ID, revision="bad"), "stable_id": "bad"}
            good = {**candidate_manifest(WHISPER_ID, revision="good"), "stable_id": "good"}
            runtime.adaptive._mutate(lambda state: state["champions"].update({"baseline": base, "current": {**bad, "stable_id": "current"}, "prior": [bad, good]}))
            runtime._validated_persisted_manifest = lambda item,**_kwargs: item if item.get("stable_id") == "good" else None  # type: ignore[method-assign]
            self.assertEqual(runtime._validated_deployment()["stable_id"], "good")
            runtime.adaptive._mutate(lambda state: state["champions"].update({"current": None, "prior": [], "baseline": base}))
            runtime._validated_persisted_manifest = lambda item,**_kwargs: None  # type: ignore[method-assign]
            runtime._receipt_manifest = lambda ident: base if ident == WHISPER_ID else None  # type: ignore[method-assign]
            self.assertEqual(runtime._validated_deployment()["stable_id"], WHISPER_ID)
            runtime._receipt_manifest = lambda ident: None  # type: ignore[method-assign]
            with self.assertRaises(RuntimeError): runtime._validated_deployment()
            class Scheduler:
                @contextmanager
                def live_request(self): yield
            class Backend:
                def __init__(self): self.calls = []
                def transcribe(self, *args): self.calls.append(args); return "", {}
            backend = Backend(); runtime.scheduler = Scheduler(); runtime.backends = backend
            with self.assertRaises(RuntimeError): runtime.live_transcribe(np.zeros(8, np.float32))
            self.assertEqual(backend.calls, [])

    def test_runtime_validates_frozen_manifest_projection_and_exact_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); manifest = candidate_manifest(WHISPER_ID, revision="rev")
            with strict_receipts(root,[manifest]) as (_receipts,snapshots):
                runtime = AdaptiveRuntime(root); snap=snapshots[WHISPER_ID]
                frozen = {**manifest, "generation_id": "g", "promotion_dependency_digests": ["x"]}
                self.assertEqual(runtime._validated_persisted_manifest(frozen)["snapshot_path"], str(snap))
                (snap / "model").write_text("y")
                self.assertIsNone(runtime._validated_persisted_manifest(frozen))

    def test_frozen_glossary_terms_and_non_glossary_injection_fail_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); glossary = candidate_manifest(WHISPER_GLOSSARY_ID, revision="rev", glossary_terms=("one", "two")); plain = candidate_manifest(WHISPER_ID, revision="rev")
            with strict_receipts(root,[glossary,plain]):
                runtime = AdaptiveRuntime(root)
                self.assertIsNotNone(runtime._validated_persisted_manifest(glossary))
                self.assertIsNone(runtime._validated_persisted_manifest({**glossary, "glossary_terms": ("two", "one")}))
                self.assertIsNone(runtime._validated_persisted_manifest({**glossary, "glossary_terms": ("one", 7)}))
                self.assertIsNone(runtime._validated_persisted_manifest({**plain, "glossary_terms": ("injected",)}))

    def test_nonadaptive_correction_revokes_adaptive_tombstone_before_mutation(self):
        import sotto
        with tempfile.TemporaryDirectory() as directory:
            guard = AdaptiveLearning(directory)
            guard.register_capture({"history_id": "old", "captured_ts": 1, "duration": 1, "audio_digest": "d", "adaptive": True, "language": "en"})
            observed = []
            class Coordinator:
                def review_transaction(self, ident, revision, before, mutation): before(); return mutation()
                def correct(self, ident, text):
                    with guard._locked(): observed.append(dict(guard._read()["captures"][ident]))
                    return True
            sotto.nonadaptive_correct(guard, Coordinator(), "old", "replacement")
            self.assertTrue(observed[0]["revoked"])
            self.assertNotIn("reference", observed[0])

    def test_nonadaptive_delete_clear_and_learning_revoke_invalidate_persisted_adaptive_state(self):
        import sotto
        from speech_backends import evaluator_hash
        with tempfile.TemporaryDirectory() as directory:
            def seeded(ident: str):
                guard = AdaptiveLearning(directory)
                guard.register_capture({"history_id": ident, "captured_ts": 1, "duration": 1, "audio_digest": ident, "adaptive": True, "language": "en"})
                champion = candidate_manifest(WHISPER_ID, revision="base")
                generation = guard.create_generation(generation_id="g-" + ident, champion=champion, evaluator_hash=evaluator_hash(), freeze_ts=0)
                guard.freeze_generation(generation, [candidate_manifest(WHISPER_ID, revision="chall")])
                guard.open_pool(generation)
                return AdaptiveLearning(directory), generation
            class Coordinator:
                def __init__(self, guard, ident): self.guard, self.ident, self.seen = guard, ident, []
                def _check(self):
                    with self.guard._locked(): self.seen.append(self.guard._read()["captures"][self.ident].get("revoked"))
                def delete(self, ident): self._check(); return True
                def clear(self): self._check()
                def revoke_history(self, ident): self._check(); return 1
            guard, generation = seeded("delete"); coordinator = Coordinator(guard, "delete")
            sotto.nonadaptive_delete(guard, coordinator, "delete")
            self.assertEqual(coordinator.seen, [True]); self.assertEqual(guard.status()["generations"][generation]["status"], "invalidated")
            guard, generation = seeded("pruned"); coordinator = Coordinator(guard, "pruned")
            # There is deliberately no HistoryStore row for this capture.
            sotto.nonadaptive_clear(guard, coordinator)
            self.assertEqual(coordinator.seen, [True]); self.assertEqual(guard.status()["generations"][generation]["status"], "invalidated")
            guard, generation = seeded("learning"); coordinator = Coordinator(guard, "learning")
            sotto.nonadaptive_revoke_learning(guard, coordinator, "learning")
            self.assertEqual(coordinator.seen, [True]); self.assertEqual(guard.status()["generations"][generation]["status"], "invalidated")

    def test_vad_gated_adaptive_no_speech_registers_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            history, learning = HistoryStore(directory), LearningStore(directory); coordinator = LearningCoordinator(history, learning)
            runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator)
            champion = candidate_manifest(WHISPER_ID, revision="base")
            generation = runtime.adaptive.create_generation(champion=champion, evaluator_hash=evaluator_hash(), freeze_ts=0)
            runtime.adaptive.freeze_generation(generation, [candidate_manifest(WHISPER_ID, revision="chall")]); runtime.adaptive.open_pool(generation)
            row = coordinator.append_live("[no speech detected]", prepare_canonical(np.zeros(16000, np.float32)), 1, "m",
                                          language="en", adaptive=True, provenance="live_no_speech")
            runtime.register_live(row)
            restarted = AdaptiveRuntime(directory)
            self.assertEqual(restarted.register_retained_history(), 1)
            restarted.adaptive.resolve_review(row["id"], "no_speech")
            self.assertEqual(restarted.adaptive.pool_status(generation)["no_speech"], 1)

    def test_stale_review_actions_do_not_mutate_adaptive_or_history_state(self):
        with tempfile.TemporaryDirectory() as directory:
            history, learning = HistoryStore(directory), LearningStore(directory); coordinator = LearningCoordinator(history, learning)
            runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator)
            row = coordinator.append_live("hyp", prepare_canonical(np.zeros(800, np.float32)), .5, "m", language="en", adaptive=True, candidate={"stable_id":"x","backend":"w","repo":"r","revision":"q"})
            runtime.register_live(row); stale = row["revision"]
            snap = coordinator.snapshot_for_retry(row["id"]); coordinator.commit_retry(row["id"], "new", snap.expected_revision)
            before = history.get(row["id"])
            for outcome in ("corrected", "correct_as_is", "no_speech", "skip"):
                with runtime.adaptive._locked(): adaptive_before = runtime.adaptive._read()
                learning_before = (learning.active(), learning.pending_gold())
                with self.assertRaises(KeyError): runtime.review(row["id"], outcome, reference="gold", reason="private", expected_revision=stale)
                self.assertEqual(history.get(row["id"]), before)
                with runtime.adaptive._locked(): self.assertEqual(runtime.adaptive._read(), adaptive_before)
                self.assertEqual((learning.active(), learning.pending_gold()), learning_before)

    def test_gold_to_skip_rolls_back_development_and_cohort_dependencies_and_revokes_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            history, learning = HistoryStore(directory), LearningStore(directory); coordinator = LearningCoordinator(history, learning)
            runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator)
            row = coordinator.append_live("hyp", prepare_canonical(np.zeros(800, np.float32)), .5, "m", language="en", adaptive=True, candidate={"stable_id":"x","backend":"w","repo":"r","revision":"q"})
            runtime.register_live(row); runtime.review(row["id"], "corrected", reference="gold", expected_revision=row["revision"])
            self.assertTrue(learning.active_for_history(row["id"]))
            def install(state):
                digest = runtime.adaptive._sample_digest(state, row["id"])
                base = candidate_manifest(WHISPER_ID, revision="base")
                state["champions"].update({"baseline": base, "current": {**candidate_manifest(WHISPER_ID, revision="cur"), "stable_id":"current", "development_dependency_digests":[digest], "invalid":False}, "prior": []})
            runtime.adaptive._mutate(install)
            revision = history.get(row["id"])["revision"]
            runtime.review(row["id"], "skip", reason="private", expected_revision=revision)
            self.assertEqual(runtime.adaptive.status()["current_champion"], WHISPER_ID)
            self.assertFalse(learning.active_for_history(row["id"]))

    def test_gold_to_skip_rolls_back_cohort_deployment_and_annotates_source(self):
        with tempfile.TemporaryDirectory() as directory:
            history, learning = HistoryStore(directory), LearningStore(directory); coordinator = LearningCoordinator(history, learning)
            runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator)
            row = coordinator.append_live("hyp", prepare_canonical(np.zeros(800, np.float32)), .5, "m", language="en", adaptive=True, candidate={"stable_id":"x","backend":"w","repo":"r","revision":"q"})
            runtime.register_live(row); runtime.review(row["id"], "corrected", reference="gold", expected_revision=row["revision"])
            def install(state):
                digest = runtime.adaptive._sample_digest(state, row["id"]); base = candidate_manifest(WHISPER_ID, revision="base")
                state["generations"]["source"] = {"id":"source", "status":"promoted", "decision":{"outcome":"promoted"}, "pool":[], "candidates":{}}
                state["champions"].update({"baseline":base, "current":{**candidate_manifest(WHISPER_ID, revision="cur"), "stable_id":"current", "promotion_dependency_generation":"source", "promotion_dependency_digests":[digest], "invalid":False}, "prior":[]})
            runtime.adaptive._mutate(install)
            runtime.review(row["id"], "skip", reason="private", expected_revision=history.get(row["id"])["revision"])
            with runtime.adaptive._locked(): source = runtime.adaptive._read()["generations"]["source"]
            self.assertEqual(runtime.adaptive.status()["current_champion"], WHISPER_ID)
            self.assertTrue(source["deployment_invalidated"]); self.assertTrue(source["decision"]["deployment_invalidated"])

    def test_first_pending_pool_skip_is_normal_open_disposition_without_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            history, learning = HistoryStore(directory), LearningStore(directory); coordinator = LearningCoordinator(history, learning)
            runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator)
            champion = candidate_manifest(WHISPER_ID, revision="base")
            generation = runtime.adaptive.create_generation(champion=champion, evaluator_hash=evaluator_hash(), freeze_ts=0)
            runtime.adaptive.freeze_generation(generation, [candidate_manifest(WHISPER_ID, revision="chall")]); runtime.adaptive.open_pool(generation)
            row = coordinator.append_live("hyp", prepare_canonical(np.zeros(800, np.float32)), .5, "m", language="en", adaptive=True, candidate={"stable_id":"x","backend":"w","repo":"r","revision":"q"})
            runtime.register_live(row); runtime.review(row["id"], "skip", reason="private", expected_revision=row["revision"])
            self.assertEqual(runtime.adaptive.pool_status(generation)["status"], "pool_open")
            self.assertEqual(runtime.adaptive.pool_status(generation)["skips"], 1)
            self.assertFalse(learning.active_for_history(row["id"]))

    def test_skip_replay_is_cas_rejected_and_refreshed_repeat_is_a_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            history, learning = HistoryStore(directory), LearningStore(directory); coordinator = LearningCoordinator(history, learning)
            runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator)
            champion = candidate_manifest(WHISPER_ID, revision="base")
            generation = runtime.adaptive.create_generation(champion=champion, evaluator_hash=evaluator_hash(), freeze_ts=0)
            runtime.adaptive.freeze_generation(generation, [candidate_manifest(WHISPER_ID, revision="chall")]); runtime.adaptive.open_pool(generation)
            row = coordinator.append_live("hyp", prepare_canonical(np.zeros(800, np.float32)), .5, "m", language="en", adaptive=True, candidate={"stable_id":"x","backend":"w","repo":"r","revision":"q"})
            runtime.register_live(row); visible_revision = row["revision"]
            runtime.review(row["id"], "skip", reason="private", expected_revision=visible_revision)
            first_history = history.get(row["id"])
            with runtime.adaptive._locked(): first_adaptive = runtime.adaptive._read()
            first_learning = (learning.active(), learning.pending_gold())
            self.assertEqual(runtime.adaptive.pool_status(generation)["status"], "pool_open")
            self.assertEqual(runtime.adaptive.pool_status(generation)["skips"], 1)
            with self.assertRaises(KeyError):
                runtime.review(row["id"], "skip", reason="private", expected_revision=visible_revision)
            self.assertEqual(history.get(row["id"]), first_history)
            with runtime.adaptive._locked(): self.assertEqual(runtime.adaptive._read(), first_adaptive)
            self.assertEqual((learning.active(), learning.pending_gold()), first_learning)
            runtime.review(row["id"], "skip", reason="other", expected_revision=first_history["revision"])
            self.assertEqual(history.get(row["id"]), first_history)
            with runtime.adaptive._locked(): self.assertEqual(runtime.adaptive._read(), first_adaptive)
            self.assertEqual((learning.active(), learning.pending_gold()), first_learning)

    def test_skip_resumes_after_durable_withdrawal_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            history, learning = HistoryStore(directory), LearningStore(directory); coordinator = LearningCoordinator(history, learning)
            runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator)
            row = coordinator.append_live("hyp", prepare_canonical(np.zeros(800, np.float32)), .5, "m", language="en", adaptive=True, candidate={"stable_id":"x","backend":"w","repo":"r","revision":"q"})
            runtime.register_live(row)
            runtime.review(row["id"], "corrected", reference="gold", expected_revision=row["revision"])
            revision = history.get(row["id"])["revision"]
            original_resolve = runtime.adaptive.resolve_review
            runtime.adaptive.resolve_review = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted"))  # type: ignore[method-assign]
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                runtime.review(row["id"], "skip", reason="private", expected_revision=revision)
            withdrawn = history.get(row["id"])
            self.assertEqual(withdrawn["revision"], revision + 1)
            self.assertEqual(withdrawn["adaptive_review"]["revision"], withdrawn["revision"])
            self.assertFalse(learning.active_for_history(row["id"]))
            runtime.adaptive.resolve_review = original_resolve  # type: ignore[method-assign]
            runtime.review(row["id"], "skip", reason="private", expected_revision=withdrawn["revision"])
            self.assertEqual(runtime.adaptive.review_disposition(row["id"]), "skip")
            self.assertFalse(learning.active_for_history(row["id"]))

    def test_all_review_outbox_operations_resume_after_restart_without_text_in_metadata(self):
        for outcome, reference in (("corrected", "gold"), ("correct_as_is", ""), ("no_speech", ""), ("skip", "")):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                history, learning = HistoryStore(directory), LearningStore(directory); coordinator = LearningCoordinator(history, learning)
                runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator)
                row = coordinator.append_live("visible", prepare_canonical(np.zeros(800, np.float32)), .5, "m", language="en", adaptive=True, candidate={"stable_id":"x","backend":"w","repo":"r","revision":"q"})
                runtime.register_live(row)
                original = runtime.adaptive.resolve_review
                runtime.adaptive.resolve_review = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted"))  # type: ignore[method-assign]
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    runtime.review(row["id"], outcome, reference=reference, reason="private", expected_revision=row["revision"])
                stored = history.get(row["id"])
                self.assertNotIn("gold", repr(stored["adaptive_review"]))
                restarted = AdaptiveRuntime(directory)
                restarted.reconcile()
                self.assertEqual(restarted.adaptive.review_disposition(row["id"]), outcome)
                if outcome == "skip":
                    self.assertFalse(restarted.learning.active_for_history(row["id"]))
                else:
                    self.assertTrue(restarted.learning.active_for_history(row["id"]))

    def test_reconcile_fails_closed_while_outbox_enrollment_is_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            history, learning = HistoryStore(directory), LearningStore(directory); coordinator = LearningCoordinator(history, learning)
            runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator)
            row = coordinator.append_live("visible", prepare_canonical(np.zeros(800, np.float32)), .5, "m", language="en", adaptive=True, candidate={"stable_id":"x","backend":"w","repo":"r","revision":"q"})
            runtime.register_live(row)
            coordinator.enroll = lambda ident: (_ for _ in ()).throw(OSError("offline"))  # type: ignore[method-assign]
            runtime.review(row["id"], "corrected", reference="gold", expected_revision=row["revision"])
            self.assertEqual(runtime.reconcile()["state"], "review_recovery_blocked")

    def test_review_invalidation_precedes_outbox_and_artifact_failure_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            history, learning = HistoryStore(directory), LearningStore(directory); coordinator = LearningCoordinator(history, learning)
            runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator)
            row = coordinator.append_live("hyp", prepare_canonical(np.zeros(800, np.float32)), .5, "m", language="en", adaptive=True, candidate={"stable_id":"x","backend":"w","repo":"r","revision":"q"})
            runtime.register_live(row); runtime.review(row["id"], "corrected", reference="gold", expected_revision=row["revision"])
            before = history.get(row["id"]); active_before = learning.active_for_history(row["id"])
            original_pre = runtime.adaptive.pre_mutate_reference
            runtime.adaptive.pre_mutate_reference = lambda ident: (_ for _ in ()).throw(RuntimeError("invalidate failed"))  # type: ignore[method-assign]
            with self.assertRaisesRegex(RuntimeError, "invalidate failed"):
                runtime.review(row["id"], "skip", reason="private", expected_revision=before["revision"])
            self.assertEqual(history.get(row["id"]), before)
            self.assertEqual(learning.active_for_history(row["id"]), active_before)
            runtime.adaptive.pre_mutate_reference = original_pre  # type: ignore[method-assign]
            with runtime.adaptive._locked(): digest = runtime.adaptive._sample_digest(runtime.adaptive._read(), row["id"])
            base = candidate_manifest(WHISPER_ID, revision="base")
            runtime.adaptive._mutate(lambda state: state["champions"].update({"baseline": base, "current": {**base, "stable_id": "dependent", "development_dependency_digests": [digest], "invalid": False}}))
            original_revoke = learning.revoke
            learning.revoke = lambda sample_id: (_ for _ in ()).throw(OSError("artifact revoke failed"))  # type: ignore[method-assign]
            with self.assertRaisesRegex(OSError, "artifact revoke failed"):
                runtime.review(row["id"], "skip", reason="private", expected_revision=before["revision"])
            interrupted = history.get(row["id"])
            self.assertEqual(interrupted["adaptive_review"]["outcome"], "skip")
            self.assertFalse(interrupted["adaptive_review"].get("completed", False))
            self.assertEqual(runtime.adaptive.status()["current_champion"], WHISPER_ID)
            learning.revoke = original_revoke  # type: ignore[method-assign]
            runtime.reconcile()
            self.assertFalse(learning.active_for_history(row["id"]))
            self.assertEqual(runtime.adaptive.review_disposition(row["id"]), "skip")

    def test_incomplete_outbox_pins_history_audio_and_blocks_direct_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            history, learning = HistoryStore(directory), LearningStore(directory); coordinator = LearningCoordinator(history, learning)
            runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator)
            row = coordinator.append_live("hyp", prepare_canonical(np.zeros(800, np.float32)), .5, "m", language="en", adaptive=True, candidate={"stable_id":"x","backend":"w","repo":"r","revision":"q"})
            runtime.register_live(row)
            original = runtime.adaptive.resolve_review
            runtime.adaptive.resolve_review = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted"))  # type: ignore[method-assign]
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                runtime.review(row["id"], "corrected", reference="gold", expected_revision=row["revision"])
            history.keep = 1
            coordinator.append_live("later", prepare_canonical(np.zeros(800, np.float32)), .5, "m", language="en", adaptive=True)
            self.assertIsNotNone(history.get(row["id"]))
            self.assertTrue(history.audio_path(row["id"]).exists())
            # Direct evaluation cannot advance; depending on whether a fresh
            # pool exists, the fail-closed visible error is the interrupted
            # outbox recovery or the absence of a closed generation.
            with self.assertRaisesRegex(RuntimeError, "interrupted|no closed adaptive generation"):
                runtime.evaluate()
            runtime.adaptive.resolve_review = original  # type: ignore[method-assign]
            runtime.reconcile()
            # The privacy spool deliberately strips the pruned review's text;
            # it must remain non-evaluable rather than reconstructing gold.
            self.assertEqual(runtime.adaptive.review_disposition(row["id"]),"pending")

    def test_restart_outbox_skip_invalidates_dependency_created_before_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            history, learning = HistoryStore(directory), LearningStore(directory); coordinator = LearningCoordinator(history, learning)
            runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator)
            row = coordinator.append_live("hyp", prepare_canonical(np.zeros(800, np.float32)), .5, "m", language="en", adaptive=True, candidate={"stable_id":"x","backend":"w","repo":"r","revision":"q"})
            runtime.register_live(row); runtime.review(row["id"], "corrected", reference="gold", expected_revision=row["revision"])
            with runtime.adaptive._locked(): digest = runtime.adaptive._sample_digest(runtime.adaptive._read(), row["id"])
            base = candidate_manifest(WHISPER_ID, revision="base")
            runtime.adaptive._mutate(lambda state: state["champions"].update({"baseline": base, "current": {**base, "stable_id": "dependent", "development_dependency_digests": [digest], "invalid": False}}))
            revision = history.get(row["id"])["revision"]
            self.assertIsNotNone(coordinator.commit_adaptive_review(row["id"], "skip", skip_reason="private", expected_revision=revision))
            restarted = AdaptiveRuntime(directory)
            restarted.reconcile()
            self.assertEqual(restarted.adaptive.review_disposition(row["id"]), "skip")
            self.assertEqual(restarted.adaptive.status()["current_champion"], WHISPER_ID)

    def test_current_revision_review_actions_apply_expected_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            history, learning = HistoryStore(directory), LearningStore(directory); coordinator = LearningCoordinator(history, learning)
            runtime = AdaptiveRuntime(directory, history=history, learning=learning, coordinator=coordinator)
            outcomes = (("corrected", "gold"), ("correct_as_is", None), ("no_speech", None), ("skip", None))
            for index, (outcome, text) in enumerate(outcomes):
                row = coordinator.append_live("visible" + str(index), prepare_canonical(np.zeros(800, np.float32)), .5, "m", language="en", adaptive=True, candidate={"stable_id":"x","backend":"w","repo":"r","revision":"q"})
                runtime.register_live(row)
                runtime.review(row["id"], outcome, reference=text or "", reason="private", expected_revision=row["revision"])
                with runtime.adaptive._locked(): capture = runtime.adaptive._read()["captures"][row["id"]]
                self.assertEqual(capture.get("outcome"), outcome)
                if outcome == "correct_as_is": self.assertEqual(history.get(row["id"])["correction"]["text"], "visible" + str(index))


    def test_comparator_spool_is_private_idempotent_and_conflict_fenced(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            runtime=AdaptiveRuntime(Path(directory)); samples=np.zeros(1600,np.float32)
            wav,digest,_=runtime._prepare_comparator_spool(samples,16000)
            self.assertEqual(runtime._comparator_capture_id('session',digest,1600,1),runtime._comparator_capture_id('session',digest,1600,1))
            self.assertEqual(runtime._comparator_capture_id('session',digest,1600,1,'request'),runtime._comparator_capture_id('session',digest,1600,1,'request'))
            self.assertNotEqual(runtime._comparator_capture_id('session',digest,1600,1),runtime._comparator_capture_id('other',digest,1600,1))
            self.assertNotEqual(runtime._comparator_capture_id('session',digest,1600,1),runtime._comparator_capture_id('session',digest,1600,2))
            path,digest=runtime._write_comparator_spool(wav,"capture",digest)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode),0o600)
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode),0o700)
            _,identity=read_canonical_wav(path); self.assertEqual(identity.sha256,digest); self.assertEqual(identity.sample_count,1600)
            self.assertEqual(runtime._write_comparator_spool(wav,"capture",digest),(path,digest))
            changed,changed_digest,_=runtime._prepare_comparator_spool(np.ones(1600,np.float32),16000)
            with self.assertRaises(RuntimeError): runtime._write_comparator_spool(changed,"capture",changed_digest)
            with self.assertRaises(ValueError): runtime._write_comparator_spool(wav,"../escape",digest)
            self.assertFalse(runtime._delete_comparator_spool("capture","0"*64))
            self.assertTrue(path.exists())
            runtime._delete_comparator_spool("capture",digest); runtime._delete_comparator_spool("capture",digest)
            self.assertFalse(path.exists())

    def test_runtime_comparator_publication_ack_and_cancel_use_exact_spool_digest(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            runtime=AdaptiveRuntime(directory); store=runtime.silver; epoch=store.epoch(); session="spool-publication"; cohort="spool-cohort"; generation=41
            with store._connect() as db:
                db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,"h","evaluated_passed","m","c","{}",epoch,0))
                db.execute("INSERT INTO route_assignments VALUES(?,?,?,?)",(generation,hashlib.sha256(session.encode()).hexdigest(),1,0))
            wav,digest,_=runtime._prepare_comparator_spool(np.zeros(800,np.float32),16_000); path,_=runtime._write_comparator_spool(wav,"published",digest)
            candidate={"success":True,"fallback":False,"latency":.1,"coverage_ok":True,"hallucination_ok":True,"identity_valid":True}
            exact={"deployment_revision":9,"capture_id":"published","session_id":session,"cohort_id":cohort,"epoch":epoch,"route_generation":generation,"candidate_arm":"silver","audio_sha256":digest,"spool_name":path.name,"candidate":candidate}
            self.assertEqual(store.enqueue_comparator_intent(**exact),"inserted")
            metadata={"comparator_publication":{key:exact[key] for key in ("deployment_revision","capture_id","spool_name","audio_sha256","candidate")}}
            self.assertTrue(runtime.acknowledge_comparator_publication(metadata))
            self.assertTrue(runtime.cancel_comparator_publication(metadata))
            self.assertFalse(path.exists())
            with store._connect() as db:
                self.assertEqual(db.execute("SELECT status FROM comparator_intents WHERE deployment_revision=9 AND capture_id='published'").fetchone()[0],"discarded")

if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import os
import sqlite3
import stat
import sys
import unittest

if sys.platform != "darwin":
    raise unittest.SkipTest("adaptive lane is macOS-only in v1 (fcntl)")

import fcntl
import tempfile
from unittest.mock import patch
from pathlib import Path

import offline_backup
from adaptive_runtime import AdaptiveRuntime
from audio_codec import prepare_canonical
from comparator_spool import ComparatorSpool
from silver_store import SilverStore


class OfflineBackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.source=self.root/"source"; self._fixture(self.source)

    def tearDown(self):
        if hasattr(self,"_db"): self._db.close()
        self.tmp.cleanup()

    def _private(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        for item in [path.parent,*path.parent.parents]:
            if item == self.root.parent: break
            if item.exists(): os.chmod(item,0o700)
        path.write_bytes(data); os.chmod(path,0o600)

    def _fixture(self, root: Path) -> None:
        root.mkdir(mode=0o700); self._private(root/".sotto.lock",b"")
        self._private(root/"history.jsonl",b'{"schema_version":2,"id":"h1"}\n')
        self._private(root/"audio"/"h1.wav",b"canonical-audio")
        self._private(root/"adaptive-learning"/"state.json",b'{"schema":1}')
        self._private(root/"adaptive-learning"/".adaptive-state.lock",b"")
        self._private(root/"adaptive-learning"/".evaluator-process.lock",b"")
        self._private(root/"adaptive-learning"/"preflight"/"whisper.json",b'{"schema":3}')
        silver=root/"adaptive-learning"/"silver"; silver.mkdir(parents=True,mode=0o700); os.chmod(silver,0o700)
        db=sqlite3.connect(silver/"silver.sqlite3",timeout=0.1)
        db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            INSERT INTO meta VALUES('schema_version','4'); INSERT INTO meta VALUES('clear_epoch','3');
            CREATE TABLE epochs(clear_epoch INTEGER PRIMARY KEY,reason TEXT,ts REAL);
            CREATE TABLE cohorts(cohort_id TEXT PRIMARY KEY,status TEXT);
            INSERT INTO cohorts VALUES('old','evaluated_passed');
            CREATE TABLE cohort_authorization(cohort_id TEXT PRIMARY KEY,disposition TEXT,updated_ts REAL);
            CREATE TABLE runtime_observations(x TEXT); CREATE TABLE runtime_pairs(x TEXT);
            CREATE TABLE route_assignments(x TEXT); CREATE TABLE comparator_intents(x TEXT);
            CREATE TABLE alpha_ledger(generation INTEGER PRIMARY KEY,alpha REAL,created_ts REAL);
            INSERT INTO alpha_ledger VALUES(1,.025,1);
            CREATE TABLE calibrations(calibration_id TEXT PRIMARY KEY,state TEXT,manifest_hash TEXT);
            INSERT INTO calibrations VALUES('public-v2','passed','public-ledger');
        """)
        db.commit(); self._db=db; os.chmod(silver/"silver.sqlite3",0o600)
        self._private(silver/"deployment.jsonl",b'{"schema":1,"revision":5,"tier":"canary","current":{"stable_id":"x"},"lkg":null,"cohort_id":"old","canary":{},"calibration":null,"route_generation":7,"audit":[]}\n')
        self._private(silver/"experiments.jsonl",b'{"schema":1}\n')
        self._private(silver/"teacher-receipts.json",b'{"schema":1,"receipts":[]}')
        self._private(silver/"evidence"/"comparators"/"capture.wav",b"comparator-private")

    def test_backup_captures_wal_json_audio_without_mutating_source(self):
        before={p.relative_to(self.source): (p.read_bytes(),p.stat().st_mtime_ns) for p in self.source.rglob("*") if p.is_file() and not p.name.endswith(("-wal","-shm"))}
        bundle=offline_backup.backup(self.source,self.root/"bundle")
        manifest=offline_backup.verify(bundle)
        self.assertTrue((bundle/"data"/"audio"/"h1.wav").is_file())
        self.assertTrue(any(e.get("path")=="data/adaptive-learning/silver/silver.sqlite3" for e in manifest["entries"]))
        after={p.relative_to(self.source): (p.read_bytes(),p.stat().st_mtime_ns) for p in self.source.rglob("*") if p.is_file() and not p.name.endswith(("-wal","-shm"))}
        self.assertEqual(before,after)

    def test_verify_is_repeatable_and_restore_after_verify_succeeds(self):
        """Bundle inspection must not materialise WAL/SHM side files that the
        next verification would reject as extra artifacts."""
        bundle=offline_backup.backup(self.source,self.root/"bundle")
        offline_backup.verify(bundle); offline_backup.verify(bundle)
        self.assertEqual([p.name for p in bundle.rglob("*") if p.name.endswith(("-wal","-shm"))],[])
        target=self.root/"restored"
        offline_backup.restore(bundle,target)
        self.assertTrue((target/"history.jsonl").is_file())

    def test_restore_tolerates_pre_comparator_store_schema(self):
        """A valid backup of an older store (lazy migrations not yet run) must
        restore; absent volatile tables have nothing to purge."""
        self._db.execute("DROP TABLE comparator_intents"); self._db.commit()
        bundle=offline_backup.backup(self.source,self.root/"bundle-old")
        offline_backup.verify(bundle)
        target=self.root/"restored-old"
        offline_backup.restore(bundle,target)
        db=sqlite3.connect(target/"adaptive-learning"/"silver"/"silver.sqlite3")
        try:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM runtime_observations").fetchone()[0],0)
        finally:
            db.close()

    def test_reconstructible_caches_are_pruned_but_unknown_state_is_rejected(self):
        self._private(self.source/"huggingface"/"model.bin",b"never-backup")
        self._private(self.source/"teacher-runtimes"/"adapter",b"never-backup")
        bundle=offline_backup.backup(self.source,self.root/"bundle")
        self.assertFalse(any("huggingface" in str(path) or "teacher-runtimes" in str(path) for path in bundle.rglob("*")))
        self._private(self.source/"unexpected"/"state",b"no")
        with self.assertRaises(RuntimeError): offline_backup.backup(self.source,self.root/"bad")

    def test_tamper_malformed_and_symlink_refuse_before_publication(self):
        self._private(self.source/"history.jsonl",b"not-json\n")
        with self.assertRaises(RuntimeError): offline_backup.backup(self.source,self.root/"bad")
        self.assertFalse((self.root/"bad").exists())
        self._private(self.source/"history.jsonl",b'{"schema_version":2}\n')
        outside=self.root/"outside"; outside.write_bytes(b"x")
        link=self.source/"audio"/"bad.wav"; link.symlink_to(outside)
        with self.assertRaises(RuntimeError): offline_backup.backup(self.source,self.root/"symlink")
        self.assertEqual(outside.read_bytes(),b"x")

    def test_restore_forces_shadow_consumes_old_cohort_and_preserves_alpha(self):
        bundle=offline_backup.backup(self.source,self.root/"bundle")
        restored=offline_backup.restore(bundle,self.root/"restored")
        row=json.loads((restored/"adaptive-learning"/"silver"/"deployment.jsonl").read_text())
        self.assertEqual(row["tier"],"shadow_only"); self.assertIsNone(row["current"]); self.assertGreater(row["route_generation"],7)
        db=sqlite3.connect(restored/"adaptive-learning"/"silver"/"silver.sqlite3")
        self.assertEqual(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0],"4")
        self.assertEqual(db.execute("SELECT disposition FROM cohort_authorization WHERE cohort_id='old'").fetchone()[0],"consumed")
        self.assertEqual(db.execute("SELECT alpha FROM alpha_ledger WHERE generation=1").fetchone()[0],.025)
        self.assertEqual(db.execute("SELECT state FROM calibrations WHERE calibration_id='public-v2'").fetchone()[0],"passed")
        db.close()
        self.assertFalse((restored/"adaptive-learning"/"silver"/"evidence"/"comparators").exists())
        self.assertTrue((restored/"adaptive-learning"/"silver"/"teacher-receipts.json").is_file())

    def test_restore_forced_shadow_removes_comparator_adoption_outbox(self):
        """A restored prepared link cannot fence runtime recovery without its row."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            # This dedicated strict fixture replaces the lightweight setUp
            # database; close it first so the test never leaks a WAL handle.
            self._db.close(); del self._db
            root=Path(directory); source=root/"source"; prior_root=self.root; self.root=root
            try: self._fixture(source)
            finally: self.root=prior_root
            self._db.close()
            silver=source/"adaptive-learning"/"silver"
            for suffix in ("silver.sqlite3","silver.sqlite3-wal","silver.sqlite3-shm"):
                (silver/suffix).unlink(missing_ok=True)
            store=SilverStore(source); epoch=store.epoch(); session="restore-session"; cohort="restore-cohort"; generation=9
            with store._connect() as db:
                db.execute("INSERT INTO cohorts VALUES(?,?,?,?,?,?,?,?)",(cohort,"h","evaluated_passed","m","c","{}",epoch,0))
                db.execute("INSERT INTO route_assignments VALUES(?,?,?,?)",(generation,__import__("hashlib").sha256(session.encode()).hexdigest(),1,0))
                db.execute("INSERT INTO calibration_universes(holdout_key,manifest_hash,source_hash,protocol_hash,policy_hash,receipt_hash,state,created_ts,updated_ts,result) VALUES('public','m','s','p','policy','receipt','complete',1,1,'passed')")
            store.alpha(1)
            prepared=prepare_canonical(__import__("numpy").zeros(800,__import__("numpy").float32))
            import io, wave
            payload=io.BytesIO()
            with wave.open(payload,"wb") as handle:
                handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16_000); handle.setcomptype("NONE","not compressed"); handle.writeframes(prepared.pcm)
            digest=prepared.identity.sha256; spool=ComparatorSpool(source); spool.write("restore-cap.wav",payload.getvalue(),digest)
            candidate={"success":True,"fallback":False,"latency":.1,"coverage_ok":True,"hallucination_ok":True,"identity_valid":True}
            args={"deployment_revision":4,"capture_id":"restore-cap","session_id":session,"cohort_id":cohort,"epoch":epoch,"route_generation":generation,"candidate_arm":"candidate","audio_sha256":digest,"spool_name":"restore-cap.wav","candidate":candidate}
            self.assertEqual(store.enqueue_comparator_intent(**args),"inserted")
            self.assertTrue(store.adopt_comparator_publication(**{key:args[key] for key in ("deployment_revision","capture_id","spool_name","audio_sha256","candidate")},history_id="missing-history",history_revision=0))
            bundle=offline_backup.backup(source,root/"bundle"); restored=offline_backup.restore(bundle,root/"restored")
            db=sqlite3.connect(restored/"adaptive-learning"/"silver"/"silver.sqlite3")
            self.assertEqual(db.execute("SELECT COUNT(*) FROM comparator_intents").fetchone()[0],0)
            self.assertIsNone(db.execute("SELECT 1 FROM meta WHERE key LIKE 'comparator_adoption:%'").fetchone())
            self.assertEqual(db.execute("SELECT alpha FROM alpha_ledger WHERE generation=1").fetchone()[0],.025)
            self.assertEqual(db.execute("SELECT result FROM calibration_universes WHERE holdout_key='public'").fetchone()[0],"passed")
            db.close()
            self.assertFalse((restored/"adaptive-learning"/"silver"/"evidence"/"comparators").exists())
            runtime=AdaptiveRuntime(restored)
            self.assertEqual(runtime.recover_comparator_publications(),0)

    def test_restore_failure_leaves_existing_target_untouched(self):
        bundle=offline_backup.backup(self.source,self.root/"bundle")
        target=self.root/"occupied"; target.mkdir(mode=0o700); sentinel=target/"sentinel"; sentinel.write_text("keep")
        with self.assertRaises(RuntimeError): offline_backup.restore(bundle,target)
        self.assertEqual(sentinel.read_text(),"keep")

    def test_verify_refuses_content_mode_and_digest_alias_tamper(self):
        bundle=offline_backup.backup(self.source,self.root/"bundle")
        alias=bundle.parent/"alias"; os.rename(bundle,alias)
        with self.assertRaises(RuntimeError): offline_backup.verify(alias)
        os.rename(alias,bundle)
        audio=bundle/"data"/"audio"/"h1.wav"; audio.write_bytes(b"tampered")
        with self.assertRaises(RuntimeError): offline_backup.verify(bundle)

    def test_backup_final_component_symlink_swap_fails_without_touching_outside(self):
        outside=self.root/"outside"; outside.write_bytes(b"outside sentinel")
        original=offline_backup._copy_private; swapped=False
        def swap_then_copy(source, destination):
            nonlocal swapped
            if not swapped and source.name == "history.jsonl":
                swapped=True; source.unlink(); source.symlink_to(outside)
            return original(source,destination)
        with patch.object(offline_backup,"_copy_private",side_effect=swap_then_copy):
            with self.assertRaises(RuntimeError): offline_backup.backup(self.source,self.root/"swap")
        self.assertEqual(outside.read_bytes(),b"outside sentinel")
        self.assertFalse((self.root/"swap").exists())

    def test_restore_final_component_symlink_swap_fails_without_touching_outside(self):
        bundle=offline_backup.backup(self.source,self.root/"bundle")
        outside=self.root/"outside"; outside.write_bytes(b"outside sentinel")
        original=offline_backup._copy_private; swapped=False
        def swap_then_copy(source, destination):
            nonlocal swapped
            if not swapped and source.name == "history.jsonl":
                swapped=True; source.unlink(); source.symlink_to(outside)
            return original(source,destination)
        with patch.object(offline_backup,"_copy_private",side_effect=swap_then_copy):
            with self.assertRaises(RuntimeError): offline_backup.restore(bundle,self.root/"restore-swap")
        self.assertEqual(outside.read_bytes(),b"outside sentinel")
        self.assertFalse((self.root/"restore-swap").exists())

    def test_concurrent_silver_writer_is_detected_not_mixed(self):
        original=offline_backup._copy_private; fired=False
        def copy_then_write(source, destination):
            nonlocal fired
            result=original(source,destination)
            if not fired:
                fired=True
                try:
                    self._db.execute("UPDATE meta SET value='4' WHERE key='clear_epoch'"); self._db.commit()
                except sqlite3.OperationalError as exc:
                    raise RuntimeError("writer blocked") from exc
            return result
        with patch.object(offline_backup,"_copy_private",side_effect=copy_then_write):
            with self.assertRaises(RuntimeError): offline_backup.backup(self.source,self.root/"raced")
        self.assertFalse((self.root/"raced").exists())

    def test_state_json_writer_race_is_detected_before_publication(self):
        original=offline_backup._copy_private; fired=False
        def copy_then_mutate(source, destination):
            nonlocal fired
            result=original(source,destination)
            if not fired:
                fired=True; self._private(self.source/"adaptive-learning"/"state.json",b'{"schema":2}')
            return result
        with patch.object(offline_backup,"_copy_private",side_effect=copy_then_mutate):
            with self.assertRaises(RuntimeError): offline_backup.backup(self.source,self.root/"raced-state")
        self.assertFalse((self.root/"raced-state").exists())

    def test_each_authority_lock_fails_bounded_without_bundle(self):
        locks=[self.source/".sotto.lock",self.source/"adaptive-learning"/".adaptive-state.lock",
               self.source/"adaptive-learning"/".evaluator-process.lock"]
        for number,path in enumerate(locks):
            fd=os.open(path,os.O_RDWR); fcntl.flock(fd,fcntl.LOCK_EX)
            try:
                with self.assertRaises(RuntimeError): offline_backup.backup(self.source,self.root/f"locked-{number}")
            finally:
                fcntl.flock(fd,fcntl.LOCK_UN); os.close(fd)
            self.assertFalse((self.root/f"locked-{number}").exists())


if __name__ == "__main__": unittest.main()

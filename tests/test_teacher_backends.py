from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

import teacher_provision
import teacher_consensus
from teacher_backends import (GRANITE, OfflineTeacherRunner, QWEN, TeacherPreempted,
                              receipt_identity, runtime_identity, sha256_file,
                              snapshot_digest, snapshot_stat_manifest, TeacherFamily,
                              validate_teacher_receipts_read_only, _import_surface,
                              _runtime_markers_current)
from teacher_consensus import policy_hash


class OfflineTeacherRunnerTests(unittest.TestCase):
    @staticmethod
    def _snapshot(root: Path, family, *, link: bool = True) -> tuple[Path, Path]:
        model = root / ("models--" + family.repo.replace("/", "--"))
        blobs = model / "blobs"; snapshot = model / "snapshots" / family.revision
        blobs.mkdir(parents=True); snapshot.mkdir(parents=True)
        blob = blobs / "weights"; blob.write_bytes(b"teacher weights")
        if link:
            (snapshot / "weights.bin").symlink_to(Path("..") / ".." / "blobs" / blob.name)
        else:
            (snapshot / "weights.bin").write_bytes(b"direct teacher weights")
        return snapshot, blob

    @staticmethod
    def _receipt(family, snapshot: Path, adapter: Path, identity: dict) -> dict:
        return {"family": family.stable_id, "repo": family.repo, "revision": family.revision,
                "interpreter": sys.executable, "interpreter_identity": identity,
                "snapshot_path": str(snapshot), "snapshot_digest": snapshot_digest(family, snapshot),
                "package_versions": identity["packages"], "adapter": str(adapter),
                "adapter_hash": sha256_file(adapter), "decode": family.decode,
                "canonicalizer_hash": policy_hash(), "canonicalizer_source_hash": sha256_file(Path(teacher_consensus.__file__))}

    def test_teacher_snapshot_confines_hierarchy_and_hf_blob_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, _blob = self._snapshot(root, QWEN)
            self.assertEqual(len(snapshot_digest(QWEN, snapshot)), 64)
            self.assertEqual(len(snapshot_stat_manifest(QWEN, snapshot)), 64)

            granite, granite_blob = self._snapshot(root, GRANITE)
            (snapshot / "weights.bin").unlink()
            (snapshot / "weights.bin").symlink_to(granite_blob)
            with self.assertRaisesRegex(RuntimeError, "blobs"):
                snapshot_digest(QWEN, snapshot)

            wrong = root / "models--wrong" / "snapshots" / QWEN.revision
            wrong.mkdir(parents=True); (wrong.parent.parent / "blobs").mkdir()
            with self.assertRaisesRegex(RuntimeError, "pinned family"):
                snapshot_digest(QWEN, wrong)

            external_snapshot, _ = self._snapshot(root / "outside", QWEN)
            alias_parent = root / "alias-parent"; alias_parent.mkdir()
            alias_model = alias_parent / external_snapshot.parent.parent.name
            alias_model.symlink_to(external_snapshot.parent.parent, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                snapshot_digest(QWEN, alias_model / "snapshots" / QWEN.revision)

    def test_teacher_snapshot_cache_rehashes_replaced_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); qwen_snapshot, qwen_blob = self._snapshot(root, QWEN); granite_snapshot, _ = self._snapshot(root, GRANITE)
            qwen_adapter = root / "qwen_adapter.py"; granite_adapter = root / "granite_adapter.py"
            qwen_adapter.write_text("# qwen\n", encoding="utf-8"); granite_adapter.write_text("# granite\n", encoding="utf-8")
            identity = {"python": "fixed", "executable": "fixed", "packages": {name: "1" for name in set(QWEN.critical_packages + GRANITE.critical_packages)}}
            receipts = {QWEN.stable_id: self._receipt(QWEN, qwen_snapshot, qwen_adapter, identity),
                        GRANITE.stable_id: self._receipt(GRANITE, granite_snapshot, granite_adapter, identity)}
            self.assertEqual(len(receipt_identity(receipts[QWEN.stable_id])), 64)
            runner = OfflineTeacherRunner(receipts)
            with mock.patch("teacher_backends.runtime_identity", return_value=identity):
                runner.validate()  # warm the cheap marker/content-digest cache
                replacement = qwen_blob.with_name("replacement"); replacement.write_bytes(b"rotated teacher weights")
                os.replace(replacement, qwen_blob)
                with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                    runner.validate()

    def test_provision_receipt_uses_one_runtime_identity_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); snapshot, _ = self._snapshot(root, QWEN); adapter = root / "adapter.py"; adapter.write_text("# adapter\n")
            identity = {"python": "fixed", "executable": "fixed", "packages": {name: "1" for name in QWEN.critical_packages}}
            with mock.patch("teacher_provision.runtime_identity", return_value=identity) as probe:
                receipt = teacher_provision._receipt(QWEN, Path(sys.executable), snapshot, adapter)
            self.assertEqual(probe.call_count, 1)
            self.assertEqual(receipt["interpreter_identity"], identity)
            self.assertEqual(receipt["package_versions"], identity["packages"])

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (macOS venv probe / POSIX process preemption)")
    def test_cancellable_teacher_process_is_preempted_without_output_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); adapter=root/'adapter.py'; audio=root/'audio.wav'
            audio.write_bytes(b'not-read')
            adapter.write_text("import time\nprint('teacher-secret-output',flush=True)\ntime.sleep(30)\n",encoding='utf-8')
            receipt={'interpreter':sys.executable,'adapter':str(adapter),'adapter_hash':sha256_file(adapter),'snapshot_path':str(root)}
            children=[]
            def popen(*args,**kwargs):
                child=subprocess.Popen(*args,**kwargs); children.append(child); return child
            runner=OfflineTeacherRunner({QWEN.stable_id:receipt},popen=popen)
            started=time.monotonic()
            def cancel() -> bool: return time.monotonic()-started >= .05
            with mock.patch.object(runner,"validate",return_value="fixture"), self.assertRaises(TeacherPreempted) as raised:
                runner.transcribe(QWEN,audio,'opaque',cancel=cancel)
            elapsed=time.monotonic()-started
            self.assertLess(elapsed,1.5); self.assertEqual(len(children),1); self.assertIsNotNone(children[0].poll())
            with self.assertRaises(ProcessLookupError): os.killpg(children[0].pid,0)
            self.assertNotIn('teacher-secret-output',str(raised.exception))

            calls=[]
            def invoke(*args,**kwargs):
                calls.append((args,kwargs)); return subprocess.CompletedProcess(args[0],0,stdout='normal-path',stderr='')
            normal=OfflineTeacherRunner({QWEN.stable_id:receipt},invoke=invoke)
            with mock.patch.object(normal,"validate",return_value="fixture"):
                self.assertEqual(normal.transcribe(QWEN,audio,'opaque'),'normal-path')
            self.assertEqual(len(calls),1)
            self.assertEqual({name:calls[0][1]['env'][name] for name in ('SOTTO_OFFLINE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')},
                             {'SOTTO_OFFLINE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})

    def test_teacher_probe_and_cancellable_inference_receive_all_offline_flags(self):
        seen=[]
        def run(*args,**kwargs):
            seen.append(kwargs['env'])
            payload={"python":"x","executable":"x","prefix":"x","base_prefix":"x","sys_path":[str(Path.cwd())],"import_roots":[{"root":str(Path.cwd()),"marker":[0,0,0,0,0],"entries":[]}],"distributions":[{"name":"numpy","version":"x","root":str(Path.cwd()),"files":[]}]}
            return subprocess.CompletedProcess(args[0],0,stdout=json.dumps(payload),stderr='')
        with mock.patch('teacher_backends.subprocess.run',side_effect=run), mock.patch('teacher_backends._import_surfaces',return_value=[]):
            runtime_identity(Path(sys.executable),('numpy',))
        receipt={'interpreter':sys.executable,'adapter':str(Path(__file__)),'adapter_hash':sha256_file(Path(__file__)),'snapshot_path':str(Path(__file__).parent)}
        class Sink:
            @staticmethod
            def write(_data): pass
            @staticmethod
            def close(): pass
        class Process:
            pid=0; returncode=0; stdin=Sink()
            def poll(_self): return 0
            def communicate(_self): return ('ok','')
        runner=OfflineTeacherRunner({QWEN.stable_id:receipt},popen=lambda *_args,**kwargs: (seen.append(kwargs['env']) or Process()))
        with mock.patch.object(runner,"validate",return_value="fixture"):
            self.assertEqual(runner.transcribe(QWEN,Path(__file__),'opaque',cancel=lambda: False),'ok')
        expected={'SOTTO_OFFLINE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
        self.assertTrue(all({name:value[name] for name in expected} == expected for value in seen))

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (macOS venv probe / POSIX process preemption)")
    def test_launcher_wrapper_and_runtime_closure_are_bound_before_inference(self):
        """A wrapper that honestly forwards probes cannot swap inference bytes."""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); family=TeacherFamily("fixture-teacher","fixture/repo","rev","numpy",{"decode":"fixture"},("numpy",))
            model=root/"models--fixture--repo"; blobs=model/"blobs"; snapshot=model/"snapshots"/"rev"; blobs.mkdir(parents=True); snapshot.mkdir(parents=True)
            blob=blobs/"weights"; blob.write_bytes(b"weights"); (snapshot/"weights").symlink_to(Path("..")/".."/"blobs"/"weights")
            adapter=root/"adapter.py"; adapter.write_text("# adapter\n")
            wrapper=root/"teacher-python"; wrapper.write_text(f"#!/bin/sh\nif [ \"$1\" = \"-I\" ] && [ \"$2\" = \"-S\" ] && [ \"$3\" = \"-B\" ] && [ \"$4\" = \"-c\" ]; then exec {sys.executable} \"$@\"; fi\nprintf 'fixed inference\\n'\n") ; os.chmod(wrapper,0o700)
            with mock.patch("teacher_backends.REQUIRED_TEACHERS",(family,)):
                receipt=teacher_provision._receipt(family,wrapper,snapshot,adapter)
                self.assertEqual(receipt["interpreter_identity"]["schema"],4)
                self.assertTrue(validate_teacher_receipts_read_only({family.stable_id:receipt}))
                runner=OfflineTeacherRunner({family.stable_id:receipt})
                self.assertEqual(runner.transcribe(family,root/"audio.wav","opaque"),"fixed inference")
                # The first validate above populated the runner cache; an
                # unchanged second pass is marker-only and must not reprobe.
                with mock.patch("teacher_backends.runtime_identity",side_effect=AssertionError("unexpected full probe")):
                    runner.validate(); runner.validate()
                wrapper.write_text(f"#!/bin/sh\nif [ \"$1\" = \"-I\" ] && [ \"$2\" = \"-S\" ] && [ \"$3\" = \"-B\" ] && [ \"$4\" = \"-c\" ]; then exec {sys.executable} \"$@\"; fi\nprintf 'rotated inference\\n'\n"); os.chmod(wrapper,0o700)
                with self.assertRaisesRegex(RuntimeError,"runtime changed"):
                    runner.validate()

    def test_isolated_import_surface_rejects_added_module_pth_and_startup_artifact(self):
        """An importable addition is a marker mismatch before adapter audio use."""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); surface=root/"site-packages"; surface.mkdir(); (surface/"known.py").write_text("VALUE=1\n")
            expected=_import_surface(surface,with_hash=True)
            # Minimal surrounding identity lets this test exercise the exact
            # marker projection without a model/runtime dependency.
            from teacher_backends import _launcher_identity
            identity={"schema":4,"launcher":_launcher_identity(Path(sys.executable)),"closure":[],"import_surface":[expected]}
            # Bind the executable hash marker tuple correctly for the helper.
            self.assertTrue(_runtime_markers_current(identity))
            for name in ("forged_sibling.py","extra.pth","sitecustomize.py"):
                (surface/name).write_text("# harmless\n")
                self.assertFalse(_runtime_markers_current(identity))
                (surface/name).unlink()

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (macOS venv probe / POSIX process preemption)")
    def test_configured_venv_probe_verifies_roots_before_insertion_and_launch(self):
        """A changed import root is rejected before any adapter can execute;
        an unchanged configured teacher venv launches normally end to end."""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); family=TeacherFamily("fixture-teacher","fixture/repo","rev","fixturepkg",{"decode":"fixture"},("fixturepkg",))
            model=root/"models--fixture--repo"; blobs=model/"blobs"; snapshot=model/"snapshots"/"rev"; blobs.mkdir(parents=True); snapshot.mkdir(parents=True)
            blob=blobs/"weights"; blob.write_bytes(b"weights"); (snapshot/"weights").symlink_to(Path("..")/".."/"blobs"/"weights")
            adapter=root/"adapter.py"
            adapter.write_text("import json,sys\n"
                               "payload=json.load(sys.stdin)\n"
                               "for row in reversed(payload.get('expected_import_roots') or []): sys.path.insert(0,row['root'])\n"
                               "import fixturepkg\n"
                               "print('adapter-ok',fixturepkg.VALUE)\n")
            subprocess.run([sys.executable,"-m","venv","--without-pip",str(root/"venv")],check=True,capture_output=True)
            interpreter=root/"venv"/"bin"/"python3"
            site=next((root/"venv"/"lib").glob("python*"))/"site-packages"
            (site/"fixturepkg.py").write_text("VALUE=1\n")
            dist=site/"fixturepkg-1.0.dist-info"; dist.mkdir()
            (dist/"METADATA").write_text("Metadata-Version: 2.1\nName: fixturepkg\nVersion: 1.0\n")
            (dist/"RECORD").write_text("fixturepkg.py,,\nfixturepkg-1.0.dist-info/METADATA,,\nfixturepkg-1.0.dist-info/RECORD,,\n")
            with mock.patch("teacher_backends.REQUIRED_TEACHERS",(family,)):
                receipt=teacher_provision._receipt(family,interpreter,snapshot,adapter)
                identity=receipt["interpreter_identity"]
                self.assertEqual(identity["schema"],4)
                self.assertTrue(any(Path(row["root"]).resolve()==site.resolve() for row in identity["import_roots"]))
                runner=OfflineTeacherRunner({family.stable_id:receipt})
                # Normal configured teacher launch through full receipt
                # validation, importing a real package from the verified root.
                self.assertEqual(runner.transcribe(family,root/"audio.wav","opaque"),"adapter-ok 1")
                # Every importable-surface addition below is rejected before the
                # adapter process is spawned, on warmed and fresh runners alike.
                sentinel=root/"executed"
                for name in ("sitecustomize.py","usercustomize.py","forged_sibling.py","extra.pth"):
                    (site/name).write_text(f"open({str(sentinel)!r},'w').write('boom')\n")
                    launches=[]
                    fresh=OfflineTeacherRunner({family.stable_id:json.loads(json.dumps(receipt))},
                                               invoke=lambda *args,**kwargs: launches.append(args),
                                               popen=lambda *args,**kwargs: launches.append(args))
                    with self.assertRaisesRegex(RuntimeError,"unavailable|changed"):
                        fresh.transcribe(family,root/"audio.wav","opaque")
                    with self.assertRaisesRegex(RuntimeError,"unavailable|changed"):
                        runner.transcribe(family,root/"audio.wav","opaque")
                    self.assertEqual(launches,[]); self.assertFalse(sentinel.exists())
                    (site/name).unlink()

    def test_real_adapters_verify_import_root_trees_before_insertion(self):
        """Both pinned adapters re-verify the frozen trees in-child and refuse
        a changed root (exit 3) before inserting or importing anything."""
        import sealed_release as release
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); site=root/"site"; site.mkdir(); module=site/"module.py"
            for adapter in ("qwen3_adapter.py","granite_adapter.py"):
                path=Path(__file__).parents[1]/"teachers"/adapter
                argv=[sys.executable,"-I","-S","-B",str(path),"--audio","x","--identity","x","--snapshot",str(root),"--decode","{}"]
                module.write_text("v1\n")
                payload=json.dumps({"expected_import_roots":[release._sealed_import_root(site)]})
                # Matching trees: verification passes, insertion happens, and
                # the adapter fails later on the fixture decode (exit 2).
                good=subprocess.run(argv,input=payload,text=True,capture_output=True)
                self.assertEqual((good.returncode,good.stderr.strip()),(2,"teacher_failed"))
                # A root changed after the runner's validation is rejected by
                # the child itself before any insertion (exit 3).
                module.write_text("changed\n")
                bad=subprocess.run(argv,input=payload,text=True,capture_output=True)
                self.assertEqual((bad.returncode,bad.stderr.strip()),(3,"teacher_failed"))

    def test_directory_link_target_descendants_and_cycles_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); surface=root/"surface"; target=root/"target"; surface.mkdir(); target.mkdir()
            leaf=target/"module.py"; leaf.write_text("v1\n"); (surface/"linked").symlink_to(target,target_is_directory=True)
            expected=_import_surface(surface,with_hash=True)
            from teacher_backends import _launcher_identity
            identity={"schema":4,"launcher":_launcher_identity(Path(sys.executable)),"closure":[],"import_surface":[expected]}
            self.assertTrue(_runtime_markers_current(identity))
            (target/"added.py").write_text("added\n"); self.assertFalse(_runtime_markers_current(identity)); (target/"added.py").unlink()
            leaf.write_text("v2 replacement\n"); self.assertFalse(_runtime_markers_current(identity))
            cycle=root/"cycle"; cycle.mkdir(); (cycle/"self").symlink_to(cycle,target_is_directory=True)
            with self.assertRaises(RuntimeError): _import_surface(cycle,with_hash=True)


if __name__ == '__main__':
    unittest.main()

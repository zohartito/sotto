from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
import os
import sys
import types
from unittest.mock import patch

import numpy as np

from audio_codec import prepare_canonical, read_canonical_wav, write_canonical_wav
from speech_backends import (LocalBackendManager, PARAKEET_ID, WHISPER_ID, WHISPER_GLOSSARY_ID, PreflightReceipts,
                             WHISPER_REPO, _candidate_runtime_identity, _package_versions, _safe_snapshot_digest,
                             candidate_manifest, evaluator_hash, resolve_snapshot)
from runtime_source_manifest import RUNTIME_SOURCE_SCHEMA, runtime_source_digest


class _Whisper:
    def transcribe(self, samples, **kwargs):
        self.samples = samples; self.kwargs = kwargs
        return {"text": " hello "}


class _Parakeet:
    def transcribe(self, path):
        self.path = Path(path)
        self.exists_during_call = self.path.exists()
        return type("Result", (), {"text": " world "})()


class SpeechBackendTests(unittest.TestCase):
    @staticmethod
    def _schema3_receipt(manifest, snapshot: Path, runtime: dict) -> dict:
        return {"schema": 3, "success": True, "manifest": {key:value for key,value in manifest.items() if key != "glossary_terms"},
                "snapshot_path": str(snapshot), "snapshot_revision": manifest["revision"],
                "snapshot_metadata_digest": _safe_snapshot_digest(snapshot), "package_versions": manifest["package_versions"],
                "runtime_identity": runtime, "runtime_source_schema": RUNTIME_SOURCE_SCHEMA,
                "runtime_source_digest": runtime_source_digest(), "evaluator_id": "sotto-paired-v1", "evaluator_hash": evaluator_hash()}

    def test_schema3_receipt_requires_runtime_and_complete_source_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); revision = "source-contract"
            snapshot = root / "hf" / "hub" / "models--mlx-community--whisper-large-v3-turbo" / "snapshots" / revision
            snapshot.mkdir(parents=True); (snapshot / "model").write_text("safe")
            packages = _package_versions(); identity = {"runtime": "test"}
            manifest = candidate_manifest(WHISPER_ID, revision=revision, package_versions=packages, runtime_identity=identity)
            receipt = {"schema": 3, "success": True, "manifest": {key:value for key,value in manifest.items() if key != "glossary_terms"},
                       "snapshot_path": str(snapshot), "snapshot_revision": revision,
                       "snapshot_metadata_digest": _safe_snapshot_digest(snapshot), "package_versions": packages,
                       "runtime_identity": identity, "runtime_source_schema": RUNTIME_SOURCE_SCHEMA,
                       "runtime_source_digest": runtime_source_digest(), "evaluator_id": "sotto-paired-v1",
                       "evaluator_hash": evaluator_hash()}
            store = PreflightReceipts(root); store.save(WHISPER_ID, receipt)
            with patch("speech_backends.resolve_snapshot", return_value=(revision, snapshot)), \
                 patch.object(store, "_current_runtime_identity", return_value=identity):
                self.assertTrue(store.valid_manifest(manifest))
                for mutate in (lambda value: value.__setitem__("schema", 2),
                               lambda value: value.pop("runtime_identity"),
                               lambda value: value.__setitem__("runtime_source_digest", "0" * 64)):
                    candidate = dict(receipt); mutate(candidate); store.save(WHISPER_ID, candidate)
                    self.assertFalse(store.valid_manifest(manifest))

    def test_whisper_uses_exact_samples_and_forced_english(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = _Whisper(); manager = LocalBackendManager(directory, whisper=fake, snapshot_resolver=lambda _: Path(directory))
            manifest = candidate_manifest(WHISPER_ID, revision="abc")
            samples = np.zeros(20, dtype=np.float32)
            text, metadata = manager.transcribe(manifest, samples)
            self.assertEqual(text, "hello")
            self.assertTrue(np.array_equal(fake.samples, prepare_canonical(samples).asr_samples))
            self.assertEqual(fake.kwargs["language"], "en")
            self.assertEqual(metadata["candidate_id"], WHISPER_ID)

    @unittest.skipUnless(sys.platform == "darwin", "adaptive lane is macOS-only in v1 (ffmpeg/venv runtime identity)")
    def test_parakeet_uses_private_mode_600_wav_then_removes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = _Parakeet(); manager = LocalBackendManager(directory, parakeet_loader=lambda _: fake, snapshot_resolver=lambda _: Path(directory))
            manifest = candidate_manifest(PARAKEET_ID, revision="abc")
            self.assertEqual(manager.transcribe(manifest, np.zeros(20, dtype=np.float32))[0], "world")
            self.assertTrue(fake.exists_during_call)
            self.assertFalse(fake.path.exists())
            self.assertEqual(fake.path.parent, Path(directory) / "adaptive-learning" / "tmp")

    @unittest.skipUnless(sys.platform == "darwin", "adaptive lane is macOS-only in v1 (ffmpeg/venv runtime identity)")
    def test_parakeet_preserves_prepared_pcm_and_evaluator_uses_validated_path(self):
        class InspectParakeet(_Parakeet):
            def transcribe(self, path):
                self.path = Path(path); self.exists_during_call = self.path.exists()
                _, self.identity_during_call = read_canonical_wav(self.path)
                return type("Result", (), {"text": " world "})()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); whisper = _Whisper(); parakeet = InspectParakeet()
            manager = LocalBackendManager(root, whisper=whisper, parakeet_loader=lambda _: parakeet,
                                          snapshot_resolver=lambda _: root)
            # Nonzero PCM can change by one LSB if decoded /32768 then
            # re-encoded with the asymmetric legacy float encoder.
            prepared = prepare_canonical(np.array([.25, -.25, .03125, -.03125], np.float32))
            parakeet_manifest = candidate_manifest(PARAKEET_ID, revision="abc")
            whisper_manifest = candidate_manifest(WHISPER_ID, revision="abc")
            manager.transcribe(parakeet_manifest, prepared)
            self.assertEqual(parakeet.identity_during_call, prepared.identity)
            manager.transcribe(whisper_manifest, prepared)
            self.assertTrue(np.array_equal(whisper.samples, prepared.asr_samples))
            capture_path = root / "capture.wav"; write_canonical_wav(capture_path, prepared.pcm)
            manager.transcribe(parakeet_manifest, np.zeros(4, np.float32), canonical_path=capture_path,
                               canonical_identity=prepared.identity)
            self.assertEqual(parakeet.path, capture_path)
            self.assertEqual(parakeet.identity_during_call, prepared.identity)

    @unittest.skipUnless(sys.platform == "darwin", "adaptive lane is macOS-only in v1 (ffmpeg/venv runtime identity)")
    def test_exact_snapshot_path_and_parakeet_reloads_for_changed_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths = [root / "one", root / "two"]
            for path in paths: path.mkdir()
            whisper = _Whisper(); loaded = []
            def loader(path):
                loaded.append(path); return _Parakeet()
            manager = LocalBackendManager(directory, whisper=whisper, parakeet_loader=loader,
                snapshot_resolver=lambda manifest: paths[0] if manifest["revision"] == "one" else paths[1])
            manager.transcribe(candidate_manifest(WHISPER_ID, revision="one"), np.zeros(8, np.float32))
            self.assertEqual(whisper.kwargs["path_or_hf_repo"], str(paths[0]))
            manager.transcribe(candidate_manifest(PARAKEET_ID, revision="one"), np.zeros(8, np.float32))
            manager.transcribe(candidate_manifest(PARAKEET_ID, revision="two"), np.zeros(8, np.float32))
            self.assertEqual(loaded, [str(paths[0]), str(paths[1])])

    @unittest.skipUnless(sys.platform == "darwin", "adaptive lane is macOS-only in v1 (ffmpeg/venv runtime identity)")
    def test_preflight_offline_fails_before_download_and_online_uses_exact_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); snapshot = root / "cache" / "rev"; snapshot.mkdir(parents=True)
            seen = []
            class Manager(LocalBackendManager):
                def transcribe(self, manifest, samples):
                    seen.append(manifest); return "", {}
            manager = Manager(root)
            with patch("speech_backends.resolve_snapshot", side_effect=FileNotFoundError("missing")), \
                 patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}, clear=False), \
                 patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=lambda **_: self.fail("download"))}):
                with self.assertRaises(RuntimeError): manager.preflight(WHISPER_ID, np.zeros(8), hf_home=root)
            with patch("speech_backends.resolve_snapshot", side_effect=[FileNotFoundError("missing"),("rev",snapshot)]), \
                 patch.dict(os.environ, {"HF_HUB_OFFLINE": "0", "SOTTO_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}, clear=False), \
                 patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=lambda **_: str(snapshot))}):
                manager.preflight(WHISPER_ID, np.zeros(8), hf_home=root)
            self.assertEqual(seen[-1]["snapshot_path"], str(snapshot.resolve()))

    def test_hf_blob_symlink_snapshot_layout_is_confined(self):
        """The real Hub cache stores snapshot files as blob links; accept
        exactly that confinement, reject any escape, bind target bytes."""
        from speech_backends import _strict_snapshot
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); model=root/"models--x--y"; blobs=model/"blobs"; snap=model/"snapshots"/"rev"
            blobs.mkdir(parents=True); snap.mkdir(parents=True)
            (blobs/"b1").write_bytes(b"weights")
            (snap/"weights.bin").symlink_to(Path("..")/".."/"blobs"/"b1")
            (snap/"config.json").write_text("{}")
            _strict_snapshot(snap)
            first=_safe_snapshot_digest(snap)
            self.assertEqual(first,_safe_snapshot_digest(snap))
            outside=root/"outside"; outside.write_bytes(b"outside")
            (snap/"escape.bin").symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError,"blobs"):
                _strict_snapshot(snap)
            with self.assertRaisesRegex(RuntimeError,"blobs"):
                _safe_snapshot_digest(snap)
            (snap/"escape.bin").unlink()
            (blobs/"b1").write_bytes(b"rotated")
            self.assertNotEqual(first,_safe_snapshot_digest(snap))
            # The cheap validation cache must also observe link/blob changes:
            # a mutated blob or a retargeted link forces a digest recheck.
            (blobs/"b1").write_bytes(b"weights")
            runtime={"runtime":"links"}
            manifest=candidate_manifest(WHISPER_ID,revision="rev",runtime_identity=runtime)
            receipts=PreflightReceipts(root)
            receipts.save(WHISPER_ID,self._schema3_receipt(manifest,snap,runtime))
            with patch("speech_backends._safe_snapshot_digest",wraps=_safe_snapshot_digest) as digest, \
                 patch("speech_backends.resolve_snapshot",return_value=("rev",snap)), \
                 patch.object(receipts,"_current_runtime_identity",return_value=runtime):
                self.assertTrue(receipts.valid_manifest(manifest)); self.assertTrue(receipts.valid_manifest(manifest))
                self.assertEqual(digest.call_count,1)
                (blobs/"b1").write_bytes(b"mutated blob")
                self.assertFalse(receipts.valid_manifest(manifest))
                (blobs/"b2").write_bytes(b"weights")
                (snap/"weights.bin").unlink(); (snap/"weights.bin").symlink_to(Path("..")/".."/"blobs"/"b2")
                self.assertFalse(receipts.valid_manifest(manifest))

    def test_each_offline_flag_fences_missing_candidate_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            class Manager(LocalBackendManager):
                def transcribe(self,*_args,**_kwargs): self.fail('must not warm/download offline')
            for flag in ('SOTTO_OFFLINE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'):
                with self.subTest(flag=flag), patch('speech_backends.resolve_snapshot',side_effect=FileNotFoundError('missing')), \
                     patch.dict(os.environ,{'SOTTO_OFFLINE':'0','HF_HUB_OFFLINE':'0','TRANSFORMERS_OFFLINE':'0',flag:'true'},clear=False), \
                     patch.dict(sys.modules,{'huggingface_hub':types.SimpleNamespace(snapshot_download=lambda **_: self.fail('download'))}):
                    with self.assertRaisesRegex(RuntimeError,'offline preflight'):
                        Manager(root).preflight(WHISPER_ID,np.zeros(8),hf_home=root)

    def test_receipt_content_cache_rehashes_on_tree_or_receipt_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); snap = root / "rev"; snap.mkdir(); model = snap / "model"; model.write_bytes(b"aaaa")
            runtime={"runtime":"cache"}; manifest = candidate_manifest(WHISPER_ID, revision="rev", runtime_identity=runtime)
            receipt = self._schema3_receipt(manifest, snap, runtime)
            receipts = PreflightReceipts(root); receipts.save(WHISPER_ID, receipt)
            with patch("speech_backends._safe_snapshot_digest", wraps=_safe_snapshot_digest) as digest, \
                 patch("speech_backends.resolve_snapshot", return_value=("rev",snap)), \
                 patch.object(receipts,"_current_runtime_identity",return_value=runtime):
                self.assertTrue(receipts.valid_manifest(manifest)); self.assertTrue(receipts.valid_manifest(manifest))
                self.assertEqual(digest.call_count, 1)
                model.write_bytes(b"bbbb")
                self.assertFalse(receipts.valid_manifest(manifest)); self.assertEqual(digest.call_count, 2)

    def test_receipt_cache_keeps_two_independent_snapshots_hot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); receipts = PreflightReceipts(root); manifests = []
            runtime={"runtime":"two"}
            for stable, revision in ((WHISPER_ID, "one"), (PARAKEET_ID, "two")):
                snap = root / revision; snap.mkdir(); (snap / "model").write_bytes(revision.encode())
                manifest = candidate_manifest(stable, revision=revision,runtime_identity=runtime); manifests.append((manifest, snap))
                receipts.save(stable, self._schema3_receipt(manifest,snap,runtime))
            with patch("speech_backends._safe_snapshot_digest", wraps=_safe_snapshot_digest) as digest, \
                 patch("speech_backends.resolve_snapshot", side_effect=lambda _repo,revision: (revision,next(path for item,path in manifests if item["revision"] == revision))), \
                 patch.object(receipts,"_current_runtime_identity",return_value=runtime):
                for manifest, _ in manifests:
                    self.assertTrue(receipts.valid_manifest(manifest)); self.assertTrue(receipts.valid_manifest(manifest))
                self.assertEqual(digest.call_count, 2)
                (manifests[0][1] / "model").write_bytes(b"changed")
                self.assertFalse(receipts.valid_manifest(manifests[0][0])); self.assertTrue(receipts.valid_manifest(manifests[1][0]))
                self.assertEqual(digest.call_count, 3)

    def test_receipt_projection_accepts_deployment_fields_but_rejects_identity_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); snap = root / "rev"; snap.mkdir(); (snap / "x").write_text("x")
            receipt_store = PreflightReceipts(root)
            runtime={"runtime":"projection"}; manifest = candidate_manifest(WHISPER_ID, revision="rev",runtime_identity=runtime)
            receipt_store.save(WHISPER_ID, self._schema3_receipt(manifest,snap,runtime))
            promoted = {**manifest, "generation_id": "g", "promotion_dependency_digests": ["d"], "invalid": False}
            glossary = candidate_manifest(WHISPER_GLOSSARY_ID, revision="rev", glossary_terms=("sotto",),runtime_identity=runtime)
            receipt_store.save(WHISPER_GLOSSARY_ID, self._schema3_receipt(glossary,snap,runtime))
            frozen = {**glossary, "development_dependency_digests": ["private"], "generation_id": "g"}
            with patch("speech_backends.resolve_snapshot",return_value=("rev",snap)), patch.object(receipt_store,"_current_runtime_identity",return_value=runtime):
                self.assertTrue(receipt_store.valid_manifest(promoted))
                self.assertFalse(receipt_store.valid_manifest({**promoted, "revision": "other"}))
                self.assertTrue(receipt_store.valid_manifest(frozen))
                self.assertFalse(receipt_store.valid_manifest({**frozen, "glossary_hash": "tampered"}))

    @unittest.skipUnless(sys.platform == "darwin", "adaptive lane is macOS-only in v1 (ffmpeg/venv runtime identity)")
    def test_candidate_runtime_identity_rotation_invalidates_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); snapshot=root/'rev'; snapshot.mkdir(); (snapshot/'model').write_text('x')
            ffmpeg=root/'ffmpeg'; ffmpeg.write_text('#!/bin/sh\nprintf "ffmpeg version Sotto-one\\n"\n'); os.chmod(ffmpeg,0o700)
            packages=_package_versions()
            identity=_candidate_runtime_identity(WHISPER_ID,ffmpeg_path=ffmpeg,package_versions=packages)
            manifest=candidate_manifest(WHISPER_ID,revision='rev',package_versions=packages,runtime_identity=identity)
            receipts=PreflightReceipts(root)
            receipts.save(WHISPER_ID,self._schema3_receipt(manifest,snapshot,identity))
            current=identity
            with patch("speech_backends.resolve_snapshot",return_value=('rev',snapshot)), patch.object(receipts,'_current_runtime_identity',side_effect=lambda _stable: current):
                self.assertTrue(receipts.valid_manifest(manifest))
                for mutate in (
                    lambda value: value['python'].__setitem__('version','rotated-python'),
                    lambda value: value['packages'].__setitem__('numpy','rotated-package'),
                    lambda value: value['ffmpeg'].__setitem__('sha256','0'*64),
                    lambda value: value['ffmpeg'].__setitem__('version_first_line','ffmpeg version rotated'),
                ):
                    current=json.loads(json.dumps(identity)); mutate(current)
                    self.assertFalse(receipts.valid_manifest(manifest))
                current=identity; self.assertTrue(receipts.valid_manifest(manifest))
            ffmpeg.write_text('#!/bin/sh\nprintf "ffmpeg version Sotto-two\\n"\n'); os.chmod(ffmpeg,0o700)
            rotated=_candidate_runtime_identity(WHISPER_ID,ffmpeg_path=ffmpeg,package_versions=packages)
            self.assertNotEqual(identity['ffmpeg']['sha256'],rotated['ffmpeg']['sha256'])
            self.assertNotEqual(identity['ffmpeg']['version_sha256'],rotated['ffmpeg']['version_sha256'])

    @unittest.skipUnless(sys.platform == "darwin", "adaptive lane is macOS-only in v1 (ffmpeg/venv runtime identity)")
    def test_candidate_snapshot_confinement_rejects_escape_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); home=root/'hf'; snapshots=home/'hub'/'models--mlx-community--whisper-large-v3-turbo'/'snapshots'
            valid=snapshots/'rev'; valid.mkdir(parents=True); (valid/'model').write_text('safe')
            self.assertEqual(resolve_snapshot(WHISPER_REPO,home,'rev'),('rev',valid))
            for revision in ('../rev','/absolute','refs/main'):
                with self.subTest(revision=revision):
                    with self.assertRaises(RuntimeError): resolve_snapshot(WHISPER_REPO,home,revision)
            external=root/'external'; external.mkdir(); (external/'model').write_text('outside')
            (snapshots/'linked').symlink_to(external,target_is_directory=True)
            with self.assertRaises(RuntimeError): resolve_snapshot(WHISPER_REPO,home,'linked')
            nested=valid/'nested-link'; nested.symlink_to(external/'model')
            with self.assertRaises(RuntimeError): resolve_snapshot(WHISPER_REPO,home,'rev')
            nested.unlink()
            linked_home=root/'linked-hf'; linked_home.symlink_to(home,target_is_directory=True)
            with self.assertRaises(RuntimeError): resolve_snapshot(WHISPER_REPO,linked_home,'rev')
            identity=_candidate_runtime_identity(WHISPER_ID)
            manifest=candidate_manifest(WHISPER_ID,revision='rev',runtime_identity=identity)
            receipts=PreflightReceipts(root); receipts.save(WHISPER_ID,{'schema':3,'success':True,
                'manifest':{key:value for key,value in manifest.items() if key != 'glossary_terms'},'snapshot_path':str(external/'rev'),
                'snapshot_revision':'rev','snapshot_metadata_digest':'x','package_versions':_package_versions(),
                'runtime_identity':identity,'evaluator_id':'sotto-paired-v1','evaluator_hash':evaluator_hash()})
            # A receipt may name neither an arbitrary sibling nor a symlinked
            # cache root, even if its leaf happens to be a valid directory.
            (external/'rev').mkdir()
            with patch.dict(os.environ,{'SOTTO_HF_HOME':str(home)},clear=False), \
                 patch.object(receipts,'_current_runtime_identity',return_value=identity):
                self.assertFalse(receipts.valid_manifest(manifest))
                manager=LocalBackendManager(root)
                with self.assertRaises(RuntimeError): manager._exact_snapshot({**manifest,'snapshot_path':str(external/'rev')})
                self.assertEqual(manager._exact_snapshot({**manifest,'snapshot_path':str(valid)}),valid)


if __name__ == "__main__":
    unittest.main()

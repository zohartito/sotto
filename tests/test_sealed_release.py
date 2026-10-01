from __future__ import annotations

import hashlib
import json
import os
import plistlib
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import runtime_source_manifest as source
import sealed_release_bootstrap as bootstrap
import sealed_release as release
from sealed_release import (_release_body, activate_release, prepare_logs, rollback_release,
                            runtime_closure, stage_release, verify_release)


class SealedReleaseTests(unittest.TestCase):
    def setUp(self):
        # bootstrap.main() intentionally forces offline flags into its process
        # environment; in-process test invocations must not leak that into
        # later test modules.
        self.enterContext(patch.dict(os.environ, {}, clear=False))

    @staticmethod
    def _fixture(root: Path) -> None:
        (root / "nested").mkdir(parents=True); (root / "alpha.py").write_bytes(b"alpha"); (root / "nested" / "beta.py").write_bytes(b"beta")

    def test_isolated_runtime_probe_rejects_added_or_replaced_shadow_before_import(self):
        """Root mismatch is checked before the child can import packaging."""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); site=root/"site-packages"; site.mkdir(); sentinel=root/"executed"
            frozen=release._sealed_import_root(site)
            policy={"distributions":[],"roots":[],"excluded":[]}
            for body in (f"open({str(sentinel)!r},'w').write('boom')\n", "# replacement\n"):
                shadow=site/"packaging.py"; shadow.write_text(body,encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError,"release runtime probe failed"):
                    release._probe_runtime(Path(sys.executable),policy=policy,import_roots=[frozen])
                self.assertFalse(sentinel.exists())
                shadow.unlink()

    def test_bootstrap_runtime_probe_rejects_shadow_before_packaging_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); site=root/"site-packages"; site.mkdir(); sentinel=root/"executed"; frozen=bootstrap._sealed_import_root(site)
            shadow=site/"packaging.py"; shadow.write_text(f"open({str(sentinel)!r},'w').write('boom')\n")
            with self.assertRaises(RuntimeError):
                bootstrap._runtime_probe(Path(sys.executable),policy={"distributions":[],"roots":[],"excluded":[]},expected=None,import_roots=[frozen])
            self.assertFalse(sentinel.exists())

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (launchd + macOS venv layout)")
    def test_content_address_modes_tamper_and_path_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); left = root / "left"; right = root / "right"; self._fixture(left); self._fixture(right)
            closure = {"schema": 1, "fixture": "runtime"}; validator = lambda value: value == closure
            with patch.object(source, "RUNTIME_SOURCE_FILES", ("alpha.py", "nested/beta.py")):
                first = stage_release(root / "releases-a", source_root=left, runtime=closure)
                same = stage_release(root / "releases-b", source_root=right, runtime=closure)
                self.assertEqual(first.name, same.name)
                manifest = verify_release(first, runtime_validator=validator)
                self.assertEqual(manifest["source_manifest"]["digest"], source.runtime_source_manifest(left)["digest"])
                self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE((first / "source" / "alpha.py").stat().st_mode), 0o600)
                (first / "source" / "alpha.py").write_bytes(b"tampered")
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    verify_release(first, runtime_validator=validator)
                untouched = stage_release(root / "releases-c", source_root=right, runtime=closure)
                os.chmod(untouched / "source" / "alpha.py", 0o644)
                with self.assertRaisesRegex(RuntimeError, "unsafe"):
                    verify_release(untouched, runtime_validator=validator)
                self.assertEqual(stat.S_IMODE((untouched / "source" / "alpha.py").stat().st_mode), 0o644)
                os.chmod(untouched / "source" / "alpha.py", 0o600)
                (untouched / "source" / "shadow.py").write_text("shadow")
                with self.assertRaisesRegex(RuntimeError, "extra"):
                    verify_release(untouched, runtime_validator=validator)

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (launchd + macOS venv layout)")
    def test_activation_lkg_rollback_logs_and_template_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source_root = root / "source"; self._fixture(source_root); releases = root / "releases"; closure = {"schema": 1, "fixture": "runtime"}; validator = lambda value: value == closure
            with patch.object(source, "RUNTIME_SOURCE_FILES", ("alpha.py", "nested/beta.py")):
                first = stage_release(releases, source_root=source_root, runtime=closure); activate_release(releases, first.name, runtime_validator=validator)
                (source_root / "alpha.py").write_bytes(b"rotated")
                second = stage_release(releases, source_root=source_root, runtime=closure); activate_release(releases, second.name, runtime_validator=validator)
                self.assertEqual(os.readlink(releases / "current"), second.name)
                self.assertEqual(os.readlink(releases / "lkg"), first.name)
                pointers={name:((releases/name).readlink(),(releases/name).lstat().st_mtime_ns) for name in ("current","lkg")}
                self.assertEqual(activate_release(releases, second.name, runtime_validator=validator),second)
                self.assertEqual({name:((releases/name).readlink(),(releases/name).lstat().st_mtime_ns) for name in ("current","lkg")},pointers)
                self.assertEqual(bootstrap.verify(releases)[0], second)
                self.assertEqual(rollback_release(releases, runtime_validator=validator), first)
                self.assertEqual(os.readlink(releases / "current"), first.name)
            logs = prepare_logs(root / "Logs" / "Sotto")
            self.assertEqual(set(logs), {"app.log", "app-error.log", "worker.log", "worker-error.log"})
            self.assertTrue(all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in logs.values()))
            templates=[]
            for template, entry, log_name in (("launchd/com.zohartito.sotto.app.plist", "sotto.py", "app.log"),
                                              ("launchd/com.zohartito.sotto.adaptive-silver.plist", "adaptive_worker.py", "worker.log")):
                value = plistlib.loads((Path(__file__).parents[1] / template).read_bytes())
                templates.append(value)
                self.assertEqual(value["ProgramArguments"][0], "/usr/bin/python3")
                self.assertEqual(value["ProgramArguments"][1:4], ["-I","-S","-B"])
                self.assertIn("sealed_release_bootstrap.py", value["ProgramArguments"][4])
                self.assertIn(entry, value["ProgramArguments"])
                self.assertTrue(value["StandardOutPath"].endswith("/Sotto/" + log_name))
                self.assertNotIn("projects/sotto", "\0".join(value["ProgramArguments"]))
            self.assertEqual(templates[0]["ProgramArguments"][-7:], ["run", "--trigger", "right-option", "--idle-release", "0", "--glossary", "__SOTTO_HOME__/Library/Application Support/sotto/releases/current/source/config/sotto-glossary.txt"])
            self.assertNotIn("--language", templates[0]["ProgramArguments"])
            self.assertEqual(templates[1]["ProgramArguments"][-3:], ["--base-dir", "__SOTTO_HOME__/Library/Application Support/sotto", "--serve"])
            self.assertEqual(templates[0]["KeepAlive"],{"SuccessfulExit":False})
            self.assertEqual(templates[0]["ThrottleInterval"],60)
            expected_env={"SOTTO_HF_HOME":"__SOTTO_HOME__/Library/Application Support/sotto/huggingface", "HF_HOME":"__SOTTO_HOME__/Library/Application Support/sotto/huggingface", "SOTTO_FFMPEG":"__SOTTO_FFMPEG__", "SOTTO_OFFLINE":"1", "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "PYTHONDONTWRITEBYTECODE":"1"}
            self.assertTrue(all({key:value["EnvironmentVariables"][key] for key in expected_env} == expected_env for value in templates))
            self.assertFalse((Path(__file__).parents[1] / "launchd/com.sotto.adaptive-silver.plist").exists())

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (launchd + macOS venv layout)")
    def test_bootstrap_refuses_manifest_alias_and_runtime_package_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source_root=root/"source"; self._fixture(source_root); releases=root/"releases"; closure={"schema":1,"fixture":"runtime"}
            with patch.object(source,"RUNTIME_SOURCE_FILES",("alpha.py","nested/beta.py")):
                bundle=stage_release(releases,source_root=source_root,runtime=closure); activate_release(releases,bundle.name,runtime_validator=lambda value:value==closure)
                alias=releases/("0"*64); os.rename(bundle,alias); os.unlink(releases/"current"); os.symlink(alias.name,releases/"current")
                with self.assertRaisesRegex(RuntimeError,"identity"):
                    bootstrap.verify(releases)
            ffmpeg=root/"ffmpeg"; ffmpeg.write_text("#!/bin/sh\nprintf 'ffmpeg version fixture\\n'\n"); os.chmod(ffmpeg,0o700)
            python=Path(sys.executable).resolve(); binary=hashlib.sha256(python.read_bytes()).hexdigest()
            observed={"python":{"version":"v","executable":"e","implementation":"i","unicode_version":"u","prefix":"p","base_prefix":"b"},"packages":{"numpy":{"version":"1","files":[{"path":"x","sha256":"a"}]},"mlx-whisper":None,"parakeet-mlx":None,"huggingface-hub":None}}
            resolved_ffmpeg=ffmpeg.resolve()
            runtime={"python":{"configured_path":str(python),"resolved_path":str(python),"launcher_chain":[{"path":str(python),"sha256":binary}],"sha256":binary,**observed["python"]},"packages":observed["packages"],"distribution_policy":{"schema":1,"digest":"d","requirements_sha256":"r","roots":[],"excluded":["pip","setuptools","wheel"],"distributions":[{"name":name,"version":value["version"]} for name,value in observed["packages"].items() if value is not None]},"ffmpeg":{"configured_path":str(ffmpeg),"resolved_path":str(resolved_ffmpeg),"sha256":hashlib.sha256(resolved_ffmpeg.read_bytes()).hexdigest(),"version_first_line":"ffmpeg version fixture","version_sha256":hashlib.sha256(b"ffmpeg version fixture").hexdigest()}}
            site=Path(sys.prefix)/"lib"/f"python{sys.version_info.major}.{sys.version_info.minor}"/"site-packages"; runtime["import_roots"]=[bootstrap._sealed_import_root(site)]
            with patch.object(bootstrap,"_runtime_probe",return_value=observed):
                self.assertEqual(bootstrap._verify_runtime(runtime),python)
                drift={**observed,"packages":{**observed["packages"],"huggingface-hub":{"version":"2","files":[]}}}
                with patch.object(bootstrap,"_runtime_probe",return_value=drift), self.assertRaisesRegex(RuntimeError,"closure"):
                    bootstrap._verify_runtime(runtime)
            with patch.object(bootstrap,"_canonical_isolation",return_value=True), patch.object(bootstrap,"verify",return_value=(root,{"runtime":runtime,"entrypoints":{"worker":"adaptive_worker.py"}})), \
                 patch.object(bootstrap,"_verify_runtime",return_value=python), self.assertRaisesRegex(RuntimeError,"declared"):
                bootstrap.main(["--release-root",str(root),"--entry","shadow.py"])

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (launchd + macOS venv layout)")
    def test_bootstrap_forwards_exact_remainder(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source_root=root/"source"; source_root.mkdir()
            recorded=root/"argv.json"
            body=f"import json,sys,pathlib\npathlib.Path({str(recorded)!r}).write_text(json.dumps(sys.argv))\n"
            entry=source_root/"sotto.py"; entry.write_text(body); os.chmod(entry,0o600)
            runtime=Path(sys.executable).resolve()
            site=Path(sys.prefix)/"lib"/f"python{sys.version_info.major}.{sys.version_info.minor}"/"site-packages"
            declared={"source_manifest":{"files":[{"path":"sotto.py","sha256":hashlib.sha256(body.encode()).hexdigest()}]}}
            saved_main=sys.modules["__main__"]; saved_argv=list(sys.argv); saved_path=list(sys.path)
            try:
                with patch.object(bootstrap,"_canonical_isolation",return_value=True), patch.object(bootstrap,"verify",return_value=(root,{**declared,"runtime":{"python":{},"import_roots":[bootstrap._sealed_import_root(site)]},"entrypoints":{"app":"sotto.py"}})), \
                     patch.object(bootstrap,"_verify_runtime",return_value=runtime), \
                     patch.object(bootstrap,"_running_pinned_launcher",return_value=True):
                    bootstrap.main(["--release-root",str(root),"--entry","sotto.py","run","--trigger","right-option","--idle-release","45","--language","en","--adaptive"])
            finally:
                sys.modules["__main__"]=saved_main; sys.argv[:]=saved_argv; sys.path[:]=saved_path
            self.assertEqual(json.loads(recorded.read_text())[1:], ["run","--trigger","right-option","--idle-release","45","--language","en","--adaptive"])

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (launchd + macOS venv layout)")
    @unittest.skipIf(sys.version_info < (3, 14),
                     "sealed -I -S venv integration requires Python 3.14; alpha foreground uses 3.12")
    def test_configured_venv_launcher_is_probed_and_bootstrapped_without_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); launcher=Path(sys.executable); resolved=launcher.resolve()
            self.assertNotEqual(subprocess.run([str(resolved),"-B","-c","import huggingface_hub"],text=True,capture_output=True).returncode,0)
            source_root=root/"source"; source_root.mkdir()
            shutil.copy2(Path(__file__).parents[1]/"sealed_release_bootstrap.py",source_root/"sealed_release_bootstrap.py")
            (source_root/"sotto.py").write_text("import huggingface_hub, os, sys\nprint('sealed-prefix='+sys.prefix)\nprint('offline='+os.environ.get('HF_HUB_OFFLINE','missing'))\nprint(huggingface_hub.__name__)\n")
            (source_root/"adaptive_worker.py").write_text("import huggingface_hub, os, sys\nprint('sealed-prefix='+sys.prefix)\nprint('offline='+os.environ.get('HF_HUB_OFFLINE','missing'))\nprint('sealed-worker='+huggingface_hub.__name__)\n")
            ffmpeg=root/"ffmpeg"; ffmpeg.write_text("#!/bin/sh\nprintf 'ffmpeg version fixture\\n'\n"); os.chmod(ffmpeg,0o700)
            closure=runtime_closure(python=launcher,ffmpeg=ffmpeg)
            self.assertTrue(all(len(import_root["marker"]) == 4 and
                                all(len(entry["marker"]) == 4 for entry in import_root["entries"])
                                for import_root in closure["import_roots"]))
            self.assertTrue(all(len(entry["marker"]) == 4
                                for package in closure["packages"].values()
                                for entry in package["files"]))
            self.assertEqual(closure["python"]["configured_path"],str(launcher))
            self.assertEqual(closure["python"]["resolved_path"],str(resolved))
            self.assertNotEqual(closure["python"]["prefix"],closure["python"]["base_prefix"])
            self.assertTrue(closure["python"]["launcher_chain"])
            with patch.object(source,"RUNTIME_SOURCE_FILES",("sotto.py","adaptive_worker.py","sealed_release_bootstrap.py")):
                bundle=stage_release(root/"releases",source_root=source_root,runtime=closure)
                activate_release(root/"releases",bundle.name,runtime_validator=lambda value:value==closure)
            bootstrap_path=bundle/"source"/"sealed_release_bootstrap.py"
            before=sorted((path.relative_to(bundle/"source").as_posix(),path.read_bytes()) for path in (bundle/"source").rglob("*") if path.is_file())
            injection=root/"injection"; injection.mkdir(); sentinel=root/"startup-executed"
            for name in ("sitecustomize.py","usercustomize.py","shadow.py"):
                (injection/name).write_text(f"open({str(sentinel)!r},'w').write('boom')\n")
            (injection/"ignored.pth").write_text(f"import pathlib; pathlib.Path({str(sentinel)!r}).write_text('boom')\n")
            environment=dict(os.environ); environment.pop("PYTHONDONTWRITEBYTECODE",None)
            for name in ("SOTTO_OFFLINE","HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE"): environment.pop(name,None)
            environment["PYTHONPATH"]=str(injection)
            for entry,marker in (("sotto.py","huggingface_hub"),("adaptive_worker.py","sealed-worker=huggingface_hub")):
                command=[sys.executable,"-I","-S","-B",str(bootstrap_path),"--release-root",str(root/"releases"),"--entry",entry]
                for _ in range(2):
                    result=subprocess.run(command,text=True,capture_output=True,env=environment,check=True)
                    self.assertIn("sealed-prefix="+sys.prefix,result.stdout); self.assertIn(marker,result.stdout)
                    # The bootstrap itself forces the offline flags even when
                    # the launching environment does not provide them.
                    self.assertIn("offline=1",result.stdout)
            self.assertFalse(sentinel.exists())
            after=sorted((path.relative_to(bundle/"source").as_posix(),path.read_bytes()) for path in (bundle/"source").rglob("*") if path.is_file())
            self.assertEqual(after,before)
            self.assertFalse(any("__pycache__" in path.as_posix() for path in (bundle/"source").rglob("*")))

    def test_pycache_directory_timestamp_churn_is_tolerated_but_bytes_are_not(self):
        """Interpreter temp files churn only __pycache__ dir timestamps; a
        frozen root must survive that while still binding every pyc byte."""
        with tempfile.TemporaryDirectory() as directory:
            site=Path(directory)/"site"; cache=site/"pkg"/"__pycache__"
            cache.mkdir(parents=True); pyc=cache/"mod.cpython-314.pyc"; pyc.write_bytes(b"bytecode")
            for module in (release,bootstrap):
                frozen=module._sealed_import_root(site)
                temp=cache/".tmp-transient"; temp.write_bytes(b"x"); temp.unlink()
                self.assertEqual(module._sealed_import_root(site),frozen)
                pyc.write_bytes(b"tampered")
                self.assertNotEqual(module._sealed_import_root(site),frozen)
                pyc.write_bytes(b"bytecode")
                extra=cache/"added.pyc"; extra.write_bytes(b"new")
                self.assertNotEqual(module._sealed_import_root(site),frozen)
                extra.unlink()

    def test_import_root_identity_survives_cross_boot_device_renumbering(self):
        """APFS device IDs may change after reboot without any byte changing."""
        with tempfile.TemporaryDirectory() as directory:
            site=Path(directory)/"site"; package=site/"pkg"; package.mkdir(parents=True)
            (package/"module.py").write_text("value = 1\n")
            real_lstat=os.lstat

            def remounted(path):
                info=real_lstat(path)
                return SimpleNamespace(
                    st_mode=info.st_mode, st_dev=info.st_dev + 1,
                    st_ino=info.st_ino, st_size=info.st_size,
                    st_mtime_ns=info.st_mtime_ns, st_ctime_ns=info.st_ctime_ns,
                )

            for module in (release,bootstrap):
                frozen=module._sealed_import_root(site)
                with patch.object(module.os,"lstat",side_effect=remounted):
                    self.assertEqual(module._sealed_import_root(site),frozen)

            (package/"module.py").write_text("value = 2\n")
            self.assertNotEqual(release._sealed_import_root(site),frozen)

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (launchd + macOS venv layout)")
    def test_mutation_during_probe_fails_before_any_execution(self):
        """Import-root, target and bootstrap mutation during the probe window
        each fail closed before any sealed application byte can execute."""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); sentinel=root/"sentinel-app"
            source_root=root/"source"; source_root.mkdir()
            (source_root/"sotto.py").write_text(f"open({str(sentinel)!r},'w').write('boom')\n")
            shutil.copy2(Path(__file__).parents[1]/"sealed_release_bootstrap.py",source_root/"sealed_release_bootstrap.py")
            site=root/"site"; site.mkdir(); (site/"module.py").write_text("v1\n")
            closure={"schema":1,"python":{},"import_roots":[bootstrap._sealed_import_root(site)]}
            releases=root/"releases"
            with patch.object(source,"RUNTIME_SOURCE_FILES",("sotto.py","sealed_release_bootstrap.py")):
                bundle=stage_release(releases,source_root=source_root,runtime=closure)
                activate_release(releases,bundle.name,runtime_validator=lambda value:value==closure)
            checked_bundle,checked_manifest=bootstrap.verify(releases)
            python=Path(sys.executable)
            def mutate(path,data):
                with open(path,"r+b") as handle: handle.seek(0); handle.truncate(); handle.write(data)
            execs=[]; argv=["--release-root",str(releases),"--entry","sotto.py"]
            # Target source mutated while the runtime probe runs: the dedicated
            # manifest-hash compare of the exact bytes to be executed rejects
            # it before execution (main executes only the compared bytes).
            original_target=(bundle/"source"/"sotto.py").read_bytes()
            with patch.object(bootstrap,"_canonical_isolation",return_value=True), \
                 patch.object(bootstrap,"verify",return_value=(checked_bundle,checked_manifest)), \
                 patch.object(bootstrap,"_verify_runtime",side_effect=lambda _r:(mutate(bundle/"source"/"sotto.py",b"tampered\n"),python)[1]), \
                 patch.object(bootstrap,"_running_pinned_launcher",return_value=True):
                with self.assertRaisesRegex(RuntimeError,"sealed target changed"):
                    bootstrap.main(argv)
            mutate(bundle/"source"/"sotto.py",original_target)
            # Bootstrap source mutated while the probe runs: rejected before execve.
            original_bootstrap=(bundle/"source"/"sealed_release_bootstrap.py").read_bytes()
            with patch.object(bootstrap,"_canonical_isolation",return_value=True), \
                 patch.object(bootstrap,"verify",return_value=(checked_bundle,checked_manifest)), \
                 patch.object(bootstrap,"_verify_runtime",side_effect=lambda _r:(mutate(bundle/"source"/"sealed_release_bootstrap.py",b"tampered\n"),python)[1]), \
                 patch.object(bootstrap,"_running_pinned_launcher",return_value=False), \
                 patch("sealed_release_bootstrap.os.execve",side_effect=lambda *a: execs.append(a)):
                with self.assertRaisesRegex(RuntimeError,"sealed bootstrap changed"):
                    bootstrap.main(argv)
            mutate(bundle/"source"/"sealed_release_bootstrap.py",original_bootstrap)
            # Import root mutated DURING the final sealed-tree verification:
            # the root recheck runs last, immediately before insertion, and
            # still rejects it before any application byte executes.
            real_verify=bootstrap.verify; verify_calls=[]
            def verify_then_mutate(releases_root):
                result=real_verify(releases_root); verify_calls.append(1)
                if len(verify_calls) == 2: mutate(site/"module.py",b"mutated\n")
                return result
            with patch.object(bootstrap,"_canonical_isolation",return_value=True), \
                 patch.object(bootstrap,"verify",side_effect=verify_then_mutate), \
                 patch.object(bootstrap,"_verify_runtime",return_value=python), \
                 patch.object(bootstrap,"_running_pinned_launcher",return_value=True):
                with self.assertRaisesRegex(RuntimeError,"import roots changed"):
                    bootstrap.main(argv)
            # A concurrently switched ``current`` pointer cannot mix releases:
            # the runtime closure was verified for the pinned digest only, so
            # any mid-start switch to another release is refused outright.
            other_source=root/"source-b"; other_source.mkdir()
            (other_source/"sotto.py").write_text(f"open({str(root/'sentinel-b')!r},'w').write('boom')\n")
            shutil.copy2(Path(__file__).parents[1]/"sealed_release_bootstrap.py",other_source/"sealed_release_bootstrap.py")
            closure_b={"schema":1,"python":{}}
            with patch.object(source,"RUNTIME_SOURCE_FILES",("sotto.py","sealed_release_bootstrap.py")):
                bundle_b=stage_release(releases,source_root=other_source,runtime=closure_b)
            def switch(_runtime):
                activate_release(releases,bundle_b.name); return python
            with patch.object(bootstrap,"_canonical_isolation",return_value=True), \
                 patch.object(bootstrap,"_verify_runtime",side_effect=switch), \
                 patch.object(bootstrap,"_running_pinned_launcher",return_value=True):
                with self.assertRaisesRegex(RuntimeError,"changed during startup"):
                    bootstrap.main(argv)
            self.assertFalse((root/"sentinel-b").exists())
            self.assertEqual(execs,[]); self.assertFalse(sentinel.exists())
            # The runtime-closure validator itself also recomputes the exact
            # import-root trees after the isolated child probe returns.
            site0=root/"site0"; site0.mkdir(); (site0/"module.py").write_text("v1\n")
            resolved=Path(sys.executable).resolve(); binary=hashlib.sha256(resolved.read_bytes()).hexdigest()
            ffmpeg=root/"ffmpeg"; ffmpeg.write_text("#!/bin/sh\nprintf 'ffmpeg version fixture\\n'\n"); os.chmod(ffmpeg,0o700)
            observed={"python":{"version":"v","executable":"e","implementation":"i","unicode_version":"u","prefix":"p","base_prefix":"b"},"packages":{}}
            runtime={"python":{"configured_path":str(resolved),"resolved_path":str(resolved),"launcher_chain":[{"path":str(resolved),"sha256":binary}],"sha256":binary,**observed["python"]},
                     "packages":{},"import_roots":[bootstrap._sealed_import_root(site0)],
                     "distribution_policy":{"schema":1,"digest":"d","requirements_sha256":"r","roots":[],"excluded":["pip","setuptools","wheel"],"distributions":[]},
                     "ffmpeg":{"configured_path":str(ffmpeg),"resolved_path":str(ffmpeg.resolve()),"sha256":hashlib.sha256(ffmpeg.resolve().read_bytes()).hexdigest(),"version_first_line":"ffmpeg version fixture","version_sha256":hashlib.sha256(b"ffmpeg version fixture").hexdigest()}}
            with patch.object(bootstrap,"_runtime_probe",side_effect=lambda *a,**k:(mutate(site0/"module.py",b"mutated\n"),observed)[1]):
                with self.assertRaisesRegex(RuntimeError,"closure changed"):
                    bootstrap._verify_runtime(runtime)

    def test_declared_provisioning_and_calibration_entrypoints_are_allowlisted(self):
        declared={"teacher_provision.py","calibration_manifest_builder.py","calibration_worker.py"}
        self.assertTrue(declared.issubset(set(source.RUNTIME_SOURCE_FILES)))
        self.assertTrue(declared.issubset(set(_release_body({}, {})["entrypoints"].values())))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); entry=root/"entry.py"; entry.write_text("pass\n"); os.chmod(entry,0o600)
            with patch.object(bootstrap,"_canonical_isolation",return_value=True), patch.object(bootstrap,"verify",return_value=(root,{"runtime":{},"entrypoints":{"teacher":"teacher_provision.py"}})), \
                 patch.object(bootstrap,"_verify_runtime",return_value=Path(sys.executable)), \
                 patch.object(bootstrap,"_running_pinned_launcher",return_value=True):
                with self.assertRaisesRegex(RuntimeError,"declared"):
                    bootstrap.main(["--release-root",str(root),"--entry","calibration_worker.py"])

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (launchd + macOS venv layout)")
    def test_runtime_policy_rejects_missing_direct_and_extra_locked_dependency(self):
        missing={"roots":["sotto-required-not-installed"],"excluded":["pip","setuptools","wheel"],"distributions":[{"name":"sotto-required-not-installed","version":"1"}]}
        with self.assertRaisesRegex(RuntimeError,"probe"):
            release._probe_runtime(Path(sys.executable),policy=missing)
        policy=release._runtime_policy()
        extra={"roots":policy["roots"],"excluded":policy["excluded"],"distributions":[*policy["distributions"],{"name":"sotto-unexpected-policy-dependency","version":"1"}]}
        with self.assertRaisesRegex(RuntimeError,"probe"):
            release._probe_runtime(Path(sys.executable),policy=extra)


if __name__ == "__main__":
    unittest.main()

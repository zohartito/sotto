"""Windows source alpha: storage paths, pinned dependency sets, source archive."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATHS_CODE = ("import json, sotto_paths as p; "
              "print(json.dumps([str(p.DATA_DIR), str(p.MODEL_CACHE_DIR)]))")


def _paths(env_updates: dict[str, str | None], cwd: str) -> list[str]:
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    for name, value in env_updates.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    result = subprocess.run([sys.executable, "-c", PATHS_CODE], env=env, cwd=cwd,
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


@unittest.skipUnless(sys.platform == "win32", "Windows storage defaults")
class WindowsPathTests(unittest.TestCase):
    def test_default_is_the_pre_alpha_appdata_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            roaming = Path(temporary) / "Roaming"
            data, cache = _paths({"SOTTO_DATA_DIR": "  ", "SOTTO_HF_HOME": "",
                                  "APPDATA": str(roaming)}, cwd=temporary)
            self.assertEqual(data, str(roaming / "sotto"))
            self.assertEqual(cache, str(roaming / "sotto" / "huggingface"))

    def test_missing_or_blank_appdata_falls_back_to_the_profile_never_cwd(self):
        with tempfile.TemporaryDirectory() as temporary:
            expected = Path.home() / "AppData" / "Roaming" / "sotto"
            for appdata in (None, " "):
                data, cache = _paths({"SOTTO_DATA_DIR": None, "SOTTO_HF_HOME": None,
                                      "APPDATA": appdata}, cwd=temporary)
                self.assertEqual(data, str(expected))
                self.assertEqual(cache, str(expected / "huggingface"))
                self.assertFalse(os.path.realpath(data).startswith(os.path.realpath(temporary)))

    def test_overrides_are_independent_and_absolute(self):
        with tempfile.TemporaryDirectory() as temporary:
            alpha = Path(temporary) / "sotto-alpha"
            self.assertEqual(_paths({"SOTTO_DATA_DIR": str(alpha), "SOTTO_HF_HOME": None}, cwd=temporary),
                             [str(alpha), str(alpha / "huggingface")])
            self.assertEqual(_paths({"SOTTO_DATA_DIR": str(alpha), "SOTTO_HF_HOME": str(alpha / "models")},
                                    cwd=temporary), [str(alpha), str(alpha / "models")])
            data, cache = _paths({"SOTTO_DATA_DIR": "relative-state", "SOTTO_HF_HOME": "~/hf"}, cwd=temporary)
            self.assertEqual(data, str(Path(temporary) / "relative-state"))
            self.assertEqual(cache, str(Path.home() / "hf"))

    def test_every_windows_consumer_follows_the_override_without_creating_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = dict(os.environ, SOTTO_DATA_DIR=str(root / "state"), SOTTO_HF_HOME=str(root / "cache"))
            code = ("import json, os, history, speech_config, vad, sotto, sotto_win; "
                    "print(json.dumps([str(history.STORE_DIR), str(speech_config.DEFAULT_LANGUAGE_MODE_PATH), "
                    "str(vad.MODEL_PATH), str(sotto.SOTTO_HF_HOME), os.environ['HF_HUB_CACHE']]))")
            result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                    check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout), [
                str(root / "state"), str(root / "state" / "language-mode"),
                str(root / "state" / "models" / "silero_vad.onnx"), str(root / "cache"),
                str(root / "cache" / "hub")])
            self.assertFalse((root / "state").exists(), "import must not create state")
            self.assertFalse((root / "cache").exists(), "import must not create a cache")


def _requirement_names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text("utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            names.add(re.sub(r"[-_.]+", "-", line).lower())
    return names


def _lock(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text("utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.+!-]+)", line)
        if match is None:
            raise AssertionError(f"{path.name}: not an exact name==version pin: {line!r}")
        pins[re.sub(r"[-_.]+", "-", match.group(1)).lower()] = match.group(2)
    return pins


class WindowsDependencyPinTests(unittest.TestCase):
    def test_locks_are_exact_disjoint_and_cover_every_direct_requirement(self):
        cpu = _lock(ROOT / "requirements-alpha-windows.lock")
        cuda = _lock(ROOT / "requirements-alpha-windows-cuda.lock")
        self.assertFalse(set(cpu) & set(cuda), "the CUDA lock only adds packages")
        self.assertLessEqual(_requirement_names(ROOT / "requirements-alpha-windows.txt"), set(cpu))
        self.assertLessEqual(_requirement_names(ROOT / "requirements-alpha-windows-cuda.txt"), set(cuda))
        self.assertIn("ctranslate2", cpu)
        self.assertIn("nvidia-cublas-cu12", cuda)

    def test_requirement_files_reference_only_local_pins(self):
        cpu = (ROOT / "requirements-alpha-windows.txt").read_text("utf-8")
        cuda = (ROOT / "requirements-alpha-windows-cuda.txt").read_text("utf-8")
        self.assertIn("-c requirements-alpha-windows.lock", cpu.splitlines())
        self.assertIn("-r requirements-alpha-windows.txt", cuda.splitlines())
        self.assertIn("-c requirements-alpha-windows-cuda.lock", cuda.splitlines())
        for text in (cpu, cuda, (ROOT / "requirements-alpha-windows.lock").read_text("utf-8"),
                     (ROOT / "requirements-alpha-windows-cuda.lock").read_text("utf-8")):
            self.assertNotRegex(text, r"(?i)https?://|file:|@|--index-url|--extra-index-url|--find-links")
        cpu_packages = (_requirement_names(ROOT / "requirements-alpha-windows.txt")
                        | set(_lock(ROOT / "requirements-alpha-windows.lock")))
        self.assertFalse([name for name in cpu_packages if name.startswith("nvidia")],
                         "CPU testers must not download CUDA packages")


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_alpha_source", ROOT / "scripts" / "build_alpha_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WindowsArchiveTests(unittest.TestCase):
    def test_windows_block_is_complete_and_the_archive_is_deterministic(self):
        builder = _load_builder()
        for name in builder.WINDOWS_FILES:
            self.assertIn(name, builder.PUBLIC_FILES)
            self.assertTrue((ROOT / name).is_file(), name)
        tracked_windows = {path.relative_to(ROOT).as_posix() for path in ROOT.glob("win_*.py")}
        tracked_windows |= {path.relative_to(ROOT).as_posix() for path in ROOT.glob("tests/test_win_*.py")}
        self.assertLessEqual(tracked_windows, set(builder.WINDOWS_FILES))
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "a.tar.gz", Path(temporary) / "b.tar.gz"
            self.assertEqual(builder.build(first)["sha256"], builder.build(second)["sha256"])
            with tarfile.open(first) as archive:
                names = archive.getnames()
        self.assertTrue(all(name.startswith("sotto-alpha/") for name in names))
        for forbidden in (".handoff/", "venv", ".git/", "huggingface", ".alpha-state", "__pycache__"):
            self.assertFalse([name for name in names if forbidden in name], forbidden)
        self.assertIn("sotto-alpha/docs/windows-alpha.md", names)


if __name__ == "__main__":
    unittest.main()

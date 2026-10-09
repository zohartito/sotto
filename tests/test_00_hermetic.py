"""Keeps a plain `python -m unittest discover -s tests` away from real user data (F64).

`sotto_paths` reads SOTTO_DATA_DIR / SOTTO_HF_HOME once, at its first import, and
without them it points at ~/Library/Application Support/sotto or %APPDATA%\\sotto.
Discovery imports test modules in sorted order, so this module loads before any
other test module and points both variables at a throwaway folder first. Every
in-process import of sotto_paths and every subprocess (they inherit os.environ)
then stays inside that folder. The tests below fail if that ever stops holding.
"""
import atexit
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SOTTO_PATHS_PRELOADED = "sotto_paths" in sys.modules
TEST_ROOT = Path(tempfile.mkdtemp(prefix="sotto-tests-"))
os.environ["SOTTO_DATA_DIR"] = str(TEST_ROOT / "data")
os.environ["SOTTO_HF_HOME"] = str(TEST_ROOT / "huggingface")
atexit.register(shutil.rmtree, TEST_ROOT, ignore_errors=True)


def _under(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _real_default_folders():
    home = Path.home()
    folders = [home / "Library/Application Support" / name for name in ("sotto", "sotto-alpha", "Sotto")]
    for variable, name in (("APPDATA", "sotto"), ("LOCALAPPDATA", "sotto-alpha")):
        if os.environ.get(variable, "").strip():
            folders.append(Path(os.environ[variable]) / name)
    return folders


class HermeticTestState(unittest.TestCase):
    def test_runs_before_anything_imports_sotto_paths(self):
        self.assertFalse(SOTTO_PATHS_PRELOADED, "sotto_paths was imported before tests/test_00_hermetic.py")

    def test_discovery_imports_this_module_first(self):
        importable = [name for name in sorted(os.listdir(HERE))
                      if (name.startswith("test") and name.endswith(".py")) or (HERE / name / "__init__.py").is_file()]
        self.assertEqual(importable[0], Path(__file__).name)

    def test_data_and_model_cache_are_temporary(self):
        import sotto_paths
        for path in (sotto_paths.DATA_DIR, sotto_paths.MODEL_CACHE_DIR):
            self.assertTrue(_under(path, TEST_ROOT), f"{path} is outside the temporary folder {TEST_ROOT}")
            for real in _real_default_folders():
                self.assertFalse(_under(path, real), f"{path} is inside the real data folder {real}")


if __name__ == "__main__":
    unittest.main()

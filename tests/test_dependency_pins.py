"""Pins that installers and the two alpha lock files must keep (F25, M6)."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]

# Places that bootstrap pip before installing the pinned set.
PIP_BOOTSTRAPS = ["scripts/install-mac.sh", "scripts/install-windows.ps1", "scripts/sotto-win.bat",
                  "README.md", "docs/windows-alpha.md", ".github/workflows/tests.yml"]
WORKFLOW = ".github/workflows/tests.yml"

# A package pinned in both the Mac and a Windows lock at different versions must
# be listed here with the reason; anything else is drift and must be aligned.
PLATFORM_DIFFERENCES = {
    "huggingface-hub": "Windows ASR needs faster-whisper -> tokenizers 0.23.2, which requires "
                       "huggingface-hub<2.0; the Mac set has no tokenizers and is locked on 2.0.0.",
}


def _pins(path: Path) -> dict:
    pins = {}
    for line in path.read_text("utf-8").splitlines():
        match = re.match(r"([A-Za-z0-9._-]+)==(\S+)", line.split("#")[0].strip())
        if match:
            pins[re.sub(r"[-_.]+", "-", match[1]).lower()] = match[2]
    return pins


class DependencyPinTests(unittest.TestCase):
    def test_pip_bootstraps_install_one_exact_pip(self):
        versions = set()
        for name in PIP_BOOTSTRAPS:
            text = (ROOT / name).read_text("utf-8")
            self.assertIsNone(re.search(r"--upgrade pip\b", text), f"{name} installs whatever pip is newest")
            found = re.findall(r"pip install (?:--[a-z-]+ )*pip==(\d+(?:\.\d+)+)\b", text)
            self.assertTrue(found, f"{name} no longer installs an exact pip")
            versions.update(found)
        self.assertEqual(len(versions), 1, f"pip versions differ: {sorted(versions)}")

    def test_ci_installs_each_pinned_set_with_the_installers_pip(self):
        # [N39] CI installed the lock files with whatever pip the runner's Python shipped.
        text = (ROOT / WORKFLOW).read_text("utf-8")
        steps = re.split(r"\n\s*- (?:name|uses):", text)
        installs = [step for step in steps if re.search(r"pip install (?:--[a-z-]+ )*-r requirements", step)]
        self.assertGreaterEqual(len(installs), 2, "the Mac and Windows jobs both install a pinned set")
        for step in installs:
            bootstrap = re.search(r"pip install (?:--[a-z-]+ )*pip==\d", step)
            pinned = re.search(r"pip install (?:--[a-z-]+ )*-r requirements", step)
            self.assertTrue(bootstrap and bootstrap.start() < pinned.start(),
                            f"{WORKFLOW} installs a pinned set before an exact pip:\n{step.strip()}")

    def test_mac_and_windows_locks_agree_or_say_why(self):
        mac = _pins(ROOT / "constraints-alpha.txt")
        differing = set()
        for lock in sorted(ROOT.glob("constraints-alpha-windows*.txt")):
            windows = _pins(lock)
            for name in sorted(mac.keys() & windows.keys()):
                if mac[name] != windows[name]:
                    differing.add(name)
                    self.assertIn(name, PLATFORM_DIFFERENCES,
                                  f"{name} is {mac[name]} on the Mac but {windows[name]} in {lock.name}")
        self.assertEqual(set(PLATFORM_DIFFERENCES) - differing, set(), "allowlisted packages that no longer differ")


if __name__ == "__main__":
    unittest.main()

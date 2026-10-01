"""Offline regression checks for executable GitHub Action provenance."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
USES = re.compile(r"^\s*(?:-\s*)?uses:\s*[^@\s]+@([^\s#]+)", re.MULTILINE)


class WorkflowActionPinningTests(unittest.TestCase):
    def test_executable_actions_are_immutable(self) -> None:
        for workflow in (ROOT / ".github" / "workflows").glob("*.y*ml"):
            for ref in USES.findall(workflow.read_text()):
                self.assertRegex(ref, r"^[0-9a-f]{40}$", workflow)

    def test_ci_remains_tombstoned_without_setup_python_provenance(self) -> None:
        self.assertFalse((ROOT / ".github" / "workflows" / "ci.yml").exists())
        tombstone = ROOT / ".github" / "workflows-disabled" / "ci.yml.disabled"
        self.assertTrue(tombstone.exists())
        self.assertIn("actions/setup-python@v7", tombstone.read_text())


if __name__ == "__main__":
    unittest.main()

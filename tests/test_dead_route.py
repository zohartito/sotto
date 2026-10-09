"""Exact-zero captures are a dead mic route, not quiet speech.

Reference capture dd45a27282ab (2026-09-20 10:38): 40.96s of 24 kHz PCM
that is bit-exact zeros (AirPods sample rate, no noise floor). Whisper
turned the collapsed 1.06s remnant into "ощ" x223. The hallucination
guard correctly refused to paste, but the garbage still filled history
and Retry re-ran ASR on the same silence.

A live mic always has a noise floor (CaptureService._tap). Quiet
whispered dictation is non-zero and must still reach the ASR
(2026-08-13/14). Exact zeros must not.
"""

from pathlib import Path
import unittest

import numpy as np

from sotto import (DEAD_ROUTE_TEXT, asr_skip_reason, is_dead_route,
                   looks_hallucinated, salvage_repetition_loop)


# the exact loop whisper produced on this machine's all-zero capture
OBSERVED_DEAD_TEXT = "ощ " * 222 + "ощ"
OBSERVED_DEAD_SECONDS = 1.06


class DeadRouteTest(unittest.TestCase):
    def test_all_zero_samples_are_a_dead_route(self):
        self.assertTrue(is_dead_route(np.zeros(16_000, dtype=np.float32)))
        self.assertEqual(asr_skip_reason(np.zeros(16_000, dtype=np.float32)),
                         "dead microphone (all-zero capture)")

    def test_empty_capture_is_a_dead_route(self):
        self.assertTrue(is_dead_route(np.zeros(0, dtype=np.float32)))

    def test_quiet_nonzero_samples_still_reach_asr(self):
        # whispered dictation sits well above exact zero; energy gates
        # on this class of audio are what ate captures in 2026-08-13/14
        quiet = np.full(16_000, 1e-5, dtype=np.float32)
        self.assertFalse(is_dead_route(quiet))
        self.assertIsNone(asr_skip_reason(quiet))

    def test_observed_oshch_loop_is_still_condemned_if_asr_runs(self):
        # defense in depth: if a near-silent capture ever reaches Whisper,
        # the existing guard must keep quarantining wall-to-wall garbage
        self.assertIsNotNone(looks_hallucinated(OBSERVED_DEAD_TEXT,
                                                OBSERVED_DEAD_SECONDS))
        self.assertIsNone(salvage_repetition_loop(OBSERVED_DEAD_TEXT,
                                                  OBSERVED_DEAD_SECONDS))

    def test_dead_route_placeholder_is_stable(self):
        self.assertEqual(DEAD_ROUTE_TEXT, "[dead microphone]")

    def test_live_and_retry_jobs_call_the_dead_route_guard(self):
        # Static check (run() cannot be called in a test yet): parse the code,
        # so a guard that is commented out no longer counts.
        import ast
        tree = ast.parse((Path(__file__).resolve().parents[1] / "sotto.py").read_text(encoding="utf-8"))
        arguments = {ast.unparse(node.args[0]) for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "asr_skip_reason"
                     and node.args}
        self.assertIn("samples", arguments)
        self.assertIn("snapshot.samples", arguments)

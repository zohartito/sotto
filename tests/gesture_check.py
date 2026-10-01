#!/usr/bin/env python3
"""GestureEngine acceptance checks (panel-derived scenarios).

Run:  venv/bin/python tests/gesture_check.py
Uses shrunken timing windows so the whole suite finishes in ~3 seconds.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sotto

sotto.HOLD_THRESHOLD_S = 0.08
sotto.DOUBLE_TAP_WINDOW_S = 0.12
HOLD = 0.10          # a press this long counts as push-to-talk
TAP = 0.02           # a press this short is a tap
GAP = 0.04           # gap between the taps of a double-tap
SETTLE = 0.20        # longer than the double-tap window


class Probe:
    def __init__(self, fail_start: bool = False):
        self.events: list[str] = []
        self.fail_start = fail_start
        self.engine = sotto.GestureEngine(self._start, self._finish, self._discard)

    def _start(self):
        if self.fail_start:
            raise RuntimeError("simulated capture failure")
        self.events.append("start")

    def _finish(self):
        self.events.append("finish")

    def _discard(self):
        self.events.append("discard")

    def press(self, seconds: float):
        self.engine.pressed()
        time.sleep(seconds)
        self.engine.released()


def check(name: str, got: list[str], expected: list[str]) -> bool:
    ok = got == expected
    print(f"{'✓' if ok else '✗'} {name}: {got}" + ("" if ok else f" (expected {expected})"))
    return ok


results = []

# 1. lone tap -> discarded after the window
p = Probe()
p.press(TAP)
time.sleep(SETTLE)
results.append(check("lone tap discarded", p.events, ["start", "discard"]))

# 2. double tap -> hands-free stays recording; next tap finishes
p = Probe()
p.press(TAP); time.sleep(GAP); p.press(TAP)
time.sleep(SETTLE)  # well past the window: hands-free must NOT discard
mid = list(p.events)
p.press(TAP)        # stop tap
time.sleep(0.05)
results.append(check("double-tap arms hands-free", mid, ["start"]))
results.append(check("hands-free stop tap finishes", p.events, ["start", "finish"]))

# 3. tap then quick HOLD -> push-to-talk finish, never a stuck hands-free mic
p = Probe()
p.press(TAP); time.sleep(GAP); p.press(HOLD)
time.sleep(SETTLE)
results.append(check("tap+hold is push-to-talk (mis-arm fixed)",
                     p.events, ["start", "finish"]))

# 4. plain hold -> push-to-talk
p = Probe()
p.press(HOLD)
time.sleep(0.05)
results.append(check("plain hold finishes", p.events, ["start", "finish"]))

# 5. stale blip then a fresh hold -> old blip discarded, new recording separate
p = Probe()
p.press(TAP)
time.sleep(SETTLE)          # discard timer fires
p.press(HOLD)
time.sleep(0.05)
results.append(check("stale blip never merges into next recording",
                     p.events, ["start", "discard", "start", "finish"]))

# 6. on_start failure resets state (no phantom recording)
p = Probe(fail_start=True)
p.engine.pressed()
time.sleep(0.02)
ok = not p.engine._recording and not p.engine._hands_free
print(f"{'✓' if ok else '✗'} capture failure resets engine state")
results.append(ok)
p.engine.released()  # must be a no-op, not a crash

# 7. force_reset drops an in-flight recording exactly once
p = Probe()
p.engine.pressed()
time.sleep(0.02)
p.engine.force_reset()
time.sleep(SETTLE)  # any stale timer must not fire a second discard
results.append(check("force_reset drops in-flight recording",
                     p.events, ["start", "discard"]))

print()
if all(results):
    print(f"ALL {len(results)} CHECKS PASSED")
else:
    print(f"{results.count(False)} of {len(results)} FAILED")
    sys.exit(1)

"""GestureEngine: the hands-free watchdog and the ordering of capture actions.

Real sotto.GestureEngine with threading.Timer replaced by a manual timer, so
every deadline fires exactly when the test says and on the thread the test
chooses. No event tap, no microphone, no wall-clock waits.

Findings covered (audit ledger ids): F5 the hands-free watchdog belongs to the
capture, not the key epoch; F6 start / finish / discard run in decision order
and never interleave across threads.
"""

import threading
import unittest
from unittest import mock

import sotto


class ManualTimers:
    """threading.Timer stand-in: arming records (delay, handler, args); the
    test fires what it wants, when it wants."""

    def __init__(self):
        self.armed = []

    def __call__(self, delay, handler, args=()):
        outer = self

        class Timer:
            daemon = False

            def start(self):
                outer.armed.append((delay, handler, tuple(args)))

        return Timer()

    def only(self, delay):
        matching = [t for t in self.armed if t[0] == delay]
        assert len(matching) == 1, matching
        _, handler, args = matching[0]
        return lambda: handler(*args)


class EngineHarness(unittest.TestCase):
    def setUp(self):
        self.timers = ManualTimers()
        for patcher in (mock.patch.object(sotto.threading, "Timer", self.timers),
                        mock.patch.object(sotto, "log", lambda msg: None)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.events = []
        self.engine = sotto.GestureEngine(lambda: self.events.append("start"),
                                          lambda: self.events.append("finish"),
                                          lambda: self.events.append("discard"))

    def arm_hands_free(self):
        """Two quick taps: the second one's release arms hands-free and its
        watchdog (HANDS_FREE_MAX_S); returns a callable that fires the watchdog."""
        before = len(self.timers.armed)
        self.engine.pressed(); self.engine.released()
        self.engine.pressed(); self.engine.released()
        self.assertEqual(self.engine.snapshot(), (True, True))
        watchdogs = [t for t in self.timers.armed[before:] if t[0] == sotto.HANDS_FREE_MAX_S]
        self.assertEqual(len(watchdogs), 1)
        _, handler, args = watchdogs[0]
        return lambda: handler(*args)


# -- F5: the watchdog belongs to the capture, not the key epoch ----------------

class HandsFreeWatchdogTests(EngineHarness):
    def test_watchdog_finishes_an_untouched_hands_free_capture(self):
        fire_watchdog = self.arm_hands_free()
        fire_watchdog()
        self.assertEqual(self.events, ["start", "finish"])
        self.assertEqual(self.engine.snapshot(), (False, False))

    def test_watchdog_survives_a_stop_press_whose_release_is_lost(self):
        fire_watchdog = self.arm_hands_free()
        self.engine.pressed()           # the stop tap goes down ... and its key-up never arrives
        self.assertEqual(self.engine.snapshot(), (True, True))
        fire_watchdog()                 # ten minutes later
        self.assertEqual(self.events, ["start", "finish"],
                         "a press disarmed the hands-free watchdog")
        self.assertEqual(self.engine.snapshot(), (False, False))
        self.engine.released()          # the late release is a no-op
        self.assertEqual(self.events, ["start", "finish"])

    def test_menu_armed_hands_free_watchdog_survives_a_press_too(self):
        self.assertTrue(self.engine.force_start())
        fire_watchdog = self.timers.only(sotto.HANDS_FREE_MAX_S)
        self.engine.pressed()
        fire_watchdog()
        self.assertEqual(self.events, ["start", "finish"])
        self.assertEqual(self.engine.snapshot(), (False, False))

    def test_a_stale_watchdog_cannot_end_a_newer_hands_free_capture(self):
        fire_first = self.arm_hands_free()
        self.engine.pressed(); self.engine.released()        # stop tap: finish
        self.assertEqual(self.events, ["start", "finish"])
        self.events.clear()
        fire_second = self.arm_hands_free()
        fire_first()                                         # the old capture's deadline
        self.assertEqual(self.events, ["start"])
        self.assertEqual(self.engine.snapshot(), (True, True))
        fire_second()
        self.assertEqual(self.events, ["start", "finish"])


# -- F6: actions run in decision order and never interleave across threads ----

class OrderedActionTests(EngineHarness):
    def test_stale_tap_expiry_cannot_discard_a_newer_capture(self):
        capture = sotto.CaptureService()
        capture._engine = mock.Mock()        # engine already live: begin() does not cold-start
        capture.idle_release_s = -1
        order = []
        discard_entered = threading.Event()
        resume_discard = threading.Event()
        second_start_ran = threading.Event()
        self.addCleanup(resume_discard.set)

        def on_start():
            order.append("start")
            capture.begin()
            if order.count("start") == 2:
                second_start_ran.set()

        def on_discard():
            discard_entered.set()
            resume_discard.wait(5)           # the timer thread is preempted right here
            order.append("discard")
            capture.abort()

        def on_finish():
            order.append("finish")
            capture.end()

        engine = sotto.GestureEngine(on_start, on_finish, on_discard)
        engine.pressed(); engine.released()  # a lone short tap arms the expiry timer
        self.assertTrue(capture.is_active())
        timer_thread = threading.Thread(target=self.timers.only(sotto.DOUBLE_TAP_WINDOW_S),
                                        daemon=True)
        timer_thread.start()                 # the window closes: discard decided, lock released
        self.assertTrue(discard_entered.wait(5))
        # The timer thread sits inside on_discard. The user presses again for a
        # real push-to-talk dictation: the decision is immediate, the action
        # must wait its turn, and the tap thread must not be held up.
        engine.pressed()
        self.assertEqual(engine.snapshot(), (True, False))
        self.assertEqual(order.count("start"), 1,
                         "start ran on the tap thread while the stale discard was mid-flight")
        resume_discard.set()
        self.assertTrue(second_start_ran.wait(5))
        timer_thread.join(5)
        self.assertFalse(timer_thread.is_alive())
        self.assertEqual(order, ["start", "discard", "start"])
        self.assertTrue(capture.is_active(), "the stale discard killed the new capture")

    def test_a_release_decided_while_start_is_blocked_is_queued_not_run_concurrently(self):
        order = []
        start_entered = threading.Event()
        resume_start = threading.Event()
        self.addCleanup(resume_start.set)

        def on_start():
            start_entered.set()
            resume_start.wait(5)             # e.g. the capture gate is busy
            order.append("start")

        engine = sotto.GestureEngine(on_start, lambda: order.append("finish"),
                                     lambda: order.append("discard"))
        tap_thread = threading.Thread(target=engine.pressed, daemon=True)
        tap_thread.start()
        self.assertTrue(start_entered.wait(5))
        with mock.patch.object(sotto, "HOLD_THRESHOLD_S", 0.0):
            engine.released()                # the resync poller's synthesized release
        self.assertEqual(engine.snapshot(), (False, False))   # decided at once ...
        self.assertEqual(order, [], "finish ran while start was still in flight")
        resume_start.set()
        tap_thread.join(5)
        self.assertEqual(order, ["start", "finish"])        # ... acted on in order


if __name__ == "__main__":
    unittest.main()

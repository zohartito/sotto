from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

import sys

if sys.platform != "darwin":
    raise unittest.SkipTest("adaptive lane is macOS-only in v1 (fcntl)")

from inference_scheduler import InferenceScheduler
from inference_scheduler import default_boot_session_id, default_process_start_token


class SchedulerTests(unittest.TestCase):
    def test_live_priority_final_recheck(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scheduler = InferenceScheduler(root, pid=11, pid_alive=lambda _: True, start_token=lambda _: "a", boot_id=lambda: "b")
            with scheduler.live_request():
                self.assertEqual(scheduler.status()["live_pending"], 1)
                with scheduler.evaluator_lease() as granted:
                    self.assertFalse(granted)
            with scheduler.evaluator_lease() as granted:
                self.assertTrue(granted)

    def test_stale_dead_and_pid_reuse_are_reclaimed(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [100.0]
            scheduler = InferenceScheduler(Path(directory), pid=1, pid_alive=lambda p: p != 22,
                                           start_token=lambda p: "new" if p == 33 else "same",
                                           boot_id=lambda: "boot", wall_clock=lambda: now[0])
            def add(state):
                state["owners"]["dead"] = {"pid": 22, "boot_id": "boot", "start_token": "same", "wall_heartbeat": 0, "role": "evaluator", "pending_live": 0}
                state["owners"]["reuse"] = {"pid": 33, "boot_id": "boot", "start_token": "old", "wall_heartbeat": 0, "role": "evaluator", "pending_live": 0}
            scheduler._update(add)
            self.assertEqual(set(scheduler.reconcile()), {"dead", "reuse"})

    def test_stale_alive_evaluator_is_suspect_and_blocks_only_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            scheduler = InferenceScheduler(Path(directory), pid=1, pid_alive=lambda _: True,
                                           start_token=lambda _: "same", boot_id=lambda: "boot", wall_clock=lambda: 100)
            scheduler._update(lambda state: state["owners"].update({"other": {"pid": 2, "boot_id": "boot", "start_token": "same", "wall_heartbeat": 0, "role": "evaluator", "pending_live": 0}}))
            self.assertTrue(scheduler.status()["evaluator_suspect"])
            with scheduler.evaluator_lease() as granted:
                self.assertFalse(granted)

    def test_default_identity_tokens_are_os_shaped_or_fail_closed(self):
        boot = default_boot_session_id()
        start = default_process_start_token(1)
        self.assertTrue(boot is None or ":" in boot)
        self.assertTrue(start is None or ":" in start)

    def test_heartbeat_updates_both_clock_values(self):
        wall, mono = [1.0], [2.0]
        with tempfile.TemporaryDirectory() as directory:
            scheduler = InferenceScheduler(Path(directory), pid=9, pid_alive=lambda _: True,
                                           start_token=lambda _: "start", boot_id=lambda: "boot",
                                           wall_clock=lambda: wall[0], monotonic=lambda: mono[0])
            scheduler.heartbeat("evaluator")
            wall[0], mono[0] = 3.0, 4.0
            scheduler.heartbeat("evaluator")
            with scheduler._state_lock.held():
                owner = scheduler._read()["owners"][scheduler.owner_id]
            self.assertEqual((owner["wall_heartbeat"], owner["monotonic_heartbeat"]), (3.0, 4.0))

    def test_heartbeat_pump_keeps_owner_registered(self):
        with tempfile.TemporaryDirectory() as directory:
            scheduler = InferenceScheduler(Path(directory), pid=7, pid_alive=lambda _: True,
                                           start_token=lambda _: "s", boot_id=lambda: "b")
            scheduler.heartbeat_seconds = 0.01
            # The pump waits one interval before its first beat; register the
            # owner explicitly so the assertion cannot race the thread.
            scheduler.heartbeat("evaluator")
            with scheduler.heartbeat_pump("evaluator"):
                time.sleep(0.03)
            self.assertEqual(scheduler.status()["owners"], 1)

    def test_evaluator_final_recheck_yields_to_live_registered_after_first_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = InferenceScheduler(root, pid=1, pid_alive=lambda _: True,
                                           start_token=lambda _: "one", boot_id=lambda: "boot")
            live = InferenceScheduler(root, pid=2, pid_alive=lambda _: True,
                                      start_token=lambda _: "two", boot_id=lambda: "boot")
            original = evaluator._try_lease
            def lease_then_register_live():
                fd = original()
                if fd is not None:
                    live._update(lambda state: state["owners"].update({live.owner_id: live._owner("live", 1)}))
                return fd
            evaluator._try_lease = lease_then_register_live  # type: ignore[method-assign]
            with evaluator.evaluator_lease() as granted:
                self.assertFalse(granted)

    def test_live_pending_observes_registered_waiter_without_taking_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            evaluator=InferenceScheduler(root,pid=1,pid_alive=lambda _: True,start_token=lambda _: 'one',boot_id=lambda: 'boot')
            live=InferenceScheduler(root,pid=2,pid_alive=lambda _: True,start_token=lambda _: 'two',boot_id=lambda: 'boot')
            entered=threading.Event(); release=threading.Event()
            def request() -> None:
                with live.live_request():
                    entered.set(); release.wait(1)
            with evaluator.evaluator_lease() as granted:
                self.assertTrue(granted); self.assertFalse(evaluator.live_pending())
                thread=threading.Thread(target=request); thread.start()
                deadline=time.monotonic()+1
                while not evaluator.live_pending() and time.monotonic()<deadline:
                    time.sleep(.005)
                self.assertTrue(evaluator.live_pending())
                self.assertFalse(entered.is_set())
            self.assertTrue(entered.wait(1)); self.assertTrue(evaluator.live_pending())
            release.set(); thread.join(1); self.assertFalse(thread.is_alive())
            self.assertFalse(evaluator.live_pending())


if __name__ == "__main__":
    unittest.main()

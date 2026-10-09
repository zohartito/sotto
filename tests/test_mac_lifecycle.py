"""How the Mac app stops, restarts and refuses work while it does (round-2
audit rows F1c, N3, N6, N8, N9, N12, N18, N27, N40, N44).

The menu actions and the main-loop teardown are closures inside sotto.run();
these tests compile the real closures against fakes (no AppKit run loop, no
mic, no model, no clipboard, no exec) and check what they do.
"""
import ast
import queue
import signal
import sys
import threading
import time
import types
import unittest
from pathlib import Path

if sys.platform != "darwin":
    raise unittest.SkipTest("the Mac app lifecycle — macOS-only")

import sotto

WAIT_S = 5.0


def closures(names, namespace):
    """Compile the named functions defined in sotto.run() — directly or under
    its ``if``/``with``/``try`` blocks, never inside another function — into
    ``namespace``, whose entries stand in for the closure variables they read."""
    source = Path(sotto.__file__).read_text(encoding="utf-8")
    run = next(node for node in ast.parse(source).body
               if isinstance(node, ast.FunctionDef) and node.name == "run")
    found = {}

    def visit(body):
        for node in body:
            if isinstance(node, ast.FunctionDef):
                found.setdefault(node.name, node)
            elif isinstance(node, (ast.If, ast.With, ast.Try)):
                for block in ("body", "orelse", "finalbody"):
                    visit(getattr(node, block, []))
                for handler in getattr(node, "handlers", []):
                    visit(handler.body)
    visit(run.body)
    missing = set(names) - set(found)
    assert not missing, f"not defined in run(): {sorted(missing)}"
    module = ast.Module(body=[found[name] for name in names], type_ignores=[])
    exec(compile(module, sotto.__file__, "exec"), namespace)
    return namespace


class Clock:
    """A virtual monotonic clock; sleep() advances it (unless frozen, for
    tests that end the dictation themselves) and yields the GIL."""

    def __init__(self, frozen=False):
        self.now = 0.0
        self.frozen = frozen

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        if not self.frozen:
            self.now += seconds
        time.sleep(0.001)


def wait_for(predicate, timeout=WAIT_S):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def gate_open(gate) -> bool:
    with gate.starting() as allowed:
        return allowed


class World:
    """A Lifecycle over a real CaptureGate and ShutdownBoundary, with a
    dictation that is in flight until the test says it is done."""

    def __init__(self, frozen=False):
        self.shutdown = sotto.ShutdownBoundary()
        self.gate = sotto.CaptureGate()
        self.clock = Clock(frozen)
        self.busy = threading.Event()
        self.ended = []
        self.lifecycle = None
        if hasattr(sotto, "Lifecycle"):
            self.lifecycle = sotto.Lifecycle(
                self.shutdown, self.gate, end_recording=self.end_recording,
                busy=self.busy.is_set, clock=self.clock, sleep=self.clock.sleep)

    def end_recording(self):
        self.ended.append(self.clock())
        return "gesture" if self.busy.is_set() else None


# -- F1c: Ctrl-C and SIGTERM finish the dictation like Quit --------------------

class SignalStopTests(unittest.TestCase):
    def test_the_first_signal_asks_for_a_drained_stop_not_an_immediate_one(self):
        boundary = sotto.ShutdownBoundary()
        with sotto.shutdown_signal_handlers(boundary):
            handler = signal.getsignal(signal.SIGTERM)
            handler(signal.SIGTERM, None)  # no OS signal: call the installed handler
            self.assertFalse(boundary.requested(), "SIGTERM discarded the dictation at once")
            self.assertTrue(boundary.stop_event.is_set())
            handler(signal.SIGINT, None)   # a second one while it drains stops at once
            self.assertTrue(boundary.requested())

    def test_a_signal_finishes_the_dictation_then_stops(self):
        world = World(frozen=True)
        world.busy.set()
        watcher = threading.Thread(target=world.lifecycle.watch_signals, daemon=True)
        watcher.start()
        world.shutdown.request_stop(signal.SIGINT)
        self.assertTrue(wait_for(lambda: world.ended), "the recording was not ended")
        self.assertFalse(gate_open(world.gate), "a new recording could start during the drain")
        self.assertEqual(world.lifecycle.closed, "quitting")
        time.sleep(0.05)
        self.assertFalse(world.shutdown.requested(), "stopped before the dictation was pasted")
        world.busy.clear()
        self.assertTrue(world.shutdown.event.wait(WAIT_S))

    def test_sigterm_stops_inside_launchds_exit_timeout(self):
        # launchd SIGKILLs Sotto's job 5 s after SIGTERM ("exit timeout = 5"):
        # the drain must leave time to keep the rest in History.
        world = World()
        world.busy.set()  # a wedged model call: never idle
        threading.Thread(target=world.lifecycle.watch_signals, daemon=True).start()
        world.shutdown.request_stop(signal.SIGTERM)
        self.assertTrue(world.shutdown.event.wait(WAIT_S))
        self.assertLessEqual(world.clock(), sotto.SIGTERM_DRAIN_S + 0.5)
        self.assertLess(sotto.SIGTERM_DRAIN_S, 5.0)

    def test_ctrl_c_waits_like_quit(self):
        world = World()
        world.busy.set()
        threading.Thread(target=world.lifecycle.watch_signals, daemon=True).start()
        world.shutdown.request_stop(signal.SIGINT)
        self.assertTrue(world.shutdown.event.wait(WAIT_S))
        self.assertGreaterEqual(world.clock(), sotto.QUIT_DRAIN_S)


# -- Quit, through the real action_quit closure (N44: behaviour, not source) ---

class QuitTests(unittest.TestCase):
    def wire(self, world):
        namespace = {"shutdown": world.shutdown, "lifecycle": world.lifecycle,
                     "capture_gate": world.gate, "log": lambda line: None}
        return closures(["action_quit"], namespace)["action_quit"]

    def test_quit_refuses_new_recordings_and_waits_for_the_paste(self):
        world = World(frozen=True)
        world.busy.set()
        self.wire(world)()
        self.assertFalse(gate_open(world.gate))
        self.assertTrue(wait_for(lambda: world.ended))
        time.sleep(0.05)
        self.assertFalse(world.shutdown.requested())
        world.busy.clear()
        self.assertTrue(world.shutdown.event.wait(WAIT_S))

    def test_quit_gives_up_after_its_cap(self):
        world = World()
        world.busy.set()
        self.wire(world)()
        self.assertTrue(world.shutdown.event.wait(WAIT_S))
        self.assertGreaterEqual(world.clock(), sotto.QUIT_DRAIN_S)
        self.assertLess(world.clock(), sotto.QUIT_DRAIN_S + 1.0)


# -- N12: the menu Restart drains like Quit ----------------------------------

class MenuRestartTests(unittest.TestCase):
    def wire(self, world, *, fail=None):
        self.requests = []

        class Restart:
            def request(restart_self, on_failure=None):
                self.requests.append(gate_open(world.gate))
                if fail:
                    on_failure(fail)
                return True
        self.errors = []
        status_ui = types.SimpleNamespace(show_error=lambda *a: self.errors.append(a))
        namespace = {"shutdown": world.shutdown, "lifecycle": world.lifecycle,
                     "capture_gate": world.gate, "restart": Restart(), "status_ui": status_ui,
                     "ui_call": lambda method, *a: method(*a), "log": lambda line: None}
        names = ["action_restart"] + (["restart_now"] if world.lifecycle else [])
        return closures(names, namespace)["action_restart"]

    def test_restart_waits_for_the_dictation_with_the_gate_closed(self):
        world = World(frozen=True)
        world.busy.set()
        self.wire(world)()
        time.sleep(0.05)
        self.assertEqual(self.requests, [], "restarted with a dictation in flight")
        self.assertFalse(gate_open(world.gate))
        self.assertTrue(world.ended, "the recording in progress was not ended")
        world.busy.clear()
        self.assertTrue(wait_for(lambda: self.requests))
        self.assertEqual(self.requests, [False])  # the gate was still closed at the restart

    def test_a_failed_restart_reopens_the_gate(self):
        world = World()
        self.wire(world, fail="launchd could not restart Sotto")()
        self.assertTrue(wait_for(lambda: self.errors))
        self.assertTrue(gate_open(world.gate))
        self.assertIsNone(world.lifecycle.closed)

    def test_quit_during_the_restart_drain_wins(self):
        world = World(frozen=True)
        world.busy.set()
        self.wire(world)()
        self.assertTrue(world.lifecycle.quit())
        world.busy.clear()
        self.assertTrue(world.shutdown.event.wait(WAIT_S))
        time.sleep(0.05)
        self.assertEqual(self.requests, [])


# -- N3 / N18: a closed gate refuses starts and Retry --------------------------

class ClosedGateTests(unittest.TestCase):
    def wire(self, world):
        self.ui, self.logs, jobs = [], [], queue.Queue()
        capture = types.SimpleNamespace(begin=lambda **kw: self.fail("the mic was opened"),
                                        is_active=lambda: False)
        status_ui = types.SimpleNamespace(
            hide=lambda: self.ui.append("hide"),
            show_hands_free=lambda hint: self.ui.append("hands_free"),
            show_recording=lambda: self.ui.append("recording"),
            show_error=lambda *a: self.ui.append("error"))
        namespace = {
            "shutdown": world.shutdown, "capture_gate": world.gate, "lifecycle": world.lifecycle,
            "capture": capture, "jobs": jobs, "use_nemotron": False,
            "model_rewarm_due": lambda *a, **k: False,
            "model_activity": {"last_finished": 0.0, "rewarming": False},
            "adaptive": False, "status_ui": status_ui, "ui_call": lambda method, *a: method(*a),
            "log": self.logs.append, "time": time, "threading": threading, "uuid": sotto.uuid,
            "current_speech_config": lambda: "config", "binding": {"trigger": "fn"},
            "hands_free_hint": lambda trigger: "",
        }
        names = ["on_start", "on_hands_free", "action_start_now", "action_retry"]
        if world.lifecycle is not None:
            names.append("hide_refused_start")
        closures(names, namespace)
        engine = sotto.GestureEngine(namespace["on_start"], lambda: self.ui.append("finish"),
                                     lambda: self.ui.append("discard"),
                                     on_hands_free=namespace["on_hands_free"])
        namespace["engine"] = engine
        return namespace, engine, jobs

    def closed_world(self):
        world = World()
        if world.lifecycle is not None:
            self.assertTrue(world.lifecycle.close("updating"))
        else:
            world.gate.close()
        return world

    def test_a_press_refused_by_the_gate_leaves_the_gesture_idle(self):
        namespace, engine, _ = self.wire(self.closed_world())
        engine.pressed()
        self.assertEqual(engine.snapshot(), (False, False), "the gesture kept recording with no mic")
        engine.released()
        self.assertNotIn("finish", self.ui, "the key-up finished a capture that never started")
        self.assertNotIn("hide", self.ui, "a refused press hid the orb of the dictation still draining")

    def test_a_menu_start_is_refused_while_the_gate_is_closed(self):
        namespace, engine, _ = self.wire(self.closed_world())
        namespace["action_start_now"]()
        self.assertEqual(engine.snapshot(), (False, False), "hands-free armed with no mic")
        self.assertNotIn("hands_free", self.ui[-1:], "the orb was left saying hands-free")

    def test_a_hands_free_start_that_races_the_gate_ends_hidden(self):
        world = World()
        namespace, engine, _ = self.wire(world)
        world.gate.close()  # closes between the menu's check and on_start
        engine.force_start()
        self.assertEqual(engine.snapshot(), (False, False))
        self.assertEqual(self.ui[-1], "hide", self.ui)

    def test_retry_is_refused_while_the_gate_is_closed(self):
        namespace, _, jobs = self.wire(self.closed_world())
        namespace["action_retry"]("row1")
        self.assertEqual(jobs.qsize(), 0, "Retry queued work during a drain")

    def test_retry_runs_while_open(self):
        namespace, _, jobs = self.wire(World())
        namespace["action_retry"]("row1")
        self.assertEqual(jobs.get_nowait()[:2], ("retry", "row1"))


# -- N8: the engine switch closes the gate like the update restart -------------

class EngineSwitchTests(unittest.TestCase):
    def wire(self, world, *, fail=None, busy_after_close=False):
        self.at_restart, self.errors = [], []
        status_ui = types.SimpleNamespace(show_error=lambda *a: self.errors.append(a))

        def persist_engine_and_restart(mode, request, on_failure=None):
            self.at_restart.append((mode, gate_open(world.gate)))
            if fail:
                on_failure(fail)

        def in_flight(*args, **kwargs):
            return busy_after_close and not gate_open(world.gate)
        namespace = {
            "adaptive": False, "engine_switching": True, "active_engine": "whisper",
            "dictation_in_flight": in_flight, "capture": None, "jobs": None, "pending_deliveries": None,
            "model_activity": {"rewarming": False}, "parakeet_cached": lambda: True,
            "persist_engine_and_restart": persist_engine_and_restart,
            "restart": types.SimpleNamespace(request=None), "status_ui": status_ui,
            "ui_call": lambda method, *a: method(*a), "lifecycle": world.lifecycle,
            "capture_gate": world.gate,
        }
        if world.lifecycle is not None:
            world.lifecycle.busy = lambda: in_flight()
        return closures(["action_set_engine"], namespace)["action_set_engine"]

    def test_the_gate_is_closed_when_the_engine_restart_is_requested(self):
        world = World()
        self.wire(world)("parakeet")
        self.assertEqual(self.at_restart, [("parakeet", False)], "a capture could start before the restart")

    def test_a_failed_switch_reopens_the_gate(self):
        world = World()
        self.wire(world, fail="launchd could not restart Sotto")("parakeet")
        self.assertTrue(self.errors)
        self.assertTrue(gate_open(world.gate))

    def test_a_capture_that_began_before_the_close_refuses_the_switch(self):
        world = World()
        self.wire(world, busy_after_close=True)("parakeet")
        self.assertEqual(self.at_restart, [])
        self.assertTrue(self.errors)
        self.assertTrue(gate_open(world.gate))


# -- N40: a model rewarm is not a dictation ------------------------------------

class WarmupIsNotDictationTests(unittest.TestCase):
    def test_a_queued_rewarm_alone_is_not_in_flight(self):
        capture = types.SimpleNamespace(is_active=lambda: False)
        jobs, deliveries = queue.Queue(), sotto.PendingDeliveries()
        jobs.put(("warmup",))
        self.assertFalse(sotto.dictation_in_flight(capture, jobs, deliveries, warming=True))
        jobs.put(("live",))
        self.assertTrue(sotto.dictation_in_flight(capture, jobs, deliveries, warming=True))
        self.assertTrue(sotto.dictation_in_flight(capture, jobs, deliveries))


# -- the main-loop teardown: N6, N9, N27 ---------------------------------------

class TeardownTests(unittest.TestCase):
    def wire(self, *, foreground=False, exec_error=None, worker_job=None, worker_alive=False,
             queued=()):
        self.events, self.exits, self.kept, self.failed = [], [], [], []
        shutdown = sotto.ShutdownBoundary()
        shutdown.request()
        jobs = queue.Queue()
        for job in queued:
            jobs.put(job)
        test = self

        class Restart:
            foreground_pending = foreground

            def exec_foreground(self):
                test.events.append("exec")
                if exec_error is not None:
                    raise exec_error

            def failed(self, message):
                test.failed.append(message)

        class Thread:
            def join(self, timeout=None):
                pass

            def is_alive(self):
                return worker_alive

        worker = types.SimpleNamespace(current_job=worker_job,
                                       keep_untranscribed=lambda job: self.kept.append(job))
        capture = types.SimpleNamespace(abort=lambda: None, shutdown=lambda: None,
                                        wait_released=lambda timeout: True)
        quartz = types.SimpleNamespace(CGEventTapEnable=lambda tap, on: None,
                                       CFRunLoopGetMain=lambda: None,
                                       CFRunLoopStop=lambda loop: self.events.append("stop"))
        namespace = {
            "shutdown": shutdown, "capture": capture, "jobs": jobs, "restart": Restart(),
            "nemotron": None, "transcription_thread": Thread(), "worker": worker,
            "restart_drain": {"deadline": None, "kept": False},
            "APP_DRAIN_TIMEOUT": 0.0, "RESTART_DRAIN_DEADLINE_S": 0.0, "RESTART_RELEASE_WAIT_S": 0.0,
            "AppHelper": types.SimpleNamespace(callLater=lambda *a: None), "time": time,
            "log": lambda line: None, "os": types.SimpleNamespace(_exit=self.exits.append),
            "Quartz": quartz, "tap": None, "status_ui": None, "ui_call": lambda *a: None,
            "flush_clipboard_restore": lambda *a, **k: self.events.append("restore"),
            "signal": signal,
        }
        for name in ("SIGTERM_RESTORE_WAIT_S", "RESTORE_DELAY_S", "RESTART_FAILED_EXIT"):
            namespace[name] = getattr(sotto, name, None)
        names = ["stop_runtime_on_main"] + (["keep_untranscribed"] if hasattr(sotto, "Lifecycle") else [])
        return closures(names, namespace)["stop_runtime_on_main"], jobs

    def test_n6_the_clipboard_is_put_back_before_the_process_ends(self):
        stop, _ = self.wire()
        stop()
        self.assertIn("restore", self.events, "the paste's clipboard restore never ran")
        self.assertLess(self.events.index("restore"), self.events.index("stop"))

    def test_n6_and_before_a_foreground_restart_execs(self):
        stop, _ = self.wire(foreground=True)
        stop()
        self.assertLess(self.events.index("restore"), self.events.index("exec"))

    def test_n9_a_queued_capture_is_kept_in_history_not_discarded(self):
        live = ("live", "raw", 48000.0, 1.0, 2.0, "cap1", None, None)
        stop, jobs = self.wire(queued=[("warmup",), live, ("retry", "row1")])
        stop()
        self.assertEqual(self.kept, [live])
        self.assertEqual(jobs.unfinished_tasks, 0)

    def test_n9_the_capture_the_model_is_still_on_is_kept(self):
        live = ("live", "raw", 48000.0, 1.0, 2.0, "cap2", None, None)
        stop, _ = self.wire(worker_job=live, worker_alive=True)
        stop()
        self.assertEqual(self.kept, [live])

    def test_n9_a_capture_the_worker_finished_is_not_kept_twice(self):
        live = ("live", "raw", 48000.0, 1.0, 2.0, "cap3", None, None)
        stop, _ = self.wire(worker_job=live, worker_alive=False)
        stop()
        self.assertEqual(self.kept, [])

    def test_n27_a_failed_exec_exits_so_launchd_starts_sotto_again(self):
        for error in (OSError(7, "Argument list too long"), IndexError("tuple index out of range")):
            with self.subTest(error=type(error).__name__):
                stop, _ = self.wire(foreground=True, exec_error=error)
                stop()
                self.assertTrue(self.failed, "the failure callback (engine choice restore) did not run")
                self.assertEqual(self.exits, [sotto.RESTART_FAILED_EXIT])
                self.assertNotEqual(sotto.RESTART_FAILED_EXIT, 0)  # KeepAlive: SuccessfulExit false


if __name__ == "__main__":
    unittest.main()

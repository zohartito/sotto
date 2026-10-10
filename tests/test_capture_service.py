"""CaptureService and its runtime wiring, with every CoreAudio / AVFoundation
entry point replaced by an in-process fake: no microphone is opened, no
event tap installed, no event posted.

Findings covered (audit ledger ids): F12 teardown releases every resource,
F17 per-block sample rates after a mid-recording route change, F21 idle
release honours -1 and never races a press, F41 doctor probes the pinned
input bus, F45 a failed microphone start surfaces, F53 the CFString from
name_of is released.
"""

import ast
import contextlib
import ctypes
import queue
import sys
import threading
import time
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np

import sotto


# -- fake AVFoundation ---------------------------------------------------------

class FakePtr:
    def __init__(self, samples):
        self._bytes = np.asarray(samples, dtype=np.float32).tobytes()

    def as_buffer(self, n):
        return self._bytes[: 4 * n]


class FakeFormat:
    def __init__(self, rate, channels=1):
        self._rate = rate
        self._channels = channels

    def sampleRate(self):
        return self._rate

    def channelCount(self):
        return self._channels


class FakeBuffer:
    def __init__(self, samples, rate):
        self._samples = np.asarray(samples, dtype=np.float32)
        self._rate = rate

    def frameLength(self):
        return len(self._samples)

    def floatChannelData(self):
        return [FakePtr(self._samples)]

    def format(self):
        return FakeFormat(self._rate)


class FakeWhen:
    def __init__(self, sample_time):
        self._t = sample_time

    def sampleTime(self):
        return self._t


class FakeNode:
    def __init__(self, rate=48000.0, remove_raises=False):
        self.rate = rate
        self.remove_raises = remove_raises
        self.tap = None
        self.calls = []

    def removeTapOnBus_(self, bus):
        self.calls.append("removeTap")
        if self.remove_raises:
            raise RuntimeError("AVAudioNode removeTapOnBus: device gone")
        self.tap = None

    def inputFormatForBus_(self, bus):
        self.calls.append("inputFormat")
        return FakeFormat(self.rate)

    def outputFormatForBus_(self, bus):
        self.calls.append("outputFormat")
        return FakeFormat(96000.0, 2)   # the system default, not the pinned mic

    def installTapOnBus_bufferSize_format_block_(self, bus, size, fmt, block):
        self.calls.append(f"installTap@{fmt.sampleRate():.0f}")
        self.tap = block


class FakeEngine:
    def __init__(self, node=None, start_ok=True, stop_raises=False):
        self.node = node or FakeNode()
        self.start_ok = start_ok
        self.stop_raises = stop_raises
        self.running = False
        self.calls = []

    def inputNode(self):
        return self.node

    def prepare(self):
        self.calls.append("prepare")

    def startAndReturnError_(self, _):
        self.calls.append("start")
        if self.start_ok:
            self.running = True
            return True, None
        return False, "Error Domain=com.apple.coreaudio.avfaudio Code=-10868"

    def stop(self):
        self.calls.append("stop")
        if self.stop_raises:
            raise RuntimeError("AVAudioEngine stop: HAL wedged")
        self.running = False

    def isRunning(self):
        return self.running


def fake_avfoundation(engine_factory):
    """sotto imports AVAudioEngine lazily inside _start_engine_locked."""
    module = types.ModuleType("AVFoundation")

    class AVAudioEngine:
        @classmethod
        def alloc(cls):
            return cls

        @classmethod
        def init(cls):
            return engine_factory()

    module.AVAudioEngine = AVAudioEngine
    module.AVAudioEngineConfigurationChangeNotification = "config-change"
    return mock.patch.dict(sys.modules, {"AVFoundation": module})


def tone(freq, seconds, rate):
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def feed(capture, samples, rate, start_sample=0, block=4096):
    """Drive the real CaptureService._tap with fake AVAudioPCMBuffers."""
    for offset in range(0, len(samples), block):
        chunk = samples[offset: offset + block]
        capture._tap(FakeBuffer(chunk, rate), FakeWhen(start_sample + offset))


def fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


# -- F53: the CFString handed to name_of is a +1 reference -------------------

class FakeCoreAudio:
    """One built-in input device whose kAudioObjectPropertyName ('lnam') is a
    CFStringRef the caller owns, exactly as CoreAudio hands it out."""
    NAME_REF = 0xC0FFEE

    def AudioObjectGetPropertyData(self, obj_id, addr_ref, _qsize, _qdata, size_ref, buf_ref):
        selector, scope = addr_ref._obj.selector, addr_ref._obj.scope
        size, buf = size_ref._obj, buf_ref._obj
        if selector == fourcc("dev#"):
            buf[0], size.value = 42, 4
        elif selector == fourcc("tran"):
            buf.value = fourcc("bltn")
        elif selector == fourcc("stm#"):
            if scope == fourcc("inpt"):
                buf[0], size.value = 7, 4
            else:
                size.value = 0
        elif selector == fourcc("term"):
            buf.value = 0x201
        elif selector == fourcc("lnam"):
            buf.value = self.NAME_REF
        return 0


class FakeCoreFoundation:
    def __init__(self):
        self.released = []

    def CFStringGetCString(self, ref, buf, size, encoding):
        buf.value = b"MacBook Pro Microphone"
        return 1

    def CFRelease(self, ref):
        self.released.append(ref.value if isinstance(ref, ctypes.c_void_p) else ref)


class DeviceNameLeakTest(unittest.TestCase):
    def test_every_device_name_cfstring_is_released_after_reading(self):
        core_foundation = FakeCoreFoundation()

        def cdll(path, *args, **kwargs):
            if path.endswith("/CoreAudio"):
                return FakeCoreAudio()
            if path.endswith("/CoreFoundation"):
                return core_foundation
            raise AssertionError(f"unexpected library {path}")

        with mock.patch("ctypes.CDLL", cdll):
            devices = sotto._audio_devices()
        self.assertEqual([d["name"] for d in devices], ["MacBook Pro Microphone"])
        self.assertEqual(core_foundation.released, [FakeCoreAudio.NAME_REF],
                         "kAudioObjectPropertyName is a +1 CFStringRef; name_of must CFRelease it")


# -- F12: teardown releases every resource even if one step raises -----------

class TeardownTest(unittest.TestCase):
    def setUp(self):
        self.logs = []
        patcher = mock.patch.object(sotto, "log", self.logs.append)
        patcher.start()
        self.addCleanup(patcher.stop)

    def live_capture(self, engine):
        capture = sotto.CaptureService()
        engine.running = True
        capture._engine_obj = engine
        capture._engine, capture._node = engine, engine.node
        capture._last_device_scan = sotto.time.monotonic()  # skip the device scan
        return capture

    def test_engine_is_stopped_even_if_tap_removal_raises(self):
        engine = FakeEngine(FakeNode(remove_raises=True))
        capture = self.live_capture(engine)
        self.assertTrue(capture._release_engine())           # key-up release
        self.assertIn("stop", engine.calls)
        self.assertFalse(engine.isRunning(), "idle Sotto left the mic engine running")
        for _ in range(3):
            capture.tick()                                   # 1 Hz idle health check
        self.assertFalse(engine.isRunning())

    def test_release_reports_failure_and_keeps_the_handle_until_the_engine_stops(self):
        engine = FakeEngine(stop_raises=True)
        capture = self.live_capture(engine)
        self.assertFalse(capture._release_engine(), "release claimed success while the engine ran")
        self.assertIs(capture._engine, engine)               # tick retries, nothing is orphaned
        self.assertEqual(engine.node.calls, ["removeTap"])
        engine.stop_raises = False
        capture.tick()                                       # idle release retried
        self.assertFalse(engine.isRunning())
        self.assertIsNone(capture._engine)
        self.assertIn("○ mic released (idle) — wakes on next press", self.logs)


# -- F21: tick's idle release honours -1 and never races a press ---------------

class IdleReleaseTest(unittest.TestCase):
    def setUp(self):
        self.logs = []
        patcher = mock.patch.object(sotto, "log", self.logs.append)
        patcher.start()
        self.addCleanup(patcher.stop)

    def live_capture(self, idle_release):
        capture = sotto.CaptureService()
        engine = FakeEngine()
        engine.running = True
        capture._engine_obj = capture._engine = engine
        capture._node = engine.node
        capture.idle_release_s = idle_release
        capture._last_device_scan = sotto.time.monotonic()  # skip the device scan
        return capture, engine

    def test_negative_idle_release_means_never_for_tick_too(self):
        capture, engine = self.live_capture(-1)
        capture.release_soon()                               # already honours -1
        capture._last_use -= 3600                            # idle for an hour
        for _ in range(3):
            capture.tick()
        self.assertTrue(engine.isRunning(), "--idle-release -1 engine released by tick()")
        self.assertIs(capture._engine, engine)

    def test_a_press_during_ticks_release_keeps_its_engine_and_audio(self):
        capture, engine = self.live_capture(30.0)            # retained engine, ring warm
        feed(capture, tone(200, 2.0, 48000), 48000)
        capture._last_use -= 31                              # idle past the threshold
        real_release = capture._release_engine

        def press_lands_first(**kwargs):                     # between tick's read and its release
            capture.begin()
            return real_release(**kwargs)

        with mock.patch.object(capture, "_release_engine", press_lands_first):
            capture.tick()
        self.assertTrue(capture.is_active())
        self.assertIs(capture._engine, engine, "tick released the engine under a new capture")
        self.assertTrue(engine.isRunning())
        feed(capture, tone(300, 3.0, 48000), 48000, start_sample=96000)
        raw = capture.end()
        self.assertGreaterEqual(len(raw) / 48000, 3.0)       # pre-roll + the 3 s spoken


# -- F17: per-block sample rates survive a mid-recording route change ----------

class InlineThread:
    """threading.Thread stand-in that runs the target on start(): the route
    rebuild happens right where the test can see it."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


def dominant_hz(samples, rate):
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return np.fft.rfftfreq(len(samples), 1 / rate)[int(np.argmax(spectrum))]


class JoinCaptureBlocksTest(unittest.TestCase):
    def test_same_rate_blocks_concatenate_untouched(self):
        blocks = [tone(440, 0.1, 48000), tone(440, 0.1, 48000)]
        joined = sotto.join_capture_blocks(blocks, [48000.0, 48000.0], 48000.0)
        np.testing.assert_array_equal(joined, np.concatenate(blocks))

    def test_each_same_rate_run_is_resampled_before_joining(self):
        blocks = [tone(440, 0.5, 48000), tone(440, 0.5, 48000), tone(440, 1.0, 24000)]
        joined = sotto.join_capture_blocks(blocks, [48000.0, 48000.0, 24000.0], 24000.0)
        self.assertEqual(len(joined), 48000)                 # 2.0 s at 24 kHz
        self.assertAlmostEqual(dominant_hz(joined[:24000], 24000), 440, delta=5)
        self.assertAlmostEqual(dominant_hz(joined[24000:], 24000), 440, delta=5)


class RouteChangeMidCaptureTest(unittest.TestCase):
    def test_batch_audio_keeps_true_duration_and_pitch(self):
        node = FakeNode(rate=48000.0)
        engine = FakeEngine(node)
        capture = sotto.CaptureService()
        capture._engine_obj = engine
        with fake_avfoundation(lambda: engine), \
             mock.patch.object(sotto, "log", lambda msg: None), \
             mock.patch.object(sotto, "_pin_input_to_builtin", lambda n: 77), \
             mock.patch.object(sotto.threading, "Thread", InlineThread):
            capture._start_engine()                          # built-in mic, 48 kHz
            capture.begin()                                  # key down
            feed(capture, tone(440, 1.0, 48000), 48000)      # 1 s spoken at 48 kHz
            node.rate = 24000.0                              # headset connects:
            capture._engine_started_at -= 5
            capture._on_config_change()                      # real rebuild path, now 24 kHz
            self.assertFalse(capture._route_rebuilding)
            self.assertEqual(node.calls[-1], "installTap@24000")
            feed(capture, tone(440, 1.0, 24000), 24000)      # 1 s spoken at 24 kHz
            raw = capture.end()
        rate = capture.native_rate                           # what on_finish enqueues
        self.assertEqual(rate, 24000.0)
        self.assertAlmostEqual(len(raw) / rate, 2.0, delta=0.05)
        whisper_in = sotto.prepare_for_whisper(raw, rate)
        third = len(whisper_in) // 3
        self.assertAlmostEqual(dominant_hz(whisper_in[:third], 16000), 440, delta=10)
        self.assertAlmostEqual(dominant_hz(whisper_in[-third:], 16000), 440, delta=10)


# -- F45: a failed microphone start surfaces ----------------------------------

def run_closures(names, namespace):
    """Compile the named functions nested directly inside sotto.run() verbatim
    (same file, same line numbers) into `namespace`, whose entries stand in
    for the closure variables they read. This exercises the real start /
    finish / discard wiring without a menu bar, an event tap or a mic."""
    tree = ast.parse(Path(sotto.__file__).read_text(encoding="utf-8"))
    run = next(node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == "run")
    found = {node.name: node for node in run.body
             if isinstance(node, ast.FunctionDef) and node.name in names}
    missing = [name for name in names if name not in found]
    if missing:
        raise AssertionError(f"sotto.run() defines no closure(s) {missing}")
    module = ast.Module(body=[found[name] for name in names], type_ignores=[])
    exec(compile(module, sotto.__file__, "exec"), namespace)
    return namespace


class StatusUI:
    """Records every overlay call as (method, *args)."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, *args))

    def names(self):
        return [call[0] for call in self.calls]


class RecordingThread(threading.Thread):
    """Every thread the code under test starts, so the test can join them."""
    started = []

    def start(self):
        RecordingThread.started.append(self)
        super().start()


class FailedMicStartTest(unittest.TestCase):
    def setUp(self):
        RecordingThread.started = []
        self.logs = []
        real_sleep = time.sleep
        for patcher in (mock.patch.object(sotto.threading, "Thread", RecordingThread),
                        mock.patch.object(sotto.time, "sleep", lambda s: None),  # retry pauses
                        mock.patch.object(sotto, "log", self.logs.append)):
            patcher.start()
            self.addCleanup(patcher.stop)
        # announce_live's own clock: its 10 s "mic live" deadline passes in 0.5 s
        self.fast_time = types.SimpleNamespace(monotonic=lambda: time.monotonic() * 20,
                                               sleep=real_sleep, time=time.time)

    def wire(self, *, with_handler):
        capture = sotto.CaptureService()
        capture._engine_obj = FakeEngine(start_ok=False)   # device busy or gone
        ui, jobs = StatusUI(), queue.Queue()

        @contextlib.contextmanager
        def starting():
            yield True

        namespace = {
            "shutdown": types.SimpleNamespace(requested=lambda: False,
                                              enqueue=lambda q, job: q.put(job) or True,
                                              stop_capture=lambda c: None),
            "capture_gate": types.SimpleNamespace(starting=starting),
            "pending_deliveries": sotto.PendingDeliveries(), "finishing": sotto.PendingDeliveries(),
            "use_nemotron": False, "capture": capture, "jobs": jobs,
            "model_rewarm_due": lambda *a, **k: False,
            "model_activity": {"last_finished": 0.0, "rewarming": False},
            "adaptive": False, "status_ui": ui, "ui_call": lambda method, *a: method(*a),
            "log": self.logs.append, "time": self.fast_time, "threading": threading,
            "uuid": uuid, "current_speech_config": lambda: None, "np": np,
        }
        names = ["on_start", "on_finish", "finish_capture", "on_discard"]
        if with_handler:
            names.append("on_mic_failed")
        run_closures(names, namespace)
        gesture = sotto.GestureEngine(namespace["on_start"], namespace["on_finish"],
                                      namespace["on_discard"])
        namespace["engine"] = gesture
        if with_handler:
            capture.on_start_failed = namespace["on_mic_failed"]
        return capture, gesture, ui, jobs

    def join_all(self):
        """Wait for every thread started so far, including ones they start."""
        seen = 0
        while seen < len(RecordingThread.started):
            batch = RecordingThread.started[seen:]
            seen = len(RecordingThread.started)
            for thread in batch:
                thread.join(10)
                self.assertFalse(thread.is_alive())

    def test_a_failed_start_never_announces_a_live_mic(self):
        with fake_avfoundation(lambda: None), \
             mock.patch.object(sotto, "_pin_input_to_builtin", lambda node: 77):
            capture, gesture, ui, jobs = self.wire(with_handler=False)
            gesture.pressed()                               # key down: five failed starts
            self.join_all()
        self.assertIn("! could not start the mic after 5 tries", self.logs)
        self.assertNotIn("● recording (mic live)", self.logs)
        after_waking = ui.names()[ui.names().index("show_waking") + 1:]
        self.assertNotIn("show_recording", after_waking, "the orb went solid on a dead mic")
        self.assertFalse(capture.is_waking())

    def test_a_failed_start_ends_the_gesture_and_tells_the_user_once(self):
        with fake_avfoundation(lambda: None), \
             mock.patch.object(sotto, "_pin_input_to_builtin", lambda node: 77):
            capture, gesture, ui, jobs = self.wire(with_handler=True)
            gesture.pressed()
            self.join_all()
            self.assertEqual(gesture.snapshot(), (False, False), "gesture kept recording a dead mic")
            self.assertFalse(capture.is_active())
            gesture.released()                              # the key-up finds nothing to finish
            self.join_all()
        errors = [call for call in ui.calls if call[0] == "show_error"]
        self.assertEqual(len(errors), 1, ui.calls)
        self.assertEqual(errors[0][1], "Could not start the microphone")
        self.assertIn("hide", ui.names()[ui.names().index("show_waking") + 1:])  # the orb goes away
        self.assertNotIn("● recording (mic live)", self.logs)
        self.assertEqual(jobs.qsize(), 0)                   # nothing pretends to record
        self.assertLessEqual(sum("captured" in line for line in self.logs), 1)


# -- F41: doctor probes the input the capture path records from ---------------

class DoctorProbeTest(unittest.TestCase):
    def test_probe_pins_the_capture_mic_then_reads_the_input_bus(self):
        node = FakeNode(rate=48000.0)
        engine = FakeEngine(node)
        pinned = []

        def pin(target_node):
            pinned.append(target_node)
            node.calls.append("pin")
            return 77

        out = __import__("io").StringIO()
        with fake_avfoundation(lambda: engine), \
             mock.patch.object(sotto, "_pin_input_to_builtin", pin), \
             contextlib.redirect_stdout(out):
            status = sotto.doctor_input_probe()
        self.assertEqual(status, 0)
        self.assertEqual(pinned, [node])
        self.assertEqual(node.calls, ["pin", "inputFormat"],
                         "probe must pin first, then read the INPUT bus — never the output bus")
        self.assertEqual(out.getvalue().strip(), "48000 Hz, 1 ch")

    @unittest.skipUnless(sys.platform == "darwin",
                         "macOS doctor: patches ApplicationServices.AXIsProcessTrusted")
    def test_doctor_runs_the_probe_through_sotto_not_an_unpinned_output_bus(self):
        logs = []
        completed = types.SimpleNamespace(returncode=0, stdout="48000 Hz, 1 ch\n",
                                          stderr="  mic: built-in mic\n")
        with mock.patch("subprocess.run", return_value=completed) as run, \
             mock.patch.object(sotto, "microphone_permission", return_value=True), \
             mock.patch("ApplicationServices.AXIsProcessTrusted", return_value=True), \
             mock.patch.object(sotto, "log", logs.append):
            sotto.doctor()
        command = run.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        probe = " ".join(command)
        self.assertIn("doctor_input_probe", probe)
        self.assertNotIn("outputFormatForBus_", probe)
        self.assertEqual(run.call_args.kwargs.get("cwd"), str(Path(sotto.__file__).resolve().parent))
        self.assertIn("✓ input device: 48000 Hz, 1 ch (built-in mic)", logs)


if __name__ == "__main__":
    unittest.main()

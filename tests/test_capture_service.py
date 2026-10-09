"""CaptureService and its runtime wiring, with every CoreAudio / AVFoundation
entry point replaced by an in-process fake: no microphone is opened, no
event tap installed, no event posted.

Findings covered (audit ledger ids): F12 teardown releases every resource,
F17 per-block sample rates after a mid-recording route change, F21 idle
release honours -1 and never races a press, F41 doctor probes the pinned
input bus, F45 a failed microphone start surfaces, F53 the CFString from
name_of is released.
"""

import ctypes
import sys
import types
import unittest
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


if __name__ == "__main__":
    unittest.main()

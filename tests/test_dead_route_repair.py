"""Bit-exact silence means this process's capture path is wedged.

2026-09-20: AirPods Pro dropped and reconnected under a new CoreAudio
device id. From then on EVERY capture in the running process came back
bit-exact zero — including ones pinned to the built-in mic — and Whisper
looped on the silence ("ощ" x223, "Liquid" x~200, "Copyright Australian
Broadcasting Corporation"). Meanwhile ffmpeg recorded those same AirPods
at peak 18609 and a fresh python process running CaptureService captured
normally at peak 0.059. The devices were fine; the process was not.

Stopping and starting the engine does not clear it: the AVAudioEngine
object is reused for the process lifetime and carries the wedged HAL
input unit across rebuilds (the 11:03 log rebuilt straight onto zeros).
So the ladder is: throw the engine object away, then the process.
"""

import time
import unittest
from unittest import mock

import numpy as np

import sotto


class RepairLadderTest(unittest.TestCase):
    def _service(self, *, active=None, repairs=0, age=9_999.0):
        service = sotto.CaptureService()
        service._engine = mock.Mock(isRunning=mock.Mock(return_value=True))
        service._engine_obj = mock.Mock()
        service._device_id = 101
        service._zero_since = 0.0          # zeros since long before now
        service._last_block_at = float("inf")   # silent, not stalled
        service._silence_repairs = repairs
        service._process_started_at = time.monotonic() - age
        service._active = active
        return service

    def _tick(self, service):
        with mock.patch.object(service, "_release_engine") as release, \
             mock.patch.object(service, "_start_engine") as start, \
             mock.patch.object(service, "_restart_app") as restart:
            service.tick()
        return release, start, restart

    def test_first_repair_discards_the_reused_engine_object(self):
        # the whole point: a stop/start that keeps _engine_obj kept the
        # wedged input unit and rebuilt onto zeros again
        service = self._service(active=[np.zeros(4096, dtype=np.float32)])
        release, start, restart = self._tick(service)
        self.assertIsNone(service._engine_obj)
        self.assertEqual(service._silence_repairs, 1)
        release.assert_called_once()
        start.assert_called_once()
        restart.assert_not_called()

    def test_second_repair_restarts_the_process(self):
        service = self._service(active=[np.zeros(4096, dtype=np.float32)],
                                repairs=1)
        with mock.patch.object(service, "_restart_app",
                               return_value=True) as restart:
            service.tick()
        restart.assert_called_once()
        self.assertEqual(service._silence_repairs, 2)

    def test_a_refused_restart_keeps_the_rung_armed(self):
        # the age guard must not burn the last rung: a process that goes deaf
        # in its first 10 minutes would otherwise stay deaf forever, since
        # real audio — the only thing that rearms the ladder — never arrives
        service = self._service(active=[np.zeros(4096, dtype=np.float32)],
                                repairs=1)
        with mock.patch.object(service, "_restart_app", return_value=False):
            service.tick()
        self.assertEqual(service._silence_repairs, 1)
        # ... and it fires once the process is old enough
        with mock.patch.object(service, "_restart_app",
                               return_value=True) as restart:
            service.tick()
        restart.assert_called_once()
        self.assertEqual(service._silence_repairs, 2)

    def test_the_ladder_stops_after_the_restart_rung(self):
        # a permanently muted input must not restart sotto in a loop
        service = self._service(active=[np.zeros(4096, dtype=np.float32)],
                                repairs=2)
        release, start, restart = self._tick(service)
        release.assert_not_called()
        start.assert_not_called()
        restart.assert_not_called()

    def test_a_young_process_is_not_restarted(self):
        service = self._service(repairs=1, age=5.0)
        service.restart_callback = mock.Mock()
        self.assertFalse(service._restart_app())
        service.restart_callback.assert_not_called()

    def test_an_old_process_requests_its_own_runtime_restart(self):
        service = self._service(repairs=1)
        service.restart_callback = mock.Mock(return_value=True)
        self.assertTrue(service._restart_app())
        service.restart_callback.assert_called_once_with()

    def test_a_failed_relaunch_does_not_claim_success(self):
        service = self._service(repairs=1)
        service.restart_callback = mock.Mock(side_effect=OSError("restart failed"))
        self.assertFalse(service._restart_app())

    def test_all_zero_capture_is_dropped_so_the_hold_restarts_clean(self):
        service = self._service(active=[np.zeros(4096, dtype=np.float32),
                                        np.zeros(4096, dtype=np.float32)])
        self._tick(service)
        self.assertEqual(service._active, [])

    def test_speech_recorded_before_the_route_died_survives(self):
        speech = np.full(4096, 0.02, dtype=np.float32)
        service = self._service(active=[speech,
                                        np.zeros(4096, dtype=np.float32)])
        self._tick(service)
        self.assertEqual(len(service._active), 2)
        self.assertTrue(np.array_equal(service._active[0], speech))


class ShortHoldRepairTest(unittest.TestCase):
    """A 1.4s press ends before the 3s in-capture watch can fire."""

    def _service(self, blocks, *, rate=48000.0, repairs=0):
        service = sotto.CaptureService()
        service._engine_obj = mock.Mock()
        service.native_rate = rate
        service._silence_repairs = repairs
        service._process_started_at = time.monotonic() - 9_999.0
        service._active = list(blocks)
        return service

    def _end(self, service):
        with mock.patch.object(service, "_rebuild_engine") as rebuild, \
             mock.patch.object(service, "_restart_app", return_value=True), \
             mock.patch("threading.Thread"):
            samples = service.end()
        return samples, rebuild

    def test_short_dead_capture_repairs_for_the_next_press(self):
        service = self._service([np.zeros(96_000, dtype=np.float32)])
        with mock.patch("threading.Thread") as thread:
            service.end()
        self.assertEqual(service._silence_repairs, 1)
        thread.call_args.kwargs["target"]()      # the deferred teardown
        self.assertIsNone(service._engine_obj)

    def test_key_up_repair_never_blocks_the_gesture_tap(self):
        # end() runs straight off the Quartz event tap (on_finish), and
        # CoreAudio teardown there can get the tap disabled (2026-09-15)
        service = self._service([np.zeros(96_000, dtype=np.float32)])
        with mock.patch("threading.Thread") as thread, \
             mock.patch.object(service, "_release_engine") as release:
            service.end()
            release.assert_not_called()
        self.assertEqual(thread.call_args.kwargs["target"],
                         service._rebuild_engine)

    def test_key_up_repair_never_starts_an_engine(self):
        # idle sotto must not hold the microphone (--idle-release 0)
        service = self._service([np.zeros(96_000, dtype=np.float32)])
        with mock.patch.object(service, "_release_engine"), \
             mock.patch.object(service, "_start_engine") as start:
            service._repair_dead_route(capturing=False)
            service._rebuild_engine()
        start.assert_not_called()

    def test_real_audio_never_triggers_a_repair(self):
        service = self._service([np.full(96_000, 1e-5, dtype=np.float32)])
        self._end(service)
        self.assertEqual(service._silence_repairs, 0)
        self.assertIsNotNone(service._engine_obj)

    def test_key_tap_artifact_is_not_enough_evidence(self):
        # a sub-second capture can be Bluetooth warm-up, not a dead route
        service = self._service([np.zeros(12_000, dtype=np.float32)])
        self._end(service)
        self.assertEqual(service._silence_repairs, 0)

    def test_empty_capture_is_not_evidence(self):
        service = self._service([])
        samples, _ = self._end(service)
        self.assertEqual(len(samples), 0)
        self.assertEqual(service._silence_repairs, 0)


class RepairRearmTest(unittest.TestCase):
    def _tap(self, service, samples):
        buffer = mock.Mock()
        buffer.frameLength.return_value = len(samples)
        buffer.floatChannelData.return_value = [
            mock.Mock(as_buffer=mock.Mock(return_value=samples.tobytes()))]
        buffer.format.return_value = mock.Mock(
            sampleRate=mock.Mock(return_value=48000.0))
        service._tap(buffer, mock.Mock(sampleTime=mock.Mock(return_value=0)))

    def test_real_audio_rearms_the_ladder(self):
        service = sotto.CaptureService()
        service._silence_repairs = 2
        self._tap(service, np.full(512, 0.01, dtype=np.float32))
        self.assertEqual(service._silence_repairs, 0)
        self.assertIsNone(service._zero_since)

    def test_quiet_nonzero_audio_counts_as_alive(self):
        # whispered dictation sits at ~1e-5, far below any energy gate but
        # far above the dead route's exact zero (2026-08-13/14)
        service = sotto.CaptureService()
        service._silence_repairs = 1
        self._tap(service, np.full(512, 1e-5, dtype=np.float32))
        self.assertEqual(service._silence_repairs, 0)

    def test_zero_buffers_start_the_silence_clock_without_rearming(self):
        service = sotto.CaptureService()
        service._silence_repairs = 1
        self._tap(service, np.zeros(512, dtype=np.float32))
        self.assertIsNotNone(service._zero_since)
        self.assertEqual(service._silence_repairs, 1)


if __name__ == "__main__":
    unittest.main()

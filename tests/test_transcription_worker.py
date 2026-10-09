"""Golden master for the transcription worker.

Each scenario drives the worker with fakes (no model, no mic, no clipboard,
no AppKit main loop) and records everything observable: History appends,
Retry commits, deliveries (inject / undo / copy), UI calls, progress totals
and log lines. The expected values were recorded from the original
``run().transcribe_worker`` closure in sotto.py, so the extracted class must
reproduce them exactly.
"""
from __future__ import annotations

import os
import queue
import re
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import dictionary
import settings
import sotto
from audio_codec import prepare_canonical
from speech_config import resolve_speech_config
from transcription import TranscriptionWorker

WHISPER = resolve_speech_config("auto", language="en")
NEMOTRON = resolve_speech_config("nemotron-en")
LOOP = "difference " * 40
PREFIX = "That is part of the plan we agreed on last week."
RECORD = os.environ.get("SOTTO_GOLDEN_RECORD") == "1"


# --- fakes -------------------------------------------------------------------

class FakeWhisper:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = 0
        self.last_language = None

    def transcribe(self, samples, **kwargs):
        self.calls += 1
        return self.behaviour(samples, **kwargs)


class FakeNemotron:
    def __init__(self):
        self.calls = 0

    def transcribe(self, samples):
        self.calls += 1
        return "nemotron transcribe should not run when the stream finished"


class FakeStream:
    """The streaming capture object a Nemotron live job carries."""

    def __init__(self, text: str, fallback: bool = False):
        self.text = text
        self.fallback = fallback
        self.finished = 0
        self.closed = 0

    def finish(self):
        self.finished += 1
        return self.text, prepare_canonical(sotto.prepare_for_whisper(self.raw, self.rate))

    def close(self):
        self.closed += 1


class FakeCoordinator:
    def __init__(self, snapshot=None, *, append=None):
        self.appended: list[dict] = []
        self.retries: list[dict] = []
        self.snapshot = snapshot
        self.lookups: list[str] = []
        self._append = append

    def append_live(self, text, prepared, seconds, model, **kw):
        if self._append is not None:
            return self._append(text, prepared, seconds, model, **kw)
        row = {"id": kw.get("entry_id") or "row1", "text": text, "provenance": kw.get("provenance")}
        self.appended.append({"text": text, "seconds": round(seconds, 3), "model": model,
                              "prepared_seconds": round(len(prepared.asr_samples) / sotto.SAMPLE_RATE, 3),
                              **kw})
        return row

    def snapshot_for_retry(self, entry_id):
        self.lookups.append(entry_id)
        return self.snapshot

    def commit_retry(self, entry_id, text, expected_revision, **kw):
        self.retries.append({"entry_id": entry_id, "text": text,
                             "expected_revision": expected_revision, **kw})
        return {"id": entry_id, "text": text}


class FakeStatusUI:
    def __init__(self):
        self.errors: list[tuple[str, str]] = []

    def hide_if_transcribing(self):
        pass

    def show_error(self, title, message):
        self.errors.append((title, message))


def speech(seconds=1.5, rate=16000, amplitude=0.1):
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def snapshot(samples, revision=3):
    return types.SimpleNamespace(samples=samples, expected_revision=revision,
                                 inference_audio_path=Path("/nonexistent/row.wav"),
                                 inference_identity=None)


# --- harness -----------------------------------------------------------------

class Harness:
    """Build the worker with fakes and run jobs through it."""

    def __init__(self, behaviour=None, *, snapshot=None, nemotron=False, append=None):
        self.logs: list[str] = []
        self.ui_calls: list[tuple] = []
        self.delivered: list[tuple] = []
        self.injected: list[str] = []
        self.undone = 0
        self.copied: list[str] = []
        self.totals: list[tuple[str, float]] = []
        self.refreshed = 0
        self.shutdown = sotto.ShutdownBoundary()
        self.jobs: queue.Queue = queue.Queue()
        self.coordinator = FakeCoordinator(snapshot, append=append)
        self.status_ui = FakeStatusUI()
        self.whisper = FakeWhisper(behaviour) if behaviour is not None else None
        self.nemotron = FakeNemotron() if nemotron else None
        self.config = NEMOTRON if nemotron else WHISPER

        def copy_text(text):
            self.copied.append(text)
        copy_text.__name__ = "_copy_text"

        def inject_when_clear(text, attempts, in_history=True):
            self.injected.append(text)

        def undo_when_clear(attempts):
            self.undone += 1

        def record_totals(text, seconds):
            self.totals.append((text, round(seconds, 3)))

        def refresh_history():
            self.refreshed += 1

        self.collaborators = dict(
            shutdown=self.shutdown, jobs=self.jobs, log=self.logs.append,
            model=self.config.model_repo, mlx_whisper=self.whisper, nemotron=self.nemotron,
            current_speech_config=lambda: self.config,
            coordinator=self.coordinator, refresh_history=refresh_history,
            adaptive_runtime=None, adaptive=False, use_nemotron=nemotron,
            status_ui=self.status_ui,
            # ui_call / deliver_call would hop to the AppKit main thread; run them inline.
            ui_call=self._ui_call, deliver_call=self._deliver_call,
            inject_when_clear=inject_when_clear, undo_when_clear=undo_when_clear,
            record_totals=record_totals,
            model_activity={"last_finished": time.monotonic(), "rewarming": False},
            glossary_terms=(),
        )
        self.worker = TranscriptionWorker(**self.collaborators, copy_text=copy_text)
        self.run = self.worker.run

    def _ui_call(self, method, *args):
        self.ui_calls.append((getattr(method, "__name__", str(method)), args))
        method(*args)

    def _deliver_call(self, method, *args):
        self.delivered.append((getattr(method, "__name__", str(method)), args))
        method(*args)

    def live_job(self, raw, rate=16000, stream=None):
        if stream is not None:
            stream.raw, stream.rate = raw, rate
        return ("live", raw, float(rate), time.time(), time.monotonic(), "cap0001", self.config, stream)

    def retry_job(self, entry_id="row1"):
        return ("retry", entry_id, time.monotonic(), "cap0002", self.config)

    def run_jobs(self, *jobs, timeout=20.0):
        # Advisory VAD would load Silero; a neutral stand-in keeps the test fast.
        fake_vad = types.ModuleType("vad")
        fake_vad.analyze = lambda samples: (1.0, [])
        fake_vad.trim_to_speech = lambda samples, spans: samples
        with patch.dict(sys.modules, {"vad": fake_vad}):
            thread = threading.Thread(target=self.run, daemon=True)
            thread.start()
            for job in jobs:
                self.jobs.put(job)
            deadline = time.monotonic() + timeout
            while self.jobs.unfinished_tasks and time.monotonic() < deadline:
                time.sleep(0.01)
            self.shutdown.request()
            thread.join(2)
        assert not self.jobs.unfinished_tasks, "worker did not finish the jobs"
        assert not any("is not defined" in line for line in self.logs), self.logs
        return self

    def observed(self) -> dict:
        """Everything a user, History or the log could see, minus wall-clock values."""
        def without_latency(row):
            row = dict(row)
            row.pop("ts", None)
            raw = row.pop("raw_samples", None)
            row["raw_kept"] = raw is not None and len(raw) > 0
            metadata = row.pop("latency", None)
            row["latency_keys"] = sorted(metadata) if metadata else None
            return row
        return {
            "appends": [without_latency(row) for row in self.coordinator.appended],
            "retries": [without_latency(row) for row in self.coordinator.retries],
            "lookups": list(self.coordinator.lookups),
            "injected": list(self.injected),
            "undone": self.undone,
            "copied": list(self.copied),
            "ui": [name for name, _ in self.ui_calls],
            "delivered": [name for name, _ in self.delivered],
            "errors": list(self.status_ui.errors),
            "totals": list(self.totals),
            "refreshed": self.refreshed,
            "asr_calls": self.whisper.calls if self.whisper else 0,
            "nemotron_calls": self.nemotron.calls if self.nemotron else 0,
            "log": [re.sub(r"\d+\.\d+", "N", line) for line in self.logs],
        }


def says(text):
    return lambda samples, **kwargs: {"text": text, "segments": []}


# --- the golden master -------------------------------------------------------

EXPECTED: dict[str, dict] = {'a_whisper_live_normal': {'appends': [{'adaptive': False,
                                        'entry_id': None,
                                        'language': 'en',
                                        'latency_keys': ['asr_seconds',
                                                         'queue_wait_seconds',
                                                         'release_to_text_seconds'],
                                        'model': 'mlx-community/whisper-large-v3-turbo',
                                        'prepared_seconds': 2.0,
                                        'preprocessing': {'resampled_normalized': True,
                                                          'silence_collapsed': False,
                                                          'vad_trimmed': False},
                                        'profile': 'auto',
                                        'prompt': {'disabled': 'no glossary terms'},
                                        'provenance': 'live',
                                        'raw_kept': True,
                                        'raw_sample_rate': 16000.0,
                                        'seconds': 2.0,
                                        'text': 'Please call me back at four',
                                        'vad': {'available': True,
                                                'span_count': 0,
                                                'speech_fraction': 1.0}}],
                           'asr_calls': 1,
                           'copied': [],
                           'delivered': ['inject_when_clear'],
                           'errors': [],
                           'injected': ['Please call me back at four'],
                           'log': ['→ Ns after release (speech model Ns) · 27 chars'],
                           'lookups': [],
                           'nemotron_calls': 0,
                           'refreshed': 1,
                           'retries': [],
                           'totals': [('Please call me back at four', 2.0)],
                           'ui': ['hide_if_transcribing'],
                           'undone': 0},
 'b1_loop_salvaged': {'appends': [{'adaptive': False,
                                   'entry_id': None,
                                   'language': 'en',
                                   'latency_keys': ['asr_seconds',
                                                    'queue_wait_seconds',
                                                    'release_to_text_seconds'],
                                   'model': 'mlx-community/whisper-large-v3-turbo',
                                   'prepared_seconds': 12.0,
                                   'preprocessing': {'repetition_trimmed': {'dropped_chars': 440,
                                                                            'repeats': 40,
                                                                            'unit': 'difference'},
                                                     'resampled_normalized': True,
                                                     'silence_collapsed': False,
                                                     'vad_trimmed': False},
                                   'profile': 'auto',
                                   'prompt': {'disabled': 'no glossary terms'},
                                   'provenance': 'live',
                                   'raw_kept': True,
                                   'raw_sample_rate': 16000.0,
                                   'seconds': 12.0,
                                   'text': 'That is part of the plan we agreed on last week.',
                                   'vad': {'available': True,
                                           'span_count': 0,
                                           'speech_fraction': 1.0}}],
                      'asr_calls': 1,
                      'copied': [],
                      'delivered': ['inject_when_clear'],
                      'errors': [],
                      'injected': ['That is part of the plan we agreed on last week.'],
                      'log': ['  cut a 40x repetition loop (440 chars) — keeping the clean prefix',
                              '→ Ns after release (speech model Ns) · 48 chars'],
                      'lookups': [],
                      'nemotron_calls': 0,
                      'refreshed': 1,
                      'retries': [],
                      'totals': [('That is part of the plan we agreed on last week.', 12.0)],
                      'ui': ['hide_if_transcribing'],
                      'undone': 0},
 'b2_pure_loop_held': {'appends': [{'adaptive': False,
                                    'entry_id': None,
                                    'language': 'en',
                                    'latency_keys': ['asr_seconds',
                                                     'queue_wait_seconds',
                                                     'release_to_text_seconds'],
                                    'model': 'mlx-community/whisper-large-v3-turbo',
                                    'prepared_seconds': 2.0,
                                    'preprocessing': {'outcome': 'suspect',
                                                      'resampled_normalized': True,
                                                      'silence_collapsed': False,
                                                      'vad_trimmed': False},
                                    'profile': 'auto',
                                    'prompt': {'disabled': 'no glossary terms'},
                                    'provenance': 'live_suspect',
                                    'raw_kept': True,
                                    'raw_sample_rate': 16000.0,
                                    'seconds': 2.0,
                                    'text': 'difference difference difference difference '
                                            'difference difference difference difference '
                                            'difference difference difference difference '
                                            'difference difference difference difference '
                                            'difference difference difference difference '
                                            'difference difference difference difference '
                                            'difference difference difference difference '
                                            'difference difference difference difference '
                                            'difference difference difference difference '
                                            'difference difference difference difference',
                                    'vad': {'available': True,
                                            'span_count': 0,
                                            'speech_fraction': 1.0}}],
                       'asr_calls': 1,
                       'copied': [],
                       'delivered': [],
                       'errors': [],
                       'injected': [],
                       'log': ['! not pasted (220 chars/sec) — kept in history',
                               '→ Ns after release (speech model Ns) · 439 chars'],
                       'lookups': [],
                       'nemotron_calls': 0,
                       'refreshed': 1,
                       'retries': [],
                       'totals': [],
                       'ui': ['hide_if_transcribing'],
                       'undone': 0},
 'c_empty_transcript': {'appends': [{'adaptive': False,
                                     'entry_id': None,
                                     'language': 'en',
                                     'latency_keys': ['asr_seconds',
                                                      'queue_wait_seconds',
                                                      'release_to_text_seconds'],
                                     'model': 'mlx-community/whisper-large-v3-turbo',
                                     'prepared_seconds': 2.0,
                                     'preprocessing': {'outcome': 'suspect',
                                                       'resampled_normalized': True,
                                                       'silence_collapsed': False,
                                                       'vad_trimmed': False},
                                     'profile': 'auto',
                                     'prompt': {'disabled': 'no glossary terms'},
                                     'provenance': 'live_suspect',
                                     'raw_kept': True,
                                     'raw_sample_rate': 16000.0,
                                     'seconds': 2.0,
                                     'text': '[no speech detected]',
                                     'vad': {'available': True,
                                             'span_count': 0,
                                             'speech_fraction': 1.0}}],
                        'asr_calls': 1,
                        'copied': [],
                        'delivered': [],
                        'errors': [],
                        'injected': [],
                        'log': ['! not pasted (empty transcript) — kept in history',
                                '→ Ns after release (speech model Ns) · 20 chars'],
                        'lookups': [],
                        'nemotron_calls': 0,
                        'refreshed': 1,
                        'retries': [],
                        'totals': [],
                        'ui': ['hide_if_transcribing'],
                        'undone': 0},
 'd_dead_route': {'appends': [{'adaptive': False,
                               'language': 'en',
                               'latency_keys': ['asr_seconds', 'queue_wait_seconds'],
                               'model': 'mlx-community/whisper-large-v3-turbo',
                               'prepared_seconds': 2.0,
                               'preprocessing': {'outcome': 'suspect',
                                                 'resampled_normalized': True,
                                                 'silence_collapsed': False,
                                                 'vad_trimmed': False},
                               'profile': 'auto',
                               'prompt': {'disabled': 'no glossary terms'},
                               'provenance': 'live_suspect',
                               'raw_kept': True,
                               'raw_sample_rate': 16000.0,
                               'seconds': 2.0,
                               'text': '[dead microphone]',
                               'vad': {'available': False,
                                       'span_count': 0,
                                       'speech_fraction': 0.0}}],
                  'asr_calls': 0,
                  'copied': [],
                  'delivered': [],
                  'errors': [],
                  'injected': [],
                  'log': ['! not pasted (dead microphone (all-zero capture))', '→ Ns · 17 chars'],
                  'lookups': [],
                  'nemotron_calls': 0,
                  'refreshed': 1,
                  'retries': [],
                  'totals': [],
                  'ui': ['hide_if_transcribing'],
                  'undone': 0},
 'e_sub_02s_dropped': {'appends': [],
                       'asr_calls': 0,
                       'copied': [],
                       'delivered': [],
                       'errors': [],
                       'injected': [],
                       'log': ['○ sub-Ns capture dropped (key-tap artifact)'],
                       'lookups': [],
                       'nemotron_calls': 0,
                       'refreshed': 0,
                       'retries': [],
                       'totals': [],
                       'ui': ['hide_if_transcribing'],
                       'undone': 0},
 'f_scratch_that': {'appends': [{'adaptive': False,
                                 'entry_id': None,
                                 'language': 'en',
                                 'latency_keys': ['asr_seconds',
                                                  'queue_wait_seconds',
                                                  'release_to_text_seconds'],
                                 'model': 'mlx-community/whisper-large-v3-turbo',
                                 'prepared_seconds': 1.5,
                                 'preprocessing': {'resampled_normalized': True,
                                                   'silence_collapsed': False,
                                                   'vad_trimmed': False,
                                                   'voice': {'action': 'scratch',
                                                             'asr_text': 'Scratch that.'}},
                                 'profile': 'auto',
                                 'prompt': {'disabled': 'no glossary terms'},
                                 'provenance': 'live',
                                 'raw_kept': True,
                                 'raw_sample_rate': 16000.0,
                                 'seconds': 1.5,
                                 'text': '',
                                 'vad': {'available': True,
                                         'span_count': 0,
                                         'speech_fraction': 1.0}}],
                    'asr_calls': 1,
                    'copied': [],
                    'delivered': ['undo_when_clear'],
                    'errors': [],
                    'injected': [],
                    'log': ['→ Ns after release (speech model Ns) · 0 chars'],
                    'lookups': [],
                    'nemotron_calls': 0,
                    'refreshed': 1,
                    'retries': [],
                    'totals': [],
                    'ui': ['hide_if_transcribing'],
                    'undone': 1},
 'g1_retry_copied': {'appends': [],
                     'asr_calls': 1,
                     'copied': ['Please call me back at four'],
                     'delivered': [],
                     'errors': [],
                     'injected': [],
                     'log': ['↻ retried · 27 chars'],
                     'lookups': ['row1'],
                     'nemotron_calls': 0,
                     'refreshed': 1,
                     'retries': [{'entry_id': 'row1',
                                  'expected_revision': 3,
                                  'language': 'en',
                                  'latency_keys': ['asr_seconds', 'queue_wait_seconds'],
                                  'model': 'mlx-community/whisper-large-v3-turbo',
                                  'preprocessing': {'retry_canonical': True,
                                                    'voice': {'asr_text': 'Um, please call me back '
                                                                          'at four'}},
                                  'profile': 'auto',
                                  'prompt': {'disabled': 'no glossary terms'},
                                  'provenance': 'retry',
                                  'raw_kept': False,
                                  'text': 'Please call me back at four',
                                  'vad': {'available': None,
                                          'span_count': None,
                                          'speech_fraction': None}}],
                     'totals': [],
                     'ui': ['_copy_text', 'hide_if_transcribing'],
                     'undone': 0},
 'g2_retry_held': {'appends': [],
                   'asr_calls': 1,
                   'copied': [],
                   'delivered': [],
                   'errors': [],
                   'injected': [],
                   'log': ['↻ retried — not copied (220 chars/sec); kept in History'],
                   'lookups': ['row1'],
                   'nemotron_calls': 0,
                   'refreshed': 1,
                   'retries': [{'entry_id': 'row1',
                                'expected_revision': 3,
                                'language': 'en',
                                'latency_keys': ['asr_seconds', 'queue_wait_seconds'],
                                'model': 'mlx-community/whisper-large-v3-turbo',
                                'preprocessing': {'outcome': 'suspect', 'retry_canonical': True},
                                'profile': 'auto',
                                'prompt': {'disabled': 'no glossary terms'},
                                'provenance': 'retry',
                                'raw_kept': False,
                                'text': 'difference difference difference difference difference '
                                        'difference difference difference difference difference '
                                        'difference difference difference difference difference '
                                        'difference difference difference difference difference '
                                        'difference difference difference difference difference '
                                        'difference difference difference difference difference '
                                        'difference difference difference difference difference '
                                        'difference difference difference difference difference',
                                'vad': {'available': None,
                                        'span_count': None,
                                        'speech_fraction': None}}],
                   'totals': [],
                   'ui': ['hide_if_transcribing'],
                   'undone': 0},
 'h_retry_deleted': {'appends': [],
                     'asr_calls': 0,
                     'copied': [],
                     'delivered': [],
                     'errors': [],
                     'injected': [],
                     'log': ['↻ retry skipped — entry deleted'],
                     'lookups': ['gone'],
                     'nemotron_calls': 0,
                     'refreshed': 0,
                     'retries': [],
                     'totals': [],
                     'ui': ['hide_if_transcribing'],
                     'undone': 0},
 'i_shutdown_mid_job': {'appends': [],
                        'asr_calls': 1,
                        'copied': [],
                        'delivered': [],
                        'errors': [],
                        'injected': [],
                        'log': [],
                        'lookups': [],
                        'nemotron_calls': 0,
                        'refreshed': 0,
                        'retries': [],
                        'totals': [],
                        'ui': ['hide_if_transcribing'],
                        'undone': 0},
 'j_nemotron_stream': {'appends': [{'adaptive': False,
                                    'entry_id': None,
                                    'language': 'en',
                                    'latency_keys': ['asr_seconds',
                                                     'queue_wait_seconds',
                                                     'release_to_text_seconds'],
                                    'model': 'nvidia/nemotron-speech-streaming-en-0.6b',
                                    'prepared_seconds': 2.0,
                                    'preprocessing': {'resampled_normalized': True,
                                                      'silence_collapsed': False,
                                                      'streaming': True,
                                                      'streaming_retry': False,
                                                      'vad_trimmed': False},
                                    'profile': 'nemotron-en',
                                    'prompt': {'disabled': 'Nemotron streaming does not use '
                                                           'Whisper glossary prompts'},
                                    'provenance': 'live',
                                    'raw_kept': True,
                                    'raw_sample_rate': 16000.0,
                                    'seconds': 2.0,
                                    'text': 'Please call me back at four',
                                    'vad': {'available': True,
                                            'span_count': 0,
                                            'speech_fraction': 1.0}}],
                       'asr_calls': 0,
                       'copied': [],
                       'delivered': ['inject_when_clear'],
                       'errors': [],
                       'injected': ['Please call me back at four'],
                       'log': ['→ Ns after release (speech model Ns) · 27 chars'],
                       'lookups': [],
                       'nemotron_calls': 0,
                       'refreshed': 1,
                       'retries': [],
                       'totals': [('Please call me back at four', 2.0)],
                       'ui': ['hide_if_transcribing'],
                       'undone': 0}}


class TemporaryUserFiles:
    """Point the dictionary and settings at a temp folder for each test."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        for target, name in ((dictionary, "DICTIONARY_PATH"), (settings, "SETTINGS_PATH")):
            patcher = patch.object(target, name, Path(folder.name) / f"{name.lower()}.txt")
            patcher.start()
            self.addCleanup(patcher.stop)


class TranscriptionWorkerGoldenMaster(TemporaryUserFiles, unittest.TestCase):
    """Scenarios a-j from the step-2 spec, recorded from the original closure."""

    def check(self, key: str, harness: Harness):
        observed = harness.observed()
        if RECORD:
            import pprint
            print(f"\n### {key}\n{pprint.pformat(observed, width=100, sort_dicts=True)}")
            return
        self.assertEqual(observed, EXPECTED[key])

    def test_a_whisper_live_normal_text(self):
        h = Harness(says("Please call me back at four"))
        self.check("a_whisper_live_normal", h.run_jobs(h.live_job(speech(2.0))))

    def test_b1_loop_after_real_dictation_is_salvaged(self):
        h = Harness(says(f"{PREFIX} {LOOP}"))
        self.check("b1_loop_salvaged", h.run_jobs(h.live_job(speech(12.0))))

    def test_b2_pure_loop_is_held(self):
        h = Harness(says(LOOP))
        self.check("b2_pure_loop_held", h.run_jobs(h.live_job(speech(2.0))))

    def test_c_empty_transcript(self):
        h = Harness(says("   "))
        self.check("c_empty_transcript", h.run_jobs(h.live_job(speech(2.0))))

    def test_d_dead_route_never_reaches_the_asr(self):
        h = Harness(says("should never be called"))
        self.check("d_dead_route", h.run_jobs(h.live_job(np.zeros(32_000, dtype=np.float32))))

    def test_e_sub_02s_capture_is_dropped(self):
        h = Harness(says("should never be called"))
        self.check("e_sub_02s_dropped", h.run_jobs(h.live_job(speech(0.1))))

    def test_f_scratch_that_undoes(self):
        h = Harness(says("Scratch that."))
        self.check("f_scratch_that", h.run_jobs(h.live_job(speech(1.5))))

    def test_g1_retry_commits_cleaned_text_and_copies(self):
        h = Harness(says("Um, please call me back at four"), snapshot=snapshot(speech(2.0)))
        self.check("g1_retry_copied", h.run_jobs(h.retry_job()))

    def test_g2_retry_of_held_text_is_not_copied(self):
        h = Harness(says(LOOP), snapshot=snapshot(speech(2.0)))
        self.check("g2_retry_held", h.run_jobs(h.retry_job()))

    def test_h_retry_of_a_deleted_entry_is_skipped(self):
        h = Harness(says("should never be called"), snapshot=None)
        self.check("h_retry_deleted", h.run_jobs(h.retry_job("gone")))

    def test_i_shutdown_mid_job_appends_and_delivers_nothing(self):
        holder: dict = {}

        def finish_then_shutdown(samples, **kwargs):
            holder["harness"].shutdown.request()
            return {"text": "Please call me back at four", "segments": []}
        h = holder["harness"] = Harness(finish_then_shutdown)
        self.check("i_shutdown_mid_job", h.run_jobs(h.live_job(speech(2.0))))

    def test_j_nemotron_stream_path(self):
        h = Harness(nemotron=True)
        stream = FakeStream("Please call me back at four")
        h.run_jobs(h.live_job(speech(2.0), stream=stream))
        self.assertEqual((stream.finished, stream.closed), (1, 0))
        self.check("j_nemotron_stream", h)


FAILED_TEXT = "[transcription failed]"


class FailureHandlingTests(TemporaryUserFiles, unittest.TestCase):
    """F4 and F16(a): neither a failed speech model nor a failed History write
    may cost the user the dictation."""

    def assert_kept_for_retry(self, h: Harness, raw, error_type: str):
        self.assertEqual(len(h.coordinator.appended), 1, h.logs)
        row = h.coordinator.appended[0]
        self.assertEqual((row["provenance"], row["text"]), ("live_suspect", FAILED_TEXT))
        self.assertIs(row["raw_samples"], raw)
        self.assertEqual(row["preprocessing"]["outcome"], "suspect")
        self.assertIn(error_type, row["preprocessing"]["error"])
        self.assertEqual(h.injected, [])
        self.assertEqual([title for title, _ in h.status_ui.errors], ["Transcription failed"])
        self.assertIn("History", h.status_ui.errors[0][1])
        self.assertTrue(any(error_type in line for line in h.logs), h.logs)
        self.assertEqual(h.refreshed, 1)

    def test_f4_whisper_exception_keeps_the_recording_and_tells_the_user(self):
        def boom(samples, **kwargs):
            raise RuntimeError("[metal::malloc] Resource limit exceeded")
        h, raw = Harness(boom), speech(2.0)
        h.run_jobs(h.live_job(raw))
        self.assert_kept_for_retry(h, raw, "RuntimeError")

    def test_f4_nemotron_stream_failure_keeps_the_recording_and_tells_the_user(self):
        class FailingStream(FakeStream):
            def finish(self):
                self.finished += 1
                raise OSError("nemotron native stream failed after its canonical retry")
        h, raw = Harness(nemotron=True), speech(2.0)
        h.run_jobs(h.live_job(raw, stream=FailingStream("never delivered")))
        self.assert_kept_for_retry(h, raw, "OSError")
        self.assertEqual(h.nemotron.calls, 0)  # the stream already retried once

    def test_f4_nemotron_failure_keeps_the_canonical_bytes_the_engine_heard(self):
        heard = prepare_canonical(np.full(32_000, 0.4, dtype=np.float32))  # boosted, unlike raw

        class FailingStream(FakeStream):
            prepared = heard

            def finish(self):
                raise OSError("nemotron native stream failed after its canonical retry")
        kept = []

        def keep(text, prepared, seconds, model, **kw):
            kept.append(prepared)
            return {"id": "row1", "text": text, "provenance": kw.get("provenance")}
        h = Harness(nemotron=True, append=keep)
        h.run_jobs(h.live_job(speech(2.0), stream=FailingStream("never delivered")))
        self.assertEqual(len(kept), 1, h.logs)
        self.assertIs(kept[0], heard)

    def test_f16_a_dead_route_row_that_cannot_be_saved_says_so_once(self):
        def disk_full(*args, **kwargs):
            raise OSError(28, "No space left on device")
        h = Harness(append=disk_full)
        h.run_jobs(h.live_job(np.zeros(32_000, dtype=np.float32)))
        self.assertEqual([title for title, _ in h.status_ui.errors], ["History could not be saved"])
        self.assertEqual(h.injected, [])

    def test_f4_after_shutdown_nothing_is_kept_or_shown(self):
        holder: dict = {}

        def fail_after_shutdown(samples, **kwargs):
            holder["harness"].shutdown.request()
            raise RuntimeError("[metal::malloc] Resource limit exceeded")
        h = holder["harness"] = Harness(fail_after_shutdown)
        h.run_jobs(h.live_job(speech(2.0)))
        self.assertEqual((h.coordinator.appended, h.status_ui.errors, h.injected), ([], [], []))

    def test_f16_history_write_failure_still_delivers_the_text_and_tells_the_user(self):
        def disk_full(*args, **kwargs):
            raise OSError(28, "No space left on device")
        h = Harness(says("Please call me back at four"), append=disk_full)
        h.run_jobs(h.live_job(speech(2.0)))
        self.assertEqual(h.injected, ["Please call me back at four"])
        self.assertEqual([title for title, _ in h.status_ui.errors], ["History could not be saved"])
        self.assertTrue(any("OSError" in line for line in h.logs), h.logs)

    def test_f16_held_text_stays_off_the_cursor_when_history_fails(self):
        def disk_full(*args, **kwargs):
            raise OSError(28, "No space left on device")
        h = Harness(says(LOOP), append=disk_full)
        h.run_jobs(h.live_job(speech(2.0)))
        self.assertEqual((h.injected, h.copied), ([], []))
        self.assertEqual([title for title, _ in h.status_ui.errors], ["History could not be saved"])

    def test_f16_after_shutdown_nothing_is_delivered_or_shown(self):
        holder: dict = {}

        def disk_full_after_shutdown(*args, **kwargs):
            holder["harness"].shutdown.request()
            raise OSError(28, "No space left on device")
        h = holder["harness"] = Harness(says("Please call me back at four"), append=disk_full_after_shutdown)
        h.run_jobs(h.live_job(speech(2.0)))
        self.assertEqual((h.injected, h.status_ui.errors), ([], []))


def read_only_history():
    """What History hands back while history.jsonl is unreadable (F16b)."""
    return sotto.UnsavedHistory(types.SimpleNamespace(unreadable="damaged line 3"), None).append_live


class RoundTwoWorkerTests(TemporaryUserFiles, unittest.TestCase):
    """F4b, N10, N16, N17 and N19: a live job or a Retry that fails anywhere is
    kept or reported once, and nothing claims History holds what it does not."""

    def test_n16_the_paste_knows_whether_history_saved_the_text(self):
        def disk_full(*args, **kwargs):
            raise OSError(28, "No space left on device")
        saved = Harness(says("Please call me back at four"))
        failed = Harness(says("Please call me back at four"), append=disk_full)
        read_only = Harness(says("Please call me back at four"), append=read_only_history())
        for h in (saved, failed, read_only):
            h.run_jobs(h.live_job(speech(2.0)))
        self.assertEqual([h.delivered[0][1][2:] for h in (saved, failed, read_only)],
                         [(True,), (False,), (False,)])


if __name__ == "__main__":
    unittest.main(verbosity=2)

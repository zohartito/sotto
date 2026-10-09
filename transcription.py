"""The single-flight transcription worker.

One prewarmed worker, strict FIFO: live dictations and retries can never run
the model (or the clipboard cycle) concurrently, and results arrive in capture
order. Jobs are tuples whose first element names the kind: ``live``,
``retry``, ``warmup``, ``stream-audio`` and ``stream-close``.

This is ``run().transcribe_worker`` from sotto.py moved into a class with its
collaborators injected (History coordinator, delivery and UI callbacks, model
handles), so tests can build it with fakes; the golden master in
tests/test_transcription_worker.py pins its observable behaviour.

Imports: the helpers below are bound from sotto at import time, and sotto
imports ``TranscriptionWorker`` inside ``run()`` rather than at module scope,
so there is no cycle at load whichever module is imported first.
"""
from __future__ import annotations

import queue
import time

import numpy as np

from pipeline import screen
from sotto import (DEAD_ROUTE_TEXT, LONG_CAPTURE_S, SAMPLE_RATE, LocalWhisper, _transcription_kwargs,
                   adaptive_live_transcribe_prepared, asr_skip_reason, collapse_silence,
                   discard_staged_adaptive_live_audio, finalize_primary_live_delivery,
                   prepare_for_whisper, transcribe_canonical_samples, transcribe_prepared,
                   warm_speech_runtime)

# A live capture whose speech model failed (F4): the audio is kept under this
# text so Retry can transcribe it again.
TRANSCRIPTION_FAILED_TEXT = "[transcription failed]"


class TranscriptionWorker:
    """Runs transcription jobs from ``jobs`` on the calling thread until
    ``shutdown`` is requested. Everything it needs is passed to ``__init__``."""

    def __init__(self, *, shutdown, jobs: queue.Queue, log, model: str, mlx_whisper, nemotron,
                 current_speech_config, coordinator, refresh_history, adaptive_runtime,
                 adaptive: bool, use_nemotron: bool, status_ui, ui_call, inject_when_clear,
                 undo_when_clear, copy_text, record_totals, model_activity: dict,
                 glossary_terms: tuple[str, ...]) -> None:
        self.shutdown = shutdown
        self.jobs = jobs
        self.log = log
        self.model = model
        self.mlx_whisper = mlx_whisper
        self.nemotron = nemotron
        self.current_speech_config = current_speech_config
        self.coordinator = coordinator
        self.refresh_history = refresh_history
        self.adaptive_runtime = adaptive_runtime
        self.adaptive = adaptive
        self.use_nemotron = use_nemotron
        self.status_ui = status_ui
        self.ui_call = ui_call
        self.inject_when_clear = inject_when_clear
        self.undo_when_clear = undo_when_clear
        self.copy_text = copy_text
        self.record_totals = record_totals
        self.model_activity = model_activity
        self.glossary_terms = glossary_terms
        self.vad_warnings: set[str] = set()
        # Comparator publication state of the job in flight. The job methods
        # set it as they go; run()'s cleanup reads it after they return or raise.
        self._publication = None
        self._publication_adopted = False
        self._publication_committed = False

    def run(self) -> None:
        """The worker loop. Blocks until shutdown is requested.

        Side effects: dispatches every job (History appends, deliveries, UI)
        and hides the transcribing indicator after each non-stream job."""
        while not self.shutdown.requested():
            try:
                job = self.jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            self._publication = None
            self._publication_adopted = False
            self._publication_committed = False
            try:
                if self.shutdown.requested():
                    continue
                if job[0] == "stream-audio":
                    self._stream_audio(job)
                elif job[0] == "stream-close":
                    self._stream_close(job)
                elif job[0] == "warmup":
                    self._warmup()
                elif job[0] == "live":
                    self._live(job)
                elif job[0] == "retry":
                    self._retry(job)
                else:
                    raise RuntimeError("unknown transcription job")
            except Exception as exc:
                self.log(f"! transcription failed: {str(exc)[:160]}")
            finally:
                if (self.adaptive_runtime is not None and self._publication is not None
                        and not self._publication_adopted and not self._publication_committed):
                    self.adaptive_runtime.cancel_comparator_publication({"comparator_publication": self._publication})
                if self.status_ui is not None and job[0] not in {"stream-audio", "stream-close"}:
                    self.ui_call(self.status_ui.hide_if_transcribing)
                self.jobs.task_done()

    def _stream_audio(self, job) -> None:
        """Feed one captured block to a Nemotron stream (Nemotron runs its
        inference here, on the worker, never on the audio tap)."""
        _, stream, block, rate = job
        stream.feed(block, rate)

    def _stream_close(self, job) -> None:
        """Close a stream whose capture was discarded or too short."""
        job[1].close()

    def _warmup(self) -> None:
        """Single-flight model rewarm after idle, serialized with dictation.

        Side effects: logs; resets ``model_activity``."""
        try:
            warm_speech_runtime(adaptive=False, model=self.model,
                                speech_config=self.current_speech_config(),
                                mlx_whisper=self.mlx_whisper)
            self.log("✓ model refreshed after idle")
        finally:
            self.model_activity["last_finished"] = time.monotonic()
            self.model_activity["rewarming"] = False

    def _live(self, job) -> None:
        """One push-to-talk capture: prepare, guard, transcribe, screen, persist,
        deliver.

        Side effects: appends a ``live`` or ``live_suspect`` History row,
        schedules the cursor insertion (or the "scratch that" undo) via
        ``ui_call``, records progress totals, refreshes the History menu, logs.
        Nothing is appended, delivered or persisted once shutdown is requested."""
        _, raw, native_rate, captured_ts, queued_at, capture_id, job_config, stream = job
        started = time.monotonic()
        stream_prepared = None
        stream_text = None
        if stream is not None and raw.any():
            try:
                stream_text, stream_prepared = stream.finish()
            except Exception as exc:
                # The native stream and its one canonical retry both failed;
                # the capture is still in hand, so keep it for Retry (F4).
                self._keep_failed_live_capture(
                    exc, raw=raw, native_rate=native_rate, captured_ts=captured_ts, queued_at=queued_at,
                    job_config=job_config,
                    preprocessing={"resampled_normalized": True, "vad_trimmed": False,
                                   "silence_collapsed": False, "streaming": True,
                                   "streaming_retry": stream.fallback})
                return
        elif stream is not None:
            stream.close()
        samples = (stream_prepared.asr_samples if stream_prepared is not None
                   else prepare_for_whisper(raw, native_rate))
        seconds = len(samples) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
        if seconds < 0.2:
            self.log("○ sub-0.2s capture dropped (key-tap artifact)")
            return
        skip = asr_skip_reason(samples)
        if skip:
            # Exact zeros never reach Whisper. This is not an
            # energy gate: quiet whispered dictation is non-zero
            # and still goes to the ASR (2026-08-13/14).
            from audio_codec import prepare_canonical
            prepared = prepare_canonical(samples)
            if self.shutdown.requested():
                return
            text = DEAD_ROUTE_TEXT
            _, attempt_metadata = _transcription_kwargs(
                job_config, self.glossary_terms, seconds)
            attempt_metadata.update({
                "vad": {"available": False, "speech_fraction": 0.0,
                        "span_count": 0},
                "preprocessing": {"resampled_normalized": True,
                                  "vad_trimmed": False,
                                  "silence_collapsed": False,
                                  "outcome": "suspect"},
                "latency": {"queue_wait_seconds": round(
                    time.monotonic() - queued_at, 4),
                            "asr_seconds": 0.0},
            })
            self.log(f"! not pasted ({skip})")
            finalize_primary_live_delivery(
                append=lambda: self.coordinator.append_live(
                    text, prepared, seconds, self.model,
                    ts=captured_ts, raw_samples=raw,
                    raw_sample_rate=native_rate,
                    provenance="live_suspect",
                    adaptive=False, **attempt_metadata),
                adaptive_runtime=None,
                appended_publication=None,
                shutdown=self.shutdown, inject=None)
            self.refresh_history()
            self.log(f"→ 0.00s · {len(text)} chars")
            return
        # VAD is advisory only: it trims dead air from long
        # captures and feeds the output-side no-speech verdict.
        # It must never block transcription — Silero scores real
        # whispered dictation at 0% speech, so a deliberate
        # capture always reaches the ASR and the transcript
        # decides what happens (2026-08-13/14 incidents).
        vad_available = True
        speech_fraction = 1.0
        speech_spans: list = []
        try:
            import vad
            speech_fraction, speech_spans = vad.analyze(samples)
        except Exception as exc:
            vad_available = False
            reason = str(exc)[:80]
            if reason not in self.vad_warnings:  # once per cause, not every dictation
                self.vad_warnings.add(reason)
                self.log(f"○ advisory VAD unavailable ({reason}); transcribing without trimming")
        vad_metadata = {"available": vad_available,
                        "speech_fraction": round(float(speech_fraction), 4),
                        "span_count": len(speech_spans)}
        preprocessing = {"resampled_normalized": True,
                         "vad_trimmed": False,
                         "silence_collapsed": False}
        if stream is not None:
            preprocessing.update({"streaming": True,
                                  "streaming_retry": stream.fallback})
        if speech_spans and seconds > 10 and not self.use_nemotron:
            before = seconds
            samples = vad.trim_to_speech(samples, speech_spans)
            trimmed = before - len(samples) / SAMPLE_RATE
            preprocessing["vad_trimmed"] = trimmed > 0
            if trimmed > 1:
                self.log(f"  trimmed {trimmed:.0f}s of non-speech (VAD)")
        if len(samples) / SAMPLE_RATE > LONG_CAPTURE_S and not self.use_nemotron:
            before = len(samples)
            samples = collapse_silence(samples)
            preprocessing["silence_collapsed"] = len(samples) < before
            if len(samples) < before:
                self.log(f"  trimmed {(before - len(samples)) / SAMPLE_RATE:.0f}s of dead air")
        from audio_codec import prepare_canonical
        prepared = stream_prepared if stream_prepared is not None else prepare_canonical(samples)
        if self.shutdown.requested():
            return
        staged_live_path = None
        entry_adaptive = self.adaptive
        if self.adaptive_runtime is not None:
            try:
                text, adaptive_metadata, staged_live_path = adaptive_live_transcribe_prepared(
                    self.adaptive_runtime,prepared,capture_id)
                self._publication=adaptive_metadata.pop("comparator_publication",None)
                attempt_metadata = {"profile": "adaptive-en", "language": "en",
                                    "prompt": {"disabled": "adaptive candidate policy"},
                                    "candidate": adaptive_metadata}
                actual_model = adaptive_metadata["repo"]
            except Exception as exc:
                discard_staged_adaptive_live_audio(self.adaptive_runtime.history,capture_id)
                if self.shutdown.requested():
                    return
                # An exception can escape after a durable comparator
                # intent/spool was created.  That state is owned by
                # the store's recovery machinery: guarantee the
                # deferred lease-window sweep FIRST (a failed
                # immediate reconcile must not prevent it), then
                # attempt prompt recovery too.
                try:
                    self.adaptive_runtime.schedule_orphan_sweep(is_shutdown=self.shutdown.requested)
                except Exception:
                    pass
                try:
                    self.adaptive_runtime.reconcile(retry_pending=False)
                except Exception:
                    pass
                # Adaptive authority drift (stale receipts, changed
                # runtime) must never cost the user their capture:
                # degrade to plain baseline dictation.  The result
                # enters ordinary non-adaptive history only, so
                # nothing reaches the adaptive lane without valid
                # receipts and no gate is weakened.
                self.log(f"! adaptive lane unavailable ({str(exc)[:80]}) — baseline dictation only")
                import mlx_whisper as mlx_whisper_fallback
                text, attempt_metadata = transcribe_prepared(
                    LocalWhisper(mlx_whisper_fallback, self.model), prepared, job_config, self.glossary_terms)
                actual_model = self.model
                entry_adaptive = False
                self._publication = None
        else:
            try:
                if self.use_nemotron:
                    from nemotron_backend import metadata
                    text = stream_text if stream_text is not None else self.nemotron.transcribe(prepared.asr_samples)
                    attempt_metadata = metadata()
                else:
                    text, attempt_metadata = transcribe_prepared(
                        self.mlx_whisper, prepared, job_config, self.glossary_terms)
            except Exception as exc:
                # A Metal/MLX or native error must not cost the user the
                # recording (F4). The adaptive lane above keeps its own fallback.
                self._keep_failed_live_capture(
                    exc, raw=raw, native_rate=native_rate, captured_ts=captured_ts, queued_at=queued_at,
                    job_config=job_config, prepared=prepared, vad_metadata=vad_metadata,
                    preprocessing=preprocessing)
                return
            actual_model = self.model
        # A backend may ignore cancellation.  Its result is never
        # pasted, registered, or persisted after shutdown.
        if self.shutdown.requested():
            if self.adaptive_runtime is not None:
                discard_staged_adaptive_live_audio(self.adaptive_runtime.history,capture_id)
            return
        elapsed = time.monotonic() - started
        self.model_activity["last_finished"] = time.monotonic()
        detected_language = attempt_metadata.pop("detected_language", None)
        if detected_language:
            preprocessing["detected_language"] = detected_language
        attempt_metadata.update({"vad": vad_metadata,
                                 "preprocessing": preprocessing,
                                 "latency": {"queue_wait_seconds": round(started - queued_at, 4),
                                             "asr_seconds": round(elapsed, 4)}})
        prepared_seconds = len(prepared.asr_samples) / SAMPLE_RATE
        # Guards, then the user's dictionary and English cleanup
        # (pipeline.screen, shared with Retry and Windows). Empty
        # output becomes the no-speech marker, so the text going
        # in is never empty; "scratch that" may clean down to "".
        screened = screen(text, seconds=prepared_seconds, speech_fraction=speech_fraction,
                          language=preprocessing.get("detected_language") or job_config.language,
                          clean=not entry_adaptive, log=self.log)
        text, reason, voice_action = screened.text, screened.held, screened.voice_action
        preprocessing.update(screened.receipts)
        if not self.shutdown.requested():
            appended_row = None
            attempt_metadata["latency"]["release_to_text_seconds"] = round(
                time.monotonic() - queued_at, 4)
            if self.shutdown.requested():
                return
            publication_meta={"comparator_publication":self._publication} if self._publication is not None else {}
            if self._publication is not None and not self.adaptive_runtime.adopt_comparator_publication(
                    publication_meta,history_id=capture_id,history_revision=0):
                self.adaptive_runtime.cancel_comparator_publication(publication_meta)
                self.adaptive_runtime.fail_comparator_publication(publication_meta)
                raise RuntimeError("comparator publication adoption unavailable")
            if reason:
                self.log(f"! not pasted ({reason}) — kept in history")
                preprocessing["outcome"] = "suspect"
                try:
                    appended_row = finalize_primary_live_delivery(
                        append=lambda: self.coordinator.append_live(
                            text, prepared, prepared_seconds, actual_model,
                            ts=captured_ts, raw_samples=raw,
                            raw_sample_rate=native_rate, provenance="live_suspect",
                            adaptive=entry_adaptive, entry_id=capture_id if entry_adaptive else None, **attempt_metadata),
                        adaptive_runtime=self.adaptive_runtime,appended_publication=publication_meta,
                        shutdown=self.shutdown,inject=None)
                    self._publication_committed = appended_row is not None
                except Exception:
                    if self.adaptive_runtime is not None: discard_staged_adaptive_live_audio(self.adaptive_runtime.history,capture_id)
                    raise
            else:
                if self.shutdown.requested():
                    return
                try:
                    appended_row = finalize_primary_live_delivery(
                        append=lambda: self.coordinator.append_live(
                            text, prepared, prepared_seconds, actual_model, ts=captured_ts,
                            raw_samples=raw, raw_sample_rate=native_rate,
                            provenance="live", adaptive=entry_adaptive,
                            entry_id=capture_id if entry_adaptive else None, **attempt_metadata),
                        adaptive_runtime=self.adaptive_runtime,appended_publication=publication_meta,
                        shutdown=self.shutdown,
                        inject=(lambda: self.ui_call(self.undo_when_clear, 0)) if voice_action == "scratch"
                        else (lambda: self.ui_call(self.inject_when_clear, text, 0)) if text else None)
                    self._publication_committed = appended_row is not None
                except Exception:
                    if self.adaptive_runtime is not None: discard_staged_adaptive_live_audio(self.adaptive_runtime.history,capture_id)
                    raise
            self._publication_adopted = (appended_row is not None and not self.shutdown.requested())
            if appended_row is not None and text and not reason and voice_action is None:
                self.record_totals(text, len(raw) / max(native_rate, 1.0))
            self.refresh_history()
        elif self.adaptive_runtime is not None:
            # No History append adopted this pre-publication WAV.
            if self._publication is not None:
                self.adaptive_runtime.cancel_comparator_publication({"comparator_publication":self._publication})
            discard_staged_adaptive_live_audio(self.adaptive_runtime.history,capture_id)
        self.log(f"→ {time.monotonic() - queued_at:.2f}s after release "
                 f"(speech model {elapsed:.2f}s) · {len(text)} chars")

    def _keep_failed_live_capture(self, exc: Exception, *, raw, native_rate, captured_ts, queued_at,
                                  job_config, preprocessing: dict, prepared=None, vad_metadata=None) -> None:
        """The speech model failed on a live capture (F4). The recording is not
        lost: it becomes a ``live_suspect`` History row reading
        TRANSCRIPTION_FAILED_TEXT, so Retry can transcribe it again.

        Side effects: appends that row with the raw audio and the error in its
        preprocessing receipt, refreshes the History menu, logs the error, and
        shows one alert. Nothing once shutdown is requested."""
        if self.shutdown.requested():
            return
        error = f"{type(exc).__name__}: {str(exc)[:160]}"
        self.log(f"! transcription failed ({error}) — audio kept in History for Retry")
        if prepared is None:  # the stream failed before any canonical audio existed
            from audio_codec import prepare_canonical
            prepared = prepare_canonical(prepare_for_whisper(raw, native_rate))
        seconds = len(prepared.asr_samples) / SAMPLE_RATE
        _, attempt_metadata = _transcription_kwargs(job_config, self.glossary_terms, seconds)
        attempt_metadata.update({
            "vad": vad_metadata or {"available": None, "speech_fraction": None, "span_count": None},
            "preprocessing": {**preprocessing, "outcome": "suspect", "error": error},
            "latency": {"queue_wait_seconds": round(time.monotonic() - queued_at, 4), "asr_seconds": 0.0},
        })
        try:
            finalize_primary_live_delivery(
                append=lambda: self.coordinator.append_live(
                    TRANSCRIPTION_FAILED_TEXT, prepared, seconds, self.model, ts=captured_ts,
                    raw_samples=raw, raw_sample_rate=native_rate, provenance="live_suspect",
                    adaptive=False, **attempt_metadata),
                adaptive_runtime=None, appended_publication=None, shutdown=self.shutdown, inject=None)
        except Exception as append_exc:
            self.log(f"! failed capture not saved ({type(append_exc).__name__}: {str(append_exc)[:160]})")
            self._show_error("Transcription failed",
                             f"{error}\n\nThe recording could not be saved to History either; please dictate again.")
            return
        self.refresh_history()
        self._show_error("Transcription failed", f"The audio is in History; use Retry.\n\n{error}")

    def _show_error(self, title: str, message: str) -> None:
        """One alert on AppKit's main thread, when there is a menu bar to show it.

        Side effects: the alert (ui.show_error also logs it); none after shutdown."""
        if self.status_ui is not None and not self.shutdown.requested():
            self.ui_call(self.status_ui.show_error, title, message)

    def _retry(self, job) -> None:
        """Transcribe a History row's saved canonical audio again.

        Side effects: commits the new text to the row (``commit_retry``),
        copies deliverable text to the clipboard via ``ui_call``, refreshes
        the History menu, logs. Nothing is committed or copied once shutdown
        is requested."""
        _, entry_id, queued_at, capture_id, job_config = job
        snapshot = self.coordinator.snapshot_for_retry(entry_id)
        if snapshot is None:
            self.log("↻ retry skipped — entry deleted")
            return
        prepared_seconds = len(snapshot.samples) / SAMPLE_RATE
        # Retry uses the immutable canonical decode from its snapshot;
        # it must never re-trim or re-collapse persisted inference input.
        if asr_skip_reason(snapshot.samples):
            text = DEAD_ROUTE_TEXT
            _, attempt_metadata = _transcription_kwargs(
                job_config, self.glossary_terms, prepared_seconds)
            attempt_metadata.update({
                "vad": {"available": None, "speech_fraction": None,
                        "span_count": None},
                "preprocessing": {"retry_canonical": True,
                                  "outcome": "suspect"},
                "latency": {"queue_wait_seconds": round(
                    time.monotonic() - queued_at, 4),
                            "asr_seconds": 0.0},
            })
            self.coordinator.commit_retry(
                entry_id, text, snapshot.expected_revision,
                model=self.model, provenance="retry",
                **attempt_metadata)
            self.log("↻ retry skipped — dead microphone (all-zero capture)")
            self.refresh_history()
            return
        retry_started = time.monotonic()
        if self.shutdown.requested():
            return
        if self.adaptive_runtime is not None:
            text, adaptive_metadata = self.adaptive_runtime.live_transcribe(
                snapshot.samples, canonical_path=snapshot.inference_audio_path,
                canonical_identity=snapshot.inference_identity, capture_id=capture_id)
            self._publication=adaptive_metadata.pop("comparator_publication",None)
            attempt_metadata = {"profile": "adaptive-en", "language": "en",
                                "prompt": {"disabled": "adaptive candidate policy"},
                                "candidate": adaptive_metadata}
            retry_model = adaptive_metadata["repo"]
        else:
            text, attempt_metadata = transcribe_canonical_samples(
                self.mlx_whisper, snapshot.samples, job_config, self.glossary_terms, nemotron=self.nemotron)
            retry_model = self.model
        self.model_activity["last_finished"] = time.monotonic()
        if self.shutdown.requested():
            return
        retry_preprocessing = {"retry_canonical": True}
        detected_language = attempt_metadata.pop("detected_language", None)
        if detected_language:
            retry_preprocessing["detected_language"] = detected_language
        held = None
        if self.adaptive_runtime is None:
            # The same guards and cleanup as live dictation. Retry has
            # no VAD score (speech_fraction 1.0) and never undoes
            # anything, so a "scratch that" result keeps its words.
            screened = screen(text, seconds=len(snapshot.samples) / SAMPLE_RATE,
                              speech_fraction=1.0,
                              language=retry_preprocessing.get("detected_language")
                              or job_config.language, log=self.log)
            retry_preprocessing.update(screened.receipts)
            text = (retry_preprocessing.pop("voice")["asr_text"]
                    if screened.voice_action == "scratch" else screened.text)
            held = screened.held
            if held:
                retry_preprocessing["outcome"] = "suspect"
        attempt_metadata.update({"vad": {"available": None, "speech_fraction": None,
                                           "span_count": None},
                                 "preprocessing": retry_preprocessing,
                                 "latency": {"queue_wait_seconds": round(retry_started - queued_at, 4),
                                             "asr_seconds": round(time.monotonic() - retry_started, 4)}})
        if self.shutdown.requested():
            return
        publication_meta={"comparator_publication":self._publication} if self._publication is not None else {}
        if self._publication is not None and not self.adaptive_runtime.adopt_comparator_publication(
                publication_meta,history_id=entry_id,
                history_revision=snapshot.expected_revision + 1):
            self.adaptive_runtime.cancel_comparator_publication(publication_meta)
            self.adaptive_runtime.fail_comparator_publication(publication_meta)
            raise RuntimeError("comparator publication adoption unavailable")
        committed = self.coordinator.commit_retry(
            entry_id, text, snapshot.expected_revision, model=retry_model,
            provenance="retry", **attempt_metadata)
        self._publication_committed = committed is not None
        if self.adaptive_runtime is not None and self._publication is not None:
            if committed is None:
                self.adaptive_runtime.cancel_comparator_publication(publication_meta)
            elif not self.shutdown.requested() and not self.adaptive_runtime.acknowledge_comparator_publication(publication_meta):
                # Retry's old revision revoke can have an exact
                # scrub marker in flight.  Keep the adopted
                # prepared row for restart reconciliation rather
                # than falsely cancelling a committed delivery.
                if not self.adaptive_runtime.comparator_publication_recovery_pending(publication_meta):
                    self.adaptive_runtime.cancel_comparator_publication(publication_meta)
                    self.adaptive_runtime.fail_comparator_publication(publication_meta)
                    raise RuntimeError("comparator publication unavailable")
            self._publication_adopted = committed is not None and not self.shutdown.requested()
        if held and committed is not None:
            self.log(f"↻ retried — not copied ({held}); kept in History")
        elif text and committed is not None and not self.shutdown.requested():
            self.ui_call(self.copy_text, text)
            self.log(f"↻ retried · {len(text)} chars")
        self.refresh_history()

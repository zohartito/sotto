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
import threading
import time


from pipeline import screen
from sotto import (DEAD_ROUTE_TEXT, LONG_CAPTURE_S, SAMPLE_RATE, LocalWhisper, _transcription_kwargs,
                   adaptive_live_transcribe_prepared, asr_skip_reason, collapse_silence,
                   discard_staged_adaptive_live_audio, finalize_primary_live_delivery,
                   prepare_for_whisper, transcribe_canonical_samples, transcribe_prepared,
                   warm_speech_runtime)

# A live capture whose speech model failed (F4): the audio is kept under this
# text so Retry can transcribe it again.
TRANSCRIPTION_FAILED_TEXT = "[transcription failed]"
# A live capture Sotto stopped before transcribing (N9): kept the same way.
NOT_TRANSCRIBED_TEXT = "[not transcribed: Sotto stopped]"


class TranscriptionWorker:
    """Runs transcription jobs from ``jobs`` on the calling thread until
    ``shutdown`` is requested. Everything it needs is passed to ``__init__``."""

    def __init__(self, *, shutdown, jobs: queue.Queue, log, model: str, mlx_whisper, nemotron,
                 current_speech_config, coordinator, refresh_history, adaptive_runtime,
                 adaptive: bool, use_nemotron: bool, status_ui, ui_call, deliver_call,
                 inject_when_clear, undo_when_clear, copy_text, record_totals, model_activity: dict,
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
        # Delivery (paste, voice undo) is always scheduled; ui_call is a no-op
        # under --no-overlay and must never carry text (F9).
        self.deliver_call = deliver_call
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
        # The job being run, and the live captures it gave up on because
        # shutdown came first: the teardown keeps both (N9).
        self.current_job = None
        self.abandoned: list = []
        # Capture ids whose fate is decided — saved here, dropped as a tap,
        # or kept by the teardown — so no capture is ever saved twice.
        self._settled: set[str] = set()
        self._settle_lock = threading.Lock()
        self._live_id = None
        # The History row the live job in flight got back (None until then),
        # whether the user was already told about it (F4/F16a), and whether
        # the Retry in flight reached commit_retry: the catch-all in run()
        # reads them to keep or report a job that raised (F4b, N17).
        self._live_row = None
        self._live_reported = False
        self._retry_committed = False

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
            self.current_job = job
            self._live_row = None
            self._live_reported = False
            self._retry_committed = False
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
                try:
                    self._job_failed(job, exc)
                except Exception:  # reporting must never end the worker thread
                    self.log(f"! transcription failed: {str(exc)[:160]}")
            finally:
                if (self.adaptive_runtime is not None and self._publication is not None
                        and not self._publication_adopted and not self._publication_committed):
                    self.adaptive_runtime.cancel_comparator_publication({"comparator_publication": self._publication})
                if self.status_ui is not None and job[0] not in {"stream-audio", "stream-close"}:
                    self.ui_call(self.status_ui.hide_if_transcribing)
                if job[0] == "live" and self.shutdown.requested():
                    self.abandoned.append(job)  # kept by the teardown unless saved here
                self.current_job = None
                self.jobs.task_done()

    def _job_failed(self, job, exc: Exception) -> None:
        """A job raised outside the paths that handle their own failures.

        A live capture that never reached History is kept for Retry exactly
        like a failed model call (F4b), whatever step raised: preparation,
        VAD trim, silence collapse, screening, the adaptive lane. A Retry that
        raised before commit_retry tells the user once, as Windows does
        (N17). An error after the result was saved, delivered or already
        reported is logged as that, not as a transcription failure.

        Side effects: may append one History row and show one alert; logs."""
        error = f"{type(exc).__name__}: {str(exc)[:160]}"
        if job[0] == "live" and self._live_row is None and not self._live_reported:
            _, raw, native_rate, captured_ts, queued_at, capture_id, job_config, _stream = job
            if self.adaptive_runtime is not None:
                # The capture is kept below as an ordinary row under a new id;
                # drop any unadopted adaptive staging of it first.
                discard_staged_adaptive_live_audio(self.adaptive_runtime.history, capture_id)
            self._keep_failed_live_capture(exc, raw=raw, native_rate=native_rate, captured_ts=captured_ts,
                                           queued_at=queued_at, job_config=job_config, preprocessing={})
        elif job[0] == "retry" and not self._retry_committed:
            self.log(f"! retry failed ({error}) — the History entry is unchanged")
            self._show_error("Retry failed", f"The History entry is unchanged.\n\n{error}")
        elif job[0] in {"live", "retry"}:
            self.log(f"! {job[0]} result already handled; a later step failed ({error})")
        else:
            self.log(f"! transcription failed: {str(exc)[:160]}")

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
        ``deliver_call``, records progress totals, refreshes the History menu, logs.
        Nothing is appended, delivered or persisted once shutdown is requested."""
        _, raw, native_rate, captured_ts, queued_at, capture_id, job_config, stream = job
        self._live_id = capture_id
        started = time.monotonic()
        stream_prepared = None
        stream_text = None
        if stream is not None and raw.any():
            try:
                stream_text, stream_prepared = stream.finish()
            except Exception as exc:
                # The native stream and its one canonical retry both failed;
                # the capture is still in hand, so keep it for Retry (F4) with
                # the exact canonical bytes the engine heard, when they exist.
                self._keep_failed_live_capture(
                    exc, raw=raw, native_rate=native_rate, captured_ts=captured_ts, queued_at=queued_at,
                    job_config=job_config, prepared=getattr(stream, "prepared", None),
                    preprocessing={"resampled_normalized": True, "vad_trimmed": False,
                                   "silence_collapsed": False, "streaming": True,
                                   "streaming_retry": stream.fallback})
                return
        elif stream is not None:
            stream.close()
        samples = (stream_prepared.asr_samples if stream_prepared is not None
                   else prepare_for_whisper(raw, native_rate))
        seconds = len(samples) / SAMPLE_RATE
        if seconds < 0.2:
            self.log("○ sub-0.2s capture dropped (key-tap artifact)")
            self._settle(capture_id)
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
            try:
                finalize_primary_live_delivery(
                    append=lambda: self._append_live(
                        text, prepared, seconds, self.model,
                        ts=captured_ts, raw_samples=raw,
                        raw_sample_rate=native_rate,
                        provenance="live_suspect",
                        adaptive=False, **attempt_metadata),
                    adaptive_runtime=None,
                    appended_publication=None,
                    shutdown=self.shutdown, inject=None)
            except Exception as exc:
                self._history_append_failed(exc)  # F16(a): held, nothing to deliver
                return
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
                preprocessing["outcome"] = "suspect"
                try:
                    appended_row = finalize_primary_live_delivery(
                        append=lambda: self._append_live(
                            text, prepared, prepared_seconds, actual_model,
                            ts=captured_ts, raw_samples=raw,
                            raw_sample_rate=native_rate, provenance="live_suspect",
                            adaptive=entry_adaptive, entry_id=capture_id if entry_adaptive else None,
                            **attempt_metadata),
                        adaptive_runtime=self.adaptive_runtime,appended_publication=publication_meta,
                        shutdown=self.shutdown,inject=None)
                    self._publication_committed = appended_row is not None
                except Exception as exc:
                    if self.adaptive_runtime is not None:
                        discard_staged_adaptive_live_audio(self.adaptive_runtime.history,capture_id)
                        raise
                    self.log(f"! not pasted ({reason})")
                    self._history_append_failed(exc)  # F16(a): held text; nothing to deliver
                else:
                    # N10: read-only History (F16b) keeps nothing; never claim it did.
                    self.log(f"! not pasted ({reason}) — "
                             + ("kept in history" if self._in_history() else "not kept (History is read-only)"))
            else:
                if self.shutdown.requested():
                    return
                deliver = ((lambda: self.deliver_call(self.undo_when_clear, 0)) if voice_action == "scratch"
                           else (lambda: self.deliver_call(self.inject_when_clear, text, 0, self._in_history()))
                           if text else None)
                try:
                    appended_row = finalize_primary_live_delivery(
                        append=lambda: self._append_live(
                            text, prepared, prepared_seconds, actual_model, ts=captured_ts,
                            raw_samples=raw, raw_sample_rate=native_rate,
                            provenance="live", adaptive=entry_adaptive,
                            entry_id=capture_id if entry_adaptive else None, **attempt_metadata),
                        adaptive_runtime=self.adaptive_runtime,appended_publication=publication_meta,
                        shutdown=self.shutdown, inject=deliver)
                    self._publication_committed = appended_row is not None
                except Exception as exc:
                    if self.adaptive_runtime is not None:
                        # The adaptive lane's row is part of its receipts:
                        # append-before-deliver stays absolute there.
                        discard_staged_adaptive_live_audio(self.adaptive_runtime.history,capture_id)
                        raise
                    self._history_append_failed(exc, deliver)  # F16(a)
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

    def _in_history(self) -> bool:
        """Did History really keep this live job's row? None when the append
        failed (F16a); read-only History hands back a row marked ``saved``
        False (F16b)."""
        return self._live_row is not None and self._live_row.get("saved") is not False

    def _keep_failed_live_capture(self, exc: Exception, *, raw, native_rate, captured_ts, queued_at,
                                  job_config, preprocessing: dict, prepared=None, vad_metadata=None) -> None:
        """The speech model (F4), or any step around it (F4b), failed on a live
        capture. The recording is not lost: it becomes a ``live_suspect``
        History row reading TRANSCRIPTION_FAILED_TEXT, so Retry can transcribe
        it again. If even that row cannot be prepared or saved, the user is
        told to dictate again.

        Side effects: appends that row with the raw audio and the error in its
        preprocessing receipt, refreshes the History menu, logs the error, and
        shows one alert. Nothing once shutdown is requested."""
        if self.shutdown.requested():
            return
        self._live_reported = True
        error = f"{type(exc).__name__}: {str(exc)[:160]}"
        self.log(f"! transcription failed ({error})")
        try:
            if prepared is None:  # no canonical audio exists yet (a stream or a step before it failed)
                from audio_codec import prepare_canonical
                prepared = prepare_canonical(prepare_for_whisper(raw, native_rate))
            seconds = len(prepared.asr_samples) / SAMPLE_RATE
            _, attempt_metadata = _transcription_kwargs(job_config, self.glossary_terms, seconds)
            attempt_metadata.update({
                "vad": vad_metadata or {"available": None, "speech_fraction": None, "span_count": None},
                "preprocessing": {**preprocessing, "outcome": "suspect", "error": error},
                "latency": {"queue_wait_seconds": round(time.monotonic() - queued_at, 4), "asr_seconds": 0.0},
            })
            finalize_primary_live_delivery(
                append=lambda: self._append_live(
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
        if not self._in_history():  # N10: read-only History (F16b) kept nothing
            self.log("! failed capture not saved (History is read-only)")
            self._show_error("Transcription failed",
                             f"{error}\n\nHistory is read-only right now, so the recording was not "
                             "kept; please dictate again.")
            return
        self.log("  audio kept in History for Retry")
        self._show_error("Transcription failed", f"The audio is in History; use Retry.\n\n{error}")

    def keep_untranscribed(self, job) -> None:
        """Sotto is stopping and this live capture was never transcribed (N9):
        Quit, Restart or SIGTERM waited as long as it may. The recording is
        not lost: it becomes a ``live_suspect`` History row reading
        NOT_TRANSCRIBED_TEXT, so Retry can transcribe it at the next start.
        Unlike every other append this runs after shutdown is requested; the
        teardown calls it only for captures nothing else will save.

        Side effects: appends that row with the raw audio; logs. Never raises.
        Nothing when this capture was already saved or kept."""
        _, raw, native_rate, captured_ts, queued_at, capture_id, job_config, _stream = job
        if not self._settle(capture_id):
            return
        try:
            from audio_codec import prepare_canonical
            prepared = prepare_canonical(prepare_for_whisper(raw, native_rate))
            seconds = len(prepared.asr_samples) / SAMPLE_RATE
            _, attempt_metadata = _transcription_kwargs(job_config, self.glossary_terms, seconds)
            attempt_metadata.update({
                "vad": {"available": None, "speech_fraction": None, "span_count": None},
                "preprocessing": {"resampled_normalized": True, "vad_trimmed": False,
                                  "silence_collapsed": False, "outcome": "suspect",
                                  "error": "not transcribed before Sotto stopped"},
                "latency": {"queue_wait_seconds": round(time.monotonic() - queued_at, 4),
                            "asr_seconds": 0.0},
            })
            self.coordinator.append_live(
                NOT_TRANSCRIBED_TEXT, prepared, seconds, self.model, ts=captured_ts,
                raw_samples=raw, raw_sample_rate=native_rate, provenance="live_suspect",
                adaptive=False, **attempt_metadata)
        except Exception as exc:
            self.log(f"! untranscribed capture not saved ({type(exc).__name__}: {str(exc)[:160]})")
            return
        self.log(f"○ {len(raw) / max(native_rate, 1.0):.1f}s capture not transcribed before "
                 "stopping — kept in History for Retry")

    def _settle(self, capture_id) -> bool:
        """Claim a live capture's one History row; False if already claimed."""
        with self._settle_lock:
            if capture_id in self._settled:
                return False
            self._settled.add(capture_id)
            return True

    def _append_live(self, *args, **kwargs):
        """coordinator.append_live for the live capture in hand, remembering
        the row it hands back, unless the teardown already kept it (a model
        that returned too late): then None, so nothing is registered or
        delivered, and an adaptive staging WAV is removed. An append that
        raises saved nothing, so the capture is unsettled again and the
        teardown may still keep it."""
        if not self._settle(self._live_id):
            if self.adaptive_runtime is not None:
                discard_staged_adaptive_live_audio(self.adaptive_runtime.history, self._live_id)
            return None
        try:
            self._live_row = self.coordinator.append_live(*args, **kwargs)
            return self._live_row
        except Exception:
            with self._settle_lock:
                self._settled.discard(self._live_id)
            raise

    def _history_append_failed(self, exc: Exception, deliver=None) -> None:
        """History could not take the live row (F16a): disk full, a locked or
        damaged store. Non-adaptive dictation still reaches the user, who is
        told once that this dictation was not saved.

        Side effects: runs ``deliver`` (the cursor insertion or the "scratch
        that" undo, exactly what a saved row would have scheduled); shows one
        alert; logs the error. Nothing once shutdown is requested."""
        self._live_reported = True
        error = f"{type(exc).__name__}: {str(exc)[:160]}"
        self.log(f"! History append failed ({error})"
                 + (" — delivering the text anyway" if deliver is not None else ""))
        if self.shutdown.requested():
            return
        if deliver is not None:
            deliver()
        outcome = ("The text was still delivered, but it is not in History."
                   if deliver is not None else "The held-back text could not be kept in History.")
        self._show_error("History could not be saved", f"{outcome}\n\n{error}")

    def _show_error(self, title: str, message: str) -> None:
        """One alert on AppKit's main thread, when there is a menu bar to show it.

        Side effects: the alert (ui.show_error also logs it); none after shutdown."""
        if self.status_ui is not None and not self.shutdown.requested():
            self.ui_call(self.status_ui.show_error, title, message)

    def _retry(self, job) -> None:
        """Transcribe a History row's saved canonical audio again.

        Side effects: commits the new text to the row (``commit_retry``),
        copies deliverable text to the clipboard via ``deliver_call``, refreshes
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
            self._retry_committed = True
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
        self._retry_committed = True
        self._publication_committed = committed is not None
        if self.adaptive_runtime is not None and self._publication is not None:
            if committed is None:
                self.adaptive_runtime.cancel_comparator_publication(publication_meta)
            elif (not self.shutdown.requested()
                  and not self.adaptive_runtime.acknowledge_comparator_publication(publication_meta)):
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
            # A delivery like a paste (N19): counted in PendingDeliveries until
            # the main thread runs it, so Quit waits for the copy.
            self.deliver_call(self.copy_text, text)
            self.log(f"↻ retried · {len(text)} chars")
        self.refresh_history()

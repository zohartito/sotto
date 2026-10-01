"""Silero VAD — advisory speech scoring: trims dead air from long captures
and feeds the output-side no-speech verdict (sotto.reads_as_no_speech).

It must never block transcription: Silero scores real whispered (unvoiced)
dictation at 0% speech frames, so any input-side gate built on it silently
eats deliberate captures (2026-08-13/14 incidents). ONNX runtime, ~2 MB
model, <1 ms per 32 ms frame. Reference scores from this machine's capture
history: voiced dictation 15-73%, whispered dictation 0%, the music capture
that transcribed as 'Woo' x200 2%, silence and dead-mic captures 0%.

Model interface (Silero v5+): 512-sample frames at 16 kHz, each prepended
with 64 samples of context from the previous frame, plus a recurrent state.
Without the context prefix the model silently returns ~0 for everything.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np

from sotto_paths import DATA_DIR

MODEL_PATH = DATA_DIR / "models" / "silero_vad.onnx"
# Silero VAD v6.2 (MIT), pinned: installed by `sotto.py setup`, verified on load.
MODEL_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2/src/silero_vad/data/silero_vad.onnx"
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
SAMPLE_RATE = 16_000
FRAME = 512
CONTEXT = 64

_session = None


def _get_session():
    global _session
    if _session is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError("optional Silero VAD model is not installed")
        if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
            raise FileNotFoundError("Silero VAD model does not match the pinned v6.2")
        import onnxruntime as ort
        _session = ort.InferenceSession(str(MODEL_PATH),
                                        providers=["CPUExecutionProvider"])
    return _session


def install(path: Path | None = None, *, fetch=None, allow_download: bool = True) -> Path:
    """Download the pinned model (2.3 MB) unless an identical copy is present.
    Nothing is kept unless its SHA-256 matches; offline, nothing is fetched."""
    path = Path(path or MODEL_PATH)
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA256:
        return path
    if not allow_download:
        raise FileNotFoundError("offline: the pinned Silero VAD is not installed yet")
    if fetch is None:
        import urllib.request
        fetch = lambda: urllib.request.urlopen(MODEL_URL, timeout=60).read()
    data = fetch()
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise RuntimeError("Silero VAD download does not match its pinned SHA-256")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".silero-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def analyze(samples: np.ndarray, threshold: float = 0.5) -> tuple[float, list]:
    """Returns (speech_fraction, speech_spans). Spans are (start, end) sample
    ranges at 16 kHz covering detected speech."""
    session = _get_session()
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(CONTEXT, dtype=np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)
    flags: list[bool] = []
    for i in range(0, len(samples) - FRAME + 1, FRAME):
        chunk = samples[i:i + FRAME].astype(np.float32)
        frame = np.concatenate([context, chunk])[None, :]
        prob, state = session.run(None, {"input": frame, "state": state,
                                         "sr": sr})
        context = chunk[-CONTEXT:]
        flags.append(float(prob[0, 0]) > threshold)
    if not flags:
        return 0.0, []
    fraction = sum(flags) / len(flags)

    pad_frames = 10  # ~0.32s of padding around each speech span
    spans: list[list[int]] = []
    for i, is_speech in enumerate(flags):
        if not is_speech:
            continue
        start = max(0, (i - pad_frames)) * FRAME
        end = min(len(flags), i + 1 + pad_frames) * FRAME
        if spans and start <= spans[-1][1]:
            spans[-1][1] = end
        else:
            spans.append([start, end])
    return fraction, [(s, e) for s, e in spans]


def trim_to_speech(samples: np.ndarray, spans: list) -> np.ndarray:
    """Keep only the detected speech spans (already padded/merged)."""
    if not spans:
        return samples
    return np.concatenate([samples[s:e] for s, e in spans])

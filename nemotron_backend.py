"""Pinned, offline Nemotron English ASR through NVIDIA's stable C ABI.

The recognizer and each stream are driven by Sotto's one transcription worker.
No network, microphone, server, or background decoder is created here.
ABI: NVIDIA/NeMo-Speech.cpp v0.1.0, include/nemo_speech/asr.h.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np

MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
SPEC_PATH = Path(__file__).resolve().parent / "config/nemotron-runtime.json"
from sotto_paths import DATA_DIR

INSTALL_ROOT = DATA_DIR / "nemotron"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def installation(*, verify: bool = True) -> tuple[Path, Path]:
    """Bind every loadable library and the model to source-controlled hashes."""
    spec = json.loads(SPEC_PATH.read_text())
    root = INSTALL_ROOT / spec["version"]
    model = root / spec["model_file"]
    expected = {"lib/" + key: value for key, value in spec["libraries"].items()}
    expected[spec["model_file"]] = spec["model_sha256"]
    if root.is_symlink() or (root / "lib").is_symlink():
        raise RuntimeError("Unsafe Nemotron installation")
    if {p.name for p in (root / "lib").glob("*.dylib")} != set(spec["libraries"]):
        raise RuntimeError("Nemotron is not installed; run scripts/setup_nemotron.py")
    for relative, digest in expected.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("Nemotron is not installed; run scripts/setup_nemotron.py")
        if verify and _sha256(path) != digest:
            raise RuntimeError("Nemotron installation changed: " + relative)
    return root / "lib/libnemo_speech_asr_c.1.dylib", model


def is_installed() -> bool:
    try:
        installation(verify=False)
        return True
    except (OSError, RuntimeError):
        return False


def metadata() -> dict:
    return {"profile": "nemotron-en", "language": "en",
            "prompt": {"disabled": "Nemotron streaming does not use Whisper glossary prompts"}}


class _Backend(C.Structure):
    _fields_ = [("size", C.c_size_t), ("gpu", C.c_int32)]


class _Model(C.Structure):
    _fields_ = [("size", C.c_size_t), ("path", C.c_char_p), ("name", C.c_char_p)]


class _Streaming(C.Structure):
    _fields_ = [("size", C.c_size_t), ("chunk_size", C.c_float),
                ("left", C.c_float), ("right", C.c_float), ("context", C.c_int32)]


class _Endpointing(C.Structure):
    _fields_ = [("size", C.c_size_t), ("enable", C.c_bool),
                ("vad_based", C.c_bool), ("stop_ms", C.c_int32)]


class _Config(C.Structure):
    _fields_ = [("size", C.c_size_t)] + [
        (name, C.c_void_p) for name in
        ("backend", "model", "streaming", "decoder", "vad", "endpointing",
         "postproc", "diar", "batching")]


class _Options(C.Structure):
    _fields_ = [("size", C.c_size_t), ("request_id", C.c_char_p),
                ("language_code", C.c_char_p), ("interim", C.c_bool),
                ("word_times", C.c_bool), ("punctuation", C.c_bool),
                ("verbatim", C.c_bool), ("profanity", C.c_bool),
                ("stop_ms", C.c_int32), ("contexts", C.c_void_p),
                ("context_count", C.c_size_t), ("alternatives", C.c_int32),
                ("diarization", C.c_bool), ("max_speakers", C.c_int32)]


class NemotronRuntime:
    def __init__(self) -> None:
        library, model = installation()
        self.lib = C.CDLL(str(library))
        self.handle = C.c_void_p()
        self.streams = set()
        signatures = {
            "create": ([C.POINTER(_Config), C.POINTER(C.c_void_p)], C.c_int),
            "destroy": ([C.c_void_p], None),
            "recognition_options_default": ([], _Options),
            "streaming_recognize": ([C.c_void_p, C.POINTER(_Options), C.POINTER(C.c_void_p)], C.c_int),
            "stream_push_f32": ([C.c_void_p, C.POINTER(C.c_float), C.c_size_t, C.c_int32], C.c_int),
            "stream_next": ([C.c_void_p, C.POINTER(C.c_void_p)], C.c_int),
            "stream_finish": ([C.c_void_p], C.c_int),
            "stream_close": ([C.c_void_p], None),
            "result_is_final": ([C.c_void_p], C.c_bool),
            "result_transcript": ([C.c_void_p, C.c_size_t], C.c_char_p),
            "result_destroy": ([C.c_void_p], None),
            "last_error": ([], C.c_char_p),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, "nemo_speech_asr_" + name)
            function.argtypes, function.restype = args, result
            setattr(self, "_" + name, function)
        backend = _Backend(C.sizeof(_Backend), 0)
        model_config = _Model(C.sizeof(_Model), str(model).encode(), None)
        # 560 ms right-context configuration; native model default is 1120 ms.
        streaming = _Streaming(C.sizeof(_Streaming), 0.56, 0, 0, 6)
        endpointing = _Endpointing(C.sizeof(_Endpointing), False, False, 0)
        config = _Config()
        config.size = C.sizeof(_Config)
        for name, value in (("backend", backend), ("model", model_config),
                            ("streaming", streaming), ("endpointing", endpointing)):
            setattr(config, name, C.addressof(value))
        try:
            self.check(self._create(C.byref(config), C.byref(self.handle)))
        except Exception:
            self.close()
            raise

    def check(self, code: int) -> None:
        if code:
            error = self._last_error()
            raise RuntimeError("Nemotron: " + (error.decode("utf-8", "replace") if error else str(code)))

    def stream(self):
        return NemotronStream(self)

    def transcribe(self, samples) -> str:
        stream = self.stream()
        try:
            for start in range(0, len(samples), 8960):
                stream.push(samples[start:start + 8960])
            return stream.finish()
        finally:
            stream.close()

    def close(self) -> None:
        for stream in tuple(self.streams):
            stream.close()
        handle, self.handle = self.handle, C.c_void_p()
        if handle:
            self._destroy(handle)


class NemotronStream:
    def __init__(self, runtime: NemotronRuntime) -> None:
        self.runtime = runtime
        self.handle = C.c_void_p()
        self.finals: list[str] = []
        self.partial = ""
        options = runtime._recognition_options_default()
        options.language_code = b"en-US"
        options.interim = True
        options.punctuation = True
        runtime.check(runtime._streaming_recognize(runtime.handle, C.byref(options), C.byref(self.handle)))
        runtime.streams.add(self)

    def _drain(self) -> None:
        while True:
            result = C.c_void_p()
            self.runtime.check(self.runtime._stream_next(self.handle, C.byref(result)))
            if not result:
                return
            try:
                value = self.runtime._result_transcript(result, 0)
                text = value.decode("utf-8", "replace").strip() if value else ""
                if self.runtime._result_is_final(result):
                    if text:
                        self.finals.append(text)
                    self.partial = ""
                else:
                    self.partial = text
            finally:
                self.runtime._result_destroy(result)

    def push(self, samples) -> None:
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        if not len(audio):
            return
        self.runtime.check(self.runtime._stream_push_f32(
            self.handle, audio.ctypes.data_as(C.POINTER(C.c_float)), len(audio), 16000))
        self._drain()

    def finish(self) -> str:
        self.runtime.check(self.runtime._stream_finish(self.handle))
        self._drain()
        # Only completed results may leave this module. Never paste an interim.
        return " ".join(self.finals).strip()

    def close(self) -> None:
        handle, self.handle = self.handle, C.c_void_p()
        if handle:
            self.runtime._stream_close(handle)
        self.runtime.streams.discard(self)

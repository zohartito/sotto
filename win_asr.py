"""Windows ASR backend: faster-whisper (CTranslate2) behind the mlx_whisper contract.

``LocalCT2Whisper.transcribe(samples, **kwargs) -> {"text": str}`` is what
``sotto.transcribe_canonical_samples`` and ``sotto.warm_speech_runtime``
expect of ``mlx_whisper``: ``samples`` is a 16 kHz float32 numpy array and
kwargs are ``path_or_hf_repo``, ``condition_on_previous_text``,
``word_timestamps``, optional ``language`` (None = per-capture auto-detect)
and optional ``initial_prompt``.

Models are pre-converted CT2 repos pinned to exact commits.  ``resolve_model_dir``
runs once at startup: a complete cached snapshot is used without any network
call, a missing one downloads only when no offline flag is set, and
``WhisperModel`` only ever receives that local directory.  Inference never
resolves a Hub id.

Device: CUDA float16 when it works, else CPU int8.  The first transcription
(the startup warmup) is the probe, because CTranslate2 loads cuBLAS lazily
and a missing DLL only surfaces at the first matmul.

Automatic language chooses only among the user's language set: detect once,
take the most probable allowed language, then transcribe with it fixed.  For a
clip that fits Whisper's 30 s window the detection's encoder pass is reused
for the decode (one encoder pass instead of two; same text, about half the
CPU time).
"""
from __future__ import annotations

import ctypes.util
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

# Model ids verified against the HuggingFace API on 2026-09-20: all ungated,
# library=ctranslate2, complete CT2 file sets.
PROFILES: dict[str, str] = {
    "auto": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "hebrew-turbo": "ivrit-ai/whisper-large-v3-turbo-ct2",
    "hebrew-quality": "ivrit-ai/whisper-large-v3-ct2",
}
# Settings speed "fast" swaps the auto profile's model for this smaller
# multilingual port (benchmark: docs/windows-alpha.md, 2026-09-29).
FAST_REPO = "Systran/faster-whisper-small"
# Exact commits (HuggingFace API, 2026-09-29).  Only these snapshots load.
REVISIONS: dict[str, str] = {
    "deepdml/faster-whisper-large-v3-turbo-ct2": "4df90f75321148c3a29a9e2351b7ddf8f5b115a8",
    "ivrit-ai/whisper-large-v3-turbo-ct2": "72ad623a37947395efcc3933132353790e5a12f5",
    "ivrit-ai/whisper-large-v3-ct2": "e9ed4a4a98d761b0f617d668303de2c514236c66",
    "Systran/faster-whisper-small": "536b0662742c02347bc0e980a01041f333bce120",
}
# Everything WhisperModel reads from a local directory.  tokenizer.json must be
# present: without it faster-whisper fetches a tokenizer from the Hub.
MODEL_FILES = ("config.json", "model.bin", "preprocessor_config.json",
               "tokenizer.json", "vocabulary.json")
# The Systran ports use Whisper's default 80 mel bins (no preprocessor config)
# and a text vocabulary; the large-v3 family needs its 128-bin config.
REPO_FILES: dict[str, tuple[str, ...]] = {
    "Systran/faster-whisper-small": ("config.json", "model.bin", "tokenizer.json",
                                     "vocabulary.txt"),
}

DEFAULT_PROFILE = "auto"
DEVICE_CHOICES = ("auto", "cuda", "cpu")
CUDA = ("cuda", "float16")
CPU = ("cpu", "int8")

_COMMIT = re.compile(r"[0-9a-f]{40}")


class SetupError(RuntimeError):
    """A startup problem with a plain, actionable message (no traceback)."""


class ModelUnavailable(SetupError):
    """The pinned model is not cached and downloading is not allowed."""


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def repo_for_profile(profile: str) -> str:
    """Return the CT2 repo for a known profile name or raise ``ValueError``."""
    try:
        return PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(PROFILES)
        raise ValueError(
            f"Unknown Windows speech profile {profile!r}; choose one of: {choices}"
        ) from exc


def pinned(model: str) -> tuple[str, str]:
    """``(repo, commit)`` for a profile repo or an explicit ``org/name@<commit>``."""
    repo, _, revision = model.partition("@")
    revision = revision or REVISIONS.get(repo, "")
    if not _COMMIT.fullmatch(revision) or repo.count("/") != 1:
        raise ValueError(
            f"model {model!r} is not pinned: use --profile, a local CTranslate2 "
            "directory, or org/name@<40-hex commit>")
    return repo, revision


def snapshot_dir(cache_dir: Path, repo: str, revision: str) -> Path:
    """Where huggingface_hub keeps this exact commit inside ``cache_dir``."""
    return (Path(cache_dir) / "hub" / f"models--{repo.replace('/', '--')}"
            / "snapshots" / revision)


def required_files(repo: str | None = None) -> tuple[str, ...]:
    """The exact file set a pinned repo (or a local directory) must provide."""
    return REPO_FILES.get(repo, MODEL_FILES)


def complete(directory: Path, files: tuple[str, ...] = MODEL_FILES) -> bool:
    return all((directory / name).is_file() for name in files)


def complete_local(directory: Path) -> bool:
    """A local directory may use either known CT2 Whisper layout."""
    return any(complete(directory, files) for files in (MODEL_FILES, *REPO_FILES.values()))


def resolve_model_dir(model: str, *, cache_dir: Path, offline: bool,
                      log: Callable[[str], None] = _log,
                      download: Callable[..., str] | None = None) -> Path:
    """Local CT2 directory for ``model``: cache first, download once, never by id later."""
    local = Path(model).expanduser()
    if local.is_dir():
        if not complete_local(local):
            missing = [name for name in MODEL_FILES if not (local / name).is_file()]
            raise SetupError(f"{local} is not a complete CTranslate2 Whisper model "
                             f"directory (missing {', '.join(missing)})")
        return local.absolute()
    repo, revision = pinned(model)
    files = required_files(repo)
    snapshot = snapshot_dir(cache_dir, repo, revision)
    if complete(snapshot, files):
        return snapshot
    if offline:
        raise ModelUnavailable(
            f"speech model {repo}@{revision[:12]} is not in {cache_dir} and offline "
            "mode is on (SOTTO_OFFLINE / HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE). "
            "Run once without those variables and with network access to download "
            "it, or point SOTTO_HF_HOME at the folder that already holds it.")
    if download is None:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from huggingface_hub import snapshot_download as download
    log(f"downloading {repo}@{revision[:12]} into {cache_dir} "
        "(first run only; the large-v3 models are 1.6–3.1 GB) ...")
    try:
        download(repo_id=repo, revision=revision, cache_dir=str(Path(cache_dir) / "hub"),
                 allow_patterns=list(files))
    except Exception as exc:
        raise SetupError(
            f"could not download {repo}@{revision[:12]}: {str(exc)[:200]} — "
            "check the network connection and run again") from exc
    if not complete(snapshot, files):
        raise SetupError(f"download of {repo}@{revision[:12]} is incomplete in {snapshot}")
    return snapshot


def choose_language(probabilities: Mapping[str, float], allowed: Iterable[str]) -> str:
    """Automatic's pick, the Mac's rule (``sotto.choose_language``).

    The most probable allowed language (ties keep the user's order) — unless
    the speech is clearly another language (every allowed one scores < 0.25
    while it scores >= 0.5): forcing it into an allowed language would make
    Whisper translate, so it is written in the spoken language instead.
    """
    allowed = tuple(allowed)
    if not allowed:
        raise ValueError("no allowed languages")
    best = max(allowed, key=lambda code: float(probabilities.get(code, 0.0)))
    if probabilities:
        top = max(probabilities, key=probabilities.get)
        if (top not in allowed and float(probabilities.get(best, 0.0)) < 0.25
                and float(probabilities[top]) >= 0.5):
            return top
    return best


class LanguageRestricted:
    """One job's view of the model: Automatic picks among ``allowed`` only.

    ``last_language`` is the code Automatic chose (what the shared
    ``sotto.transcribe_canonical_samples`` records as ``detected_language``).
    """

    def __init__(self, model: Any, allowed: Iterable[str]) -> None:
        self._model = model
        self.allowed = tuple(allowed)
        self.last_language: str | None = None

    def transcribe(self, samples: Any, *args: Any, **kwargs: Any) -> dict:
        self.last_language = None
        result = self._model.transcribe(samples, *args, allowed_languages=self.allowed, **kwargs)
        self.last_language = result.get("language")
        return result


_MISSING = object()


def _first_window(model: Any, samples: Any):
    """(features, encoder output) of the exact first window faster-whisper
    decodes, when the clip fits in that one 30 s window; else None."""
    extractor = getattr(model, "feature_extractor", None)
    if extractor is None or not getattr(getattr(model, "model", None), "is_multilingual", False):
        return None
    try:
        from faster_whisper.audio import pad_or_trim
    except ImportError:
        return None
    features = extractor(samples)
    content = features.shape[-1] - 1  # generate_segments' own window arithmetic
    if not 0 < content <= extractor.nb_max_frames:
        return None
    window = pad_or_trim(features[:, :content])
    return window, model.encode(window)


def _transcribe_reusing(model: Any, samples: Any, options: dict, window: Any, output: Any) -> str:
    """Decode with ``output`` standing in for the encoder on that exact window.

    The stored output is used only for bit-identical input, once; anything
    else (a later window, a changed feature pipeline) encodes normally.
    """
    import numpy as np
    encode = model.encode
    previous = model.__dict__.get("encode", _MISSING)
    unused = [True]

    def reuse(features):
        if unused[0] and features.shape == window.shape and np.array_equal(features, window):
            unused[0] = False
            return output
        return encode(features)

    model.encode = reuse
    try:
        segments, _info = model.transcribe(samples, **options)
        return "".join(segment.text for segment in segments)  # decoding runs here
    finally:
        if previous is _MISSING:
            del model.encode
        else:
            model.encode = previous


def requested_device(cli: str | None = None, environ=None) -> str:
    """``--device`` wins, then ``SOTTO_DEVICE``; blank means ``auto``."""
    source = os.environ if environ is None else environ
    value = (cli or source.get("SOTTO_DEVICE", "") or "auto").strip().lower() or "auto"
    if value not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {', '.join(DEVICE_CHOICES)}; got {value!r}")
    return value


def device_plan(requested: str, cuda_devices: int) -> list[tuple[str, str]]:
    """Ordered (device, compute_type) attempts for a requested device."""
    if requested == "cpu":
        return [CPU]
    if requested == "cuda":
        return [CUDA]
    return [CUDA, CPU] if cuda_devices > 0 else [CPU]


def cuda_device_count() -> int:
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def _ensure_cublas() -> None:
    """Make cublas64_12.dll resolvable before CTranslate2 loads it.

    The ctranslate2 wheel links CUDA 12 and bundles cuDNN but not cuBLAS.  The
    optional pip nvidia-cublas-cu12 package (requirements-alpha-windows-cuda.txt)
    carries the 12-series DLLs; add its bin dir to the process DLL search path
    so CUDA inference works from any launch context.
    """
    if ctypes.util.find_library("cublas64_12"):
        return  # already on PATH or in a system directory
    try:
        import nvidia.cublas as _cublas
    except ImportError:
        return  # the CUDA attempt fails at first use and falls back to CPU
    # namespace package: __file__ is None, __path__ carries the dir
    dll_dir = Path(next(iter(_cublas.__path__))) / "bin"
    if dll_dir.is_dir():
        os.add_dll_directory(str(dll_dir))
        os.environ["PATH"] = f"{dll_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def _whisper_model(path: str, *, device: str, compute_type: str) -> Any:
    from faster_whisper import WhisperModel
    return WhisperModel(path, device=device, compute_type=compute_type,
                        local_files_only=True)


class LocalCT2Whisper:
    """One resident model on the first device that actually transcribes."""

    def __init__(self, model_dir: Path | str, *, device: str = "auto",
                 model_factory: Callable[..., Any] = _whisper_model,
                 cuda_devices: Callable[[], int] = cuda_device_count,
                 log: Callable[[str], None] = _log) -> None:
        self.path = str(Path(model_dir))
        self.requested = requested_device(device, environ={})
        self.device: str | None = None
        self.compute_type: str | None = None
        self._factory = model_factory
        self._cuda_devices = cuda_devices
        self._log = log
        self._model: Any = None
        self._lock = threading.Lock()

    def transcribe(self, samples: Any, path_or_hf_repo: str | None = None, *,
                   condition_on_previous_text: bool = False,
                   word_timestamps: bool = False,
                   language: str | None = None,
                   initial_prompt: str | None = None,
                   allowed_languages: Iterable[str] = (),
                   **_ignored: Any) -> dict:
        """mlx_whisper-compatible; ``path_or_hf_repo`` is a label, never resolved.

        With ``language=None`` and a non-empty ``allowed_languages``, the
        language is detected once and the most probable allowed one is fixed
        for the decode (``language`` in the result says which).
        """
        options = {"language": language, "initial_prompt": initial_prompt,
                   "condition_on_previous_text": condition_on_previous_text,
                   "word_timestamps": word_timestamps}
        allowed = tuple(allowed_languages)
        with self._lock:
            if self._model is None:
                return self._load(samples, options, allowed)
            return self._run(self._model, samples, options, allowed)

    @staticmethod
    def _run(model: Any, samples: Any, options: dict, allowed: tuple[str, ...] = ()) -> dict:
        if options["language"] is not None or not allowed:
            segments, _info = model.transcribe(samples, **options)
            return {"text": "".join(segment.text for segment in segments)}
        if len(allowed) == 1:
            first = None
            language = allowed[0]
        else:
            first = _first_window(model, samples)
            if first is not None:  # one encoder pass for detection and decode
                ranked = model.model.detect_language(first[1])[0]
                probabilities = {token[2:-2]: probability for token, probability in ranked}
            else:
                _top, _probability, ranked = model.detect_language(audio=samples)
                probabilities = dict(ranked)
            language = choose_language(probabilities, allowed)
        options = dict(options, language=language)
        if first is not None:
            text = _transcribe_reusing(model, samples, options, *first)
        else:
            segments, _info = model.transcribe(samples, **options)
            text = "".join(segment.text for segment in segments)
        return {"text": text, "language": language}

    def _load(self, samples: Any, options: dict, allowed: tuple[str, ...] = ()) -> dict:
        plan = device_plan(self.requested, self._cuda_devices())
        failures = []
        for device, compute_type in plan:
            try:
                if device == "cuda":
                    _ensure_cublas()
                model = self._factory(self.path, device=device, compute_type=compute_type)
                result = self._run(model, samples, options, allowed)
            except Exception as exc:
                failures.append(f"{device} {compute_type}: {str(exc)[:160]}")
                if (device, compute_type) != plan[-1]:
                    self._log(f"! {device} ({compute_type}) unavailable — "
                              f"{str(exc)[:160]}; falling back to CPU (int8)")
                continue
            self._model, self.device, self.compute_type = model, device, compute_type
            self._log(f"✓ ASR device: {device} ({compute_type})")
            return result
        hint = (" — run with --device cpu (or SOTTO_DEVICE=cpu), or install the "
                "CUDA extras" if self.requested == "cuda" else "")
        raise SetupError("no usable ASR device: " + "; ".join(failures) + hint)

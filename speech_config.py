"""Pure, local configuration helpers for Sotto speech models.

This module deliberately has no MLX or platform imports.  Resolving a profile is
therefore safe to use in a preferences UI, a CLI ``--dry-run``, or tests without
causing a model download.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Iterable
import uuid


# Every language Whisper can transcribe (code -> English name).
WHISPER_LANGUAGES: dict[str, str] = {
    "af": "Afrikaans", "sq": "Albanian", "am": "Amharic", "ar": "Arabic", "hy": "Armenian",
    "as": "Assamese", "az": "Azerbaijani", "ba": "Bashkir", "eu": "Basque", "be": "Belarusian",
    "bn": "Bengali", "bs": "Bosnian", "br": "Breton", "bg": "Bulgarian", "yue": "Cantonese",
    "ca": "Catalan", "zh": "Chinese", "hr": "Croatian", "cs": "Czech", "da": "Danish",
    "nl": "Dutch", "en": "English", "et": "Estonian", "fo": "Faroese", "fi": "Finnish",
    "fr": "French", "gl": "Galician", "ka": "Georgian", "de": "German", "el": "Greek",
    "gu": "Gujarati", "ht": "Haitian Creole", "ha": "Hausa", "haw": "Hawaiian", "he": "Hebrew",
    "hi": "Hindi", "hu": "Hungarian", "is": "Icelandic", "id": "Indonesian", "it": "Italian",
    "ja": "Japanese", "jw": "Javanese", "kn": "Kannada", "kk": "Kazakh", "km": "Khmer",
    "ko": "Korean", "lo": "Lao", "la": "Latin", "lv": "Latvian", "ln": "Lingala",
    "lt": "Lithuanian", "lb": "Luxembourgish", "mk": "Macedonian", "mg": "Malagasy",
    "ms": "Malay", "ml": "Malayalam", "mt": "Maltese", "mi": "Maori", "mr": "Marathi",
    "mn": "Mongolian", "my": "Myanmar", "ne": "Nepali", "no": "Norwegian", "nn": "Nynorsk",
    "oc": "Occitan", "ps": "Pashto", "fa": "Persian", "pl": "Polish", "pt": "Portuguese",
    "pa": "Punjabi", "ro": "Romanian", "ru": "Russian", "sa": "Sanskrit", "sr": "Serbian",
    "sn": "Shona", "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian",
    "so": "Somali", "es": "Spanish", "su": "Sundanese", "sw": "Swahili", "sv": "Swedish",
    "tl": "Tagalog", "tg": "Tajik", "ta": "Tamil", "tt": "Tatar", "te": "Telugu", "th": "Thai",
    "bo": "Tibetan", "tr": "Turkish", "tk": "Turkmen", "uk": "Ukrainian", "ur": "Urdu",
    "uz": "Uzbek", "vi": "Vietnamese", "cy": "Welsh", "yi": "Yiddish", "yo": "Yoruba",
}
SUPPORTED_LANGUAGES = ("auto", *WHISPER_LANGUAGES)
# Automatic chooses among these unless Settings names others.
DEFAULT_AUTOMATIC_LANGUAGES = ("en", "he")
LANGUAGE_LABELS = {"auto": "Automatic (English + Hebrew)", **WHISPER_LANGUAGES}


def automatic_languages(chosen=None) -> tuple[str, ...]:
    """The languages Automatic may pick; unknown codes are dropped, none = default."""
    codes = tuple(dict.fromkeys(code for code in (chosen or ()) if code in WHISPER_LANGUAGES))
    return codes or DEFAULT_AUTOMATIC_LANGUAGES


def automatic_label(chosen=None) -> str:
    names = [WHISPER_LANGUAGES[code] for code in automatic_languages(chosen)]
    shown = " + ".join(names[:3]) + (f" +{len(names) - 3}" if len(names) > 3 else "")
    return f"Automatic ({shown})"
from sotto_paths import DATA_DIR

DEFAULT_LANGUAGE_MODE_PATH = DATA_DIR / "language-mode"
DEFAULT_ENGINE_MODE_PATH = DEFAULT_LANGUAGE_MODE_PATH.with_name("engine-mode")
ENGINE_CHOICES = {"whisper": "Whisper",
                  "parakeet": "Parakeet (fastest · 25 European languages)",
                  "nemotron": "Nemotron (English streaming)"}
# Parakeet TDT 0.6B v3 detects these itself; no Hebrew, Arabic or Asian languages.
PARAKEET_LANGUAGES = ("bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it",
                      "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk")
PARAKEET_DOWNLOAD = "2.5 GB"
MAX_GLOSSARY_TERMS = 100
MAX_GLOSSARY_CHARACTERS = 2_000
MAX_GLOSSARY_TERM_CHARACTERS = 120
GLOSSARY_MAX_CAPTURE_SECONDS = 30.0


@dataclass(frozen=True)
class ModelProfile:
    """A named local speech model and its language default."""

    name: str
    repo: str
    language: str | None
    backend: str = "whisper"


@dataclass(frozen=True)
class SpeechConfig:
    """The fully resolved configuration passed to a transcription call."""

    profile: ModelProfile
    model_repo: str
    language: str | None


@dataclass(frozen=True)
class GlossaryPrompt:
    """A prompt decision, including why no prompt was supplied when disabled."""

    prompt: str | None
    reason: str | None


MODEL_PROFILES: dict[str, ModelProfile] = {
    "auto": ModelProfile("auto", "mlx-community/whisper-large-v3-turbo", None),
    "hebrew-turbo": ModelProfile(
        "hebrew-turbo", "mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx", "he"
    ),
    "hebrew-quality": ModelProfile(
        "hebrew-quality", "mlx-community/ivrit-ai-whisper-large-v3-mlx", "he"
    ),
    "nemotron-en": ModelProfile(
        "nemotron-en", "nvidia/nemotron-speech-streaming-en-0.6b", "en", "nemotron"
    ),
    "parakeet": ModelProfile("parakeet", "mlx-community/parakeet-tdt-0.6b-v3", None, "parakeet"),
}

# Exact Hugging Face commits for the built-in Whisper profiles (verified
# 2026-09-29). A cached snapshot of the pinned commit is used without any Hub
# request; a first download fetches exactly these bytes, never a moved `main`.
MODEL_REVISIONS: dict[str, str] = {
    "mlx-community/whisper-large-v3-turbo": "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
    "mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx": "53ad8c6cd8b32eb0303f093a404ae13c1b1d567f",
    "mlx-community/ivrit-ai-whisper-large-v3-mlx": "097c0cb2bb4288a3c82f6d524a39c4ce09afa187",
    "mlx-community/parakeet-tdt-0.6b-v3": "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
}


def pinned_snapshot_cached(repo: str, hub_dir, required=("config.json",)) -> bool:
    """Is the pinned snapshot of ``repo`` fully present in a Hugging Face hub
    cache? A pure path check: never resolves or downloads anything."""
    revision = MODEL_REVISIONS.get(repo)
    if revision is None:
        return False
    folder = Path(hub_dir) / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    return all((folder / name).is_file() for name in required)


def get_profile(name: str = "auto") -> ModelProfile:
    """Return a known model profile or raise ``ValueError`` with valid choices."""

    try:
        return MODEL_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(MODEL_PROFILES)
        raise ValueError(f"Unknown speech profile {name!r}; choose one of: {choices}") from exc


def normalize_language(language: str | None) -> str | None:
    """Validate a language CLI value; ``auto`` (and ``None``) means auto-detect."""

    if language is None or language == "auto":
        return None
    if language not in SUPPORTED_LANGUAGES:
        choices = ", ".join(SUPPORTED_LANGUAGES)
        raise ValueError(f"Unknown language {language!r}; choose one of: {choices}")
    return language


def language_mode(language: str | None) -> str:
    """Return the user-facing mode for a resolved Whisper language value."""

    normalized = normalize_language(language)
    return "auto" if normalized is None else normalized


def with_language(config: SpeechConfig, mode: str) -> SpeechConfig:
    """Return ``config`` with only its per-capture language hint changed."""

    language = normalize_language(mode)
    if config.profile.backend == "parakeet":
        return replace(config, language=None)  # Parakeet detects its own language
    if config.profile.backend == "nemotron" and language != "en":
        raise ValueError("Nemotron is English-only; select Whisper for Hebrew or automatic language.")
    return replace(config, language=language)


def load_engine_mode(path: str | Path = DEFAULT_ENGINE_MODE_PATH) -> str:
    try:
        mode = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return "whisper"
    return mode if mode in ENGINE_CHOICES else "whisper"


def save_engine_mode(mode: str, path: str | Path = DEFAULT_ENGINE_MODE_PATH) -> str:
    if mode not in ENGINE_CHOICES:
        raise ValueError("Unknown speech engine")
    # The existing private atomic preference writer validates language modes.
    # Keep the same storage contract while accepting this preference's values.
    _save_private_choice(mode, Path(path))
    return mode


def load_language_mode(
    path: str | Path = DEFAULT_LANGUAGE_MODE_PATH, *, default: str = "auto"
) -> str:
    """Read the last menu choice, failing closed to a validated default."""

    fallback = language_mode(default)
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return fallback
    try:
        return language_mode(value)
    except ValueError:
        return fallback


def save_language_mode(mode: str, path: str | Path = DEFAULT_LANGUAGE_MODE_PATH) -> str:
    """Atomically persist a private language choice and return its canonical mode."""

    selected = language_mode(mode)
    _save_private_choice(selected, Path(path))
    return selected


def _save_private_choice(selected: str, path: Path) -> None:
    target = Path(path)
    from storage_lock import ensure_private_directory, ensure_private_file

    ensure_private_directory(target.parent)
    ensure_private_file(target)
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(selected + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
        os.chmod(target, 0o600)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def resolve_speech_config(
    profile: str = "auto", model: str | None = None, language: str | None = None
) -> SpeechConfig:
    """Resolve explicit model/language overrides over a named profile.

    ``language=None`` means no CLI override was supplied.  Passing ``auto`` is
    an explicit override which enables Whisper's language auto-detection.
    """

    selected_profile = get_profile(profile)
    if model is not None and not model.strip():
        raise ValueError("Model repository must not be empty")
    model_repo = model.strip() if model is not None else selected_profile.repo
    resolved_language = (
        selected_profile.language if language is None else normalize_language(language)
    )
    if selected_profile.backend == "nemotron":
        if model is not None:
            raise ValueError("Nemotron uses its pinned local model; --model is a Whisper override.")
        if resolved_language != "en":
            raise ValueError("Nemotron is English-only; use --language en or select Whisper.")
    if selected_profile.backend == "parakeet":
        if model is not None:
            raise ValueError("Parakeet uses its pinned model; --model is a Whisper override.")
        resolved_language = None  # Parakeet detects its own language
    return SpeechConfig(selected_profile, model_repo, resolved_language)


def _clean_terms(raw_terms: Iterable[object]) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    characters = 0
    for raw in raw_terms:
        if not isinstance(raw, str):
            raise ValueError("Glossary terms must be strings")
        term = " ".join(raw.strip().split())
        if not term or term.startswith("#") or term in seen:
            continue
        # Long terms can become an unexpectedly large initial prompt.  Ignore
        # them instead of silently splitting a name or code word in half.
        if len(term) > MAX_GLOSSARY_TERM_CHARACTERS:
            continue
        prompt_cost = len(term) + (2 if terms else 0)
        if len(terms) >= MAX_GLOSSARY_TERMS or characters + prompt_cost > MAX_GLOSSARY_CHARACTERS:
            break
        terms.append(term)
        seen.add(term)
        characters += prompt_cost
    return tuple(terms)


def load_glossary(path: str | Path) -> tuple[str, ...]:
    """Load a local UTF-8 text file or a JSON string list, safely bounded.

    Text files support one term per line; blank lines and lines beginning with
    ``#`` are ignored.  Duplicate terms preserve the first spelling and order.
    """

    glossary_path = Path(path)
    try:
        contents = glossary_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Unable to read glossary {glossary_path}: {exc}") from exc
    if glossary_path.suffix.lower() == ".json":
        try:
            parsed = json.loads(contents)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Glossary JSON is invalid: {exc.msg}") from exc
        if not isinstance(parsed, list):
            raise ValueError("Glossary JSON must contain a list of strings")
        return _clean_terms(parsed)
    return _clean_terms(contents.splitlines())


def glossary_prompt(terms: Iterable[str], duration_seconds: float | None) -> GlossaryPrompt:
    """Return a short-capture initial prompt, otherwise a concrete disable reason."""

    bounded_terms = _clean_terms(terms)
    if not bounded_terms:
        return GlossaryPrompt(None, "no glossary terms")
    if duration_seconds is None:
        return GlossaryPrompt(None, "capture duration is unknown")
    if duration_seconds > GLOSSARY_MAX_CAPTURE_SECONDS:
        return GlossaryPrompt(
            None,
            f"capture exceeds {GLOSSARY_MAX_CAPTURE_SECONDS:g} seconds; glossary prompting is disabled",
        )
    if duration_seconds < 0:
        return GlossaryPrompt(None, "capture duration must not be negative")
    # Whisper responds more reliably to sentence-like context than a comma-only
    # word bag.  This remains deterministic and bounded; callers still decide
    # explicitly which local terms and phrases enter the prompt.
    phrases = ". ".join(term.rstrip(".?!") for term in bounded_terms)
    return GlossaryPrompt("Speech context: " + phrases + ".", None)

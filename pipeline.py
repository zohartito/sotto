"""What happens to a finished transcript before anything is delivered.

One path for live dictation and Retry, on macOS and Windows:

    raw transcript -> no-speech verdict -> hallucination guard (with
    repetition-loop salvage) -> personal dictionary -> English voice cleanup

`screen()` composes the steps; the pieces stay importable on their own. Pure
apart from reading the user's dictionary and cleanup settings and an optional
log callback. Platform modules import from here; nothing here imports them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import zlib

NO_SPEECH_TEXT = "[no speech detected]"

LOOP_MIN_REPEATS = 6        # far past what real speech repeats verbatim
LOOP_MAX_UNIT_WORDS = 6     # loops cycle a word or a short phrase, not a clause
LOOP_MIN_PREFIX_CHARS = 40  # below this there is no dictation worth rescuing


def _quiet(message: str) -> None:
    """Default log callback: say nothing."""


def looks_hallucinated(text: str, seconds: float) -> str | None:
    """Whisper can turn a second of breath into a thousand characters of
    looped phrases. Speech is ~15 chars/sec; loops compress absurdly well.
    Returns the reason if the transcript can't be real speech, else None."""
    chars_per_sec = len(text) / max(seconds, 0.1)
    if chars_per_sec > 40:
        return f"{chars_per_sec:.0f} chars/sec"
    if len(text) > 120:
        ratio = len(zlib.compress(text.encode())) / len(text.encode())
        if ratio < 0.25:
            return f"compression ratio {ratio:.2f}"
    return None


def _find_repetition_loop(text: str) -> tuple[int, str, int] | None:
    """Earliest degenerate repeated run in `text`, as (char offset where the
    run starts, the repeated unit, repeat count). Words are matched
    case-insensitively and without trailing punctuation, so whisper's
    "Okay. Okay. Okay." loops count as repeats of one unit."""
    spans = [match.span() for match in re.finditer(r"\S+", text)]
    words = [text[start:end].strip(".,!?;:-").casefold() for start, end in spans]
    best: tuple[int, str, int] | None = None
    for unit in range(1, LOOP_MAX_UNIT_WORDS + 1):
        index = 0
        while index + unit * LOOP_MIN_REPEATS <= len(words):
            first = words[index:index + unit]
            repeats, cursor = 1, index + unit
            while cursor + unit <= len(words) and words[cursor:cursor + unit] == first:
                repeats += 1
                cursor += unit
            if repeats >= LOOP_MIN_REPEATS:
                if best is None or spans[index][0] < best[0]:
                    unit_text = text[spans[index][0]:spans[index + unit - 1][1]]
                    best = (spans[index][0], unit_text, repeats)
                index = cursor
            else:
                index += 1
    return best


def salvage_repetition_loop(text: str, seconds: float) -> tuple[str, dict] | None:
    """Whisper can transcribe real dictation and only then fall into a
    repetition loop ("...that's part of the plan." + "difference" x223,
    observed 2026-08-15). Blocking the whole transcript costs the user
    everything they said, so cut at the loop and keep the clean prefix.

    Returns (prefix, receipt) only when the prefix stands on its own as real
    speech by the same measures that condemned the whole; otherwise None and
    the caller quarantines as before."""
    found = _find_repetition_loop(text)
    if found is None:
        return None
    offset, unit_text, repeats = found
    prefix = text[:offset].strip()
    if len(prefix) < LOOP_MIN_PREFIX_CHARS:
        return None
    if looks_hallucinated(prefix, seconds) is not None:
        return None
    return prefix, {"unit": unit_text[:40], "repeats": repeats,
                    "dropped_chars": len(text) - len(prefix)}


def reads_as_no_speech(text: str, speech_fraction: float) -> str | None:
    """Output-side no-speech verdict, judged AFTER transcription. Whisper
    turns silence and dead-mic captures into short stock phrases ("you",
    "Thank you."), so a tiny transcript from a capture the VAD scored as
    speechless is silence, not dictation. A substantial transcript wins over
    the VAD score — Silero scores real whispered dictation at 0% speech
    (whispers on this machine: 240+ chars at VAD 0%; silence: <=15 chars)."""
    if not text.strip():
        return "empty transcript"
    if speech_fraction < 0.06 and len(text.strip()) <= 20:
        return f"tiny transcript, VAD {speech_fraction*100:.0f}% speech"
    return None


def apply_personal_dictionary(text: str, preprocessing: dict, log=_quiet) -> str:
    """The user's own spellings for text that is about to be delivered.

    Runs after every hallucination/no-speech guard; the recognizer's original
    output stays in the History row's preprocessing receipt.
    Side effects: adds a "dictionary" receipt to `preprocessing`; logs.
    """
    import dictionary
    try:
        updated, receipt = dictionary.apply(text, dictionary.load(dictionary.DICTIONARY_PATH))
    except Exception as exc:  # a broken dictionary must never cost a dictation
        log(f"! dictionary skipped ({type(exc).__name__})")
        return text
    if receipt:
        preprocessing["dictionary"] = {"rules": receipt, "asr_text": text}
        log(f"  dictionary: {sum(item['count'] for item in receipt)} replacement(s)")
    return updated


def apply_voice_cleanup(text: str, preprocessing: dict, language: str | None) -> tuple[str, str | None]:
    """English voice commands and filler removal, after the dictionary. Returns
    the text to deliver and an action ("scratch" undoes the last dictation).
    Side effects: adds a "voice" receipt to `preprocessing`."""
    import settings
    import voice_commands
    preferences = settings.load(settings.SETTINGS_PATH)
    commands, fillers = preferences["voice_commands"], preferences["remove_fillers"]
    if not (commands or fillers) or not voice_commands.is_english(language):
        return text, None
    cleaned, action = voice_commands.clean(text, fillers=fillers, commands=commands)
    if action or cleaned != text:
        preprocessing["voice"] = {"asr_text": text, **({"action": action} if action else {})}
    return cleaned, action


@dataclass(frozen=True)
class Screened:
    """A transcript after the guards. `held` is why it is kept back from
    delivery (it still goes to History as suspect); `voice_action` "scratch"
    means undo the last dictation instead of inserting anything."""
    text: str
    held: str | None = None
    voice_action: str | None = None
    receipts: dict = field(default_factory=dict)


def screen(text: str, *, seconds: float, speech_fraction: float, language: str | None,
           clean: bool = True, log=_quiet) -> Screened:
    """Run every output guard, then (for text that passes) the user's
    dictionary and English cleanup. `clean=False` skips the last two (the
    adaptive lane keeps the recognizer's exact text)."""
    receipts: dict = {}
    no_speech = reads_as_no_speech(text, speech_fraction)
    if no_speech == "empty transcript":
        text = NO_SPEECH_TEXT
    hallucinated = looks_hallucinated(text, seconds)
    if hallucinated and not no_speech:
        # A loop that starts partway through must not cost the user the real
        # dictation in front of it.
        salvaged = salvage_repetition_loop(text, seconds)
        if salvaged is not None:
            text, trim_receipt = salvaged
            receipts["repetition_trimmed"] = trim_receipt
            log(f"  cut a {trim_receipt['repeats']}x repetition loop "
                f"({trim_receipt['dropped_chars']} chars) — keeping the clean prefix")
            hallucinated = None
    held = hallucinated or no_speech
    voice_action = None
    if not held and clean:
        text = apply_personal_dictionary(text, receipts, log)
        text, voice_action = apply_voice_cleanup(text, receipts, language)
    return Screened(text, held, voice_action, receipts)

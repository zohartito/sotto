"""Dependency-light transcription quality metrics for offline evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import random
import unicodedata
from typing import Iterable, Sequence


LOW_COUNT_THRESHOLD = 10
PAIRED_EVALUATOR_ID = "sotto-paired-v1"
PAIRED_BOOTSTRAP_REPLICATES = 10_000


def tokenize(text: str) -> tuple[str, ...]:
    """Normalize Hebrew/English text and retain word and numeric tokens.

    NFKC handles compatibility variants, combining marks (such as niqqud) are
    omitted, and punctuation is a separator.  Hebrew letters and all Unicode
    decimal digits are retained; no ASCII-only filtering is used.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    output: list[str] = []
    current: list[str] = []
    for char in normalized:
        if unicodedata.category(char).startswith("M"):
            continue
        if char.isalnum():
            current.append(char)
        elif char in "'_’" and current:
            # Keep apostrophes only when they bridge two alphanumeric pieces.
            current.append("'")
        else:
            if current:
                output.append("".join(current).strip("'"))
                current = []
    if current:
        output.append("".join(current).strip("'"))
    return tuple(token for token in output if token)


def normalize_text(text: str) -> str:
    """Return a deterministic display form used by the metrics."""

    return " ".join(tokenize(text))


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance with linear auxiliary space."""

    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, reference_item in enumerate(reference, start=1):
        current = [i]
        for j, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (reference_item != hypothesis_item),
            ))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    """Return WER, or ``None`` where the reference has zero words."""

    ref_tokens, hyp_tokens = tokenize(reference), tokenize(hypothesis)
    if not ref_tokens:
        return None
    return _edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def character_error_rate(reference: str, hypothesis: str) -> float | None:
    """Return CER over normalized non-space characters, or ``None`` for empty ref."""

    ref_chars = tuple(normalize_text(reference).replace(" ", ""))
    hyp_chars = tuple(normalize_text(hypothesis).replace(" ", ""))
    if not ref_chars:
        return None
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def _contains_tokens(hypothesis: Sequence[str], required: Sequence[str]) -> bool:
    if not required:
        return False
    width = len(required)
    return any(tuple(hypothesis[index:index + width]) == tuple(required)
               for index in range(len(hypothesis) - width + 1))


def required_term_accuracy(hypothesis: str, required_terms: Iterable[str]) -> tuple[float | None, int, int]:
    """Return accuracy, matched-term count, and usable required-term count."""

    hyp_tokens = tokenize(hypothesis)
    terms = [tokenize(term) for term in required_terms if isinstance(term, str)]
    usable = [term for term in terms if term]
    if not usable:
        return None, 0, 0
    matched = sum(_contains_tokens(hyp_tokens, term) for term in usable)
    return matched / len(usable), matched, len(usable)


def numeric_token_accuracy(reference: str, hypothesis: str) -> tuple[float | None, int, int]:
    """Score every numeric token from reference against hypothesis token counts."""

    reference_numbers = [token for token in tokenize(reference) if any(ch.isdecimal() for ch in token)]
    if not reference_numbers:
        return None, 0, 0
    remaining = list(token for token in tokenize(hypothesis) if any(ch.isdecimal() for ch in token))
    matched = 0
    for token in reference_numbers:
        try:
            remaining.remove(token)
        except ValueError:
            continue
        matched += 1
    return matched / len(reference_numbers), matched, len(reference_numbers)


@dataclass(frozen=True)
class SampleResult:
    sample_id: str
    language: str | None
    tags: tuple[str, ...]
    reference: str
    hypothesis: str
    normalized_reference: str
    normalized_hypothesis: str
    word_distance: int
    reference_words: int
    character_distance: int
    reference_characters: int
    wer: float | None
    cer: float | None
    required_term_accuracy: float | None
    matched_required_terms: int
    required_term_count: int
    numeric_token_accuracy: float | None
    matched_numeric_tokens: int
    numeric_token_count: int
    hallucination_proxy: bool

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_sample(
    reference: str,
    hypothesis: str,
    *,
    sample_id: str = "",
    language: str | None = None,
    tags: Iterable[str] = (),
    required_terms: Iterable[str] = (),
) -> SampleResult:
    """Compute all metrics for a transcription/reference pair."""

    ref_tokens, hyp_tokens = tokenize(reference), tokenize(hypothesis)
    ref_text, hyp_text = " ".join(ref_tokens), " ".join(hyp_tokens)
    ref_chars, hyp_chars = tuple(ref_text.replace(" ", "")), tuple(hyp_text.replace(" ", ""))
    word_distance = _edit_distance(ref_tokens, hyp_tokens)
    char_distance = _edit_distance(ref_chars, hyp_chars)
    term_accuracy, matched_terms, term_count = required_term_accuracy(hypothesis, required_terms)
    numeric_accuracy, matched_numbers, number_count = numeric_token_accuracy(reference, hypothesis)
    return SampleResult(
        sample_id=sample_id,
        language=language,
        tags=tuple(sorted({tag for tag in tags if isinstance(tag, str)})),
        reference=reference,
        hypothesis=hypothesis,
        normalized_reference=ref_text,
        normalized_hypothesis=hyp_text,
        word_distance=word_distance,
        reference_words=len(ref_tokens),
        character_distance=char_distance,
        reference_characters=len(ref_chars),
        wer=(word_distance / len(ref_tokens)) if ref_tokens else None,
        cer=(char_distance / len(ref_chars)) if ref_chars else None,
        required_term_accuracy=term_accuracy,
        matched_required_terms=matched_terms,
        required_term_count=term_count,
        numeric_token_accuracy=numeric_accuracy,
        matched_numeric_tokens=matched_numbers,
        numeric_token_count=number_count,
        hallucination_proxy=not ref_tokens and bool(hyp_tokens),
    )


def _sample_stderr(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _metric(value: float | None, denominator: int, sample_values: list[float]) -> dict:
    return {
        "value": value,
        "denominator": denominator,
        "scored_samples": len(sample_values),
        "sample_stderr": _sample_stderr(sample_values),
    }


def summarize_results(results: Iterable[SampleResult]) -> dict:
    """Aggregate micro metrics and expose sample count/uncertainty explicitly."""

    rows = list(results)
    word_units = sum(row.reference_words for row in rows)
    character_units = sum(row.reference_characters for row in rows)
    term_units = sum(row.required_term_count for row in rows)
    number_units = sum(row.numeric_token_count for row in rows)
    count = len(rows)
    low_count = count < LOW_COUNT_THRESHOLD
    return {
        "count": count,
        "low_count": low_count,
        "directional": low_count,
        "uncertainty_note": (
            f"Directional only: fewer than {LOW_COUNT_THRESHOLD} samples."
            if low_count else
            "Sample size alone does not establish statistical sufficiency."
        ),
        "zero_reference_samples": sum(row.reference_words == 0 for row in rows),
        "metrics": {
            "wer": _metric(_ratio(sum(row.word_distance for row in rows), word_units), word_units,
                           [row.wer for row in rows if row.wer is not None]),
            "cer": _metric(_ratio(sum(row.character_distance for row in rows), character_units), character_units,
                           [row.cer for row in rows if row.cer is not None]),
            "required_term_accuracy": _metric(
                _ratio(sum(row.matched_required_terms for row in rows), term_units), term_units,
                [row.required_term_accuracy for row in rows if row.required_term_accuracy is not None],
            ),
            "numeric_token_accuracy": _metric(
                _ratio(sum(row.matched_numeric_tokens for row in rows), number_units), number_units,
                [row.numeric_token_accuracy for row in rows if row.numeric_token_accuracy is not None],
            ),
            "hallucination_proxy_rate": _metric(
                _ratio(sum(row.hallucination_proxy for row in rows), count), count,
                [float(row.hallucination_proxy) for row in rows],
            ),
        },
    }


def aggregate_results(results: Iterable[SampleResult]) -> dict:
    """Return overall, language, and tag slices with stable key ordering."""

    rows = list(results)
    language_groups: dict[str, list[SampleResult]] = {}
    tag_groups: dict[str, list[SampleResult]] = {}
    for row in rows:
        language_groups.setdefault(row.language or "unspecified", []).append(row)
        for tag in row.tags:
            tag_groups.setdefault(tag, []).append(row)
    return {
        "overall": summarize_results(rows),
        "by_language": {key: summarize_results(language_groups[key]) for key in sorted(language_groups)},
        "by_tag": {key: summarize_results(tag_groups[key]) for key in sorted(tag_groups)},
    }


@dataclass(frozen=True)
class PairedPromotionRow:
    """Content-free metric inputs for one frozen paired evaluation sample.

    The durable evaluator stores only these sufficient statistics in its admin
    surface; references and transcripts remain in the private learning store.
    """
    sample_id: str
    champion_word_distance: int
    candidate_word_distance: int
    reference_words: int
    champion_character_distance: int = 0
    candidate_character_distance: int = 0
    reference_characters: int = 0
    champion_hallucination: bool = False
    candidate_hallucination: bool = False
    champion_latency: float = 0.0
    candidate_latency: float = 0.0


def _paired_micro(rows: Sequence[PairedPromotionRow], candidate: bool) -> tuple[float | None, float | None]:
    words = sum(row.reference_words for row in rows)
    chars = sum(row.reference_characters for row in rows)
    word_distance = sum((row.candidate_word_distance if candidate else row.champion_word_distance) for row in rows)
    char_distance = sum((row.candidate_character_distance if candidate else row.champion_character_distance) for row in rows)
    return (word_distance / words if words else None, char_distance / chars if chars else None)


def paired_bootstrap_statistics(
    rows: Iterable[PairedPromotionRow], generation_id: str, *, evaluator_id: str = PAIRED_EVALUATOR_ID,
    replicates: int = PAIRED_BOOTSTRAP_REPLICATES,
) -> dict:
    """Run the specified deterministic paired micro-WER bootstrap.

    Empty references must not be supplied here.  This deliberate validation
    prevents a caller accidentally treating no-speech examples as WER data.
    """
    frozen = tuple(rows)
    if evaluator_id != PAIRED_EVALUATOR_ID:
        raise ValueError("unsupported paired evaluator")
    if not frozen or any(row.reference_words <= 0 for row in frozen):
        raise ValueError("paired bootstrap requires nonempty references")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    champion_wer, champion_cer = _paired_micro(frozen, False)
    candidate_wer, candidate_cer = _paired_micro(frozen, True)
    assert champion_wer is not None and candidate_wer is not None
    observed = candidate_wer - champion_wer
    seed_digest = hashlib.sha256((generation_id + evaluator_id).encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(seed_digest, "big"))
    n = len(frozen)
    draws: list[float] = []
    for _ in range(replicates):
        sampled = [frozen[rng.randrange(n)] for _ in range(n)]
        cw, _ = _paired_micro(sampled, False)
        nw, _ = _paired_micro(sampled, True)
        assert cw is not None and nw is not None
        draws.append(nw - cw)
    draws.sort()
    upper_ci = draws[math.ceil(.95 * replicates) - 1]
    # One-sided centered bootstrap, H1: candidate - champion < 0.
    p_value = (1 + sum((draw - observed) <= observed for draw in draws)) / (replicates + 1)
    wins_or_ties = sum(
        row.candidate_word_distance / row.reference_words <= row.champion_word_distance / row.reference_words
        for row in frozen
    ) / n
    return {
        "evaluator_id": evaluator_id,
        "seed_sha256": seed_digest.hex(),
        "replicates": replicates,
        "nonempty_count": n,
        "champion_wer": champion_wer,
        "candidate_wer": candidate_wer,
        "champion_cer": champion_cer,
        "candidate_cer": candidate_cer,
        "difference": observed,
        "upper_ci": upper_ci,
        "p_value": p_value,
        "win_or_tie_rate": wins_or_ties,
        "champion_hallucinations": sum(row.champion_hallucination for row in frozen),
        "candidate_hallucinations": sum(row.candidate_hallucination for row in frozen),
        "champion_latency_median": _median([row.champion_latency for row in frozen]),
        "candidate_latency_median": _median([row.candidate_latency for row in frozen]),
    }


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def holm_fixed_family(p_values: Iterable[tuple[str, float]], *, alpha: float = .05) -> dict[str, bool]:
    """Fixed-family Holm step-down, sorted by stable candidate id on ties."""
    ordered = sorted(p_values, key=lambda pair: (pair[1], pair[0]))
    total = len(ordered)
    accepted: dict[str, bool] = {}
    blocked = False
    for index, (stable_id, p_value) in enumerate(ordered):
        passes = not blocked and p_value <= alpha / (total - index)
        accepted[stable_id] = passes
        if not passes:
            blocked = True
    return accepted


# Explicit aliases make the protocol discoverable to integrations without
# changing the established evaluate_sample / aggregate_results API.
paired_promotion_statistics = paired_bootstrap_statistics
holm_step_down = holm_fixed_family

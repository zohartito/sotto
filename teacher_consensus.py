"""Deterministic, deliberately conservative two-family silver consensus."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

POLICY_VERSION = "silver-v1-exact-unanimity"
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+", re.ASCII)
_COMMAND = frozenset({"delete", "send", "submit", "pay", "transfer", "call", "open", "close", "cancel", "confirm"})
_NEGATION = frozenset({"no", "not", "never", "none", "cannot", "can't", "don't", "won't"})
_NUMBER = frozenset("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth dollar dollars cent cents percent am pm".split())


def normalized_tokens(text: str) -> tuple[str, ...]:
    """Frozen normalization used for both equality and persisted references."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return tuple(_TOKEN.findall(folded))

def normalization_preserves_words(text: str) -> bool:
    """V1 is ASCII-token based; never silently erase non-ASCII word content."""
    return not any((ch.isalpha() or ch.isdigit()) and ord(ch) > 127 for ch in unicodedata.normalize("NFKC", text))


def canonical_reference(tokens: Iterable[str]) -> str:
    return " ".join(tokens)


def high_risk(text: str, tokens: tuple[str, ...] | None = None) -> bool:
    """V1 intentionally abstains from terms where exact text is still risky."""
    words = tokens if tokens is not None else normalized_tokens(text)
    if any(token.isdigit() for token in words) or any(token in _COMMAND or token in _NEGATION or token in _NUMBER for token in words):
        return True
    # Uppercase initials/acronyms and name-like capitalized words are excluded.
    raw_words = re.findall(r"[A-Za-z]+", unicodedata.normalize("NFKC", text))
    # Sentence-initial conventional capitalization is ordinary English, not a
    # personal-name signal.  Any later title-cased word remains conservative.
    for index, word in enumerate(raw_words):
        if len(word) >= 2 and word.isupper():
            return True
        if index > 0 and word[:1].isupper() and word[1:].islower():
            return True
    return False


def policy_hash() -> str:
    # The policy is executable normalization/risk control flow, not just its
    # visible constants.  Bind the complete source and Unicode runtime so a
    # changed tokenizer/high-risk branch cannot reuse earlier consensus.
    body=(POLICY_VERSION + "\0" + _TOKEN.pattern + "\0" + ",".join(sorted(_COMMAND | _NEGATION | _NUMBER))).encode()
    body += b"\0" + Path(__file__).read_bytes() + b"\0" + sys.version.encode() + b"\0" + unicodedata.unidata_version.encode()
    return hashlib.sha256(body).hexdigest()


@dataclass(frozen=True)
class Consensus:
    state: str                 # accepted or abstained
    reference: str | None
    reason: str                # content-free code
    vote_digest: str


def exact_unanimous(results: dict[str, str | None], required_families: Iterable[str]) -> Consensus:
    """Accept only two successful, nonempty, equal low-risk teacher outputs.

    The returned diagnostic is intentionally a fixed code and digest: callers
    must never log raw teacher strings.
    """
    families = tuple(required_families)
    # Do not retain a hash of teacher text: common phrases make an unhashed
    # local dictionary attack practical.  This is an opaque protocol/event
    # fingerprint only, with no transcript-derived material.
    digest = hashlib.sha256(json.dumps({"policy": POLICY_VERSION, "families": sorted(families),
                                        "present": [isinstance(results.get(f), str) for f in sorted(families)]},
                                       sort_keys=True).encode()).hexdigest()
    if len(families) != 2 or len(set(families)) != 2:
        return Consensus("abstained", None, "teacher_configuration", digest)
    values = [results.get(f) for f in families]
    if any(not isinstance(v, str) for v in values):
        return Consensus("abstained", None, "teacher_failure", digest)
    if any(not normalization_preserves_words(v or "") for v in values):
        return Consensus("abstained", None, "unicode_unsupported", digest)
    tokens = [normalized_tokens(v or "") for v in values]
    if any(not x for x in tokens):
        return Consensus("abstained", None, "blank_or_no_speech", digest)
    if tokens[0] != tokens[1]:
        return Consensus("abstained", None, "mismatch", digest)
    if any(high_risk(v or "", token) for v, token in zip(values, tokens)):
        return Consensus("abstained", None, "high_risk", digest)
    return Consensus("accepted", canonical_reference(tokens[0]), "exact_unanimous", digest)

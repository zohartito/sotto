"""Personal dictionary: deterministic "heard => write" replacements.

Applied to the final dictation text right before it is pasted, so Sotto
spells names and terms your way. Corrections in History can propose new
rules; nothing is added without an explicit confirmation. Plain local text,
pure logic (no platform imports), shared by the macOS and Windows apps.
"""
from __future__ import annotations

from dataclasses import dataclass
import difflib
import os
from pathlib import Path
import re
import tempfile

from sotto_paths import DATA_DIR

DICTIONARY_PATH = DATA_DIR / "dictionary.txt"
SEPARATOR = "=>"
COMMENT = "#"  # starts a comment line, so no rule may start with it
MAX_RULES = 500
MAX_SIDE_CHARS = 120
MAX_SUGGESTIONS = 5
HEADER = (
    "# Sotto dictionary: one rule per line, written as\n"
    "#   what Sotto hears => what you want written\n"
    "# Matching ignores case and only replaces whole words or phrases.\n"
    "# Edits apply to the next dictation. Example:\n"
    "#   whisper flow => Wispr Flow\n"
)
# Single common words make poor global rules; a correction of one of these is
# almost always about that one sentence, so it is never suggested.
_COMMON_WORDS = frozenset(
    "a an and are as at be but by for from he her his i if in is it its me my no not "
    "of on or our she so that the their them then there they this to up us was we "
    "were what when which who will with you your".split())


@dataclass(frozen=True)
class Rule:
    heard: str
    write: str


def _clean(side: str) -> str:
    return " ".join(side.split())


def parse(text: str) -> list[Rule]:
    """Valid rules in file order; blank, comment and malformed lines are skipped."""
    rules: list[Rule] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith(COMMENT) or SEPARATOR not in line:
            continue
        heard, write = (_clean(part) for part in line.split(SEPARATOR, 1))
        key = heard.lower()
        if (not heard or not write or key in seen
                or len(heard) > MAX_SIDE_CHARS or len(write) > MAX_SIDE_CHARS):
            continue
        seen.add(key)
        rules.append(Rule(heard, write))
        if len(rules) == MAX_RULES:
            break
    return rules


_cache: dict[str, tuple[tuple[int, int], list[Rule]]] = {}


def load(path: Path | str = DICTIONARY_PATH) -> list[Rule]:
    """Current rules, re-read only when the file changes; missing file = none."""
    path = Path(path)
    try:
        info = path.stat()
    except OSError:
        return []
    stamp = (info.st_mtime_ns, info.st_size)
    cached = _cache.get(str(path))
    if cached is None or cached[0] != stamp:
        try:
            rules = parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            return []
        cached = _cache[str(path)] = (stamp, rules)
    return list(cached[1])


def _pattern(rules: list[Rule]) -> re.Pattern | None:
    if not rules:
        return None
    # Longest first so "open ai codex" wins over "open ai"; any whitespace run
    # between words matches, and word boundaries are Unicode-aware.
    alternatives = sorted({r"\s+".join(map(re.escape, rule.heard.split())) for rule in rules},
                          key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.IGNORECASE)


def apply(text: str, rules: list[Rule]) -> tuple[str, list[dict]]:
    """Rewrite whole-word matches; returns (text, receipt of rules used)."""
    pattern = _pattern(rules)
    if pattern is None or not text:
        return text, []
    by_key = {rule.heard.lower(): rule for rule in rules}
    counts: dict[str, int] = {}

    def replace(match: re.Match) -> str:
        rule = by_key.get(_clean(match.group(0)).lower())
        if rule is None:  # exotic case-folding mismatch: leave the text alone
            return match.group(0)
        counts[rule.heard] = counts.get(rule.heard, 0) + 1
        write = rule.write
        # "Gonna" at a sentence start stays capitalized when the rule is lowercase.
        if match.group(0)[:1].isupper() and write[:1].islower():
            write = write[:1].upper() + write[1:]
        return write

    result = pattern.sub(replace, text)
    receipt = [{"heard": rule.heard, "write": rule.write, "count": counts[rule.heard]}
               for rule in rules if rule.heard in counts]
    return result, receipt


# Sentence punctuation and quotes around a word are not part of the term;
# symbols such as + # @ & are (a correction to "C++" must stay "C++").
_EDGE = ".,;:!?\"'“”‘’«»()[]{}…"


def _strip_edges(word: str) -> str:
    return word.strip(_EDGE)


def suggest(before: str, after: str, existing: list[Rule] = ()) -> list[Rule]:
    """Word-level substitutions a correction made that could become rules."""
    raw = before.split()
    old = [_strip_edges(word) for word in raw]
    new = [_strip_edges(word) for word in after.split()]
    known = {rule.heard.lower() for rule in existing}
    # Align case-insensitively so a sentence-start capital never glues its
    # neighbours into a rule. Words whose case changed next to a replacement
    # belong to it ("whisper flow" -> "Wispr Flow"); other interior-case fixes
    # (iphone -> iPhone) become single-word rules.
    matcher = difflib.SequenceMatcher(a=[word.lower() for word in old],
                                      b=[word.lower() for word in new], autojunk=False)
    opcodes = matcher.get_opcodes()
    aligned = {i1 + k: j1 + k for tag, i1, i2, j1, _j2 in opcodes if tag == "equal" for k in range(i2 - i1)}

    def recased(i: int) -> bool:
        if i not in aligned or old[i] == new[aligned[i]]:
            return False
        at_sentence_start = i == 0 or raw[i - 1][-1:] in ".!?"
        return not (at_sentence_start and old[i][1:] == new[aligned[i]][1:])

    consumed: set[int] = set()
    candidates: list[tuple[str, str, int]] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "replace":
            continue
        while i1 - 1 in aligned and recased(i1 - 1) and i2 - i1 < 4:
            i1, j1 = i1 - 1, j1 - 1
        while i2 in aligned and recased(i2) and i2 - i1 < 4:
            i2, j2 = i2 + 1, j2 + 1
        if 1 <= i2 - i1 <= 4 and 1 <= j2 - j1 <= 6:
            consumed.update(range(i1, i2))
            candidates.append((" ".join(old[i1:i2]), " ".join(new[j1:j2]), i2 - i1))
    candidates += [(old[i], new[j], 1) for i, j in sorted(aligned.items())
                   if i not in consumed and old[i][1:] != new[j][1:]]
    suggestions: list[Rule] = []
    for heard, write, words in candidates:
        heard, write = _clean(heard), _clean(write)
        if (not heard or not write or heard == write or heard.lower() in known
                or heard.startswith(COMMENT)
                or len(heard) > MAX_SIDE_CHARS or len(write) > MAX_SIDE_CHARS
                or (words == 1 and heard.lower() in _COMMON_WORDS)):
            continue
        known.add(heard.lower())
        suggestions.append(Rule(heard, write))
        if len(suggestions) == MAX_SUGGESTIONS:
            break
    return suggestions


def ensure_file(path: Path | str = DICTIONARY_PATH) -> Path:
    """Create a private dictionary with instructions if none exists yet."""
    path = Path(path)
    if not path.exists():
        _write_private(path, HEADER)
    return path


def add(new_rules: list[Rule], path: Path | str = DICTIONARY_PATH) -> int:
    """Append rules whose 'heard' side is new; returns how many were added.

    A heard side starting with ``COMMENT`` raises ValueError and nothing is
    written: the file would read that line as a comment, so the rule would be
    reported as added and never apply (N31)."""
    for rule in new_rules:
        heard = _clean(rule.heard)
        if heard.startswith(COMMENT):
            raise ValueError(f'"{heard}" cannot be a dictionary rule: a line starting with '
                             f'"{COMMENT}" is a comment in the dictionary file.')
    path = Path(path)
    try:
        current = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        current = HEADER
    known = {rule.heard.lower() for rule in parse(current)}
    lines = []
    for rule in new_rules:
        heard, write = _clean(rule.heard), _clean(rule.write)
        if (heard and write and heard.lower() not in known and SEPARATOR not in heard
                and len(heard) <= MAX_SIDE_CHARS and len(write) <= MAX_SIDE_CHARS
                and len(known) < MAX_RULES):
            known.add(heard.lower())
            lines.append(f"{heard} {SEPARATOR} {write}")
    if lines:
        if current and not current.endswith("\n"):
            current += "\n"
        _write_private(path, current + "\n".join(lines) + "\n")
    return len(lines)


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".dictionary-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)

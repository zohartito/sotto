"""Is Sotto getting better for you? A local summary of recent History.

Pure logic shared by the macOS menu and the Windows tray: counts and timings
only, never transcript text. Everything comes from History rows on this
machine (the retained window, at most the last 200 dictations).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import tempfile
import time

DAY = 86_400.0
TYPING_WPM = 40          # a typical typing speed; the estimate says so
TOTALS_NAME = "stats.json"  # lifetime counts in the data folder, never text


def _latency(entry: dict):
    value = (entry.get("latency") or {}).get("release_to_text_seconds")
    return float(value) if isinstance(value, (int, float)) else None


def _replacements(entry: dict) -> int:
    receipt = (entry.get("preprocessing") or {}).get("dictionary") or {}
    return sum(int(item.get("count", 0)) for item in receipt.get("rules", []) if isinstance(item, dict))


def _corrected(entry: dict) -> bool:
    """A real fix, not "Transcript is correct" or "no speech" reviews."""
    return (entry.get("correction") or {}).get("outcome") == "corrected"


def _window(entries: list[dict], start: float, end: float) -> list[dict]:
    return [entry for entry in entries
            if isinstance(entry.get("ts"), (int, float)) and start < entry["ts"] <= end]


def summarize(entries: list[dict], *, rules: int, now: float | None = None, days: int = 7) -> dict:
    """Numbers for this period and the one before it (for a trend)."""
    now = time.time() if now is None else now
    span = days * DAY
    current = _window(entries, now - span, now)
    previous = _window(entries, now - 2 * span, now - span)

    def period(rows: list[dict]) -> dict:
        latencies = [value for value in map(_latency, rows) if value is not None]
        corrected = sum(1 for row in rows if _corrected(row))
        return {"dictations": len(rows),
                "median_latency": statistics.median(latencies) if latencies else None,
                "corrected_share": corrected / len(rows) if rows else None,
                "replacements": sum(map(_replacements, rows))}

    return {"days": days, "current": period(current), "previous": period(previous),
            "rules": rules,
            "corrections": sum(1 for entry in entries if _corrected(entry)),
            "learning_samples": sum(1 for entry in entries if entry.get("learning_state") == "active")}


def _delivered(entry: dict) -> bool:
    """Text that was actually inserted (not held back as suspect)."""
    if not str(entry.get("text") or "").strip():
        return False
    if (entry.get("preprocessing") or {}).get("outcome") == "suspect":
        return False
    attempts = entry.get("attempts") or [{}]
    return attempts[-1].get("provenance") != "live_suspect"


def _zero() -> dict:
    return {"dictations": 0, "words": 0, "seconds": 0.0}


def _write(path: Path, totals: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".stats-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(totals, stream)
            # On disk before it replaces the old file (N32): a torn stats.json
            # is never rewritten again under record()'s strict parse (F38).
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _parse_totals(text: str, *, strict: bool = False) -> dict:
    """The totals in ``text``; malformed text reads as zero for display, or
    raises ValueError when ``strict`` (before an update would overwrite it)."""
    try:
        data = json.loads(text)
        return {"dictations": int(data.get("dictations", 0)), "words": int(data.get("words", 0)),
                "seconds": float(data.get("seconds", 0.0))}
    except (ValueError, TypeError, AttributeError) as exc:
        if strict:
            raise ValueError(f"stats file is malformed: {type(exc).__name__}") from exc
        return _zero()


def load_totals(path: Path) -> dict:
    try:
        return _parse_totals(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return _zero()


def ensure_totals(path: Path, entries: list[dict]) -> dict:
    """Start the lifetime counts from History once, so they are not zero on day one."""
    path = Path(path)
    if path.exists():
        return load_totals(path)
    totals = _zero()
    for entry in entries:
        if _delivered(entry):
            totals["dictations"] += 1
            totals["words"] += len(str(entry["text"]).split())
            totals["seconds"] += float(entry.get("duration") or 0.0)
    _write(path, totals)
    return totals


def record(path: Path, *, text: str, seconds: float) -> None:
    """Add one inserted dictation: its word count and how long it took to say.

    Side effects: rewrites ``path``. Only a missing file starts from zero: any
    other read error (a passing lock or permission glitch) raises OSError, and
    a malformed file raises ValueError, writing nothing, so neither can replace
    the lifetime totals with this one dictation (F38)."""
    try:
        totals = _parse_totals(Path(path).read_text(encoding="utf-8"), strict=True)
    except FileNotFoundError:
        totals = _zero()
    totals["dictations"] += 1
    totals["words"] += len(text.split())
    totals["seconds"] += max(0.0, float(seconds))
    _write(Path(path), totals)


def saved_minutes(totals: dict) -> float:
    """Typing the same words at TYPING_WPM, minus the time spent talking."""
    return max(0.0, totals["words"] / TYPING_WPM - totals["seconds"] / 60.0)


def totals_line(totals: dict) -> str | None:
    if not totals["dictations"]:
        return None
    minutes = saved_minutes(totals)
    if minutes >= 60:
        saved = f"about {minutes / 60:.1f} hours"
    elif minutes >= 1:
        saved = f"about {round(minutes)} minute{'' if round(minutes) == 1 else 's'}"
    else:
        saved = "less than a minute"
    return f"All time: {totals['words']:,} words · {saved} saved vs typing at {TYPING_WPM} wpm"


def lines(summary: dict, totals: dict | None = None) -> list[str]:
    """Short menu-sized lines; no transcript text ever appears here."""
    days, now, before = summary["days"], summary["current"], summary["previous"]
    lifetime = totals_line(totals) if totals else None
    first = f"Last {days} days: {now['dictations']} dictation{'' if now['dictations'] == 1 else 's'}"
    if now["median_latency"] is not None:
        first += f" · ready {now['median_latency']:.2f}s after release (median)"
        if before["median_latency"] is not None:
            first += f", was {before['median_latency']:.2f}s"
    result = ([lifetime] if lifetime else []) + [first]
    if now["corrected_share"] is not None:
        share = f"Needed a correction: {now['corrected_share']:.0%} of dictations"
        if before["corrected_share"] is not None:
            share += f" (was {before['corrected_share']:.0%})"
        result.append(share)
    result.append(f"Dictionary: {summary['rules']} rule{'' if summary['rules'] == 1 else 's'}"
                  f" · fixed {now['replacements']} word{'' if now['replacements'] == 1 else 's'} this period")
    result.append(f"Corrections saved: {summary['corrections']}"
                  f" · in learning set: {summary['learning_samples']}")
    return result

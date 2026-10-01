"""Is Sotto getting better for you? A local summary of recent History.

Pure logic shared by the macOS menu and the Windows tray: counts and timings
only, never transcript text. Everything comes from History rows on this
machine (the retained window, at most the last 200 dictations).
"""
from __future__ import annotations

import statistics
import time

DAY = 86_400.0


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


def lines(summary: dict) -> list[str]:
    """Short menu-sized lines; no transcript text ever appears here."""
    days, now, before = summary["days"], summary["current"], summary["previous"]
    first = f"Last {days} days: {now['dictations']} dictation{'' if now['dictations'] == 1 else 's'}"
    if now["median_latency"] is not None:
        first += f" · ready {now['median_latency']:.2f}s after release (median)"
        if before["median_latency"] is not None:
            first += f", was {before['median_latency']:.2f}s"
    result = [first]
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

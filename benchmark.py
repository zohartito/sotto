"""Headless, offline transcription model A/B benchmark CLI."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import wave
from typing import Any, Iterable

from evaluation import aggregate_results, evaluate_sample
from speech_config import glossary_prompt, load_glossary, resolve_speech_config


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    audio: Path
    reference: str
    language: str | None
    tags: tuple[str, ...]
    required_terms: tuple[str, ...]


@dataclass(frozen=True)
class ModelCandidate:
    label: str
    repo: str


def load_manifest(path: str | Path, limit: int | None = None) -> list[ManifestRow]:
    """Read and validate a JSONL manifest, resolving audio relative to it."""

    manifest_path = Path(path).resolve()
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Unable to read manifest {manifest_path}: {exc}") from exc
    rows: list[ManifestRow] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Manifest line {line_number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Manifest line {line_number}: expected an object")
        audio, reference = value.get("audio"), value.get("reference")
        if not isinstance(audio, str) or not audio.strip():
            raise ValueError(f"Manifest line {line_number}: 'audio' must be a non-empty string")
        if not isinstance(reference, str):
            raise ValueError(f"Manifest line {line_number}: 'reference' must be a string")
        tags = value.get("tags", [])
        required_terms = value.get("required_terms", [])
        if isinstance(tags, str):
            tags = [tags]
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"Manifest line {line_number}: 'tags' must be a string list")
        if not isinstance(required_terms, list) or not all(isinstance(term, str) for term in required_terms):
            raise ValueError(f"Manifest line {line_number}: 'required_terms' must be a string list")
        language = value.get("language")
        if language is not None and not isinstance(language, str):
            raise ValueError(f"Manifest line {line_number}: 'language' must be a string")
        audio_path = (manifest_path.parent / audio).resolve()
        if not audio_path.is_file():
            raise ValueError(f"Manifest line {line_number}: audio file does not exist: {audio_path}")
        sample_id = value.get("id", str(line_number))
        if not isinstance(sample_id, str):
            sample_id = str(sample_id)
        rows.append(ManifestRow(sample_id, audio_path, reference, language,
                                tuple(sorted(set(tags))), tuple(required_terms)))
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError("Manifest contains no samples")
    return rows


def parse_model_candidate(value: str) -> ModelCandidate:
    """Parse ``label=repo`` or use the repository itself as the label."""

    label, separator, repo = value.partition("=")
    if separator:
        if not label.strip() or not repo.strip():
            raise ValueError("Model must be 'label=repo' with non-empty values")
        return ModelCandidate(label.strip(), repo.strip())
    if not value.strip():
        raise ValueError("Model repository must not be empty")
    return ModelCandidate(value.strip(), value.strip())


def _duration_seconds(audio_path: Path) -> float | None:
    """Get WAV duration when possible; unknown formats intentionally get no prompt."""

    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            if not wav_file.getframerate():
                return None
            return wav_file.getnframes() / wav_file.getframerate()
    except (wave.Error, OSError):
        return None


def _transcribe(audio_path: Path, model_repo: str, language: str | None, prompt: str | None) -> str:
    """Lazy MLX import: importing this CLI and dry runs never load MLX Whisper."""

    import mlx_whisper  # type: ignore[import-not-found]

    kwargs: dict[str, Any] = {"path_or_hf_repo": model_repo}
    if language is not None:
        kwargs["language"] = language
    if prompt is not None:
        kwargs["initial_prompt"] = prompt
    response = mlx_whisper.transcribe(str(audio_path), **kwargs)
    if not isinstance(response, dict) or not isinstance(response.get("text"), str):
        raise RuntimeError("mlx_whisper.transcribe returned no text")
    return response["text"]


def run_benchmark(
    rows: Iterable[ManifestRow], candidates: Iterable[ModelCandidate], *, profile: str,
    language: str | None, glossary_terms: Iterable[str],
) -> dict:
    """Transcribe locally and return raw hypotheses plus per-model aggregates."""

    output_models: list[dict] = []
    for candidate in candidates:
        config = resolve_speech_config(profile, model=candidate.repo, language=language)
        sample_reports: list[dict] = []
        metric_rows = []
        for row in rows:
            prompt_decision = glossary_prompt(glossary_terms, _duration_seconds(row.audio))
            started = time.perf_counter()
            hypothesis = _transcribe(row.audio, config.model_repo, config.language, prompt_decision.prompt)
            latency_seconds = time.perf_counter() - started
            result = evaluate_sample(
                row.reference, hypothesis, sample_id=row.sample_id, language=row.language,
                tags=row.tags, required_terms=row.required_terms,
            )
            metric_rows.append(result)
            sample_reports.append({
                "id": row.sample_id,
                "audio": str(row.audio),
                "model_repo": config.model_repo,
                "profile": config.profile.name,
                "language": config.language,
                "prompt_used": prompt_decision.prompt,
                "prompt_reason": prompt_decision.reason,
                "latency_seconds": latency_seconds,
                "hypothesis": hypothesis,
                "hallucination_proxy": result.hallucination_proxy,
                "metrics": result.to_dict(),
            })
        output_models.append({
            "label": candidate.label,
            "model_repo": config.model_repo,
            "profile": config.profile.name,
            "language": config.language,
            "samples": sample_reports,
            "aggregate": aggregate_results(metric_rows),
        })
    return {"models": output_models}


def markdown_report(report: dict) -> str:
    """Create a compact, deterministic human-readable summary."""

    if report.get("dry_run"):
        models = report.get("models", [])
        lines = [
            "# Sotto offline speech benchmark dry run",
            "",
            f"Validated {report.get('sample_count', 0)} manifest samples without loading a model.",
            "",
            "| Label | Repository |",
            "| --- | --- |",
        ]
        lines.extend(f"| {model['label']} | `{model['repo']}` |" for model in models)
        return "\n".join(lines) + "\n"
    lines = ["# Sotto offline speech benchmark", "", "This report is comparative; sample count alone does not establish statistical sufficiency.", ""]
    for model in report.get("models", []):
        overall = model["aggregate"]["overall"]
        metrics = overall["metrics"]
        lines.extend([
            f"## {model['label']}",
            "",
            f"- Repository: `{model['model_repo']}`",
            f"- Profile: `{model['profile']}`; language: `{model['language'] or 'auto'}`",
            f"- Samples: {overall['count']} ({overall['uncertainty_note']})",
            f"- WER: {metrics['wer']['value']!s}; CER: {metrics['cer']['value']!s}",
            "",
            "### Slices",
            "",
            "| Slice | Samples | Directional | WER |",
            "| --- | ---: | --- | ---: |",
        ])
        for kind in ("by_language", "by_tag"):
            for key, slice_summary in model["aggregate"][kind].items():
                wer = slice_summary["metrics"]["wer"]["value"]
                lines.append(f"| {kind[3:]}: {key} | {slice_summary['count']} | {slice_summary['directional']} | {wer!s} |")
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an offline Sotto speech-model benchmark.")
    parser.add_argument("manifest", help="JSONL manifest containing audio and reference fields")
    parser.add_argument("--model", action="append", default=[], metavar="LABEL=REPO",
                        help="Candidate MLX repo (repeatable); use label=repo to name it")
    parser.add_argument("--profile", default="auto", help="Base profile (default: auto)")
    parser.add_argument("--language", default=None,
                        help="Explicit language override")
    parser.add_argument("--glossary", help="Local UTF-8 text or JSON glossary")
    parser.add_argument("--limit", type=int, help="Evaluate at most this many manifest rows")
    parser.add_argument("--output-json", help="Write the full JSON report to this path")
    parser.add_argument("--output-markdown", help="Write a Markdown summary to this path")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without importing MLX or transcribing")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    try:
        rows = load_manifest(args.manifest, args.limit)
        glossary_terms = load_glossary(args.glossary) if args.glossary else ()
        base_config = resolve_speech_config(args.profile, language=args.language)
        candidates = [parse_model_candidate(value) for value in args.model]
        if not candidates:
            candidates = [ModelCandidate(base_config.profile.name, base_config.model_repo)]
        if len({candidate.label for candidate in candidates}) != len(candidates):
            raise ValueError("Model labels must be unique")
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        report = {
            "dry_run": True,
            "manifest": str(Path(args.manifest).resolve()),
            "sample_count": len(rows),
            "glossary_terms": len(glossary_terms),
            "models": [asdict(candidate) for candidate in candidates],
            "profile": base_config.profile.name,
            "language_override": args.language,
        }
    else:
        report = run_benchmark(rows, candidates, profile=args.profile,
                               language=args.language, glossary_terms=glossary_terms)
        report.update({"dry_run": False, "manifest": str(Path(args.manifest).resolve()),
                       "sample_count": len(rows), "glossary_terms": len(glossary_terms)})
    json_payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(json_payload, encoding="utf-8")
    if args.output_markdown:
        Path(args.output_markdown).write_text(markdown_report(report), encoding="utf-8")
    if not args.output_json:
        print(json_payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

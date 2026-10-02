#!/usr/bin/env python3
"""Build a deterministic source-only archive from an explicit public allowlist.

No Git database, environments, state, models, recordings or local audit evidence.
Run from an inspected candidate: python scripts/build_alpha_source.py OUTPUT.tar.gz
"""
import gzip
import hashlib
import io
from pathlib import Path
import sys
import tarfile

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from runtime_source_manifest import RUNTIME_SOURCE_FILES


PUBLIC_TEST_FILES = (
    'tests/fixtures/benchmark_gate/manifest.jsonl',
    'tests/fixtures/benchmark_gate/silence.wav',
    'tests/fixtures/calibration_v2_expected/part-00.json',
    'tests/fixtures/calibration_v2_expected/part-01.json',
    'tests/fixtures/calibration_v2_expected/part-02.json',
    'tests/fixtures/calibration_v2_expected/part-03.json',
    'tests/fixtures/calibration_v2_expected/part-04.json',
    'tests/fixtures/calibration_v2_expected/part-05.json',
    'tests/fixtures/calibration_v2_expected/part-06.json',
    'tests/fixtures/calibration_v2_expected/part-07.json',
    'tests/fixtures/calibration_v2_expected/part-08.json',
    'tests/fixtures/calibration_v2_expected/part-09.json',
    'tests/fixtures/calibration_v2_expected/part-10.json',
    'tests/fixtures/calibration_v2_expected/part-11.json',
    'tests/fixtures/calibration_v2_expected/part-12.json',
    'tests/fixtures/calibration_v2_expected/part-13.json',
    'tests/fixtures/calibration_v2_expected/part-14.json',
    'tests/fixtures/calibration_v2_expected/part-15.json',
    'tests/fixtures/calibration_v2_expected/part-16.json',
    'tests/fixtures/calibration_v2_expected/part-17.json',
    'tests/fixtures/calibration_v2_expected/part-18.json',
    'tests/fixtures/calibration_v2_expected/part-19.json',
    'tests/gesture_check.py',
    'tests/test_adaptive_learning.py',
    'tests/test_adaptive_runtime.py',
    'tests/test_alpha_onboarding.py',
    'tests/test_comparator_spool.py',
    'tests/test_dead_route.py',
    'tests/test_dictionary.py',
    'tests/test_settings.py',
    'tests/test_insertion.py',
    'tests/test_settings_window.py',
    'tests/test_progress.py',
    'tests/test_updates.py',
    'tests/test_voice_commands.py',
    'tests/test_app_install.py',
    'tests/test_parakeet_engine.py',
    'tests/test_dead_route_repair.py',
    'tests/test_evaluation.py',
    'tests/test_history_learning.py',
    'tests/test_hotkey_state.py',
    'tests/test_inference_scheduler.py',
    'tests/test_nemotron.py',
    'tests/test_nemotron_pipeline.py',
    'tests/test_offline_backup.py',
    'tests/test_repetition_loop.py',
    'tests/test_runtime_source_manifest.py',
    'tests/test_sealed_release.py',
    'tests/test_silver_lane.py',
    'tests/test_speech_backends.py',
    'tests/test_speech_config.py',
    'tests/test_speech_runtime.py',
    'tests/test_teacher_backends.py',
    'tests/test_ui_overlay.py',
    'tests/test_vad_gate.py',
    'tests/test_workflow_action_pinning.py',
)

# Windows source alpha: entry points, Windows I/O and tray, installer, pins,
# docs and tests, plus the shared dictionary/settings modules it imports.
WINDOWS_FILES = (
    'sotto_win.py',
    'win_asr.py',
    'win_capture.py',
    'win_hotkey.py',
    'win_inject.py',
    'win_launch.py',
    'win_startup.py',
    'win_ui.py',
    'dictionary.py',
    'settings.py',
    'voice_commands.py',
    'updates.py',
    'scripts/install-windows.ps1',
    'scripts/update-windows.ps1',
    'scripts/sotto-win.bat',
    'requirements-alpha-windows.txt',
    'constraints-alpha-windows.txt',
    'requirements-alpha-windows-cuda.txt',
    'constraints-alpha-windows-cuda.txt',
    'docs/windows-alpha.md',
    'tests/test_audio_codec_write.py',
    'tests/test_dictionary.py',
    'tests/test_settings.py',
    'tests/test_sotto_win.py',
    'tests/test_storage_lock.py',
    'tests/test_win_alpha.py',
    'tests/test_win_asr.py',
    'tests/test_win_capture.py',
    'tests/test_win_hotkey.py',
    'tests/test_win_inject.py',
    'tests/test_win_installer.py',
    'tests/test_win_startup.py',
    'tests/test_win_ui.py',
)

PUBLIC_FILES = sorted(set(RUNTIME_SOURCE_FILES) | set(PUBLIC_TEST_FILES) | set(WINDOWS_FILES) | {
    'README.md', 'LICENSE', 'AGENTS.md', 'requirements-alpha.txt', 'constraints-alpha.txt',
    'scripts/setup_nemotron.py', 'scripts/test_alpha.py', 'scripts/build_alpha_source.py',
    'scripts/install-mac.sh', 'scripts/install_app.py',
    'scripts/rollout.sh', 'benchmark.py', 'docs/alpha-release-notes.md',
    'docs/alpha-validation.md', 'docs/advanced-development.md',
    'docs/speech-learning.md', 'docs/english-adaptive-research.md',
    'teachers/requirements-qwen.txt', 'teachers/requirements-granite.txt',
    '.gitignore', '.github/workflows/secret-scan.yml',
    '.github/workflows/benchmark-gate.yml', '.github/workflows/claude-pr-review.yml',
    '.github/workflows-disabled/README.md', '.github/workflows-disabled/ci.yml.disabled',
})


def build(output: Path) -> dict:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w', format=tarfile.PAX_FORMAT) as archive:
        for name in PUBLIC_FILES:
            source = root / name
            if source.is_symlink() or not source.is_file():
                raise RuntimeError('Public source file missing or unsafe: ' + name)
            data = source.read_bytes()
            info = tarfile.TarInfo('sotto-alpha/' + name)
            info.size = len(data)
            # Keep only the executable bit (rollout.sh); owner/time are normalized.
            info.mode = 0o755 if source.stat().st_mode & 0o100 else 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    with output.open('wb') as handle:
        with gzip.GzipFile(filename='', mode='wb', fileobj=handle, mtime=0) as zipped:
            zipped.write(buffer.getvalue())
    return {'files': len(PUBLIC_FILES), 'bytes': output.stat().st_size,
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == '__main__':
    import json
    if len(sys.argv) != 2:
        raise SystemExit('usage: python scripts/build_alpha_source.py OUTPUT.tar.gz')
    print(json.dumps(build(Path(sys.argv[1])), indent=2))

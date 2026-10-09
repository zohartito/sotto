# Alpha validation

What was checked before this alpha was shared, and what those checks do and do
not establish. It is not a performance benchmark. Every check used its own
temporary state and cache; no personal Sotto data was involved.

## Reproduce the checks (macOS)

From a checkout you own outside shared folders (not under `/tmp`), in a native
arm64 Python 3.12 environment installed with `requirements-alpha.txt`:

```bash
venv-alpha/bin/python -m pip check
venv-alpha/bin/python scripts/test_alpha.py
SOTTO_DATA_DIR="$PWD/.alpha-state" SOTTO_HF_HOME="$PWD/.alpha-cache" \
  SOTTO_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  venv-alpha/bin/python tests/gesture_check.py
venv-alpha/bin/python -m compileall -q *.py scripts tests
python3 benchmark.py tests/fixtures/benchmark_gate/manifest.jsonl --dry-run
bash -n scripts/install-mac.sh scripts/rollout.sh
plutil -lint launchd/*.plist
venv-alpha/bin/python scripts/build_alpha_source.py /absolute/output/sotto-alpha-source.tar.gz
```

`test_alpha.py` points data, Hugging Face, XDG and Numba caches at a temporary
folder and sets every offline flag; it refuses a checkout below a world-writable
or foreign-owned folder, because Sotto's privacy guards reject such paths.
Inference in the suite is mocked or fixture-based. Five calibration integration
checks need separately provisioned `SOTTO_CALIBRATION_FIXTURES` and are skipped
otherwise. The sealed LaunchAgent integration needs Python 3.14 and is skipped
on 3.12. Windows-only tests are skipped on macOS (and vice versa) with reasons.
Windows steps are in [windows-alpha.md](windows-alpha.md).

## Results (2026-10-01)

macOS 27.2, M4 Max, CPython 3.12.14 unless noted.

| Check | Result | What it establishes |
| --- | --- | --- |
| Source archive built twice from the same commit | Byte-identical, 170 files (2026-10-09; 150 on 2026-10-01) | Reproducible allowlist; no Git data, environments, models or personal data |
| Fresh install from that archive | 21 s with a warm pip cache (46 s cold on 2026-09-29), `pip check` clean, installed set == lock (71 packages) | The pinned dependency set is complete |
| Full suite from the extracted archive | 827 tests OK, 156 skipped (Windows-only, calibration, Python 3.14) on 2026-10-09; 597 OK on 2026-10-01 | Unit, fixture and mocked-runtime behaviour |
| Fresh Python 3.14.7 environment on the production dependency policy (2026-09-29) | 12/12 sealed-release tests, none skipped | The sealed runtime path still verifies and isolates |
| Gesture acceptance | 8/8 | Hold / double-tap / tap state machine, not real key delivery |
| First launch download | Pinned Whisper revision fetched once; later launches make no Hub request | Measured against a local fake endpoint: 0 requests when cached |
| Foreground run | Ctrl-C and SIGTERM exit 0 in < 0.1 s; Restart re-executes with arguments and environment kept | Real Cocoa event loop, no microphone |
| Sotto.app build | Launcher compiled, signed ad hoc, verified; SIGTERM reaches the app; exit status passes through | Local app bundle mechanics, not notarization |
| Automatic language, one encoder pass | 0.63–0.73 s → 0.32–0.40 s (synthetic speech) | Same text on 48/56 clips; the rest equal or better |
| Fast speed (experimental) | 0.30 s → 0.13 s median; mean error 0.041 → 0.066 (synthetic, 13 languages) | English unaffected on these clips |
| Parakeet engine | 50–190 ms per sentence; en/es/de/fr correct (numbers as digits); silence and noise → empty; an unsupported language → a Latin-letter guess, not empty | Array input, pinned snapshot, chunked long input |
| Nemotron engine (2026-09-30, real model) | 27–29 ms from end of audio to final text while streaming; 3 English sentences, 1 word wrong; silence and noise → empty | Streaming decode on pinned local libraries |
| Silence handling with the pinned Silero VAD | "Thank you."/"you" on silence or noise held back; speech passes | Output-side verdict only; no input gate |
| Out-of-set language guard | In-set languages unchanged (two-language set); es/fr/de/ja written as spoken, not translated | Detection probabilities: in-set speech ≥ 0.989, other ≤ 0.112 |
| Nemotron install | Pinned hashes enforced; tampered model, extra library and symlink refused | Checksum gate, synthetic decode only |
| Windows (i9-11900K, RTX 4070 Ti SUPER) | 527 tests OK, 78 skipped from a fresh clone in the CPU environment (2026-10-09); 333 OK from a fresh archive in CPU and CUDA environments (2026-10-01); tray start/Quit exit 0 | See windows-alpha.md; no desktop session was used |

## Not yet established

- Real microphone → application dictation for this exact build on more than the
  maintainer's Mac, and on Windows at all (a person pressing the key, speaking
  and checking the text in Notes, TextEdit, Notepad, Word, browsers).
- Accuracy on real voices, accents, noisy rooms or whispering; every number
  above uses synthetic speech from system voices.
- macOS 14–26, Apple Silicon Macs other than one M4 Max, Windows machines
  other than one desktop, and ARM64 Windows.
- Notarized distribution (needs a Developer ID) and automatic updates.

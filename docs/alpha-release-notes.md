# First public alpha — release candidate

Status: prepared for review; not published. Source installation only. The
repository remains private until the owner approves visibility and historical
privacy exposure. No packaged app, signing, notarization, installer or login
service is included.

## Supported setup

Apple Silicon / native arm64 Python 3.12 / macOS 14 or later (the oldest
release the pinned dependencies install on). Verified candidate host: M4 Max,
macOS 27.2, CPython 3.12.14. macOS 14–26 and other chips are untested.
Initial engine: local Whisper large-v3-turbo; Automatic chooses among the
languages you tick (English + Hebrew by default, 100 available), with an
experimental Fast speed. Optional engines: Parakeet v3 (fastest, 25 European
languages, no Hebrew) and Nemotron en-0.6b (English-only streaming; final text
is pasted after release). Adaptive routing and background learning workers
remain off.

## Install from the approved source

After this candidate reaches the approved public `master`:

```bash
git clone https://github.com/zohartito/sotto.git
cd sotto
python3.12 -m venv venv-alpha
venv-alpha/bin/python -m pip install --upgrade pip
venv-alpha/bin/python -m pip install -r requirements-alpha.txt
venv-alpha/bin/python -m pip check
export SOTTO_DATA_DIR="$HOME/Library/Application Support/sotto-alpha"
export SOTTO_HF_HOME="$SOTTO_DATA_DIR/huggingface"
venv-alpha/bin/python sotto.py setup
venv-alpha/bin/python sotto.py doctor
venv-alpha/bin/python sotto.py run --idle-release 0
```

For the source archive, verify its separately supplied SHA-256 with
`shasum -a 256 sotto-alpha-source.tar.gz`, extract with
`tar -xzf sotto-alpha-source.tar.gz`, `cd sotto-alpha`, and run the commands
above starting at `python3.12 -m venv`. The prepared archive has no Git history,
credentials, model weights or personal data. Dependencies/models download
separately; models keep their own licensing. The alpha dependency pins are
separate from the strict production sealed-runtime policy.

Enable your terminal/Python in System Settings → Privacy & Security →
Microphone and Accessibility, then repeat `doctor`. Initial Whisper download
must finish before the listening banner appears. The menu's Restart and engine
switch work in a foreground run without an existing LaunchAgent. Quit or
Ctrl-C stops it. Keep the alpha environment variables set for every launch.

## Known limitations

- This is an early terminal-run alpha. Recognition mistakes and hallucinated
  output remain possible; guards are conservative and can withhold text.
- No claims about speed, accuracy or superiority to other dictation products.
- No verified end-to-end human microphone → Notes/TextEdit capture for this
  exact candidate yet. The earlier public demo is not this candidate's receipt.
- Secure fields and some applications can refuse simulated paste. Use History
  → Copy and test your target app before relying on it.
- Clipboard managers, target apps, Desktop exports and backups can retain
  separate copies of text/audio. See the README's storage/removal instructions.
- Whisper can invent short phrases ("Thank you.") for silence. `sotto.py setup`
  installs the pinned Silero VAD so those are held back (kept in History);
  without it they may be pasted. No input-side speech gate is ever applied.
- Bluetooth device changes can need a fresh process; lost beginnings and quiet
  whispered speech need human acceptance testing on more than one Mac.
- The first model download requires network access and fetches a pinned
  Hugging Face revision. Once it is cached, startup and inference use it
  without contacting Hugging Face; offline flags forbid any download.
- Keep the alpha data folder (and a checkout you run tests from) out of
  `/tmp`, `/Users/Shared` and other world-writable or symlinked locations.
  Privacy guards refuse them, and History → Clear then reports an error.
- If a sealed LaunchAgent cannot be confirmed as this process's supervisor,
  Restart refuses rather than re-executing outside the verified bootstrap.
- A harmless Python multiprocessing semaphore cleanup warning was observed on
  foreground shutdown/re-exec; verified shutdown returned zero.
- Explicit `--profile` or `--model` locks engine selection for that launch.
- Automatic login/sealed deployment is a separate maintainer workflow, not
  supported alpha onboarding. Do not install raw LaunchAgent templates.

## Windows source alpha

Windows 10/11 x64 with Python 3.13 runs a separate console alpha,
`sotto_win.py`: right Ctrl push-to-talk, pinned faster-whisper models, CUDA
float16 with automatic CPU int8 fallback, typed (not pasted) output and
command-line History. No menu or overlay. Nemotron, adaptive learning and
sealed deployment stay macOS-only. See [windows-alpha.md](windows-alpha.md).

[First-run instructions and troubleshooting](../README.md) ·
[Validation evidence](alpha-validation.md) ·
[Advanced safeguards](advanced-development.md)

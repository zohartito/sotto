# Sotto — source alpha

Local push-to-talk dictation for Apple Silicon Macs (and, separately, Windows).
Hold a key, speak, release; final text lands at the cursor. Whisper is the
default and understands 100 languages; Automatic chooses among the languages
you pick. Optional English-only Nemotron streams while you speak. Corrections
teach Sotto your spellings, and a Settings window holds every choice.

This alpha installs from source: one script builds a local **Sotto.app** (with
its own permissions, launch at login and updates), or you can run it from a
terminal. It is not notarized. Audio and inference stay on your Mac;
installing dependencies and downloading model files require internet access.
No speech-upload service or telemetry is configured. [MIT license](LICENSE); models and dependencies have
their own licenses.

## Requirements

- Apple Silicon Mac, running native **arm64** Python (not Rosetta). Intel Macs
  and Linux are unsupported; Windows has a separate source alpha
  ([Windows instructions](docs/windows-alpha.md)).
- **macOS 14 (Sonoma) or later**, **Python 3.12**, Git, and a local graphical
  login. macOS 14 is the oldest release the pinned dependencies install on.
  This candidate was only checked on macOS 27.2 / M4 Max / Python 3.12.14;
  macOS 14–26 and other Apple Silicon chips are untested, so please report
  results.
- A microphone, permission to use it, and Accessibility permission for the
  terminal/Python running Sotto.
- Allow several GB of memory and at least 8 GB of free disk for the environment,
  model and download caches. Latency and accuracy vary; this alpha makes no
  benchmark or comparative accuracy claims.

If needed, install native Python 3.12 with Homebrew:

```bash
brew install python@3.12
python3.12 --version
python3.12 -c 'import platform; print(platform.machine())'  # must say arm64
```

See [Python for macOS](https://www.python.org/downloads/macos/) and
[MLX installation](https://ml-explore.github.io/mlx/build/html/install.html).

## Install as an app (recommended)

From the clone (or the extracted source archive):

```bash
scripts/install-mac.sh
open -a Sotto
```

The script checks your Mac (Apple Silicon, macOS 14+, Python 3.12, Apple's
command line tools), installs the pinned dependencies into `venv-alpha`,
downloads the speech model once, and builds `Sotto.app` in `/Applications` (or `~/Applications` when you cannot write there). The app
is built on your Mac, so macOS does not block it, and it asks for the
Microphone and Accessibility as **Sotto**. Sotto lives in the menu bar; choose
**Start Sotto when I log in** under **Settings…**. Its log (counts and timings,
never your words) is `logs/sotto.log` in the data folder.

- Update: `scripts/install-mac.sh --update` (pulls a git clone, reinstalls
  dependencies; your permissions and settings stay).
- Remove: `scripts/install-mac.sh --uninstall` (removes the app and its login
  item; your data folder stays until you delete it).

Anything that can change this source folder runs with Sotto's permissions, as
with any source install; keep it in a folder only you control.

## Install and first launch from a terminal

Once this candidate is approved and available on `master`:

```bash
git clone https://github.com/zohartito/sotto.git
cd sotto
python3.12 -m venv venv-alpha
venv-alpha/bin/python -m pip install --upgrade pip
venv-alpha/bin/python -m pip install -r requirements-alpha.txt
venv-alpha/bin/python -m pip check

# Keep alpha preferences, recordings and models separate from any installed Sotto.
export SOTTO_DATA_DIR="$HOME/Library/Application Support/sotto-alpha"
export SOTTO_HF_HOME="$SOTTO_DATA_DIR/huggingface"

venv-alpha/bin/python sotto.py setup   # downloads the pinned models once
venv-alpha/bin/python sotto.py doctor
venv-alpha/bin/python sotto.py run --idle-release 0
```

While the repo is private, a clone requires authorized GitHub access. Reviewers
can use `git clone --branch codex/public-alpha ...` after that branch is shared,
or use the prepared source archive. Nothing in these commands installs a
LaunchAgent or enables adaptive routing/background learning. Re-export the two
variables in each new terminal before launching. To use a different folder,
set absolute paths for both variables consistently.

`doctor` requests microphone permission if undecided, checks Accessibility and
probes the input format. It does **not** record or verify text delivery. In
**System Settings → Privacy & Security**, enable your terminal (Terminal,
iTerm, etc.) under **Microphone** and **Accessibility**. The entry may appear as
**Python** depending on how you launched it. If macOS asks for Input Monitoring
for the event tap, allow the same terminal/Python. Quit/reopen the terminal and
rerun `doctor` after changing permissions. Do not grant a different Python
installation by mistake.

On first launch, Sotto downloads a pinned revision of the default
`mlx-community/whisper-large-v3-turbo` model (about 1.6 GB) from Hugging Face
into the alpha cache, then warms it before printing **listening on
right-option**. This can take time; it is not ready while downloading/warming.
A failed download can be retried with the same command. Once that exact
revision is cached, later launches use it without contacting Hugging Face. To
forbid network access entirely (for example, to confirm offline use):

```bash
SOTTO_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  venv-alpha/bin/python sotto.py run --idle-release 0
```

Click a plain text field in Notes or TextEdit, hold **right Option** (or the key
you chose in Settings) for at least
0.35 seconds, speak a short sentence, then release. Wait for final text before
changing fields. Compare it with what you said. Sotto uses the clipboard to
paste and attempts to restore its previous contents; clipboard managers and
apps may retain what was pasted. In **Settings** you can switch to **Type** (the
clipboard is never touched) and choose the spacing: **Smart** (default) adds a
space before the text only where it would touch the previous word, in apps
that expose the cursor; elsewhere it adds a space after, like before. Password
fields are never typed or pasted into.

## Gestures and controls

| Gesture/control | Effect |
| --- | --- |
| Hold trigger, then release | Push-to-talk; transcribe/paste after release |
| Double-tap trigger | Hands-free; next tap stops (10-minute maximum) |
| Lone short tap | Ignored |
| Control–Option–D | Alternate hotkey: hold/release or tap to start/stop |
| Menu → Start / Finish dictation | Hands-free controls without the key |
| Menu → Settings… (⌘,) | Key, languages, engine, how text is inserted, dictionary, data folder |
| Menu → Language | Automatic (among your languages) or one language; saved locally |
| Menu → Speech engine | Optional Nemotron or Whisper; restart while idle |
| Menu → Dictionary | Your personal spellings (see below) |
| Menu → Restart sotto | Restart this foreground process, retaining args/settings |
| Menu → Quit, or Ctrl-C in terminal | Stop this foreground run and release microphone |

The dictation key starts as **right Option**; choose left or right Option,
Command, Control or Shift, or Fn, in **Settings**. The change applies to the
next press. Holding the key while pressing another key (a shortcut such as
⌘-Tab) never starts dictation. Never let two running dictation tools share a
key: if another tool (or another Sotto) already uses right Option or
Control–Option–D, pick a different key in Settings. `--trigger` and `--hotkey`
override Settings for a single run (`--hotkey ''` disables the alternate hotkey).
`--no-overlay` hides both overlay and menu bar; use Ctrl-C to stop that run.
With `--idle-release 0`, the mic starts for capture and stops on release; the
speech model stays in memory. Hands-free intentionally keeps the mic active
until stopped. **Restart** discards an unfinished capture; finish dictation
before restarting or switching engines.

Recent items in the menu offer Copy, Retry, Save audio to Desktop, Delete,
Correct Transcript and Clear history. Copy lets you recover text when an app
refuses paste. A correction can explicitly enroll paired audio/reference in a
local learning corpus; this does not start a worker or enable adaptive routing.

## Dictionary: Sotto learns your spellings

When you fix a transcript with **History → Correct Transcript**, Sotto offers
the words you changed as rules ("open ai" → "OpenAI", "john smith" →
"Jon Smyth"). Checked rules go into your personal dictionary and apply to every
later dictation, and nothing is added without your confirmation. Open
**Dictionary** in the menu to edit the plain-text file (`dictionary.txt` in
your data folder), one rule per line:

```text
whisper flow => Wispr Flow
```

Matching ignores case and only replaces whole words or phrases; edits apply to
the next dictation. History keeps what the model originally heard next to the
rewritten text. `venv-alpha/bin/python sotto.py dictionary` lists the rules.

To help Whisper hear names and terms in the first place, list them one per line
in `glossary.txt` in the same data folder (up to 100 terms; lines starting with
`#` are ignored) and restart Sotto. `config/sotto-glossary.txt` is an example.
The glossary is used with Whisper on captures up to 30 seconds; a terminal run
can pass its own file with `--glossary FILE`.

## Optional Parakeet (fastest; 25 European languages)

**Speech engine → Parakeet** uses NVIDIA's Parakeet TDT 0.6B v3 through MLX:
about 0.05 s per sentence on an M4 Max once warm, and it writes nothing for
silence. It detects its own language among Bulgarian, Croatian, Czech, Danish,
Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian,
Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Russian, Slovak,
Slovenian, Spanish, Swedish and Ukrainian. It does **not** understand other
languages (it pastes a Latin-letter guess for them): use Whisper for those.
Download it once from **Settings → Download Parakeet (2.5 GB)** (or
`venv-alpha/bin/python sotto.py setup --profile parakeet`), then choose it
while idle; Sotto restarts on the new engine. The model is pinned to one
commit, loaded from the local copy, and never needs ffmpeg. Glossary prompts
do not apply; your dictionary still does. If it cannot load, Sotto returns to
Whisper.

## Optional Nemotron (English only)

Keep the same alpha data/cache variables exported, then explicitly provision:

```bash
venv-alpha/bin/python scripts/setup_nemotron.py
```

This downloads the pinned NeMo-Speech.cpp 0.1.0 Metal libraries and Q8 model,
checks SHA-256 values and retains upstream notices under
`$SOTTO_DATA_DIR/nemotron/0.1.0`. Choose **Speech engine → Nemotron (English
streaming)** while idle. Audio is decoded during capture; only finalized,
validated text is pasted after release. Partial text is not pasted. Other
languages, automatic language detection and Whisper glossary prompts are unavailable.
Returning to Whisper restores your saved language choice. Native load failure
falls back to Whisper; it does not turn on adaptive routing.

Explicit profiles override saved engine choice and disable engine switching
in that run:

```bash
venv-alpha/bin/python sotto.py run --profile nemotron-en
venv-alpha/bin/python sotto.py run --profile auto
```

Nemotron is experimental and is not established as more accurate than Whisper
for your voice. [NVIDIA model](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
and [native runtime](https://github.com/NVIDIA/NeMo-Speech.cpp).

## Local data and removal

With the commands above, all Sotto-managed alpha state lives under
`~/Library/Application Support/sotto-alpha`:

| Location relative to that folder | Contents |
| --- | --- |
| `history.jsonl`, `audio/`, `audio-raw/` | Transcripts/metadata, and the processed and raw-capture WAV audio of each item; ordinary history capped at 200 |
| `language-mode`, `engine-mode`, `settings.json` | Saved language, engine and Settings choices |
| `dictionary.txt` | Your personal spelling rules |
| `models/silero_vad.onnx` | Voice-detection model (Silero v6.2, MIT) used to drop silence phrases |
| `learning/`, `adaptive-learning/` | Consented corrections and experimental state/receipts |
| `huggingface/` | Whisper model cache (the exported `SOTTO_HF_HOME`) |
| `nemotron/` | Optional native runtime, model and notices |

Recovery-pinned entries may outlive normal history retention. History can
retain audio and suspect output even when no text was pasted. Without the
alpha variables, legacy defaults are `~/Library/Application Support/sotto`
for data and `~/Library/Application Support/Sotto/huggingface` for Whisper
cache; capitalization can matter on case-sensitive volumes. Keep the data
folder somewhere only you can write, such as the default above. Sotto's
privacy guards refuse storage below `/tmp`, `/Users/Shared`, other
world-writable folders or a symlinked path, and History → Clear then reports an
error. The same applies to the checkout when you run the test suite.

**Delete** removes a history item and revokes linked learning material.
**Clear history** clears retained history and revokes the active local corpus.
Models and preferences stay. See [storage and revocation details](docs/speech-learning.md).
For a complete alpha uninstall, Quit/Ctrl-C first, then move **only your alpha
folder** and this clone (including `venv-alpha`) to Trash in Finder. This
removes alpha recordings/transcripts, learning data, settings and model caches.
Never remove an existing installed Sotto's folder/LaunchAgent as part of this
alpha uninstall. To remove only alpha models, quit and move its `huggingface`
and optional `nemotron` folders to Trash; they need downloading again later.

Saved audio on Desktop, deliberately exported corpora, clipboard-manager
history, text pasted into other apps, pip's package cache, backups and filesystem
snapshots are separate copies; remove them separately if needed. Foreground
logs go to the terminal, not a Sotto log file, unless you redirect them yourself.
Deletion cannot guarantee erasure from backups or SSD snapshots.

## Troubleshooting

- **No Sotto icon in the menu bar:** on a full menu bar (especially beside the
  notch) macOS hides new icons behind **«**. Click it to find Sotto's small `◦`,
  then hold ⌘ and drag the icon to a visible spot. Dictation works either way.
- **Recording won't stop:** a double-tap starts hands-free mode; one tap of the
  same key stops it (it also stops by itself after 10 minutes).
- **Permission error / no hotkey:** rerun `doctor` in the same terminal and
  check that terminal/Python's Accessibility and microphone grants. After a
  Python/venv replacement the old grant may no longer apply. For Fn conflicts,
  use right Option or another supported trigger. Keep one Sotto per trigger.
- **No input / all-zero audio:** select a working input in System Settings →
  Sound. Check headset/Bluetooth routing; try the built-in mic. Use Restart
  from the menu, or Quit and rerun. A silent/dead route is retained as suspect
  history, not pasted as ordinary dictation.
- **No text in target app:** click a plain text field and release the key.
  Secure/password fields may refuse paste. Look in the history menu and Copy;
  `doctor` passing does not prove paste works in every app.
- **Download/TLS error or "Whisper model unavailable":** check internet access
  and disk space. Python.org installs may need their `Install
  Certificates.command`. "Not downloaded yet" means an offline flag is set and
  the model is not cached yet: run once without the offline flags. Do not
  delete a shared cache.
- **Wrong language:** add your languages under **Settings… → Languages** so
  Automatic can pick them, or force one in the Language menu for short phrases.
  Use Whisper for languages Parakeet and Nemotron do not cover. Switching
  languages does not load a second model.
- **"Thank you." appears when you said nothing:** Whisper invents short stock
  phrases for silence. `sotto.py setup` (run by the installer) adds the small
  Silero voice-detection model, which keeps those out of the text field (they
  stay in History). It never blocks real dictation, including whispers.
- **Restart / engine switch fails:** use the reported error, Quit and rerun.
  A source run does not need a LaunchAgent. Explicit `--profile`/`--model`
  locks engine selection; omit them for menu switching.

When reporting an issue, include OS/chip/Python version, command, permission
results and a short reproduction. Redact usernames/paths. Do not attach private
recordings, transcripts, caches or a full data folder.

## Windows

A separate console source alpha runs on Windows 10/11 x64 with Python 3.13:
`sotto_win.py`, hold right Ctrl, Whisper on an NVIDIA GPU or the CPU. Install,
data locations and limitations: [docs/windows-alpha.md](docs/windows-alpha.md).

[Alpha release notes](docs/alpha-release-notes.md) ·
[Maintainer validation](docs/alpha-validation.md) ·
[Advanced development and adaptive safeguards](docs/advanced-development.md)

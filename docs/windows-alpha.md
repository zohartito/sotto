# Sotto — Windows source alpha

Local push-to-talk dictation for Windows 10/11. Hold right Ctrl, speak,
release; the text lands at the cursor. Whisper only (faster-whisper /
CTranslate2), on an NVIDIA GPU when one works and on the CPU otherwise.

Sotto runs as a **tray app** (Start Menu shortcut, no console window) with a
menu for History, Language, Speed, Progress (including all-time words and time
saved), the personal Dictionary, Settings (with English voice commands and
filler cleanup) and
**Check for updates…** (it contacts GitHub only when you click it), or as
a plain **console program** for tests and headless use. Audio and inference
stay on your PC; installing dependencies and the first model download need
internet access. No speech-upload service or telemetry is configured.
Nemotron, adaptive/"silver" learning, sealed releases and LaunchAgents are
macOS-only and never run here. History actions create no silver ledger; one
left in the data folder by an earlier alpha is still consulted, so a deleted
recording cannot leave silver records behind. [MIT license](../LICENSE); models and
dependencies have their own licenses.

## Requirements

- Windows 10 or 11, **x64**, with a local desktop session (not a service or
  remote session without a microphone).
- **Python 3.13, 64-bit**, from [python.org](https://www.python.org/downloads/windows/)
  (checked with CPython 3.13.14). Keep **py launcher** and **tcl/tk and IDLE**
  ticked in its installer: the Settings and Correct windows use tkinter. Other
  Python versions are untested against the lock.
- A microphone, and **Settings → Privacy & security → Microphone → Let desktop
  apps access your microphone** turned on.
- Optional NVIDIA GPU: a driver with CUDA 12 support (release 528 or newer).
  No CUDA Toolkit install is needed; the pip CUDA extras carry cuBLAS and the
  ctranslate2 wheel carries cuDNN. Without a working GPU Sotto uses the CPU.
- About 6 GB free disk for the environment, caches and the default model
  (1.6 GB; the Fast speed adds 0.5 GB). Several GB of
  RAM (CPU) or about 2 GB of VRAM (GPU) for the default model.

Checked on Windows 11 x64 (build 26200), i9-11900K, RTX 4070 Ti SUPER, NVIDIA
driver 610.88. Other hardware has not been validated. No accuracy claims
beyond the synthetic-speech benchmark below.

## Install (recommended)

In **PowerShell**, from a clone of the repository or the extracted source
archive (see below):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install-windows.ps1 -WhatIf   # show the plan only
powershell -ExecutionPolicy Bypass -File scripts\install-windows.ps1
```

The installer checks for Windows x64 and Python 3.13 64-bit (it never
installs Python for you; it prints what to install instead), creates
`venv-alpha` in that folder, installs the pinned CUDA set when `nvidia-smi`
sees a GPU and the CPU set otherwise (`-Cpu` / `-Cuda` override), creates the
data folder `%LOCALAPPDATA%\sotto-alpha` (`-DataDir` overrides), downloads the
pinned speech model (about 1.6 GB, into `huggingface\` there) and the pinned
Silero VAD v6.2 voice detection (2.3 MB, SHA-256 checked) with
`sotto_win.py setup` (`-SkipSetup` leaves the model to the first launch), and
adds a Start Menu shortcut **Sotto** that starts the tray app without a
console window, with that data folder applied. No admin rights, services or
scheduled tasks. Running it again is safe and updates the environment; it
keeps the CPU or CUDA set recorded at install (`-Cpu` / `-Cuda` switch it). It
stops without changing anything while Sotto runs from that environment (quit
Sotto first), and when the Start Menu already has a **Sotto** shortcut that
starts another copy (remove that copy with its own `-Uninstall`, or pass
`-ShortcutDir`); a shortcut to a copy that no longer exists is replaced.

Then start **Sotto** from the Start Menu. A tray icon appears (grey while the
model warms up); a "Ready — hold Right Ctrl to dictate" notification says when
you can dictate. Launches use the cached model with no network access.

For the source archive, check its separately supplied SHA-256 with
`Get-FileHash sotto-alpha-source.tar.gz -Algorithm SHA256`, extract it with
`tar -xzf sotto-alpha-source.tar.gz` (built into Windows 10/11), `cd sotto-alpha`,
and run the installer. The archive has no Git history, credentials, model
weights or personal data.

## Manual install and console use

```powershell
py -3.13 -m venv venv-alpha
venv-alpha\Scripts\python -m pip install pip==26.2.1
# CPU only:
venv-alpha\Scripts\python -m pip install -r requirements-alpha-windows.txt
# ...or, with an NVIDIA GPU (includes the CPU set plus cuBLAS):
venv-alpha\Scripts\python -m pip install -r requirements-alpha-windows-cuda.txt
venv-alpha\Scripts\python -m pip check

# Keep alpha history, settings and models in their own folder.
$env:SOTTO_DATA_DIR = "$env:LOCALAPPDATA\sotto-alpha"
$env:SOTTO_HF_HOME  = "$env:SOTTO_DATA_DIR\huggingface"

venv-alpha\Scripts\python sotto_win.py setup      # model + voice detection, once
venv-alpha\Scripts\python sotto_win.py doctor
venv-alpha\Scripts\python sotto_win.py            # console; Ctrl-C stops it
venv-alpha\Scripts\python sotto_win.py --tray     # tray app from a console
venv-alpha\Scripts\pythonw win_launch.py --data-dir "$env:LOCALAPPDATA\sotto-alpha"   # tray, no console
```

In **cmd.exe** use `set SOTTO_DATA_DIR=%LOCALAPPDATA%\sotto-alpha` and
`set SOTTO_HF_HOME=%LOCALAPPDATA%\sotto-alpha\huggingface`. Set both variables
again in every new console, and use the same values for the `history`
commands below. `win_launch.py --data-dir DIR` sets them for you (models go to
`DIR\huggingface` unless `--hf-home` says otherwise).

`setup` downloads the pinned model for your profile and speed and the pinned
VAD; with the offline variables set it downloads nothing (and says what is
missing). A run without `setup` downloads the model on its first start but
never the VAD. `doctor` shows the data and model folders, the CUDA devices
CTranslate2 can see, whether the pinned model is cached, the default
microphone, whether the pinned VAD is installed, SendInput and tkinter. It
downloads nothing and does not open the microphone. The console run is ready when it prints **listening on
right-ctrl**. To forbid any network access:

```powershell
$env:SOTTO_OFFLINE = "1"; $env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
```

Offline with no cached model, Sotto stops with a one-line explanation (in the
tray app, a message box) instead of downloading. Only one Sotto runs per
Windows sign-in (two would insert every dictation twice); a second start says
it is already running.

Open **Notepad**, click in the text area, hold **right Ctrl** for at least 0.35
seconds, say a short sentence, release, and wait for the text.

## Dictating

| Gesture | Effect |
| --- | --- |
| Hold the trigger, then release | Push-to-talk; transcribe and insert after release |
| Double-tap the trigger | Hands-free; the next tap stops (10-minute maximum) |
| Lone short tap | Ignored |
| Another key while holding the trigger (e.g. Ctrl-C) | A shortcut: the capture is discarded, nothing inserted |
| Tray → Start dictation / Finish dictation | Hands-free without the key; finishing from the tray copies the text (paste it with Ctrl+V) |

The microphone opens when you press the trigger and closes when you release
it (hands-free: when you stop). It is never held while idle. The tray icon
turns red while recording and amber while transcribing. Very first syllables
can be clipped; start speaking after a beat.

Only final text that passes the output checks is inserted: an all-zero (dead)
microphone never reaches the model, and empty, no-speech or looping output is
kept in History instead (a clean prefix before a repetition loop is kept).
History is written before anything is inserted. Insertions happen one at a
time, in order, and wait while any Shift, Ctrl, Alt or Windows key is held, so
the text cannot combine with a modifier; a text not inserted within 2 minutes of
being ready (counted for each text, however many wait in line) is kept in
History instead.

If the speech model itself fails on a dictation (for example CUDA runs out of
memory), the recording is kept in History as **[transcription failed]** and a
notification says so; **History → Retry** transcribes that audio again.

## Tray menu

Right-click the tray icon (left-click opens Settings):

- **Start dictation (hands-free)** / **Finish dictation**
- **History** — the 10 newest dictations, each with **Copy**, **Retry**,
  **Correct…** and **Delete**, plus **Clear history…**. Retry transcribes the
  saved audio again and copies the new text to the clipboard (it never pastes).
- **Language** — Automatic (among your languages), each of your languages,
  and **Other languages** (common ones); a specific language is always forced.
- **Speed** — Accurate or Fast; switching restarts Sotto once the current
  dictation is done.
- **Progress** — see below. **Dictionary** — opens your dictionary in Notepad.
- **Settings…**, **Restart** (finishes a dictation in flight first, however
  long the model takes, and ignores new presses meanwhile), **Quit**.
- **Check for updates…** — compares this git copy with GitHub. **Update**
  waits for Sotto to quit, installs the new version's pinned packages (the CPU
  or CUDA set recorded at install) before it switches the source, and puts the
  previous packages back if that fails, so the source never runs on the wrong
  packages. Sotto starts again either way, and the tray then says whether the
  update worked; the details are in `update.log` in the data folder. The
  one-line `get.ps1` refuses to update a copy while Sotto runs from it.

Opening the tray menu takes the focus away from the app you were dictating
into, so a dictation finished with **Finish dictation** is copied to the
clipboard (a notification says so) instead of being inserted.

## Settings

| Setting | Choices | Applies |
| --- | --- | --- |
| Trigger key | Right Ctrl (default), Left Ctrl, Left Alt, Left Shift | Immediately (a `--trigger` flag wins for that run) |
| Languages for Automatic | Any Whisper languages, 1 to 12 (default English) | Next dictation |
| Insert text by | Paste (clipboard put back) or Type (clipboard untouched) | Next dictation |
| Spacing | Smart, Add a space, No space | Next dictation |
| Speed | Accurate or Fast | After restart |
| Launch Sotto when I sign in | On/off | Immediately |

Settings also has **Open dictionary**, **Show data folder** and the Progress
numbers. Settings live in `settings.json` in the data folder.

Trigger keys left out on purpose: Fn (handled by the keyboard itself,
invisible to Windows); right Alt (it is AltGr on many non-US layouts, and
arrives with a synthetic left Ctrl that would cancel every
capture); right Shift (holding it 8 seconds opens the Filter Keys prompt); the
Windows key (Start menu, Win+H voice typing, PowerToys hold guides). With
**Left Alt**, Sotto sends an unassigned key on each press so releasing Alt
does not open the app's menu bar.

## Personal dictionary and corrections

The dictionary is a plain text file, `dictionary.txt` in the data folder, with
one rule per line: `what Sotto hears => what you want written`, for example
`whisper flow => Wispr Flow`. Matching ignores case and replaces whole words or
phrases. Rules apply to the final text right before it is inserted (and to a
Retry), never to text kept back as suspect. When a rule changes a dictation,
History keeps the original recognition next to the rules used; the log only
counts replacements.

**History → Correct…** opens the text for editing. Saving stores the
correction (same shared History logic as the Mac). If the correction replaced
words, Sotto asks **"Always write these your way?"** with a checkbox per
suggested rule; only the rules you confirm are added.

## Languages

**Automatic** chooses among your languages (Settings → Languages, English by
default). With only one language chosen it skips detection and transcribes in
that language, so add every language you speak. Otherwise Sotto detects the
language once, takes the most likely of
your languages, and transcribes with that language fixed. Speech that is
clearly another language (your languages score below 25% and another language
at least 50%) is written in the language spoken, because forcing it into one
of yours would make Whisper translate it (the Mac's rule). For a dictation
of up to 30 seconds the same encoder pass serves both the detection and the
transcription (identical text, about half the CPU time of detecting
separately). History records the pick as `detected_language`. Choosing a
specific language in the tray always forces it; `--language CODE` does the
same for one run.

## Speed

**Accurate** uses `deepdml/faster-whisper-large-v3-turbo-ct2`. **Fast** uses
`Systran/faster-whisper-small` (multilingual, MIT), pinned to commit
`536b0662742c`, downloaded once on first use (about 0.5 GB) like the other
models. Fast matters on the CPU; on a GPU both are equally quick. `--model`
ignores the speed setting.

Benchmark, 2026-09-29, this check machine, synthetic speech: 10 dictation-style
sentences spoken by the 5 installed en-US Windows voices (System.Speech),
clean and with white noise at 10 dB SNR; seconds per clip (clips average
4.5 s); word error rate (WER) after lowercasing and removing punctuation.
CPU: 20 clips, int8, i9-11900K. CUDA: 50 clips, float16, RTX 4070 Ti SUPER.

Shipped Automatic path (two-language set, one encoder pass):

| Setting (model) | CPU s/clip clean / noisy | CPU WER clean / noisy | CUDA s/clip | CUDA WER clean / noisy |
| --- | --- | --- | --- | --- |
| Accurate (large-v3-turbo) | 4.92 / 6.25 | 0.0% / 0.0% | 0.20 | 0.0% / 0.3% |
| Fast (small) | 1.46 / 1.47 | 0.0% / 1.2% | 0.21 | 0.0% / 0.9% |

Candidate selection (detection as a separate pass, noisy clips; CPU s/clip,
WER CPU / CUDA): base 0.84 s, 0.8% / 1.7%; small 2.9–3.4 s, 1.2% / 0.9%;
medium 7.1–7.9 s, 0.4% / 0.2%; large-v3-turbo 12.3–14.4 s, 0.0% / 0.3%. With
a fixed language the CPU times roughly halve (turbo 5.5–6.2 s, small
1.6–1.7 s). Medium was barely faster than Accurate on the CPU; base was
faster still but is the weakest multilingual model. Small is the smallest
multilingual port with a clear CPU gain and a low error rate here. The
distilled large-v3 ports are English-only and would break other languages.

Accuracy outside English was not measured per language on the check machine;
smaller Whisper models are known to be much weaker than large ones on many
languages. Use Accurate if Fast makes mistakes in yours.

## Insertion: paste or type

- **Paste** (default): Sotto offers the text on the clipboard, marked so that
  Windows Clipboard History, cloud clipboard sync and clipboard managers skip
  it, and sends Ctrl+V. The text is handed over only when the app actually
  reads it (delayed rendering); a moment later Sotto puts your previous
  clipboard back, unless you copied something new meanwhile (your copy wins).
  If the app never reads it (for example a terminal that ignores Ctrl+V), the
  clipboard comes back after 10 seconds. If the clipboard is busy, very large,
  holds content that cannot be restored exactly (for example Office drawings)
  or holds content its owner marked private (password managers that clear the
  clipboard themselves), Sotto types the text instead and leaves the clipboard
  alone. If another program holds the clipboard when Sotto puts yours back,
  it retries for a few seconds, then with the next paste and at Quit.
  Restored content is not added to Clipboard History again, and nothing Sotto
  puts on the clipboard (restores, History → Copy, Retry) is uploaded by
  cloud clipboard sync.
- **Type**: the text arrives as Unicode keystrokes (`SendInput`); the
  clipboard is never used.
- **Spacing**: *Add a space* appends one space so consecutive dictations do
  not run together; *No space* appends nothing. *Smart* behaves like *Add a
  space* on Windows: there is no reliable, app-independent way to read the
  character before the cursor, so Sotto never guesses a leading space.

## Launch at login

Settings → **Launch Sotto when I sign in** adds one per-user value, `Sotto`,
under `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`, which starts the
tray app with pythonw and the same data folder. Turning it off removes that
value; Task Manager → Startup apps can also disable it. No service or
scheduled task is used.

## Progress

Tray → **Progress** (and the Settings window) shows the same summary as the
Mac, computed from local History only: dictations in the last 7 days and
their median time from key release to text (with the previous 7 days for
comparison), how many needed a correction, dictionary rules and the words they
fixed this period, and corrections saved. Each History row stores
`latency.release_to_text_seconds` (key release until the text is ready to
insert, the Mac's measure), and the log prints it, e.g.
`→ 0.41s after release (speech model 0.22s) · 23 chars`. History keeps the
newest 200 dictations, so older activity drops out of these numbers.

## Device: GPU or CPU

| Setting | Behavior |
| --- | --- |
| default (`auto`) | Try CUDA float16; if the GPU, driver, cuBLAS or cuDNN fails, use CPU int8 |
| `--device cpu` or `$env:SOTTO_DEVICE = "cpu"` | CPU int8 only |
| `--device cuda` or `$env:SOTTO_DEVICE = "cuda"` | CUDA only; stop with an error instead of falling back |

`--device` wins over `SOTTO_DEVICE`. On the CPU, Accurate takes about 5
seconds per short clip and Fast about 1.5 (see Speed); on a GPU both take
about 0.2 seconds.

## Command-line options

- `--tray` runs the tray app; without it Sotto is a console program.
- `--trigger right-ctrl|left-ctrl|left-option|left-shift` for one run.
- `--language auto|en|fr|…` fixes or frees the decode language for one run.
- `--glossary FILE` adds a local UTF-8 word list as a decoding hint.

## History

History keeps the newest 200 dictations (text plus WAV audio). Besides the
tray menu, with the same `SOTTO_DATA_DIR` as the running app:

```powershell
venv-alpha\Scripts\python sotto_win.py history              # newest 20: id, time, length, text
venv-alpha\Scripts\python sotto_win.py history --limit 200
venv-alpha\Scripts\python sotto_win.py history-delete --id ENTRY_ID
venv-alpha\Scripts\python sotto_win.py history-clear --yes
```

Delete removes the entry, its audio and any linked local learning material.
Clear removes every entry and recording; models, settings and the dictionary
stay. History is also how you recover text that an application refused.

## Local data and removal

Everything Sotto writes lives in the data folder
(`%LOCALAPPDATA%\sotto-alpha` with the installer):

| Location in that folder | Contents |
| --- | --- |
| `history.jsonl`, `audio\`, `audio-raw\` | Transcripts/metadata and WAV audio (capped at 200) |
| `dictionary.txt`, `settings.json`, `language-mode` | Your rules and preferences |
| `learning\`, `adaptive-learning\` | Local learning and revocation bookkeeping (no worker runs on Windows) |
| `sotto.log` (+ `sotto.log.1`) | Timings, devices, character counts — no transcript text |
| `huggingface\` | Model cache (`SOTTO_HF_HOME`) |
| `models\silero_vad.onnx` | Pinned Silero VAD v6.2, installed by `setup` |

Without the variables (or the installer's shortcut), Sotto uses
`%APPDATA%\sotto` for data and `%APPDATA%\sotto\huggingface` for models.

To uninstall: quit Sotto, then

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install-windows.ps1 -Uninstall              # keeps your data
powershell -ExecutionPolicy Bypass -File scripts\install-windows.ps1 -Uninstall -RemoveData  # also deletes it
```

`-Uninstall` removes the Start Menu shortcut and the login entry when they
start this copy of Sotto, and `venv-alpha` if the installer created it.
`-RemoveData` deletes the data folder only if the installer created it, and
never `%APPDATA%\sotto`. Then delete the source folder. pip's download cache
(`%LOCALAPPDATA%\pip\cache`), text inserted into other apps, backups and File
History copies are separate.

## Known limitations

- Real microphone dictation has been checked by a person on one Windows 11
  desktop only; the automated checks use synthetic audio, simulated key
  events and a readiness/shutdown test.
- The tray menu and dialogs were exercised from a non-interactive session
  (menu callbacks invoked in a real process); a real right-click and the
  Settings/Correct windows on a desktop have not been verified by a person.
- Elevated (administrator) windows, UAC prompts and the sign-in screen ignore
  synthetic input, as can some games, remote-desktop clients and terminals.
  Do not run Sotto elevated to work around this.
- The trigger uses a low-level keyboard hook (pynput). Windows passes every key
  event through it; Sotto uses only the trigger key and whether another key
  joins a hold, and records no keystrokes. Some security software may warn
  about keyboard hooks.
- Paste restores the clipboard shortly after the foreground app reads the
  offered text. When another program reads it first (a clipboard tool that
  ignores the "skip me" markers), Sotto waits up to 10 seconds instead; an app
  busy for longer than that would paste your previous clipboard. A clipboard
  tool that ignores the markers can also keep its own copy of the dictation.
  Use Type for such setups.
- Tk windows do not reorder right-to-left text: in the Correct window it
  may display in visual order, although it is saved correctly.
- A console run ends immediately when its window is closed; an unfinished
  dictation is lost. Prefer Ctrl-C or the tray's Quit.
- A failed CUDA attempt adds a few seconds to startup before the CPU fallback.
- Without the VAD (a manual install that skipped `setup`, or an offline
  setup) transcription still runs, but long-capture trimming and the advisory
  no-speech score are unavailable, so holding the key without speaking can
  insert a Whisper stock phrase such as "Thank you." (seen with synthetic
  silence). There is never an input-side speech gate: the VAD only judges the
  output.
- Only Python 3.13 x64 has been checked against the lock; ARM64 Windows is
  untested.

## Troubleshooting

- **No tray icon:** check the hidden-icons area (^) on the taskbar; the icon is
  grey while the model loads. `sotto.log` in the data folder says what it is
  doing.
- **"Sotto is already running":** another Sotto (tray or console) is running
  in this Windows sign-in; quit it from its tray menu or its console.
- **`✗ microphone`:** pick a default input in **Settings → System → Sound**
  and check the microphone privacy switch for desktop apps. Sotto records from
  the Windows default input through PortAudio's default host API (MME), which
  converts to 16 kHz. `! mic open failed` in the log usually means another app
  holds the device exclusively.
- **`! cuda (float16) unavailable … falling back to CPU`:** expected on a
  CPU-only install. For the GPU, run the installer with `-Cuda` (or install
  `requirements-alpha-windows-cuda.txt`) and update the NVIDIA driver;
  `--device cuda` shows the full error without falling back.
- **Download error:** check network, proxy and disk space, then start again;
  the download resumes. Clear the offline variables only when you intend to
  download.
- **"not in … and offline mode is on":** run once without the offline variables,
  or point `SOTTO_HF_HOME` at the folder that already holds the model.
- **No text appears:** click into a normal text field first, release the key,
  check that the target is not elevated, and try Settings → Type. History
  shows what was recognized; `! not pasted (…)` in the log means the text was
  kept back on purpose.
- **Wrong language:** limit Settings → Languages to the ones you speak, or
  pick one language in the tray. **Slow on CPU:** Speed → Fast.
- **Settings or Correct does not open:** reinstall Python 3.13 with
  **tcl/tk and IDLE** (python.org installer → Modify).

## Candidate checks

Run 2026-09-29 on the check machine from the extracted source archive, with
fresh venvs and temporary data, model and Start Menu folders and a throwaway
registry key (no personal data, no real login entry):

| Check | Result |
| --- | --- |
| Installer from the archive, `-Cuda` (with `setup`) and `-Cpu -SkipSetup` | Both OK; `setup` fetched the pinned model and the VAD (hash matches v6.2) |
| Fresh CPU and CUDA venvs: `pip check`, `pip freeze` vs lock | Clean; freeze equals lock (32 / 34 packages) |
| `scripts\test_alpha.py` in each fresh venv (temporary state, offline flags) | 333 tests OK in each; 52 explicit skips (CUDA venv) / 53 (CPU venv), all macOS-only or CUDA-extras |
| `doctor` on the installed CUDA layout | Every check green (model, VAD, microphone, SendInput, tkinter) |
| Tray process, offline: listening → real Settings window clicked, saved, reloaded → tray Quit | Listening in about 6 s on CUDA; settings applied; exit 0, no thread or process left |
| Ctrl-C / Ctrl-Break, console and `--tray`, CUDA and forced CPU | Exit 0 every time; no leftover process |
| Tray Restart | First process exits 0; the relaunched pythonw copy starts listening, then quits cleanly |
| pythonw launcher (the Start Menu shortcut's command) | Listening in about 6 s with the given data and model folders |
| Paste with delayed rendering, private window station | The reading app got the text (non-Latin text included); clipboard restored; the session's own clipboard untouched |
| `-Uninstall`, `-Uninstall -RemoveData` | Shortcut, login entry and created venv removed; data kept unless asked |

These checks use synthetic audio, simulated key events and programmatic menu
clicks from a non-interactive session; they do not exercise a real
microphone, a physical key press, a real right-click or typing into another
application.

When reporting an issue, include Windows version, CPU/GPU, Python version, the
command, the `doctor` output and a short reproduction. Redact usernames/paths.
Do not attach recordings, transcripts, the dictionary, caches or the data
folder.

`scripts\sotto-win.bat` is a developer shortcut that creates `.\venv` and uses
the default `%APPDATA%\sotto` data folder; testers should use the installer.

[Alpha release notes](alpha-release-notes.md) · [macOS instructions](../README.md)

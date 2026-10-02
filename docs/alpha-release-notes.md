# Sotto 0.1.0-alpha

Local, offline push-to-talk dictation for macOS and Windows: hold a key, speak,
release, and the text lands at the cursor. Speech is transcribed on your
computer; there is no account, cloud service or telemetry.

## What is in it

- **Hold to talk, or hands-free.** Hold the key (right Option on a Mac, right
  Ctrl on Windows) and release to paste. Double-tap for hands-free; one tap
  stops it. A Ctrl-Option-D toggle works through remote-desktop apps (Mac).
- **Three engines on the Mac.** Whisper large-v3-turbo (100 languages),
  Parakeet v3 (fastest; 25 European languages) and Nemotron (English,
  streaming while you talk). Windows runs Whisper on an NVIDIA GPU or the CPU.
- **Your languages.** Automatic picks among the languages you choose (English
  by default). Speech that is clearly another language is written as spoken
  rather than translated.
- **Fast.** On an M4 Max with synthetic speech: about 0.3 s from release to
  text with Whisper, about 0.13 s at the Fast speed, 50–190 ms with Parakeet and
  about 30 ms with Nemotron.
- **Learns your spellings.** Correct a transcript in History and Sotto offers
  the change as a dictionary rule for every later dictation. A glossary helps
  Whisper hear names and terms. **Your progress** shows what it has learned and
  your all-time words and time saved versus typing.
- **Voice commands (English).** "New line", "new paragraph" and "scratch that"
  (undo the last dictation); um/uh fillers removed. Both can be switched off.
- **Live words.** With Nemotron, the pill shows your words as you speak.
- **History.** Copy, Retry, Correct and Delete recent dictations; audio stays on
  your computer.
- **Careful output.** Paste (clipboard put back) or type; smart spacing; the
  short phrases Whisper invents on silence ("Thank you.") and repetition loops
  are held back in History instead of pasted.
- **A real app.** Menu-bar app with a Settings window on the Mac (built as
  Sotto.app by the installer), a tray app on Windows, and start at login on both.

## Install

Mac — Apple Silicon, macOS 14 or later, Python 3.12 and Apple's command line
tools (`xcode-select --install`). One line in Terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/zohartito/sotto/main/scripts/get.sh | bash
```

or clone it yourself and run `scripts/install-mac.sh`.

The script installs the pinned dependencies, downloads the speech model once
(about 1.6 GB, pinned to an exact revision) and builds Sotto.app in
Applications. Open it, allow the Microphone and turn Sotto on under Privacy &
Security → Accessibility. Update later with **Check for Updates…** in Sotto's
menu (or `scripts/install-mac.sh --update`).

Windows — Windows 10/11 x64 with Git and 64-bit Python 3.13. One line in
PowerShell: `irm https://raw.githubusercontent.com/zohartito/sotto/main/scripts/get.ps1 | iex`
(or clone it and run `scripts\install-windows.ps1`), then start Sotto from the
Start menu. Details: [windows-alpha.md](windows-alpha.md).

The source archive attached to the release is built deterministically from the
tagged commit; check it with `shasum -a 256` against the published SHA-256.

## Known limitations

- Early alpha: recognition mistakes and invented text are still possible. The
  guards are conservative and may hold text back; it stays in History.
- Speed and accuracy figures come from synthetic speech. Real voices, accents,
  noisy rooms and whispering have only been tried on the maintainer's machines.
- Checked on one M4 Max (macOS 27.2) and one Windows 11 desktop (RTX 4070 Ti
  SUPER). Other Macs, macOS 14–26 and ARM Windows are untested.
- The app is built on your computer and is not notarized. Updates are manual:
  Check for Updates contacts GitHub only when you click it. After an update
  that rebuilds the app, macOS may need Sotto removed and re-added under
  Accessibility; the app says so when needed.
- Secure fields and some apps refuse simulated paste. Use History → Copy.
- A Bluetooth microphone that reconnects can need **Restart Sotto** from the
  menu.
- Parakeet and Nemotron do not use the glossary. Nemotron is English only.

[README](../README.md) · [Windows](windows-alpha.md) ·
[Validation evidence](alpha-validation.md) · [Advanced](advanced-development.md)

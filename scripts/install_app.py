#!/usr/bin/env python3
"""Build (or remove) Sotto.app in /Applications (or ~/Applications) for this source install.

The bundle holds a tiny compiled launcher that stays the parent of the Python
process, so macOS asks for Microphone/Accessibility as "Sotto" and remembers
the grants. The launcher only embeds absolute paths, so updating the source
or dependencies never changes it and the grants survive updates. Nothing is
notarized: the app is built locally, so Gatekeeper does not quarantine it.
Run by scripts/install-mac.sh; see README "Install as an app".
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ID = "org.sotto.alpha"
APP_NAME = "Sotto.app"

LAUNCHER_C = r"""
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <errno.h>

extern char **environ;
static volatile pid_t child = 0;

static void forward(int sig) { if (child > 0) kill(child, sig); }

int main(void) {
    setenv("SOTTO_DATA_DIR", %(data_dir)s, 1);
    setenv("SOTTO_HF_HOME", %(hf_home)s, 1);
    setenv("SOTTO_LAUNCHER", "app", 1);
    setenv("SOTTO_APP_EXECUTABLE", %(app_executable)s, 1);
    char *argv[] = {%(python)s, "-u", %(script)s, "run", "--idle-release", "0", NULL};
    /* Block the forwarded signals until the child PID is known, so a quit
       that arrives during the spawn is delivered to it, never lost. */
    sigset_t forwarded, empty;
    sigemptyset(&forwarded); sigemptyset(&empty);
    sigaddset(&forwarded, SIGTERM); sigaddset(&forwarded, SIGINT); sigaddset(&forwarded, SIGHUP);
    sigprocmask(SIG_BLOCK, &forwarded, NULL);
    signal(SIGTERM, forward); signal(SIGINT, forward); signal(SIGHUP, forward);
    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
    posix_spawnattr_setsigmask(&attributes, &empty);
    posix_spawnattr_setsigdefault(&attributes, &forwarded);
    posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF);
    pid_t pid;
    int spawned = posix_spawn(&pid, %(python)s, NULL, &attributes, argv, environ);
    posix_spawnattr_destroy(&attributes);
    if (spawned != 0) return 127;
    child = pid;
    sigprocmask(SIG_UNBLOCK, &forwarded, NULL);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}
"""


def c_string(value: str) -> str:
    """A C string literal for an arbitrary path (quotes, backslashes, UTF-8)."""
    escaped = "".join(f"\\{byte:03o}" if byte < 0x20 or byte > 0x7E or byte in (0x22, 0x5C)
                      else chr(byte) for byte in value.encode("utf-8"))
    return f'"{escaped}"'


def launcher_source(*, python: Path, script: Path, data_dir: Path, hf_home: Path, app_executable: Path) -> str:
    for path in (python, script, data_dir, hf_home, app_executable):
        if not path.is_absolute():
            raise ValueError(f"launcher paths must be absolute: {path}")
    return LAUNCHER_C % {"python": c_string(str(python)), "script": c_string(str(script)),
                         "data_dir": c_string(str(data_dir)), "hf_home": c_string(str(hf_home)),
                         "app_executable": c_string(str(app_executable))}


def info_plist(version: str) -> bytes:
    return plistlib.dumps({
        "CFBundleIdentifier": BUNDLE_ID, "CFBundleName": "Sotto", "CFBundleDisplayName": "Sotto",
        "CFBundleExecutable": "Sotto", "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version, "CFBundleVersion": version,
        "CFBundleIconFile": "Sotto", "LSMinimumSystemVersion": "14.0", "LSUIElement": True,
        "NSMicrophoneUsageDescription": "Sotto records only while you hold the dictation key, and "
                                        "transcribes on this Mac. Audio never leaves it.",
        "NSHighResolutionCapable": True,
    })


def draw_icon(destination: Path) -> bool:
    """A warm ember orb icon (.icns) drawn locally; skipped if AppKit is missing."""
    try:
        import AppKit
    except ImportError:
        return False
    with tempfile.TemporaryDirectory() as temporary:
        iconset = Path(temporary) / "Sotto.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                pixels = size * scale
                image = AppKit.NSImage.alloc().initWithSize_((pixels, pixels))
                image.lockFocus()
                AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.11, 0.10, 0.12, 1).set()
                AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    ((0, 0), (pixels, pixels)), pixels * 0.22, pixels * 0.22).fill()
                gradient = AppKit.NSGradient.alloc().initWithStartingColor_endingColor_(
                    AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.72, 0.35, 1),
                    AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.45, 0.08, 1))
                inset = pixels * 0.3
                orb = AppKit.NSBezierPath.bezierPathWithOvalInRect_(
                    ((inset, inset), (pixels - 2 * inset, pixels - 2 * inset)))
                gradient.drawInBezierPath_relativeCenterPosition_(orb, (-0.3, 0.3))
                image.unlockFocus()
                rep = AppKit.NSBitmapImageRep.imageRepWithData_(image.TIFFRepresentation())
                png = rep.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, {})
                suffix = "" if scale == 1 else "@2x"
                png.writeToFile_atomically_(str(iconset / f"icon_{size}x{size}{suffix}.png"), True)
        subprocess.run(["/usr/bin/iconutil", "-c", "icns", str(iconset), "-o", str(destination)], check=True)
    return True


def compile_launcher(source: str, output: Path, compiler: str = "clang") -> Path:
    with tempfile.TemporaryDirectory() as temporary:
        c_file = Path(temporary) / "launcher.c"
        c_file.write_text(source, encoding="utf-8")
        subprocess.run(["/usr/bin/xcrun", compiler, "-O2", "-Wall", "-mmacosx-version-min=14.0",
                        "-o", str(output), str(c_file)], check=True)
    return output


def build(*, applications: Path, data_dir: Path, hf_home: Path, python: Path, compiler: str = "clang") -> Path:
    app = applications / APP_NAME
    executable = app / "Contents" / "MacOS" / "Sotto"
    source = launcher_source(python=python, script=ROOT / "sotto.py", data_dir=data_dir,
                             hf_home=hf_home, app_executable=executable)
    version = "0.1.0-alpha"
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary) / APP_NAME
        (staging / "Contents" / "MacOS").mkdir(parents=True)
        (staging / "Contents" / "Resources").mkdir()
        (staging / "Contents" / "Info.plist").write_bytes(info_plist(version))
        compile_launcher(source, staging / "Contents" / "MacOS" / "Sotto", compiler)
        draw_icon(staging / "Contents" / "Resources" / "Sotto.icns")
        subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(staging)], check=True,
                       capture_output=True)
        # Keep an identical existing launcher (and so its permissions) untouched.
        if app.exists() and _same_bundle(app, staging):
            return app
        applications.mkdir(parents=True, exist_ok=True)
        if app.exists():
            shutil.rmtree(app)
        shutil.copytree(staging, app, symlinks=True)
    return app


def _same_bundle(installed: Path, staged: Path) -> bool:
    def files(root: Path) -> dict:
        return {path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*") if path.is_file() and "_CodeSignature" not in path.parts}
    try:
        return files(installed) == files(staged)
    except OSError:
        return False


def candidate_folders(home: Path | None = None) -> list[Path]:
    return [Path("/Applications"), (home or Path.home()) / "Applications"]


def default_applications(home: Path | None = None) -> Path:
    """/Applications, where Launchpad, Spotlight and 'open app' tools look,
    when this user can write there (admins can); otherwise ~/Applications."""
    system, personal = candidate_folders(home)
    return system if os.access(system, os.W_OK) else personal


def ours(app: Path) -> bool:
    """Only ever remove a Sotto.app this installer built (our bundle id)."""
    try:
        return plistlib.loads((app / "Contents" / "Info.plist").read_bytes()).get("CFBundleIdentifier") == BUNDLE_ID
    except (OSError, plistlib.InvalidFileException):
        return False


def remove_copies(folders: list[Path], keep: Path | None = None) -> list[Path]:
    removed = []
    for folder in folders:
        app = folder / APP_NAME
        if app != keep and app.exists() and ours(app):
            shutil.rmtree(app)
            removed.append(app)
    return removed


def uninstall(folders: list[Path]) -> bool:
    import login_item
    login_item.disable(unload=True)
    return bool(remove_copies(folders))


def main() -> None:
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--applications", type=Path, default=None,
                        help="install folder (default: /Applications if writable, else ~/Applications)")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get("SOTTO_DATA_DIR", "").strip() or
                                     Path.home() / "Library/Application Support/sotto-alpha"))
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()
    folders = [args.applications.expanduser().absolute()] if args.applications else candidate_folders()
    if args.uninstall:
        print("removed Sotto.app" if uninstall(folders) else "Sotto.app was not installed")
        return
    data_dir = args.data_dir.expanduser().absolute()
    target = folders[0] if args.applications else default_applications()
    app = build(applications=target, data_dir=data_dir,
                hf_home=data_dir / "huggingface", python=Path(sys.executable).absolute())
    for old in remove_copies(candidate_folders(), keep=app):  # one Sotto.app, not two
        print(f"removed the older copy at {old}")
    print(f"built {app}")


if __name__ == "__main__":
    main()

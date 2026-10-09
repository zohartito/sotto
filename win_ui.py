"""Windows tray app: a pystray icon and menu plus Tkinter dialogs.

Threads: the icon runs its own message loop (pystray) and rebuilds the menu
on every right-click, so History, Language and Progress are always current.
Menu callbacks never do real work on that thread: actions run on short-lived
worker threads, and every dialog lives on one Tk thread started on first use
(a console run never imports Tk).  The icon colour shows the state: grey
starting, ember idle, red recording, amber transcribing.

Menu labels show transcript previews (local UI, like the Mac menu); nothing
here logs them.
"""
from __future__ import annotations

import ctypes
import queue
import sys
import threading
import time
from typing import Callable

if sys.platform != "win32":
    raise ImportError("win_ui is Windows-only")

from speech_config import WHISPER_LANGUAGES, automatic_label, automatic_languages
import win_hotkey

HISTORY_ITEMS = 10
# Language menu extras (the brief's "common languages"), shown under
# "Other languages" when they are not already in the user's set.
COMMON_LANGUAGES = ("en", "ar", "zh", "nl", "fr", "de", "he", "hi", "it", "ja", "ko",
                    "pl", "pt", "ru", "es", "tr", "uk")
PREVIEW_CHARS = 48
COLORS = {"starting": (140, 140, 140), "idle": (232, 116, 59),
          "recording": (214, 40, 40), "transcribing": (240, 176, 0)}
SPEED_LABELS = {"accurate": "Accurate — large-v3-turbo",
                "fast": "Fast — small (quicker on CPU, weaker outside English)"}
INSERT_LABELS = {"paste": "Paste (the clipboard is put back)",
                 "type": "Type (the clipboard is never touched)"}
SPACING_LABELS = {"smart": "Smart (same as trailing on Windows)",
                  "trailing": "Add a space after the text", "none": "No space"}


def message_box(title: str, text: str, *, error: bool = False) -> None:
    """A native message box (no Tk needed); blocks the calling thread only."""
    flags = 0x00010000 | 0x00040000 | (0x10 if error else 0x40)  # SETFOREGROUND|TOPMOST|icon
    ctypes.windll.user32.MessageBoxW(None, text, title, flags)


def menu_text(text: str) -> str:
    """One line, and '&' shown literally (Windows menus treat it as a mnemonic)."""
    return " ".join(text.split()).replace("&", "&&")


def history_label(entry: dict, now: float) -> str:
    stamp = entry.get("ts") or 0
    when = time.strftime("%H:%M" if now - stamp < 86400 else "%b %d %H:%M", time.localtime(stamp))
    text = " ".join(str(entry.get("text", "")).split())
    if len(text) > PREVIEW_CHARS:
        text = text[:PREVIEW_CHARS - 1].rstrip() + "…"
    return menu_text(f"{when} · {text or '(empty)'}")


def language_name(code: str) -> str:
    return WHISPER_LANGUAGES.get(code, code)


def language_menu_options(active: str, saved) -> list[tuple[str, str, bool]]:
    """The Mac's menu: Automatic (among the chosen languages), each chosen
    language, and the active one if it lives outside that set."""
    codes = automatic_languages(saved)
    choices = [("auto", automatic_label(saved))] + [(code, language_name(code)) for code in codes]
    if active != "auto" and active not in codes:
        choices.append((active, language_name(active)))
    return [(mode, label, mode == active) for mode, label in choices]


def other_languages(active: str, saved) -> list[str]:
    """Common languages not already in the main Language menu."""
    shown = {mode for mode, _label, _checked in language_menu_options(active, saved)}
    return [code for code in COMMON_LANGUAGES if code not in shown]


def tooltip(state: str, trigger: str, phase: str) -> str:
    if state == "starting":
        return f"Sotto — {phase}"[:120]
    return {"recording": "Sotto — recording", "transcribing": "Sotto — transcribing"}.get(
        state, f"Sotto — hold {win_hotkey.label(trigger)} to dictate")


def icon_image(state: str):
    from PIL import Image, ImageDraw
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((4, 4, 60, 60), fill=COLORS.get(state, COLORS["idle"]) + (255,))
    draw.ellipse((18, 14, 40, 36), fill=(255, 255, 255, 70))  # ember highlight
    return image


def update_report(data_dir=None) -> str | None:
    """The outcome of the last tray update, reported once at the next start."""
    from pathlib import Path
    from sotto_paths import DATA_DIR
    path = Path(data_dir or DATA_DIR) / "update-status.txt"
    try:
        line = path.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return None
    path.unlink(missing_ok=True)
    if line.startswith("ok"):
        return f"Sotto updated ({line[2:].strip()})."
    return "Update failed: " + line.removeprefix("failed").strip()

class DialogHost:
    """Every Tk window on one thread, created on first use and stopped once."""

    def __init__(self, log: Callable[[str], None]) -> None:
        self._log = log
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stopped = False

    def call(self, build: Callable) -> None:
        """Run ``build(root)`` on the Tk thread."""
        with self._lock:
            if self._stopped:
                return
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True, name="sotto-dialogs")
                self._thread.start()
        self._queue.put(build)

    def _run(self) -> None:
        try:
            import tkinter as tk
        except ImportError:
            message_box("Sotto", "This Python has no tkinter, so Settings and Correct can't open. "
                        "Re-run the python.org installer, choose Modify, and tick 'tcl/tk and IDLE'.",
                        error=True)
            return
        root = tk.Tk()
        root.withdraw()

        def pump() -> None:
            while True:
                try:
                    build = self._queue.get_nowait()
                except queue.Empty:
                    break
                if build is None:
                    root.quit()
                    return
                try:
                    build(root)
                except Exception as exc:
                    self._log(f"! dialog failed: {str(exc)[:120]}")
            root.after(50, pump)

        root.after(50, pump)
        root.mainloop()
        # Tear Tk down on its own thread: Tcl objects (windows, variables in
        # reference cycles) must never be finalized by another thread.
        root.destroy()
        root = None
        import gc
        gc.collect()

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            thread = self._thread
        if thread is not None:
            self._queue.put(None)
            thread.join(3)


class TrayApp:
    """The icon, its menu and the dialogs; a Controller is attached once listening."""

    def __init__(self, *, quit: Callable[[], None], log: Callable[[str], None]) -> None:
        self._quit = quit
        self._log = log
        self.controller = None
        self._phase = "starting…"
        self._state = "starting"
        self._icon = None
        self._images: dict = {}
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._changed = threading.Event()
        self._stopping = False
        self.dialogs = DialogHost(log)
        self._settings_window = None  # Tk thread only

    # lifecycle ---------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="sotto-tray")
        self._thread.start()
        threading.Thread(target=self._apply_changes, daemon=True, name="sotto-tray-icon").start()
        self._ready.wait(5)

    def _run(self) -> None:
        import pystray
        from pystray._util import win32 as pystray_win32

        class Icon(pystray.Icon):
            def _on_notify(icon, wparam, lparam):  # noqa: N805 — pystray's own naming
                if lparam == pystray_win32.WM_RBUTTONUP:
                    icon._update_menu()  # fresh History/Language/Progress, on this thread
                return super()._on_notify(wparam, lparam)

        def setup(icon) -> None:
            icon.visible = True
            self._ready.set()
            self._changed.set()  # anything that changed before the icon existed

        self._icon = Icon("sotto", self._image("starting"), self._title(),
                          menu=pystray.Menu(self.items))
        try:
            self._icon.run(setup=setup)
        except Exception as exc:
            self._log(f"! tray icon failed: {str(exc)[:120]}")
            self._ready.set()

    def stop(self) -> None:
        self._stopping = True
        self._changed.set()
        self.dialogs.stop()
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(3)

    def attach(self, controller) -> None:
        self.controller = controller
        self.set_state("idle")
        ready = f"Ready — hold {win_hotkey.label(controller.hook.trigger)} to dictate."
        report = update_report()
        self.notify(f"{report} {ready}" if report else ready)

    # state -------------------------------------------------------------------
    def _image(self, state: str):
        if state not in self._images:
            self._images[state] = icon_image(state)
        return self._images[state]

    def _title(self) -> str:
        trigger = self.controller.hook.trigger if self.controller is not None else "right-ctrl"
        return tooltip(self._state, trigger, self._phase)

    def set_phase(self, phase: str) -> None:
        self._phase = phase
        self._changed.set()

    def set_state(self, state: str) -> None:
        """Never blocks: callers include the keyboard-hook thread."""
        if state != self._state:
            self._state = state
            self._changed.set()

    def _apply_changes(self) -> None:
        """The only thread that touches the icon image and tooltip."""
        shown = None
        while not self._stopping:
            self._changed.wait()
            self._changed.clear()
            icon = self._icon
            wanted = (self._state, self._title())
            if icon is None or wanted == shown or self._stopping:
                continue
            try:
                icon.icon = self._image(wanted[0])
                icon.title = wanted[1]
                shown = wanted
            except Exception:
                pass  # the icon is not shown (yet) or is going away

    def notify(self, text: str) -> None:
        try:
            if self._icon is not None:
                self._icon.notify(text, "Sotto")
        except Exception:
            pass

    # menu --------------------------------------------------------------------
    def _act(self, action: Callable, *args, done: str | None = None) -> Callable:
        """A menu callback that runs ``action`` off the tray thread."""
        def work() -> None:
            try:
                result = action(*args)
                message = done if done is not None else (result if isinstance(result, str) else None)
                if message:
                    self.notify(message)
            except Exception as exc:
                self._log(f"! tray action failed: {str(exc)[:120]}")
                message_box("Sotto", str(exc)[:300], error=True)
        return lambda icon, item: threading.Thread(target=work, daemon=True).start()

    def _dialog(self, build: Callable, *args) -> Callable:
        return lambda icon, item: self.dialogs.call(lambda root: build(root, *args))

    def items(self):
        import pystray
        Item, Menu = pystray.MenuItem, pystray.Menu
        controller = self.controller
        if controller is None:
            yield Item(menu_text(f"Sotto — {self._phase}"), None, enabled=False)
            yield Menu.SEPARATOR
            yield Item("Quit", lambda icon, item: self._quit())
            return
        yield Item(menu_text(self._title()), None, enabled=False)
        if controller.recording():
            yield Item("Finish dictation (copy the text)", self._act(controller.finish_now))
        else:
            yield Item("Start dictation (hands-free)", self._act(controller.start_now))
        yield Menu.SEPARATOR
        yield Item("History", Menu(*self._history_items(controller)))
        yield Item("Language", Menu(*self._language_items(controller)))
        yield Item("Speed", Menu(*self._speed_items(controller)))
        yield Item("Progress", Menu(*[Item(menu_text(line), None, enabled=False)
                                      for line in controller.progress()]))
        yield Item("Dictionary", self._act(controller.open_dictionary))
        yield Item("Settings…", self._dialog(self.settings_dialog), default=True)
        yield Menu.SEPARATOR
        yield Item("Check for updates…", self._act(self._check_updates))
        yield Item("Restart", self._act(controller.restart))
        yield Item("Quit", lambda icon, item: controller.quit())

    def _history_items(self, controller) -> list:
        import pystray
        Item, Menu = pystray.MenuItem, pystray.Menu
        entries = controller.entries(HISTORY_ITEMS)
        if not entries:
            return [Item("No dictations yet", None, enabled=False)]
        now = time.time()
        items = [Item(history_label(entry, now), Menu(
            Item("Copy", self._act(controller.copy, entry["id"], done="Copied.")),
            Item("Retry", self._act(controller.retry, entry["id"])),  # it says what it did
            Item("Correct…", self._dialog(self.correct_dialog, entry)),
            Item("Delete", self._act(controller.delete, entry["id"])),
        )) for entry in entries]
        items += [Menu.SEPARATOR, Item("Clear history…", self._dialog(self.confirm_clear))]
        return items

    def _language_items(self, controller) -> list:
        import pystray
        Item, Menu = pystray.MenuItem, pystray.Menu
        saved, current = controller.saved_languages(), controller.language()

        def choice(mode: str, label: str):
            return Item(menu_text(label), self._act(controller.set_language, mode),
                        checked=lambda item, mode=mode: controller.language() == mode, radio=True)

        items = [choice(mode, label) for mode, label, _checked in language_menu_options(current, saved)]
        extra = other_languages(current, saved)
        if extra:
            items += [Menu.SEPARATOR,
                      Item("Other languages", Menu(*[choice(code, language_name(code)) for code in extra]))]
        return items

    def _speed_items(self, controller) -> list:
        import pystray
        Item = pystray.MenuItem
        return [Item(label, self._act(controller.set_speed, speed),
                     checked=lambda item, speed=speed: controller.speed == speed, radio=True)
                for speed, label in SPEED_LABELS.items()]

    # dialogs (Tk thread) -----------------------------------------------------
    @staticmethod
    def _window(root, title: str):
        import tkinter as tk
        window = tk.Toplevel(root)
        window.title(title)
        window.resizable(False, False)
        window.attributes("-topmost", True)
        window.after(300, lambda: window.attributes("-topmost", False))
        window.after(50, window.focus_force)
        return window

    def _check_updates(self) -> None:
        result = self.controller.check_updates()
        self.dialogs.call(lambda root: self.update_dialog(root, result))

    def update_dialog(self, root, result) -> None:
        from tkinter import messagebox
        import updates
        title, text = updates.describe(
            result, "run git pull, then scripts\\install-windows.ps1, from the Sotto folder.")
        if result.state == "available":
            if messagebox.askyesno(title, text, parent=root):
                self._act(self.controller.update_and_restart)(None, None)
        elif result.state == "not-git":
            if messagebox.askyesno("Download the latest release",
                                   f"{result.detail}\n\nOpen the releases page?", parent=root):
                import webbrowser
                webbrowser.open(updates.RELEASES_URL)
        elif result.state == "failed":
            messagebox.showerror(title, text, parent=root)
        else:
            messagebox.showinfo(title, text, parent=root)

    def confirm_clear(self, root) -> None:
        from tkinter import messagebox
        if messagebox.askyesno("Clear history", "Delete every transcript and recording?\n"
                               "Models, settings and the dictionary stay.", parent=root):
            self._act(self.controller.clear, done="History cleared.")(None, None)

    def correct_dialog(self, root, entry: dict) -> None:
        import tkinter as tk
        from tkinter import messagebox, ttk
        before = str(entry.get("text", ""))
        window = self._window(root, "Correct transcript")
        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Edit the text the way it should have been written:").pack(anchor="w")
        text = tk.Text(frame, width=64, height=8, wrap="word", undo=True)
        text.insert("1.0", before)
        text.pack(fill="both", expand=True, pady=(6, 10))
        text.focus_set()
        buttons = ttk.Frame(frame)
        buttons.pack(anchor="e")

        def save() -> None:
            after = text.get("1.0", "end-1c").strip()
            if not after:
                messagebox.showerror("Correct transcript", "The corrected text is empty.", parent=window)
                return
            try:
                suggestions = self.controller.correct(entry["id"], before, after, entry.get("revision"))
            except Exception as exc:
                messagebox.showerror("Could not save correction", str(exc)[:300], parent=window)
                return
            window.destroy()
            if suggestions:
                self.suggestions_dialog(root, suggestions)
            else:
                self.notify("Correction saved.")

        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="right")
        ttk.Button(buttons, text="Save", command=save).pack(side="right", padx=(0, 6))

    def suggestions_dialog(self, root, suggestions: list) -> None:
        import tkinter as tk
        from tkinter import messagebox, ttk
        window = self._window(root, "Always write these your way?")
        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Add these to your dictionary so every future dictation "
                              "is written this way:").pack(anchor="w", pady=(0, 6))
        choices = []
        for rule in suggestions:
            chosen = tk.BooleanVar(window, value=True)
            ttk.Checkbutton(frame, text=f"{rule.heard}  →  {rule.write}", variable=chosen).pack(anchor="w")
            choices.append((rule, chosen))
        buttons = ttk.Frame(frame)
        buttons.pack(anchor="e", pady=(10, 0))

        def add() -> None:
            selected = [rule for rule, chosen in choices if chosen.get()]
            try:
                added = self.controller.add_rules(selected) if selected else 0
            except Exception as exc:
                messagebox.showerror("Could not update the dictionary", str(exc)[:300], parent=window)
                return
            window.destroy()
            self.notify(f"Correction saved; {added} dictionary rule(s) added.")

        ttk.Button(buttons, text="Not now",
                   command=lambda: (window.destroy(), self.notify("Correction saved."))).pack(side="right")
        ttk.Button(buttons, text="Add selected", command=add).pack(side="right", padx=(0, 6))

    def settings_dialog(self, root) -> None:
        import tkinter as tk
        from tkinter import messagebox, ttk
        import settings as user_settings
        if self._settings_window is not None and self._settings_window.winfo_exists():
            self._settings_window.lift()
            self._settings_window.focus_force()
            return
        controller = self.controller
        saved = controller.settings()
        window = self._settings_window = self._window(root, "Sotto settings")
        frame = ttk.Frame(window, padding=14)
        frame.pack(fill="both", expand=True)

        def section(title: str) -> ttk.LabelFrame:
            box = ttk.LabelFrame(frame, text=title, padding=8)
            box.pack(fill="x", pady=(0, 8))
            return box

        box = section("Trigger key (hold to talk, double-tap for hands-free)")
        labels = {name: label for name, (_key, _vk, label) in win_hotkey.TRIGGERS.items()}
        trigger = tk.StringVar(window, value=labels[win_hotkey.supported(saved["trigger"])])
        ttk.Combobox(box, textvariable=trigger, values=list(labels.values()), state="readonly",
                     width=24).pack(anchor="w")

        box = section(f"Automatic chooses only among these languages "
                      f"(1 to {user_settings.MAX_LANGUAGES})")
        chosen_languages = {}
        current = set(automatic_languages(saved["languages"]))
        canvas = tk.Canvas(box, height=150, highlightthickness=0)
        scrollbar = ttk.Scrollbar(box, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        # The list is the window's only scrolling area: the wheel scrolls it
        # from anywhere in the window, one line per notch or touchpad nudge.
        window.bind("<MouseWheel>", lambda event: canvas.yview_scroll(
            -1 if event.delta > 0 else 1, "units") if event.delta else None)
        for index, (code, name) in enumerate(sorted(WHISPER_LANGUAGES.items(), key=lambda item: item[1])):
            variable = tk.BooleanVar(window, value=code in current)
            ttk.Checkbutton(inner, text=name, variable=variable).grid(
                row=index // 3, column=index % 3, sticky="w", padx=(0, 14))
            chosen_languages[code] = variable

        def radios(title: str, options: dict, value: str) -> tk.StringVar:
            variable = tk.StringVar(window, value=value)
            box = section(title)
            for key, label in options.items():
                ttk.Radiobutton(box, text=label, value=key, variable=variable).pack(anchor="w")
            return variable

        insert_mode = radios("Insert text by", INSERT_LABELS, saved["insert_mode"])
        spacing = radios("Spacing after each dictation", SPACING_LABELS, saved["spacing"])
        box = section("Cleanup (English dictation)")
        fillers = tk.BooleanVar(window, value=bool(saved["remove_fillers"]))
        ttk.Checkbutton(box, text="Remove filler words (um, uh)", variable=fillers).pack(anchor="w")
        commands = tk.BooleanVar(window, value=bool(saved["voice_commands"]))
        ttk.Checkbutton(box, text="Voice commands: new line, new paragraph, scratch that",
                        variable=commands).pack(anchor="w")
        speed = radios("Speed (applies after restart)", SPEED_LABELS, saved["speed"])

        box = section("General")
        login = tk.BooleanVar(window, value=bool(saved["launch_at_login"]))
        ttk.Checkbutton(box, text="Launch Sotto when I sign in", variable=login).pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(anchor="w", pady=(6, 0))
        ttk.Button(row, text="Open dictionary", command=lambda: self._act(
            controller.open_dictionary)(None, None)).pack(side="left")
        ttk.Button(row, text="Show data folder", command=lambda: self._act(
            controller.show_data_folder)(None, None)).pack(side="left", padx=(6, 0))

        box = section("Progress (this PC only)")
        for line in controller.progress():
            ttk.Label(box, text=line).pack(anchor="w")

        status = ttk.Label(frame, text="", wraplength=440, justify="left")
        status.pack(fill="x", pady=(0, 8))
        buttons = ttk.Frame(frame)
        buttons.pack(anchor="e")

        def save() -> None:
            languages = [code for code, variable in chosen_languages.items() if variable.get()]
            if not 1 <= len(languages) <= user_settings.MAX_LANGUAGES:
                messagebox.showerror("Sotto settings",
                                     f"Choose between 1 and {user_settings.MAX_LANGUAGES} languages "
                                     "for Automatic.", parent=window)
                return
            by_label = {label: name for name, label in labels.items()}
            updates = {"trigger": by_label[trigger.get()], "languages": languages,
                       "insert_mode": insert_mode.get(), "spacing": spacing.get(),
                       "speed": speed.get(), "launch_at_login": bool(login.get()),
                       "remove_fillers": bool(fillers.get()), "voice_commands": bool(commands.get())}
            try:
                notes = controller.save_settings(updates)
            except Exception as exc:
                messagebox.showerror("Could not save settings", str(exc)[:300], parent=window)
                return
            status.configure(text="Saved. " + " ".join(notes))

        ttk.Button(buttons, text="Close", command=window.destroy).pack(side="right")
        ttk.Button(buttons, text="Save", command=save).pack(side="right", padx=(0, 6))

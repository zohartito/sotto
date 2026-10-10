"""sotto's voice indicator — a quiet ember, not a waveform.

A small glassy island at top-center holds a single warm orb that breathes
with the smoothed voice level (fast attack, slow decay, eased core-animation
transitions). While transcribing, the orb cools to pale white and pulses.

Main-thread only — call every method through PyObjCTools.AppHelper.callAfter
when coming from the tap, audio, or transcription threads.
"""

from __future__ import annotations

import math
import time

import AppKit
import objc
import Quartz
from PyObjCTools import AppHelper
from speech_config import LANGUAGE_LABELS, WHISPER_LANGUAGES, automatic_label, automatic_languages

ISLAND_W, ISLAND_H = 88, 34
HINT_FONT_SIZE = 12.0   # hands-free "tap … to stop" beside the orb
LIVE_CHARS = 48         # live words shown beside the orb (the newest end)
LIVE_REFRESH_S = 0.2
ORB_SIZE = 19.0
DECAY = 0.85            # per-push falloff — the orb sinks, never snaps
EASE_S = 0.14           # animation duration per level change

EMBER = (1.0, 0.52, 0.12)        # unmistakable orange — recording
PALE = (0.95, 0.92, 0.86)        # cooled ember — transcribing



def screen_containing_point(screens, point):
    """Return the display containing an AppKit global point, if any.

    A menu-bar accessory has no key window, so ``NSScreen.mainScreen()`` can
    remain pinned to the menu-bar display even while the user works on a
    different monitor.  The pointer is the stable local signal available at
    the instant push-to-talk starts.
    """
    for screen in screens:
        frame = screen.frame()
        if (frame.origin.x <= point.x < frame.origin.x + frame.size.width
                and frame.origin.y <= point.y < frame.origin.y + frame.size.height):
            return screen
    return None


def window_info_is_visible(rows) -> bool:
    """Interpret WindowServer metadata without treating missing data as live."""
    return any(bool(row.get(Quartz.kCGWindowIsOnscreen))
               and float(row.get(Quartz.kCGWindowAlpha, 1.0)) > 0.0
               for row in rows)


def language_menu_options(active: str, languages=None) -> tuple[tuple[str, str, bool], ...]:
    """Pure menu description: Automatic (among the chosen languages), each
    chosen language, and the active one if it lives outside that set."""
    codes = automatic_languages(languages)
    choices = [("auto", automatic_label(languages))]
    choices += [(code, LANGUAGE_LABELS[code]) for code in codes]
    if active != "auto" and active not in codes:
        if active not in WHISPER_LANGUAGES:
            raise ValueError(f"Unknown language mode: {active}")
        choices.append((active, LANGUAGE_LABELS[active]))
    return tuple((mode, label, mode == active) for mode, label in choices)


def correction_disclosure(adaptive: bool) -> str:
    return ("Saving is an explicit review action and automatically enters Sotto's private learning set; deletion or revocation removes it."
            if adaptive else
            "Save a corrected transcript locally. Adding it to Sotto's learning set remains a separate action.")


def show_manual_enroll_action(*, adaptive_mode: bool, corrected: bool, active: bool,
                              adaptive_eligible: bool = True) -> bool:
    """Whether an inactive corrected row needs the nonadaptive enroll action."""
    return corrected and not active and (not adaptive_mode or not adaptive_eligible)


def show_adaptive_review_actions(*, adaptive_mode: bool, reviewable: bool) -> bool:
    return adaptive_mode and reviewable


def _color(rgb: tuple[float, float, float], alpha: float = 1.0) -> AppKit.NSColor:
    return AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(*rgb, alpha)


def live_snippet(text: str, limit: int = LIVE_CHARS) -> str:
    """The newest end of the live words, cut at a word where possible."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    tail = text[-(limit - 1):]
    space = tail.find(" ")
    if 0 <= space < limit // 3:
        tail = tail[space + 1:]
    return "…" + tail


_app_icon = {"image": None}  # Sotto.app's icon once use_app_icon found it


def use_app_icon(icon_path) -> None:
    """Sotto.app runs Python as a child, so dialogs would show Python's icon.

    Side effects: sets the application icon and remembers it for new_alert();
    AppKit can put Python's bundle icon back when the app finishes launching,
    so every dialog also carries it itself."""
    image = AppKit.NSImage.alloc().initWithContentsOfFile_(str(icon_path)) if icon_path else None
    if image is not None:
        _app_icon["image"] = image
        AppKit.NSApplication.sharedApplication().setApplicationIconImage_(image)


def new_alert():
    """An NSAlert showing Sotto's icon (when Sotto.app started this process)."""
    alert = AppKit.NSAlert.alloc().init()
    if _app_icon["image"] is not None:
        alert.setIcon_(_app_icon["image"])
    return alert


class _PermissionWait(AppKit.NSObject):
    """Keeps a permission dialog up while System Settings is open, and closes
    it by itself once the permission is on."""

    url = None
    granted = None
    register = None

    def openSettings_(self, _sender):
        if self.register is not None:
            self.register()  # makes macOS list Sotto in the pane
        AppKit.NSWorkspace.sharedWorkspace().openURL_(self.url)

    def poll_(self, _timer):
        if self.granted():
            # stopModal from a timer waits for the next event; abort does not.
            AppKit.NSApp.abortModal()


def permission_dialog(title: str, message: str, url: str, *, granted=None, register=None,
                      icon_path=None) -> bool:
    """Explain a missing permission and offer its System Settings pane.

    Without ``granted`` either button closes the dialog. With it, "Open
    System Settings" leaves the dialog up, the dialog closes as soon as
    ``granted()`` is true and this returns True; Quit returns False. The main
    thread keeps serving events throughout, so the dialog never freezes."""
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    use_app_icon(icon_path)
    app.activateIgnoringOtherApps_(True)
    alert = new_alert()
    alert.setMessageText_(title)
    alert.setInformativeText_(message)
    settings = alert.addButtonWithTitle_("Open System Settings")
    alert.addButtonWithTitle_("Quit")
    target = _PermissionWait.alloc().init()
    target.url = AppKit.NSURL.URLWithString_(url)
    target.granted = granted
    target.register = register
    timer = None
    if granted is not None:
        settings.setTarget_(target)
        settings.setAction_("openSettings:")
        timer = AppKit.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0, target, "poll:", None, True)
        AppKit.NSRunLoop.currentRunLoop().addTimer_forMode_(timer, AppKit.NSRunLoopCommonModes)
    response = alert.runModal()
    if timer is not None:
        timer.invalidate()
    if granted is None and response == AppKit.NSAlertFirstButtonReturn:
        target.openSettings_(None)
    # Let the window server take the dialog down before slow work holds this thread.
    AppKit.NSRunLoop.currentRunLoop().runUntilDate_(AppKit.NSDate.dateWithTimeIntervalSinceNow_(0.1))
    return granted is not None and response != AppKit.NSAlertSecondButtonReturn and bool(granted())


class _MenuTarget(AppKit.NSObject):
    """Bridges menu clicks and system notifications to python callbacks."""

    callbacks: dict = {}
    ui = None

    def quit_(self, _sender):
        callback = self.callbacks.get("quit")
        if callback is not None:
            callback()
        else:
            AppKit.NSApp.terminate_(None)

    def restart_(self, _sender):
        self.callbacks["restart"]()

    def screensChanged_(self, _note):
        if self.ui is not None:
            self.ui.reposition_if_visible()

    def didWake_(self, _note):
        if self.ui is not None and self.ui.on_wake is not None:
            self.ui.on_wake()

    def levelTick_(self, _timer):
        if self.ui is not None and self.ui.level_source is not None:
            self.ui.push_level(self.ui.level_source())
        if self.ui is not None:
            self.ui.refresh_live_text()

    def copyEntry_(self, sender):
        self.callbacks["copy"](sender.representedObject())

    def retryEntry_(self, sender):
        self.callbacks["retry"](sender.representedObject())

    def saveEntry_(self, sender):
        self.callbacks["save"](sender.representedObject())

    def deleteEntry_(self, sender):
        self.callbacks["delete"](sender.representedObject())

    def correctEntry_(self, sender):
        """Open the correction editor on the AppKit main thread."""
        if self.ui is not None:
            payload = sender.representedObject()
            self.ui.correct_transcript(payload["id"], payload["text"], payload.get("revision"),
                                       bool(payload.get("adaptive_eligible", False)))

    def enrollEntry_(self, sender):
        """Ask for explicit local-learning consent before enrolling."""
        if self.ui is not None:
            self.ui.confirm_enrollment(sender.representedObject())

    def revokeEntry_(self, sender):
        """Ask for explicit confirmation before removing local learning data."""
        if self.ui is not None:
            self.ui.confirm_revocation(sender.representedObject())

    def correctAsIs_(self, sender):
        payload = sender.representedObject()
        self.callbacks["correct_as_is"](payload["id"], payload.get("revision"))

    def noSpeech_(self, sender):
        payload = sender.representedObject()
        self.callbacks["no_speech"](payload["id"], payload.get("revision"))

    def skipLearningReview_(self, sender):
        payload = sender.representedObject()
        if isinstance(payload, dict):
            self.callbacks["skip_learning_review"](payload["id"], payload["reason"], payload.get("revision"))
        else:
            self.callbacks["skip_learning_review"](payload, "private")

    def clearHistory_(self, _sender):
        self.callbacks["clear"]()

    def finishNow_(self, _sender):
        self.callbacks["finish_now"]()

    def startNow_(self, _sender):
        self.callbacks["start_now"]()

    def setLanguage_(self, sender):
        self.callbacks["set_language"](sender.representedObject())

    def setEngine_(self, sender):
        self.callbacks["set_engine"](sender.representedObject())

    def openDictionary_(self, _sender):
        self.callbacks["open_dictionary"]()

    def openSettings_(self, _sender):
        self.callbacks["open_settings"]()

    def checkUpdates_(self, _sender):
        self.callbacks["check_updates"]()


def init_app() -> "StatusUI":
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    return StatusUI()


class StatusUI:
    def __init__(self) -> None:
        bar = AppKit.NSStatusBar.systemStatusBar()
        self._status = bar.statusItemWithLength_(AppKit.NSVariableStatusItemLength)
        self._status.button().setTitle_("◦")
        self._target = _MenuTarget.alloc().init()
        self._menu = AppKit.NSMenu.alloc().init()
        self._menu.setAutoenablesItems_(False)
        self._status.setMenu_(self._menu)
        self.language_mode = "auto"
        self.automatic_languages = None  # Settings -> languages
        self.language_switching_enabled = True
        self.engine_mode = "whisper"
        self.engine_switching_enabled = True
        from nemotron_backend import is_installed
        self.nemotron_available = is_installed()
        self._history_entries: list[dict] = []
        self.refresh_history([])

        self._display_level = 0.0
        self._tap_healthy = True  # False shows ⚠ in the menu bar while idle
        self._mode = "idle"  # idle | recording | transcribing
        self._indicator_state = "idle"  # idle | waking | recording | transcribing
        self._visibility_generation = 0
        self._last_screen_id = None
        self.adaptive_mode = False
        self.on_wake = None       # callable set by the daemon (gesture reset)
        self.on_overlay_event = None  # callable(message), set by the daemon
        self.level_source = None  # callable() -> rms, polled by a main-thread
        self._level_timer = None  # timer — the audio thread never touches UI
        self._hint = None         # hands-free stop hint, shown beside the orb
        self.live_text_source = None  # callable() -> words so far (streaming engines)
        self._live_text = ""
        self._live_checked = 0.0
        self._panel = self._make_island()
        self._orb = self._make_orb()

        self._target.ui = self
        AppKit.NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
            self._target, "screensChanged:",
            AppKit.NSApplicationDidChangeScreenParametersNotification, None)
        AppKit.NSWorkspace.sharedWorkspace().notificationCenter(
        ).addObserver_selector_name_object_(
            self._target, "didWake:", AppKit.NSWorkspaceDidWakeNotification, None)

    # -- history menu ------------------------------------------------------

    def set_history_callbacks(self, callbacks: dict) -> None:
        """Register history actions.

        copy/retry/save/delete/enroll/revoke take an entry id;
        correct takes (entry_id, corrected_text, expected_revision);
        correct_as_is/no_speech take (entry_id, expected_revision);
        skip_learning_review takes (entry_id, reason, expected_revision);
        clear/finish_now/start_now take no arguments; set_language takes a mode.
        """
        self._target.callbacks = callbacks

    def refresh_history(self, entries: list[dict]) -> None:
        """Rebuild the dropdown: recent transcripts, each with an action submenu."""
        self._history_entries = list(entries)
        with objc.autorelease_pool():
            self._rebuild_menu(self._history_entries)

    def set_language_mode(self, mode: str, enabled: bool = True) -> None:
        """Show the active per-capture language choice in the menu bar menu."""
        language_menu_options(mode, self.automatic_languages)  # validate before mutating visible state
        self.language_mode = mode
        self.language_switching_enabled = enabled
        self.refresh_history(self._history_entries)

    def set_engine_mode(self, mode: str, enabled: bool = True) -> None:
        from speech_config import ENGINE_CHOICES
        if mode not in ENGINE_CHOICES:
            raise ValueError("Unknown speech engine")
        self.engine_mode = mode
        self.engine_switching_enabled = enabled
        self.refresh_history(self._history_entries)

    def set_tap_health(self, healthy: bool) -> None:
        """Make a deaf hotkey visible: ⚠ in the menu bar while the event tap
        cannot be re-enabled, restored to the idle glyph once it recovers.
        Side effects: remembered, so hiding the pill does not erase the ⚠."""
        self._tap_healthy = healthy
        if self._mode == "idle":
            self._status.button().setTitle_(self._idle_glyph())

    def _idle_glyph(self) -> str:
        return "◦" if getattr(self, "_tap_healthy", True) else "⚠"

    def set_adaptive_stage(self, stage: str) -> None:
        """A compact idle indicator; it deliberately contains no user text."""
        if self._mode == "idle":
            self._status.button().setTitle_(stage)

    def _rebuild_menu(self, entries: list[dict]) -> None:
        self._menu.removeAllItems()

        # Actions first — they must stay reachable without scrolling past a
        # long history.
        for label, selector, key in (("Start dictation (hands-free)", "startNow:", ""),
                                     ("Finish recording now", "finishNow:", ""),
                                     ("Settings…", "openSettings:", ","),
                                     ("Check for Updates…", "checkUpdates:", ""),
                                     ("Restart sotto", "restart:", ""),
                                     ("Quit sotto", "quit:", "")):
            item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                label, selector, key)
            item.setTarget_(self._target)
            self._menu.addItem_(item)

        from speech_config import ENGINE_CHOICES
        from nemotron_backend import is_installed
        # Re-check on every rebuild: a runtime installed after launch shows up.
        self.nemotron_available = is_installed()
        from sotto_paths import MODEL_CACHE_DIR
        from speech_config import MODEL_PROFILES, pinned_snapshot_cached
        parakeet_available = pinned_snapshot_cached(MODEL_PROFILES["parakeet"].repo, MODEL_CACHE_DIR / "hub",
                                                    ("config.json", "model.safetensors"))
        active_engine = getattr(self, "engine_mode", "whisper")
        engine_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            f"Speech engine: {ENGINE_CHOICES[active_engine]}", None, "")
        engine_menu = AppKit.NSMenu.alloc().init()
        engine_menu.setAutoenablesItems_(False)
        for mode, label in ENGINE_CHOICES.items():
            available = {"whisper": True, "parakeet": parakeet_available,
                         "nemotron": getattr(self, "nemotron_available", False)}[mode]
            title = label if available else label + (" — not downloaded" if mode == "parakeet"
                                                     else " — not installed")
            choice = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                title, "setEngine:", "")
            choice.setTarget_(self._target)
            choice.setRepresentedObject_(mode)
            choice.setState_(AppKit.NSControlStateValueOn if mode == active_engine
                             else AppKit.NSControlStateValueOff)
            choice.setEnabled_(available and getattr(self, "engine_switching_enabled", True))
            engine_menu.addItem_(choice)
        engine_item.setSubmenu_(engine_menu)
        self._menu.addItem_(engine_item)

        active_label = next(
            label for mode, label, selected in language_menu_options(self.language_mode, self.automatic_languages)
            if selected
        )
        language_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            f"Language: {active_label}", None, "")
        if self.language_switching_enabled:
            submenu = AppKit.NSMenu.alloc().init()
            submenu.setAutoenablesItems_(False)
            for mode, label, selected in language_menu_options(self.language_mode,
                                                               self.automatic_languages):
                choice = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    label, "setLanguage:", "")
                choice.setTarget_(self._target)
                choice.setRepresentedObject_(mode)
                choice.setState_(AppKit.NSControlStateValueOn if selected
                                 else AppKit.NSControlStateValueOff)
                submenu.addItem_(choice)
            submenu.addItem_(AppKit.NSMenuItem.separatorItem())
            more = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "More languages…", "openSettings:", "")
            more.setTarget_(self._target)
            submenu.addItem_(more)
            language_item.setSubmenu_(submenu)
        else:
            language_item.setTitle_({"nemotron": "Language: English (Nemotron)",
                                     "parakeet": "Language: detected by Parakeet (25 European)"}
                                    .get(active_engine, "Language: English (adaptive mode)"))
            language_item.setEnabled_(False)
        self._menu.addItem_(language_item)
        import dictionary
        rule_count = len(dictionary.load(dictionary.DICTIONARY_PATH))
        dictionary_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            f"Dictionary ({rule_count} rule{'' if rule_count == 1 else 's'})…", "openDictionary:", "")
        dictionary_item.setTarget_(self._target)
        self._menu.addItem_(dictionary_item)
        import progress
        progress_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Your progress", None, "")
        progress_menu = AppKit.NSMenu.alloc().init()
        progress_menu.setAutoenablesItems_(False)
        from sotto_paths import DATA_DIR
        totals = progress.load_totals(DATA_DIR / progress.TOTALS_NAME)
        for line in progress.lines(progress.summarize(entries, rules=rule_count), totals):
            info = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(line, None, "")
            info.setEnabled_(False)
            progress_menu.addItem_(info)
        progress_item.setSubmenu_(progress_menu)
        self._menu.addItem_(progress_item)
        self._menu.addItem_(AppKit.NSMenuItem.separatorItem())

        # Keep the visible menu to about half the screen; the rest lives in an
        # "Earlier" submenu so nothing is lost but nothing runs off-screen.
        screen = AppKit.NSScreen.mainScreen()
        height = screen.frame().size.height if screen else 900
        inline_max = max(6, min(24, int((height * 0.5) / 24) - 5))
        inline, overflow = entries[:inline_max], entries[inline_max:]
        self._add_entry_rows(self._menu, inline)
        if overflow:
            older = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                f"Earlier ({len(overflow)})", None, "")
            submenu = AppKit.NSMenu.alloc().init()
            submenu.setAutoenablesItems_(False)
            self._add_entry_rows(submenu, overflow)
            older.setSubmenu_(submenu)
            self._menu.addItem_(older)
        self._menu.addItem_(AppKit.NSMenuItem.separatorItem())
        clear_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Clear history", "clearHistory:", "")
        clear_item.setTarget_(self._target)
        self._menu.addItem_(clear_item)
        if not entries:
            empty = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "No transcripts yet", None, "")
            empty.setEnabled_(False)
            self._menu.addItem_(empty)

    def _add_entry_rows(self, menu, entries: list[dict]) -> None:
        """Transcript rows, newest first, grouped under day headers."""
        import datetime
        today = datetime.date.today()
        current_day = None
        for entry in entries:
            day = datetime.datetime.fromtimestamp(entry["ts"]).date()
            if day != current_day:
                current_day = day
                delta = (today - day).days
                label = ("Today" if delta == 0 else
                         "Yesterday" if delta == 1 else
                         day.strftime("%a %b %-d"))
                header = (AppKit.NSMenuItem.alloc()
                          .initWithTitle_action_keyEquivalent_(label, None, ""))
                header.setEnabled_(False)
                menu.addItem_(header)
            menu.addItem_(self._entry_item(entry))

    def _entry_item(self, entry: dict) -> AppKit.NSMenuItem:
        import datetime
        when = datetime.datetime.fromtimestamp(entry["ts"]).strftime("%H:%M")
        text = " ".join(entry["text"].split())  # collapse newlines for the row
        title = text if len(text) <= 46 else text[:45] + "…"
        is_active = entry.get("learning_state") == "active"
        learned_marker = " · learned" if is_active else ""
        item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            f"{when}  {title}{learned_marker}", None, "")
        ready = (entry.get("latency") or {}).get("release_to_text_seconds")
        ready_note = f" · ready {ready:.2f}s after release" if isinstance(ready, (int, float)) else ""
        tooltip = (f"{when} · {entry['duration']:.1f}s · "
                   f"{entry['model'].split('/')[-1]}{ready_note}\n\n{text}")
        if is_active:
            tooltip += "\n\nIn Sotto's local learning set"
        item.setToolTip_(tooltip)
        submenu = AppKit.NSMenu.alloc().init()
        submenu.setAutoenablesItems_(False)
        for label, selector in (("Copy", "copyEntry:"),
                                ("Retry transcription", "retryEntry:"),
                                ("Save audio to Desktop", "saveEntry:"),
                                ("Delete", "deleteEntry:")):
            action_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                label, selector, "")
            action_item.setTarget_(self._target)
            action_item.setRepresentedObject_(entry["id"])
            submenu.addItem_(action_item)
        correction_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Correct Transcript", "correctEntry:", "")
        correction_item.setTarget_(self._target)
        correction_item.setRepresentedObject_({"id": entry["id"], "text": entry["text"], "revision": entry.get("revision"),
                                                "adaptive_eligible": bool(entry.get("adaptive_eligible", False))})
        submenu.addItem_(correction_item)
        if show_adaptive_review_actions(adaptive_mode=self.adaptive_mode,
                                        reviewable=bool(entry.get("adaptive_eligible", False))):
            for label, selector in (("Transcript is correct", "correctAsIs:"),
                                    ("No speech / should be blank", "noSpeech:")):
                review_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(label, selector, "")
                review_item.setTarget_(self._target); review_item.setRepresentedObject_({"id": entry["id"], "revision": entry.get("revision")})
                submenu.addItem_(review_item)
            for reason, label in (("private", "Skip learning review (private)"),
                                  ("corrupt", "Skip learning review (corrupt)"),
                                  ("wrong_capture", "Skip learning review (wrong capture)")):
                review_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(label, "skipLearningReview:", "")
                review_item.setTarget_(self._target); review_item.setRepresentedObject_({"id": entry["id"], "reason": reason, "revision": entry.get("revision")})
                submenu.addItem_(review_item)
        if is_active:
            learning_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Remove from Learning Set", "revokeEntry:", "")
            learning_item.setTarget_(self._target)
            learning_item.setRepresentedObject_(entry["id"])
            submenu.addItem_(learning_item)
        elif show_manual_enroll_action(adaptive_mode=self.adaptive_mode,
                                       corrected=bool(entry.get("correction")), active=is_active,
                                       adaptive_eligible=bool(entry.get("adaptive_eligible", False))):
            learning_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Add to Learning Set", "enrollEntry:", "")
            learning_item.setTarget_(self._target)
            learning_item.setRepresentedObject_(entry["id"])
            submenu.addItem_(learning_item)
        item.setSubmenu_(submenu)
        return item

    def correct_transcript(self, entry_id: str, current_text: str, revision: int | None = None,
                           adaptive_eligible: bool = False) -> None:
        """Present the native multi-line correction editor on the main thread."""
        alert = new_alert()
        alert.setMessageText_("Correct transcript")
        alert.setInformativeText_(correction_disclosure(self.adaptive_mode and adaptive_eligible))
        alert.addButtonWithTitle_("Save Correction")
        alert.addButtonWithTitle_("Cancel")

        scroll = AppKit.NSScrollView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, 420, 150))
        scroll.setHasVerticalScroller_(True)
        scroll.setBorderType_(AppKit.NSBezelBorder)
        editor = AppKit.NSTextView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, 420, 150))
        editor.setString_(current_text)
        editor.setMinSize_(AppKit.NSMakeSize(0, 150))
        editor.setVerticallyResizable_(True)
        editor.setHorizontallyResizable_(False)
        editor.textContainer().setWidthTracksTextView_(True)
        scroll.setDocumentView_(editor)
        alert.setAccessoryView_(scroll)

        if alert.runModal() != AppKit.NSAlertFirstButtonReturn:
            return
        corrected_text = str(editor.string()).strip()
        if corrected_text and corrected_text != str(current_text).strip():
            self._target.callbacks["correct"](entry_id, corrected_text, revision)

    def offer_dictionary_rules(self, rules: list, saved_message: str) -> None:
        """After a correction, offer its substitutions as dictionary rules."""
        alert = new_alert()
        alert.setMessageText_("Correction saved. Always write these your way?")
        alert.setInformativeText_(
            saved_message + "\n\nChecked rules go into your dictionary and apply to every "
            "future dictation. Edit or remove them any time from Dictionary in the menu.")
        alert.addButtonWithTitle_("Add to Dictionary")
        alert.addButtonWithTitle_("Not Now")
        row = 24
        view = AppKit.NSView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, 420, row * len(rules)))
        boxes = []
        for index, (heard, write) in enumerate(rules):
            box = AppKit.NSButton.alloc().initWithFrame_(
                AppKit.NSMakeRect(0, row * (len(rules) - 1 - index), 420, row))
            box.setButtonType_(AppKit.NSButtonTypeSwitch)
            box.setTitle_(f"“{heard}” → “{write}”")
            box.setState_(AppKit.NSControlStateValueOn)
            view.addSubview_(box)
            boxes.append((box, heard, write))
        alert.setAccessoryView_(view)
        if alert.runModal() != AppKit.NSAlertFirstButtonReturn:
            return
        chosen = [(heard, write) for box, heard, write in boxes
                  if box.state() == AppKit.NSControlStateValueOn]
        if chosen:
            self._target.callbacks["add_rules"](chosen)

    def confirm_enrollment(self, entry_id: str) -> None:
        """Legacy manual enrollment control for existing callers."""
        alert = new_alert()
        alert.setMessageText_("Add corrected sample to learning set?")
        alert.setInformativeText_(
            "The corrected transcript and audio will be copied into Sotto's "
            "local learning corpus and are never uploaded automatically.")
        alert.addButtonWithTitle_("Add to Learning Set")
        alert.addButtonWithTitle_("Cancel")
        if alert.runModal() == AppKit.NSAlertFirstButtonReturn:
            self._target.callbacks["enroll"](entry_id)

    def confirm_revocation(self, entry_id: str) -> None:
        """Confirm removal of the local copied learning audio."""
        alert = new_alert()
        alert.setMessageText_("Remove from learning set?")
        alert.setInformativeText_(
            "The local copied learning audio for this sample will be deleted.")
        alert.addButtonWithTitle_("Remove from Learning Set")
        alert.addButtonWithTitle_("Cancel")
        if alert.runModal() == AppKit.NSAlertFirstButtonReturn:
            self._target.callbacks["revoke"](entry_id)

    def show_error(self, title: str, message: str) -> None:
        """Show a native error alert. Call this method on AppKit's main thread.
        Error text comes from Sotto itself, so it is also logged."""
        if self.on_overlay_event is not None:
            self.on_overlay_event(f"! {title}: {message}")
        self._alert(AppKit.NSAlertStyleCritical, title, message)

    def show_info(self, title: str, message: str) -> None:
        """Show a native informational alert. Call this method on the main thread.
        Not logged: an info message can quote dictated text."""
        self._alert(AppKit.NSAlertStyleInformational, title, message)

    def ask(self, title: str, message: str, yes: str, no: str) -> bool:
        """A two-button question on the main thread; True for the first button."""
        alert = new_alert()
        alert.setMessageText_(title)
        alert.setInformativeText_(message)
        alert.addButtonWithTitle_(yes)
        alert.addButtonWithTitle_(no)
        AppKit.NSApp.activateIgnoringOtherApps_(True)
        return alert.runModal() == AppKit.NSAlertFirstButtonReturn

    @staticmethod
    def _alert(style, title: str, message: str) -> None:
        alert = new_alert()
        alert.setAlertStyle_(style)
        alert.setMessageText_(title)
        alert.setInformativeText_(message)
        alert.addButtonWithTitle_("OK")
        # A menu bar app is rarely active: without this the alert can open
        # behind the frontmost window and look like a frozen menu.
        AppKit.NSApp.activateIgnoringOtherApps_(True)
        alert.runModal()

    # -- states ------------------------------------------------------------

    def show_recording(self) -> None:
        self._mode = "recording"
        self._indicator_state = "recording"
        self._display_level = 0.0
        self._apply_indicator_state()
        self._show_panel()
        if self._level_timer is None and self.level_source is not None:
            self._level_timer = (
                AppKit.NSTimer
                .scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                    0.05, self._target, "levelTick:", None, True))

    def show_waking(self) -> None:
        """Cold mic start: blink until audio actually flows, so the user knows
        not to speak yet (a cold AVAudioEngine can take seconds)."""
        self._mode = "recording"
        self._indicator_state = "waking"
        self._apply_indicator_state()
        self._show_panel()

    def show_hands_free(self, hint: str) -> None:
        """Hands-free keeps recording after the key is up: widen the island
        to say how to stop it, until the recording ends."""
        self._hint = hint
        if self._mode != "idle":
            self._position()

    def refresh_live_text(self) -> None:
        """Show the streaming engine's words while recording (display only)."""
        source = getattr(self, "live_text_source", None)
        if source is None or self._indicator_state != "recording":
            return
        now = time.monotonic()
        if now - self._live_checked < LIVE_REFRESH_S:
            return
        self._live_checked = now
        text = source() or ""
        if text != self._live_text:
            self._live_text = text
            self._position()

    def show_transcribing(self) -> None:
        self._mode = "transcribing"
        self._hint = None
        self._live_text = ""
        self._indicator_state = "transcribing"
        self._stop_level_timer()
        self._apply_indicator_state()
        # Do not assume the recording callback made the panel visible.  A cold
        # mic can finish before its delayed live announcement, and the main
        # queue may coalesce state changes under load.  Transcribing is still a
        # real user-visible state and must bring the indicator forward itself.
        self._show_panel()

    def hide(self) -> None:
        self._mode = "idle"
        self._hint = None
        self._live_text = ""
        self._indicator_state = "idle"
        self._visibility_generation += 1
        self._stop_level_timer()
        self._status.button().setTitle_(self._idle_glyph())
        self._orb.removeAnimationForKey_("pulse")
        self._panel.orderOut_(None)

    def _stop_level_timer(self) -> None:
        if self._level_timer is not None:
            self._level_timer.invalidate()
            self._level_timer = None

    def hide_if_transcribing(self) -> None:
        """Worker-completion cleanup that must never kill a live recording."""
        if self._mode == "transcribing":
            self.hide()

    def reposition_if_visible(self) -> None:
        if self._panel.isVisible():
            self._position()

    def push_level(self, rms: float) -> None:
        # Fast attack, slow decay — speech lifts the orb instantly, silence
        # lets it settle instead of flickering block-to-block.
        shaped = min(1.0, math.sqrt(max(0.0, rms)) * 3.2)
        self._display_level = max(shaped, self._display_level * DECAY)
        self._set_orb(self._display_level, animated=True)

    # -- drawing -----------------------------------------------------------

    def _apply_indicator_state(self) -> None:
        """Restore the current visual state, including after panel recovery."""
        self._orb.removeAnimationForKey_("pulse")
        self._orb.setOpacity_(1.0)
        if self._indicator_state == "recording":
            self._status.button().setTitle_("●")
            self._orb.setBackgroundColor_(_color(EMBER).CGColor())
            self._orb.setShadowColor_(_color(EMBER).CGColor())
            self._set_orb(self._display_level, animated=False)
        elif self._indicator_state == "waking":
            self._status.button().setTitle_("◍")
            self._orb.setBackgroundColor_(_color(PALE).CGColor())
            self._orb.setShadowColor_(_color(PALE).CGColor())
            blink = Quartz.CABasicAnimation.animationWithKeyPath_("opacity")
            blink.setFromValue_(1.0)
            blink.setToValue_(0.15)
            blink.setDuration_(0.35)
            blink.setAutoreverses_(True)
            blink.setRepeatCount_(1e9)
            self._orb.addAnimation_forKey_(blink, "pulse")
        elif self._indicator_state == "transcribing":
            self._status.button().setTitle_("…")
            self._orb.setBackgroundColor_(_color(PALE).CGColor())
            self._orb.setShadowColor_(_color(PALE).CGColor())
            pulse = Quartz.CABasicAnimation.animationWithKeyPath_("transform.scale")
            pulse.setFromValue_(0.62)
            pulse.setToValue_(1.0)
            pulse.setDuration_(0.55)
            pulse.setAutoreverses_(True)
            pulse.setRepeatCount_(1e9)
            pulse.setTimingFunction_(Quartz.CAMediaTimingFunction.functionWithName_(
                Quartz.kCAMediaTimingFunctionEaseInEaseOut))
            self._orb.addAnimation_forKey_(pulse, "pulse")

    def _set_orb(self, level: float, animated: bool) -> None:
        scale = 0.55 + 0.45 * level
        glow = 0.25 + 0.75 * level
        Quartz.CATransaction.begin()
        if animated:
            Quartz.CATransaction.setAnimationDuration_(EASE_S)
            Quartz.CATransaction.setAnimationTimingFunction_(
                Quartz.CAMediaTimingFunction.functionWithName_(
                    Quartz.kCAMediaTimingFunctionEaseOut))
        else:
            Quartz.CATransaction.setDisableActions_(True)
        self._orb.setTransform_(Quartz.CATransform3DMakeScale(scale, scale, 1.0))
        self._orb.setShadowOpacity_(glow)
        Quartz.CATransaction.commit()

    def _make_island(self) -> AppKit.NSPanel:
        panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            AppKit.NSMakeRect(0, 0, ISLAND_W, ISLAND_H),
            AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel,
            AppKit.NSBackingStoreBuffered, False)
        panel.setFloatingPanel_(True)
        panel.setLevel_(AppKit.NSStatusWindowLevel)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(AppKit.NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setIgnoresMouseEvents_(True)
        panel.setHidesOnDeactivate_(False)
        panel.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorStationary
            | AppKit.NSWindowCollectionBehaviorIgnoresCycle
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary)

        island = AppKit.NSView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, ISLAND_W, ISLAND_H))
        island.setWantsLayer_(True)
        island.layer().setBackgroundColor_(
            AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.075, 0.06, 0.05, 0.85).CGColor())
        island.layer().setCornerRadius_(ISLAND_H / 2)
        island.layer().setBorderWidth_(0.5)
        island.layer().setBorderColor_(_color(EMBER, 0.25).CGColor())
        panel.setContentView_(island)
        return panel

    def _make_orb(self) -> Quartz.CALayer:
        orb = Quartz.CALayer.layer()
        orb.setBounds_(Quartz.CGRectMake(0, 0, ORB_SIZE, ORB_SIZE))
        orb.setPosition_(Quartz.CGPointMake(ISLAND_W / 2, ISLAND_H / 2))
        orb.setCornerRadius_(ORB_SIZE / 2)
        orb.setBackgroundColor_(_color(EMBER).CGColor())
        orb.setShadowColor_(_color(EMBER).CGColor())
        orb.setShadowRadius_(14.0)
        orb.setShadowOpacity_(0.55)
        orb.setShadowOffset_(Quartz.CGSizeMake(0, 0))
        self._panel.contentView().layer().addSublayer_(orb)
        return orb

    def _show_panel(self, recovery_attempted: bool = False) -> None:
        """Raise the indicator and verify WindowServer actually made it visible."""
        self._visibility_generation += 1
        generation = self._visibility_generation
        self._position()
        self._panel.setLevel_(AppKit.NSStatusWindowLevel)
        self._panel.orderFrontRegardless()
        self._panel.displayIfNeeded()
        AppHelper.callLater(0.1, self._verify_panel_visible,
                            generation, recovery_attempted)

    def _verify_panel_visible(self, generation: int,
                              recovery_attempted: bool) -> None:
        if generation != self._visibility_generation or self._mode == "idle":
            return
        rows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionIncludingWindow,
            self._panel.windowNumber()) or []
        visible = window_info_is_visible(rows) if rows else self._panel.isVisible()
        if visible:
            return
        if recovery_attempted:
            self._report_overlay("! overlay recovery failed — panel still hidden")
            return
        self._report_overlay("! overlay hidden after raise — rebuilding panel")
        self._panel.orderOut_(None)
        self._panel = self._make_island()
        self._orb = self._make_orb()
        self._apply_indicator_state()
        self._show_panel(recovery_attempted=True)

    def _report_overlay(self, message: str) -> None:
        if self.on_overlay_event is not None:
            self.on_overlay_event(message)

    def _position(self) -> None:
        screens = tuple(AppKit.NSScreen.screens())
        screen = screen_containing_point(screens, AppKit.NSEvent.mouseLocation())
        if screen is None:
            screen = AppKit.NSScreen.mainScreen() or (screens[0] if screens else None)
        if screen is None:
            return
        visible = screen.visibleFrame()
        width = self._layout_island()
        x = visible.origin.x + (visible.size.width - width) / 2
        y = visible.origin.y + visible.size.height - ISLAND_H - 8
        self._panel.setFrame_display_(
            AppKit.NSMakeRect(x, y, width, ISLAND_H), True)
        screen_id = screen.deviceDescription().get("NSScreenNumber")
        if screen_id != self._last_screen_id:
            self._last_screen_id = screen_id
            self._report_overlay(
                f"  overlay: display {screen_id} at {int(x)},{int(y)}")


    def _layout_island(self) -> float:
        """Place the orb (and the hands-free hint) and return the island width.
        Runs on every position, so a rebuilt panel gets the same layout."""
        live = getattr(self, "_live_text", "")
        hint = live_snippet(live) if live else getattr(self, "_hint", None)
        root = self._panel.contentView().layer()
        layer = getattr(self, "_hint_layer", None)
        Quartz.CATransaction.begin()
        Quartz.CATransaction.setDisableActions_(True)
        if not hint:
            self._orb.setPosition_(Quartz.CGPointMake(ISLAND_W / 2, ISLAND_H / 2))
            if layer is not None:
                layer.setHidden_(True)
            Quartz.CATransaction.commit()
            return ISLAND_W
        if layer is None or layer.superlayer() != root:
            layer = Quartz.CATextLayer.layer()
            layer.setFont_(AppKit.NSFont.systemFontOfSize_(HINT_FONT_SIZE))
            layer.setFontSize_(HINT_FONT_SIZE)
            layer.setForegroundColor_(_color(PALE, 0.9).CGColor())
            screen = AppKit.NSScreen.mainScreen()
            layer.setContentsScale_(screen.backingScaleFactor() if screen else 2.0)
            root.addSublayer_(layer)
            self._hint_layer = layer
        font = AppKit.NSFont.systemFontOfSize_(HINT_FONT_SIZE)
        size = AppKit.NSAttributedString.alloc().initWithString_attributes_(
            hint, {AppKit.NSFontAttributeName: font}).size()
        layer.setString_(hint)
        layer.setFrame_(Quartz.CGRectMake(ISLAND_H, (ISLAND_H - size.height) / 2,
                                          size.width + 2, size.height))
        layer.setHidden_(False)
        self._orb.setPosition_(Quartz.CGPointMake(ISLAND_H / 2 + 3, ISLAND_H / 2))
        Quartz.CATransaction.commit()
        return ISLAND_H + size.width + 14


def run_loop() -> None:
    AppHelper.runEventLoop()

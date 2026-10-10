"""Sotto's Settings window (macOS): every choice in one place, applied live.

The window only renders state and reports changes. A small model object owned
by the running app supplies ``state()`` and applies ``change(key, value)`` and
the named actions, so nothing here touches capture, models or storage directly.
"""
from __future__ import annotations

import AppKit
import objc

HOTKEY_SPEC = "ctrl-opt-d"
INSERT_LABELS = {"paste": "Paste (your clipboard is restored)",
                 "type": "Type it (clipboard untouched)"}
SPACING_LABELS = {"smart": "Smart: a space only where needed",
                  "trailing": "Always add a space after",
                  "none": "No automatic spaces"}
SPEED_LABELS = {"accurate": "Accurate (default)", "fast": "Fast (experimental, about 3x quicker)"}
WIDTH, LABEL_W, CONTROL_X = 560, 170, 190
CONTROL_W = WIDTH - CONTROL_X - 24


class _Flipped(AppKit.NSView):
    def isFlipped(self):
        return True


def _label(text: str, frame, *, bold: bool = False, small: bool = False):
    field = AppKit.NSTextField.labelWithString_(text)
    field.setFrame_(frame)
    if bold:
        field.setFont_(AppKit.NSFont.boldSystemFontOfSize_(13))
    elif small:
        field.setFont_(AppKit.NSFont.systemFontOfSize_(11))
        field.setTextColor_(AppKit.NSColor.secondaryLabelColor())
    return field


class SettingsController(AppKit.NSObject):
    """Owns the Settings window; every control reports through the model."""

    def initWithModel_(self, model):
        self = objc.super(SettingsController, self).init()
        if self is None:
            return None
        self.model = model
        self.window = None
        self.controls = {}
        self.language_boxes = []
        return self

    # -- building ---------------------------------------------------------

    @objc.python_method
    def show(self):
        if self.window is None:
            self._build()
        self._refresh()
        AppKit.NSApp.activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)

    @objc.python_method
    def _popup(self, view, y, options, selector, *, key):
        popup = AppKit.NSPopUpButton.alloc().initWithFrame_pullsDown_(
            AppKit.NSMakeRect(CONTROL_X, y - 3, CONTROL_W, 26), False)
        popup.setAutoenablesItems_(False)
        for value, title in options:
            popup.addItemWithTitle_(title)
            popup.lastItem().setRepresentedObject_(value)
        popup.setTarget_(self)
        popup.setAction_(selector)
        view.addSubview_(popup)
        self.controls[key] = popup
        return popup

    @objc.python_method
    def _checkbox(self, view, y, title, selector, *, key, x=CONTROL_X, width=CONTROL_W):
        box = AppKit.NSButton.alloc().initWithFrame_(AppKit.NSMakeRect(x, y, width, 22))
        box.setButtonType_(AppKit.NSButtonTypeSwitch)
        box.setTitle_(title)
        box.setTarget_(self)
        box.setAction_(selector)
        view.addSubview_(box)
        self.controls[key] = box
        return box

    @objc.python_method
    def _button(self, view, y, title, selector, *, key, x=CONTROL_X, width=220):
        button = AppKit.NSButton.alloc().initWithFrame_(AppKit.NSMakeRect(x, y - 4, width, 28))
        button.setBezelStyle_(AppKit.NSBezelStyleRounded)
        button.setTitle_(title)
        button.setTarget_(self)
        button.setAction_(selector)
        view.addSubview_(button)
        self.controls[key] = button
        return button

    @objc.python_method
    def _row_label(self, view, y, text):
        view.addSubview_(_label(text, AppKit.NSMakeRect(20, y, LABEL_W - 10, 20)))

    @objc.python_method
    def _build(self):
        state = self.model.state()
        height = 868
        self.window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            AppKit.NSMakeRect(0, 0, WIDTH, height),
            AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable,
            AppKit.NSBackingStoreBuffered, False)
        self.window.setTitle_("Sotto Settings")
        self.window.setReleasedWhenClosed_(False)
        self.window.center()
        view = _Flipped.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, WIDTH, height))
        self.window.setContentView_(view)
        y = 18

        view.addSubview_(_label("Dictation", AppKit.NSMakeRect(20, y, 300, 20), bold=True)); y += 30
        self._row_label(view, y, "Hold to dictate")
        self._popup(view, y, list(state["trigger_labels"].items()), "triggerChanged:", key="trigger"); y += 32
        self._row_label(view, y, "Also")
        self._checkbox(view, y, "Control–Option–D: hold or tap (works over remote desktop)",
                       "hotkeyToggled:", key="hotkey"); y += 30
        self.controls["locked_note"] = _label("", AppKit.NSMakeRect(CONTROL_X, y, CONTROL_W, 16), small=True)
        view.addSubview_(self.controls["locked_note"]); y += 28

        view.addSubview_(_label("Recognition", AppKit.NSMakeRect(20, y, 300, 20), bold=True)); y += 30
        self._row_label(view, y, "Speech engine")
        self._popup(view, y, list(state["engine_labels"].items()), "engineChanged:", key="engine"); y += 32
        self._row_label(view, y, "")
        self._button(view, y, "Install Nemotron (700 MB)…", "installNemotron:", key="install_nemotron",
                     width=CONTROL_W); y += 32
        self._button(view, y, "Download Parakeet (2.5 GB)…", "installParakeet:", key="install_parakeet",
                     width=CONTROL_W); y += 34
        self._row_label(view, y, "Whisper speed")
        self._popup(view, y, list(SPEED_LABELS.items()), "speedChanged:", key="speed"); y += 32
        self._row_label(view, y, "Automatic language")
        view.addSubview_(_label("With two or more, clearly other speech is written as spoken.",
                                AppKit.NSMakeRect(CONTROL_X, y + 2, CONTROL_W, 16), small=True)); y += 24
        languages = sorted(state["language_names"].items(), key=lambda item: item[1])
        scroll = AppKit.NSScrollView.alloc().initWithFrame_(AppKit.NSMakeRect(CONTROL_X, y, CONTROL_W, 150))
        scroll.setHasVerticalScroller_(True)
        scroll.setBorderType_(AppKit.NSBezelBorder)
        list_view = _Flipped.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, CONTROL_W - 20, 22 * len(languages)))
        for index, (code, name) in enumerate(languages):
            box = AppKit.NSButton.alloc().initWithFrame_(AppKit.NSMakeRect(6, index * 22, CONTROL_W - 30, 22))
            box.setButtonType_(AppKit.NSButtonTypeSwitch)
            box.setTitle_(name)
            box.setTarget_(self)
            box.setAction_("languageToggled:")
            list_view.addSubview_(box)
            self.language_boxes.append((code, box))
        scroll.setDocumentView_(list_view)
        view.addSubview_(scroll)
        self.controls["languages"] = scroll
        y += 162

        view.addSubview_(_label("Inserting text", AppKit.NSMakeRect(20, y, 300, 20), bold=True)); y += 30
        self._row_label(view, y, "Insert by")
        self._popup(view, y, list(INSERT_LABELS.items()), "insertChanged:", key="insert_mode"); y += 32
        self._row_label(view, y, "Spacing")
        self._popup(view, y, list(SPACING_LABELS.items()), "spacingChanged:", key="spacing"); y += 32
        self._row_label(view, y, "Cleanup")
        self._checkbox(view, y, "Remove filler words (um, uh)", "fillersToggled:", key="remove_fillers"); y += 24
        self._checkbox(view, y, "Voice commands", "commandsToggled:", key="voice_commands"); y += 22
        view.addSubview_(_label("Say “new line”, “new paragraph” or “scratch that”. English only.",
                                AppKit.NSMakeRect(CONTROL_X, y, CONTROL_W, 16), small=True)); y += 30

        view.addSubview_(_label("General", AppKit.NSMakeRect(20, y, 300, 20), bold=True)); y += 30
        self._row_label(view, y, "Login")
        self._checkbox(view, y, "Start Sotto when I log in", "loginToggled:", key="launch_at_login"); y += 24
        self.controls["login_note"] = _label("", AppKit.NSMakeRect(CONTROL_X, y, CONTROL_W, 16), small=True)
        view.addSubview_(self.controls["login_note"]); y += 26
        self._row_label(view, y, "Dictionary")
        self._button(view, y, "Open Dictionary…", "openDictionary:", key="dictionary"); y += 34
        self._row_label(view, y, "Your data")
        self._button(view, y, "Show in Finder", "showDataFolder:", key="data_folder"); y += 30
        self.controls["data_path"] = _label("", AppKit.NSMakeRect(CONTROL_X, y, CONTROL_W, 16), small=True)
        self.controls["data_path"].setLineBreakMode_(AppKit.NSLineBreakByTruncatingMiddle)
        view.addSubview_(self.controls["data_path"]); y += 30
        view.addSubview_(_label("Changes apply to your next dictation. Switching the speech engine restarts Sotto.",
                                AppKit.NSMakeRect(20, y, WIDTH - 40, 16), small=True))

    # -- state ------------------------------------------------------------

    @objc.python_method
    def _select(self, key, value):
        popup = self.controls[key]
        for index in range(popup.numberOfItems()):
            if popup.itemAtIndex_(index).representedObject() == value:
                popup.selectItemAtIndex_(index)
                return

    @objc.python_method
    def _refresh(self):
        state = self.model.state()
        prefs = state["settings"]
        locked = state["locked"]
        self._select("trigger", state["trigger"])
        self.controls["trigger"].setEnabled_("trigger" not in locked)
        self.controls["hotkey"].setState_(AppKit.NSControlStateValueOn if state["hotkey"]
                                          else AppKit.NSControlStateValueOff)
        self.controls["hotkey"].setEnabled_("hotkey" not in locked)
        self.controls["locked_note"].setStringValue_(
            "Set by a command-line flag for this run." if locked & {"trigger", "hotkey"} else "")
        self._select("engine", state["engine"])
        self.controls["engine"].setEnabled_(state["engine_switching"])
        for index in range(self.controls["engine"].numberOfItems()):
            item = self.controls["engine"].itemAtIndex_(index)
            item.setEnabled_({"nemotron": state["nemotron_installed"],
                              "parakeet": state.get("parakeet_installed", False)}
                             .get(item.representedObject(), True))
        self.controls["install_nemotron"].setHidden_(state["nemotron_installed"])
        self.controls["install_nemotron"].setEnabled_(not state["nemotron_installing"])
        self.controls["install_nemotron"].setTitle_(
            "Installing Nemotron…" if state["nemotron_installing"] else "Install Nemotron (700 MB)…")
        self.controls["install_parakeet"].setHidden_(state.get("parakeet_installed", False))
        self.controls["install_parakeet"].setEnabled_(not state.get("parakeet_installing", False))
        self.controls["install_parakeet"].setTitle_(
            "Downloading Parakeet…" if state.get("parakeet_installing") else "Download Parakeet (2.5 GB)…")
        self._select("speed", prefs["speed"])
        self.controls["speed"].setEnabled_(state["fast_available"] and state["engine"] == "whisper")
        chosen = set(state["automatic_languages"])
        for code, box in self.language_boxes:
            box.setState_(AppKit.NSControlStateValueOn if code in chosen else AppKit.NSControlStateValueOff)
        self._select("insert_mode", prefs["insert_mode"])
        self._select("spacing", prefs["spacing"])
        for key in ("remove_fillers", "voice_commands"):
            self.controls[key].setState_(AppKit.NSControlStateValueOn if prefs.get(key, True)
                                         else AppKit.NSControlStateValueOff)
        self.controls["launch_at_login"].setState_(AppKit.NSControlStateValueOn if state["login_enabled"]
                                                   else AppKit.NSControlStateValueOff)
        self.controls["launch_at_login"].setEnabled_(state["login_available"])
        self.controls["login_note"].setStringValue_(
            "" if state["login_available"] else "Available when Sotto is installed as Sotto.app.")
        count = state["dictionary_rules"]
        self.controls["dictionary"].setTitle_(f"Open Dictionary ({count} rule{'' if count == 1 else 's'})…")
        self.controls["data_path"].setStringValue_(state["data_dir"])

    @objc.python_method
    def _apply(self, key, value):
        error = self.model.change(key, value)
        if error:
            import ui
            alert = ui.new_alert()
            alert.setMessageText_("Could not change that setting")
            alert.setInformativeText_(str(error)[:200])
            alert.runModal()
        self._refresh()

    # -- actions ----------------------------------------------------------

    def triggerChanged_(self, sender):
        self._apply("trigger", sender.selectedItem().representedObject())

    def hotkeyToggled_(self, sender):
        self._apply("hotkey", HOTKEY_SPEC if sender.state() == AppKit.NSControlStateValueOn else "")

    def engineChanged_(self, sender):
        self._apply("engine", sender.selectedItem().representedObject())

    def speedChanged_(self, sender):
        self._apply("speed", sender.selectedItem().representedObject())

    def languageToggled_(self, sender):
        chosen = [code for code, box in self.language_boxes if box.state() == AppKit.NSControlStateValueOn]
        if not chosen:  # Automatic needs at least one language
            sender.setState_(AppKit.NSControlStateValueOn)
            return
        self._apply("languages", chosen)

    def insertChanged_(self, sender):
        self._apply("insert_mode", sender.selectedItem().representedObject())

    def spacingChanged_(self, sender):
        self._apply("spacing", sender.selectedItem().representedObject())

    def fillersToggled_(self, sender):
        self._apply("remove_fillers", sender.state() == AppKit.NSControlStateValueOn)

    def commandsToggled_(self, sender):
        self._apply("voice_commands", sender.state() == AppKit.NSControlStateValueOn)

    def loginToggled_(self, sender):
        self._apply("launch_at_login", sender.state() == AppKit.NSControlStateValueOn)

    def openDictionary_(self, _sender):
        self.model.action("open_dictionary")

    def showDataFolder_(self, _sender):
        self.model.action("show_data_folder")

    def installNemotron_(self, _sender):
        self.model.action("install_nemotron")
        self._refresh()

    def installParakeet_(self, _sender):
        self.model.action("install_parakeet")
        self._refresh()

    @objc.python_method
    def refresh(self):
        """Re-read state (e.g. after an install finishes); safe if never shown."""
        if self.window is not None:
            self._refresh()

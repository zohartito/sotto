"""Settings window wiring and the language menu, built headless (never shown)."""
import sys
import unittest

if sys.platform != "darwin":
    raise unittest.SkipTest("macOS-only: AppKit Settings window; Windows: tests/test_win_ui.py")

import AppKit

import settings
from settings_window import HOTKEY_SPEC, SettingsController
from speech_config import WHISPER_LANGUAGES
import ui


class FakeModel:
    def __init__(self, **overrides):
        self.changes = []
        self.actions = []
        self.overrides = overrides

    def state(self):
        state = {
            "settings": dict(settings.DEFAULTS), "trigger": "right-option", "hotkey": "ctrl-opt-d",
            "locked": set(), "trigger_labels": {"right-option": "Right Option", "right-cmd": "Right Command"},
            "engine_labels": {"whisper": "Whisper", "parakeet": "Parakeet", "nemotron": "Nemotron (English streaming)"},
            "engine": "whisper", "engine_switching": True, "nemotron_installed": False,
            "nemotron_installing": False, "fast_available": False, "language_names": WHISPER_LANGUAGES,
            "automatic_languages": ["en", "pt"], "login_enabled": False, "login_available": False,
            "dictionary_rules": 3, "data_dir": "/tmp/sotto-alpha",
        }
        state.update(self.overrides)
        return state

    def change(self, key, value):
        self.changes.append((key, value))
        return None

    def action(self, name):
        self.actions.append(name)


def select(popup, value):
    for index in range(popup.numberOfItems()):
        if popup.itemAtIndex_(index).representedObject() == value:
            popup.selectItemAtIndex_(index)
            return
    raise AssertionError(value)


class SettingsWindowTests(unittest.TestCase):
    def controller(self, **overrides):
        model = FakeModel(**overrides)
        controller = SettingsController.alloc().initWithModel_(model)
        controller._build()
        controller._refresh()
        self.addCleanup(controller.window.close)
        return controller, model

    def test_controls_report_changes_through_the_model(self):
        controller, model = self.controller()
        popup = controller.controls["trigger"]
        select(popup, "right-cmd")
        controller.triggerChanged_(popup)
        controller.controls["hotkey"].setState_(AppKit.NSControlStateValueOff)
        controller.hotkeyToggled_(controller.controls["hotkey"])
        select(controller.controls["spacing"], "none")
        controller.spacingChanged_(controller.controls["spacing"])
        select(controller.controls["insert_mode"], "type")
        controller.insertChanged_(controller.controls["insert_mode"])
        self.assertEqual(model.changes, [("trigger", "right-cmd"), ("hotkey", ""),
                                         ("spacing", "none"), ("insert_mode", "type")])
        controller.openDictionary_(None)
        controller.showDataFolder_(None)
        self.assertEqual(model.actions, ["open_dictionary", "show_data_folder"])
        self.assertIn("3 rules", controller.controls["dictionary"].title())

    def test_language_checkboxes_keep_at_least_one_language(self):
        controller, model = self.controller(automatic_languages=["en"])
        boxes = dict(controller.language_boxes)
        self.assertEqual(boxes["en"].state(), AppKit.NSControlStateValueOn)
        boxes["fr"].setState_(AppKit.NSControlStateValueOn)
        controller.languageToggled_(boxes["fr"])
        self.assertEqual(sorted(model.changes[-1][1]), ["en", "fr"])
        controller.model.overrides["automatic_languages"] = ["en"]
        controller._refresh()
        boxes["en"].setState_(AppKit.NSControlStateValueOff)
        controller.languageToggled_(boxes["en"])
        self.assertEqual(boxes["en"].state(), AppKit.NSControlStateValueOn)
        self.assertEqual(len(model.changes), 1)

    def test_locked_uninstalled_and_unavailable_states_are_visible(self):
        controller, _model = self.controller(locked={"trigger"}, nemotron_installed=False)
        self.assertFalse(controller.controls["trigger"].isEnabled())
        self.assertIn("command-line flag", controller.controls["locked_note"].stringValue())
        self.assertFalse(controller.controls["install_nemotron"].isHidden())
        self.assertFalse(controller.controls["install_parakeet"].isHidden())
        parakeet = [controller.controls["engine"].itemAtIndex_(i) for i in range(controller.controls["engine"].numberOfItems())
                    if controller.controls["engine"].itemAtIndex_(i).representedObject() == "parakeet"]
        self.assertTrue(parakeet and not parakeet[0].isEnabled())
        controller.installParakeet_(None)
        self.assertEqual(controller.model.actions, ["install_parakeet"])
        engine = controller.controls["engine"]
        nemotron = [engine.itemAtIndex_(i) for i in range(engine.numberOfItems())
                    if engine.itemAtIndex_(i).representedObject() == "nemotron"][0]
        self.assertFalse(nemotron.isEnabled())
        self.assertFalse(controller.controls["launch_at_login"].isEnabled())
        self.assertFalse(controller.controls["speed"].isEnabled())
        controller.model.overrides.update(nemotron_installed=True)
        controller._refresh()
        self.assertTrue(controller.controls["install_nemotron"].isHidden())
        self.assertEqual(HOTKEY_SPEC, "ctrl-opt-d")


class LanguageMenuTests(unittest.TestCase):
    def test_default_menu_is_automatic_english_only(self):
        self.assertEqual(ui.language_menu_options("auto"),
                         (("auto", "Automatic (English)", True), ("en", "English", False)))

    def test_custom_set_and_an_active_language_outside_it(self):
        options = ui.language_menu_options("ja", ["en", "fr"])
        self.assertEqual([mode for mode, _label, _selected in options], ["auto", "en", "fr", "ja"])
        self.assertEqual(options[0][1], "Automatic (English + French)")
        self.assertTrue(options[-1][2])
        with self.assertRaises(ValueError):
            ui.language_menu_options("xx")


if __name__ == "__main__":
    unittest.main()

"""win_ui: labels and the tray menu model (no icon shown, no Tk window opened).

``TrayApp.items`` is the generator pystray calls on every right-click; here it
runs against a fake controller and the resulting pystray items are inspected
and invoked exactly as pystray would.
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from unittest import mock


def _pystray_available() -> bool:
    try:
        import pystray  # noqa: F401
    except ImportError:
        return False
    return True


if sys.platform == "win32":
    import win_ui


@unittest.skipUnless(sys.platform == "win32", "Windows-only module")
class LabelTest(unittest.TestCase):
    def test_history_labels_are_one_short_line_with_literal_ampersands(self) -> None:
        now = time.time()
        label = win_ui.history_label({"ts": now, "text": "Tom & Jerry\nsecond line " + "x" * 80}, now)
        self.assertNotIn("\n", label)
        self.assertIn("Tom && Jerry second line", label)
        self.assertTrue(label.endswith("…"))
        self.assertLessEqual(len(label), 60)
        self.assertIn("(empty)", win_ui.history_label({"ts": now, "text": ""}, now))

    def test_language_menu_matches_the_mac_plus_other_common_languages(self) -> None:
        # The Mac's ui.language_menu_options: Automatic, the chosen set, and
        # the active language when it is outside that set.
        self.assertEqual(win_ui.language_menu_options("auto", []), [
            ("auto", "Automatic (English + Hebrew)", True), ("en", "English", False),
            ("he", "Hebrew", False)])
        self.assertEqual(win_ui.language_menu_options("sw", ["he", "fr"]), [
            ("auto", "Automatic (Hebrew + French)", False), ("he", "Hebrew", False),
            ("fr", "French", False), ("sw", "Swahili", True)])
        extra = win_ui.other_languages("sw", ["he", "fr"])
        self.assertIn("en", extra)
        self.assertFalse({"he", "fr", "sw"} & set(extra))
        self.assertLessEqual(set(win_ui.COMMON_LANGUAGES), set(win_ui.WHISPER_LANGUAGES))

    def test_tooltips(self) -> None:
        self.assertEqual(win_ui.tooltip("idle", "left-option", ""), "Sotto — hold Left Alt to dictate")
        self.assertEqual(win_ui.tooltip("recording", "right-ctrl", ""), "Sotto — recording")
        self.assertEqual(win_ui.tooltip("starting", "right-ctrl", "Warming up…"), "Sotto — Warming up…")

    @unittest.skipUnless(_pystray_available(), "pystray/Pillow not installed (requirements-alpha-windows.txt)")
    def test_icon_images_differ_by_state(self) -> None:
        images = {state: win_ui.icon_image(state).getpixel((32, 50)) for state in win_ui.COLORS}
        self.assertEqual(len(set(images.values())), len(win_ui.COLORS))


class FakeController:
    def __init__(self) -> None:
        self.calls: list = []
        self.speed = "accurate"
        self.hook = mock.Mock(trigger="right-ctrl")
        self._language = "auto"
        self.done = threading.Event()

    def _record(self, *call):
        self.calls.append(call)
        self.threads = getattr(self, "threads", []) + [threading.get_ident()]
        self.done.set()

    def recording(self): return False
    def entries(self, limit): return [{"id": "abc", "ts": time.time(), "text": "hello", "revision": 0}]
    def saved_languages(self): return ["en", "he", "sw"]
    def language(self): return self._language
    def progress(self): return ["Corrections saved: 0", "Dictionary rules: 0"]
    def start_now(self): self._record("start_now")
    def copy(self, entry_id): self._record("copy", entry_id)
    def retry(self, entry_id): self._record("retry", entry_id)
    def delete(self, entry_id): self._record("delete", entry_id)
    def set_language(self, mode): self._record("set_language", mode)
    def set_speed(self, speed): self._record("set_speed", speed)
    def open_dictionary(self): self._record("open_dictionary")
    def restart(self): self._record("restart")
    def quit(self): self._record("quit")


@unittest.skipUnless(sys.platform == "win32", "Windows-only module")
@unittest.skipUnless(_pystray_available(), "pystray/Pillow not installed (requirements-alpha-windows.txt)")
class MenuTest(unittest.TestCase):
    def setUp(self) -> None:
        self.quits = []
        self.app = win_ui.TrayApp(quit=lambda: self.quits.append(1), log=lambda message: None)
        self.controller = FakeController()

    @staticmethod
    def by_text(items):
        import pystray
        return {item.text: item for item in items
                if item is not pystray.Menu.SEPARATOR and getattr(item, "text", "")}

    def invoke(self, item) -> None:
        self.controller.done.clear()
        item(None)  # what pystray does on a click
        self.assertTrue(self.controller.done.wait(5), item.text)

    def test_before_listening_only_status_and_quit(self) -> None:
        items = list(self.app.items())
        texts = [getattr(item, "text", "") for item in items]
        self.assertEqual(texts[0], "Sotto — starting…")
        self.assertFalse(items[0].enabled)
        self.by_text(items)["Quit"](None)
        self.assertEqual(self.quits, [1])

    def test_menu_has_every_section_and_actions_run_off_the_menu_thread(self) -> None:
        self.app.controller = self.controller
        menu = self.by_text(self.app.items())
        for name in ("Start dictation (hands-free)", "History", "Language", "Speed", "Progress",
                     "Dictionary", "Settings…", "Restart", "Quit"):
            self.assertIn(name, menu)
        self.assertTrue(menu["Settings…"].default)
        history = self.by_text(menu["History"].submenu.items)
        entry = next(item for text, item in history.items() if text.endswith("hello"))
        actions = self.by_text(entry.submenu.items)
        self.assertEqual(list(actions), ["Copy", "Retry", "Correct…", "Delete"])
        self.assertIn("Clear history…", history)
        with mock.patch.object(win_ui.TrayApp, "notify"):
            for name in ("Copy", "Retry", "Delete"):
                self.invoke(actions[name])
            self.invoke(menu["Start dictation (hands-free)"])
        self.assertEqual(self.controller.calls, [("copy", "abc"), ("retry", "abc"), ("delete", "abc"),
                                                 ("start_now",)])
        self.assertNotIn(threading.get_ident(), self.controller.threads,
                         "actions never run on the thread that clicked (pystray's menu thread)")
        progress = [item.text for item in menu["Progress"].submenu.items]
        self.assertEqual(progress, ["Corrections saved: 0", "Dictionary rules: 0"])
        self.assertTrue(all(not item.enabled for item in menu["Progress"].submenu.items))

    def test_language_and_speed_are_radio_groups_with_the_current_choice(self) -> None:
        self.app.controller = self.controller
        menu = self.by_text(self.app.items())
        languages = self.by_text(menu["Language"].submenu.items)
        automatic = "Automatic (English + Hebrew + Swahili)"
        self.assertEqual(list(languages), [automatic, "English", "Hebrew", "Swahili", "Other languages"])
        self.assertTrue(languages[automatic].checked)
        self.assertFalse(languages["Hebrew"].checked)
        self.controller._language = "he"
        self.assertTrue(languages["Hebrew"].checked)
        others = self.by_text(languages["Other languages"].submenu.items)
        self.assertNotIn("Hebrew", others)
        with mock.patch.object(win_ui.TrayApp, "notify"):
            self.invoke(others["French"])
        self.assertEqual(self.controller.calls[-1], ("set_language", "fr"))
        speeds = menu["Speed"].submenu.items
        self.assertEqual([item.checked for item in speeds], [True, False])
        with mock.patch.object(win_ui.TrayApp, "notify"):
            self.invoke(speeds[1])
        self.assertEqual(self.controller.calls[-1], ("set_speed", "fast"))


if __name__ == "__main__":
    unittest.main()

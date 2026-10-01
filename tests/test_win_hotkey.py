"""win_hotkey: edge detection feeding a GestureEngine (no real hook installed).

_on_press/_on_release are driven directly with pynput Key objects; the
physically-down query is patched.  The live WH_KEYBOARD_LL hook is exercised
end-to-end by sotto_win.py, not here.
"""
from __future__ import annotations

import sys
import unittest
from unittest import mock

if sys.platform == "win32":
    import pynput.keyboard as kb
    import win_hotkey


class FakeEngine:
    """Counts edges; the counters must not shadow the pressed()/released()
    method names the real GestureEngine exposes."""

    def __init__(self) -> None:
        self.press_count = 0
        self.release_count = 0
        self.chord_count = 0

    def pressed(self) -> None:
        self.press_count += 1

    def released(self) -> None:
        self.release_count += 1

    def chorded(self) -> None:
        self.chord_count += 1


@unittest.skipUnless(sys.platform == "win32", "Windows-only module")
class TriggerHookTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FakeEngine()
        self.hook = win_hotkey.TriggerHook(self.engine, trigger="right-ctrl")

    def test_press_release_edges(self) -> None:
        self.hook._on_press(kb.Key.ctrl_r)
        self.hook._on_release(kb.Key.ctrl_r)
        self.assertEqual((self.engine.press_count, self.engine.release_count), (1, 1))

    def test_auto_repeat_is_filtered(self) -> None:
        self.hook._on_press(kb.Key.ctrl_r)
        self.hook._on_press(kb.Key.ctrl_r)  # OS auto-repeat while held
        self.hook._on_press(kb.Key.ctrl_r)
        self.assertEqual(self.engine.press_count, 1)
        self.hook._on_release(kb.Key.ctrl_r)
        self.hook._on_release(kb.Key.ctrl_r)  # stray release
        self.assertEqual(self.engine.release_count, 1)

    def test_other_ctrl_twin_is_ignored(self) -> None:
        self.hook._on_press(kb.Key.ctrl_l)
        self.hook._on_release(kb.Key.ctrl_l)
        self.assertEqual((self.engine.press_count, self.engine.release_count), (0, 0))

    def test_other_key_during_a_hold_is_a_shortcut(self) -> None:
        self.hook._on_press(kb.Key.ctrl_r)
        self.hook._on_press(kb.KeyCode.from_char("c"))  # right Ctrl + C
        self.hook._on_press(kb.Key.ctrl_l)
        self.assertEqual(self.engine.chord_count, 2)
        self.hook._on_release(kb.Key.ctrl_r)
        self.hook._on_press(kb.KeyCode.from_char("c"))  # ordinary typing afterwards
        self.assertEqual(self.engine.chord_count, 2)
        self.assertEqual((self.engine.press_count, self.engine.release_count), (1, 1))

    def test_release_without_press_is_ignored(self) -> None:
        self.hook._on_release(kb.Key.ctrl_r)
        self.assertEqual((self.engine.press_count, self.engine.release_count), (0, 0))

    def test_left_ctrl_trigger_variant(self) -> None:
        hook = win_hotkey.TriggerHook(self.engine, trigger="left-ctrl")
        hook._on_press(kb.Key.ctrl_l)
        hook._on_release(kb.Key.ctrl_l)
        self.assertEqual((self.engine.press_count, self.engine.release_count), (1, 1))

    def test_unknown_trigger_rejected(self) -> None:
        with self.assertRaises(ValueError):
            win_hotkey.TriggerHook(self.engine, trigger="right-option")

    def test_physically_down_delegates_to_getasync_keystate(self) -> None:
        with mock.patch.object(win_hotkey, "get_async_key_state", return_value=True):
            self.assertTrue(self.hook.physically_down)
        with mock.patch.object(win_hotkey, "get_async_key_state", return_value=False):
            self.assertFalse(self.hook.physically_down)

    def test_not_alive_before_start(self) -> None:
        self.assertFalse(self.hook.alive())


@unittest.skipUnless(sys.platform == "win32", "Windows-only module")
class TriggerChoiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FakeEngine()
        self.masks = []

    def hook(self, trigger):
        return win_hotkey.TriggerHook(self.engine, trigger=trigger,
                                      mask_menu=lambda: self.masks.append(trigger))

    def test_windows_triggers_are_a_safe_subset_of_the_shared_settings(self) -> None:
        import settings
        self.assertLessEqual(set(win_hotkey.TRIGGERS), set(settings.TRIGGER_KEYS))
        self.assertEqual(set(win_hotkey.TRIGGERS),
                         {"right-ctrl", "left-ctrl", "left-option", "left-shift"})
        for unsafe in ("fn", "right-option", "right-shift", "left-cmd", "right-cmd"):
            self.assertEqual(win_hotkey.supported(unsafe), "right-ctrl", unsafe)
            with self.assertRaises(ValueError):
                self.hook(unsafe)
        self.assertEqual(win_hotkey.label("left-option"), "Left Alt")

    def test_left_alt_masks_the_menu_bar_once_per_press(self) -> None:
        hook = self.hook("left-option")
        hook._on_press(kb.Key.alt_l)
        hook._on_press(kb.Key.alt_l)  # auto-repeat
        hook._on_release(kb.Key.alt_l)
        self.assertEqual((self.engine.press_count, self.engine.release_count), (1, 1))
        self.assertEqual(self.masks, ["left-option"])
        shift = self.hook("left-shift")
        shift._on_press(kb.Key.shift_l)
        shift._on_release(kb.Key.shift_l)
        self.assertEqual(self.masks, ["left-option"])  # only Alt needs masking
        self.assertEqual(self.engine.press_count, 2)

    def test_set_trigger_switches_keys_live(self) -> None:
        hook = self.hook("right-ctrl")
        hook._on_press(kb.Key.ctrl_r)
        hook.set_trigger("left-shift")
        hook._on_release(kb.Key.ctrl_r)  # the old key no longer counts
        hook._on_press(kb.Key.shift_l)
        self.assertEqual((hook.trigger, self.engine.press_count, self.engine.release_count),
                         ("left-shift", 2, 0))
        with self.assertRaises(ValueError):
            hook.set_trigger("fn")

    def test_start_installs_the_own_event_filter_and_replaces_a_dead_listener(self) -> None:
        listeners = []

        class FakeListener:
            def __init__(self, **kwargs):
                self.kwargs, self.alive, self.daemon = kwargs, True, False
                listeners.append(self)

            def start(self): pass
            def stop(self): self.alive = False
            def is_alive(self): return self.alive

        hook = self.hook("right-ctrl")
        with mock.patch.object(win_hotkey.kb, "Listener", FakeListener):
            hook.start()
            hook.start()  # alive: no second hook
            self.assertEqual(len(listeners), 1)
            self.assertIs(listeners[0].kwargs["win32_event_filter"], win_hotkey.own_event)
            listeners[0].alive = False  # Windows removed the hook / the thread died
            self.assertFalse(hook.alive())
            hook.start()
        self.assertEqual(len(listeners), 2)
        self.assertTrue(hook.alive())

    def test_own_injected_events_are_filtered_before_the_callbacks(self) -> None:
        import win_inject
        ours = mock.Mock(dwExtraInfo=win_inject.INJECTED_TAG)
        theirs = mock.Mock(dwExtraInfo=0)
        remote = mock.Mock(dwExtraInfo=None)
        self.assertFalse(win_hotkey.own_event(0x100, ours))
        self.assertTrue(win_hotkey.own_event(0x100, theirs))
        self.assertTrue(win_hotkey.own_event(0x100, remote))

    def test_modifiers_held_checks_shift_ctrl_alt_and_windows(self) -> None:
        hook = self.hook("right-ctrl")
        for held in win_hotkey.MODIFIER_VKS:
            with mock.patch.object(win_hotkey, "get_async_key_state", side_effect=lambda vk: vk == held):
                self.assertTrue(hook.modifiers_held, hex(held))
        with mock.patch.object(win_hotkey, "get_async_key_state", return_value=False):
            self.assertFalse(hook.modifiers_held)
        with mock.patch.object(win_hotkey, "get_async_key_state", side_effect=lambda vk: vk == 0x12):
            self.assertFalse(hook.physically_down)  # Alt is not this trigger's key


if __name__ == "__main__":
    unittest.main()

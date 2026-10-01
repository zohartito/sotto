import unittest

import sotto


class HotkeyReleaseStateTests(unittest.TestCase):
    def test_unmatched_key_up_is_ignored_instead_of_using_machine_uptime(self):
        state = {"pressed_at": None, "skip_up": False}

        held = sotto.hotkey_release_held_seconds(state, 1_517_376.6)

        self.assertIsNone(held)
        self.assertIsNone(state["pressed_at"])

    def test_matching_key_up_returns_duration_and_consumes_press(self):
        state = {"pressed_at": 100.0, "skip_up": False}

        held = sotto.hotkey_release_held_seconds(state, 101.25)

        self.assertEqual(held, 1.25)
        self.assertIsNone(state["pressed_at"])
        self.assertIsNone(
            sotto.hotkey_release_held_seconds(state, 102.0))

    def test_toggle_stop_release_is_consumed_once(self):
        state = {"pressed_at": 200.0, "skip_up": True}

        held = sotto.hotkey_release_held_seconds(state, 200.5)

        self.assertIsNone(held)
        self.assertFalse(state["skip_up"])
        self.assertIsNone(state["pressed_at"])

    def test_stale_skip_flag_cannot_authorize_unmatched_release(self):
        state = {"pressed_at": None, "skip_up": True}

        held = sotto.hotkey_release_held_seconds(state, 300.0)

        self.assertIsNone(held)
        self.assertFalse(state["skip_up"])


if __name__ == "__main__":
    unittest.main()

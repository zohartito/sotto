from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import sys

if sys.platform != "darwin":
    raise unittest.SkipTest("AppKit-only UI — macOS-only in v1")

import AppKit
import Quartz

from ui import StatusUI, screen_containing_point, window_info_is_visible


class _Screen:
    def __init__(self, x: float, y: float, width: float, height: float) -> None:
        self._frame = AppKit.NSMakeRect(x, y, width, height)

    def frame(self):
        return self._frame


class OverlayPlacementTests(unittest.TestCase):
    def test_screen_selection_handles_stacked_displays(self):
        built_in = _Screen(0, 0, 1728, 1117)
        external = _Screen(-84, 1117, 1920, 1080)

        self.assertIs(
            screen_containing_point((built_in, external), AppKit.NSMakePoint(700, 500)),
            built_in,
        )
        self.assertIs(
            screen_containing_point((built_in, external), AppKit.NSMakePoint(700, 1600)),
            external,
        )

    def test_screen_selection_rejects_points_outside_every_display(self):
        screen = _Screen(0, 0, 1728, 1117)
        self.assertIsNone(
            screen_containing_point((screen,), AppKit.NSMakePoint(-500, -500))
        )

    def test_window_server_visibility_requires_onscreen_nonzero_alpha(self):
        self.assertTrue(window_info_is_visible([{
            Quartz.kCGWindowIsOnscreen: True,
            Quartz.kCGWindowAlpha: 1.0,
        }]))
        self.assertFalse(window_info_is_visible([{
            Quartz.kCGWindowIsOnscreen: False,
            Quartz.kCGWindowAlpha: 1.0,
        }]))
        self.assertFalse(window_info_is_visible([{
            Quartz.kCGWindowIsOnscreen: True,
            Quartz.kCGWindowAlpha: 0.0,
        }]))

    def test_hidden_live_panel_is_rebuilt_and_raised_once(self):
        status = StatusUI.__new__(StatusUI)
        status._visibility_generation = 7
        status._mode = "recording"
        old_panel = MagicMock()
        old_panel.windowNumber.return_value = 42
        replacement_panel = MagicMock()
        replacement_orb = MagicMock()
        status._panel = old_panel
        status._orb = MagicMock()
        messages = []
        status.on_overlay_event = messages.append

        hidden = [{
            Quartz.kCGWindowIsOnscreen: False,
            Quartz.kCGWindowAlpha: 1.0,
        }]
        with (patch.object(Quartz, "CGWindowListCopyWindowInfo", return_value=hidden),
              patch.object(status, "_make_island", return_value=replacement_panel),
              patch.object(status, "_make_orb", return_value=replacement_orb),
              patch.object(status, "_apply_indicator_state") as apply_state,
              patch.object(status, "_show_panel") as show_panel):
            status._verify_panel_visible(7, recovery_attempted=False)

        old_panel.orderOut_.assert_called_once_with(None)
        self.assertIs(status._panel, replacement_panel)
        self.assertIs(status._orb, replacement_orb)
        apply_state.assert_called_once_with()
        show_panel.assert_called_once_with(recovery_attempted=True)
        self.assertEqual(messages, ["! overlay hidden after raise — rebuilding panel"])

    def test_recovery_does_not_loop_if_replacement_stays_hidden(self):
        status = StatusUI.__new__(StatusUI)
        status._visibility_generation = 8
        status._mode = "recording"
        status._panel = MagicMock()
        status._panel.windowNumber.return_value = 43
        messages = []
        status.on_overlay_event = messages.append

        hidden = [{
            Quartz.kCGWindowIsOnscreen: False,
            Quartz.kCGWindowAlpha: 1.0,
        }]
        with patch.object(Quartz, "CGWindowListCopyWindowInfo", return_value=hidden):
            status._verify_panel_visible(8, recovery_attempted=True)

        status._panel.orderOut_.assert_not_called()
        self.assertEqual(messages, ["! overlay recovery failed — panel still hidden"])


class AlertIconTests(unittest.TestCase):
    """Sotto.app runs Python as a child, and AppKit loads Python's bundle icon
    when the app finishes launching, so dialogs showed Python's rocket even
    after use_app_icon. Every dialog carries Sotto's icon itself."""

    def test_every_dialog_is_built_with_sottos_icon(self):
        import tempfile
        from pathlib import Path
        import ui
        with tempfile.TemporaryDirectory() as folder:
            icon = Path(folder) / "Sotto.png"
            image = AppKit.NSImage.alloc().initWithSize_((16, 16))
            image.lockFocus(); AppKit.NSColor.orangeColor().set()
            AppKit.NSBezierPath.fillRect_(((0, 0), (16, 16))); image.unlockFocus()
            rep = AppKit.NSBitmapImageRep.imageRepWithData_(image.TIFFRepresentation())
            rep.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, {}).writeToFile_atomically_(
                str(icon), True)
            ui.use_app_icon(icon)
        alert = ui.new_alert()
        self.assertIsNotNone(alert.icon())
        self.assertEqual(tuple(alert.icon().size()), (16.0, 16.0))

    def test_no_dialog_is_built_without_it(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        counts = {name: (root / name).read_text(encoding="utf-8").count("NSAlert.alloc().init()")
                  for name in ("ui.py", "settings_window.py")}
        self.assertEqual(counts, {"ui.py": 1, "settings_window.py": 0}, "only new_alert() builds an NSAlert")


if __name__ == "__main__":
    unittest.main()

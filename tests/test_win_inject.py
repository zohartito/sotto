"""win_inject: SendInput layout, spacing, type/paste routing, clipboard restore.

No real keystrokes are sent: SendInput is patched and inputs are asserted
structurally.  The paste logic runs against a fake clipboard; the real Win32
clipboard code runs in a child process moved to a private window station, so
it never touches the user's clipboard.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock

if sys.platform == "win32":
    import win_inject

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == "win32", "Windows-only module")
class WinInjectTest(unittest.TestCase):
    def test_input_struct_has_exact_win32_size(self) -> None:
        # x64 Win32 layout: 4 (type) + 4 (padding) + 32 (union = MOUSEINPUT,
        # whose ULONG_PTR dwExtraInfo lands at offset 24).  28 on 32-bit.
        self.assertEqual(ctypes.sizeof(win_inject._INPUT), 40)

    def test_build_inputs_down_up_pairs(self) -> None:
        inputs = win_inject.build_inputs("ab")
        self.assertEqual(len(inputs), 4)
        scans = [i.u.ki.wScan for i in inputs]
        flags = [i.u.ki.dwFlags for i in inputs]
        self.assertEqual(scans, [0x61, 0x61, 0x62, 0x62])
        self.assertEqual(flags, [
            win_inject.KEYEVENTF_UNICODE,
            win_inject.KEYEVENTF_UNICODE | win_inject.KEYEVENTF_KEYUP,
            win_inject.KEYEVENTF_UNICODE,
            win_inject.KEYEVENTF_UNICODE | win_inject.KEYEVENTF_KEYUP,
        ])
        self.assertTrue(all(i.type == win_inject.INPUT_KEYBOARD for i in inputs))
        self.assertTrue(all(i.u.ki.wVk == 0 for i in inputs))

    def test_astral_characters_become_surrogate_pairs(self) -> None:
        # U+1F600 -> UTF-16 D83D DE00: two units, four events, in order.
        inputs = win_inject.build_inputs("\U0001F600")
        self.assertEqual([i.u.ki.wScan for i in inputs],
                         [0xD83D, 0xD83D, 0xDE00, 0xDE00])

    def test_inject_appends_trailing_space(self) -> None:
        with mock.patch.object(win_inject, "send_inputs") as send:
            win_inject.inject("hi")
        args, _ = send.call_args
        self.assertEqual([i.u.ki.wScan for i in args[0]],
                         [0x68, 0x68, 0x69, 0x69, 0x20, 0x20])

    def test_inject_no_trailing_space_when_text_ends_with_space(self) -> None:
        # "hi " is three chars (h, i, space) -> six events, no added space.
        with mock.patch.object(win_inject, "send_inputs") as send:
            win_inject.inject("hi ")
        args, _ = send.call_args
        self.assertEqual(len(args[0]), 6)

    def test_inject_empty_text_sends_nothing(self) -> None:
        with mock.patch.object(win_inject, "send_inputs") as send:
            win_inject.inject("")
        send.assert_not_called()

    def test_send_inputs_chunks_and_reports_short_writes(self) -> None:
        inputs = win_inject.build_inputs("x" * 300)  # 600 events
        calls = []

        def fake_send(count, array, size):
            calls.append(count)
            return count

        with mock.patch("ctypes.windll.user32.SendInput", side_effect=fake_send):
            win_inject.send_inputs(inputs)
        self.assertEqual(calls, [256, 256, 88])

    def test_send_inputs_raises_on_short_write(self) -> None:
        inputs = win_inject.build_inputs("x")
        with mock.patch("ctypes.windll.user32.SendInput", return_value=0):
            with self.assertRaises(OSError):
                win_inject.send_inputs(inputs)

    def test_every_injected_event_carries_the_sotto_tag(self) -> None:
        # win_hotkey drops exactly these, so Ctrl+V can never look like a trigger.
        events = win_inject.build_inputs("a") + win_inject.chord_inputs(win_inject.VK_CONTROL, win_inject.VK_V)
        self.assertTrue(all(event.u.ki.dwExtraInfo == win_inject.INJECTED_TAG for event in events))
        self.assertEqual([(e.u.ki.wVk, e.u.ki.dwFlags) for e in events[2:]],
                         [(0x11, 0), (0x56, 0), (0x56, win_inject.KEYEVENTF_KEYUP),
                          (0x11, win_inject.KEYEVENTF_KEYUP)])


@unittest.skipUnless(sys.platform == "win32", "Windows-only module")
class SpacingAndRoutingTest(unittest.TestCase):
    def test_spacing_modes(self) -> None:
        for spacing in ("smart", "trailing"):
            self.assertEqual(win_inject.spaced("hi", spacing), "hi ")
            self.assertEqual(win_inject.spaced("hi\n", spacing), "hi\n")
            self.assertEqual(win_inject.spaced("", spacing), "")
        self.assertEqual(win_inject.spaced("hi", "none"), "hi")
        with self.assertRaises(ValueError):
            win_inject.spaced("hi", "double")

    @staticmethod
    def _typed(send) -> str:
        units = [event.u.ki.wScan for call in send.call_args_list for event in call.args[0]
                 if not event.u.ki.dwFlags & win_inject.KEYEVENTF_KEYUP]
        return "".join(map(chr, units))

    def test_type_mode_types_and_never_touches_the_clipboard(self) -> None:
        paster = mock.Mock()
        with mock.patch.object(win_inject, "send_inputs") as send, \
                mock.patch.object(win_inject, "paster", return_value=paster):
            self.assertEqual(win_inject.deliver("hello", mode="type", spacing="smart"), "type")
        self.assertEqual(self._typed(send), "hello ")
        paster.paste.assert_not_called()

    def test_paste_mode_pastes_and_falls_back_to_typing(self) -> None:
        paster = mock.Mock()
        paster.paste.return_value = True
        with mock.patch.object(win_inject, "send_inputs") as send, \
                mock.patch.object(win_inject, "paster", return_value=paster):
            self.assertEqual(win_inject.deliver("hello", mode="paste", spacing="none"), "paste")
            send.assert_not_called()
            paster.paste.assert_called_once_with("hello")
            paster.paste.return_value = False  # the clipboard can't be preserved
            self.assertEqual(win_inject.deliver("hello", mode="paste", spacing="trailing"), "type")
        self.assertEqual(self._typed(send), "hello ")
        with self.assertRaises(ValueError):
            win_inject.deliver("x", mode="shout")


class FakeClipboard:
    """The _ClipboardWindow surface; ``rendered`` is set when the "target app" reads."""

    def __init__(self) -> None:
        self.contents = [(13, "user\0".encode("utf-16-le"))]
        self.markers: tuple = ()
        self.seq = 100
        self.unrestorable = self.busy = self.fail_offer = False
        self.fail_restores = 0
        self.writes: list = []
        self.rendered = threading.Event()

    def sequence(self):
        return self.seq

    def snapshot(self):
        if self.busy:
            raise win_inject.ClipboardBusy("busy")
        return None if self.unrestorable else list(self.contents)

    def write(self, items, markers=None):
        self.seq += 1
        self.contents = list(items)
        self.markers = tuple(win_inject.RESTORE_MARKERS if markers is None else markers)
        self.writes.append(list(items))
        return self.seq

    def restore_if_unchanged(self, items, own_sequence):
        if self.fail_restores:
            self.fail_restores -= 1
            raise win_inject.ClipboardBusy("the clipboard is busy")
        if self.seq != own_sequence:
            return False
        self.write(items)
        return True

    def offer_text(self, text):
        if self.fail_offer:
            self.contents = []  # emptied, then the offer failed
            raise OSError("could not offer text on the clipboard")
        self.rendered.clear()
        self.seq += 1
        self.contents, self.markers = [(13, (text + "\0").encode("utf-16-le"))], win_inject.OFFER_MARKERS
        return self.seq

    def text(self):
        return dict(self.contents)[13].decode("utf-16-le").rstrip("\0")


@unittest.skipUnless(sys.platform == "win32", "Windows-only module")
class ClipboardPasterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clipboard = FakeClipboard()
        self.sent: list = []
        self.watchers: list = []
        self.paster = win_inject.ClipboardPaster(self.clipboard, send=self.sent.append,
                                                 render_wait=0.05, grace=0.0,
                                                 spawn=self.watchers.append, sleep=lambda s: None)

    def target_reads_and_watch(self, index: int = -1) -> None:
        self.clipboard.rendered.set()  # WM_RENDERFORMAT: the app read the text
        self.watchers[index]()

    def test_paste_offers_the_text_sends_ctrl_v_and_restores_after_the_app_reads(self) -> None:
        self.assertTrue(self.paster.paste("dictated "))
        self.assertEqual(self.clipboard.text(), "dictated ")
        self.assertEqual(self.clipboard.markers, win_inject.OFFER_MARKERS)
        self.assertEqual([[e.u.ki.wVk for e in chord] for chord in self.sent], [[0x11, 0x56, 0x56, 0x11]])
        self.assertTrue(all(e.u.ki.wScan for e in self.sent[0]), "real scan codes")
        self.target_reads_and_watch()
        self.assertEqual(self.clipboard.text(), "user")
        self.assertEqual(self.clipboard.markers, win_inject.RESTORE_MARKERS)

    def test_an_app_that_never_reads_still_gets_the_clipboard_back(self) -> None:
        self.paster.paste("dictated ")
        started = time.monotonic()
        self.watchers[0]()  # render_wait elapses without a read
        self.assertGreaterEqual(time.monotonic() - started, 0.04)
        self.assertEqual(self.clipboard.text(), "user")

    def test_a_copy_made_meanwhile_wins_over_the_restore(self) -> None:
        self.paster.paste("dictated ")
        self.clipboard.write([(13, "newer copy\0".encode("utf-16-le"))])
        self.target_reads_and_watch()
        self.assertEqual(self.clipboard.text(), "newer copy")

    def test_back_to_back_pastes_restore_the_original_not_the_first_dictation(self) -> None:
        self.paster.paste("first ")
        self.paster.paste("second ")
        self.assertEqual(self.clipboard.text(), "second ")
        self.target_reads_and_watch(0)  # stale generation: no-op
        self.assertEqual(self.clipboard.text(), "second ")
        self.target_reads_and_watch(1)
        self.assertEqual(self.clipboard.text(), "user")

    def test_unrestorable_sensitive_or_busy_clipboard_is_left_alone(self) -> None:
        self.clipboard.unrestorable = True
        self.assertFalse(self.paster.paste("x"))
        self.clipboard.unrestorable, self.clipboard.busy = False, True
        self.assertFalse(self.paster.paste("x"))
        self.assertEqual((self.clipboard.writes, self.sent), ([], []))

    def test_failed_offer_puts_the_snapshot_back(self) -> None:
        self.clipboard.fail_offer = True
        self.assertFalse(self.paster.paste("x"))
        self.assertEqual(self.clipboard.text(), "user")
        self.assertEqual(self.sent, [])

    def test_failed_ctrl_v_releases_the_keys_restores_and_raises(self) -> None:
        sends: list = []

        def send(inputs):
            sends.append([(e.u.ki.wVk, e.u.ki.dwFlags) for e in inputs])
            if len(sends) == 1:
                raise OSError("SendInput delivered 1/4 events")

        paster = win_inject.ClipboardPaster(self.clipboard, send=send, spawn=self.watchers.append)
        with self.assertRaises(OSError):
            paster.paste("dictated ")
        up = win_inject.KEYEVENTF_KEYUP
        self.assertEqual(sends[1], [(0x56, up), (0x11, up)])  # never leave Ctrl logically down
        self.assertEqual(self.clipboard.text(), "user")
        self.assertEqual(self.watchers, [])

    def test_a_busy_restore_is_retried_and_never_strands_the_dictation(self) -> None:
        self.paster.paste("first ")
        self.clipboard.fail_restores = 100  # busy for every attempt of this watcher
        self.target_reads_and_watch()
        self.assertEqual(self.clipboard.text(), "first ")
        # The pending snapshot survived: the next paste reuses it (never
        # snapshotting our own offer), and its restore brings the user's back.
        self.clipboard.fail_restores = 0
        self.assertTrue(self.paster.paste("second "))
        self.target_reads_and_watch()
        self.assertEqual(self.clipboard.text(), "user")

    def test_flush_retries_a_restore_that_failed_earlier(self) -> None:
        self.paster.paste("dictated ")
        self.clipboard.fail_restores = 2  # two busy attempts, then free
        self.paster.flush()
        self.assertEqual(self.clipboard.text(), "user")

    def test_no_clipboard_window_means_typing_not_an_error(self) -> None:
        paster = win_inject.ClipboardPaster(send=self.sent.append, spawn=self.watchers.append)
        with mock.patch.object(win_inject, "_ClipboardWindow", side_effect=OSError("no window")):
            self.assertFalse(paster.paste("x"))
        self.assertEqual(self.sent, [])

    def test_flush_restores_immediately_at_shutdown(self) -> None:
        self.paster.paste("dictated ")
        self.paster.flush()
        self.assertEqual(self.clipboard.text(), "user")
        self.target_reads_and_watch()  # nothing left to restore
        self.assertEqual(len(self.clipboard.writes), 1)


PRIVATE_STATION_SCRIPT = r'''
import ctypes, os, sys, threading
from ctypes import wintypes
u = ctypes.WinDLL("user32", use_last_error=True)
u.CreateWindowStationW.restype = wintypes.HANDLE
u.CreateWindowStationW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p)
u.SetProcessWindowStation.argtypes = (wintypes.HANDLE,)
u.GetProcessWindowStation.restype = wintypes.HANDLE
u.CreateDesktopW.restype = wintypes.HANDLE
u.CreateDesktopW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.c_void_p, wintypes.DWORD,
                             wintypes.DWORD, ctypes.c_void_p)
u.GetUserObjectInformationW.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                        wintypes.DWORD, ctypes.POINTER(wintypes.DWORD))
u.GetClipboardSequenceNumber.restype = wintypes.DWORD
u.GetClipboardData.restype = wintypes.HANDLE
u.OpenClipboard.argtypes = (wintypes.HWND,)
k = ctypes.WinDLL("kernel32")
k.GlobalLock.restype = wintypes.LPVOID
k.GlobalLock.argtypes = (wintypes.HANDLE,)
k.GlobalUnlock.argtypes = (wintypes.HANDLE,)
# The clipboard belongs to the process window station: move to a private one
# (with a desktop for the owner window) so the user's clipboard is never used.
original = u.GetProcessWindowStation()
original_sequence = u.GetClipboardSequenceNumber()
name = "sotto-test-%d" % os.getpid()
u.SetProcessWindowStation(u.CreateWindowStationW(name, 0, 0x37F, None))
buf = ctypes.create_unicode_buffer(256)
u.GetUserObjectInformationW(u.GetProcessWindowStation(), 2, buf, 512, ctypes.byref(wintypes.DWORD()))
if buf.value != name or not u.CreateDesktopW("Default", None, None, 0, 0x01FF, None):
    sys.exit(3)  # never touch a shared clipboard
import win_inject
clipboard = win_inject._ClipboardWindow()
clipboard.write([(13, "user copy\0".encode("utf-16-le"))])
sent, restored = [], threading.Event()
read = {}


def app_reads(inputs):
    """Stands in for the focused app handling Ctrl+V: it reads the clipboard."""
    sent.append(inputs)

    def target():
        u.OpenClipboard(None)
        handle = u.GetClipboardData(13)
        read["text"] = ctypes.wstring_at(k.GlobalLock(handle))
        k.GlobalUnlock(handle)
        u.CloseClipboard()

    threading.Thread(target=target).start()


def spawn(watch):
    def run():
        watch()
        restored.set()

    threading.Thread(target=run).start()


paster = win_inject.ClipboardPaster(clipboard, send=app_reads, render_wait=30, grace=0.05, spawn=spawn)
text = "dictated こんにちは "
assert paster.paste(text)
assert restored.wait(20), "no restore after the app read the text"
assert read["text"] == text, read
assert clipboard.rendered.is_set()
formats = dict(clipboard.snapshot())
assert formats[13].decode("utf-16-le") == "user copy\0", formats
names = {n: u.RegisterClipboardFormatW(n) for n in (
    "ExcludeClipboardContentFromMonitorProcessing", "CanIncludeInClipboardHistory",
    "CanUploadToCloudClipboard", "Clipboard Viewer Ignore")}
assert names["ExcludeClipboardContentFromMonitorProcessing"] not in formats  # restore stays pasteable
assert names["Clipboard Viewer Ignore"] not in formats
assert names["CanIncludeInClipboardHistory"] in formats and names["CanUploadToCloudClipboard"] in formats


def mark_sensitive():
    clipboard._open()
    try:
        clipboard._put(names["Clipboard Viewer Ignore"], b"\0\0\0\0")
    finally:
        u.CloseClipboard()


# A password manager's clipboard (owner-marked) is never snapshotted for restore.
clipboard.write([(13, "secret\0".encode("utf-16-le"))], markers=())
clipboard.call(mark_sensitive)
assert paster.paste("x") is False
assert len(sent) == 1
u.SetProcessWindowStation(original)
assert u.GetClipboardSequenceNumber() == original_sequence, "the user's clipboard changed"
print("ok")
'''


@unittest.skipUnless(sys.platform == "win32", "Windows-only module")
class RealClipboardTest(unittest.TestCase):
    def test_delayed_render_paste_and_restore_in_a_private_window_station(self) -> None:
        result = subprocess.run([sys.executable, "-c", PRIVATE_STATION_SCRIPT], cwd=ROOT,
                                capture_output=True, text=True, timeout=90)
        if result.returncode == 3:
            self.skipTest("could not isolate a private window station")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()

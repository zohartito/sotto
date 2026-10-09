"""Delivering finished text to the frontmost app: the clipboard snapshot and
restore around a synthetic ⌘V, the FIFO delivery queue, Sotto's own tagged
key events, and "scratch that".

Everything runs against fakes: no key event is posted, the real pasteboard is
never touched, and a virtual clock drives threading.Timer, so nothing sleeps.
"""
import threading
import types
import unittest
from unittest.mock import patch

import Quartz as RealQuartz

import sotto

PLAIN_TEXT = "public.utf8-plain-text"


# -- fakes -------------------------------------------------------------------

class FakeItem:
    def __init__(self, data=None):
        self.data = dict(data or {})

    @classmethod
    def alloc(cls):
        return cls

    @classmethod
    def init(cls):
        return cls()

    def types(self):
        return list(self.data)

    def dataForType_(self, pb_type):
        return self.data.get(pb_type)

    def setData_forType_(self, data, pb_type):
        self.data[pb_type] = data


class FakePasteboard:
    """The NSPasteboard semantics that matter here: clearContents bumps
    changeCount; reading never does."""

    def __init__(self, text):
        self.change = 1
        self.items = [FakeItem({PLAIN_TEXT: text})]

    def generalPasteboard(self):
        return self

    def pasteboardItems(self):
        return list(self.items)

    def clearContents(self):
        self.change += 1
        self.items = []

    def setString_forType_(self, text, pb_type):
        self.items = [FakeItem({pb_type: text})]

    def writeObjects_(self, items):
        self.items.extend(items)

    def changeCount(self):
        return self.change

    def text(self):
        return self.items[0].data.get(PLAIN_TEXT) if self.items else None

    def user_copies(self, text):
        """What a ⌘C in another app does to the general pasteboard."""
        self.clearContents()
        self.setString_forType_(text, PLAIN_TEXT)


class FakeQuartz:
    """Records the key events Sotto posts instead of posting them; real
    Quartz constants, so ALL_MODIFIERS and TRIGGERS keep their values."""

    def __init__(self):
        self.posted = []
        self.flags_state = 0  # what CGEventSourceFlagsState reports

    def __getattr__(self, name):
        return getattr(RealQuartz, name)

    def CGEventCreateKeyboardEvent(self, _source, keycode, key_down):
        return {"keycode": keycode, "down": key_down, "flags": 0, "user_data": 0}

    def CGEventSetFlags(self, event, flags):
        event["flags"] = flags

    def CGEventGetFlags(self, event):
        return event["flags"]

    def CGEventSetIntegerValueField(self, event, field, value):
        if field == RealQuartz.kCGEventSourceUserData:
            event["user_data"] = value
        else:
            event[field] = value

    def CGEventGetIntegerValueField(self, event, field):
        if field == RealQuartz.kCGKeyboardEventKeycode:
            return event["keycode"]
        if field == RealQuartz.kCGEventSourceUserData:
            return event.get("user_data", 0)
        return event.get(field, 0)

    def CGEventKeyboardSetUnicodeString(self, event, _units, text):
        event["text"] = text

    def CGEventPost(self, _tap, event):
        self.posted.append(dict(event))

    def CGEventSourceFlagsState(self, _state):
        return self.flags_state

    def key_downs(self, keycode):
        return [e for e in self.posted if e["keycode"] == keycode and e["down"]]


class VirtualTimers:
    """threading.Timer replacement driven by a virtual clock."""

    def __init__(self):
        self.now = 0.0
        self.pending = []  # (fire_at, seq, fn)
        self.seq = 0
        outer = self

        class Timer:
            def __init__(self, delay, fn, args=()):
                self.delay, self.fn, self.args = delay, fn, args
                self.daemon = False

            def start(self):
                outer.seq += 1
                outer.pending.append((outer.now + self.delay, outer.seq,
                                      lambda: self.fn(*self.args)))

        self.Timer = Timer

    def advance_to(self, t):
        while True:
            due = sorted(p for p in self.pending if p[0] <= t)
            if not due:
                break
            item = due[0]
            self.pending.remove(item)
            self.now = item[0]
            item[2]()
        self.now = t


class _ThreadingWithTimer:
    """sotto.threading with Timer swapped for the virtual one."""

    def __init__(self, timer):
        self.Timer = timer

    def __getattr__(self, name):
        return getattr(threading, name)


def patch_insertion(test, *, clipboard="ORIGINAL user clipboard", secure=False):
    """Point sotto.inject / _restore_clipboard at the fakes for one test."""
    timers = VirtualTimers()
    pasteboard = FakePasteboard(clipboard)
    quartz = FakeQuartz()
    logs = []
    for target, value in (
            ("Quartz", quartz), ("NSPasteboard", pasteboard), ("NSPasteboardItem", FakeItem),
            ("NSPasteboardTypeString", PLAIN_TEXT), ("secure_input_active", lambda: secure),
            ("log", logs.append), ("threading", _ThreadingWithTimer(timers.Timer)),
            ("_restore_generation", 0)):
        patcher = patch.object(sotto, target, value)
        patcher.start()
        test.addCleanup(patcher.stop)
    patcher = patch.object(sotto, "_pending_restore", None, create=True)
    patcher.start()
    test.addCleanup(patcher.stop)
    patcher = patch("PyObjCTools.AppHelper.callAfter", lambda fn, *args: fn(*args))
    patcher.start()
    test.addCleanup(patcher.stop)
    patcher = patch.object(sotto.time, "sleep", lambda _s: None)
    patcher.start()
    test.addCleanup(patcher.stop)
    return types.SimpleNamespace(timers=timers, pasteboard=pasteboard, quartz=quartz, logs=logs)


# -- F8: overlapping pastes keep the user's clipboard ---------------------------

class ClipboardCarryForwardTests(unittest.TestCase):
    def test_overlapping_pastes_keep_the_users_clipboard(self):
        world = patch_insertion(self)
        sotto.inject("first dictation")                       # t = 0
        world.timers.advance_to(0.3)
        sotto.inject("second dictation")                      # inside the restore window
        self.assertEqual(len(world.quartz.key_downs(9)), 2)   # both ⌘V went out
        world.timers.advance_to(sotto.RESTORE_DELAY_S * 2 + 1.0)
        self.assertEqual(world.pasteboard.text(), "ORIGINAL user clipboard")

    def test_three_quick_pastes_carry_the_original_forward(self):
        world = patch_insertion(self)
        for index, text in enumerate(["one", "two", "three"]):
            world.timers.advance_to(index * 0.2)
            sotto.inject(text)
        self.assertEqual(world.pasteboard.text(), "three ")   # the app can still read the last one
        world.timers.advance_to(sotto.RESTORE_DELAY_S * 2 + 1.0)
        self.assertEqual(world.pasteboard.text(), "ORIGINAL user clipboard")

    def test_a_user_copy_between_overlapping_pastes_wins(self):
        world = patch_insertion(self)
        sotto.inject("first dictation")
        world.timers.advance_to(0.2)
        world.pasteboard.user_copies("USER COPY")             # ⌘C in the window
        world.timers.advance_to(0.3)
        sotto.inject("second dictation")                      # must snapshot the user's copy
        world.timers.advance_to(sotto.RESTORE_DELAY_S * 2 + 1.0)
        self.assertEqual(world.pasteboard.text(), "USER COPY")

    def test_restore_is_skipped_when_the_user_copied_after_the_paste(self):
        world = patch_insertion(self)
        sotto.inject("dictation")
        world.timers.advance_to(0.2)
        world.pasteboard.user_copies("USER COPY")
        world.timers.advance_to(sotto.RESTORE_DELAY_S * 2 + 1.0)
        self.assertEqual(world.pasteboard.text(), "USER COPY")


if __name__ == "__main__":
    unittest.main()

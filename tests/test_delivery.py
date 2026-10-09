"""Delivering finished text to the frontmost app: the clipboard snapshot and
restore around a synthetic ⌘V, the FIFO delivery queue, Sotto's own tagged
key events, and "scratch that".

Everything runs against fakes: no key event is posted, the real pasteboard is
never touched, and a virtual clock drives threading.Timer, so nothing sleeps.
"""
import ast
import sys
import threading
import types
import unittest
from unittest.mock import patch

if sys.platform != "darwin":
    raise unittest.SkipTest("Quartz key events and the AppKit pasteboard — macOS-only")

import Quartz as RealQuartz

import sotto

PLAIN_TEXT = "public.utf8-plain-text"
CMD = RealQuartz.kCGEventFlagMaskCommand


def run_closures(names, namespace):
    """Compile the named functions nested directly inside sotto.run() into
    `namespace`, whose entries stand in for the closure variables they read."""
    source = open(sotto.__file__, encoding="utf-8").read()
    tree = ast.parse(source)
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    found = {node.name: node for node in run.body
             if isinstance(node, ast.FunctionDef) and node.name in names}
    missing = set(names) - set(found)
    assert not missing, f"not nested in run(): {sorted(missing)}"
    module = ast.Module(body=[found[name] for name in names], type_ignores=[])
    exec(compile(module, sotto.__file__, "exec"), namespace)
    return namespace


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


# -- F42: the restore gives a slow app time to read the pasteboard ---------------

class RestoreWindowTests(unittest.TestCase):
    def test_restore_waits_for_a_slow_app_then_puts_the_original_back(self):
        world = patch_insertion(self)
        sotto.inject("dictation")
        world.timers.advance_to(1.5)  # the app is still busy handling ⌘V (the audit's repro)
        self.assertEqual(world.pasteboard.text(), "dictation ", "restored before the app could read it")
        world.timers.advance_to(sotto.RESTORE_DELAY_S + 0.01)
        self.assertEqual(world.pasteboard.text(), "ORIGINAL user clipboard")

    def test_restore_delay_is_a_few_seconds_and_bounded(self):
        self.assertGreaterEqual(sotto.RESTORE_DELAY_S, 2.0)
        self.assertLessEqual(sotto.RESTORE_DELAY_S, 5.0)

    def test_restore_still_checks_change_count_at_restore_time(self):
        world = patch_insertion(self)
        sotto.inject("dictation")
        world.timers.advance_to(sotto.RESTORE_DELAY_S - 0.05)
        world.pasteboard.user_copies("USER COPY")  # a ⌘C just before the restore fires
        world.timers.advance_to(sotto.RESTORE_DELAY_S + 1.0)
        self.assertEqual(world.pasteboard.text(), "USER COPY")


# -- F10: Sotto's own key events are tagged ------------------------------------

class OwnEventTagTests(unittest.TestCase):
    def test_paste_tags_both_cmd_v_events(self):
        world = patch_insertion(self)
        sotto.inject("hello", spacing="none")
        self.assertEqual([(e["keycode"], e["flags"], e["user_data"]) for e in world.quartz.posted],
                         [(9, CMD, sotto.SOTTO_EVENT_TAG), (9, CMD, sotto.SOTTO_EVENT_TAG)])
        self.assertTrue(all(sotto.is_own_event(e) for e in world.quartz.posted))

    def test_typing_mode_tags_every_event(self):
        world = patch_insertion(self)
        sotto.inject("typed across chunks, well over sixteen characters",
                     insert_mode="type", spacing="none")
        self.assertGreater(len(world.quartz.posted), 2)
        self.assertTrue(all(e["user_data"] == sotto.SOTTO_EVENT_TAG for e in world.quartz.posted))

    def test_undo_shortcut_is_tagged(self):
        world = patch_insertion(self)
        sotto.post_command_key(6)
        self.assertEqual([(e["keycode"], e["down"], e["flags"], e["user_data"]) for e in world.quartz.posted],
                         [(6, True, CMD, sotto.SOTTO_EVENT_TAG), (6, False, CMD, sotto.SOTTO_EVENT_TAG)])

    def test_a_users_key_event_is_not_sottos(self):
        world = patch_insertion(self)
        event = world.quartz.CGEventCreateKeyboardEvent(None, 9, True)
        world.quartz.CGEventSetFlags(event, CMD)
        self.assertFalse(sotto.is_own_event(event))


# -- F10 / F7: the FIFO delivery queue -----------------------------------------

class DeliveryQueueHarness:
    """A DeliveryQueue over fakes: a virtual clock, scripted key/recording
    state, and recorders for inserts, undos, notes and log lines."""

    def __init__(self, *, held_until=0.0, recording_while_held=True, pid=4242):
        self.timers = VirtualTimers()
        self.held_until = held_until
        self.recording_while_held = recording_while_held
        self.pid = pid
        self.keydowns = 0
        self.secure = False
        self.inserts_ok = True
        self.shutdown = False
        self.delivered, self.undos, self.notes, self.logs = [], [], [], []
        self.queue = sotto.DeliveryQueue(
            insert=self._insert, undo_keys=lambda: self.undos.append(round(self.timers.now, 2)),
            keys_held=self.held, recording=lambda: self.held() and self.recording_while_held,
            frontmost_pid=lambda: self.pid, keydowns=lambda: self.keydowns,
            secure_input=lambda: self.secure, call_after=lambda fn, *args: fn(*args),
            note=lambda title, message: self.notes.append((title, message)), log=self.logs.append,
            shutdown_requested=lambda: self.shutdown, clock=lambda: self.timers.now,
            timer=self.timers.Timer)

    def held(self):
        return self.timers.now < self.held_until

    def _insert(self, text):
        self.delivered.append((text, round(self.timers.now, 2), self.held()))
        return self.inserts_ok


class DeliveryQueueOrderTests(unittest.TestCase):
    def test_results_behind_a_held_key_paste_in_capture_order(self):
        # The key is held for dictation 3 from t=0 to t=0.5; results 1 and 2 of
        # the serial worker land while it is held.
        world = DeliveryQueueHarness(held_until=0.5)
        world.queue.paste("result-1 ")                        # t = 0.0
        world.timers.advance_to(0.1)
        world.queue.paste("result-2 ")                        # t = 0.1
        world.timers.advance_to(3.0)
        self.assertEqual([text for text, _t, _held in world.delivered], ["result-1 ", "result-2 "])
        self.assertFalse(any(held for _text, _t, held in world.delivered))

    def test_nothing_is_pasted_while_the_trigger_is_held_during_a_recording(self):
        world = DeliveryQueueHarness(held_until=90.0)         # a long push-to-talk hold
        world.queue.paste("result-1 ")
        world.timers.advance_to(60.0)
        self.assertEqual(world.delivered, [])
        self.assertEqual(world.notes, [])                     # a recording is active: keep waiting
        world.timers.advance_to(91.0)                         # released
        self.assertEqual([text for text, _t, _held in world.delivered], ["result-1 "])
        self.assertFalse(world.delivered[0][2])

    def test_a_key_held_with_no_recording_drops_the_paste_after_the_bound_with_one_note(self):
        world = DeliveryQueueHarness(held_until=10_000.0, recording_while_held=False)  # a stuck modifier
        world.queue.paste("result-1 ")
        world.timers.advance_to(5.0)
        world.queue.paste("result-2 ")
        world.timers.advance_to(sotto.DELIVERY_WAIT_MAX_S - 0.5)
        self.assertEqual(world.notes, [])                     # still inside the bound
        world.timers.advance_to(sotto.DELIVERY_WAIT_MAX_S + 1.0)
        self.assertEqual(world.delivered, [])
        self.assertEqual(len(world.notes), 1)
        self.assertIn("History", world.notes[0][0] + world.notes[0][1])
        world.timers.advance_to(2 * sotto.DELIVERY_WAIT_MAX_S + 5.0)
        self.assertEqual(len(world.notes), 1)                 # nothing left to drop, no second note
        world.held_until = 0.0                                # key released: the queue works again
        world.queue.paste("result-3 ")
        self.assertEqual([text for text, _t, _held in world.delivered], ["result-3 "])

    def test_undo_waits_in_the_same_queue_behind_the_paste(self):
        world = DeliveryQueueHarness(held_until=0.5)
        world.queue.paste("result-1 ")
        world.timers.advance_to(0.1)
        world.queue.undo()                                    # "scratch that" dictated right after
        self.assertEqual(world.undos, [])
        world.timers.advance_to(3.0)
        self.assertEqual([text for text, _t, _held in world.delivered], ["result-1 "])
        self.assertEqual(len(world.undos), 1)
        self.assertGreaterEqual(world.undos[0], world.delivered[0][1])

    def test_shutdown_drops_pending_deliveries(self):
        world = DeliveryQueueHarness(held_until=0.5)
        world.queue.paste("result-1 ")
        world.shutdown = True
        world.timers.advance_to(3.0)
        self.assertEqual(world.delivered, [])


# -- F7: inject() reports whether anything reached the app ----------------------

class InjectReportsInsertTests(unittest.TestCase):
    def test_secure_input_decline_reports_no_insert(self):
        world = patch_insertion(self, secure=True)
        self.assertIs(sotto.inject("secret"), False)
        self.assertIs(sotto.inject("secret", insert_mode="type"), False)
        self.assertEqual(world.quartz.posted, [])
        self.assertEqual(world.pasteboard.text(), "ORIGINAL user clipboard")

    def test_paste_and_typing_report_an_insert(self):
        world = patch_insertion(self)
        self.assertIs(sotto.inject("hello"), True)
        self.assertIs(sotto.inject("hello", insert_mode="type"), True)
        self.assertGreater(len(world.quartz.posted), 2)


# -- F7: "scratch that" undoes only Sotto's own insert ---------------------------

class ScratchThatTests(unittest.TestCase):
    def test_secure_input_decline_does_not_arm_undo(self):
        world = DeliveryQueueHarness()
        world.inserts_ok = False                              # a password field had focus
        world.queue.paste("my dictation")
        world.secure = False                                  # the user tabs out of the field
        world.queue.undo()
        self.assertEqual(world.undos, [])
        self.assertTrue(any("nothing recent" in line for line in world.logs))

    def test_undo_refused_when_another_app_is_frontmost(self):
        world = DeliveryQueueHarness(pid=100)
        world.queue.paste("hello")                            # pasted into app 100
        world.pid = 200                                       # the user switched to app 200
        world.queue.undo()
        self.assertEqual(world.undos, [])
        self.assertTrue(any("another app" in line for line in world.logs), world.logs)

    def test_undo_refused_when_the_frontmost_app_is_unknown(self):
        world = DeliveryQueueHarness(pid=None)                # NSWorkspace gave no answer at paste time
        world.queue.paste("hello")
        world.queue.undo()                                    # ...nor now: None == None proves nothing
        self.assertEqual(world.undos, [])
        self.assertTrue(any("unknown" in line for line in world.logs), world.logs)

    def test_undo_refused_after_the_user_typed(self):
        world = DeliveryQueueHarness()
        world.queue.paste("hello")
        world.keydowns += 3                                   # the user kept typing
        world.queue.undo()
        self.assertEqual(world.undos, [])
        self.assertTrue(any("typed" in line for line in world.logs), world.logs)

    def test_undo_fires_once_for_sottos_own_insert_in_the_same_app(self):
        world = DeliveryQueueHarness()
        world.queue.paste("hello")
        world.timers.advance_to(2.0)
        world.queue.undo()
        self.assertEqual(world.undos, [2.0])
        world.queue.undo()                                    # a second "scratch that" has nothing left
        self.assertEqual(world.undos, [2.0])

    def test_undo_window_expires(self):
        import voice_commands
        world = DeliveryQueueHarness()
        world.queue.paste("hello")
        world.timers.advance_to(voice_commands.SCRATCH_WINDOW_S + 1.0)
        world.queue.undo()
        self.assertEqual(world.undos, [])

    def test_undo_is_not_sent_into_a_secure_field(self):
        world = DeliveryQueueHarness()
        world.queue.paste("hello")
        world.secure = True
        world.queue.undo()
        self.assertEqual(world.undos, [])


# -- the event tap: Sotto's own events are never chords or user typing ----------

class EventTapCallbackHarness:
    def __init__(self, test, quartz, trigger="right-option"):
        self.events = []
        self.engine = sotto.GestureEngine(lambda: self.events.append("start"),
                                          lambda: self.events.append("finish"),
                                          lambda: self.events.append("discard"))
        test.addCleanup(self.engine.force_reset)
        mask, keycode, device_bit = sotto.TRIGGERS[trigger]
        self.binding = {"keycode": keycode, "mask": mask, "device_bit": device_bit, "trigger": trigger}
        self.trigger_state = {"down": False}
        self.user_keydowns = {"count": 0}
        self.quartz = quartz
        namespace = dict(vars(sotto))
        namespace.update({
            "Quartz": quartz, "shutdown": types.SimpleNamespace(requested=lambda: False),
            "trigger_state": self.trigger_state, "binding": self.binding, "engine": self.engine,
            "hotkey": sotto.parse_hotkey("ctrl-opt-d"),        # run()'s default
            "hotkey_state": {"pressed_at": None, "skip_up": False},
            "log": lambda _m: None, "revive_tap": lambda _reason: None,
            "user_keydowns": self.user_keydowns,
        })
        self.callback = run_closures(["callback"], namespace)["callback"]

    def press_trigger(self):
        event = {"keycode": self.binding["keycode"], "flags": self.binding["mask"] | self.binding["device_bit"],
                 "down": True, "user_data": 0}
        self.callback(None, RealQuartz.kCGEventFlagsChanged, event, None)

    def key_down(self, keycode, flags=0, user_data=0):
        event = {"keycode": keycode, "flags": flags, "down": True, "user_data": user_data}
        self.callback(None, RealQuartz.kCGEventKeyDown, event, None)


class EventTapCallbackTests(unittest.TestCase):
    def test_sottos_own_cmd_v_during_a_hold_is_not_a_chord(self):
        world = patch_insertion(self)
        tap = EventTapCallbackHarness(self, world.quartz)
        tap.press_trigger()
        self.assertEqual(tap.events, ["start"])
        sotto.inject("earlier result", spacing="none")        # a deferred paste going out
        own_cmd_v = world.quartz.key_downs(9)[0]              # exactly what a session tap receives
        tap.callback(None, RealQuartz.kCGEventKeyDown, own_cmd_v, None)
        self.assertEqual(tap.events, ["start"])
        self.assertTrue(tap.engine.snapshot()[0], "live dictation discarded by Sotto's own ⌘V")

    def test_a_real_key_during_a_hold_is_still_a_chord(self):
        world = patch_insertion(self)
        tap = EventTapCallbackHarness(self, world.quartz)
        tap.press_trigger()
        tap.key_down(9, flags=CMD)                             # the user's own ⌘V
        self.assertEqual(tap.events, ["start", "discard"])

    def test_user_keydowns_are_counted_but_not_sottos_or_the_hotkey(self):
        world = patch_insertion(self)
        tap = EventTapCallbackHarness(self, world.quartz)
        tap.key_down(0)                                        # "a"
        tap.key_down(1)                                        # "s"
        self.assertEqual(tap.user_keydowns["count"], 2)
        sotto.inject("hello", spacing="none")
        for event in world.quartz.key_downs(9):
            tap.callback(None, RealQuartz.kCGEventKeyDown, event, None)
        self.assertEqual(tap.user_keydowns["count"], 2)        # Sotto's own ⌘V is not typing
        hotkey_flags, hotkey_code = sotto.parse_hotkey("ctrl-opt-d")
        tap.key_down(hotkey_code, flags=hotkey_flags)          # starting a dictation is not typing
        self.assertEqual(tap.user_keydowns["count"], 2)
        tap.key_down(hotkey_code)                              # a plain "d" is
        self.assertEqual(tap.user_keydowns["count"], 3)


if __name__ == "__main__":
    unittest.main()

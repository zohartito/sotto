"""Windows trigger-key hook: edge-detected, feeding the shared GestureEngine.

pynput runs the WH_KEYBOARD_LL listener on its own thread; this module only
consumes edges and feeds sotto.GestureEngine, exactly like the macOS
CGEventTap callback.  Auto-repeat presses are filtered by the same down-state
guard the Mac uses, and ``physically_down`` (GetAsyncKeyState) drives the
lost-release recovery poller.

Trigger keys (settings names, shared with the Mac):
  right-ctrl (default), left-ctrl
  left-option  -> left Alt.  Each press also injects an unassigned key so the
                  Alt release never opens the focused app's menu bar.
  left-shift
Omitted on purpose: fn (handled by keyboard firmware, invisible to Windows);
right-option (right Alt is AltGr on many non-US layouts, and
arrives with a synthetic left Ctrl that would cancel every capture);
right-shift (holding it 8 seconds opens the Filter Keys prompt); the Windows
key (Start menu on release, Win+H voice typing, PowerToys hold-guides).

Events Sotto injects itself (typing, Ctrl+V, the Alt mask) carry
win_inject.INJECTED_TAG and are ignored, so a paste can never look like a
trigger press or a chord.
"""
from __future__ import annotations

import ctypes
import sys
import threading

if sys.platform != "win32":
    raise ImportError("win_hotkey is Windows-only")

import pynput.keyboard as kb

import win_inject

VK_SHIFT, VK_CONTROL, VK_MENU = 0x10, 0x11, 0x12  # either side of each pair
VK_LWIN, VK_RWIN = 0x5B, 0x5C
MODIFIER_VKS = (VK_SHIFT, VK_CONTROL, VK_MENU, VK_LWIN, VK_RWIN)
# Modifier keys alone type nothing, so they never count as the user typing
# (the Mac counts key-downs, and its modifiers arrive as flag changes).
MODIFIER_KEYS = frozenset((kb.Key.shift, kb.Key.shift_l, kb.Key.shift_r, kb.Key.ctrl,
                           kb.Key.ctrl_l, kb.Key.ctrl_r, kb.Key.alt, kb.Key.alt_l, kb.Key.alt_r,
                           kb.Key.alt_gr, kb.Key.cmd, kb.Key.cmd_l, kb.Key.cmd_r))

# settings name -> (pynput key, virtual key for GetAsyncKeyState, menu label)
TRIGGERS: dict[str, tuple[object, int, str]] = {
    "right-ctrl": (kb.Key.ctrl_r, VK_CONTROL, "Right Ctrl"),
    "left-ctrl": (kb.Key.ctrl_l, VK_CONTROL, "Left Ctrl"),
    "left-option": (kb.Key.alt_l, VK_MENU, "Left Alt"),
    "left-shift": (kb.Key.shift_l, VK_SHIFT, "Left Shift"),
}
DEFAULT_TRIGGER = "right-ctrl"


def supported(trigger: str | None) -> str:
    """A trigger this platform can use; anything else falls back to right Ctrl."""
    return trigger if trigger in TRIGGERS else DEFAULT_TRIGGER


def label(trigger: str) -> str:
    return TRIGGERS[supported(trigger)][2]


def get_async_key_state(vk: int) -> bool:
    """True while the virtual key is physically down (either side for VK_CONTROL etc.)."""
    return (ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000) != 0


def own_event(_msg, data) -> bool:
    """pynput win32_event_filter: False drops Sotto's own injected keys."""
    return (data.dwExtraInfo or 0) != win_inject.INJECTED_TAG


class TriggerHook:
    """Feed trigger-key edges into a GestureEngine; safe from any thread."""

    def __init__(self, engine, *, trigger: str = DEFAULT_TRIGGER,
                 mask_menu=win_inject.send_menu_mask) -> None:
        if trigger not in TRIGGERS:
            raise ValueError(f"unknown Windows trigger {trigger!r}; choose one of "
                             + ", ".join(TRIGGERS))
        self._engine = engine
        self._mask_menu = mask_menu
        self._down = False
        # The user's own key-downs so far (Sotto's injected keys never reach
        # the callbacks): typing after an insert means "scratch that" must
        # not undo it.  Read from other threads; only the hook thread writes.
        self.keydowns = 0
        self._listener: kb.Listener | None = None
        self._lock = threading.Lock()
        self._set(trigger)

    def _set(self, trigger: str) -> None:
        self.trigger = trigger
        self._key, self._vk, _label = TRIGGERS[trigger]

    def set_trigger(self, trigger: str) -> None:
        """Switch keys live; an in-flight hold is finished by the resync poller."""
        if trigger not in TRIGGERS:
            raise ValueError(f"unknown Windows trigger {trigger!r}")
        with self._lock:
            self._set(trigger)
            self._down = False

    def start(self) -> None:
        """Start the hook, or replace one whose thread died (never a second live one)."""
        with self._lock:
            if self._listener is not None and self._listener.is_alive():
                return
            self._down = False
            self._listener = kb.Listener(on_press=self._on_press,
                                         on_release=self._on_release,
                                         win32_event_filter=own_event)
            self._listener.daemon = True
            self._listener.start()

    def stop(self) -> None:
        with self._lock:
            if self._listener is not None:
                self._listener.stop()
                self._listener = None

    def alive(self) -> bool:
        # pynput's Listener is a threading.Thread: is_alive() covers both a
        # stopped listener and one whose hook thread died on its own.
        with self._lock:
            return self._listener is not None and self._listener.is_alive()

    def _on_press(self, key, _injected=False) -> None:
        if key != self._key:
            if key not in MODIFIER_KEYS:
                self.keydowns += 1  # one integer bump: nothing slow on the hook thread
            # Any other key during a trigger hold (Ctrl-C, Alt-Tab, the other
            # Ctrl) makes it a shortcut, not dictation — same rule as the Mac.
            if self._down:
                self._engine.chorded()
            return
        # Auto-repeat of the trigger itself is filtered by the edge guard.
        if self._down:
            return
        self._down = True
        if self._vk == VK_MENU:
            try:
                self._mask_menu()
            except OSError:
                pass  # worst case the app's menu bar opens; dictation still works
        self._engine.pressed()

    def _on_release(self, key, _injected=False) -> None:
        if key != self._key or not self._down:
            return
        self._down = False
        self._engine.released()

    @property
    def physically_down(self) -> bool:
        # Either side counts (both Ctrl keys for a Ctrl trigger): for the
        # lost-release poller that is the conservative direction.
        return get_async_key_state(self._vk)

    @property
    def modifiers_held(self) -> bool:
        """Any Shift/Ctrl/Alt/Windows key down: injected text or Ctrl+V would combine with it."""
        return any(get_async_key_state(vk) for vk in MODIFIER_VKS)

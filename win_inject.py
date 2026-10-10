"""Windows text delivery: type (SendInput Unicode) or paste (clipboard + Ctrl+V).

``type`` emits the text as KEYEVENTF_UNICODE keystrokes and never touches the
clipboard.

``paste`` snapshots the clipboard, offers the text with *delayed rendering*
from a small clipboard-owner window (marked so Clipboard History, cloud sync
and clipboard monitors skip it), and sends Ctrl+V.  Windows asks that window
for the text at the moment the target app reads it (WM_RENDERFORMAT); a short
grace later the user's clipboard is put back, unless something else was copied
meanwhile (the user's copy wins).  If the app never reads it, the clipboard is
restored after RENDER_WAIT_S.  A clipboard that cannot be put back exactly —
busy, oversized, holding GDI/private handles, or marked sensitive by its owner
(password managers) — is left alone and the text is typed instead.

Spacing: ``trailing`` appends one space so consecutive dictations don't run
together; ``none`` appends nothing.  ``smart`` behaves like ``trailing``:
Windows offers no reliable, app-independent way to read the character before
the caret, so no leading space is ever guessed (sotto.compose_insertion with
an unknown context).

Every event Sotto injects carries ``INJECTED_TAG`` in ``dwExtraInfo`` so the
trigger hook can ignore exactly its own keystrokes (win_hotkey).

UIPI: Windows silently drops synthetic input aimed at elevated (admin)
windows — the same class of limit as the macOS secure-input skip.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import queue
import struct
import sys
import threading
import time
from typing import Callable

if sys.platform != "win32":
    raise ImportError("win_inject is Windows-only")

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_CONTROL = 0x11
VK_V = 0x56
VK_MENU_MASK = 0xE8  # unassigned; cancels Alt's menu activation (win_hotkey)
INJECTED_TAG = 0x534F54544F  # "SOTTO" — marks Sotto's own synthetic keys
_CHUNK = 256  # inputs per SendInput call (dictations are far smaller)

RENDER_WAIT_S = 10.0    # an app that has not read the offer by now never pastes it
RESTORE_GRACE_S = 0.4   # after the first read, let the app finish its paste
MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
INSERT_MODES = ("paste", "type")
SPACING_MODES = ("smart", "trailing", "none")

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_user32.MapVirtualKeyW.restype = wintypes.UINT
_user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_uint),
        ("time", ctypes.c_uint),
        ("dwExtraInfo", ctypes.c_size_t),
    )


class _MOUSEINPUT(ctypes.Structure):
    # Present so the union — and therefore _INPUT — has the exact Win32 size.
    _fields_ = (
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_uint),
        ("dwFlags", ctypes.c_uint),
        ("time", ctypes.c_uint),
        ("dwExtraInfo", ctypes.c_size_t),
    )


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", ctypes.c_uint),
        ("wParamL", ctypes.c_ushort),
        ("wParamH", ctypes.c_ushort),
    )


class _INPUT_UNION(ctypes.Union):
    _fields_ = (
        ("mi", _MOUSEINPUT),
        ("ki", _KEYBDINPUT),
        ("hi", _HARDWAREINPUT),
    )


class _INPUT(ctypes.Structure):
    _fields_ = (
        ("type", ctypes.c_uint),
        ("u", _INPUT_UNION),
    )


def _key(vk: int, scan: int, flags: int) -> _INPUT:
    return _INPUT(INPUT_KEYBOARD, _INPUT_UNION(
        ki=_KEYBDINPUT(vk, scan, flags, 0, INJECTED_TAG)))


def build_inputs(text: str) -> list[_INPUT]:
    """One down+up KEYBDINPUT pair per UTF-16 code unit of ``text``."""
    units = struct.unpack(f"<{len(text.encode('utf-16-le')) // 2}H",
                          text.encode("utf-16-le"))
    inputs: list[_INPUT] = []
    for unit in units:
        inputs.append(_key(0, unit, KEYEVENTF_UNICODE))
        inputs.append(_key(0, unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
    return inputs


def _scan(vk: int) -> int:
    # Real scan codes: remote-desktop clients, VMs and some games ignore wScan=0.
    return _user32.MapVirtualKeyW(vk, 0)


def chord_inputs(modifier: int, vk: int) -> list[_INPUT]:
    """modifier down, key down, key up, modifier up (e.g. Ctrl+V)."""
    return [_key(modifier, _scan(modifier), 0), _key(vk, _scan(vk), 0),
            _key(vk, _scan(vk), KEYEVENTF_KEYUP), _key(modifier, _scan(modifier), KEYEVENTF_KEYUP)]


def release_inputs(*vks: int) -> list[_INPUT]:
    return [_key(vk, _scan(vk), KEYEVENTF_KEYUP) for vk in vks]


def send_inputs(inputs: list[_INPUT]) -> None:
    for i in range(0, len(inputs), _CHUNK):
        chunk = inputs[i:i + _CHUNK]
        array = (_INPUT * len(chunk))(*chunk)
        sent = ctypes.windll.user32.SendInput(
            len(chunk), ctypes.byref(array), ctypes.sizeof(_INPUT))
        if sent != len(chunk):
            raise OSError(f"SendInput delivered {sent}/{len(chunk)} events")


def send_menu_mask() -> None:
    """A tagged no-op key: Alt released after it no longer opens the menu bar."""
    send_inputs([_key(VK_MENU_MASK, 0, 0), _key(VK_MENU_MASK, 0, KEYEVENTF_KEYUP)])


def spaced(text: str, spacing: str) -> str:
    """Apply the spacing setting; ``smart`` == ``trailing`` on Windows (see module doc)."""
    if spacing not in SPACING_MODES:
        raise ValueError(f"unknown spacing {spacing!r}")
    if not text or spacing == "none" or text[-1].isspace():
        return text
    return text + " "


def inject(text: str) -> None:
    """Type ``text`` at the focused cursor as synthetic Unicode keystrokes."""
    if not text:
        return
    send_inputs(build_inputs(spaced(text, "trailing")))


# -- clipboard ---------------------------------------------------------------

CF_BITMAP, CF_METAFILEPICT, CF_PALETTE, CF_ENHMETAFILE = 2, 3, 9, 14
CF_UNICODETEXT = 13
CF_OWNERDISPLAY, CF_DSPBITMAP, CF_DSPMETAFILEPICT, CF_DSPENHMETAFILE = 0x80, 0x82, 0x83, 0x8E
CF_DIB, CF_DIBV5 = 8, 17
# Handle-based formats whose bytes can't be copied: a clipboard holding one
# can't be restored faithfully, so paste falls back to typing.  CF_BITMAP and
# CF_PALETTE are the exception: Windows synthesizes them back from CF_DIB.
_UNRESTORABLE = frozenset({CF_METAFILEPICT, CF_ENHMETAFILE, CF_OWNERDISPLAY, CF_DSPBITMAP,
                           CF_DSPMETAFILEPICT, CF_DSPENHMETAFILE, *range(0x200, 0x400)})
_SYNTHESIZED_FROM_DIB = frozenset({CF_BITMAP, CF_PALETTE})
# Registered formats honoured by Clipboard History, cloud clipboard sync and
# clipboard monitors.  The pasted offer is skipped by all of them; a restore
# or an explicit Copy is never uploaded, and a restore adds no History entry.
EXCLUDE_MONITORS = "ExcludeClipboardContentFromMonitorProcessing"
VIEWER_IGNORE = "Clipboard Viewer Ignore"  # the older convention many managers honour
NO_HISTORY = "CanIncludeInClipboardHistory"
NO_CLOUD = "CanUploadToCloudClipboard"
OFFER_MARKERS = (EXCLUDE_MONITORS, VIEWER_IGNORE, NO_HISTORY, NO_CLOUD)
RESTORE_MARKERS = (NO_HISTORY, NO_CLOUD)
COPY_MARKERS = (NO_CLOUD,)
# An owner that asks monitors to skip its content (password managers) may
# clear it on a timer; putting it back would defeat that, so Sotto types.
SENSITIVE_MARKERS = (VIEWER_IGNORE, EXCLUDE_MONITORS)
RESTORE_ATTEMPTS = 6  # a busy clipboard is retried with backoff (about 5 s in all)
GMEM_MOVEABLE = 0x0002
WM_RENDERFORMAT, WM_RENDERALLFORMATS, WM_DESTROYCLIPBOARD = 0x0305, 0x0306, 0x0307
WM_APP_CALL = 0x8000 + 0x51
HWND_MESSAGE = -3
PM_NOREMOVE = 0x0000
_LRESULT = ctypes.c_ssize_t
_WNDPROC = ctypes.WINFUNCTYPE(_LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)


class ClipboardBusy(OSError):
    """Another process holds the clipboard open."""


class _WNDCLASSW(ctypes.Structure):
    _fields_ = (("style", wintypes.UINT), ("lpfnWndProc", _WNDPROC), ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR))


for _name, _restype, _argtypes in (
        ("OpenClipboard", wintypes.BOOL, (wintypes.HWND,)),
        ("CloseClipboard", wintypes.BOOL, ()),
        ("EmptyClipboard", wintypes.BOOL, ()),
        ("EnumClipboardFormats", wintypes.UINT, (wintypes.UINT,)),
        ("GetClipboardData", wintypes.HANDLE, (wintypes.UINT,)),
        ("SetClipboardData", wintypes.HANDLE, (wintypes.UINT, wintypes.HANDLE)),
        ("GetClipboardOwner", wintypes.HWND, ()),
        ("GetClipboardSequenceNumber", wintypes.DWORD, ()),
        ("RegisterClipboardFormatW", wintypes.UINT, (wintypes.LPCWSTR,)),
        ("DefWindowProcW", _LRESULT, (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)),
        ("RegisterClassW", wintypes.ATOM, (ctypes.POINTER(_WNDCLASSW),)),
        ("CreateWindowExW", wintypes.HWND, (wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR,
                                            wintypes.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, wintypes.HWND, wintypes.HMENU,
                                            wintypes.HINSTANCE, wintypes.LPVOID)),
        ("GetOpenClipboardWindow", wintypes.HWND, ()),
        ("GetForegroundWindow", wintypes.HWND, ()),
        ("GetWindowThreadProcessId", wintypes.DWORD, (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))),
        ("PostMessageW", wintypes.BOOL, (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)),
        ("PeekMessageW", wintypes.BOOL, (ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT,
                                         wintypes.UINT, wintypes.UINT)),
        ("GetMessageW", wintypes.BOOL, (ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT,
                                        wintypes.UINT)),
        ("TranslateMessage", wintypes.BOOL, (ctypes.POINTER(wintypes.MSG),)),
        ("DispatchMessageW", _LRESULT, (ctypes.POINTER(wintypes.MSG),))):
    _function = getattr(_user32, _name)
    _function.restype, _function.argtypes = _restype, _argtypes
for _name, _restype, _argtypes in (
        ("GlobalAlloc", wintypes.HGLOBAL, (wintypes.UINT, ctypes.c_size_t)),
        ("GlobalLock", wintypes.LPVOID, (wintypes.HGLOBAL,)),
        ("GlobalUnlock", wintypes.BOOL, (wintypes.HGLOBAL,)),
        ("GlobalSize", ctypes.c_size_t, (wintypes.HGLOBAL,)),
        ("GlobalFree", wintypes.HGLOBAL, (wintypes.HGLOBAL,)),
        ("GetModuleHandleW", wintypes.HMODULE, (wintypes.LPCWSTR,))):
    _function = getattr(_kernel32, _name)
    _function.restype, _function.argtypes = _restype, _argtypes

_WINDOW_CLASS = "SottoClipboardOwner"
_windows: dict[int, "_ClipboardWindow"] = {}
_class_lock = threading.Lock()
_class_registered = False


@_WNDPROC
def _window_proc(hwnd, message, wparam, lparam):
    window = _windows.get(hwnd)
    if window is not None:
        try:
            handled = window._handle(message, wparam)
        except Exception:
            handled = 0  # never let an exception cross the Win32 callback
        if handled is not None:
            return handled
    return _user32.DefWindowProcW(hwnd, message, wparam, lparam)


class _ClipboardWindow:
    """Every clipboard operation, on one thread owning a message-only window.

    The window is the clipboard owner Windows asks for delayed-rendered text
    (WM_RENDERFORMAT); ``rendered`` is set when the foreground app (the paste
    target) read it.  Any other reader still gets the text, but does not
    start the restore; RENDER_WAIT_S covers that case.
    """

    def __init__(self) -> None:
        global _class_registered
        with _class_lock:
            if not _class_registered:
                wndclass = _WNDCLASSW(lpfnWndProc=_window_proc, lpszClassName=_WINDOW_CLASS,
                                      hInstance=_kernel32.GetModuleHandleW(None))
                if not _user32.RegisterClassW(ctypes.byref(wndclass)) and ctypes.get_last_error() != 1410:
                    raise OSError("could not register the clipboard window class")
                _class_registered = True
        self.hwnd = None
        self.offer: str | None = None
        self.rendered = threading.Event()
        self._calls: queue.Queue = queue.Queue()
        self._markers = {name: _user32.RegisterClipboardFormatW(name)
                         for name in (*OFFER_MARKERS, *SENSITIVE_MARKERS)}
        started = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(started,), daemon=True,
                                        name="sotto-clipboard")
        self._thread.start()
        if not started.wait(5) or not self.hwnd:
            raise OSError("the clipboard window could not be created")

    def _run(self, started: threading.Event) -> None:
        hwnd = _user32.CreateWindowExW(0, _WINDOW_CLASS, None, 0, 0, 0, 0, 0,
                                       wintypes.HWND(HWND_MESSAGE), None,
                                       _kernel32.GetModuleHandleW(None), None)
        if hwnd:
            _windows[hwnd] = self
        self.hwnd = hwnd
        started.set()
        if not hwnd:
            return
        message = wintypes.MSG()
        while _user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            _user32.TranslateMessage(ctypes.byref(message))
            _user32.DispatchMessageW(ctypes.byref(message))
        _windows.pop(hwnd, None)

    def _handle(self, message: int, wparam: int) -> int | None:
        if message == WM_APP_CALL:
            self._drain()
            return 0
        if message == WM_RENDERFORMAT:  # the requester holds the clipboard open
            self._render(wparam)
            return 0
        if message == WM_RENDERALLFORMATS:
            # Going away with an unread offer: drop it rather than leave the
            # dictated text on the clipboard for good.
            self.offer = None
            return 0
        if message == WM_DESTROYCLIPBOARD:  # someone (maybe us) emptied the clipboard
            self.offer = None
            return 0
        return None

    @staticmethod
    def _reader_is_foreground() -> bool:
        """Is the process holding the clipboard open the foreground app?"""
        reader, foreground = _user32.GetOpenClipboardWindow(), _user32.GetForegroundWindow()
        if not reader or not foreground:
            return True  # unknowable: count it, as before
        reader_pid, foreground_pid = wintypes.DWORD(), wintypes.DWORD()
        _user32.GetWindowThreadProcessId(reader, ctypes.byref(reader_pid))
        _user32.GetWindowThreadProcessId(foreground, ctypes.byref(foreground_pid))
        return reader_pid.value == foreground_pid.value

    def _render(self, fmt: int) -> None:
        if fmt == CF_UNICODETEXT and self.offer is not None:
            self._put(fmt, (self.offer + "\0").encode("utf-16-le"))
            if self._reader_is_foreground():
                self.rendered.set()

    def _drain(self) -> None:
        while True:
            try:
                function, done, box = self._calls.get_nowait()
            except queue.Empty:
                return
            if box.get("cancelled"):
                continue  # its caller gave up: never run it late
            try:
                box["value"] = function()
            except BaseException as exc:
                box["error"] = exc
            finally:
                done.set()

    def call(self, function: Callable, timeout: float = 10.0):
        """Run ``function`` on the window thread and return its result."""
        if threading.current_thread() is self._thread:
            return function()
        done, box = threading.Event(), {}
        self._calls.put((function, done, box))
        if not _user32.PostMessageW(self.hwnd, WM_APP_CALL, 0, 0):
            box["cancelled"] = True
            raise OSError("the clipboard window is gone")
        if not done.wait(timeout):
            box["cancelled"] = True
            raise OSError("the clipboard window did not answer")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    # -- window thread only ------------------------------------------------
    def _open(self) -> None:
        message = wintypes.MSG()
        for _ in range(40):  # clipboard managers hold it for milliseconds
            if _user32.OpenClipboard(self.hwnd):
                return
            # Answer any render request sent to us meanwhile, or both sides wait.
            _user32.PeekMessageW(ctypes.byref(message), None, 0, 0, PM_NOREMOVE)
            time.sleep(0.01)
        raise ClipboardBusy("the clipboard is busy")

    def _put(self, fmt: int, data: bytes) -> None:
        handle = _kernel32.GlobalAlloc(GMEM_MOVEABLE, max(len(data), 1))
        if not handle:
            raise MemoryError("GlobalAlloc failed")
        pointer = _kernel32.GlobalLock(handle)
        if not pointer:
            _kernel32.GlobalFree(handle)
            raise MemoryError("GlobalLock failed")
        ctypes.memmove(pointer, data, len(data))
        _kernel32.GlobalUnlock(handle)
        if not _user32.SetClipboardData(fmt, handle):
            _kernel32.GlobalFree(handle)  # ownership passes only on success
            raise OSError(f"SetClipboardData({fmt}) failed")

    def _put_markers(self, names) -> None:
        for name in names:
            self._put(self._markers[name], b"\x00\x00\x00\x00")

    def _snapshot(self) -> list[tuple[int, bytes]] | None:
        self._open()
        try:
            formats, fmt = [], 0
            while True:
                fmt = _user32.EnumClipboardFormats(fmt)
                if not fmt:
                    break
                formats.append(fmt)
            sensitive = {self._markers[name] for name in SENSITIVE_MARKERS}
            if _UNRESTORABLE.intersection(formats) or sensitive.intersection(formats):
                return None
            has_dib = CF_DIB in formats or CF_DIBV5 in formats
            items, total = [], 0
            for fmt in formats:
                if fmt in _SYNTHESIZED_FROM_DIB:
                    if has_dib:
                        continue
                    return None
                handle = _user32.GetClipboardData(fmt)
                if not handle:
                    continue
                size = _kernel32.GlobalSize(handle)
                total += size
                if total > MAX_SNAPSHOT_BYTES:
                    return None
                pointer = _kernel32.GlobalLock(handle)
                if not pointer:
                    return None  # not an HGLOBAL: can't be copied
                try:
                    items.append((fmt, ctypes.string_at(pointer, size)))
                finally:
                    _kernel32.GlobalUnlock(handle)
            return items
        finally:
            _user32.CloseClipboard()

    def _write(self, items: list[tuple[int, bytes]], markers) -> int:
        self._open()
        try:
            _user32.EmptyClipboard()
            for fmt, data in items:
                self._put(fmt, data)
            self._put_markers(markers)
        finally:
            _user32.CloseClipboard()
        return self.sequence()

    def _offer_text(self, text: str) -> int:
        self._open()
        try:
            _user32.EmptyClipboard()  # WM_DESTROYCLIPBOARD clears any older offer first
            self.offer = text
            self.rendered.clear()
            ctypes.set_last_error(0)
            if not _user32.SetClipboardData(CF_UNICODETEXT, None) and ctypes.get_last_error():
                self.offer = None
                raise OSError("could not offer text on the clipboard")
            self._put_markers(OFFER_MARKERS)
        finally:
            _user32.CloseClipboard()
        return self.sequence()

    def _restore_if_unchanged(self, items: list[tuple[int, bytes]], own_sequence: int) -> bool:
        self._open()  # while it is open nothing else can change the clipboard
        try:
            if self.sequence() != own_sequence:
                return False  # something was copied meanwhile: that copy wins
            _user32.EmptyClipboard()
            for fmt, data in items:
                self._put(fmt, data)
            self._put_markers(RESTORE_MARKERS)
            return True
        finally:
            _user32.CloseClipboard()

    # -- any thread ---------------------------------------------------------
    @staticmethod
    def sequence() -> int:
        return int(_user32.GetClipboardSequenceNumber())

    def snapshot(self) -> list[tuple[int, bytes]] | None:
        """Every format as bytes, or None when it can't (or mustn't) be put back."""
        return self.call(self._snapshot)

    def write(self, items: list[tuple[int, bytes]], markers=RESTORE_MARKERS) -> int:
        """Replace the clipboard with ``items`` plus ``markers``; returns the sequence."""
        return self.call(lambda: self._write(items, markers))

    def offer_text(self, text: str) -> int:
        """Offer ``text`` (delayed rendering); returns the sequence."""
        return self.call(lambda: self._offer_text(text))

    def restore_if_unchanged(self, items: list[tuple[int, bytes]], own_sequence: int) -> bool:
        """Put ``items`` back only if the clipboard is still ours (checked while open)."""
        return self.call(lambda: self._restore_if_unchanged(items, own_sequence))


def _text_item(text: str) -> tuple[int, bytes]:
    return CF_UNICODETEXT, (text + "\0").encode("utf-16-le")


@dataclass
class _Pending:
    snapshot: list[tuple[int, bytes]]
    own_sequence: int
    generation: int


def _spawn(function: Callable[[], None]) -> None:
    threading.Thread(target=function, daemon=True, name="sotto-clipboard-restore").start()


class ClipboardPaster:
    """Paste through the clipboard and put the user's clipboard back."""

    def __init__(self, clipboard=None, *, send: Callable[[list], None] = send_inputs,
                 render_wait: float = RENDER_WAIT_S, grace: float = RESTORE_GRACE_S,
                 spawn: Callable[[Callable[[], None]], None] = _spawn,
                 sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] = lambda message: None) -> None:
        self._clipboard = clipboard
        self._send = send
        self._render_wait = render_wait
        self._grace = grace
        self._spawn = spawn
        self._sleep = sleep
        self._log = log
        # Reentrant: a failed Ctrl+V restores while the paste still holds it.
        self._lock = threading.RLock()
        self._pending: _Pending | None = None
        self._generation = 0
        self._unread = 0  # pastes whose Ctrl+V is sent but whose watch has not ended

    @property
    def clipboard(self):
        with self._lock:  # one owner window, however many threads ask first
            if self._clipboard is None:
                self._clipboard = _ClipboardWindow()
            return self._clipboard

    def paste(self, text: str) -> bool:
        """True when pasted; False means the clipboard was left untouched (type instead).

        Raises OSError when the keystrokes could not be sent (the clipboard is
        put back first).  The offer and its Ctrl+V happen under one lock, so a
        restore or a shutdown flush can never land between them.
        """
        with self._lock:
            try:
                clipboard = self.clipboard
                pending = self._pending
                if pending is not None and clipboard.sequence() == pending.own_sequence:
                    # The previous paste's restore is still due (or failed):
                    # its snapshot is the user's clipboard, not our dictation.
                    snapshot = pending.snapshot
                else:
                    snapshot = clipboard.snapshot()
            except OSError as exc:
                self._log(f"! clipboard unavailable ({str(exc)[:80]}) — typing instead")
                return False
            if snapshot is None:
                return False
            try:
                own = clipboard.offer_text(text)
            except Exception as exc:
                try:
                    clipboard.write(snapshot)  # never leave the user's clipboard emptied
                except Exception:
                    pass
                self._log(f"! clipboard write failed ({str(exc)[:80]}) — typing instead")
                return False
            self._generation += 1
            generation = self._generation
            self._pending = _Pending(snapshot, own, generation)
            try:
                self._send(chord_inputs(VK_CONTROL, VK_V))
            except OSError:
                try:
                    self._send(release_inputs(VK_V, VK_CONTROL))  # never leave Ctrl logically down
                except OSError:
                    pass
                self._restore(generation)  # nothing was pasted: put the clipboard back now
                raise
            self._unread += 1
        try:
            self._spawn(lambda: self._watch(generation))
        except Exception as exc:  # "can't start new thread"
            with self._lock:
                self._unread -= 1  # no watch will ever settle this paste
            # The restore stays pending: the next paste reuses its snapshot,
            # and the shutdown flush puts it back.
            self._log(f"! clipboard restore not scheduled ({str(exc)[:80]}); "
                      "it is put back on the next paste or at quit")
        return True

    def settling(self) -> bool:
        """A Ctrl+V was sent and the app has not read the text yet (nor has
        the render wait run out): the paste is not delivered until then."""
        with self._lock:
            return self._unread > 0

    def _watch(self, generation: int) -> None:
        try:
            if self.clipboard.rendered.wait(self._render_wait):
                self._sleep(self._grace)
            self._restore(generation)
        finally:
            with self._lock:
                self._unread -= 1

    def _restore(self, generation: int, attempts: int = RESTORE_ATTEMPTS) -> None:
        """Put the snapshot back; a busy clipboard is retried, and a restore
        that still fails stays pending (the next paste reuses it, flush retries)."""
        for attempt in range(attempts):
            with self._lock:
                pending = self._pending
                if pending is None or pending.generation != generation:
                    return  # a newer paste owns the restore
                try:
                    self.clipboard.restore_if_unchanged(pending.snapshot, pending.own_sequence)
                    self._pending = None  # restored, or the user's newer copy won
                    return
                except (OSError, MemoryError) as exc:
                    reason = str(exc)[:80]
            self._sleep(0.2 * (attempt + 1))
        self._log(f"! clipboard not restored yet ({reason}); it will be retried")

    def flush(self) -> None:
        """Restore now (shutdown): never leave dictated text on the clipboard."""
        with self._lock:
            generation = self._pending.generation if self._pending else None
        if generation is not None:
            self._restore(generation)


_paster: ClipboardPaster | None = None
_paster_lock = threading.Lock()


def paster(log: Callable[[str], None] = lambda message: None) -> ClipboardPaster:
    global _paster
    with _paster_lock:
        if _paster is None:
            _paster = ClipboardPaster(log=log)
        return _paster


def flush_clipboard() -> None:
    """Put a pending clipboard restore in place now (shutdown)."""
    with _paster_lock:
        current = _paster
    if current is not None:
        current.flush()


def paste_settling() -> bool:
    """A paste is still waiting for the app to read it (Quit waits for it)."""
    with _paster_lock:
        current = _paster
    return current is not None and current.settling()


def foreground_identity() -> tuple[int, int] | None:
    """(window handle, process id) of the foreground window — where a
    keystroke lands — or None when there is none ("scratch that" checks it)."""
    window = _user32.GetForegroundWindow()
    if not window:
        return None
    pid = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(window, ctypes.byref(pid))
    return (window, pid.value) if pid.value else None


def copy_text(text: str) -> None:
    """Put ``text`` on the clipboard (History Copy, Retry); never uploaded to the cloud."""
    paster().clipboard.write([_text_item(text)], COPY_MARKERS)


def deliver(text: str, *, mode: str = "type", spacing: str = "trailing",
            log: Callable[[str], None] = lambda message: None) -> str:
    """Insert ``text`` at the cursor; returns how: "paste", "type" or "" (nothing)."""
    if mode not in INSERT_MODES:
        raise ValueError(f"unknown insert mode {mode!r}")
    text = spaced(text, spacing)
    if not text:
        return ""
    if mode == "paste" and paster(log).paste(text):
        return "paste"
    send_inputs(build_inputs(text))
    return "type"

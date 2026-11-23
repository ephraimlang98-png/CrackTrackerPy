"""Side B from the phone's own camera app (Google Camera on the Pixel), driven over USB by adb.

Same interface as phone_b.PhoneB (side-A trigger, saving under the A name, trigger log): only the camera
link differs. The shutter is the camera key; the photo Google Camera saves (HDR+, full quality) is copied to
the PC and deleted from the phone at once. Live view = screenshots of the camera screen; mouse wheel on it =
phone zoom (a short swipe on the zoom bar), click = focus there (a tap).
Only for the screen layout of the Pixel 8 Pro camera app (zoom chips .5 / 1 / 2 / 5 under the viewfinder).
"""
from __future__ import annotations

import calendar
import io
import queue
import re
import subprocess
import threading
import time

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtGui import QImage

import fc_link
import phone_b

PKG = "com.google.android.GoogleCamera"
DCIM = "/sdcard/DCIM/Camera"
CHIPS = {0.5: 0.341, 1.0: 0.448, 2.0: 0.554, 5.0: 0.661}   # zoom chip x / screen width (Pixel 8 Pro)
CHIP_UP = 72                     # chip row: px above the viewfinder bottom
STEP_PX = 25                     # one wheel notch = a swipe this long on the zoom bar
SAVE_WAIT_S = 12.0               # shutter -> final JPEG (HDR+ takes ~3.5 s)
EXTRA_S = 1.2                    # after the first final file: wait for a second (processed) version
KEEPALIVE_S = 15.0               # the phone screen times out after 30 s: touch the top bar meanwhile


def _run(args, timeout=15, binary=False):
    try:
        r = subprocess.run([fc_link.adb_path(), *args], capture_output=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.returncode, (r.stdout if binary else r.stdout.decode("utf-8", "replace"))
    except (OSError, subprocess.TimeoutExpired) as ex:
        return -1, (b"" if binary else str(ex))


def sh(cmd, timeout=15):
    return _run(["shell", cmd], timeout)


class GCamB(phone_b.PhoneB):
    _ev = Signal(str, object)                 # worker thread -> GUI thread

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = "gcam"
        self.timeout_s = 15.0                 # Google Camera processes each photo for a few seconds
        self.focus_locked = "auto"            # Google Camera focuses by itself
        self.jobs: queue.Queue = queue.Queue()
        self._run_ev = threading.Event()
        self._threads = []
        self.screen = (1008, 2244)
        self.vf = None                        # viewfinder rows on the screen (top, bottom)
        self.zoom_chip, self.zoom_steps = 1.0, 0
        self._zacc, self._zlock = 0, threading.Lock()   # wheel notches not sent yet (merged into one swipe)
        self.inflight = []                    # shutter presses whose photo was not collected yet
        self._known = None                    # DCIM names that are not ours (there before / already handled)
        self._sizes = {}                      # name -> (size, since) to see a file finished writing
        self._offset, self._offs = 0.0, []    # phone clock - PC clock [s], learned from the matches
        self._ev.connect(self._on_ev)

    # ------------------------------------------------------------------ link
    def start(self):
        if self._run_ev.is_set():
            return
        self._run_ev.set()
        self._threads = [threading.Thread(target=self._worker, daemon=True),
                         threading.Thread(target=self._previewer, daemon=True)]
        for t in self._threads:
            t.start()
        self.status.emit("warn", "מתחבר למצלמת גוגל...")

    def stop(self):
        self.disarm()
        self._run_ev.clear()
        self.jobs.put(None)
        self.connected = False

    def send(self, msg):
        """Commands of the FatigueCam app: only "capture" means something here."""
        if msg.get("type") == "capture":
            if not self.connected:
                return False
            self.jobs.put(("capture", msg.get("id")))
        return True

    def phone_to_pc(self, t):
        return t                              # capture times are PC times already

    @property
    def rtt(self):
        return np.nan

    # ------------------------------------------------------------------ public controls
    def zoom_preset(self, z):
        self.jobs.put(("chip", float(z)))

    def zoom_step(self, n):
        with self._zlock:
            first = self._zacc == 0
            self._zacc += int(n)
        if first:
            self.jobs.put(("step", None))

    def focus_at(self, fx, fy):
        """fx, fy: 0..1 in the live view (the viewfinder)."""
        self.jobs.put(("tap", (float(fx), float(fy))))

    def start_focus(self):
        x, y, w, h = self.roi
        self.focus_at(x + w / 2, y + h / 2)

    # ------------------------------------------------------------------ GUI-thread events
    def _on_ev(self, kind, data):
        if kind == "conn":
            ok, text = data
            was = self.connected
            self.connected = ok
            if ok:
                self.cam_ok, self.t_frame = True, time.time()
            if was and not ok:
                self.alert.emit(f"צד B: מצלמת גוגל לא זמינה ({text})")
            self.status.emit("ok" if ok else "err", text)
            self.changed.emit()
        elif kind == "capture":
            cid, data_, t_press = data
            self._on_capture({"id": cid, "t": t_press, "sharp": None}, data_, time.time())
        elif kind == "fail":
            cid, why = data
            if cid in self.setup_req:
                self.setup_req.pop(cid, None)
                self.alert.emit(f"התמונה ממצלמת גוגל לא הגיעה: {why}")
            r = self.recs.get(cid)
            if r is not None and r["status"] == "pending":
                self._finish(r, f"missed: {why}")
        elif kind == "log":
            self.log.emit(data)

    def _check_timeouts(self):
        now = time.time()
        with self._lock:
            late = [r for r in self.recs.values() if r["status"] == "pending" and r["t_sent"]
                    and now - r["t_sent"] > self.timeout_s]
        for r in late:
            self._finish(r, f"missed: no image within {self.timeout_s:g} s")
        for cid, (purpose, t) in list(self.setup_req.items()):
            if now - t > self.timeout_s + 3:
                self.setup_req.pop(cid, None)
                self.alert.emit("התמונה ממצלמת גוגל לא הגיעה – נסה שוב")

    # ------------------------------------------------------------------ worker thread (all phone actions)
    def _worker(self):
        last_ok = None
        t_keep = t_check = 0.0
        ok, why = False, ""
        while self._run_ev.is_set():
            try:
                job = self.jobs.get(timeout=0.25 if self.inflight else 1.0)
            except queue.Empty:
                job = ("idle", None)
            if job is None or not self._run_ev.is_set():
                break
            kind, arg = job
            now = time.time()
            # the camera check (dumpsys, ~0.4 s) is not repeated before every shutter press
            need = (not ok) or now - t_check > (3.0 if kind != "idle" else 2.5)
            if self.inflight and kind == "idle":
                need = (not ok) or now - t_check > 8.0
            if need:
                ok, why = self._ensure_camera(); t_check = time.time()
            if ok != last_ok:
                last_ok = ok
                self._ev.emit("conn", (ok, "מחובר · מצלמת גוגל" if ok else why))
            if not ok:
                if kind == "capture":
                    self._ev.emit("fail", (arg, why))
                for p in self.inflight:
                    if now - p["t"] > SAVE_WAIT_S + 8:
                        self._ev.emit("fail", (p["cid"], why))
                self.inflight = [p for p in self.inflight if now - p["t"] <= SAVE_WAIT_S + 8]
                continue
            try:
                if kind == "capture":
                    self._press(arg)
                elif kind == "chip":
                    self._tap_chip(arg); self.zoom_chip, self.zoom_steps = arg, 0
                elif kind == "step":
                    with self._zlock:
                        n, self._zacc = self._zacc, 0
                    if n:
                        self._swipe_zoom(n); self.zoom_steps += n
                elif kind == "tap":
                    x, y = self._vf_to_screen(*arg)
                    sh(f"input tap {x} {y}")
                if kind != "idle" or time.time() - t_keep > KEEPALIVE_S:
                    if kind == "idle" and not self.inflight:
                        sh(f"input tap {self.screen[0] // 2} 40")   # empty top bar: keeps the screen on
                    t_keep = time.time()
                if self.inflight:
                    self._collect()
            except Exception as ex:                                  # never kill the worker
                if kind == "capture":
                    self._ev.emit("fail", (arg, str(ex)))

    def _ensure_camera(self):
        code, out = _run(["get-state"], 5)
        if code != 0 or "device" not in out:
            return False, "הטלפון לא מחובר בכבל (או שלא אושר ניפוי USB)"
        _, win = sh("dumpsys window | grep -E 'mCurrentFocus|mDreamingLockscreen|isKeyguardShowing'")
        focus = next((ln for ln in win.splitlines() if "mCurrentFocus" in ln), "")
        if PKG in focus:
            if self.vf is None:
                self._measure_screen()
            return True, ""
        # not in front: wake the phone and open the camera (also works on the lock screen)
        sh("input keyevent KEYCODE_WAKEUP")
        locked = "mDreamingLockscreen=true" in win or "isKeyguardShowing=true" in win
        sh("am start -a android.media.action." + ("STILL_IMAGE_CAMERA_SECURE" if locked else "STILL_IMAGE_CAMERA"))
        time.sleep(2.0)
        _, win = sh("dumpsys window | grep mCurrentFocus")
        if PKG not in win:
            return False, "מצלמת גוגל לא נפתחה בטלפון"
        self._ev.emit("log", "צד B: מצלמת גוגל נפתחה בטלפון")
        self._measure_screen()
        self._restore_zoom()
        return True, ""

    def _measure_screen(self):
        _, out = sh("wm size")
        try:
            w, h = (int(v) for v in out.strip().split()[-1].split("x"))
            self.screen = (w, h)
        except ValueError:
            pass
        img = self._screencap()
        w, h = self.screen
        top, bot = int(0.074 * h), int(0.673 * h)                 # Pixel 8 Pro defaults
        if img is not None and img.shape[0] == h:
            mid = img[:, w // 5: 4 * w // 5].max(axis=(1, 2)) if img.ndim == 3 else img[:, w // 5: 4 * w // 5].max(axis=1)
            dark = mid < 12
            rows = np.nonzero(~dark[: h // 2])[0]
            if len(rows):
                top = int(rows[0])
            lo = np.nonzero(dark[h // 2:])[0]
            if len(lo):
                bot = int(h // 2 + lo[0])
        self.vf = (top, bot)

    def _restore_zoom(self):
        """Google Camera starts at 1x after it was closed: back to the zoom set from the PC."""
        if self.zoom_chip != 1.0 or self.zoom_steps:
            self._tap_chip(self.zoom_chip)
            if self.zoom_steps:
                self._swipe_zoom(self.zoom_steps)

    def _chip_y(self):
        return (self.vf[1] if self.vf else int(0.673 * self.screen[1])) - CHIP_UP

    def _tap_chip(self, z):
        x = int(CHIPS.get(z, CHIPS[1.0]) * self.screen[0])
        sh(f"input tap {x} {self._chip_y()}")
        time.sleep(0.8)

    def _swipe_zoom(self, n):
        """One swipe on the zoom bar (to the left = zoom in), n wheel notches long; the bar is
        logarithmic, ~1.6 notches per doubling, 0.5x..30x across its width."""
        y, w = self._chip_y(), self.screen[0]
        left = n > 0
        while n:
            k = int(np.clip(n, -12, 12))
            n -= k
            x0 = int(0.68 * w) if left else int(0.32 * w)   # start inside the zoom pill
            sh(f"input swipe {x0} {y} {x0 - k * STEP_PX} {y} {150 + 25 * abs(k)}")

    def _vf_to_screen(self, fx, fy):
        top, bot = self.vf or (int(0.074 * self.screen[1]), int(0.673 * self.screen[1]))
        return int(np.clip(fx, 0, 1) * self.screen[0]), int(top + np.clip(fy, 0, 1) * (bot - top))

    def _listing(self):
        _, out = sh(f"cd {DCIM} && ls -at | head -15 | while read f; do stat -c '%Y %s %n' \"$f\"; done")
        d = {}
        for line in out.splitlines():
            p = line.split(" ", 2)
            if len(p) == 3 and p[2] not in (".", ".."):
                d[p[2]] = (int(p[0]), int(p[1]))
        return d

    # ------------------------------------------------------------------ capture: press now, collect later
    # Google Camera processes every photo for a few seconds (HDR+) but takes the next one at once: the
    # shutter is pressed immediately at every stop, and the saved photos are collected in the background
    # and matched to their presses by the capture time in the file name (PXL_yyyymmdd_hhmmssmmm, UTC).
    # One photo every ~4 s keeps up; waiting for each photo before the next press did not.
    def _press(self, cid):
        if self._known is None:
            self._known = set(self._listing())
        t = time.time()
        sh("input keyevent KEYCODE_CAMERA")
        self.inflight.append(dict(cid=cid, t=t))

    @staticmethod
    def _name_time(n):
        m = re.match(r"PXL_(\d{8})_(\d{6})(\d{3})", n)
        if not m:
            return None
        try:
            tt = time.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            return calendar.timegm(tt) + int(m.group(3)) / 1000.0
        except ValueError:
            return None

    def _collect(self):
        cur = self._listing()
        now = time.time()
        known = self._known or set()
        new = {n: v for n, v in cur.items() if n not in known}
        pending = [n for n in new if n.startswith(".")]
        finals = {}
        for n, (mt, size) in new.items():
            if n.startswith(".") or not n.lower().endswith(".jpg"):
                continue
            prev = self._sizes.get(n)
            if prev is None or prev[0] != size:
                self._sizes[n] = (size, now)
            tn = self._name_time(n)
            if tn is not None:
                finals[n] = (tn, mt, size, now - self._sizes[n][1])
        taken = set()
        done = []
        for p in sorted(self.inflight, key=lambda q: q["t"]):
            # this press's photos: capture time (phone clock, learned offset) within -2..+3 s of the press
            pt = p["t"] + self._offset
            cand = {n: f for n, f in finals.items() if n not in taken and -2.0 <= f[0] - pt <= 3.0}
            if not cand:
                if now - p["t"] > SAVE_WAIT_S + 8:
                    done.append(p); self._ev.emit("fail", (p["cid"], "מצלמת גוגל לא שמרה תמונה"))
                continue
            taken |= set(cand)
            p.setdefault("t_first", now)
            stable = all(f[3] >= 0.4 and f[2] > 0 for f in cand.values())
            waited = now - p["t_first"] >= EXTRA_S                     # a second (processed) version?
            if not (stable and waited and (not pending or now - p["t"] > 8.0)):
                continue
            final = max(cand, key=lambda n: (cand[n][1], cand[n][2]))  # newest write = the processed one
            code, data = _run(["exec-out", f"cat {DCIM}/{final}"], 30, binary=True)
            self._delete(list(cand))
            known |= set(cand)
            done.append(p)
            if code != 0 or len(data) < 1000 or data[-2:] != b"\xff\xd9":
                self._ev.emit("fail", (p["cid"], "העתקת התמונה מהטלפון נכשלה"))
                continue
            # learn the phone-clock offset from the matches (both clocks are network-synced: ~0 s)
            self._offs.append(cand[final][0] - p["t"])
            self._offset = float(np.clip(np.median(self._offs[-9:]), -30, 30))
            self._ev.emit("capture", (p["cid"], data, p["t"]))
        self.inflight = [p for p in self.inflight if p not in done]
        self._known = known

    def _delete(self, names):
        """Removes the photos this program took (file + gallery entry)."""
        for n in names:
            if n.startswith("PXL_") or n.startswith(".pending"):
                sh(f"content delete --uri content://media/external/images/media --where \"_display_name='{n}'\"")
        left = [n for n in names if n in self._listing()]
        for n in left:
            sh(f"rm -f '{DCIM}/{n}'")

    # ------------------------------------------------------------------ live view
    def _screencap(self):
        code, data = _run(["exec-out", "screencap -p"], 10, binary=True)
        if code != 0 or not data:
            return None
        from PIL import Image
        try:
            return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception:
            return None

    def _previewer(self):
        while self._run_ev.is_set():
            if not (self.want_preview and self.connected):
                time.sleep(0.3); continue
            img = self._screencap()
            if img is None:
                time.sleep(0.5); continue
            top, bot = self.vf or (0, img.shape[0])
            a = np.ascontiguousarray(img[top:bot])
            q = QImage(a.data, a.shape[1], a.shape[0], 3 * a.shape[1], QImage.Format_RGB888).copy()
            if self.want_preview:
                self.preview.emit(q)

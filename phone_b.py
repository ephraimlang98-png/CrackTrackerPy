"""Side B from the phone (Crack Tracker 7.0): every new image in the side-A folder triggers a phone capture.

Re-uses the FatigueCam phone app and its PC link code (fc_proto / fc_link / fc_autosetup, copied unchanged).
Nothing is stored on the phone: each JPEG comes over the USB cable and is saved here, in the side-B folder,
under the side-A file's name (same stem, .jpg), so A <-> B pair by name.
Every trigger is logged to <side-B folder>/phone_trigger_log.csv (times of A, command, phone exposure, arrival).
"""
from __future__ import annotations

import csv
import os
import re
import threading
import time

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QImage

import crack_core as cc
import fc_link
from fc_autosetup import AutoSetup

LOG_NAME = "phone_trigger_log.csv"
SETUP_DIR = "_setup"          # base / calibration captures (not measured: list_images does not recurse)
LATE_DIR = "_late"            # captures that arrived after the timeout (machine probably running again)
POLL_S = 0.04                 # side-A folder polling
STABLE_S = 0.25               # size unchanged this long = the camera finished writing (non-JPEG / no end marker)
WATCH_GIVEUP_S = 15.0         # a new A file that never completes
LOG_COLS = ["id", "file_A", "file_B", "status", "A_shot", "A_created", "A_written", "A_detected", "A_complete",
            "complete_rule", "cmd_sent", "phone_exposure", "received", "saved", "A_shot_to_exposure_s",
            "A_created_to_exposure_s", "A_complete_to_exposure_s", "A_created_to_received_s",
            "A_complete_to_cmd_s", "cmd_to_received_s", "sharpness", "sharpness_vs_base", "rtt_ms", "focus_D", "iso", "exposure_us", "retries"]


def b_name(a_path: str) -> str:
    return os.path.splitext(os.path.basename(a_path))[0] + ".jpg"


def _ts(t):
    if t is None or not np.isfinite(t):
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"


def _dt(a, b):
    return "" if a is None or b is None else f"{b - a:.3f}"


def _complete_rule(path, size):
    """Is the file fully written? Decided from its own bytes where the format allows (no waiting):
    JPEG end marker, PNG IEND chunk, BMP size field in the header (the side-A camera writes BMP)."""
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "rb") as fh:
            if ext in (".jpg", ".jpeg"):
                fh.seek(max(0, size - 2))
                return "jpeg-end" if fh.read(2) == b"\xff\xd9" else None
            if ext == ".png":
                fh.seek(max(0, size - 8))
                return "png-end" if fh.read(8) == b"IEND\xaeB`\x82" else None
            if ext == ".bmp":
                h = fh.read(6)
                n = int.from_bytes(h[2:6], "little") if len(h) == 6 and h[:2] == b"BM" else 0
                return "bmp-size" if n > 54 and size >= n else None
    except OSError:
        pass
    return None


def a_shot_time(path):
    """Capture time written by the side-A camera software in the file name, e.g.
    Pic_20260609170658811-514.bmp = 2026-06-09 17:06:58.811 (None if the name has no such stamp)."""
    m = re.search(r"(20\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{3})", os.path.basename(path))
    if not m:
        return None
    try:
        y, mo, d, h, mi, s, ms = (int(x) for x in m.groups())
        return time.mktime((y, mo, d, h, mi, s, 0, 0, -1)) + ms / 1000.0
    except (ValueError, OverflowError):
        return None


def _readable(path):
    try:
        with open(path, "rb") as fh:
            fh.read(1)
        return True
    except OSError:
        return False


class FolderTrigger(threading.Thread):
    """Polls the side-A folder and calls on_ready(path, t_created, t_written, t_seen, t_ready, rule) once per
    new image, as soon as the camera has finished writing it (_complete_rule, else size stable for STABLE_S).
    Images already in the folder when it starts are never triggered."""

    def __init__(self, folder, on_ready):
        super().__init__(daemon=True)
        self.folder, self.on_ready = folder, on_ready
        self._stop = threading.Event()
        self.initial = set(self._names())
        self.seen = set(self.initial)
        self.cand = {}                     # name -> [t_seen, size, t_size_since]
        self.error = None
        self._dm, self._t_scan = self._dir_mtime(), time.time()

    def _dir_mtime(self):
        try:
            return os.stat(self.folder).st_mtime_ns
        except OSError:
            return None

    def _names(self):
        try:
            with os.scandir(self.folder) as it:
                self.error = None
                return [e.name for e in it
                        if os.path.splitext(e.name)[1].lower() in cc.IMAGE_EXT and e.is_file()]
        except OSError as ex:
            self.error = str(ex)
            return []

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            now = time.time()
            dm = self._dir_mtime()
            # listing thousands of images costs tens of ms: only when the folder changed (new / renamed
            # file), and once a second anyway
            if dm != self._dm or now - self._t_scan > 1.0:
                self._dm, self._t_scan = dm, now
                for n in self._names():
                    if n not in self.seen and n not in self.cand:
                        self.cand[n] = [now, -1, now]
            for n, c in list(self.cand.items()):
                p = os.path.join(self.folder, n)
                try:
                    st = os.stat(p)                 # live size (opens the file's metadata)
                except OSError:
                    if now - c[0] > WATCH_GIVEUP_S:
                        del self.cand[n]            # vanished (temporary name)
                    continue
                if st.st_size != c[1]:
                    c[1], c[2] = st.st_size, now
                if st.st_size <= 0:
                    continue
                rule = _complete_rule(p, st.st_size)
                if rule is None and now - c[2] >= STABLE_S and _readable(p):
                    rule = "size-stable"
                if rule:
                    del self.cand[n]
                    self.seen.add(n)
                    try:
                        self.on_ready(p, st.st_ctime, st.st_mtime, c[0], time.time(), rule)
                    except Exception:
                        pass
            self._stop.wait(POLL_S)


class PhoneB(QObject):
    """Phone link + side-A trigger + capture saving + trigger log. Lives in the GUI thread; the folder
    trigger sends the capture command from its own thread (no GUI delay)."""
    status = Signal(str, str)                 # level ok/warn/err, text
    preview = Signal(object)                  # QImage
    setup_captured = Signal(str, str, object)  # purpose (base/calib), saved path, capture message
    triggered = Signal(dict)                  # one finished trigger (log row)
    alert = Signal(str)                       # warning for the user
    log = Signal(str)
    focus_progress = Signal(str)
    focus_done = Signal(bool, str, object)
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        sim = os.environ.get("CT_PHONE_SIM")            # testing without the phone: fc_sim.py --port N
        self.mode, self.sim_port = ("sim", int(sim)) if sim else ("usb", 8766)
        self.link = None
        self.hello = None
        self.connected = False
        self.cfg = {"camera": {"ae": "manual", "exposure_ns": 500000, "iso": 400, "af": "manual", "awb": "lock"}}
        self.roi = (0.3, 0.35, 0.4, 0.3)
        self.focus_locked = None
        self.timeout_s = 5.0
        self.burst, self.gap_ms = 3, 60
        self.folder_a = self.folder_b = ""
        self.enabled = True
        self.trig: FolderTrigger | None = None
        self.recs = {}                        # id -> trigger record
        self.state = {}                       # B file name -> pending / ok / missed
        self.setup_req = {}                   # id -> (purpose, t_sent)
        self._lock = threading.Lock()
        self._next_id = 1
        self.offsets = []                     # (rtt, pc_time - phone_time) from pongs
        self.done = []                        # finished records of this session
        self.last_metrics = (np.nan, np.nan)
        self.ref_sharp = None                 # sharpness of the base capture (focus watch)
        self.blurred = 0
        self.want_preview = False
        self.t_connected = self.t_frame = 0.0
        self.cam_ok = None                    # False: connected, but the phone camera sends nothing
        self._t_kick = 0.0
        self.auto = AutoSetup(self)
        self.auto.progress.connect(self.focus_progress)
        self.auto.finished.connect(self._focus_finished)
        self._tick = QTimer(self); self._tick.setInterval(250); self._tick.timeout.connect(self._check_timeouts)
        self._tick.start()

    # ------------------------------------------------------------------ link
    def start(self):
        if self.link is not None:
            return
        self.link = fc_link.Link(self.mode, sim_port=self.sim_port)
        self.link.message.connect(self._on_msg)
        self.link.status.connect(self._on_status)
        self.link.start()

    def stop(self):
        self.disarm()
        self.auto.cancel()
        if self.link is not None:
            self.link.stop(); self.link.wait(3000)
            self.link = None
        self.connected = False

    def send(self, msg):
        return self.link is not None and self.link.connected and self.link.send(msg)

    def _on_status(self, level, text):
        was = self.connected
        self.connected = bool(self.link and self.link.connected)
        if level == "ok":
            self.t_connected, self.cam_ok = time.time(), None
            self._configure()
        if was and not self.connected:
            self.alert.emit(f"צד B: הטלפון התנתק ({text})")
            with self._lock:
                pend = [r for r in self.recs.values() if r["status"] == "pending"]
            for r in pend:
                self._finish(r, "missed: disconnected")
        self.status.emit(level, text)
        self.changed.emit()

    def _configure(self):
        """On every (re)connect: the phone app starts with its defaults - send our settings again
        (the locked focus above all)."""
        self.send({"type": "preview", "fps": 5, "quality": 60, "width": 960})
        x, y, w, h = self.roi
        self.send({"type": "roi", "x": x, "y": y, "w": w, "h": h})
        if "focus" in self.cfg["camera"] or self.cfg["camera"].get("zoom", 1.0) != 1.0:
            self.send({"type": "set", "camera": self.cfg["camera"]})

    def _on_msg(self, msg, payload, tr):
        ty = msg.get("type")
        if ty in ("metrics", "preview"):
            self.t_frame = tr
            if self.cam_ok is not True:
                if self.cam_ok is False:
                    self.log.emit("צד B: המצלמה בטלפון חזרה לעבוד")
                    self.status.emit("ok", "מחובר")
                self.cam_ok = True
        if ty == "metrics":
            for row in msg.get("rows", []):
                self.auto.feed(row[1], row[3])
                self.last_metrics = (row[1], row[3])
        elif ty == "preview":
            if self.want_preview and payload:
                q = QImage.fromData(payload, "JPG")
                if not q.isNull():
                    self.preview.emit(q)
        elif ty == "capture":
            self._on_capture(msg, payload, tr)
        elif ty == "hello":
            self.hello = msg
            self.log.emit(f"טלפון: {msg.get('device')} · אנדרואיד {msg.get('android')} · אפליקציה {msg.get('app')}")
        elif ty == "pong":
            rtt = self.link.rtt if self.link else np.nan
            if np.isfinite(rtt) and msg.get("t"):
                self.offsets.append((rtt, tr - rtt / 2 - msg["t"] * 1e-9))
                self.offsets = self.offsets[-30:]
        elif ty in ("error", "warn"):
            m = str(msg.get("msg", ""))
            self.log.emit(f"טלפון: {m}")
            if m.startswith("capture ") and "no frame" in m:
                try:
                    cid = int(m.split()[1].rstrip(":"))
                except ValueError:
                    cid = None
                r = self.recs.get(cid)
                if r is not None and r["status"] == "pending":
                    # the phone camera drops frames for a few seconds e.g. right after the screen turns off
                    # (face-unlock warm-up): one immediate retry, while the stop is surely still on
                    if not r["retries"] and time.time() - r["t_ready"] < self.timeout_s - 1.0 and \
                            self.send({"type": "capture", "id": cid, "burst": self.burst, "gap_ms": self.gap_ms}):
                        r["retries"] = 1
                        self.log.emit(f"צד B: הטלפון לא החזיר תמונה עבור {os.path.basename(r['file_A'])} – מנסה שוב")
                    else:
                        self._finish(r, "missed: phone returned no frame")

    def phone_to_pc(self, t_ns):
        if not self.offsets or not t_ns:
            return None
        return t_ns * 1e-9 + min(self.offsets)[1]

    @property
    def rtt(self):
        return self.link.rtt if self.link is not None else np.nan

    # ------------------------------------------------------------------ camera
    def set_roi(self, x, y, w, h):
        self.roi = (float(x), float(y), float(w), float(h))
        self.ref_sharp = None                 # sharpness of another area is not comparable
        self.send({"type": "roi", "x": x, "y": y, "w": w, "h": h})

    def set_focus(self, f):
        self.cfg["camera"].update(af="manual", focus=round(float(f), 3))
        self.send({"type": "set", "camera": self.cfg["camera"]})

    def set_camera(self, cam: dict):
        self.cfg["camera"].update({k: v for k, v in cam.items() if v is not None})
        if "focus" in self.cfg["camera"]:
            self.send({"type": "set", "camera": self.cfg["camera"]})

    def start_focus(self):
        self.auto.start()

    def _focus_finished(self, ok, msg, res):
        if ok:
            self.focus_locked = res["focus"]
        self.focus_done.emit(ok, msg, res)
        self.changed.emit()

    def capture_setup(self, purpose):
        if not self.connected:
            self.alert.emit("אין חיבור לטלפון"); return False
        with self._lock:
            cid = self._next_id; self._next_id += 1
        self.setup_req[cid] = (purpose, time.time())
        return self.send({"type": "capture", "id": cid, "burst": self.burst, "gap_ms": self.gap_ms})

    # ------------------------------------------------------------------ trigger
    def arm(self, folder_a, folder_b):
        """(Re)starts the side-A folder trigger. Images already there are not triggered."""
        self.disarm()
        self.folder_a, self.folder_b = folder_a or "", folder_b or ""
        if not (self.enabled and self.folder_a and self.folder_b and os.path.isdir(self.folder_a)):
            self.changed.emit(); return
        os.makedirs(self.folder_b, exist_ok=True)
        self.trig = FolderTrigger(self.folder_a, self._fire)
        self.trig.start()
        self.log.emit(f"צד B: הטריגר פעיל – כל תמונה חדשה ב-{self.folder_a} תצלם בטלפון")
        self.changed.emit()

    def disarm(self):
        if self.trig is not None:
            self.trig.stop()
            self.trig = None
            self.changed.emit()

    @property
    def armed(self):
        return self.trig is not None

    def _fire(self, path, t_c, t_w, t_seen, t_ready, rule):
        """Runs in the folder-trigger thread."""
        with self._lock:
            cid = self._next_id; self._next_id += 1
        cam = self.cfg["camera"]
        rec = dict(id=cid, file_A=path, file_B=b_name(path), status="pending", t_shot=a_shot_time(path),
                   t_created=t_c, t_written=t_w,
                   t_seen=t_seen, t_ready=t_ready, rule=rule, t_sent=None, t_exposure=None, t_recv=None,
                   t_saved=None, sharp=None, rtt=self.rtt, focus=cam.get("focus"), iso=cam.get("iso"),
                   exposure_ns=cam.get("exposure_ns"), retries=0)
        with self._lock:
            self.recs[cid] = rec
            self.state[rec["file_B"]] = "pending"
        rec["t_sent"] = time.time()
        if not self.send({"type": "capture", "id": cid, "burst": self.burst, "gap_ms": self.gap_ms}):
            rec["t_sent"] = None
            self._finish(rec, "missed: phone not connected")

    def _save(self, path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)

    def _on_capture(self, msg, payload, tr):
        cid = msg.get("id")
        if cid in self.setup_req:
            purpose, _ = self.setup_req.pop(cid)
            p = os.path.join(self.folder_b or os.path.expanduser("~"), SETUP_DIR,
                             f"{purpose}_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
            try:
                self._save(p, payload)
            except OSError as ex:
                self.alert.emit(f"לא ניתן לשמור את תמונת הכיול: {ex}"); return
            self.setup_captured.emit(purpose, p, msg)
            return
        rec = self.recs.get(cid)
        if rec is None:
            return
        t_exp = self.phone_to_pc(msg.get("t"))
        if rec["status"] != "pending":           # too late: kept aside, not measured
            p = os.path.join(self.folder_b, LATE_DIR, rec["file_B"])
            try:
                self._save(p, payload)
            except OSError:
                pass
            late = dict(rec, status=f"late ({tr - rec['t_sent']:.1f} s, saved in {LATE_DIR})" if rec["t_sent"] else "late",
                        t_recv=tr, t_exposure=t_exp, sharp=msg.get("sharp"))
            self._write_log(late)
            self.log.emit(f"צד B: תמונה מאוחרת {rec['file_B']} נשמרה ב-{LATE_DIR} (לא נמדדת)")
            return
        rec.update(t_recv=tr, t_exposure=t_exp, sharp=msg.get("sharp"), rtt=self.rtt)
        try:
            self._save(os.path.join(self.folder_b, rec["file_B"]), payload)
        except OSError as ex:
            self._finish(rec, f"missed: cannot save ({ex})"); return
        rec["t_saved"] = time.time()
        self._finish(rec, "ok")
        self._check_sharpness(rec)

    def _check_sharpness(self, rec):
        """Focus is locked: if the specimen moved (or the focus is off) the captures get blurred.
        Warns when 3 captures in a row are less than half as sharp as the base image."""
        if not self.ref_sharp or rec["sharp"] is None:
            return
        rel = rec["sharp"] / self.ref_sharp
        self.blurred = self.blurred + 1 if rel < 0.5 else 0
        if self.blurred == 3:
            self.alert.emit(f"צד B: 3 תמונות ברצף פחות חדות מתמונת הבסיס ({rel:.0%}) – אולי הדגם זז או שהפוקוס "
                            "לא מתאים. בדוק בתצוגה חיה; אם צריך – כוון פוקוס מחדש וצלם כיול מחדש.")

    def _finish(self, rec, status):
        with self._lock:
            if rec["status"] != "pending":
                return
            rec["status"] = status
            self.state[rec["file_B"]] = "ok" if status == "ok" else "missed"
            self.done.append(rec)
        self._write_log(rec)
        self.triggered.emit(rec)
        if status != "ok":
            self.alert.emit(f"צד B: פספוס – {os.path.basename(rec['file_A'])}: {status}")
        self.changed.emit()

    def _camera_stalled(self, first):
        """Connected but no camera frames: Android blocks the camera when the app was started with the
        screen locked (or another app took the camera). Says so, and brings the app to the front, which
        restores the camera once the phone is unlocked. Runs in its own thread (adb calls)."""
        _, out = fc_link.adb("shell", "dumpsys", "window")
        locked = "isKeyguardShowing=true" in out
        if first:
            msg = ("צד B: הטלפון מחובר אבל המצלמה חסומה כי מסך הטלפון נעול. פתח את נעילת הטלפון – "
                   "האפליקציה תעלה לבד, ואחר כך אפשר לכבות את המסך." if locked else
                   "צד B: הטלפון מחובר אבל המצלמה לא שולחת תמונה (אפליקציה אחרת משתמשת במצלמה?) – מפעיל מחדש את האפליקציה")
            self.alert.emit(msg)
            self.status.emit("err", "מחובר, אבל המצלמה חסומה – פתח את נעילת הטלפון" if locked else
                             "מחובר, אבל המצלמה לא שולחת תמונה")
        if not locked:
            fc_link.adb("shell", "am", "start", "-n", f"{fc_link.APP_ID}/.MainActivity")

    def _check_timeouts(self):
        now = time.time()
        if self.connected and self.cam_ok is not True and now - max(self.t_connected, self.t_frame) > 5.0 \
                and now - self._t_kick > 10.0 and self.mode == "usb":
            first = self.cam_ok is None
            self.cam_ok, self._t_kick = False, now
            threading.Thread(target=self._camera_stalled, args=(first,), daemon=True).start()
        elif self.connected and self.cam_ok and now - self.t_frame > 5.0:
            self.cam_ok = None                   # frames stopped: checked again on the next tick
            self.t_connected = now - 5.0
        with self._lock:
            late = [r for r in self.recs.values() if r["status"] == "pending" and r["t_sent"]
                    and now - r["t_sent"] > self.timeout_s]
        for r in late:
            self._finish(r, f"missed: no image within {self.timeout_s:g} s")
        for cid, (purpose, t) in list(self.setup_req.items()):
            if now - t > max(8.0, self.timeout_s):
                self.setup_req.pop(cid, None)
                self.alert.emit("תמונת הכיול לא הגיעה מהטלפון – נסה שוב")

    def b_state(self, a_path):
        """For the measuring worker: is side B of this A image ok / missed / pending / none (never triggered)?"""
        n = b_name(a_path)
        st = self.state.get(n)
        if st:
            return st
        if self.folder_b and os.path.exists(os.path.join(self.folder_b, n)):
            return "ok"
        t = self.trig
        if t is not None and os.path.normcase(os.path.dirname(a_path)) == os.path.normcase(self.folder_a) \
                and os.path.basename(a_path) not in t.initial:
            try:
                if time.time() - os.path.getmtime(a_path) < WATCH_GIVEUP_S:
                    return "pending"                 # the trigger has not seen it complete yet
            except OSError:
                pass
            return "missed"
        return "none"

    # ------------------------------------------------------------------ log
    def log_path(self):
        return os.path.join(self.folder_b, LOG_NAME) if self.folder_b else ""

    def _write_log(self, r):
        p = self.log_path()
        if not p:
            return
        row = [r["id"], os.path.basename(r["file_A"]), r["file_B"] if r["status"] == "ok" else "", r["status"],
               _ts(r["t_shot"]), _ts(r["t_created"]), _ts(r["t_written"]), _ts(r["t_seen"]), _ts(r["t_ready"]), r["rule"],
               _ts(r["t_sent"]), _ts(r["t_exposure"]), _ts(r["t_recv"]), _ts(r["t_saved"]),
               _dt(r["t_shot"], r["t_exposure"]), _dt(r["t_created"], r["t_exposure"]), _dt(r["t_ready"], r["t_exposure"]),
               _dt(r["t_created"], r["t_recv"]),
               _dt(r["t_ready"], r["t_sent"]), _dt(r["t_sent"], r["t_recv"]),
               "" if r["sharp"] is None else f"{r['sharp']:.1f}",
               "" if r["sharp"] is None or not self.ref_sharp else f"{r['sharp'] / self.ref_sharp:.2f}",
               "" if r["rtt"] is None or not np.isfinite(r["rtt"]) else f"{r['rtt'] * 1000:.0f}",
               "" if r["focus"] is None else r["focus"], "" if r["iso"] is None else r["iso"],
               "" if r["exposure_ns"] is None else f"{r['exposure_ns'] / 1000:.0f}", r.get("retries", 0)]
        with self._lock:
            try:
                new = not os.path.exists(p)
                with open(p, "a", newline="", encoding="utf-8-sig") as fh:
                    w = csv.writer(fh)
                    if new:
                        w.writerow(LOG_COLS)
                    w.writerow(row)
            except OSError as ex:
                self.log.emit(f"לא ניתן לכתוב ללוג הטריגרים: {ex}")

    def summary(self):
        """Counters and timing of this session's triggers."""
        with self._lock:
            done = list(self.done)
            pend = sum(r["status"] == "pending" for r in self.recs.values())
        ok = [r for r in done if r["status"] == "ok"]
        exp = [r["t_exposure"] - r["t_created"] for r in ok if r["t_exposure"] is not None]
        cexp = [r["t_exposure"] - r["t_ready"] for r in ok if r["t_exposure"] is not None]
        sexp = [r["t_exposure"] - r["t_shot"] for r in ok if r["t_exposure"] is not None and r["t_shot"] is not None]
        rec = [r["t_recv"] - r["t_created"] for r in ok]
        return dict(n=len(done) + pend, ok=len(ok), missed=len(done) - len(ok), pending=pend,
                    exp_med=float(np.median(exp)) if exp else np.nan, exp_max=float(np.max(exp)) if exp else np.nan,
                    cexp_med=float(np.median(cexp)) if cexp else np.nan,
                    cexp_max=float(np.max(cexp)) if cexp else np.nan,
                    sexp_med=float(np.median(sexp)) if sexp else np.nan,
                    sexp_max=float(np.max(sexp)) if sexp else np.nan,
                    recv_med=float(np.median(rec)) if rec else np.nan, recv_max=float(np.max(rec)) if rec else np.nan,
                    last=done[-1] if done else None)

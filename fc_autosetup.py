"""Automatic camera setup, done by the program (not the user):
1. exposure: keep the short exposure time (sharp images on a vibrating frame) and adjust ISO until the
   ROI brightness is on target; warn if there is not enough light.
2. focus: sweep the whole focus range, measure the ROI sharpness at each step, refine around the best,
   lock there. Focus then stays locked for the whole run (constant magnification = valid um/px)."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

SETTLE_S = 0.30          # lens / exposure settling after a change (frames discarded)
MEASURE_S = 0.25         # frames averaged per step
TARGET_BRIGHT = 120.0    # ROI mean on 0..255


class AutoSetup(QObject):
    progress = Signal(str)
    finished = Signal(bool, str, object)       # ok, message, result dict

    def __init__(self, engine):
        super().__init__()
        self.e = engine
        self.samples = []                      # (sharp, bright) of the current step
        self._t = QTimer(self); self._t.setSingleShot(True); self._t.timeout.connect(self._next)
        self.running = False
        self._collect = False
        self.warn = None
        self.curve = []

    # ------------------------------------------------------------------ public
    def start(self):
        caps = (self.e.hello or {}).get("caps", {})
        if not self.e.link or not self.e.link.connected:
            self.finished.emit(False, "אין חיבור לטלפון", None); return
        self.iso_rng = caps.get("iso") or [50, 3200]
        self.exp_rng = caps.get("exposure_ns") or [100000, 100000000]
        self.fmax = float(getattr(self.e, "focus_max", None) or caps.get("min_focus_diopter") or 10.0)
        cam = self.e.cfg["camera"]
        self.exposure = int(cam.get("exposure_ns", 500000))
        self.iso = int(cam.get("iso", 400))
        self.focus = float(cam.get("focus", self.fmax / 2))
        self.running = True; self.warn = None
        self.phase = "exposure"; self.iter = 0; self.curve = []
        self._set(ae="manual", af="manual", exposure_ns=self.exposure, iso=self.iso, focus=self.focus)
        self.progress.emit("כיוון אוטומטי: חשיפה...")
        self._wait()

    def cancel(self):
        self.running = False; self._t.stop()

    def feed(self, sharp, bright):
        """Called by the engine for every camera metrics row."""
        if self.running and self._collect:
            self.samples.append((sharp, bright))

    # ------------------------------------------------------------------ steps
    def _set(self, **kw):
        self.e.cfg["camera"].update(kw)
        self.e.link.send({"type": "set", "camera": self.e.cfg["camera"]})

    def _wait(self):
        """Discard frames while settling, then collect MEASURE_S of frames."""
        self._collect = False; self.samples = []
        QTimer.singleShot(int(SETTLE_S * 1000), self._begin_collect)

    def _begin_collect(self):
        if not self.running:
            return
        self.samples = []; self._collect = True
        self._t.start(int(MEASURE_S * 1000))

    def _measure(self):
        if len(self.samples) < 2:
            return None, None
        a = np.array(self.samples, float)
        return float(np.median(a[:, 0])), float(np.median(a[:, 1]))

    def _next(self):
        if not self.running:
            return
        self._collect = False
        sharp, bright = self._measure()
        if sharp is None:
            self._fail("לא מגיעים נתונים מהמצלמה"); return
        if self.phase == "exposure":
            self.iter += 1
            r = TARGET_BRIGHT / max(bright, 1.0)
            if 0.85 < r < 1.18 or self.iter > 8:
                self._start_focus(bright); return
            iso = int(np.clip(self.iso * r, *self.iso_rng))
            if iso == self.iso:                      # ISO at its limit
                if r > 1 and self.iso >= self.iso_rng[1]:
                    self.warn = f"חסר אור: גם ב-ISO {self.iso} הבהירות {bright:.0f}/255. להגביר תאורה (או זמן חשיפה ארוך יותר – פחות חד ברעד)."
                    self._start_focus(bright); return
                if r < 1:                            # too bright even at the lowest ISO: shorter exposure
                    self.exposure = int(np.clip(self.exposure * r, *self.exp_rng))
                    self._set(exposure_ns=self.exposure)
                    self._wait(); return
                self._start_focus(bright); return
            self.iso = iso
            self._set(iso=iso)
            self.progress.emit(f"כיוון אוטומטי: חשיפה – ISO {iso} (בהירות {bright:.0f})")
            self._wait()
        elif self.phase in ("coarse", "fine"):
            self.curve.append((self.steps[self.k], sharp))
            self.k += 1
            if self.k < len(self.steps):
                self._set(focus=float(self.steps[self.k]))
                self.progress.emit(f"כיוון אוטומטי: פוקוס {'גס' if self.phase == 'coarse' else 'עדין'} "
                                   f"{self.k}/{len(self.steps)} (חדות {sharp:.1f})")
                self._wait(); return
            pts = sorted(self.curve)
            best_f, best_s = max(pts, key=lambda x: x[1])
            if self.phase == "coarse":
                step = self.fmax / (len(self.steps) - 1)
                lo, hi = max(0.0, best_f - step), min(self.fmax, best_f + step)
                self.phase = "fine"; self.steps = list(np.linspace(lo, hi, 11)); self.k = 0
                self._set(focus=float(self.steps[0])); self._wait(); return
            self._done(best_f, best_s)

    def _start_focus(self, bright):
        self.bright = bright
        self.phase = "coarse"
        self.steps = list(np.linspace(0.0, self.fmax, 25)); self.k = 0
        self._set(focus=float(self.steps[0]))
        self.progress.emit("כיוון אוטומטי: סורק פוקוס...")
        self._wait()

    def _done(self, f, s):
        self.running = False
        self._set(focus=round(float(f), 3))
        sh = np.array([c[1] for c in self.curve])
        contrast = float(s / max(np.median(sh), 1e-9))
        res = {"focus": round(float(f), 3), "sharp": s, "iso": self.iso, "exposure_ns": self.exposure,
               "bright": getattr(self, "bright", None), "contrast": contrast, "curve": self.curve}
        msg = f"פוקוס ננעל על {f:.2f} דיופטר (חדות {s:.1f}, פי {contrast:.1f} מהממוצע) · ISO {self.iso} · חשיפה {self.exposure / 1000:.0f} µs"
        if f >= 0.93 * self.fmax:
            msg += (f" · ⚠ הפוקוס בקצה הטווח – הדגם קרוב מדי לעדשה (מתמקדת רק מ-{100 / self.fmax:.0f} ס״מ): "
                    f"להרחיק את הטלפון {'10–20' if self.fmax < 3 else '1–2'} ס״מ ולכוון שוב")
        elif f <= 0.03 * self.fmax:
            msg += " · ⚠ הפוקוס על אינסוף – כנראה שאזור העניין לא על הדגם"
        if contrast < 1.5:
            msg += " · ⚠ שיא חדות חלש – לוודא שאזור העניין על הדגם ושהמרחק בטווח העדשה"
        if getattr(self, "warn", None):
            msg += " · ⚠ " + self.warn
        self.finished.emit(True, msg, res)

    def _fail(self, why):
        self.running = False
        self.finished.emit(False, "הכיוון האוטומטי נכשל: " + why, None)

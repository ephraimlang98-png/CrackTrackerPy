"""CT Tracker 2.0 - fatigue crack growth of C(T) specimens (ASTM E647), one app for the whole test:
1. specimen photo on the table + measured height H and thickness B -> all dimensions vs. the drawing
2. measurement setup: side A = machine camera folder (turned so that the notch is on the right),
   side B = phone on the USB cable (camera tuned and locked, or automatic). Specimen edges, mm grid and
   notch tip found automatically; the start point is on the load line, W from the back edge.
3. measuring: every new side-A image triggers the phone (side B), then both are measured; a growth
   guard keeps the tip in place when the crack did not grow.
4. results / E647 points, 5. alerts, 6. load calibration.
Built on Crack Tracker 7.0 (app.py, unchanged); 1.4 is kept as ct_app_v14_backup.py."""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QKeySequence, QPainterPath, QPen, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame, QGraphicsItem, QGraphicsPathItem,
                               QGraphicsTextItem, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QScrollArea, QTableWidget,
                               QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

import app as base
import crack_core as cc
import ct_core as ct
import ct_detect2 as cdet2
import ct_detect3 as cdet3
import ct_frame as cf
import ct_track as tk
import gcam_b
from app import (PAGE_ALERTS, PAGE_MEASURE, PAGE_RESULTS, PAGE_SETUP, SIDE_NAMES, SetupView, _btn, _fmt, _ltr,
                 _num, _repolish)

APP_NAME = "CT Tracker"
VERSION = "2.2"
PAGE_SPEC, PAGE_LOADS = 4, 5
NAV_ORDER = [(PAGE_SPEC, "דגם ומידות"), (PAGE_SETUP, "הגדרת מדידה"), (PAGE_MEASURE, "מדידה"),
             (PAGE_RESULTS, "תוצאות וייצוא"), (PAGE_ALERTS, "התראות"), (PAGE_LOADS, "כיול עומסים")]
GRID_MINOR = QColor(125, 211, 252, 55)
GRID_MAJOR = QColor(125, 211, 252, 120)
C_LOAD = "#facc15"
C_EDGE = "#4ade80"

# Illustrative values only; no laboratory drawing or tolerances are distributed.
SPEC_ROWS = [("H", "Height H (side to side, caliper)", ct.NOMINAL["H"], ct.TOL["H"]),
             ("W125", "Overall width (front to back edge)", ct.NOMINAL["W125"], ct.TOL["W125"]),
             ("W", "W (load line to back edge)", ct.NOMINAL["W"], ct.TOL["W"]),
             ("F", "Front edge to load line", ct.NOMINAL["W125"] - ct.NOMINAL["W"], (-0.5, 0.5)),
             ("an", "a_n (load line to notch tip)", ct.NOMINAL["an"], ct.TOL["an"]),
             ("D1", "Hole 1 diameter", ct.NOMINAL["D1"], ct.TOL["D1"]),
             ("D2", "Hole 2 diameter", ct.NOMINAL["D2"], ct.TOL["D2"]),
             ("S", "Hole spacing (centre to centre)", ct.NOMINAL["S"], ct.TOL["S"]),
             ("h1", "Hole 1 to notch axis", ct.NOMINAL["S"] / 2, (-0.5, 0.5)),
             ("h2", "Hole 2 to notch axis", ct.NOMINAL["S"] / 2, (-0.5, 0.5)),
             ("off", "Notch offset from specimen centre", ct.NOMINAL["off"], ct.TOL["off"]),
             ("B", "Thickness B (caliper)", ct.NOMINAL["B"], ct.TOL["B"])]
SPEC_NOM = {k: n for k, _, n, _ in SPEC_ROWS}
SPEC_INPUT = ("H", "B")                 # entered by the user (caliper), not from the photo

# ---- every image is read 'upright' (notch on the right): quarter turns per image folder
ROT = {}
_read_raw = cc.read_gray


def _norm(p):
    return os.path.normcase(os.path.abspath(p))


def rot_of(path):
    d = _norm(os.path.dirname(path))
    for f, k in ROT.items():
        if d == f or d.startswith(f + os.sep):
            return k
    return 0


def read_upright(path):
    return cf.upright(_read_raw(path), rot_of(path))


cc.read_gray = read_upright


def to_raw(pt, k, raw_shape):
    """Point of an upright image (np.rot90 k) -> the same point in the raw image."""
    x, y = float(pt[0]), float(pt[1]); h, w = raw_shape
    k %= 4
    if k == 1:
        return w - 1 - y, x
    if k == 2:
        return w - 1 - x, h - 1 - y
    if k == 3:
        return y, h - 1 - x
    return x, y


# ---- own settings, E647 alert defaults, extra columns (a/W and the back-edge check)
base.QSettings = lambda *a: QSettings("CTTracker", "CTTracker")
PHONE_ZOOMS = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0)   # phone camera zoom ratios; 5x and up = telephoto lens on the Pixel
PHONE_TELE_MIN_FOCUS_D = 1.25    # Pixel 8 Pro telephoto (physical camera 4): focuses from 80 cm only
base.ALERT_DEFAULTS.update(diff_on=True, diff_mode="a", diff_thr=1.25, ang_new_on=True, ang_new_thr=10.0,
                           ang_all_on=True, ang_all_thr=10.0,
                           # growth alerts (popup + beep): crack start, every step of growth, target length
                           start_on=True, start_thr=0.10, step_on=True, step_mm=0.5,
                           len_on=False, len_mode="a", len_val=15.0, beep_on=True, beep_repeat=True)
base.HEADERS.insert(len(base.HEADERS) - 1, "a/W")
base.HEADERS.insert(len(base.HEADERS) - 1, "a מהקצה האחורי")
C_AW = len(base.HEADERS) - 3
C_AB = len(base.HEADERS) - 2
C_ST = len(base.HEADERS) - 1
base.C_ST = C_ST
BACK_TOL = 0.10                         # mm: a from the load line vs. a from the back edge
_status_he0 = base.status_he


def status_he(st):
    if st == tk.NO_GROWTH:
        return "תקין · לא גדל"
    if st == "uncertain (re-anchored)":
        return "לא בטוח · עוגן עודכן"
    return _status_he0(st)


base.status_he = status_he


def _a_back_mm(res):
    return getattr(res, "a_back", np.nan) if res is not None else np.nan


class CTResultsModel(base.ResultsModel):
    def data(self, idx, role=Qt.DisplayRole):
        r, c = idx.row(), idx.column()
        R = self.m.R
        if c in (C_AW, C_AB) and r < len(R["k"]):
            if role == Qt.TextAlignmentRole:
                return int(Qt.AlignCenter)
            if c == C_AW and role == Qt.DisplayRole:
                W = self.m.spec_dim("W")
                a = R["a"][r]
                return _ltr(f"{a / W:.4f}") if np.isfinite(a) and W else ""
            if c == C_AB:
                res = R["res"][r]
                vals = [(_a_back_mm(res[i]), res[i].a if res[i] is not None else np.nan) for i in self.m._active()]
                if role == Qt.DisplayRole:
                    txt = [_fmt(ab) for ab, _ in vals]
                    return _ltr("  |  ".join(txt)) if any(txt) else ""
                bad = [abs(ab - a) > BACK_TOL for ab, a in vals if np.isfinite(ab) and np.isfinite(a)]
                if role == Qt.ForegroundRole and any(bad):
                    return QBrush(QColor("#b45309"))
                if role == Qt.ToolTipRole:
                    return ("a לפי הקצה האחורי שזוהה מחדש בתמונה הזו: W − (קצה אחורי ← קצה הסדק).\n"
                            f"צריך להיות שווה ל-a מקו העומס; הפרש מעל {BACK_TOL:.2f} מ״מ מסומן בכתום "
                            "(הדגם זז / היישור או נקודת ההתחלה לא במקום).")
        if role == Qt.ToolTipRole and r < len(R["k"]) and c == C_ST:
            notes = [getattr(x, "guard", "") for x in R["res"][r] if x is not None]
            notes = [n for n in notes if n]
            if notes:
                return R["status"][r] + "\n" + "\n".join(notes)
        if role == Qt.ToolTipRole and r in self.m.gap_tip:
            return self.m.gap_tip[r]
        return super().data(idx, role)


base.ResultsModel = CTResultsModel


# =============================================================================
# Workers
# =============================================================================
class CTWorker(base.Worker):
    """Crack Tracker's measuring loop + the growth guard + the back-edge check, per side.
    revised(row, side, result): an earlier row corrected by the guard (growth not confirmed)."""
    revised = Signal(int, int, object)

    def __init__(self, sides, live, start_index, guard=True):
        super().__init__(sides, live, start_index)
        self.guards = {s["index"]: tk.GrowthGuard(s) for s in sides} if guard else {}

    def run(self):
        super().run()
        # no more images: growth still waiting for its confirming images is not confirmed (the rows go
        # back to the confirmed tip before the final alert check)
        for idx, g in self.guards.items():
            try:
                for row, rr in g.finish():
                    self.revised.emit(row, idx, rr)
            except Exception as ex:
                self.message.emit(f"שגיאה בסיום שומר הצמיחה: {ex}")

    def _do(self, k, lists):
        out = dict(k=k, file=["", ""], img=[None, None], res=[None, None], sens=[None, None])
        want_img = time.time() - self._last_img > 0.3
        for j, s in enumerate(self.sides):
            idx = s["index"]
            out["sens"][idx] = "fit" if s.get("fitted") else s.get("sens_now", 0)
            img = None
            try:
                use_prior = s["P"].get("use_prior", True)
                if s.get("notch") and not s.get("noise_from") and not s.get("_noise_try") \
                        and os.path.normcase(lists[j][k]) != os.path.normcase(s.get("base_file") or "") \
                        and os.path.exists(lists[j][k]):
                    # set up with only the baseline in the folder (live start): the threshold came from
                    # the scratches' own contrast - far too high on bare metal. Take the new-darkening
                    # noise from this first image after the baseline instead (no crack yet).
                    s["_noise_try"] = True
                    try:
                        if cc.notch_noise_threshold(s, cc.load_frame(lists[j][k], s), os.path.basename(lists[j][k])):
                            s["thr_eff0"] = s.get("thr_eff")
                    except Exception:
                        pass
                if not os.path.exists(lists[j][k]):
                    # an image that never arrived (side B: the phone missed it) is a missed row, not
                    # an 'unreadable image' error
                    r = cc.FrameResult(lists[j][k], False, "no phone image", np.nan, np.nan)
                else:
                    r = cc.measure_frame(lists[j][k], s, keep_image=True,
                                         prev_reach=self.prev[idx] if use_prior else None)
                    img = r.img
                    g = self.guards.get(idx)
                    if g is not None:
                        r, revs = g.apply(r, img, k)
                        for row, rr in revs:
                            if row is not None:
                                self.revised.emit(row, idx, rr)
                    r.a_back = back_check(s, r, img)
                if r.ok and np.isfinite(r.reach):
                    self.prev[idx] = r.reach
            except Exception as ex:
                r = cc.FrameResult(lists[j][k], False, f"unreadable image ({type(ex).__name__})", np.nan, np.nan)
            out["img"][idx] = img if want_img else None
            r.img = None
            out["res"][idx] = r
            out["file"][idx] = os.path.basename(lists[j][k])
        if want_img:
            self._last_img = time.time()
        self.result.emit(out)


base.Worker = CTWorker


def back_check(prep, res, img):
    """a from the back edge found again in this image: W - (back edge -> tip) along the crack axis."""
    if img is None or not res.ok or prep.get("back_axis") is None or not np.all(np.isfinite(res.tip_xy)):
        return np.nan
    try:
        k = 1.0 / prep["mm_per_px"]
        bp = cf.back_edge_at(img, res.d, prep["back_axis"], prep["u"], k)
        if bp is None:
            return np.nan
        return float(prep["W_mm"] - np.dot(bp - np.asarray(res.tip_xy, float), prep["u"]) / k)
    except Exception:
        return np.nan


class FrameWorker(QThread):
    """Specimen edges + notch tip on a measuring image (upright, or every orientation for the phone)."""
    done = Signal(object)

    def __init__(self, img, dims, any_orient):
        super().__init__()
        self.img, self.dims, self.any = img, dims, any_orient

    def run(self):
        try:
            if self.any:
                k, r = cf.detect_any_orientation(self.img, *self.dims)
                r = dict(r or {}); r["rot"] = k; r["any"] = True
            else:
                r = cf.detect(self.img, *self.dims); r["rot"] = None; r["any"] = False
        except Exception as ex:
            r = dict(notes=[f"שגיאה בזיהוי: {ex}"], rot=None, any=self.any)
        self.done.emit(r)


class DetectWorker(QThread):
    """Specimen photo on the table: holes, notch tip and the four edges."""
    done = Signal(object, object)

    def __init__(self, img, path=None):
        super().__init__()
        self.img, self.path = img, path

    def run(self):
        try:
            feats, notes, rgb = {}, [], None
            if self.path:
                try:
                    rgb = np.asarray(Image.open(self.path).convert("RGB"))
                    if rgb.shape[:2] != self.img.shape[:2]:
                        rgb = None
                except Exception:
                    rgb = None
            ok = lambda f: "hole1" in f and "hole2" in f
            colour = rgb is not None and cdet3.is_color(rgb)
            if colour:                           # phone photo: geometry first + colour edges
                feats, notes = cdet3.detect_specimen_v3(rgb)
            if not ok(feats):
                feats, notes = cdet2.detect_specimen_v2(self.img)
            if not ok(feats) and rgb is not None and not colour:
                f3, n3 = cdet3.detect_specimen_v3(rgb)
                if ok(f3):
                    feats, notes = f3, n3
        except Exception as ex:
            feats, notes = {}, [f"שגיאה בזיהוי: {ex}"]
        self.done.emit(feats, notes)


# =============================================================================
# Drawing helpers
# =============================================================================
def _label(view, x, y, text, color="#ffffff", bg="rgba(15,23,42,190)", size=9):
    it = QGraphicsTextItem()
    it.setHtml(f"<div style='background:{bg}; color:{color}; font-family:Segoe UI; font-size:{size}pt;"
               f" padding:1px 4px'>{text}</div>")
    it.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
    it.setPos(x, y)
    view._add(it, 12)
    return it


def _path_lines(view, segs, color, width=1.0, dash=False, z=4):
    if not segs:
        return
    path = QPainterPath()
    for a, b in segs:
        path.moveTo(a[0] + 0.5, a[1] + 0.5); path.lineTo(b[0] + 0.5, b[1] + 0.5)
    it = QGraphicsPathItem(path)
    pen = QPen(color, width, Qt.DashLine if dash else Qt.SolidLine); pen.setCosmetic(True)
    it.setPen(pen)
    view._add(it, z)


def draw_grid(view, origin, u, mmpp, a_min, a_max, lat_min, lat_max, labels=True):
    """Subtle 1-mm grid over the whole specimen (stronger every 5 mm), aligned with the crack axis u,
    origin on the load line. a runs along u (labels = a [mm] from the load line), lat across it."""
    if mmpp is None or not np.isfinite(mmpp) or origin is None:
        return
    k = 1.0 / mmpp
    u = np.asarray(u, float) / np.linalg.norm(u); v = np.array([-u[1], u[0]]); o = np.asarray(origin, float)
    a0, a1 = int(np.floor(a_min)), int(np.ceil(a_max))
    l0, l1 = lat_min, lat_max
    minor, major = [], []
    for a in range(a0, a1 + 1):
        seg = (o + a * k * u + l0 * k * v, o + a * k * u + l1 * k * v)
        (major if a % 5 == 0 else minor).append(seg)
    for t in range(int(np.ceil(l0)), int(np.floor(l1)) + 1):
        seg = (o + t * k * v + a_min * k * u, o + t * k * v + a_max * k * u)
        (major if t % 5 == 0 else minor).append(seg)
    _path_lines(view, minor, GRID_MINOR, 1.0)
    _path_lines(view, major, GRID_MAJOR, 1.0)
    _path_lines(view, [(o + l0 * k * v, o + l1 * k * v)], QColor(250, 204, 21, 190), 1.6, True, 5)
    if labels:
        for a in range(a0 - a0 % 5, a1 + 1, 5):
            if a_min <= a <= a_max:
                p = o + a * k * u + (l1 + 0.3) * k * v
                _label(view, p[0], p[1], f"a={a}" if a else "קו עומס", "#bae6fd", "rgba(15,23,42,140)", 7)


# =============================================================================
# Side panel of the measurement setup
# =============================================================================
class CTSidePanel(base.SidePanel):
    def __init__(self, idx, main):
        super().__init__(idx, main)
        lay = self.layout()
        self.g2.hide(); self.g3.hide()
        pos = lay.indexOf(self.g2)
        # ---- 2: specimen edges -> scale and load line
        self.c2 = QGroupBox("2 · זיהוי הדגם וקנה מידה")
        l2 = QVBoxLayout(self.c2)
        self.b_eauto = _btn("זהה אוטומטית: קצוות הדגם + קצה החריץ", "primary", slot=lambda: main.ct_auto(idx),
                            icon="SP_BrowserReload",
                            tip="מוצא את הקצה האחורי, שני הצדדים וקצה החריץ (הנקודה העמוקה של החריץ).\n"
                                "קנה מידה = המרחק בין הצדדים / הגובה שנמדד במסך 1;\n"
                                "קו העומס (נקודת ההתחלה) = W מהקצה האחורי.")
        l2.addWidget(self.b_eauto)
        row = QHBoxLayout()
        self.b_back = _btn("קצה אחורי", slot=lambda: main.ct_pick(idx, "ct_back"))
        self.b_s1 = _btn("צד 1", slot=lambda: main.ct_pick(idx, "ct_side1"))
        self.b_s2 = _btn("צד 2", slot=lambda: main.ct_pick(idx, "ct_side2"))
        for b in (self.b_back, self.b_s1, self.b_s2):
            b.setToolTip("תיקון ידני: 2 לחיצות על הקו, רחוקות זו מזו – הקו מיושר לשפה")
            row.addWidget(b)
        l2.addLayout(row)
        self.chk_mcal = QCheckBox("כיול ידני: מרחק ידוע במקום הגובה")
        self.chk_mcal.setToolTip("כשהצדדים לא נראים בתמונה (למשל צילום קרוב בטלפון): סמן מרחק ידוע בתמונה")
        self.chk_mcal.toggled.connect(lambda on: main.ct_mcal(idx, on))
        l2.addWidget(self.chk_mcal)
        self.w_mcal = QWidget(); lm = QVBoxLayout(self.w_mcal); lm.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout(); row.addWidget(QLabel("מרחק:")); row.addWidget(self.known, 1)
        row.addWidget(self.btn_cal); row.addWidget(self.btn_cal_del)
        lm.addLayout(row); lm.addWidget(self.cal_table)
        self.btn_cal.setText("סמן 2 קצוות")
        self.btn_cal_img.hide()
        l2.addWidget(self.w_mcal); self.w_mcal.hide()
        l2.addWidget(self.lbl_scale)
        lay.insertWidget(pos, self.c2)
        # ---- 3: notch tip + start point on the load line
        self.c3 = QGroupBox("3 · קצה החריץ ונקודת ההתחלה")
        l3 = QVBoxLayout(self.c3)
        row = QHBoxLayout()
        self.b_tip = _btn("סמן קצה חריץ ידנית", slot=lambda: main.ct_pick(idx, "ct_tip"))
        self.b_mouth = _btn("סמן כיוון החריץ", slot=lambda: main.ct_pick(idx, "ct_mouth"))
        self.b_mouth.setToolTip("רק כשהקצה האחורי לא נראה: לחץ על נקודה באמצע החריץ, לפני הקצה שלו")
        row.addWidget(self.b_tip); row.addWidget(self.b_mouth); l3.addLayout(row)
        row = QHBoxLayout()
        self.b_start = QToolButton(); self.b_start.setText("הזז נקודת התחלה ידנית")
        self.b_start.clicked.connect(lambda: main.ct_pick(idx, "ct_start"))
        self.b_auto = QToolButton(); self.b_auto.setText("חזור לאוטומטי")
        self.b_auto.clicked.connect(lambda: main.ct_start_auto(idx))
        row.addWidget(self.b_start); row.addWidget(self.b_auto); row.addStretch(1); l3.addLayout(row)
        self.lbl_start = QLabel(""); self.lbl_start.setObjectName("Muted"); self.lbl_start.setWordWrap(True)
        self.lbl_start.setTextFormat(Qt.RichText)
        l3.addWidget(self.lbl_start)
        row = QHBoxLayout(); row.addWidget(self.btn_roi); row.addStretch(1); l3.addLayout(row)
        l3.addWidget(self.lbl_crack)
        lay.insertWidget(pos + 1, self.c3)
        self.g4.setTitle("4 · רגישות הזיהוי")
        row = QHBoxLayout()
        row.addWidget(QLabel("איך הסדק נראה:"))
        self.pol = QComboBox()
        self.pol.addItem("אוטומטי – כהה או בהיר (מומלץ)", "auto")
        self.pol.addItem("כהה – צל על פני הדגם", "dark")
        self.pol.addItem("בהיר – אור מאחור עובר דרך הסדק", "bright")
        self.pol.setToolTip("סדק שנסגר (ליד הקצה) נראה בדרך כלל כקו כהה על פני הדגם. עם אור מאחורי הדגם, סדק\n"
                            "שנפתח לכל העובי מעביר אור ונראה בהיר. אוטומטי = שניהם, לכל קטע לפי מה שמופיע בו.")
        self.pol.currentIndexChanged.connect(lambda *_: main.set_polarity(idx))
        row.addWidget(self.pol, 1)
        self.g4.layout().insertLayout(0, row)
        if idx == 1:
            self._ct_phone_box()

    def _ct_phone_box(self):
        main = self.main
        pl = self.phone_box.layout()
        self.btn_pfocus.setText("כוון מצלמה ונעל")
        self.btn_pfocus.setToolTip("חשיפה (ISO על זמן חשיפה קצר – חד גם ברעד) ופוקוס: סריקה של כל הטווח\n"
                                   "ונעילה על החד ביותר באזור קצה החריץ. איזון לבן נעול.\n"
                                   "הכל נשאר נעול לכל הריצה – ההגדלה (קנה המידה) לא משתנה.")
        self.chk_autocam = QCheckBox("מצלמה אוטומטית (בלי נעילה) – הטלפון מכוון את עצמו בכל תמונה")
        self.chk_autocam.setToolTip("כמו מצלמת הטלפון הרגילה: פוקוס, חשיפה ואיזון לבן אוטומטיים בכל צילום.\n"
                                    "שים לב: פוקוס אוטומטי משנה מעט את ההגדלה – קנה המידה יכול לזוז מעט בין תמונות.")
        self.chk_autocam.toggled.connect(main.phone_autocam)
        pl.insertWidget(pl.indexOf(self.lbl_pfocus) + 1, self.chk_autocam)
        row = QHBoxLayout()
        row.addWidget(QLabel("זום:"))
        self.ph_zoom = QComboBox()
        for z in PHONE_ZOOMS:
            kind = "רחב" if z == 1 else ("עדשת טלפוטו – אופטי, חד" if z >= 5 else "חיתוך דיגיטלי")
            self.ph_zoom.addItem(f"{z:g}×  ({kind})", z)
        self.ph_zoom.setToolTip("זום של מצלמת הטלפון עצמה (לא הגדלה על המסך).\n"
                                "מ-5× הטלפון עובר לעדשת הטלפוטו – הגדלה אופטית אמיתית, חדה, מרחוק.\n"
                                "2×–3× = חיתוך מהעדשה הרגילה (פחות חד). אחרי שינוי: כוון מצלמה ונעל, וצלם תמונת בסיס מחדש.")
        self.ph_zoom.currentIndexChanged.connect(main.phone_zoom)
        row.addWidget(self.ph_zoom); row.addStretch(1)
        self.chk_gcam = QCheckBox("מצלמת גוגל של הטלפון (איכות מלאה – מומלץ)")
        self.chk_gcam.setToolTip("התוכנה מפעילה את אפליקציית המצלמה הרגילה של הטלפון: לוחצת על הצמצם בכל עצירה,\n"
                                 "מעתיקה את התמונה (עם העיבוד המלא של גוגל) למחשב ומוחקת אותה מהטלפון מיד.\n"
                                 "בתצוגה החיה: גלגלת = זום בטלפון, לחיצה = פוקוס בנקודה.\n"
                                 "כבוי = אפליקציית FatigueCam (חשיפה ופוקוס נעולים, כלום לא נשמר בטלפון).")
        self.chk_gcam.toggled.connect(main.phone_gcam)
        pl.insertWidget(pl.indexOf(self.lbl_conn) + 1 if pl.indexOf(self.lbl_conn) >= 0 else 0, self.chk_gcam)
        at = next(i for i in range(pl.count()) if pl.itemAt(i).layout() is not None
                  and pl.itemAt(i).layout().indexOf(self.btn_plive) >= 0)
        pl.insertLayout(at, row)
        self.btn_pbase.setText("צלם תמונה + זהה דגם וקצה חריץ")
        self.btn_pbase.setToolTip("תמונה אחת מהטלפון (לפני תחילת הסדק), כמו שהיא – בלי סיבוב.\n"
                                  "התוכנה מזהה את קצוות הדגם וקצה החריץ ומציירת את רשת המ״מ")

    def controls(self):
        c = super().controls() + [self.b_eauto, self.b_back, self.b_s1, self.b_s2, self.chk_mcal, self.b_tip,
                                  self.b_mouth, self.b_start, self.b_auto]
        return c

    def set_done(self, folder, scale, crack):
        for g, ok in ((self.g1, folder), (self.c2, scale), (self.c3, crack)):
            t = g.title().replace("✓ ", "")
            g.setTitle(("✓ " + t) if ok else t)
            g.setProperty("done", "true" if ok else "false"); _repolish(g)


# =============================================================================
# Main window
# =============================================================================
class CTMainWindow(base.MainWindow):
    gap_rows = frozenset()
    gap_tip = {}
    grid_on = True
    nav_pages = [p for p, _ in NAV_ORDER]

    def __init__(self):
        super().__init__()
        self.spec = dict(name="", image="", img=None, feats={}, H=np.nan, B=np.nan, gpx={}, pxmm=np.nan, dims={})
        self.loads = dict(stages=[list(s) for s in ct.DEFAULT_STAGES], W0=ct.NOMINAL["W"], B0=ct.NOMINAL["B"],
                          lbf=ct.LBF_TO_N, tol=2.0)
        self.grid_on = True
        self.gap_tip = {}
        self.gap_rows = set()
        self.e647_sel = None
        self._fw = {}
        self._cam_saved = None
        z = float(self.settings.value("phone_zoom", 1.0) or 1.0)
        zb = self.panels[1].ph_zoom
        zb.blockSignals(True); zb.setCurrentIndex(max(0, zb.findData(z))); zb.blockSignals(False)
        g = self.panels[1].chk_gcam
        g.blockSignals(True); g.setChecked(str(self.settings.value("phone_gcam", "true")).lower() == "true")
        g.blockSignals(False)
        for i in (0, 1):
            pb = self.panels[i].pol
            pb.blockSignals(True)
            pb.setCurrentIndex(max(0, pb.findData(self.settings.value(f"crack_polarity_{i}", "auto"))))
            pb.blockSignals(False)
        self._gcam_widgets()
        self.su.view.viewport().installEventFilter(self)
        self.stack.addWidget(self._build_spec_page())        # PAGE_SPEC
        self.stack.addWidget(self._build_loads_page())       # PAGE_LOADS
        self._extend_measure_page()
        self._extend_results_page()
        self._extend_alerts_page()
        sc = QShortcut(QKeySequence("Return"), self); sc.activated.connect(self._finish_pick)
        sc = QShortcut(QKeySequence("Enter"), self); sc.activated.connect(self._finish_pick)
        self._update_title()
        self._recompute_spec()
        self._goto(PAGE_SPEC)
        self.log(f"{APP_NAME} {VERSION} מוכן.")

    # ------------------------------------------------------------------ chrome
    def _build_header(self):
        h = super()._build_header()
        lay = h.layout()
        for b in self.nav_btns:                      # rebuild the navigation in C(T) order
            lay.removeWidget(b); self.nav.removeButton(b); b.deleteLater()
        logo_idx = 3
        btns = {}
        for n, (page, txt) in enumerate(NAV_ORDER):
            b = QPushButton(f"{n + 1}   {txt}"); b.setObjectName("Nav"); b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor); b.setToolTip(f"Ctrl+{n + 1}")
            self.nav.addButton(b, page); lay.insertWidget(logo_idx + n, b); btns[page] = b
        self.nav_btns = [btns[p] for p in range(len(NAV_ORDER))]
        self.logo_lbl = h.findChildren(QLabel)[0]
        self.logo_lbl.setText("CT Tracker")
        h.findChildren(QLabel)[1].setText(f"v{VERSION}")
        return h

    def _build_settings_tab(self):
        w = super()._build_settings_tab()
        g = w.findChild(QGroupBox)
        self.chk_guard = QCheckBox("שומר צמיחה: הסדק גדל רק כשיש התכהות חדשה מעבר לקצה הקודם")
        self.chk_guard.setChecked(True)
        self.chk_guard.setToolTip("משווה כל תמונה לתמונה האחרונה שבה אושרה צמיחה. בלי התכהות חדשה מעבר לקצה –\n"
                                  "הקצה נשאר במקום (\"תקין · לא גדל\"), כך שהסדק לא 'זוחל' כשהוא לא גדל.\n"
                                  "סדק לא מתקצר: תוצאה קצרה יותר נשארת על הקצה הקודם (אלא אם זה קורה 5 פעמים ברצף).")
        g.layout().addRow(self.chk_guard)
        return w

    def _goto(self, page):
        if page >= self.stack.count():
            return
        super()._goto(page)
        if page == PAGE_LOADS:
            self._recompute_loads()
        if page == PAGE_SETUP and not self.pick:
            QTimer.singleShot(0, self._setup_autozoom)
        if page == PAGE_SPEC and self.spec["img"] is not None and not self.pick:
            QTimer.singleShot(0, self.sp.view.fit_all)

    def _setup_autozoom(self):
        s = self.sides[self.setup_side]
        if s["base"] is None:
            return
        p = s["prep"]
        if p:
            k = 1.0 / p["mm_per_px"]
            pts = np.array([p["ref"], p["base_tip_xy"]], float)
            lo, hi = pts.min(0) - 6 * k, pts.max(0) + 6 * k
            self.su.view.zoom_to(lo[0], lo[1], hi[0], hi[1])
        else:
            self.su.view.fit_all()

    def _update_title(self):
        n = self.spec["name"].strip() if hasattr(self, "spec") else ""
        self.setWindowTitle(f"{APP_NAME} {VERSION}" + (f"  –  {n}" if n else ""))

    def _update_ready(self):
        ready = all(self.sides[i]["prep"] for i in self._active())
        self.btn_go.setEnabled(ready)
        n = [p for p, _ in NAV_ORDER].index(PAGE_SETUP) + 1
        self.nav_btns[PAGE_SETUP].setText(f"{n}   הגדרת מדידה{'  ✓' if ready else ''}")
        for i in (0, 1):
            s = self.sides[i]
            self.panels[i].set_done(s["base"] is not None, bool(np.isfinite(s["mm_per_px"])), bool(s["prep"]))
        return ready

    @staticmethod
    def _new_side():
        s = base.MainWindow._new_side()
        s["ct"] = dict(tip=None, mouth=None, start_manual=None, back=None, side1=None, side2=None, slot_u=None,
                       mcal=False, rot=None)
        return s

    def _build_setup_page(self):
        orig = base.SidePanel
        base.SidePanel = CTSidePanel
        try:
            w = super()._build_setup_page()
        finally:
            base.SidePanel = orig
        self.set_setup_hint("צד A: בחר את תיקיית המצלמה של המכונה. בתיקייה צריכה להיות תמונה אחת של הדגם במכונה "
                            "לפני תחילת הסדק – התוכנה מסובבת אותה (החריץ מימין) ומזהה את הדגם לבד.")
        return w

    # ------------------------------------------------------------------ screen 1: specimen
    def _build_spec_page(self):
        w = self._page(); lay = QHBoxLayout(w); lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(12)
        right = QFrame(); right.setObjectName("Card"); right.setFixedWidth(760)
        rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        sa = QScrollArea(); sa.setWidgetResizable(True); sa.setFrameShape(QFrame.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget(); il = QVBoxLayout(inner); il.setContentsMargins(12, 10, 12, 12); il.setSpacing(6)
        sa.setWidget(inner); rl.addWidget(sa)
        g0 = QGroupBox("הדגם")
        f0 = QFormLayout(g0)
        self.name_edit = QLineEdit(); self.name_edit.setPlaceholderText("למשל CT-1")
        self.name_edit.textChanged.connect(self._name_changed)
        f0.addRow("שם / מספר דגם:", self.name_edit)
        row = QHBoxLayout()
        row.addWidget(_btn("שמור פרויקט", slot=self.save_setup, icon="SP_DialogSaveButton"))
        row.addWidget(_btn("פתח פרויקט", slot=self.load_setup, icon="SP_DialogOpenButton"))
        f0.addRow(row)
        il.addWidget(g0)
        g1 = QGroupBox("1 · תמונת הדגם על השולחן")
        l1 = QVBoxLayout(g1)
        l1.addWidget(_btn("בחר תמונה של הדגם...", slot=self._pick_spec_image, icon="SP_DirOpenIcon"))
        self.spec_img_lbl = QLabel("לא נבחרה תמונה"); self.spec_img_lbl.setObjectName("Muted"); self.spec_img_lbl.setWordWrap(True)
        l1.addWidget(self.spec_img_lbl)
        il.addWidget(g1)
        g2 = QGroupBox("2 · מידות שנמדדו בקליבר")
        f2 = QFormLayout(g2)
        self.in_H = QDoubleSpinBox(); self.in_B = QDoubleSpinBox()
        for sp in (self.in_H, self.in_B):
            sp.setRange(0.0, 200.0); sp.setDecimals(3); sp.setSingleStep(0.01); sp.setSuffix("  מ״מ")
            sp.setSpecialValueText("—"); sp.setValue(0.0)
            sp.valueChanged.connect(self._inputs_changed)
        f2.addRow("גובה כולל H (בין שני הצדדים):", self.in_H)
        f2.addRow("עובי B:", self.in_B)
        n = QLabel("H קובע את קנה המידה של התמונה – מכאן כל שאר המידות. B לא נראה בתמונה (לחישוב העומסים ול-0.25B).")
        n.setObjectName("Muted"); n.setWordWrap(True); f2.addRow(n)
        il.addWidget(g2)
        g3 = QGroupBox("3 · זיהוי (ותיקון ידני אם צריך)")
        l3 = QVBoxLayout(g3)
        self.btn_auto = _btn("זהה אוטומטית: חורים, קצה חריץ וקצוות", "primary", slot=self.auto_detect_spec,
                             icon="SP_BrowserReload")
        l3.addWidget(self.btn_auto)
        grid = [("sp_hole1", "חור 1"), ("sp_hole2", "חור 2"), ("sp_tip", "קצה החריץ"),
                ("sp_back", "קצה אחורי"), ("sp_front", "קצה קדמי"), ("sp_side1", "צד 1"), ("sp_side2", "צד 2")]
        self.sp_btns = {}
        for rr in (grid[0:3], grid[3:5], grid[5:7]):
            row = QHBoxLayout()
            for key, txt in rr:
                b = _btn(txt, slot=lambda _=False, key=key: self.spec_pick(key)); self.sp_btns[key] = b; row.addWidget(b)
            l3.addLayout(row)
        hint = QLabel("חור – 3+ נקודות על השפה ואז Enter · קצה – 2 לחיצות על הקו · קצה חריץ – לחיצה אחת (בזום).")
        hint.setObjectName("Muted"); hint.setWordWrap(True); l3.addWidget(hint)
        il.addWidget(g3)
        g4 = QGroupBox("4 · Specimen dimensions — illustrative nominal values only")
        l4 = QVBoxLayout(g4)
        self.scale_lbl = QLabel("Scale: —"); self.scale_lbl.setStyleSheet("font-weight:600"); self.scale_lbl.setWordWrap(True)
        self.scale_lbl.setLayoutDirection(Qt.LeftToRight); self.scale_lbl.setAlignment(Qt.AlignLeft)
        l4.addWidget(self.scale_lbl)
        self.dim_tbl = QTableWidget(len(SPEC_ROWS), 6)
        self.dim_tbl.setLayoutDirection(Qt.LeftToRight)
        self.dim_tbl.setHorizontalHeaderLabels(["Dimension [mm]", "Measured", "Nominal", "Tolerance", "Deviation",
                                                "Status"])
        self.dim_tbl.verticalHeader().setVisible(False)
        self.dim_tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.dim_tbl.setSelectionMode(QAbstractItemView.NoSelection)
        hh = self.dim_tbl.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch); hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.dim_tbl.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.dim_tbl.setFixedHeight(len(SPEC_ROWS) * 30 + 32)
        l4.addWidget(self.dim_tbl)
        self.dim_note = QLabel(""); self.dim_note.setObjectName("Muted"); self.dim_note.setWordWrap(True)
        self.dim_note.setLayoutDirection(Qt.LeftToRight); self.dim_note.setAlignment(Qt.AlignLeft)
        l4.addWidget(self.dim_note)
        row = QHBoxLayout()
        row.addWidget(_btn("שמור תמונת מידות...", slot=self._export_dim_image, icon="SP_DialogSaveButton"))
        row.addWidget(_btn("העתק טבלה", slot=self._copy_dims))
        row.addStretch(1)
        row.addWidget(_btn("המשך להגדרת מדידה  ←", "primary", slot=lambda: self._goto(PAGE_SETUP)))
        l4.addLayout(row)
        il.addWidget(g4)
        il.addStretch(1)
        lay.addWidget(right)
        left = QVBoxLayout(); left.setSpacing(8)
        self.spec_hint = QLabel("צלם את הדגם מונח על השולחן (פני הדגם ישר מול המצלמה), בחר את התמונה והזן את הגובה "
                                "הכולל H ואת העובי B. התוכנה מזהה את החורים, קצה החריץ והקצוות ומחשבת את כל המידות.")
        self.spec_hint.setObjectName("Banner"); self.spec_hint.setWordWrap(True)
        left.addWidget(self.spec_hint)
        self.sp = SetupView(); self.sp.title.setText("הדגם")
        self.sp.view.clear_image("בחר תמונה של הדגם")
        self.sp.zoom.caption = "הגדלה · גלגלת = זום"
        self.sp.view.clicked.connect(self._on_spec_click)
        self.sp.zoom.clicked.connect(self._on_spec_click)
        left.addWidget(self.sp, 1)
        lay.addLayout(left, 1)
        return w

    def _name_changed(self, t):
        self.spec["name"] = t.strip()
        self._update_title()

    def _inputs_changed(self, *_):
        self.spec["H"] = self.in_H.value() if self.in_H.value() > 0 else np.nan
        self.spec["B"] = self.in_B.value() if self.in_B.value() > 0 else np.nan
        self._recompute_spec()
        for i in (0, 1):
            if self.sides[i]["base"] is not None and not self._busy():
                self._ct_update(i)

    def _pick_spec_image(self, path=None):
        if not path:
            start = os.path.dirname(self.spec["image"]) if self.spec["image"] else \
                self.settings.value("last_folder", os.path.expanduser("~"))
            path, _ = QFileDialog.getOpenFileName(self, "תמונת הדגם", start,
                                                  "Images (*.bmp *.jpg *.jpeg *.png *.tif *.tiff)")
            if not path:
                return
        try:
            img = cc.read_gray(path)
        except Exception as ex:
            QMessageBox.warning(self, APP_NAME, f"לא ניתן לקרוא את התמונה:\n{ex}"); return
        new = path != self.spec.get("image")
        self.spec.update(image=path, img=img)
        if new:
            self.spec["feats"] = {}              # marks of a previous photo do not belong to this one
        self.spec_img_lbl.setText(_ltr(os.path.basename(path)))
        self.sp.view.set_image(img); self.sp.view.fit_all()
        self._draw_spec()
        if not self.spec["feats"]:
            self.auto_detect_spec()

    def auto_detect_spec(self):
        if self.spec["img"] is None:
            self.toast("בחר קודם תמונה של הדגם.", "warn"); return
        if getattr(self, "_det", None) is not None and self._det.isRunning():
            return
        self._cancel_pick()
        self.btn_auto.setEnabled(False)
        self.spec_hint.setText("מזהה את הדגם... (כמה שניות)")
        self._det = DetectWorker(self.spec["img"], self.spec.get("image"))
        self._det.done.connect(self._spec_detected)
        self._det.start()

    def _spec_detected(self, feats, notes):
        self.btn_auto.setEnabled(True)
        if feats:
            self.spec["feats"].update(feats)
        found = [n for k, n in (("hole1", "חור 1"), ("hole2", "חור 2"), ("tip", "קצה חריץ"), ("back", "קצה אחורי"),
                                ("front", "קצה קדמי"), ("side1", "צד 1"), ("side2", "צד 2")) if k in feats]
        msg = ("זוהו: " + ", ".join(found) + ". בדוק בעין שהסימונים יושבים על הדגם (גלגלת = זום); "
               "לתיקון – כפתור התכונה וסימון ידני.") if found else ""
        if not np.isfinite(self.spec["H"]):
            msg += "  הזן את הגובה H כדי לקבל מידות במ״מ."
        self.spec_hint.setProperty("kind", "action" if notes else ""); _repolish(self.spec_hint)
        self.spec_hint.setText((msg + ("  ·  " if msg and notes else "") + "  ".join(notes)) or "לא זוהה דבר.")
        self._recompute_spec()
        if found:
            self.toast(f"זוהו {len(found)} מתוך 7 תכונות", "ok" if not notes else "warn")

    def spec_dim(self, k):
        """Dimension in use [mm]: H and B as entered, everything else from the photo (scaled by H);
        nominal when not available."""
        if k in SPEC_INPUT and np.isfinite(self.spec.get(k, np.nan)):
            return float(self.spec[k])
        if k == "D":
            d = [self.spec["dims"].get(x, np.nan) for x in ("D1", "D2")]
            d = [x for x in d if np.isfinite(x)]
            return float(np.mean(d)) if d else 7.5
        ph = self.spec["dims"].get(k, np.nan)
        if np.isfinite(ph):
            return float(ph)
        return SPEC_NOM.get(k, ct.NOMINAL.get(k, np.nan))

    def spec_pick(self, key):
        if self.spec["img"] is None:
            self.toast("בחר קודם תמונה של הדגם.", "warn"); return
        self._cancel_pick()
        self.pick = dict(mode=key, side=-1, pts=[])
        self.sp.pair.set_pick(True)
        msg = {"sp_hole1": "חור 1: לחץ 3 נקודות או יותר על שפת החור (על פני הדגם), ואז Enter",
               "sp_hole2": "חור 2: לחץ 3 נקודות או יותר על שפת החור, ואז Enter",
               "sp_tip": "לחץ על קצה החריץ (מומלץ בחלון ההגדלה)",
               "sp_back": "קצה אחורי (מול החריץ): 2 לחיצות על הקו, רחוקות זו מזו",
               "sp_front": "קצה קדמי (הצד של החריץ): 2 לחיצות על הקו, לא בתוך החריץ",
               "sp_side1": "צד 1: 2 לחיצות על הקו", "sp_side2": "צד 2 (הנגדי): 2 לחיצות על הקו"}[key]
        self.spec_hint.setText(msg + "  ·  Esc = ביטול")
        self.spec_hint.setProperty("kind", "action"); _repolish(self.spec_hint)

    def _on_spec_click(self, x, y):
        if not self.pick or not self.pick["mode"].startswith("sp_"):
            return
        mode = self.pick["mode"]
        self.pick["pts"].append((x, y))
        self.sp.view.add_marker(x, y, "+", "#22d3ee", 7, 2)
        pts = self.pick["pts"]
        if mode in ("sp_hole1", "sp_hole2"):
            if len(pts) >= 3:
                self.spec_hint.setText(f"{len(pts)} נקודות – Enter לסיום, או עוד נקודות לדיוק")
            return
        if mode == "sp_tip":
            self._cancel_pick()
            self.spec["feats"]["tip"] = (float(x), float(y))
            self._spec_done("קצה החריץ סומן.")
            return
        if len(pts) < 2:
            return
        self._cancel_pick()
        q1, q2 = ct.refine_line(self.spec["img"], pts[0], pts[1])
        self.spec["feats"][mode[3:]] = (tuple(map(float, q1)), tuple(map(float, q2)))
        self._spec_done("הקו סומן ויושר לשפה.")

    def _finish_pick(self):
        if not self.pick:
            return
        mode = self.pick["mode"]
        if mode in ("sp_hole1", "sp_hole2") and len(self.pick["pts"]) >= 3:
            pts = self.pick["pts"]
            self._cancel_pick()
            cx, cy, r = ct.fit_circle(pts)
            cx, cy, r = ct.refine_circle(self.spec["img"], cx, cy, r, search=max(4.0, 0.03 * r))
            self.spec["feats"][mode[3:]] = (cx, cy, r)
            self._spec_done("החור סומן (מעגל מותאם לשפה).")

    def _spec_done(self, msg):
        self.spec_hint.setProperty("kind", ""); _repolish(self.spec_hint)
        self.spec_hint.setText(msg)
        self._recompute_spec()

    def _cancel_pick(self):
        super()._cancel_pick()
        if hasattr(self, "sp"):
            self.sp.pair.set_pick(False)

    def _recompute_spec(self):
        if not hasattr(self, "dim_tbl"):
            return
        sp = self.spec
        f = sp["feats"]
        gpx = ct.geometry_px(f)
        sp["gpx"] = gpx
        H = sp["H"]
        if "H" in gpx and np.isfinite(H):
            sp["pxmm"] = gpx["H"] / H
        else:
            sp["pxmm"] = np.nan
        k = sp["pxmm"]
        dims = {}
        if np.isfinite(k):
            for key in ("W", "W125", "D1", "D2", "S", "an", "off"):
                if key in gpx:
                    dims[key] = gpx[key] / k
            if "W" in dims and "W125" in dims:
                dims["F"] = dims["W125"] - dims["W"]
            if "tip" in f and "load_u" in gpx and "hole1" in f and "hole2" in f:
                t0 = np.asarray(f["tip"], float); lu = gpx["load_u"]
                for key, h in (("h1", "hole1"), ("h2", "hole2")):
                    dims[key] = abs(float(np.dot(np.asarray(f[h][:2], float) - t0, lu))) / k
        sp["dims"] = dims
        if np.isfinite(k):
            self.scale_lbl.setText(f"Scale: {k:.3f} px/mm  ({1000.0 / k:.1f} µm/px)  ·  from H = {H:.3f} mm")
        elif "H" not in gpx:
            self.scale_lbl.setText("Scale: — (the two side edges are needed)")
        else:
            self.scale_lbl.setText("Scale: — (enter the measured height H)")
        res = (1.0 / k) if np.isfinite(k) else np.nan
        n_out = 0
        RED_BG, RED_FG, OK_FG = QColor("#fee2e2"), QColor("#991b1b"), QColor("#15803d")
        for r, (key, name, nom, (lo, hi)) in enumerate(SPEC_ROWS):
            if key in SPEC_INPUT:
                val = sp[key]; src = ""
            else:
                val = dims.get(key, np.nan); src = ""
            have = np.isfinite(val)
            ok = have and nom + lo - 1e-9 <= val <= nom + hi + 1e-9
            note = ""
            if have and not ok and key not in SPEC_INPUT and np.isfinite(res):
                over = (nom + lo - val) if val < nom + lo else (val - nom - hi)
                if over <= 1.5 * res:
                    note = " *"
            status = "" if not have else ("OK" if ok else "OUT OF TOL" + note)
            tol = f"{lo:+.2f} / {hi:+.2f}" if lo != -hi else f"±{hi:.2f}"
            vals = [name, (f"{val:.3f}{src}" if have else "—"), f"{nom:.2f}", tol,
                    (f"{val - nom:+.3f}" if have else ""), status]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment((Qt.AlignLeft | Qt.AlignVCenter) if c == 0 else Qt.AlignCenter)
                if have and not ok:
                    it.setBackground(QBrush(RED_BG)); it.setForeground(QBrush(RED_FG))
                    if c == 5:
                        fb = QFont(); fb.setBold(True); it.setFont(fb)
                elif have and c == 5:
                    it.setForeground(QBrush(OK_FG))
                self.dim_tbl.setItem(r, c, it)
            n_out += int(have and not ok)
        notes = []
        if np.isfinite(res):
            notes.append(f"Photo resolution: 1 px = {res:.3f} mm.")
        notes.append("* = outside the tolerance by less than 1.5 px of the photo – check it with a caliper / pin gauge.")
        if n_out:
            notes.insert(0, f"{n_out} dimension(s) out of tolerance (red).")
        self.dim_note.setText("  ".join(notes))
        for key, b in self.sp_btns.items():
            done = key[3:] in f
            b.setText(("✓ " if done else "") + b.text().replace("✓ ", ""))
        self._draw_spec()
        if hasattr(self, "loads_tbl"):
            self._recompute_loads()

    def _dims_rows(self):
        out = [["Dimension [mm]", "Measured", "Nominal", "Tolerance", "Deviation", "Status"]]
        for r in range(self.dim_tbl.rowCount()):
            out.append([(self.dim_tbl.item(r, c).text() if self.dim_tbl.item(r, c) else "") for c in range(6)])
        return out

    def _copy_dims(self):
        QApplication.clipboard().setText("\n".join("\t".join(r) for r in self._dims_rows()))
        self.toast("הטבלה הועתקה – אפשר להדביק באקסל.", "ok")

    def _draw_spec(self):
        v = self.sp.view
        if self.spec["img"] is None:
            return
        v.clear_overlays()
        f, g = self.spec["feats"], self.spec["gpx"]
        pxmm = self.spec["pxmm"]
        mm = (lambda px: f"{px / pxmm:.3f} mm") if np.isfinite(pxmm) else (lambda px: f"{px:.0f} px")
        for key in ("hole1", "hole2"):
            if key in f:
                cx, cy, r = f[key]
                path = QPainterPath(); path.addEllipse(cx + 0.5 - r, cy + 0.5 - r, 2 * r, 2 * r)
                it = QGraphicsPathItem(path); pen = QPen(QColor("#22d3ee"), 2); pen.setCosmetic(True); it.setPen(pen)
                v._add(it, 8)
                v.add_marker(cx, cy, "+", "#22d3ee", 8, 1.5)
        for key in ("back", "front", "side1", "side2"):
            if key in f:
                v.add_polyline(list(f[key]), C_EDGE, 2.0, z=8)
        if "tip" in f:
            v.add_marker(*f["tip"], "circle", "#ff2d2d", 9, 2)
        if "load_p" in g:
            M, lu = g["load_p"], g["load_u"]
            half = (g.get("S", 200) * 0.9)
            v.add_polyline([M - lu * half, M + lu * half], C_LOAD, 1.4, Qt.DashLine, z=7)
            o, ax = g["start"], g["axis_u"]
            far = g.get("W", g.get("an", 100) * 1.5)
            v.add_polyline([o, o + ax * far], "#7dd3fc", 1.2, Qt.DashLine, z=7)
            v.add_marker(*o, "+", C_LOAD, 9, 2)
            n = np.array([-ax[1], ax[0]])
            off = 0.22 * g.get("S", 200)
            if "an" in g:
                p1, p2 = o + n * off * 0.5, o + n * off * 0.5 + ax * g["an"]
                v.add_polyline([p1, p2], "#fb923c", 2.0, z=9)
                m = (p1 + p2) / 2
                _label(v, m[0], m[1], f"a_n = {mm(g['an'])}", "#fed7aa")
            if "W" in g:
                p1, p2 = o + n * off, o + n * off + ax * g["W"]
                v.add_polyline([p1, p2], "#fbbf24", 2.0, z=9)
                m = p1 + 0.62 * (p2 - p1)
                _label(v, m[0], m[1], f"W = {mm(g['W'])}", "#fde68a")
            if "W125" in g:
                p1, p2 = g["front_pt"] - n * off * 1.6, g["back_pt"] - n * off * 1.6
                v.add_polyline([p1, p2], "#a3e635", 2.0, z=9)
                m = p1 + 0.45 * (p2 - p1)
                _label(v, m[0], m[1], f"Width = {mm(g['W125'])}", "#d9f99d")
            if "S" in g:
                c1, c2 = np.array(f["hole1"][:2]), np.array(f["hole2"][:2])
                m = (c1 + c2) / 2 - ax * g["D"] * 0.35
                _label(v, m[0], m[1], f"S = {mm(g['S'])}", "#a5f3fc")
                c = c1 + ax * f["hole1"][2] * 0.1
                _label(v, c[0] - f["hole1"][2] * 0.6, c[1], f"Ø {mm(2 * f['hole1'][2])}", "#a5f3fc")
                c = c2 + ax * f["hole2"][2] * 0.1
                _label(v, c[0] - f["hole2"][2] * 0.6, c[1], f"Ø {mm(2 * f['hole2'][2])}", "#a5f3fc")
        if "H" in g:
            a, b = g["side_pts"]
            v.add_polyline([a, b], "#c4b5fd", 2.0, z=9)
            m = a + 0.22 * (b - a)
            _label(v, m[0], m[1], f"H = {mm(g['H'])}", "#ddd6fe")
        if np.isfinite(pxmm) and "load_p" in g:
            Wm = g.get("W", 30 * pxmm) / pxmm; Fm = (g["W125"] - g["W"]) / pxmm if "W125" in g and "W" in g else 7.5
            Hm = g.get("H", 36 * pxmm) / pxmm; off = g.get("off", 0.0) / pxmm
            vv = np.array([-g["axis_u"][1], g["axis_u"][0]]); sg = np.sign(np.dot(vv, g["load_u"])) or 1.0
            draw_grid(v, g["start"], g["axis_u"], 1.0 / pxmm, -Fm, Wm, -Hm / 2 - sg * off, Hm / 2 - sg * off, labels=False)
        if "tip" in f:
            self.sp.pair.show_zoom(self.spec["img"], f["tip"], 1.0 / pxmm if np.isfinite(pxmm) else None,
                                   lambda z: z.add_marker(*f["tip"], "ticks", "#ff2d2d", 10, 2))

    def _export_dim_image(self):
        if self.spec["img"] is None:
            self.toast("אין תמונת דגם.", "warn"); return
        from PIL import ImageDraw, ImageFont
        name = self.spec["name"] or "specimen"
        start = os.path.join(os.path.dirname(self.spec["image"]), f"{name}_dimensions.png")
        out, _ = QFileDialog.getSaveFileName(self, "שמירת תמונת מידות", start, "PNG (*.png)")
        if not out:
            return
        im = Image.fromarray((np.clip(self.spec["img"], 0, 1) * 255).astype(np.uint8)).convert("RGB")
        dr = ImageDraw.Draw(im)
        H = im.size[1]; lw = max(2, H // 500)
        try:
            font = ImageFont.truetype("segoeui.ttf", max(14, H // 45))
        except Exception:
            font = ImageFont.load_default()
        f, g, pxmm = self.spec["feats"], self.spec["gpx"], self.spec["pxmm"]
        mm = (lambda px: f"{px / pxmm:.3f} mm") if np.isfinite(pxmm) else (lambda px: f"{px:.0f} px")

        def txt(p, s, col):
            x, y = float(p[0]), float(p[1])
            bb = dr.textbbox((x, y), s, font=font)
            dr.rectangle([bb[0] - 4, bb[1] - 2, bb[2] + 4, bb[3] + 2], fill=(15, 23, 42))
            dr.text((x, y), s, fill=col, font=font)
        for key in ("hole1", "hole2"):
            if key in f:
                cx, cy, r = f[key]
                dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(34, 211, 238), width=lw)
        for key in ("back", "front", "side1", "side2"):
            if key in f:
                dr.line([tuple(f[key][0]), tuple(f[key][1])], fill=(74, 222, 128), width=lw)
        if "load_p" in g:
            o, ax = g["start"], g["axis_u"]; n = np.array([-ax[1], ax[0]]); off = 0.22 * g.get("S", 200)
            M, lu = g["load_p"], g["load_u"]; half = g.get("S", 200) * 0.9
            dr.line([tuple(M - lu * half), tuple(M + lu * half)], fill=(250, 204, 21), width=lw)
            for key, sc, col in (("an", 0.5, (251, 146, 60)), ("W", 1.0, (251, 191, 36))):
                if key in g:
                    p1, p2 = o + n * off * sc, o + n * off * sc + ax * g[key]
                    dr.line([tuple(p1), tuple(p2)], fill=col, width=lw)
                    txt((p1 + p2) / 2, f"{'a_n' if key == 'an' else 'W'} = {mm(g[key])}", col)
            if "W125" in g:
                p1, p2 = g["front_pt"] - n * off * 1.6, g["back_pt"] - n * off * 1.6
                dr.line([tuple(p1), tuple(p2)], fill=(163, 230, 53), width=lw)
                txt((p1 + p2) / 2, f"Width = {mm(g['W125'])}", (163, 230, 53))
            if "S" in g:
                c1, c2 = np.array(f["hole1"][:2]), np.array(f["hole2"][:2])
                txt((c1 + c2) / 2 - ax * g["D"] * 0.35, f"S = {mm(g['S'])}", (165, 243, 252))
                txt(c1, f"D = {mm(2 * f['hole1'][2])}", (165, 243, 252))
                txt(c2, f"D = {mm(2 * f['hole2'][2])}", (165, 243, 252))
        if "H" in g:
            a, b = g["side_pts"]
            dr.line([tuple(a), tuple(b)], fill=(196, 181, 253), width=lw)
            txt((a + b) / 2, f"H = {mm(g['H'])}", (196, 181, 253))
        if "tip" in f:
            x, y = f["tip"]; r = 4 * lw
            dr.ellipse([x - r, y - r, x + r, y + r], outline=(255, 45, 45), width=lw)
        B = self.spec_dim("B")
        txt((20, 20), f"{name}   B = {B:.3f} mm   scale {pxmm:.3f} px/mm", (255, 255, 255))
        im.save(out)
        self.toast(f"נשמר: {os.path.basename(out)}", "ok")

    # ------------------------------------------------------------------ loads page
    def _build_loads_page(self):
        w = self._page(); lay = QVBoxLayout(w); lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(10)
        head = QLabel("<span style='font-size:15pt; font-weight:700; color:#0f172a'>כיול עומסים</span><br>"
                      "<span style='color:#64748b'>K היעד מחושב מהעומס הנומינלי והמידות הנומינליות; העומסים לדגם "
                      "מחושבים מחדש לפי B ו-W שנמדדו, כך ש-K נשאר זהה (ASTM E647, C(T)). Pmin = R·Pmax.</span>")
        head.setWordWrap(True); lay.addWidget(head)
        top = QHBoxLayout(); top.setSpacing(12)
        g1 = QGroupBox("מידות")
        f1 = QFormLayout(g1)
        self.ld_W0 = QDoubleSpinBox(); self.ld_B0 = QDoubleSpinBox(); self.ld_lbf = QDoubleSpinBox(); self.ld_tol = QDoubleSpinBox()
        for sp, val, dec, suf in ((self.ld_W0, ct.NOMINAL["W"], 3, " מ״מ"), (self.ld_B0, ct.NOMINAL["B"], 3, " מ״מ"),
                                  (self.ld_lbf, ct.LBF_TO_N, 6, " N/lbf"), (self.ld_tol, 2.0, 1, " %")):
            sp.setRange(0.0, 1000.0); sp.setDecimals(dec); sp.setValue(val); sp.setSuffix(suf)
            sp.valueChanged.connect(self._recompute_loads)
        f1.addRow("W נומינלי:", self.ld_W0); f1.addRow("B נומינלי:", self.ld_B0)
        f1.addRow("המרת lbf ל-N:", self.ld_lbf); f1.addRow("טולרנס עומס:", self.ld_tol)
        self.ld_used = QLabel(""); self.ld_used.setWordWrap(True); self.ld_used.setStyleSheet("font-weight:600")
        f1.addRow(self.ld_used)
        top.addWidget(g1, 2)
        g2 = QGroupBox("שלבים (עומסים נומינליים)")
        l2 = QVBoxLayout(g2)
        self.st_tbl = QTableWidget(0, 4)
        self.st_tbl.setHorizontalHeaderLabels(["שלב", "a [מ״מ]", "Pmax נומינלי [lbf]", "Pmin נומינלי [lbf]"])
        self.st_tbl.verticalHeader().setVisible(False)
        self.st_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.st_tbl.itemChanged.connect(self._stages_changed)
        l2.addWidget(self.st_tbl)
        row = QHBoxLayout()
        row.addWidget(_btn("הוסף שלב", slot=self._add_stage)); row.addWidget(_btn("מחק שלב", slot=self._del_stage))
        row.addStretch(1); l2.addLayout(row)
        top.addWidget(g2, 3)
        lay.addLayout(top)
        card = QFrame(); card.setObjectName("Card"); cl = QVBoxLayout(card); cl.setContentsMargins(8, 8, 8, 8)
        t = QLabel("עומסים לדגם"); t.setObjectName("PanelTitle"); cl.addWidget(t)
        self.loads_tbl = QTableWidget(0, 11)
        self.loads_tbl.setHorizontalHeaderLabels(["שלב", "a [מ״מ]", "Kmax יעד [MPa√m]", "ΔK [MPa√m]", "α = a/W",
                                                  "f(α)", "Pmax [N]", "Pmin [N]", "גבול עליון [N]",
                                                  "גבול תחתון [N]", "Pmax נומינלי [N]"])
        self.loads_tbl.verticalHeader().setVisible(False)
        self.loads_tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.loads_tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        cl.addWidget(self.loads_tbl)
        row = QHBoxLayout()
        row.addWidget(_btn("ייצא עומסים (CSV)", "primary", slot=self._export_loads, icon="SP_DialogSaveButton"))
        row.addWidget(_btn("העתק טבלה", slot=self._copy_loads))
        row.addStretch(1)
        cl.addLayout(row)
        lay.addWidget(card, 1)
        self._fill_stages()
        return w

    def _fill_stages(self):
        self._filling_st = True
        self.st_tbl.setRowCount(len(self.loads["stages"]))
        for r, st in enumerate(self.loads["stages"]):
            for c, v in enumerate(st):
                it = QTableWidgetItem(str(v) if c == 0 else f"{float(v):g}"); it.setTextAlignment(Qt.AlignCenter)
                self.st_tbl.setItem(r, c, it)
        self._filling_st = False

    def _stages_changed(self, *_):
        if getattr(self, "_filling_st", False):
            return
        st = []
        for r in range(self.st_tbl.rowCount()):
            try:
                name = self.st_tbl.item(r, 0).text()
                vals = [float(self.st_tbl.item(r, c).text().replace(",", ".")) for c in (1, 2, 3)]
                st.append([name] + vals)
            except Exception:
                continue
        self.loads["stages"] = st
        self._recompute_loads()

    def _add_stage(self):
        self.loads["stages"].append([f"Stage {len(self.loads['stages']) + 1}", self.spec_dim("an"), 0.0, 0.0])
        self._fill_stages(); self._recompute_loads()

    def _del_stage(self):
        r = self.st_tbl.currentRow()
        if 0 <= r < len(self.loads["stages"]):
            self.loads["stages"].pop(r); self._fill_stages(); self._recompute_loads()

    def _load_rows(self):
        W, B = self.spec_dim("W"), self.spec_dim("B")
        return ct.load_table([tuple(s) for s in self.loads["stages"]], self.ld_W0.value(), self.ld_B0.value(),
                             W, B, self.ld_lbf.value(), self.ld_tol.value() / 100.0), W, B

    def _recompute_loads(self, *_):
        if not hasattr(self, "loads_tbl") or not hasattr(self, "spec"):
            return
        rows, W, B = self._load_rows()
        srcW = "מהתמונה" if np.isfinite(self.spec["dims"].get("W", np.nan)) else "נומינלי"
        srcB = "קליבר" if np.isfinite(self.spec["B"]) else "נומינלי"
        a0 = self.spec_dim("an")
        txt = f"לדגם {self.spec['name'] or '—'}:  W = {W:.4f} מ״מ ({srcW}),  B = {B:.4f} מ״מ ({srcB}),  a0 = {a0:.3f} מ״מ"
        short = [st[0] for st in self.loads["stages"] if float(st[1]) < a0]
        if short:
            txt += (f"<br><span style='color:#b45309'>שים לב: בשלבים {', '.join(short)} אורך הסדק a קטן מ-a0 של הדגם "
                    f"({a0:.2f} מ״מ) – שלב יכול להתחיל רק מאורך הסדק בפועל (לפחות a0).</span>")
        self.ld_used.setText(txt)
        self.loads_tbl.setRowCount(len(rows))
        for r, d in enumerate(rows):
            vals = [d["name"], f"{d['a']:.3f}", f"{d['K0']:.4f}", f"{d['dK']:.4f}", f"{d['alpha']:.5f}",
                    f"{d['f']:.5f}", f"{d['pmax']:.2f}", f"{d['pmin']:.2f}", f"{d['upper']:.2f}",
                    f"{d['lower']:.2f}", f"{d['pmax0']:.2f}"]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(_ltr(v) if c else v); it.setTextAlignment(Qt.AlignCenter)
                if c in (6, 7):
                    f = QFont(); f.setBold(True); it.setFont(f)
                self.loads_tbl.setItem(r, c, it)

    def _loads_csv_rows(self):
        rows, W, B = self._load_rows()
        out = [["specimen", self.spec["name"]], ["W_mm", f"{W:.4f}"], ["B_mm", f"{B:.4f}"],
               ["W0_mm", f"{self.ld_W0.value():.3f}"], ["B0_mm", f"{self.ld_B0.value():.3f}"], [],
               ["stage", "a_mm", "Kmax_target_MPa_sqrt_m", "dK_MPa_sqrt_m", "alpha", "f_alpha", "Pmax_N", "Pmin_N",
                "upper_N", "lower_N", "Pmax_nominal_N", "Pmin_nominal_N", "Pmax_nominal_lbf", "Pmin_nominal_lbf"]]
        for d in rows:
            out.append([d["name"], f"{d['a']:.3f}", f"{d['K0']:.5f}", f"{d['dK']:.5f}", f"{d['alpha']:.6f}",
                        f"{d['f']:.6f}", f"{d['pmax']:.3f}", f"{d['pmin']:.3f}", f"{d['upper']:.3f}",
                        f"{d['lower']:.3f}", f"{d['pmax0']:.3f}", f"{d['pmin0']:.3f}", f"{d['pmax0_lb']:g}",
                        f"{d['pmin0_lb']:g}"])
        return out

    def _export_loads(self):
        name = self.spec["name"] or "specimen"
        start = os.path.join(os.path.dirname(self.spec["image"]) if self.spec["image"] else os.path.expanduser("~"),
                             f"{name}_loads.csv")
        f, _ = QFileDialog.getSaveFileName(self, "שמירת עומסים", start, "CSV (*.csv)")
        if not f:
            return
        try:
            with open(f, "w", newline="", encoding="utf-8-sig") as fh:
                csv.writer(fh).writerows(self._loads_csv_rows())
            self.toast("טבלת העומסים נשמרה.", "ok")
        except OSError as ex:
            QMessageBox.warning(self, APP_NAME, f"לא ניתן לשמור:\n{ex}")

    def _copy_loads(self):
        QApplication.clipboard().setText("\n".join("\t".join(r) for r in self._loads_csv_rows()))
        self.toast("הטבלה הועתקה – אפשר להדביק באקסל.", "ok")

    # ------------------------------------------------------------------ screen 2: measurement setup
    def _dims4(self):
        return (self.spec_dim("H"), self.spec_dim("W"), self.spec_dim("W125"), self.spec_dim("an"))

    def set_folder(self, i, folder, base_file=None, rot=None):
        if i == 0:
            ROT[_norm(folder)] = cf.ROT_MACHINE         # machine camera: always the same quarter turn
        else:
            ROT.pop(_norm(folder), None)                # side B (phone): images are used as they are
        keep = self.sides[i]["ct"] if (self.sides[i]["folder"] == folder) else None
        super().set_folder(i, folder, base_file)
        s = self.sides[i]
        if s["base"] is None:
            return
        if keep is None:
            s["ct"] = self._new_side()["ct"]
        s["ct"]["rot"] = ROT.get(_norm(folder))
        self._ct_update(i)
        if s["ct"]["tip"] is None:
            self.ct_auto(i)

    def ct_mcal(self, i, on):
        self.sides[i]["ct"]["mcal"] = bool(on)
        self.panels[i].w_mcal.setVisible(bool(on))
        self._ct_update(i)

    def ct_pick(self, i, mode):
        s = self.sides[i]
        if s["base"] is None:
            self.toast("בחר קודם תיקיית תמונות." if i == 0 else "צלם קודם תמונה מהטלפון.", "warn"); return
        if self._busy():
            self.toast("עצור קודם את המדידה.", "warn"); return
        if i == 1 and hasattr(self.panels[1], "btn_plive"):
            self.panels[1].btn_plive.setChecked(False)
        self._cancel_pick()
        self._goto(PAGE_SETUP)
        self.show_side(i, "base")
        msg = {"ct_tip": "לחץ על קצה החריץ – הנקודה העמוקה ביותר של החריץ (מומלץ בחלון ההגדלה)",
               "ct_mouth": "לחץ על נקודה באמצע החריץ, לפני הקצה שלו",
               "ct_start": "לחץ על נקודת ההתחלה (על קו העומס)",
               "ct_back": "קצה אחורי של הדגם (מול החריץ): 2 לחיצות על הקו, רחוקות זו מזו",
               "ct_side1": "צד 1 של הדגם: 2 לחיצות על הקו (איפה שהתפסנית לא מסתירה)",
               "ct_side2": "צד 2 של הדגם: 2 לחיצות על הקו"}[mode]
        self.set_setup_hint(msg + "  ·  Esc = ביטול", True)
        self.pick = dict(mode=mode, side=i, pts=[])
        self.su.pair.set_pick(True)

    def ct_auto(self, i):
        s = self.sides[i]
        if s["base"] is None:
            self.toast("בחר קודם תיקיית תמונות." if i == 0 else "צלם קודם תמונה מהטלפון.", "warn"); return
        if self._busy():
            self.toast("עצור קודם את המדידה.", "warn"); return
        if self._fw.get(i) is not None and self._fw[i].isRunning():
            return
        self._cancel_pick()
        self._goto(PAGE_SETUP)
        self.panels[i].b_eauto.setEnabled(False)
        any_orient = i == 1
        if any_orient:                                  # phone / second camera: try every orientation
            img = _read_raw(s["base_file"])
            self.set_setup_hint(f"צד {SIDE_NAMES[i]}: מזהה את קצוות הדגם וקצה החריץ (בכל כיוון של התמונה)...")
        else:
            img = s["base"]
            self.set_setup_hint(f"צד {SIDE_NAMES[i]}: מזהה את קצוות הדגם וקצה החריץ...")
        fw = self._fw[i] = FrameWorker(img, self._dims4(), any_orient)
        fw.done.connect(lambda r, i=i: self._ct_frame_detected(i, r))
        fw.start()

    def _ct_frame_detected(self, i, r):
        self.panels[i].b_eauto.setEnabled(True)
        s = self.sides[i]; c = s["ct"]
        notes = list(r.get("notes", []))
        if r.get("any"):
            if r.get("rot") is None:
                self.set_setup_hint(f"צד {SIDE_NAMES[i]}: הדגם לא זוהה באף כיוון. " + "  ".join(notes) +
                                    " – אפשר לסמן ידנית (קצה אחורי, צדדים, קצה חריץ) או כיול ידני.", True)
                return
            # found on a turned copy; the image itself stays as it is (marks mapped back to it)
            k, shape = r["rot"], s["base"].shape
            back = lambda p: to_raw(p, k, shape)
            for key in ("back", "side1", "side2"):
                if r.get(key):
                    r[key] = (back(r[key][0]), back(r[key][1]))
            if r.get("tip") is not None:
                if r.get("slot_u") is not None:
                    t = np.asarray(r["tip"], float)
                    a, b = np.asarray(back(t)), np.asarray(back(t + 10 * np.asarray(r["slot_u"], float)))
                    r["slot_u"] = tuple((b - a) / np.linalg.norm(b - a))
                r["tip"] = back(r["tip"])
            c["rot"] = 0
        for k in ("back", "side1", "side2"):
            c[k] = r.get(k)
        c["tip"] = r.get("tip"); c["slot_u"] = r.get("slot_u"); c["start_manual"] = None
        s["roi"] = None if not s["roi_user"] else s["roi"]
        self._ct_update(i)
        self.show_side(i, "base")
        self._setup_autozoom()
        ok = all(c.get(k) is not None for k in ("back", "side1", "side2", "tip"))
        msg = ("זוהו הקצה האחורי, שני הצדדים וקצה החריץ – בדוק בעין בחלון ההגדלה. " if ok else "") + "  ".join(notes)
        self.set_setup_hint(msg + ("  " + s.get("start_txt", "") if ok else ""), not ok)

    def ct_start_auto(self, i):
        self.sides[i]["ct"]["start_manual"] = None
        self._ct_update(i)

    def _on_setup_click(self, x, y):
        if not self.pick or not self.pick["mode"].startswith("ct_"):
            return super()._on_setup_click(x, y)
        i, mode = self.pick["side"], self.pick["mode"]
        c = self.sides[i]["ct"]
        self.pick["pts"].append((x, y))
        self.su.view.add_marker(x, y, "+", "#22d3ee", 7, 2)
        pts = self.pick["pts"]
        if mode in ("ct_back", "ct_side1", "ct_side2") and len(pts) < 2:
            return
        self._cancel_pick()
        base_img = self.sides[i]["base"]
        if mode in ("ct_back", "ct_side1", "ct_side2"):
            q1, q2 = ct.refine_line(base_img, pts[0], pts[1])
            c[mode[3:]] = (tuple(map(float, q1)), tuple(map(float, q2)))
        elif mode == "ct_tip":
            c["tip"] = (float(x), float(y))
        elif mode == "ct_mouth":
            c["mouth"] = (float(x), float(y))
        elif mode == "ct_start":
            c["start_manual"] = (float(x), float(y))
        self._ct_update(i)
        self.set_setup_hint(self.sides[i]["msg"] if self.sides[i]["prep"] else self.panels[i].lbl_start.text(),
                            not self.sides[i]["prep"])

    def _ct_geometry(self, i):
        """Scale (mm/px), start point on the load line, crack axis and the back-edge point on the axis."""
        s = self.sides[i]; c = s["ct"]
        H, W, an = self.spec_dim("H"), self.spec_dim("W"), self.spec_dim("an")
        mmpp = np.nan; start = None; u = None; note = ""; back_axis = None
        s["ct_frame"] = None
        man = bool(c.get("mcal") and s["cal"])
        if man:
            pxmm = [np.hypot(b[0] - a[0], b[1] - a[1]) / mmv for a, b, mmv in s["cal"]]
            mmpp = 1.0 / float(np.mean(pxmm))
        tip = np.asarray(c["tip"], float) if c["tip"] is not None else None
        if c["back"] and c["side1"] and c["side2"]:
            fg = ct.frame_geometry(c["back"], c["side1"], c["side2"], H, W)
            if fg is not None and man:            # same frame, scale from the manual calibration
                fg = ct.frame_geometry(c["back"], c["side1"], c["side2"], fg["H_px"] * mmpp, W)
            if fg is not None and tip is not None:
                fg = ct.frame_geometry(c["back"], c["side1"], c["side2"], H if not man else fg["H_px"] * mmpp, W,
                                       tip=tip, a0_mm=an)
            if fg is not None:
                s["ct_frame"] = fg
                mmpp = fg["mmpp"]
                if tip is not None:
                    start, u, back_axis = fg["start"], fg["n"], fg["axis_back"]
                    note = f"על קו העומס: W = {W:.3f} מ״מ מהקצה האחורי"
        elif c["back"] and tip is not None and np.isfinite(mmpp):
            b1, b2 = (np.asarray(p, float) for p in c["back"])
            e1 = (b2 - b1) / np.linalg.norm(b2 - b1); n = np.array([-e1[1], e1[0]])
            if np.dot(tip - b1, n) > 0:
                n = -n                                  # n points out through the back edge
            back_axis = b1 + np.dot(tip - b1, e1) * e1
            u = n; start = back_axis - n * W / mmpp
            note = f"על קו העומס: W = {W:.3f} מ״מ מהקצה האחורי (כיול ידני)"
        if start is None and tip is not None and np.isfinite(mmpp):
            if c.get("slot_u") is not None:
                u = np.asarray(c["slot_u"], float)
            elif c.get("mouth") is not None:
                u = tip - np.asarray(c["mouth"], float)
            if u is not None:
                u = u / max(1e-9, np.linalg.norm(u))
                start = tip - u * an / mmpp
                note = f"a_n = {an:.3f} מ״מ לפני קצה החריץ (מתמונת הדגם – הקצה האחורי לא סומן)"
        if c["start_manual"] is not None and tip is not None:
            start = np.asarray(c["start_manual"], float); note = "ידנית"
            u = tip - start; u = u / max(1e-9, np.linalg.norm(u))
        return mmpp, start, u, note, back_axis

    def _ct_update(self, i):
        s, p = self.sides[i], self.panels[i]
        mmpp, start, u, note, back_axis = self._ct_geometry(i)
        s["mm_per_px"] = mmpp
        s["back_axis"] = back_axis
        txt_scale = (f"קנה מידה: {1.0 / mmpp:.3f} px/mm  ·  {mmpp * 1000:.2f} µm/px" if np.isfinite(mmpp)
                     else "קנה מידה: לא נקבע")
        if np.isfinite(mmpp) and not s["ct"].get("mcal"):
            txt_scale += f"  (מהגובה H = {self.spec_dim('H'):.3f})"
        p.lbl_scale.setText(txt_scale)
        if start is not None:
            s["root"] = tuple(map(float, start)); s["seed"] = tuple(map(float, s["ct"]["tip"]))
            a0 = float(np.dot(np.asarray(s["seed"]) - start, u)) * mmpp
            an = self.spec_dim("an")
            dd = a0 - an
            col = "#15803d" if abs(dd) <= 0.15 else ("#b45309" if abs(dd) <= 0.4 else "#b91c1c")
            txt = (f"התחלה: {note}<br>a0 בתמונה = <b>{a0:.3f}</b> מ״מ · בתמונת הדגם a_n = {an:.3f} · "
                   f"<span style='color:{col}; font-weight:600'>הפרש {dd:+.3f}</span>")
            p.lbl_start.setText(txt)
            if back_axis is not None:
                txt += (f"<br>בדיקה מהקצה האחורי: קצה אחורי ← קצה חריץ = "
                        f"{np.dot(np.asarray(back_axis) - np.asarray(s['seed']), u) * mmpp:.3f} מ״מ (W − a0)")
            s["start_txt"] = txt
        else:
            s["root"] = s["seed"] = None
            need = []
            if not np.isfinite(mmpp):
                need.append("קנה מידה (קצה אחורי + שני הצדדים, או כיול ידני)")
            if s["ct"]["tip"] is None:
                need.append("קצה החריץ")
            elif not s["ct"]["back"] and s["ct"].get("slot_u") is None and s["ct"]["mouth"] is None:
                need.append("קצה אחורי או כיוון החריץ")
            p.lbl_start.setText("חסר: " + ", ".join(need))
        if not s["roi_user"]:
            s["roi"] = None
        self._prepare(i)
        self.show_side(i, "base")

    def _update_scale(self, i):              # manual calibration (known distance)
        s, p = self.sides[i], self.panels[i]
        p.cal_table.setRowCount(0)
        for (a, b, mm) in s["cal"]:
            px = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            r = p.cal_table.rowCount(); p.cal_table.insertRow(r)
            for c, val in enumerate((f"{px:.1f}", f"{mm:g}", f"{px / mm:.3f}")):
                it = QTableWidgetItem(val); it.setTextAlignment(Qt.AlignCenter)
                p.cal_table.setItem(r, c, it)
        if s["base"] is not None:
            self._ct_update(i)

    def set_crack(self, i, start_pt, tip):  # not used in C(T) mode
        pass

    def _spec_poly(self, i):
        fg = self.sides[i].get("ct_frame")
        if fg is None:
            return None
        k = fg["pxmm"]; n = fg["n"]; W125 = self.spec_dim("W125")
        P1, P2 = np.asarray(fg["P1"], float), np.asarray(fg["P2"], float)
        return [tuple(P1), tuple(P2), tuple(P2 - n * W125 * k), tuple(P1 - n * W125 * k)]

    def _prepare(self, i):
        s = self.sides[i]
        s["prep"] = None
        ok = False
        if s["base"] is None:
            msg = "בחר תיקייה." if i == 0 else "צלם תמונה מהטלפון (או בחר תיקייה)."
        elif not np.isfinite(s["mm_per_px"]):
            msg = "קבע קנה מידה (שלב 2)."
        elif s["seed"] is None:
            msg = "חסר קצה החריץ / נקודת ההתחלה (שלב 3)."
        else:
            try:
                P = self._params(); P["sensitivity"] = self.panels[i].sens.value()
                P["crack_polarity"] = self.panels[i].pol.currentData() or "auto"
                files = s["files"]
                d = dict(mm_per_px=s["mm_per_px"], seed=s["seed"], root=s["root"], ref=s["root"],
                         base=s["base"], P=P, roi=s["roi"] if s["roi_user"] else None, base_file=s["base_file"],
                         notch=True, a_mode="proj", spec_poly=self._spec_poly(i),
                         back_axis=s.get("back_axis"), W_mm=self.spec_dim("W"),
                         noise_file=files[1] if len(files) > 1 and files[0] == s["base_file"] else None)
                prep = cc.prepare_side(d)
                prep.update(index=i, folder=s["folder"], files=s["files"])
                prep["thr_eff0"] = prep.get("thr_eff")
                s["prep"], s["roi"] = prep, prep["roi"]
                a0 = prep["a0_px"] * prep["mm_per_px"]
                msg = f"מוכן · a0 = {a0:.3f} מ״מ (מקו העומס עד קצה החריץ)"
                if not prep.get("noise_from"):
                    msg += " · הסף מתמונה אחת בלבד – עדיף שתהיה בתיקייה עוד תמונה לפני תחילת הסדק"
                ok = True
            except ValueError as ex:
                msg = str(ex)
        s["msg"] = msg
        lbl = self.panels[i].lbl_crack
        lbl.setText(msg)
        lbl.setStyleSheet("color:#15803d; font-weight:600" if ok else "color:#b91c1c")
        self._update_ready()

    def _draw_setup(self, i, img):
        super()._draw_setup(i, img)
        s, v = self.sides[i], self.su.view
        c = s.get("ct") or {}
        if s.get("view") != "base":
            return
        if c.get("mouth"):
            v.add_marker(*c["mouth"], "+", "#7dd3fc", 7, 1.5)
        if c.get("tip") and not s["prep"]:
            v.add_marker(*c["tip"], "circle", "#ff2d2d", 9, 2)
            self.su.pair.show_zoom(img, c["tip"], s["mm_per_px"] if np.isfinite(s["mm_per_px"]) else None,
                                   lambda z: z.add_marker(*c["tip"], "ticks", "#ff2d2d", 10, 2))
        for key in ("back", "side1", "side2"):
            if c.get(key):
                v.add_polyline(list(c[key]), C_EDGE, 2.0, z=8)
        if s.get("back_axis") is not None:
            v.add_marker(*s["back_axis"], "+", C_EDGE, 9, 2)
        p = s["prep"]
        if p and self.grid_on:
            draw_grid(v, p["ref"], p["u"], p["mm_per_px"], *self._grid_ext(i))

    def _render(self, pair, i, res, img):
        super()._render(pair, i, res, img)
        p = self.sides[i]["prep"]
        if p and self.grid_on:
            d = np.asarray(res.d if res.d is not None and np.all(np.isfinite(res.d)) else (0.0, 0.0), float)
            draw_grid(pair.view, np.asarray(p["ref"], float) + d, p["u"], p["mm_per_px"], *self._grid_ext(i))

    def _grid_ext(self, i):
        """Grid extent in mm around the start point: a from the front edge to the back edge,
        across from one side edge to the other."""
        W, W125, H = self.spec_dim("W"), self.spec_dim("W125"), self.spec_dim("H")
        lo, hi = -H / 2, H / 2
        s, p = self.sides[i], self.sides[i]["prep"]
        fg = s.get("ct_frame")
        if fg is not None and p is not None:
            k = fg["pxmm"]; u = np.asarray(p["u"], float); v = np.array([-u[1], u[0]])
            o = np.asarray(p["ref"], float)
            a_ = [float(np.dot(np.asarray(P, float) - o, v)) / k for P in (fg["P1"], fg["P2"])]
            lo, hi = min(a_), max(a_)
        return -(W125 - W), W, lo, hi

    # ------------------------------------------------------------------ side B: the phone
    def set_use_b(self, on):
        super().set_use_b(on)
        if on and hasattr(self, "spec") and not self.b_phone and not self.sides[1]["folder"]:
            self.panels[1].rb_phone.setChecked(True)          # side B is the phone by default

    def _set_b_base(self, path, img=None):
        super()._set_b_base(path, img)
        self.sides[1]["ct"] = self._new_side()["ct"]
        self.ct_auto(1)

    def _phone_ui(self):
        super()._phone_ui()
        ph, p = self.phone, self.panels[1]
        if ph is None or not self.b_phone:
            return
        auto = p.chk_autocam.isChecked()
        p.btn_pfocus.setText("עצור כיוון" if ph.auto.running else
                             ("כוון מצלמה מחדש ונעל" if ph.focus_locked is not None else "כוון מצלמה ונעל"))
        p.btn_pfocus.setEnabled(not auto or ph.auto.running)

    def phone_autocam(self, on):
        ph = self._phone()
        cam = ph.cfg["camera"]
        if on:
            if ph.auto.running:
                ph.auto.cancel()
            self._cam_saved = dict(cam)
            cam.update(ae="auto", af="auto", awb="auto")
            cam.setdefault("focus", 5.0)                # keeps the settings sent again on every reconnect
            ph.focus_locked = None
            self.panels[1].lbl_pfocus.setText("מצלמה אוטומטית: פוקוס, חשיפה ואיזון לבן בכל צילום")
            self.log("צד B: מצלמה אוטומטית (בלי נעילה)")
        else:
            if self._cam_saved:
                cam.update(self._cam_saved)
            cam.update(ae="manual", af="manual", awb="lock")
            if self._cam_saved and "focus" in self._cam_saved and self._cam_saved.get("af") == "manual":
                ph.focus_locked = self._cam_saved["focus"]
            self.panels[1].lbl_pfocus.setText("נעילה הוחזרה" + (f" · פוקוס {cam['focus']:.2f} D" if "focus" in cam else "")
                                              + " – מומלץ \"כוון מצלמה ונעל\"")
            self.log("צד B: מצלמה נעולה")
        ph.send({"type": "set", "camera": cam})
        self._phone_ui()

    def _gcam(self):
        return isinstance(self.phone, gcam_b.GCamB)

    def _phone(self):
        new = self.phone is None
        if new and self.panels[1].chk_gcam.isChecked():
            orig, base.phone_b.PhoneB = base.phone_b.PhoneB, gcam_b.GCamB
            try:
                ph = super()._phone()
            finally:
                base.phone_b.PhoneB = orig
            ph.timeout_s = max(ph.timeout_s, 15.0)
            z = float(self.panels[1].ph_zoom.currentData() or 1.0)
            ph.zoom_chip = z if z in gcam_b.CHIPS else 1.0
            return ph
        ph = super()._phone()
        if new and not self._gcam():
            z = ph.cfg["camera"]["zoom"] = float(self.panels[1].ph_zoom.currentData() or 1.0)
            ph.focus_max = PHONE_TELE_MIN_FOCUS_D if z >= 5 else None
        return ph

    def phone_zoom(self, *_):
        z = float(self.panels[1].ph_zoom.currentData() or 1.0)
        self.settings.setValue("phone_zoom", z)
        ph = self._phone()
        if self._gcam():
            ph.zoom_preset(z)
            self.panels[1].lbl_pfocus.setText(f"זום {z:g}× בטלפון · גלגלת על התצוגה החיה = זום עדין, לחיצה = פוקוס. "
                                              "אחרי שינוי זום – צלם תמונת בסיס מחדש.")
            self.log(f"צד B: זום מצלמת גוגל {z:g}×")
            return
        if ph.cfg["camera"].get("zoom") == z:
            return
        if ph.auto.running:
            ph.auto.cancel()
        ph.cfg["camera"]["zoom"] = z
        ph.focus_max = PHONE_TELE_MIN_FOCUS_D if z >= 5 else None   # focus scan over the telephoto's range only
        if z >= 5 and ph.cfg["camera"].get("focus", 0) > PHONE_TELE_MIN_FOCUS_D:
            ph.cfg["camera"]["focus"] = PHONE_TELE_MIN_FOCUS_D / 2
        ph.focus_locked = None                          # another lens / magnification: focus and scale are new
        ph.ref_sharp = None
        ph.send({"type": "set", "camera": ph.cfg["camera"]})
        self.panels[1].lbl_pfocus.setStyleSheet("")
        self.panels[1].lbl_pfocus.setText(f"זום {z:g}× – עכשיו \"כוון מצלמה ונעל\" וצלם תמונת בסיס מחדש (קנה המידה השתנה)"
            + (" · עדשת הטלפוטו מתמקדת רק מ-80 ס״מ ומעלה" if z >= 5 else ""))
        self.log(f"צד B: זום הטלפון {z:g}×")
        self._phone_ui()

    def phone_gcam(self, on):
        self.settings.setValue("phone_gcam", "true" if on else "false")
        p = self.panels[1]
        p.btn_plive.setChecked(False)
        if self.phone is not None:
            self.phone.stop()
            self.phone.deleteLater()
            self.phone = None
        self._gcam_widgets()
        if self.b_phone:
            self._phone().start()
            self.arm_phone() if hasattr(self, "arm_phone") else None
        self.log("צד B: " + ("מצלמת גוגל" if on else "אפליקציית FatigueCam"))
        self._phone_ui()

    def _gcam_widgets(self):
        p = self.panels[1]
        g = p.chk_gcam.isChecked()
        for w in (p.chk_autocam, p.ph_focus):
            w.setVisible(not g)
        lay = p.ph_focus.parentWidget().layout() if p.ph_focus.parentWidget() else None
        zb = p.ph_zoom
        z = float(zb.currentData() or 1.0)
        zb.blockSignals(True)
        zb.clear()
        for v in ((1.0, 2.0, 5.0) if g else PHONE_ZOOMS):
            kind = ("רגילה" if v == 1 else "טלפוטו – מ-80 ס״מ" if v >= 5 else "חיתוך") if g else                 ("רחב" if v == 1 else ("עדשת טלפוטו – אופטי, חד" if v >= 5 else "חיתוך דיגיטלי"))
            zb.addItem(f"{v:g}×  ({kind})", v)
        i = zb.findData(z)
        zb.setCurrentIndex(i if i >= 0 else zb.count() - 1)
        zb.blockSignals(False)
        p.btn_pfocus.setText("פוקוס על אזור החריץ" if g else "כוון מצלמה ונעל")
        p.btn_pfocus.setToolTip("נוגע במסך הטלפון במרכז אזור הפוקוס (קצה החריץ אם סומן) – מצלמת גוגל מתמקדת שם.\n"
                                "אפשר גם ללחוץ ישירות על התצוגה החיה." if g else p.btn_pfocus.toolTip())

    def eventFilter(self, obj, ev):
        """Live view of Google Camera: wheel = phone zoom, click = focus there."""
        if obj is self.su.view.viewport() and self._gcam() and self.panels[1].btn_plive.isChecked()                 and self.pick is None and self.su.view.img_shape:
            t = ev.type()
            if t == ev.Type.Wheel:
                n = 1 if ev.angleDelta().y() > 0 else -1
                self.phone.zoom_step(n)
                self.su.info.setText("זום " + ("+" if n > 0 else "−") + " בטלפון...")
                return True
            if t == ev.Type.MouseButtonPress and ev.button() == Qt.LeftButton:
                self._gc_press = ev.position()
            if t == ev.Type.MouseButtonRelease and ev.button() == Qt.LeftButton and getattr(self, "_gc_press", None) is not None                     and (ev.position() - self._gc_press).manhattanLength() < 5:
                sp = self.su.view.mapToScene(ev.position().toPoint())
                h, w = self.su.view.img_shape
                if 0 <= sp.x() < w and 0 <= sp.y() < h:
                    self.phone.focus_at(sp.x() / w, sp.y() / h)
                    self.su.info.setText("פוקוס בנקודה שנבחרה...")
                self._gc_press = None
        return super().eventFilter(obj, ev)

    def _phone_preview(self, q):
        if not self._gcam():
            return super()._phone_preview(q)
        p = self.panels[1]
        if not (self.b_phone and p.btn_plive.isChecked() and self.pick is None and self.tabs.currentIndex() == 1
                and self.stack.currentIndex() == PAGE_SETUP):
            return
        v = self.su.view
        v.clear_overlays()
        v.set_ruler(None)
        v.set_qimage(q)
        self.su.info.setText("מסך מצלמת גוגל · גלגלת = זום בטלפון · לחיצה = פוקוס בנקודה")

    def phone_focus(self):
        p = self.panels[1]
        if self._gcam():
            self._phone_roi_from_crack()
            self.phone.start_focus()
            p.lbl_pfocus.setText("מצלמת גוגל: פוקוס על אזור החריץ (נגיעה במסך)")
            return
        if p.chk_autocam.isChecked():
            p.chk_autocam.setChecked(False)
        super().phone_focus()

    def _phone_roi_from_crack(self):
        """Focus / exposure area of the phone: around the notch tip (raw phone frame), else the centre."""
        s = self.sides[1]
        if s["seed"] is None or s["base"] is None or not s["base_file"]:
            return
        k = s["ct"].get("rot") or 0
        try:
            h0, w0 = Image.open(s["base_file"]).size[::-1]
        except Exception:
            return
        x, y = to_raw(s["seed"], k, (h0, w0))
        rw, rh = 0.3, 0.3
        self.phone.set_roi(float(np.clip(x / w0 - rw / 2, 0, 1 - rw)), float(np.clip(y / h0 - rh / 2, 0, 1 - rh)), rw, rh)

    def phone_capture(self, purpose):
        ph = self._phone()
        if not self.sides[1]["folder"]:
            self.toast("בחר קודם תיקיית שמירה לצד B.", "warn"); return
        if not ph.connected:
            self.toast("אין חיבור לטלפון.", "warn"); return
        if ph.focus_locked is None and not self.panels[1].chk_autocam.isChecked():
            self.toast("המצלמה עוד לא כוונה – מומלץ \"כוון מצלמה ונעל\" לפני הצילום.", "warn", 5000)
        if ph.capture_setup(purpose):
            self.set_setup_hint("מצלם בטלפון...")

    # ------------------------------------------------------------------ measuring
    def start_worker(self, live):
        self._guard_on = self.chk_guard.isChecked()
        orig = base.Worker
        base.Worker = lambda sides, lv, k0: CTWorker(sides, lv, k0, guard=self._guard_on)
        try:
            super().start_worker(live)
        finally:
            base.Worker = orig
        if isinstance(self.worker, CTWorker):
            self.worker.revised.connect(self._on_revised)

    def _on_revised(self, row, i, res):
        """The growth guard corrected an earlier row (its growth was not confirmed by the next image)."""
        R = self.R
        if row >= len(R["res"]) or R["orig"][row][i] is None or res.file != R["orig"][row][i].file:
            return
        old = R["orig"][row][i]
        res.a_back = getattr(old, "a_back", np.nan) + (res.a - old.a)     # same back edge, tip moved along u
        R["orig"][row][i] = res
        if R["res"][row][i] is None or R["res"][row][i].status != "manual":
            R["res"][row][i] = res
        self._update_row(row, full=(row == 0))
        if row == self.cur_row:
            self.show_row(row)
        if getattr(res, "guard", ""):
            self.log(f"תמונה {R['k'][row]}, צד {SIDE_NAMES[i]}: {res.guard} → a = {res.a:.3f}")

    def set_polarity(self, i):
        """Dark / bright / auto crack: applies to the next images, also during a run (the worker uses
        the same prepared side)."""
        pol = self.panels[i].pol.currentData() or "auto"
        self.settings.setValue(f"crack_polarity_{i}", pol)
        p = self.sides[i]["prep"]
        if self._busy() and p is not None:
            p["P"]["crack_polarity"] = pol
            self.log(f"צד {SIDE_NAMES[i]}: סוג הסדק = {self.panels[i].pol.currentText()} (מהתמונה הבאה)")
        elif self.sides[i]["base"] is not None:
            self._prepare(i)

    def start_move_start(self, i):
        """Measuring page: pick a new start point (load line) for side i - also during a run."""
        if i == 1 and not self.use_b:
            return
        p = self.sides[i]["prep"]
        if p is None:
            self.toast("הצד הזה עוד לא מוכן.", "warn"); return
        if self.pick and self.pick.get("mode") == "start" and self.pick["side"] == i:
            self._cancel_pick(); self.set_hint(""); return              # second press = cancel
        self._cancel_pick()
        row = self.cur_row if 0 <= self.cur_row < len(self.R["res"]) else None
        if row is not None:
            self._set_follow(False)
        self.pick = dict(mode="start", side=i, pts=[], row=row)
        self.sv[i].pair.set_pick(True)
        self.set_hint(f"נקודת התחלה · צד {SIDE_NAMES[i]}: לחץ על הנקודה הנכונה על קו העומס (מומלץ בהגדלה). "
                      f"כל השורות יחושבו מחדש – קצה הסדק לא זז.  Esc = ביטול", True)

    def _move_start(self, i, x, y, row):
        """New start point: a = projection of (tip - start) on the crack axis, so every row changes by
        the same amount; the tips, the growth guard and Δa are unchanged."""
        s, p = self.sides[i], self.sides[i]["prep"]
        d = np.zeros(2)
        if row is not None and self.R["res"][row][i] is not None:
            rd = self.R["res"][row][i].d
            if rd is not None and np.all(np.isfinite(rd)):
                d = np.asarray(rd, float)
        u = np.asarray(p["u"], float)
        old = np.asarray(p["ref"], float)
        shift_px = float(np.dot(np.asarray((x, y), float) - d - old, u))
        new = old + shift_px * u                       # stays on the crack axis
        delta = shift_px * p["mm_per_px"]              # every a gets shorter by this much
        p["ref"] = p["root"] = (float(new[0]), float(new[1]))
        p["a0_px"] = float(p["a0_px"] - shift_px)
        s["root"] = p["root"]
        s["ct"]["start_manual"] = p["ref"]
        seen = set()
        for rr in range(len(self.R["res"])):
            for key in ("res", "orig"):
                res = self.R[key][rr][i]
                if res is not None and id(res) not in seen and np.isfinite(res.a):
                    seen.add(id(res)); res.a -= delta
        w = self.worker
        g = getattr(w, "guards", {}).get(i) if w is not None else None
        if g is not None:
            if g.anchor is not None:
                g.anchor["a"] -= delta
            if g.pending is not None and id(g.pending["res"]) not in seen:
                g.pending["res"].a -= delta
        self._full_refresh()
        a0 = p["a0_px"] * p["mm_per_px"]
        self.log(f"צד {SIDE_NAMES[i]}: נקודת ההתחלה הוזזה ב-{delta:+.3f} מ״מ לאורך הציר – a0 = {a0:.3f} מ״מ, "
                 f"כל השורות עודכנו.")
        self.set_hint(f"נקודת ההתחלה של צד {SIDE_NAMES[i]} עודכנה ({delta:+.3f} מ״מ) · a0 = {a0:.3f} מ״מ.")

    def _on_fix_click(self, i, x, y):
        if self.pick and self.pick.get("mode") == "start" and self.pick["side"] == i:
            row = self.pick["row"]
            self._cancel_pick()
            self._move_start(i, x, y, row)
            return
        r = self.pick["row"] if self.pick and self.pick.get("mode") == "fix" else None
        super()._on_fix_click(i, x, y)
        w = self.worker
        if r is None or w is None or not w.isRunning() or r != len(self.R["k"]) - 1:
            return
        g = getattr(w, "guards", {}).get(i)
        res = self.R["res"][r][i]
        if g is not None and res is not None and res.status == "manual":
            g.set_manual(res)
            w.prev[i] = res.reach
            self.log(f"צד {SIDE_NAMES[i]}: הקצה המתוקן הוא הבסיס להמשך המעקב.")

    def _extend_measure_page(self):
        lay = self.btn_full.parentWidget().layout()
        self.chk_grid = QCheckBox("רשת מ״מ"); self.chk_grid.setChecked(True)
        self.chk_grid.setToolTip("רשת עדינה של 1 מ״מ על הדגם, מקו העומס (תוויות = a במ״מ)")
        self.chk_grid.toggled.connect(self._toggle_grid)
        lay.insertWidget(lay.indexOf(self.btn_full), self.chk_grid)
        for i, sv in enumerate(self.sv):
            b = QToolButton(); b.setText(" הזז נקודת התחלה")
            b.setToolTip("לחץ, ואז לחץ על נקודת ההתחלה הנכונה על קו העומס.\n"
                         "אפשר גם באמצע ריצה: כל השורות (גם הקודמות) מחושבות מחדש – קצה הסדק לא זז.")
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon); b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, i=i: self.start_move_start(i))
            head = sv.btn_fix.parentWidget().layout().itemAt(0).layout()
            head.insertWidget(head.indexOf(sv.btn_fix), b)
            sv.btn_start = b

    def _toggle_grid(self, on):
        self.grid_on = bool(on)
        if self.R["k"] and self.cur_row >= 0:
            self.show_row(self.cur_row)
        elif self.stack.currentIndex() == PAGE_MEASURE:
            self._preview_bases()
        self.redraw()

    def _extend_results_page(self):
        page = self.stack.widget(PAGE_RESULTS)
        lay = page.layout()
        bar = QFrame(); bar.setObjectName("Toolbar")
        bl = QHBoxLayout(bar); bl.setContentsMargins(10, 6, 10, 6); bl.setSpacing(8)
        bl.addWidget(_btn("בחר נקודות לפי E647", "primary", slot=self.select_e647,
                          tip="מסמן לייצוא נקודה אחת לכל מרווח מדידה: Δa ≥ המרווח המינימלי, ולא יותר מהמרווח "
                              "המקסימלי לפי a/W (0.25–0.40: 1.2 · 0.40–0.60: 0.6 · ≥0.60: 0.3 מ״מ).\n"
                              "פערים גדולים מהמותר בלי מדידה נצבעים באדום."))
        bl.addWidget(QLabel("Δa מינימלי:"))
        self.e_min = QDoubleSpinBox(); self.e_min.setRange(0.01, 5); self.e_min.setDecimals(2); self.e_min.setValue(0.25)
        self.e_min.setSuffix(" מ״מ"); bl.addWidget(self.e_min)
        self.e_res = QCheckBox("ולפחות 10× רזולוציה"); self.e_res.setChecked(True); bl.addWidget(self.e_res)
        bl.addWidget(QLabel("צעד:"))
        self.e_step = QComboBox()
        self.e_step.addItem("המינימלי – הכי הרבה נקודות", "min")
        self.e_step.addItem("המקסימלי המותר – הכי מעט נקודות", "max")
        bl.addWidget(self.e_step)
        self.e_info = QLabel(""); self.e_info.setObjectName("Muted"); bl.addWidget(self.e_info, 1)
        lay.insertWidget(2, bar)

    def _extend_alerts_page(self):
        g = self.alert_w["diff_thr"].parentWidget()
        b = _btn("0.25·B לפי הדגם", tip="E647: |a₁ − a₂| ≤ 0.25B", slot=self._diff_from_B)
        g.layout().addRow("", b)
        # growth alerts: a window that stays on top + a beep, during the measurement
        aw = self.alert_w
        gg = QGroupBox("התראה קופצת + צפצוף על התקדמות הסדק")
        f = QFormLayout(gg); f.setSpacing(10)
        aw["start_on"] = QCheckBox("כשהסדק מתחיל לצמוח מקצה החריץ – Δa לפחות")
        aw["start_thr"] = QDoubleSpinBox(); aw["start_thr"].setRange(0.01, 5); aw["start_thr"].setDecimals(2)
        aw["start_thr"].setSingleStep(0.05); aw["start_thr"].setSuffix("  מ״מ")
        aw["step_on"] = QCheckBox("כל פעם שהסדק מתקדם עוד")
        aw["step_mm"] = QDoubleSpinBox(); aw["step_mm"].setRange(0.05, 20); aw["step_mm"].setDecimals(2)
        aw["step_mm"].setSingleStep(0.25); aw["step_mm"].setSuffix("  מ״מ")
        aw["len_on"] = QCheckBox("כשהסדק מגיע לאורך יעד")
        aw["len_mode"] = QComboBox()
        aw["len_mode"].addItem("a (מ״מ, מקו העומס)", "a"); aw["len_mode"].addItem("Δa (מ״מ, מקצה החריץ)", "da")
        aw["len_mode"].addItem("a/W", "aw")
        aw["len_val"] = QDoubleSpinBox(); aw["len_val"].setRange(0.0, 200); aw["len_val"].setDecimals(3)
        aw["len_val"].setSingleStep(0.5)
        aw["beep_on"] = QCheckBox("צפצוף")
        aw["beep_repeat"] = QCheckBox("לצפצף שוב כל כמה שניות עד שלוחצים \"הבנתי\"")
        f.addRow(aw["start_on"]); f.addRow("סף התחלה:", aw["start_thr"])
        f.addRow(aw["step_on"]); f.addRow("צעד:", aw["step_mm"])
        f.addRow(aw["len_on"]); f.addRow("לפי:", aw["len_mode"]); f.addRow("יעד:", aw["len_val"])
        f.addRow(aw["beep_on"]); f.addRow(aw["beep_repeat"])
        n = QLabel("נבדק על כל תמונה חדשה בזמן המדידה, אחרי שהצמיחה אושרה (בתמונה שאחריה) – בלי צפצופי שווא "
                   "מרעש. הערך = a הממוצע של הצדדים הפעילים.")
        n.setObjectName("Muted"); n.setWordWrap(True); f.addRow(n)
        f.addRow("", _btn("נסה צפצוף", slot=lambda: self._growth_popup("בדיקה: כך תיראה ההתראה.", test=True)))
        g.parentWidget().layout().insertWidget(g.parentWidget().layout().indexOf(g) + 1, gg)
        for key in ("start_on", "start_thr", "step_on", "step_mm", "len_on", "len_mode", "len_val", "beep_on",
                    "beep_repeat"):
            wd = aw[key]
            wd.blockSignals(True)
            if isinstance(wd, QCheckBox):
                wd.setChecked(bool(self.alerts[key]))
            elif isinstance(wd, QComboBox):
                wd.setCurrentIndex(max(0, wd.findData(self.alerts[key])))
            else:
                wd.setValue(float(self.alerts[key]))
            wd.blockSignals(False)
            sig = wd.toggled if isinstance(wd, QCheckBox) else (
                wd.currentIndexChanged if isinstance(wd, QComboBox) else wd.valueChanged)
            sig.connect(lambda *_, key=key: self._alert_widget_changed(key))
        self._ga = None
        self._alert_dlg = None
        self._beep_timer = QTimer(self); self._beep_timer.setInterval(4000); self._beep_timer.timeout.connect(self._beep)
        self._alerts_enable()

    def _alerts_enable(self):
        super()._alerts_enable()
        aw = self.alert_w
        if "start_on" in aw:
            aw["start_thr"].setEnabled(aw["start_on"].isChecked())
            aw["step_mm"].setEnabled(aw["step_on"].isChecked())
            aw["len_mode"].setEnabled(aw["len_on"].isChecked()); aw["len_val"].setEnabled(aw["len_on"].isChecked())
            aw["beep_repeat"].setEnabled(aw["beep_on"].isChecked())

    # ------------------------------------------------------------------ growth alerts
    def _growth_check(self, r):
        """Row r is final (the guard confirmed / corrected it): crack start, growth steps, target length."""
        R, a_ = self.R, self.alerts
        if r < 0 or r >= len(R["k"]):
            return
        a, da = R["a"][r], R["da"][r]
        if not (np.isfinite(a) and np.isfinite(da)):
            return
        ga = self._ga = self._ga or dict(started=False, level=0, len_done=False, last_row=-1)
        if r <= ga["last_row"]:
            return
        ga["last_row"] = r
        W = self.spec_dim("W")
        msgs = []
        if a_["start_on"] and not ga["started"] and da >= a_["start_thr"]:
            ga["started"] = True
            msgs.append(f"הסדק התחיל לצמוח: Δa = {da:.3f} מ״מ")
        if a_["step_on"] and a_["step_mm"] > 0:
            lev = int(np.floor(da / a_["step_mm"] + 1e-9))
            if lev > ga["level"]:
                ga["level"] = lev
                msgs.append(f"הסדק התקדם: Δa = {da:.3f} מ״מ (עבר {lev * a_['step_mm']:.2f})")
        if a_["len_on"] and not ga["len_done"]:
            v = {"a": a, "da": da, "aw": a / W if W > 0 else np.nan}[a_["len_mode"]]
            if np.isfinite(v) and v >= a_["len_val"]:
                ga["len_done"] = True
                name = {"a": "a", "da": "Δa", "aw": "a/W"}[a_["len_mode"]]
                msgs.append(f"<b>הסדק הגיע לאורך היעד: {name} = {v:.3f} (יעד {a_['len_val']:g})</b>")
        if msgs:
            self._growth_popup(f"תמונה {R['k'][r]} · a = {a:.3f} מ״מ · a/W = {a / W:.4f}<br>" + "<br>".join(msgs))

    def _growth_popup(self, html, test=False):
        stamp = time.strftime("%H:%M:%S")
        for m in html.split("<br>"):
            self.log("התראה: " + m.replace("<b>", "").replace("</b>", ""))
        dlg = self._alert_dlg
        if dlg is None:
            dlg = self._alert_dlg = QMessageBox(self)
            dlg.setWindowTitle("CT Tracker – התראה")
            dlg.setIcon(QMessageBox.Warning)
            dlg.setTextFormat(Qt.RichText)
            dlg.setStandardButtons(QMessageBox.Ok)
            dlg.button(QMessageBox.Ok).setText("הבנתי")
            dlg.setWindowModality(Qt.NonModal)
            dlg.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            dlg.finished.connect(lambda *_: self._beep_timer.stop())
            dlg._lines = []
        dlg._lines = (dlg._lines + [f"<b>{stamp}</b> · {html}"])[-6:]
        dlg.setText("<br><br>".join(dlg._lines))
        dlg.show(); dlg.raise_(); dlg.activateWindow()
        self.toast(html.split("<br>")[-1].replace("<b>", "").replace("</b>", ""), "warn", 8000)
        if self.alerts.get("beep_on", True):
            self._beep()
            if self.alerts.get("beep_repeat", True) and not test:
                self._beep_timer.start()

    def _beep(self):
        try:
            import winsound
            for f in (1400, 1000, 1400):
                winsound.Beep(f, 180)
        except Exception:
            QApplication.beep()

    def _diff_from_B(self):
        self.alert_w["diff_thr"].setValue(round(0.25 * self.spec_dim("B"), 3))

    def select_e647(self):
        n = len(self.R["k"])
        if not n:
            self.toast("עדיין לא נמדד כלום.", "warn"); return
        W = self.spec_dim("W")
        res = max((self.sides[i]["mm_per_px"] for i in self._active() if np.isfinite(self.sides[i]["mm_per_px"])),
                  default=0.0)
        min_inc = self.e_min.value()
        if self.e_res.isChecked():
            min_inc = max(min_inc, 10 * res)
        # failed measurements are skipped; uncertain ones are kept (they are flagged in the table)
        a = np.array([x if np.isfinite(x) and base._kind(st) != "bad" else np.nan
                      for x, st in zip(self.R["a"], self.R["status"])], float)
        step = None if self.e_step.currentData() == "min" else 1e9
        sel, gaps = ct.select_points(a, W, min_inc, step)
        self.R["export"] = [bool(x) for x in sel]
        self.gap_rows = set(); self.gap_tip = {}
        for i0, i1, jump, allowed in gaps:
            for r in (i0, i1):
                self.gap_rows.add(r)
                self.gap_tip[r] = (f"פער מדידה: Δa = {jump:.3f} מ״מ בין תמונה {self.R['k'][i0]} לתמונה "
                                   f"{self.R['k'][i1]} – המותר ב-a/W = {a[i0] / W:.3f} הוא {allowed:.2f} מ״מ")
        self.model.all_changed()
        self._update_stats()
        self.e_info.setText(f"נבחרו {int(sel.sum())} מתוך {n} · Δa מינימלי {min_inc:.3f} מ״מ "
                            f"(רזולוציה {res * 1000:.1f} µm/px) · W = {W:.3f} · "
                            + (f"<span style='color:#b91c1c'><b>{len(gaps)} פערים</b></span>" if gaps else "אין פערים"))
        self.e647_sel = dict(min_inc=min_inc, W=W, gaps=gaps)
        if gaps:
            self.toast(f"{len(gaps)} פערים גדולים מהמותר – מסומנים באדום בטבלה", "warn", 5000)

    def alert_cells(self, r):
        out = super().alert_cells(r)
        if r in self.gap_rows:
            out = out | set(range(1, len(base.HEADERS)))
        return out

    def _reset_results(self):
        super()._reset_results()
        self.gap_rows = set(); self.gap_tip = {}; self.e647_sel = None
        self._ga = None

    def _on_result(self, out):
        super()._on_result(out)
        try:
            # a row is final once the guard has seen the images that confirm its growth
            lag = 1 + int(cc.default_params().get("guard_confirm_n", 2)) if getattr(self, "_guard_on", True) else 1
            self._growth_check(len(self.R["k"]) - lag)
        except Exception as ex:
            self.log(f"שגיאה בבדיקת ההתראות: {ex}")

    def _on_finished(self):
        super()._on_finished()
        try:
            for r in range(max(0, len(self.R["k"]) - 3), len(self.R["k"])):
                self._growth_check(r)
        except Exception as ex:
            self.log(f"שגיאה בבדיקת ההתראות: {ex}")

    # ------------------------------------------------------------------ export / project
    def _base_dir(self):
        return self.sides[0]["folder"] or (os.path.dirname(self.spec["image"]) if self.spec["image"] else os.path.expanduser("~"))

    def export_csv(self):
        if not self.R["k"]:
            self.toast("עדיין לא נמדד כלום.", "warn"); return
        rows = self._export_rows()
        if not rows:
            return
        name = self.spec["name"] or "specimen"
        f, _ = QFileDialog.getSaveFileName(self, "שמירת תוצאות", os.path.join(self._base_dir(), f"{name}_results.csv"),
                                           "CSV (*.csv)")
        if f:
            self._write_csv(f, rows)

    def _write_csv(self, f, rows):
        R = self.R; W = self.spec_dim("W")
        names = {base.C_DIFF: "diff", base.C_THA: "angle_new_A", base.C_THB: "angle_new_B",
                 base.C_THA0: "angle_total_A", base.C_THB0: "angle_total_B"}
        try:
            with open(f, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow(["specimen", self.spec["name"]]); w.writerow(["W_mm", f"{W:.4f}"])
                w.writerow(["B_mm", f"{self.spec_dim('B'):.4f}"]); w.writerow(["a_measured_from", "load line (projected, ASTM E647)"])
                w.writerow(["growth_guard", "on" if getattr(self, "_guard_on", True) else "off"])
                if self.e647_sel:
                    w.writerow(["E647_min_increment_mm", f"{self.e647_sel['min_inc']:.4f}"])
                w.writerow([])
                w.writerow(["image", "file_A", "file_B", "a_mm_sideA", "a_mm_sideB", "a_mm", "a_over_W", "delta_a_mm",
                            "diff_AB_mm", "a_back_edge_A", "a_back_edge_B", "raw_a_A", "raw_a_B",
                            "angle_new_A_deg", "angle_new_B_deg", "angle_total_A_deg",
                            "angle_total_B_deg", "sensitivity_A", "sensitivity_B", "measurement_gap", "alerts", "status"])
                for i in rows:
                    ang = R["ang"][i]
                    sens = [("" if x is None else x) for x in R["sens"][i]]
                    a = R["a"][i]
                    rs = R["res"][i]
                    raw = [getattr(x, "raw_a", np.nan) if x is not None else np.nan for x in rs]
                    w.writerow([R["k"][i], R["file"][i], R["fileB"][i] if self.use_b else "", _num(R["aA"][i]),
                                _num(R["aB"][i]), _num(a), f"{a / W:.5f}" if np.isfinite(a) else "", _num(R["da"][i]),
                                _num(self._diff(i)), _num(_a_back_mm(rs[0])), _num(_a_back_mm(rs[1])) if self.use_b else "",
                                _num(raw[0]), _num(raw[1]) if self.use_b else "",
                                _num(ang[0][0]), _num(ang[1][0]), _num(ang[0][1]), _num(ang[1][1]),
                                sens[0], sens[1] if self.use_b else "", "GAP" if i in self.gap_rows else "",
                                ";".join(names[c] for c in sorted(base.MainWindow.alert_cells(self, i))), R["status"][i]])
            self.toast(f"הטבלה נשמרה · {len(rows)} שורות", "ok")
        except OSError as ex:
            QMessageBox.warning(self, APP_NAME, f"לא ניתן לשמור את הקובץ:\n{ex}")

    def export_annotated(self):
        self.marked_dirname = f"{self.spec['name'] or 'specimen'}_marked"
        super().export_annotated()

    def save_setup(self):
        name = self.spec["name"] or "specimen"
        f, _ = QFileDialog.getSaveFileName(self, "שמירת פרויקט", os.path.join(self._base_dir(), f"{name}.json"),
                                           "JSON (*.json)")
        if not f:
            return
        sp = self.spec
        data = dict(app=APP_NAME, version=VERSION, use_b=self.use_b, b_phone=self.b_phone,
                    thr=self.thr.value(), thr_auto=self.thr_auto.isChecked(),
                    ahead=self.ahead.value(), prior=self.chk_prior.isChecked(), check=self.chk_check.isChecked(),
                    guard=self.chk_guard.isChecked(), alerts=self.alerts,
                    autocam=self.panels[1].chk_autocam.isChecked(),
                    specimen=dict(name=sp["name"], image=sp["image"], feats=sp["feats"],
                                  H=None if not np.isfinite(sp["H"]) else sp["H"],
                                  B=None if not np.isfinite(sp["B"]) else sp["B"]),
                    loads=dict(stages=self.loads["stages"], W0=self.ld_W0.value(), B0=self.ld_B0.value(),
                               lbf=self.ld_lbf.value(), tol=self.ld_tol.value()),
                    sides=[dict(folder=s["folder"], base_file=s["base_file"], cal=s["cal"], ct=s["ct"],
                                sens=self.panels[j].sens.value(), roi=s["roi"] if s["roi_user"] else None,
                                roi_user=s["roi_user"]) for j, s in enumerate(self.sides)])
        with open(f, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        self.toast(f"הפרויקט נשמר: {os.path.basename(f)}", "ok")

    def load_setup(self):
        f, _ = QFileDialog.getOpenFileName(self, "פתיחת פרויקט", self._base_dir(), "JSON (*.json)")
        if not f:
            return
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        sp = data.get("specimen", {})
        self.name_edit.setText(sp.get("name", ""))
        feats = {}
        for k, v in (sp.get("feats") or {}).items():
            feats[k] = tuple(v) if k in ("hole1", "hole2", "tip") else (tuple(v[0]), tuple(v[1]))
        self.spec["feats"] = feats
        for key, wdg in (("H", self.in_H), ("B", self.in_B)):
            val = sp.get(key)
            if val is None and isinstance(sp.get("caliper"), dict) and sp["caliper"].get(key):   # 1.x projects
                val = float(np.mean(sp["caliper"][key]))
            wdg.blockSignals(True); wdg.setValue(float(val) if val else 0.0); wdg.blockSignals(False)
        self.spec["H"] = self.in_H.value() if self.in_H.value() > 0 else np.nan
        self.spec["B"] = self.in_B.value() if self.in_B.value() > 0 else np.nan
        if sp.get("image") and os.path.exists(sp["image"]):
            self._pick_spec_image(sp["image"])
        ld = data.get("loads")
        if ld:
            self.loads["stages"] = [list(s) for s in ld["stages"]]
            for wdg, key in ((self.ld_W0, "W0"), (self.ld_B0, "B0"), (self.ld_lbf, "lbf"), (self.ld_tol, "tol")):
                wdg.blockSignals(True); wdg.setValue(float(ld[key])); wdg.blockSignals(False)
            self._fill_stages()
        if isinstance(data.get("alerts"), dict):
            self.alerts.update(data["alerts"]); self._alerts_to_widgets(); self._alerts_changed(recompute=False)
        if "guard" in data:
            self.chk_guard.setChecked(bool(data["guard"]))
        self.set_use_b(data.get("use_b", False))
        self._recompute_spec()
        sides = data.get("sides", [])
        for i, sd in enumerate(sides):
            if i == 1 and data.get("b_phone") is not None:
                (self.panels[1].rb_phone if data["b_phone"] else self.panels[1].rb_folder).setChecked(True)
            s = self.sides[i] = self._new_side()
            s["cal"] = [(tuple(a), tuple(b), mm) for a, b, mm in sd.get("cal", [])]
            ctd = sd.get("ct") or {}
            for k in ("tip", "mouth", "start_manual", "slot_u"):
                if ctd.get(k) is not None:
                    s["ct"][k] = tuple(ctd[k])
            for k in ("back", "side1", "side2"):
                if ctd.get(k):
                    s["ct"][k] = (tuple(ctd[k][0]), tuple(ctd[k][1]))
            s["ct"]["mcal"] = bool(ctd.get("mcal", False)); s["ct"]["rot"] = ctd.get("rot")
            s["roi_user"] = sd.get("roi_user", False); s["roi"] = tuple(sd["roi"]) if sd.get("roi") else None
            p = self.panels[i]
            p.chk_mcal.blockSignals(True); p.chk_mcal.setChecked(s["ct"]["mcal"]); p.chk_mcal.blockSignals(False)
            p.w_mcal.setVisible(s["ct"]["mcal"])
            p.sens.blockSignals(True); p.sens.setValue(int(sd.get("sens", 0))); p.sens.blockSignals(False)
            p.set_sens_label(p.sens.value())
            folder = sd.get("folder")
            if not folder or not os.path.isdir(folder):
                continue
            if i == 1 and self.b_phone:
                self.set_phone_folder(folder)
                self.sides[1].update(s | dict(folder=folder, files=cc.list_images(folder)))
                if sd.get("base_file") and os.path.exists(sd["base_file"]):
                    ct_keep = dict(s["ct"])
                    base.MainWindow._set_b_base(self, sd["base_file"])
                    self.sides[1]["ct"] = ct_keep
                    self._update_scale(1)
                continue
            s["folder"] = folder
            self.set_folder(i, folder, sd.get("base_file"), rot=s["ct"]["rot"])
            self._update_scale(i)
        if data.get("autocam"):
            self.panels[1].chk_autocam.setChecked(True)
        self.show_side(0, "base")
        self._recompute_loads()
        self.toast("הפרויקט נטען.", "ok")

    def show_help(self):
        QMessageBox.information(self, "CT Tracker", CT_HELP)


CT_HELP = """CT Tracker 2.2 – מדידת התקדמות סדק בדגמי C(T) לפי ASTM E647

1 · דגם ומידות: תמונה של הדגם מונח על השולחן + הגובה הכולל H והעובי B (קליבר). התוכנה מזהה את החורים,
    קצה החריץ והקצוות ומחשבת מהתמונה את כל השאר (W, a_n, רוחב, קוטר החורים, מרחקים, היסט החריץ).
    טבלה באנגלית מול השרטוט; שורה מחוץ לטולרנס – באדום.
2 · הגדרת מדידה:
    צד A – תיקיית המצלמה של המכונה. התמונה מסובבת תמיד באותו כיוון (החריץ מימין). התוכנה מוצאת את
    הקצה האחורי ושני הצדדים (קנה מידה מ-H), את קצה החריץ – הנקודה העמוקה של החריץ, בתוך רצועת
    החריץ בלבד – ואת נקודת ההתחלה על קו העומס, W מהקצה האחורי. רשת מ״מ על הדגם.
    בדיקה: a0 בתמונה מול a_n מתמונת הדגם. אפשר לתקן כל דבר ידנית, וכיול ידני (מרחק ידוע).
    צד B – הטלפון: "כוון מצלמה ונעל" (חשיפה + פוקוס על אזור החריץ, ננעל), או "מצלמה אוטומטית".
    "צלם תמונה" – התמונה נשארת כמו שהיא (בלי סיבוב); התוכנה מוצאת את הדגם, קצה החריץ והרשת.
3 · מדידה: "מעקב חי" – כל תמונה חדשה של המכונה מצלמת קודם את צד B בטלפון, ואז שתי התמונות נמדדות
    ונוספת שורה לטבלה. שומר צמיחה: הסדק גדל רק כשיש התכהות חדשה מעבר לקצה הקודם ("תקין · לא גדל").
    עמודת "a מהקצה האחורי" – אותו a, מחושב מהקצה האחורי שזוהה מחדש בכל תמונה (בדיקת נקודת ההתחלה).
4 · תוצאות וייצוא: "בחר נקודות לפי E647"; פערים באדום.  5 · התראות: |a₁ − a₂| ≤ 0.25B, זווית ±10°.
6 · כיול עומסים: Pmax/Pmin לדגם לפי B ו-W שנמדדו."""


def main():
    app = base.make_app(sys.argv)
    app.setApplicationName(APP_NAME)
    w = CTMainWindow()
    if w.settings.value("geometry") is None:
        w.showMaximized()
    else:
        w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

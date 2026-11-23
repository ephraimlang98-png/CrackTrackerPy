"""Crack Tracker - desktop app for measuring fatigue crack growth from image folders.
7.0: side B can come from a phone on the USB cable, triggered by every new side-A image (phone_b.py)."""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback

import numpy as np
from PySide6.QtCore import (QAbstractTableModel, QModelIndex, QPointF, QRectF, QSettings, Qt,
                            QThread, QTimer, Signal)
from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QIcon, QImage, QKeySequence, QPainter,
                           QPainterPath, QPen, QPixmap, QShortcut, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                               QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem,
                               QGraphicsRectItem, QGraphicsScene, QGraphicsView, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QScrollArea, QSlider, QSpinBox,
                               QSplitter, QStackedWidget, QStyle, QTableView, QTableWidget, QTableWidgetItem,
                               QTabWidget, QTextBrowser, QToolButton, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import crack_core as cc
import fc_link
import phone_b

APP_NAME = "Crack Tracker"
VERSION = "7.0"
SIDE_NAMES = ("A", "B")
C_MEAS = "#ff9600"
C_REF = "#ffeb00"
C_TIP = "#ff2d2d"
C_MAN = "#e040fb"
C_UNC = "#ffb020"
COL_A_PLOT, COL_B_PLOT = "#2563eb", "#ea580c"
C_EXP, C_N, C_FILE, C_A, C_B, C_AM, C_DA, C_DIFF, C_THA, C_THB, C_THA0, C_THB0, C_ST = range(13)
HEADERS = ["ייצוא", "#", "קובץ (A | B)", "a צד A", "a צד B", "a ממוצע", "Δa", "הפרש A−B", "θ חדש A", "θ חדש B",
           "θ כולל A", "θ כולל B", "סטטוס"]
B_COLS = (C_B, C_AM, C_DIFF, C_THB, C_THB0)
ALERT_DEFAULTS = dict(diff_on=True, diff_mode="da", diff_thr=0.5, ang_new_on=True, ang_new_thr=10.0,
                      ang_last_mm=0.0, ang_all_on=False, ang_all_thr=5.0, row_mode=False)
ALERT_BG, ALERT_ROW_BG, ALERT_FG = QColor("#fecaca"), QColor("#fee2e2"), QColor("#991b1b")
ZOOM0 = 3.0
PAGE_SETUP, PAGE_MEASURE, PAGE_RESULTS, PAGE_ALERTS = range(4)

PILL = {"ok": ("#dcfce7", "#166534"), "unc": ("#fef3c7", "#92400e"), "man": ("#f3e8ff", "#6b21a8"),
        "bad": ("#fee2e2", "#991b1b"), "none": ("#e2e8f0", "#334155")}
ROW_FG = {"bad": QColor("#b91c1c"), "unc": QColor("#b45309"), "man": QColor("#7e22ce")}
ROW_BG = {"bad": QColor("#fef2f2"), "unc": QColor("#fffbeb"), "man": QColor("#faf5ff")}

QSS = """
* { font-family: "Segoe UI"; font-size: 10pt; }
QMainWindow, QWidget#Page { background: #f1f5f9; }
QFrame#Header { background: #0f172a; }
QLabel#Logo { color: #ffffff; font-size: 13pt; font-weight: 600; }
QLabel#Ver { color: #64748b; font-size: 9pt; }
QPushButton#Nav { background: transparent; color: #94a3b8; border: none; border-radius: 0;
    padding: 12px 20px 9px 20px; font-size: 10.5pt; border-bottom: 3px solid transparent; }
QPushButton#Nav:hover { color: #e2e8f0; background: transparent; }
QPushButton#Nav:checked { color: #ffffff; border-bottom: 3px solid #3b82f6; font-weight: 600; }
QLabel#Chip { border-radius: 11px; padding: 3px 12px; color: #ffffff; background: #334155; font-weight: 600; }
QFrame#Card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; }
QFrame#Toolbar { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; }
QFrame#VSep { background: #e2e8f0; }
QPushButton { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 14px; color: #0f172a; }
QPushButton:hover { background: #f8fafc; border-color: #94a3b8; }
QPushButton:pressed { background: #e2e8f0; }
QPushButton:disabled { color: #94a3b8; background: #f8fafc; border-color: #e2e8f0; }
QPushButton[kind="primary"] { background: #2563eb; color: #ffffff; border: 1px solid #2563eb; font-weight: 600; }
QPushButton[kind="primary"]:hover { background: #1d4ed8; border-color: #1d4ed8; }
QPushButton[kind="primary"]:disabled { background: #bfdbfe; border-color: #bfdbfe; color: #eff6ff; }
QPushButton[kind="danger"] { color: #dc2626; border-color: #fecaca; }
QPushButton[kind="danger"]:hover { background: #fef2f2; border-color: #f87171; }
QPushButton[kind="danger"]:disabled { color: #fca5a5; border-color: #fee2e2; background: #ffffff; }
QPushButton[kind="nav"] { padding: 6px 10px; min-width: 22px; }
QToolButton { border: 1px solid transparent; border-radius: 6px; padding: 5px 9px; color: #334155; background: transparent; }
QToolButton:hover { background: #f1f5f9; border-color: #e2e8f0; }
QToolButton:checked { background: #dbeafe; border-color: #93c5fd; color: #1e3a8a; }
QToolButton:disabled { color: #cbd5e1; }
QLabel#Banner { background: #eff6ff; color: #1e3a8a; border: 1px solid #bfdbfe; border-radius: 6px; padding: 7px 12px; }
QLabel#Banner[kind="action"] { background: #fffbeb; color: #78350f; border: 1px solid #fcd34d; }
QLabel#Muted { color: #64748b; }
QLabel#SideTitle { font-size: 11pt; font-weight: 700; color: #ffffff; background: #0f172a; border-radius: 6px; padding: 2px 10px; }
QLabel#MetricA { font-size: 16pt; font-weight: 700; color: #0f172a; }
QLabel#MetricSub { font-size: 11pt; color: #334155; }
QLabel#Pill { border-radius: 9px; padding: 2px 10px; font-weight: 600; }
QLabel#PanelTitle { font-size: 10pt; font-weight: 700; color: #334155; }
QTableView { background: #ffffff; border: none; gridline-color: #f1f5f9; selection-background-color: #dbeafe;
    selection-color: #0f172a; alternate-background-color: #f8fafc; }
QHeaderView::section { background: #f8fafc; color: #475569; border: none; border-bottom: 1px solid #e2e8f0;
    padding: 5px 6px; font-weight: 600; }
QTabWidget::pane { border: none; background: #ffffff; }
QTabBar::tab { background: transparent; padding: 9px 18px; color: #64748b; border: none; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #1d4ed8; border-bottom: 2px solid #2563eb; font-weight: 600; }
QTabBar::tab:disabled { color: #cbd5e1; }
QGroupBox { border: 1px solid #e2e8f0; border-radius: 8px; margin-top: 16px; padding: 12px 10px 10px 10px;
    background: #ffffff; font-weight: 600; color: #0f172a; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top right; padding: 0 8px; right: 8px; }
QGroupBox[done="true"] { border: 1px solid #86efac; }
QGroupBox[done="true"]::title { color: #15803d; }
QDoubleSpinBox, QSpinBox { border: 1px solid #cbd5e1; border-radius: 6px; padding: 4px 6px; background: #ffffff; }
QDoubleSpinBox:focus, QSpinBox:focus { border-color: #3b82f6; }
QProgressBar { border: 1px solid #e2e8f0; border-radius: 6px; background: #f1f5f9; text-align: center; color: #0f172a; max-height: 16px; }
QProgressBar::chunk { background: #3b82f6; border-radius: 5px; }
QStatusBar { background: #ffffff; border-top: 1px solid #e2e8f0; color: #475569; }
QStatusBar QLabel { color: #475569; padding: 0 8px; }
QSplitter::handle { background: transparent; }
QSplitter::handle:vertical { height: 8px; }
QSplitter::handle:horizontal { width: 8px; }
QFrame#Stat { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; }
QLabel#StatVal { font-size: 17pt; font-weight: 700; color: #0f172a; }
QLabel#StatLbl { color: #64748b; font-size: 9pt; }
QPlainTextEdit#Log { background: #0f172a; color: #cbd5e1; border: none; border-radius: 6px; font-family: Consolas; font-size: 9pt; }
QToolTip { background: #0f172a; color: #ffffff; border: none; padding: 6px; }
QLabel#Toast { border-radius: 8px; padding: 10px 16px; font-weight: 600; }
QGraphicsView { border: none; }
QCheckBox { spacing: 6px; color: #0f172a; }
QMenuBar { background: #0f172a; color: #cbd5e1; }
QMenuBar::item { padding: 4px 10px; }
QMenuBar::item:selected { background: #1e293b; color: #ffffff; }
QMenu { background: #ffffff; border: 1px solid #e2e8f0; }
QMenu::item { padding: 6px 24px; }
QMenu::item:selected { background: #eff6ff; color: #1e3a8a; }
"""


def resource_path(name: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def _fmt(x, nd=3):
    return "" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _num(x):
    return "" if x is None or not np.isfinite(x) else f"{x:.4f}"


def _bad(st):
    return not (st.startswith("ok") or st == "manual" or st.startswith("uncertain"))


def _kind1(st):
    if not st:
        return "none"
    if st == "manual":
        return "man"
    if st.startswith("uncertain"):
        return "unc"
    if _bad(st):
        return "bad"
    return "ok"


def _kind(st):
    """Row kind from a combined 'A | B' status: worst first."""
    ks = [_kind1(x) for x in st.split(" | ")]
    for k in ("bad", "unc", "man", "ok"):
        if k in ks:
            return k
    return "none"


def _needs_check(st):
    return _kind(st) in ("bad", "unc")


def status_he(st):
    if not st:
        return ""
    if st == "manual":
        return "תוקן ידנית"
    if st == "no phone image":
        return "אין תמונה מהטלפון"
    if st.startswith("uncertain"):
        return "לא בטוח " + st[len("uncertain"):].strip()
    if st.startswith("ok"):
        return "תקין" + ("" if st == "ok" else " " + st[2:].strip())
    return "נכשל: " + st


def _ltr(t):
    """Keeps a file name / path left-to-right inside Hebrew text."""
    return "‎" + str(t) + "‎"


def _fmt_deg(x):
    return "" if x is None or not np.isfinite(x) else f"{x:+.1f}°"


def _vsep():
    f = QFrame(); f.setObjectName("VSep"); f.setFixedWidth(1); f.setFixedHeight(26)
    return f


def _icon(name):
    return QApplication.style().standardIcon(getattr(QStyle, name))


def _btn(text, kind=None, tip=None, slot=None, icon=None):
    b = QPushButton(text)
    if icon:
        b.setIcon(_icon(icon))
        if text:
            b.setText("  " + text)
    if kind:
        b.setProperty("kind", kind)
    if tip:
        b.setToolTip(tip)
    if slot:
        b.clicked.connect(slot)
    b.setCursor(Qt.PointingHandCursor)
    return b


def _repolish(w):
    w.style().unpolish(w); w.style().polish(w)


# =============================================================================
# Image view with zoom / pan / click picking, overlays and mm rulers
# =============================================================================
class ImageView(QGraphicsView):
    clicked = Signal(float, float)

    def __init__(self, smooth=False):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor(24, 26, 31))
        self.setFrameShape(QFrame.NoFrame)
        self.setRenderHint(QPainter.SmoothPixmapTransform, smooth)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setLayoutDirection(Qt.LeftToRight)
        self.pix = QGraphicsPixmapItem()
        self.pix.setTransformationMode(Qt.SmoothTransformation if smooth else Qt.FastTransformation)
        self.scene().addItem(self.pix)
        self.overlays: list[QGraphicsItem] = []
        self.img_shape = None
        self.pick_mode = False
        self._press = None
        self.mm_per_px = None
        self.auto_fit = True
        self.caption = ""
        self.empty_text = ""

    def set_image(self, img: np.ndarray):
        a = np.ascontiguousarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
        h, w = a.shape
        q = QImage(a.data, w, h, w, QImage.Format_Grayscale8).copy()
        self.pix.setPixmap(QPixmap.fromImage(q))
        new = self.img_shape != (h, w)
        self.img_shape = (h, w)
        self.empty_text = ""
        self.scene().setSceneRect(QRectF(-w, -h, 3 * w, 3 * h))
        if new and self.auto_fit:
            self.fit_all()
        self.viewport().update()

    def set_qimage(self, q: QImage):
        """Live phone preview (colour, no overlays)."""
        self.pix.setPixmap(QPixmap.fromImage(q))
        new = self.img_shape != (q.height(), q.width())
        self.img_shape = (q.height(), q.width())
        self.empty_text = ""
        self.scene().setSceneRect(QRectF(-q.width(), -q.height(), 3 * q.width(), 3 * q.height()))
        if new and self.auto_fit:
            self.fit_all()
        self.viewport().update()

    def clear_image(self, text=""):
        self.pix.setPixmap(QPixmap())
        self.clear_overlays()
        self.img_shape = None
        self.empty_text = text
        self.viewport().update()

    def fit_all(self):
        if self.img_shape:
            self.fitInView(self.pix, Qt.KeepAspectRatio)

    def zoom_to(self, x0, y0, x1, y1):
        self.fitInView(QRectF(x0, y0, x1 - x0, y1 - y0), Qt.KeepAspectRatio)

    def magnify_at(self, x, y, mag):
        self.setTransform(QTransform.fromScale(mag, mag))
        self.centerOn(x + 0.5, y + 0.5)

    def set_ruler(self, mm_per_px):
        self.mm_per_px = mm_per_px if mm_per_px and np.isfinite(mm_per_px) else None
        self.viewport().update()

    def wheelEvent(self, e):
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(f, f)

    def mouseDoubleClickEvent(self, e):
        if not self.pick_mode and self.auto_fit:
            self.fit_all()
        super().mouseDoubleClickEvent(e)

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self.viewport().update()

    def set_pick_mode(self, on: bool):
        self.pick_mode = on
        self.viewport().setCursor(Qt.CrossCursor if on else Qt.OpenHandCursor)
        self.viewport().update()

    def mousePressEvent(self, e):
        self._press = e.position()
        super().mousePressEvent(e)
        if self.pick_mode:
            self.viewport().setCursor(Qt.CrossCursor)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if self.pick_mode:
            self.viewport().setCursor(Qt.CrossCursor)
        if (self.pick_mode and e.button() == Qt.LeftButton and self._press is not None
                and (e.position() - self._press).manhattanLength() < 5):
            p = self.mapToScene(e.position().toPoint())
            self.clicked.emit(p.x() - 0.5, p.y() - 0.5)
        self._press = None

    def drawForeground(self, painter, rect):
        painter.save()
        painter.resetTransform()
        vw, vh = self.viewport().width(), self.viewport().height()
        if self.img_shape is None and self.empty_text:
            painter.setPen(QColor(148, 163, 184))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(QRectF(20, 0, vw - 40, vh), Qt.AlignCenter | Qt.TextWordWrap, self.empty_text)
        if self.pick_mode:
            painter.setPen(QPen(QColor(59, 130, 246), 3)); painter.setBrush(Qt.NoBrush)
            painter.drawRect(1, 1, vw - 3, vh - 3)
        if self.caption and self.img_shape is not None:
            painter.setFont(QFont("Segoe UI", 8, QFont.DemiBold))
            fm = painter.fontMetrics(); w = fm.horizontalAdvance(self.caption) + 14
            painter.fillRect(vw - w - 6, vh - 24, w, 18, QColor(15, 23, 42, 170))
            painter.setPen(QColor(226, 232, 240))
            painter.drawText(QRectF(vw - w - 6, vh - 24, w, 18), Qt.AlignCenter, self.caption)
        if self.mm_per_px and self.img_shape:
            tl = self.mapToScene(0, 0); br = self.mapToScene(vw, vh)
            ppm = vw / max(1e-9, (br.x() - tl.x()) * self.mm_per_px)
            minor = cc.nice_step(ppm, 7); major = cc.nice_step(ppm, 70)
            band = 16
            painter.fillRect(0, 0, vw, band, QColor(0, 0, 0, 90))
            painter.fillRect(0, 0, band, vh, QColor(0, 0, 0, 90))
            pen = QPen(QColor(255, 235, 120, 190)); pen.setWidth(1); painter.setPen(pen)
            painter.setFont(QFont("Segoe UI", 7))
            for horiz in (True, False):
                lo_mm = (tl.x() if horiz else tl.y()) * self.mm_per_px
                hi_mm = (br.x() if horiz else br.y()) * self.mm_per_px
                a = np.ceil(lo_mm / minor) * minor
                n = 0
                while a <= hi_mm and n < 2000:
                    p = (a - lo_mm) * ppm
                    is_major = abs(a / major - round(a / major)) < 1e-6
                    if horiz:
                        painter.drawLine(int(p), 0, int(p), 12 if is_major else 5)
                        if is_major and p > band + 2:
                            painter.drawText(int(p) + 2, 12, cc._lbl(a))
                    else:
                        painter.drawLine(0, int(p), 12 if is_major else 5, int(p))
                        if is_major and p > band + 2:
                            painter.save(); painter.translate(12, int(p) - 2); painter.rotate(-90)
                            painter.drawText(0, 0, cc._lbl(a)); painter.restore()
                    a += minor; n += 1
            painter.drawText(vw - 34, 12, "mm")
        painter.restore()

    def clear_overlays(self):
        for it in self.overlays:
            self.scene().removeItem(it)
        self.overlays.clear()

    def _pen(self, color, width=2, style=Qt.SolidLine, alpha=255):
        c = QColor(color); c.setAlpha(alpha)
        p = QPen(c, width, style)
        p.setCosmetic(True); p.setCapStyle(Qt.RoundCap); p.setJoinStyle(Qt.RoundJoin)
        return p

    def _add(self, it, z):
        it.setZValue(z)
        self.scene().addItem(it)
        self.overlays.append(it)

    def add_marker(self, x, y, shape="dot", color="red", size=9, width=2.5):
        if not (np.isfinite(x) and np.isfinite(y)):
            return
        path = QPainterPath()
        if shape == "+":
            path.moveTo(-size, 0); path.lineTo(size, 0); path.moveTo(0, -size); path.lineTo(0, size)
        elif shape == "ticks":
            g, L = size, size + 14
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                path.moveTo(dx * g, dy * g); path.lineTo(dx * L, dy * L)
        else:
            path.addEllipse(QPointF(0, 0), size, size)
        it = QGraphicsPathItem(path)
        it.setPen(self._pen(color, width))
        if shape == "dot":
            it.setBrush(QBrush(QColor(color)))
        it.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        it.setPos(x + 0.5, y + 0.5)
        self._add(it, 10)

    def add_polyline(self, pts, color, width=2.0, style=Qt.SolidLine, z=6, alpha=255):
        if pts is None or len(pts) < 2:
            return
        path = QPainterPath(QPointF(pts[0][0] + 0.5, pts[0][1] + 0.5))
        for p in pts[1:]:
            path.lineTo(p[0] + 0.5, p[1] + 0.5)
        it = QGraphicsPathItem(path)
        it.setPen(self._pen(color, width, style, alpha))
        self._add(it, z)

    def add_rect(self, r, color="yellow"):
        it = QGraphicsRectItem(r[0], r[1], r[2], r[3])
        it.setPen(self._pen(color, 1.2, Qt.DashLine))
        it.setBrush(QBrush(Qt.NoBrush))
        self._add(it, 5)


class Toast(QLabel):
    """Small non-blocking notification at the bottom of the window."""
    COLORS = {"info": "#0f172a", "ok": "#166534", "warn": "#92400e", "error": "#991b1b"}

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("Toast")
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignCenter)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.timer = QTimer(self); self.timer.setSingleShot(True); self.timer.timeout.connect(self.hide)
        self.hide()

    def place(self):
        p = self.parentWidget()
        self.move((p.width() - self.width()) // 2, p.height() - self.height() - 56)

    def pop(self, text, kind="info", ms=3500):
        self.setStyleSheet(f"background:{self.COLORS.get(kind, '#0f172a')}; color:#ffffff;")
        self.setText(text)
        self.setMaximumWidth(640); self.setMinimumWidth(240)
        self.adjustSize()
        self.place()
        self.show(); self.raise_()
        self.timer.start(ms)


# =============================================================================
# Results table model (shared by the compact and the full table; cheap with thousands of rows)
# =============================================================================
class ResultsModel(QAbstractTableModel):
    def __init__(self, main):
        super().__init__()
        self.m = main

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.m.R["k"])

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(HEADERS)

    def headerData(self, s, orient, role=Qt.DisplayRole):
        if orient == Qt.Horizontal and role == Qt.DisplayRole:
            if s == C_DIFF:
                return "הפרש a A−B" if self.m.alerts["diff_mode"] == "a" else "הפרש Δa A−B"
            return HEADERS[s]
        if orient == Qt.Horizontal and role == Qt.ToolTipRole and s in (C_THA, C_THB, C_THA0, C_THB0):
            return "זווית ביחס לכיוון ההתחלתי של הסדק; + = נגד כיוון השעון בתמונה"
        return None

    def flags(self, idx):
        f = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if idx.column() == C_EXP:
            f |= Qt.ItemIsUserCheckable
        return f

    def data(self, idx, role=Qt.DisplayRole):
        r, c = idx.row(), idx.column()
        R = self.m.R
        if r >= len(R["k"]):
            return None
        if role == Qt.DisplayRole:
            if c == C_N:
                return str(R["k"][r])
            if c == C_FILE:
                return R["file"][r] if not self.m.use_b else f"{R['file'][r]}  |  {R['fileB'][r]}"
            if c == C_A:
                return _ltr(_fmt(R["aA"][r]))
            if c == C_B:
                return _ltr(_fmt(R["aB"][r]))
            if c == C_AM:
                return _ltr(_fmt(R["a"][r]))
            if c == C_DA:
                return _ltr(_fmt(R["da"][r]))
            if c == C_DIFF:
                v = self.m._diff(r)
                return "" if not np.isfinite(v) else _ltr(f"{v:+.3f}")
            if c in (C_THA, C_THB, C_THA0, C_THB0):
                side = 0 if c in (C_THA, C_THA0) else 1
                return _ltr(_fmt_deg(R["ang"][r][side][0 if c in (C_THA, C_THB) else 1]))
            if c == C_ST:
                return "  |  ".join(status_he(x) for x in R["status"][r].split(" | "))
            return None
        if role == Qt.CheckStateRole and c == C_EXP:
            return Qt.CheckState.Checked if R["export"][r] else Qt.CheckState.Unchecked
        if role == Qt.TextAlignmentRole and c not in (C_FILE, C_ST):
            return int(Qt.AlignCenter)
        if role in (Qt.ForegroundRole, Qt.BackgroundRole, Qt.FontRole):
            fl = self.m.alert_cells(r)
            if fl and (c in fl or (self.m.alerts["row_mode"] and c != C_EXP)):
                if role == Qt.BackgroundRole:
                    return QBrush(ALERT_BG if c in fl else ALERT_ROW_BG)
                if role == Qt.ForegroundRole:
                    return QBrush(ALERT_FG)
                if c in fl:
                    f = QFont(); f.setBold(True); return f
                return None
            if role == Qt.FontRole:
                return None
            k = _kind(R["status"][r])
            table = ROW_FG if role == Qt.ForegroundRole else ROW_BG
            return QBrush(table[k]) if k in table else None
        if role == Qt.ToolTipRole and c == C_ST:
            return R["status"][r]
        if role == Qt.ToolTipRole and c in self.m.alert_cells(r):
            return "חורג מהסף שנקבע במסך \"התראות\""
        return None

    def setData(self, idx, value, role=Qt.EditRole):
        if role == Qt.CheckStateRole and idx.column() == C_EXP and idx.row() < len(self.m.R["export"]):
            v = value.value if hasattr(value, "value") else value
            self.m.R["export"][idx.row()] = int(v) == Qt.CheckState.Checked.value
            self.dataChanged.emit(idx, idx, [Qt.CheckStateRole])
            return True
        return False

    def row_changed(self, r):
        self.dataChanged.emit(self.index(r, 0), self.index(r, len(HEADERS) - 1))

    def headers_changed(self):
        self.headerDataChanged.emit(Qt.Horizontal, 0, len(HEADERS) - 1)

    def all_changed(self):
        n = self.rowCount()
        if n:
            self.dataChanged.emit(self.index(0, 0), self.index(n - 1, len(HEADERS) - 1))


# =============================================================================
# Image + tip zoom pair; measuring-screen side card; setup-screen view
# =============================================================================
class ImagePair(QWidget):
    def __init__(self):
        super().__init__()
        self.view = ImageView()
        self.zoom = ImageView(smooth=True)
        self.zoom.auto_fit = False
        self.zoom.caption = "הגדלת הקצה · גלגלת = זום"
        self.zoom.setToolTip("הגדלה של אזור קצה הסדק: גלגלת = זום, גרירה = הזזה")
        self.view.setToolTip("גלגלת = זום, גרירה = הזזה, לחיצה כפולה = כל התמונה")
        self.zoom.setMinimumWidth(220)
        self._zoom_mag = ZOOM0
        self.fitted = False
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        sp = QSplitter(Qt.Horizontal); sp.setLayoutDirection(Qt.LeftToRight)
        sp.addWidget(self.view); sp.addWidget(self.zoom)
        sp.setStretchFactor(0, 3); sp.setStretchFactor(1, 2)
        lay.addWidget(sp)

    def show_zoom(self, img, tip_xy, mmpp, draw):
        if img is None or tip_xy is None or not np.all(np.isfinite(tip_xy)):
            self.zoom.clear_image("אין קצה להצגה"); return
        if self.zoom.img_shape is not None:
            self._zoom_mag = max(0.2, self.zoom.transform().m11())
        self.zoom.set_image(img)
        self.zoom.set_ruler(mmpp)
        self.zoom.clear_overlays()
        draw(self.zoom)
        self.zoom.magnify_at(tip_xy[0], tip_xy[1], self._zoom_mag)

    def set_pick(self, on):
        self.view.set_pick_mode(on); self.zoom.set_pick_mode(on)


class SideView(QFrame):
    def __init__(self, idx, main):
        super().__init__()
        self.setObjectName("Card")
        self.idx = idx
        lay = QVBoxLayout(self); lay.setContentsMargins(10, 8, 10, 10); lay.setSpacing(6)
        head = QHBoxLayout(); head.setSpacing(12)
        self.title = QLabel(f"צד {SIDE_NAMES[idx]}"); self.title.setObjectName("SideTitle")
        self.a_lbl = QLabel("a = —"); self.a_lbl.setObjectName("MetricA")
        self.da_lbl = QLabel("Δa = —"); self.da_lbl.setObjectName("MetricSub")
        self.pill = QLabel(""); self.pill.setObjectName("Pill"); self.pill.hide()
        self.file_lbl = QLabel(""); self.file_lbl.setObjectName("Muted")
        self.btn_fix = QToolButton(); self.btn_fix.setText(" " + "תקן קצה"); self.btn_fix.setIcon(_icon("SP_DialogApplyButton"))
        self.btn_fix.setToolTip("לחץ, ואז לחץ על הקצה הנכון של הסדק (מומלץ בהגדלה).\n"
                                "התצוגה נעצרת על התמונה הזו – המדידה ממשיכה ברקע."
                                + ("  [T]" if idx == 0 else ""))
        self.btn_fix.setCheckable(True)
        self.btn_fix.clicked.connect(lambda: main.start_fix(idx))
        self.btn_undo = QToolButton(); self.btn_undo.setText(" " + "בטל תיקון"); self.btn_undo.setIcon(_icon("SP_DialogResetButton"))
        self.btn_undo.clicked.connect(lambda: main.undo_fix(idx))
        self.btn_all = QToolButton(); self.btn_all.setText(" " + "החל תיקון על כולם"); self.btn_all.setIcon(_icon("SP_BrowserReload"))
        self.btn_all.setToolTip("מתאים את סף הזיהוי כך שהקצה האוטומטי יגיע לנקודה שתיקנת, ומודד מחדש ברקע\n"
                                "את כל התמונות של הצד הזה (תמונות שתוקנו ידנית נשארות).")
        self.btn_all.clicked.connect(lambda: main.apply_fix_to_all(idx))
        self.ang_lbl = QLabel(""); self.ang_lbl.setObjectName("Pill"); self.ang_lbl.hide()
        self.ang_lbl.setToolTip("זווית החלק החדש של הסדק ביחס לכיוון ההתחלתי (+ = נגד כיוון השעון בתמונה)")
        for w in (self.title, self.a_lbl, self.da_lbl, self.pill, self.ang_lbl, self.file_lbl):
            head.addWidget(w)
        head.addStretch(1)
        sl = QLabel("רגישות"); sl.setObjectName("Muted")
        self.sens = QSpinBox(); self.sens.setRange(-5, 5); self.sens.setValue(0); self.sens.setFixedWidth(64)
        self.sens.setAlignment(Qt.AlignCenter)
        self.sens.setToolTip("שינוי רגישות בזמן ריצה – חל רק על התמונות שעוד לא נמדדו.\n"
                             "+ = גם קווים דקים יותר נחשבים סדק, − = רק קווים ברורים.\n"
                             "פעיל רק בזמן מדידה (לפני הריצה – במסך ההגדרה).")
        self.sens.setEnabled(False)
        self.sens.valueChanged.connect(lambda v: main.change_run_sens(idx, v))
        head.addWidget(sl); head.addWidget(self.sens); head.addSpacing(6)
        for w in (self.btn_fix, self.btn_undo, self.btn_all):
            w.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            w.setCursor(Qt.PointingHandCursor); head.addWidget(w)
        lay.addLayout(head)
        self.pair = ImagePair()
        lay.addWidget(self.pair, 1)
        self.set_fix_enabled(False)

    @property
    def view(self):
        return self.pair.view

    @property
    def zoom(self):
        return self.pair.zoom

    @property
    def fitted(self):
        return self.pair.fitted

    @fitted.setter
    def fitted(self, v):
        self.pair.fitted = v

    def set_fix_enabled(self, on, manual=False):
        self.btn_fix.setEnabled(on)
        self.btn_undo.setEnabled(on and manual)
        self.btn_all.setEnabled(on and manual)

    def set_metrics(self, a, da, status, label):
        self.a_lbl.setText(f"{_ltr('a = ' + _fmt(a))} מ״מ" if np.isfinite(a) else "a = —")
        self.da_lbl.setText(_ltr(f"Δa = {_fmt(da)}") if np.isfinite(da) else "Δa = —")
        if status:
            bg, fg = PILL[_kind1(status)]
            self.pill.setStyleSheet(f"background:{bg}; color:{fg};")
            self.pill.setText(status_he(status)); self.pill.show()
        else:
            self.pill.hide()
        self.file_lbl.setText(label)

    def set_angle(self, ang, alert):
        if ang is None or not np.isfinite(ang):
            self.ang_lbl.hide(); return
        bg, fg = ("#fecaca", "#991b1b") if alert else PILL["none"]
        self.ang_lbl.setStyleSheet(f"background:{bg}; color:{fg};")
        self.ang_lbl.setText(f"θ חדש  {_ltr(_fmt_deg(ang))}"); self.ang_lbl.show()


class AngleSketch(QWidget):
    """Small explanatory drawing of the two crack angles."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 190); self.setMaximumHeight(260)
        self.setLayoutDirection(Qt.LeftToRight)

    def paintEvent(self, e):
        qp = QPainter(self); qp.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        qp.fillRect(0, 0, w, h, QColor("#f8fafc"))
        y = h * 0.62; x0, xb = w * 0.08, w * 0.50
        tip = QPointF(w * 0.90, h * 0.22)
        qp.setPen(QPen(QColor("#38bdf8"), 1.4, Qt.DashLine)); qp.drawLine(QPointF(x0, y), QPointF(w * 0.95, y))
        qp.setPen(QPen(QColor("#334155"), 4)); qp.drawLine(QPointF(x0, y), QPointF(xb, y))
        qp.setPen(QPen(QColor("#ea580c"), 3)); qp.drawLine(QPointF(xb, y), tip)
        qp.setPen(QPen(QColor("#2563eb"), 1.4, Qt.DotLine)); qp.drawLine(QPointF(x0, y), tip)
        qp.setPen(QPen(QColor("#ca8a04"), 2)); qp.drawLine(QPointF(x0, y - 7), QPointF(x0, y + 7))
        qp.drawLine(QPointF(x0 - 7, y), QPointF(x0 + 7, y))
        qp.setPen(QPen(QColor("#dc2626"), 2)); qp.setBrush(Qt.NoBrush); qp.drawEllipse(tip, 6, 6)
        f = QFont("Segoe UI", 9); qp.setFont(f)
        qp.setPen(QColor("#ea580c")); qp.drawArc(QRectF(xb - 60, y - 60, 120, 120), 0, 30 * 16)
        qp.drawText(QPointF(xb + 64, y - 10), "θ חדש")
        qp.setPen(QColor("#2563eb")); qp.drawArc(QRectF(x0 - 110, y - 110, 220, 220), 0, 14 * 16)
        qp.drawText(QPointF(x0 + 112, y - 14), "θ כולל")
        qp.setPen(QColor("#334155")); qp.drawText(QPointF(x0 - 4, y + 24), "נקודת התחלה")
        qp.drawText(QPointF(xb - 34, y + 24), "קצה בבסיס")
        qp.setPen(QColor("#0284c7")); qp.drawText(QPointF(w * 0.68, y + 24), "כיוון התחלתי")
        qp.setPen(QColor("#ea580c")); qp.drawText(QPointF(w * 0.74, h * 0.50), "החלק החדש")
        qp.end()


class SetupView(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("Card")
        lay = QVBoxLayout(self); lay.setContentsMargins(10, 8, 10, 10); lay.setSpacing(6)
        head = QHBoxLayout(); head.setSpacing(12)
        self.title = QLabel("תמונת בסיס"); self.title.setObjectName("SideTitle")
        self.info = QLabel(""); self.info.setObjectName("Muted")
        head.addWidget(self.title); head.addWidget(self.info); head.addStretch(1)
        lay.addLayout(head)
        self.pair = ImagePair()
        lay.addWidget(self.pair, 1)
        self.view.clear_image("בחר תיקיית תמונות כדי להתחיל")
        self.zoom.clear_image("")

    @property
    def view(self):
        return self.pair.view

    @property
    def zoom(self):
        return self.pair.zoom


# =============================================================================
# Background workers (the window never waits for image work)
# =============================================================================
class Worker(QThread):
    """Measures images (batch or live). Never stops on a bad image."""
    result = Signal(dict)
    message = Signal(str)

    def __init__(self, sides: list[dict], live: bool, start_index: int):
        super().__init__()
        self.sides = sides
        self.live = live
        self.k = start_index
        self._stop = False
        self._last_img = 0.0
        self.prev = {s["index"]: None for s in sides}
        self.phone = any(s.get("b_state") for s in sides)      # side B from the phone, paired by name
        self._dirs = {}                                         # folder -> (mtime, t, image list)

    def stop(self):
        self._stop = True

    def _list(self, folder):
        """list_images, re-read only when the folder changed (listing thousands of images takes ~1 s);
        every 5 s anyway."""
        try:
            dm = os.stat(folder).st_mtime_ns
        except OSError:
            dm = None
        c = self._dirs.get(folder)
        if c is None or c[0] != dm or time.time() - c[1] > 5.0:
            c = self._dirs[folder] = (dm, time.time(), cc.list_images(folder))
        return list(c[2])

    def _available(self):
        lists = []
        for s in self.sides:
            if s.get("b_state"):
                lists.append([]); continue
            files = self._list(s["folder"])
            if self.live and not self.phone:          # phone: the trigger already knows when A is complete
                now = time.time()
                while files:
                    try:
                        if now - os.path.getmtime(files[-1]) > 1.0:
                            break
                    except OSError:
                        pass
                    files = files[:-1]
            lists.append(files)
        return self._pair(lists)

    def _pair(self, lists):
        """Side B from the phone: pairs by name with side A (a missed capture never shifts the pairs).
        In live mode stops at the first A image whose phone capture is still on its way."""
        for j, s in enumerate(self.sides):
            st = s.get("b_state")
            if not st:
                continue
            out = []
            now = time.time()
            for i, f in enumerate(lists[0]):
                if self.live and i >= self.k:            # only images not measured yet are checked
                    state = st(f)
                    if state == "pending":
                        break
                    try:                                 # never triggered (trigger off): wait until written
                        if state == "none" and now - os.path.getmtime(f) <= 1.0:
                            break
                    except OSError:
                        break
                out.append(os.path.join(s["folder"], phone_b.b_name(f)))
            lists[0], lists[j] = lists[0][:len(out)], out
        return lists

    def run(self):
        try:
            if not self.live:
                lists = self._pair([[] if s.get("b_state") else s["files"] for s in self.sides])
                n = min(len(l) for l in lists)
                if len(lists) == 2 and len(lists[0]) != len(lists[1]):
                    self.message.emit(f"אזהרה: בצד A יש {len(lists[0])} תמונות ובצד B {len(lists[1])}. "
                                      f"מתאים לפי הסדר, רק {n} הראשונות.")
                for k in range(self.k, n):
                    if self._stop:
                        break
                    self._do(k, lists)
                return
            while not self._stop:
                try:
                    lists = self._available()
                except Exception as ex:
                    self.message.emit(f"שגיאה בקריאת התיקייה: {ex}"); lists = None
                if lists:
                    n = min(len(l) for l in lists)
                    while self.k < n and not self._stop:
                        self._do(self.k, lists)
                        self.k += 1
                for _ in range(3 if self.phone else 10):
                    if self._stop:
                        break
                    self.msleep(100)
        except Exception as ex:
            self.message.emit(f"שגיאה: {ex}")

    def _do(self, k, lists):
        out = dict(k=k, file=["", ""], img=[None, None], res=[None, None], sens=[None, None])
        want_img = time.time() - self._last_img > 0.3
        for j, s in enumerate(self.sides):
            idx = s["index"]
            out["sens"][idx] = "fit" if s.get("fitted") else s.get("sens_now", 0)
            try:
                use_prior = s["P"].get("use_prior", True)
                if s.get("b_state") and not os.path.exists(lists[j][k]):
                    r = cc.FrameResult(lists[j][k], False, "no phone image", np.nan, np.nan)
                else:
                    r = cc.measure_frame(lists[j][k], s, keep_image=want_img,
                                         prev_reach=self.prev[idx] if use_prior else None)
                if r.ok and np.isfinite(r.reach):
                    self.prev[idx] = r.reach
            except Exception as ex:
                r = cc.FrameResult(lists[j][k], False, f"unreadable image ({type(ex).__name__})",
                                   np.nan, np.nan)
            out["img"][idx] = r.img
            r.img = None
            out["res"][idx] = r
            out["file"][idx] = os.path.basename(lists[j][k])
        if want_img:
            self._last_img = time.time()
        self.result.emit(out)


class RemeasureWorker(QThread):
    """'Apply correction to all': finds the threshold that puts the automatic tip at the manual
    correction, then re-measures all existing rows of one side with it."""
    thr_found = Signal(object, object)          # (thr or None, a reached)
    row_done = Signal(int, int, object)
    progress = Signal(int, int)

    def __init__(self, side_idx, prep, fix, file_of, n_rows, measuring):
        super().__init__()
        # fix: args of threshold_for_tip; file_of(row) -> file or None; n_rows() -> rows so far;
        # measuring() -> True while the main measurement is still running
        self.side_idx, self.prep, self.fix = side_idx, prep, fix
        self.file_of, self.n_rows, self.measuring = file_of, n_rows, measuring
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            thr, a_got = cc.threshold_for_tip(*self.fix)
        except Exception:
            thr, a_got = None, None
        self.thr_found.emit(thr, a_got)
        if thr is None:
            return
        self.prep["thr_eff"] = thr          # new images of the running measurement use it too
        self.prep["fitted"] = True
        prev = None
        use_prior = self.prep["P"].get("use_prior", True)
        # rows measured until now (+2 that may be in flight in the main measurement with the old threshold)
        total = self.n_rows() + (2 if self.measuring() else 0)
        row = 0
        while row < total and not self._stop:
            f = self.file_of(row)
            if f is None:
                if row >= self.n_rows() and not self.measuring():
                    break
                if row >= self.n_rows():
                    self.msleep(50); continue        # row not measured yet - wait for it
                row += 1; continue                   # this side has no result in that row
            try:
                r = cc.measure_frame(f, self.prep, prev_reach=prev if use_prior else None)
                if r.ok and np.isfinite(r.reach):
                    prev = r.reach
            except Exception as ex:
                r = cc.FrameResult(f, False, f"unreadable image ({type(ex).__name__})", np.nan, np.nan)
            self.row_done.emit(row, self.side_idx, r)
            if row % 5 == 0:
                self.progress.emit(row + 1, total)
            row += 1
        self.progress.emit(total, total)


class ExportWorker(QThread):
    """Renders and saves annotated images in the background (no freezing, can be stopped)."""
    progress = Signal(int, int)
    done = Signal(int, str)

    def __init__(self, jobs, out_dir):
        super().__init__()
        self.jobs, self.out_dir = jobs, out_dir     # jobs: [(res, prep, label, filename)]
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        n = 0; err = ""
        for i, (res, prep, label, fname) in enumerate(self.jobs):
            if self._stop:
                break
            try:
                img = cc.load_frame(res.file, prep)
                text = f"{label}   a = {_fmt(res.a)} mm   da = {_fmt(res.da)} mm   {res.status}"
                im = cc.render_annotated(img, prep, res, text, crop=None)
                im.save(os.path.join(self.out_dir, fname))
                n += 1
            except Exception as ex:
                err = f"{os.path.basename(res.file)}: {ex}"
            if i % 2 == 0:
                self.progress.emit(i + 1, len(self.jobs))
        self.progress.emit(len(self.jobs), len(self.jobs))
        self.done.emit(n, err)


# =============================================================================
# Side setup panel
# =============================================================================
class SidePanel(QWidget):
    def __init__(self, idx: int, main: "MainWindow"):
        super().__init__()
        self.idx, self.main = idx, main
        lay = QVBoxLayout(self)
        lay.setSpacing(4); lay.setContentsMargins(12, 10, 12, 10)
        if idx == 1:
            self.use = QCheckBox("להשתמש בצד B (מצלמה שנייה / הצד השני של הדגימה)")
            self.use.toggled.connect(main.set_use_b)
            lay.addWidget(self.use)
        self.g1 = QGroupBox("1 · תיקיית תמונות")
        l1 = QVBoxLayout(self.g1)
        self.btn_folder = _btn("בחר תיקייה...", slot=lambda: main.pick_folder(idx), icon="SP_DirOpenIcon")
        self.lbl_folder = QLabel("לא נבחרה תיקייה"); self.lbl_folder.setObjectName("Muted")
        self.lbl_folder.setWordWrap(True)
        if idx == 1:
            row = QHBoxLayout()
            row.addWidget(QLabel("מקור:"))
            self.rb_folder = QRadioButton("תיקייה"); self.rb_folder.setChecked(True)
            self.rb_phone = QRadioButton("טלפון (טריגר מצד A)")
            self.rb_phone.setToolTip("כל תמונה חדשה בתיקייה של צד A מצלמת את צד B בטלפון שמחובר בכבל USB.\n"
                                     "התמונה נשמרת במחשב בשם של תמונת A. שום דבר לא נשמר בטלפון.")
            src = QButtonGroup(self); src.addButton(self.rb_folder); src.addButton(self.rb_phone)
            self.rb_phone.toggled.connect(main.set_b_phone)
            row.addWidget(self.rb_folder); row.addWidget(self.rb_phone); row.addStretch(1)
            l1.addLayout(row)
        l1.addWidget(self.btn_folder); l1.addWidget(self.lbl_folder)
        if idx == 1:
            self._build_phone_box(l1)
        lay.addWidget(self.g1)
        self.g2 = QGroupBox("2 · קנה מידה")
        l2 = QVBoxLayout(self.g2)
        row = QHBoxLayout()
        row.addWidget(QLabel("מרחק ידוע:"))
        self.known = QDoubleSpinBox(); self.known.setRange(0.001, 10000); self.known.setDecimals(3)
        self.known.setValue(25.4); self.known.setSuffix("  מ״מ")
        self.known.setToolTip("אם הסרגל באינצ'ים: אינץ' אחד = 25.4 מ״מ. ודא את המרחק המסומן בפועל.")
        row.addWidget(self.known, 1)
        l2.addLayout(row)
        row = QHBoxLayout()
        self.btn_cal = _btn("סמן 2 קצוות", "primary", slot=lambda: main.start_pick(idx, "calib"))
        self.btn_cal_img = _btn("תמונת כיול אחרת...", slot=lambda: main.pick_calib_image(idx))
        row.addWidget(self.btn_cal); row.addWidget(self.btn_cal_img)
        l2.addLayout(row)
        self.cal_table = QTableWidget(0, 3)
        self.cal_table.setHorizontalHeaderLabels(["פיקסלים", "מ״מ", "px/mm"])
        self.cal_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cal_table.verticalHeader().setVisible(False)
        self.cal_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cal_table.setMaximumHeight(92)
        l2.addWidget(self.cal_table)
        row = QHBoxLayout()
        self.lbl_scale = QLabel("לא נקבע"); self.lbl_scale.setStyleSheet("font-weight:600")
        self.btn_cal_del = QToolButton(); self.btn_cal_del.setText("מחק אחרון")
        self.btn_cal_del.clicked.connect(lambda: main.remove_calib(idx))
        row.addWidget(self.lbl_scale, 1); row.addWidget(self.btn_cal_del)
        l2.addLayout(row)
        lay.addWidget(self.g2)
        self.g3 = QGroupBox("3 · נקודת התחלה וקצה הסדק")
        l3 = QVBoxLayout(self.g3)
        self.btn_crack = _btn("סמן נקודת התחלה ← קצה סדק", "primary",
                              tip="לחיצה ראשונה: נקודת ההתחלה של המדידה (למשל קצה החריץ).\n"
                                  "לחיצה שנייה: קצה הסדק בתמונת הבסיס.",
                              slot=lambda: main.start_pick(idx, "crack"))
        self.lbl_crack = QLabel("לא סומן"); self.lbl_crack.setWordWrap(True)
        self.btn_roi = QToolButton(); self.btn_roi.setText("אזור יישור (לא חובה)")
        self.btn_roi.clicked.connect(lambda: main.start_pick(idx, "roi"))
        row = QHBoxLayout(); row.addWidget(self.btn_roi); row.addStretch(1)
        l3.addWidget(self.btn_crack); l3.addLayout(row); l3.addWidget(self.lbl_crack)
        lay.addWidget(self.g3)
        self.g4 = QGroupBox("4 · רגישות הזיהוי")
        self.g4.setToolTip("כמה דק / חלש קו יכול להיות ועדיין להיחשב סדק.\n"
                           "יותר רגיש = גם קווים דקים מאוד (הסדק ייצא ארוך יותר, אבל עלול לתפוס רעש).\n"
                           "פחות רגיש = רק קווים ברורים (הסדק ייצא קצר יותר).")
        l4 = QVBoxLayout(self.g4)
        row = QHBoxLayout()
        self.sens = QSlider(Qt.Horizontal); self.sens.setRange(-5, 5); self.sens.setValue(0)
        self.sens.setTickPosition(QSlider.TicksBelow); self.sens.setTickInterval(1); self.sens.setPageStep(1)
        self.sens_lbl = QLabel(""); self.sens_lbl.setMinimumWidth(150)
        row.addWidget(self.sens, 1); row.addWidget(self.sens_lbl)
        l4.addLayout(row)
        row = QHBoxLayout()
        lo = QLabel("רק קווים ברורים"); lo.setObjectName("Muted")
        hi = QLabel("גם קווים דקים מאוד"); hi.setObjectName("Muted")
        row.addWidget(lo); row.addStretch(1); row.addWidget(hi); row.addSpacing(156)
        l4.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("בדוק על תמונה מס׳"))
        self.prev_spin = QSpinBox(); self.prev_spin.setRange(1, 1); self.prev_spin.setMinimumWidth(80)
        self.prev_spin.setToolTip("בחר תמונה מאוחרת יותר, שבה הסדק כבר צמח, ובדוק איפה נמצא הקצה")
        self.btn_test = _btn("הצג", slot=lambda: main.preview_image(idx, self.prev_spin.value() - 1))
        row.addWidget(self.prev_spin); row.addWidget(self.btn_test); row.addStretch(1)
        l4.addLayout(row)
        self.prev_info = QLabel(""); self.prev_info.setObjectName("Muted"); self.prev_info.setWordWrap(True)
        l4.addWidget(self.prev_info)
        lay.addWidget(self.g4)
        self.set_sens_label(0)
        self.sens_timer = QTimer(self); self.sens_timer.setSingleShot(True); self.sens_timer.setInterval(250)
        self.sens_timer.timeout.connect(lambda: main.apply_sensitivity(idx))
        self.sens.valueChanged.connect(self._sens_moved)
        lay.addStretch(1)

    def _build_phone_box(self, l1):
        main = self.main
        self.phone_box = QWidget(); pl = QVBoxLayout(self.phone_box); pl.setContentsMargins(0, 2, 0, 0); pl.setSpacing(5)
        self.btn_bfolder = _btn("תיקיית שמירה לצד B...", slot=main.pick_b_folder, icon="SP_DirOpenIcon",
                                tip="כאן נשמרות התמונות מהטלפון (בשם של תמונת A) ולוג הטריגרים")
        self.lbl_bfolder = QLabel("לא נבחרה תיקייה"); self.lbl_bfolder.setObjectName("Muted"); self.lbl_bfolder.setWordWrap(True)
        pl.addWidget(self.btn_bfolder); pl.addWidget(self.lbl_bfolder)
        row = QHBoxLayout()
        self.lbl_conn = QLabel("● לא מחובר"); self.lbl_conn.setWordWrap(True)
        self.btn_install = QToolButton(); self.btn_install.setText("התקן אפליקציה")
        self.btn_install.setToolTip("מתקין / מעדכן את אפליקציית הטלפון (FatigueCam) דרך הכבל ומפעיל אותה")
        self.btn_install.clicked.connect(main.install_phone_app)
        row.addWidget(self.lbl_conn, 1); row.addWidget(self.btn_install)
        pl.addLayout(row)
        row = QHBoxLayout()
        self.btn_plive = _btn("תצוגה חיה", icon="SP_MediaPlay", tip="תמונה חיה מהטלפון – לכיוון הטלפון על הדגם")
        self.btn_plive.setCheckable(True); self.btn_plive.toggled.connect(main.phone_live)
        self.btn_pfocus = _btn("כוון פוקוס ונעל", tip="חשיפה אוטומטית, סריקת כל טווח הפוקוס, ונעילה על החד ביותר.\n"
                                                      "הפוקוס נשאר נעול לכל הריצה – ההגדלה (קנה המידה) לא משתנה.\n"
                                                      "לחיצה נוספת אחר כך = כיוון מחדש.",
                               slot=main.phone_focus)
        row.addWidget(self.btn_plive); row.addWidget(self.btn_pfocus)
        pl.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("פוקוס:"))
        self.ph_focus = QDoubleSpinBox(); self.ph_focus.setRange(0, 20); self.ph_focus.setDecimals(2)
        self.ph_focus.setSingleStep(0.05); self.ph_focus.setSuffix("  D"); self.ph_focus.setKeyboardTracking(False)
        self.ph_focus.setToolTip("מרחק הפוקוס הנעול (דיופטר). שינוי ידני אפשרי – אבל אז צריך לצלם כיול מחדש")
        self.ph_focus.valueChanged.connect(main.phone_focus_manual)
        row.addWidget(self.ph_focus); row.addStretch(1)
        pl.addLayout(row)
        self.lbl_pfocus = QLabel("הפוקוס עוד לא כוון"); self.lbl_pfocus.setObjectName("Muted"); self.lbl_pfocus.setWordWrap(True)
        pl.addWidget(self.lbl_pfocus)
        self.btn_pbase = _btn("צלם תמונת בסיס", "primary", icon="SP_DialogApplyButton",
                              tip="תמונה מהטלפון לסימון נקודת ההתחלה וקצה הסדק (שלב 3), וגם לכיול אם אין תמונת כיול נפרדת",
                              slot=lambda: main.phone_capture("base"))
        pl.addWidget(self.btn_pbase)
        self.chk_trig = QCheckBox("טריגר פעיל: כל תמונה חדשה בצד A מצלמת את צד B"); self.chk_trig.setChecked(True)
        self.chk_trig.toggled.connect(lambda _: main.phone_arm())
        pl.addWidget(self.chk_trig)
        row = QHBoxLayout()
        row.addWidget(QLabel("אזהרה אם לא הגיעה תוך"))
        self.ph_timeout = QDoubleSpinBox(); self.ph_timeout.setRange(1, 30); self.ph_timeout.setDecimals(1)
        self.ph_timeout.setValue(5.0); self.ph_timeout.setSuffix("  ש׳")
        self.ph_timeout.valueChanged.connect(main.phone_params)
        row.addWidget(self.ph_timeout)
        row.addWidget(QLabel("רצף:"))
        self.ph_burst = QSpinBox(); self.ph_burst.setRange(1, 5); self.ph_burst.setValue(3)
        self.ph_burst.setToolTip("כמה תמונות הטלפון מצלם בכל טריגר (60 מ״ש ביניהן); נשמרת החדה ביותר")
        self.ph_burst.valueChanged.connect(main.phone_params)
        row.addWidget(self.ph_burst); row.addStretch(1)
        pl.addLayout(row)
        self.lbl_trig = QLabel(""); self.lbl_trig.setWordWrap(True); self.lbl_trig.setTextFormat(Qt.RichText)
        pl.addWidget(self.lbl_trig)
        row = QHBoxLayout()
        b = QToolButton(); b.setText("פתח לוג טריגרים"); b.clicked.connect(main.open_phone_log)
        row.addWidget(b); row.addStretch(1)
        pl.addLayout(row)
        self.phone_box.hide()
        l1.addWidget(self.phone_box)

    def controls(self):
        c = [self.btn_folder, self.known, self.btn_cal, self.btn_cal_img, self.btn_cal_del,
             self.btn_crack, self.btn_roi, self.btn_test, self.sens, self.prev_spin]
        return c + ([self.rb_folder, self.rb_phone, self.phone_box] if self.idx == 1 else [])

    def _sens_moved(self, v):
        self.set_sens_label(v)
        self.sens_timer.start()

    def set_sens_label(self, v):
        if v == 0:
            self.sens_lbl.setText("<b>0</b>  ברירת מחדל"); return
        txt = "יותר רגיש" if v > 0 else "פחות רגיש"
        col = "#b45309" if v >= 4 else "#0f172a"
        extra = " · עלול לתפוס רעש" if v >= 4 else ""
        self.sens_lbl.setText(f"<b style='color:{col}'>{v:+d}</b>  {txt}{extra}")

    def set_done(self, folder, scale, crack):
        for g, ok in ((self.g1, folder), (self.g2, scale), (self.g3, crack)):
            t = g.title().replace("✓ ", "")
            g.setTitle(("✓ " + t) if ok else t)
            g.setProperty("done", "true" if ok else "false"); _repolish(g)


# =============================================================================
# Main window
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {VERSION}")
        if os.path.exists(resource_path("icon.ico")):
            self.setWindowIcon(QIcon(resource_path("icon.ico")))
        self.settings = QSettings("CrackTracker", "CrackTracker")
        self.sides = [self._new_side(), self._new_side()]
        self.use_b = False
        self.b_phone = False                      # side B from the phone (triggered by side A)
        self.phone: phone_b.PhoneB | None = None
        self.setup_side = 0
        self.cur_row = -1
        self.follow = True
        self.pick = None
        self.worker: Worker | None = None
        self.remeasure: RemeasureWorker | None = None
        self.exporter: ExportWorker | None = None
        self._prog_select = False
        self._t0 = time.time()
        self._total = 0
        self._export_dir = ""
        self._fix_info = None
        self._plot_dirty = False
        self.alerts = dict(ALERT_DEFAULTS)
        try:
            self.alerts.update(json.loads(self.settings.value("alerts", "{}")))
        except Exception:
            pass
        self.sens_events = []
        self._reset_results()
        self.model = ResultsModel(self)

        root = QWidget(); vl = QVBoxLayout(root); vl.setContentsMargins(0, 0, 0, 0); vl.setSpacing(0)
        vl.addWidget(self._build_header())
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_setup_page())
        self.stack.addWidget(self._build_measure_page())
        self.stack.addWidget(self._build_results_page())
        self.stack.addWidget(self._build_alerts_page())
        vl.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        self._build_statusbar()
        self._build_menu()
        self._build_shortcuts()
        self.toast_w = Toast(root)

        self.ui_timer = QTimer(self); self.ui_timer.setInterval(1000)
        self.ui_timer.timeout.connect(self._periodic)
        self.ui_timer.start()
        self.set_use_b(False)
        self._update_ready()
        self._set_state("idle")
        g = self.settings.value("geometry")
        if g is not None:
            self.restoreGeometry(g)
        else:
            self.resize(1600, 980)
        self._goto(PAGE_SETUP)
        self.log(f"{APP_NAME} {VERSION} מוכן.")

    # ------------------------------------------------------------------ chrome
    def _build_header(self):
        h = QFrame(); h.setObjectName("Header")
        lay = QHBoxLayout(h); lay.setContentsMargins(16, 0, 16, 0); lay.setSpacing(4)
        logo = QLabel("Crack Tracker"); logo.setObjectName("Logo")
        ver = QLabel(f"v{VERSION}"); ver.setObjectName("Ver")
        lay.addWidget(logo); lay.addWidget(ver); lay.addSpacing(30)
        self.nav = QButtonGroup(self); self.nav.setExclusive(True)
        self.nav_btns = []
        for i, t in enumerate(("1   הגדרה", "2   מדידה", "3   תוצאות וייצוא", "4   התראות")):
            b = QPushButton(t); b.setObjectName("Nav"); b.setCheckable(True); b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(f"Ctrl+{i + 1}")
            self.nav.addButton(b, i); lay.addWidget(b); self.nav_btns.append(b)
        self.nav.idClicked.connect(self._goto)
        lay.addStretch(1)
        self.chip = QLabel("מוכן"); self.chip.setObjectName("Chip")
        lay.addWidget(self.chip)
        return h

    def _build_statusbar(self):
        sb = self.statusBar()
        self.sb_state = QLabel("מוכן")
        self.sb_rate = QLabel("")
        self.sb_prog = QProgressBar(); self.sb_prog.setFixedWidth(260); self.sb_prog.hide()
        sb.addWidget(self.sb_state)
        sb.addWidget(self.sb_rate)
        self.sb_phone = QLabel(""); self.sb_phone.hide()
        sb.addPermanentWidget(self.sb_phone)
        sb.addPermanentWidget(self.sb_prog)

    def _build_menu(self):
        m = self.menuBar().addMenu("קובץ")
        for text, fn, key in [("שמור הגדרות...", self.save_setup, "Ctrl+S"),
                              ("טען הגדרות...", self.load_setup, "Ctrl+O"),
                              (None, None, None),
                              ("ייצא טבלה ל-CSV...", self.export_csv, "Ctrl+Shift+E"),
                              ("שמור תמונות מסומנות...", self.export_annotated, "Ctrl+E")]:
            if text is None:
                m.addSeparator(); continue
            a = QAction(text, self); a.setShortcut(QKeySequence(key)); a.triggered.connect(fn); m.addAction(a)
        v = self.menuBar().addMenu("תצוגה")
        a = QAction("מסך מלא", self); a.setShortcut(QKeySequence("F11")); a.triggered.connect(self._toggle_fullscreen)
        v.addAction(a)
        h = self.menuBar().addMenu("עזרה")
        a = QAction("איך משתמשים", self); a.setShortcut(QKeySequence("F1")); a.triggered.connect(self.show_help)
        h.addAction(a)

    def _build_shortcuts(self):
        def sc(key, fn, widget=None):
            s = QShortcut(QKeySequence(key), widget or self)
            if widget is not None:
                s.setContext(Qt.WidgetWithChildrenShortcut)
            s.activated.connect(fn)
        for n, page in enumerate(getattr(self, "nav_pages", range(4))):
            sc(f"Ctrl+{n + 1}", lambda page=page: self._goto(page))
        sc("Escape", self._escape)
        sc("F5", lambda: self.start_worker(False))
        mp = self.measure_page
        sc("Left", lambda: self._step(-1), mp)
        sc("Right", lambda: self._step(1), mp)
        sc("Home", lambda: self._jump(0), mp)
        sc("End", lambda: self._set_follow(True), mp)
        sc("N", self._next_check, mp)
        sc("F", lambda: self._set_follow(not self.follow), mp)
        sc("T", lambda: self.start_fix(0), mp)

    def _goto(self, page):
        self.stack.setCurrentIndex(page)
        self.nav_btns[page].setChecked(True)
        if page == PAGE_MEASURE and not self.R["k"]:
            self._preview_bases()
        if page == PAGE_RESULTS:
            self._update_stats()

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    def _escape(self):
        if self.pick:
            self._cancel_pick()
            self.set_hint("")
            self.set_setup_hint("הפעולה בוטלה.")
        elif self.isFullScreen():
            self.showMaximized()

    def toast(self, text, kind="info", ms=3500):
        self.toast_w.pop(text, kind, ms)

    def _set_state(self, st, extra=""):
        txt, col = {"idle": ("מוכן", "#334155"), "run": ("מודד", "#2563eb"), "live": ("מעקב חי", "#16a34a"),
                    "stop": ("עוצר…", "#b45309"), "remeasure": ("מודד מחדש", "#7c3aed")}[st]
        self.chip.setText(("●  " + txt) if st != "idle" else txt)
        self.chip.setStyleSheet(f"background:{col};")
        self.sb_state.setText(txt + (f"  ·  {extra}" if extra else ""))

    # ------------------------------------------------------------------ pages
    def _page(self):
        w = QWidget(); w.setObjectName("Page")
        return w

    def _build_setup_page(self):
        w = self._page(); lay = QHBoxLayout(w); lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(12)
        right = QFrame(); right.setObjectName("Card"); right.setFixedWidth(440)
        rl = QVBoxLayout(right); rl.setContentsMargins(0, 4, 0, 12); rl.setSpacing(8)
        self.tabs = QTabWidget(); self.tabs.setDocumentMode(True)
        self.panels = [SidePanel(0, self), SidePanel(1, self)]
        self.tabs.addTab(self.panels[0], "צד A")
        sa = QScrollArea(); sa.setWidgetResizable(True); sa.setFrameShape(QFrame.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setWidget(self.panels[1])
        self.tabs.addTab(sa, "צד B")
        self.tabs.addTab(self._build_settings_tab(), "הגדרות זיהוי")
        self.tabs.currentChanged.connect(self._tab_changed)
        rl.addWidget(self.tabs, 1)
        self.btn_go = _btn("המשך למדידה  ←", "primary", slot=lambda: self._goto(PAGE_MEASURE))
        self.btn_go.setMinimumHeight(40)
        bl = QHBoxLayout(); bl.setContentsMargins(12, 0, 12, 0); bl.addWidget(self.btn_go)
        rl.addLayout(bl)
        lay.addWidget(right)
        left = QVBoxLayout(); left.setSpacing(8)
        self.setup_hint = QLabel(""); self.setup_hint.setObjectName("Banner"); self.setup_hint.setWordWrap(True)
        left.addWidget(self.setup_hint)
        self.su = SetupView()
        self.su.view.clicked.connect(self._on_setup_click)
        self.su.zoom.clicked.connect(self._on_setup_click)
        left.addWidget(self.su, 1)
        lay.addLayout(left, 1)
        self.set_setup_hint("שלב 1: בחר תיקיית תמונות לצד A.")
        return w

    def _build_settings_tab(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(12, 10, 12, 10)
        g = QGroupBox("זיהוי הסדק")
        f = QFormLayout(g); f.setSpacing(10)
        self.thr_auto = QCheckBox("סף אוטומטי (לפי הרקע בתמונת הבסיס)"); self.thr_auto.setChecked(True)
        self.thr = QDoubleSpinBox(); self.thr.setRange(0.01, 0.95); self.thr.setSingleStep(0.01)
        self.thr.setDecimals(3); self.thr.setValue(0.20); self.thr.setEnabled(False)
        self.thr_auto.toggled.connect(lambda on: self.thr.setEnabled(not on))
        self.thr_auto.toggled.connect(self._params_changed)
        self.thr.valueChanged.connect(self._params_changed)
        self.ahead = QDoubleSpinBox(); self.ahead.setRange(2, 500); self.ahead.setValue(40)
        self.ahead.setSuffix("  מ״מ")
        self.ahead.valueChanged.connect(self._params_changed)
        self.chk_prior = QCheckBox("חפש קודם סדק באורך התמונה הקודמת לפחות"); self.chk_prior.setChecked(True)
        self.chk_prior.setToolTip("בכל תמונה החיפוש מניח קודם שהסדק לפחות באורך של התמונה הקודמת "
                                  "(מגשר על פערים קצרים עד שם). אם אין שם סדק – יימצא קצר יותר.\n"
                                  "המדידה עצמה לא מסוננת.")
        self.chk_prior.toggled.connect(self._params_changed)
        self.chk_check = QCheckBox("בדיקה נוספת של כל קצה (מסמן קצה לא בטוח)"); self.chk_check.setChecked(True)
        self.chk_check.toggled.connect(self._params_changed)
        f.addRow(self.thr_auto)
        f.addRow("סף ידני:", self.thr)
        f.addRow("אורך מעקב מקסימלי:", self.ahead)
        f.addRow(self.chk_prior)
        f.addRow(self.chk_check)
        lay.addWidget(g)
        g2 = QGroupBox("קובץ הגדרות")
        r = QHBoxLayout(g2)
        r.addWidget(_btn("שמור הגדרות...", slot=self.save_setup))
        r.addWidget(_btn("טען הגדרות...", slot=self.load_setup))
        lay.addWidget(g2)
        lay.addStretch(1)
        return w

    def _build_measure_page(self):
        w = self._page(); self.measure_page = w
        lay = QVBoxLayout(w); lay.setContentsMargins(12, 10, 12, 8); lay.setSpacing(8)
        tb = QFrame(); tb.setObjectName("Toolbar")
        tl = QHBoxLayout(tb); tl.setContentsMargins(10, 6, 10, 6); tl.setSpacing(8)
        self.btn_run = _btn("מדוד את כל התמונות", "primary", "F5", lambda: self.start_worker(live=False), "SP_MediaPlay")
        self.btn_live = _btn("מעקב חי", icon="SP_BrowserReload", tip="מודד כל תמונה חדשה שנשמרת בתיקייה, עד לעצירה",
                             slot=lambda: self.start_worker(live=True))
        self.btn_stop = _btn("עצור", "danger", slot=self.stop_worker, icon="SP_MediaStop")
        self.btn_stop.setEnabled(False)
        tl.addWidget(self.btn_run); tl.addWidget(self.btn_live); tl.addWidget(self.btn_stop)
        tl.addWidget(_vsep())
        navw = QWidget(); navw.setLayoutDirection(Qt.LeftToRight)
        nl = QHBoxLayout(navw); nl.setContentsMargins(0, 0, 0, 0); nl.setSpacing(4)
        self.btn_first = _btn("", "nav", "התמונה הראשונה (Home)", lambda: self._jump(0), "SP_MediaSkipBackward")
        self.btn_prev = _btn("", "nav", "הקודמת (←)", lambda: self._step(-1), "SP_MediaSeekBackward")
        self.spin_img = QSpinBox(); self.spin_img.setRange(0, 0); self.spin_img.setMinimumWidth(80)
        self.spin_img.setAlignment(Qt.AlignCenter)
        self.spin_img.setButtonSymbols(QSpinBox.NoButtons)
        self.spin_img.setKeyboardTracking(False)
        self.spin_img.setToolTip("מספר תמונה – הקלד ולחץ Enter")
        self.spin_img.valueChanged.connect(self._spin_changed)
        self.lbl_count = QLabel("/ 0"); self.lbl_count.setObjectName("Muted")
        self.btn_next = _btn("", "nav", "הבאה (→)", lambda: self._step(1), "SP_MediaSeekForward")
        self.btn_last = _btn("", "nav", "האחרונה (End)", lambda: self._set_follow(True), "SP_MediaSkipForward")
        for x in (self.btn_first, self.btn_prev, self.spin_img, self.lbl_count, self.btn_next, self.btn_last):
            nl.addWidget(x)
        tl.addWidget(navw)
        self.chk_follow = QCheckBox("עקוב אחרי התמונה האחרונה"); self.chk_follow.setChecked(True)
        self.chk_follow.setToolTip("כשמסומן – התצוגה עוברת לכל תמונה חדשה שנמדדה (F).\n"
                                   "נכבה לבד כשבוחרים תמונה או לוחצים \"תקן קצה\"; המדידה ממשיכה ברקע.")
        self.chk_follow.toggled.connect(self._set_follow)
        tl.addWidget(self.chk_follow)
        tl.addWidget(_vsep())
        self.btn_next_check = _btn("לבדיקה: 0", icon="SP_MessageBoxWarning",
                                   tip="קפוץ לתמונה הבאה שבה הקצה לא בטוח או שהמדידה נכשלה (N)",
                                   slot=self._next_check)
        tl.addWidget(self.btn_next_check)
        self.diff_lbl = QLabel(""); self.diff_lbl.setObjectName("Pill"); self.diff_lbl.hide()
        self.diff_lbl.setToolTip("הפרש בין צד A לצד B בתמונה המוצגת (הסף נקבע במסך \"התראות\")")
        tl.addWidget(self.diff_lbl)
        tl.addStretch(1)
        self.btn_full = QToolButton(); self.btn_full.setText(" " + "מסך מלא"); self.btn_full.setIcon(_icon("SP_TitleBarMaxButton")); self.btn_full.setToolTip("F11")
        self.btn_full.clicked.connect(self._toggle_fullscreen)
        self.btn_full.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        tl.addWidget(self.btn_full)
        lay.addWidget(tb)
        self.hint = QLabel(""); self.hint.setObjectName("Banner"); self.hint.setWordWrap(True); self.hint.hide()
        lay.addWidget(self.hint)
        self.sv = [SideView(0, self), SideView(1, self)]
        for i in (0, 1):
            self.sv[i].view.clicked.connect(lambda x, y, i=i: self._on_fix_click(i, x, y))
            self.sv[i].zoom.clicked.connect(lambda x, y, i=i: self._on_fix_click(i, x, y))
        vs = QSplitter(Qt.Vertical); vs.setChildrenCollapsible(False)
        sides = QWidget(); sl = QVBoxLayout(sides); sl.setContentsMargins(0, 0, 0, 0); sl.setSpacing(8)
        sl.addWidget(self.sv[0], 1); sl.addWidget(self.sv[1], 1)
        vs.addWidget(sides)
        bottom = QWidget(); bl = QHBoxLayout(bottom); bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(8)
        tcard = QFrame(); tcard.setObjectName("Card")
        tcl = QVBoxLayout(tcard); tcl.setContentsMargins(8, 6, 8, 8); tcl.setSpacing(4)
        th = QHBoxLayout(); t = QLabel("תוצאות אחרונות"); t.setObjectName("PanelTitle"); th.addWidget(t)
        th.addStretch(1)
        more = QToolButton(); more.setText("לטבלה המלאה ←"); more.setCursor(Qt.PointingHandCursor)
        more.clicked.connect(lambda: self._goto(PAGE_RESULTS))
        th.addWidget(more); tcl.addLayout(th)
        self.compact = self._make_table(compact=True)
        tcl.addWidget(self.compact, 1)
        pcard = QFrame(); pcard.setObjectName("Card")
        pcl = QVBoxLayout(pcard); pcl.setContentsMargins(8, 6, 8, 4); pcl.setSpacing(0)
        t2 = QLabel("אורך הסדק לאורך הניסוי"); t2.setObjectName("PanelTitle"); pcl.addWidget(t2)
        self.fig = Figure(figsize=(5, 1.5))
        self.fig.patch.set_facecolor("white")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setLayoutDirection(Qt.LeftToRight)
        self.canvas.setMinimumHeight(90)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.26)
        pcl.addWidget(self.canvas, 1)
        bl.addWidget(tcard, 5); bl.addWidget(pcard, 6)
        vs.addWidget(bottom)
        vs.setStretchFactor(0, 1); vs.setStretchFactor(1, 0)
        vs.setSizes([760, 215])
        lay.addWidget(vs, 1)
        legend = QLabel(f"<span style='color:#b8a000'>+</span> נקודת התחלה &nbsp;&nbsp; "
                        f"<span style='color:{C_TIP}'>◯</span> קצה &nbsp;&nbsp; "
                        f"<span style='color:#d97706'>◯</span> קצה לא בטוח &nbsp;&nbsp; "
                        f"<span style='color:{C_MAN}'>◯</span> תוקן ידנית &nbsp;&nbsp; "
                        f"<span style='color:{C_MEAS}'>━</span> a לאורך הסדק &nbsp;&nbsp;·&nbsp;&nbsp; "
                        f"← → מעבר בין תמונות · N לבדיקה · T תיקון · F מעקב · Esc ביטול")
        legend.setObjectName("Muted")
        lay.addWidget(legend)
        return w

    def _make_table(self, compact):
        tv = QTableView()
        tv.setModel(self.model)
        tv.setAlternatingRowColors(True)
        tv.setSelectionBehavior(QAbstractItemView.SelectRows)
        tv.setSelectionMode(QAbstractItemView.SingleSelection)
        tv.verticalHeader().setVisible(False)
        tv.verticalHeader().setDefaultSectionSize(24 if compact else 26)
        tv.setShowGrid(False)
        tv.setWordWrap(False)
        hh = tv.horizontalHeader()
        hh.setHighlightSections(False)
        hh.setStretchLastSection(True)
        hh.setSectionResizeMode(QHeaderView.Interactive)
        nw = 72 if compact else 88
        widths = {C_EXP: 52, C_N: 56, C_FILE: 240, C_A: nw, C_B: nw, C_AM: nw, C_DA: nw, C_DIFF: nw + 14,
                  C_THA: nw, C_THB: nw, C_THA0: nw, C_THB0: nw}
        for c, wd in widths.items():
            tv.setColumnWidth(c, wd)
        if compact:
            tv.setColumnHidden(C_EXP, True); tv.setColumnHidden(C_FILE, True)
            tv.setMinimumHeight(100)
            tv.selectionModel().currentRowChanged.connect(self._compact_row_changed)
            tv.setToolTip("לחיצה על שורה = הצגת התמונה")
        else:
            tv.doubleClicked.connect(lambda idx: self._open_row(idx.row()) if idx.column() != C_EXP else None)
            tv.setToolTip("לחיצה כפולה = פתיחת התמונה במסך המדידה.\nתיבת הסימון קובעת אילו שורות ייוצאו.")
        return tv

    def _build_results_page(self):
        w = self._page(); lay = QVBoxLayout(w); lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(10)
        stats = QHBoxLayout(); stats.setSpacing(10)
        self.stat = {}
        for key, lbl in (("n", "תמונות שנמדדו"), ("a", "a אחרון [מ״מ]"), ("da", "Δa כולל [מ״מ]"),
                         ("man", "תיקונים ידניים"), ("chk", "לבדיקה"), ("alr", "חריגות (התראות)"),
                         ("exp", "מסומנות לייצוא")):
            f = QFrame(); f.setObjectName("Stat"); fl = QVBoxLayout(f); fl.setContentsMargins(14, 10, 14, 10)
            v = QLabel("—"); v.setObjectName("StatVal"); l = QLabel(lbl); l.setObjectName("StatLbl")
            fl.addWidget(v); fl.addWidget(l); stats.addWidget(f, 1); self.stat[key] = v
        lay.addLayout(stats)
        tb = QFrame(); tb.setObjectName("Toolbar")
        tl = QHBoxLayout(tb); tl.setContentsMargins(10, 6, 10, 6); tl.setSpacing(8)
        tl.addWidget(_btn("סמן הכל", slot=lambda: self._set_all_export(True)))
        tl.addWidget(_btn("נקה סימון", slot=lambda: self._set_all_export(False)))
        tl.addWidget(_btn("סמן רק חריגות / ידני / לבדיקה",
                          tip="מסמן לייצוא רק שורות עם חריגת התראה, שתוקנו ידנית, לא בטוחות או נכשלו",
                          slot=self._select_flagged))
        tl.addWidget(_vsep())
        tl.addWidget(_btn("ייצא טבלה (CSV)", "primary", "Ctrl+Shift+E", self.export_csv, "SP_DialogSaveButton"))
        tl.addWidget(_btn("שמור תמונות מסומנות", "primary", "Ctrl+E", self.export_annotated, "SP_DialogSaveButton"))
        self.btn_stop_export = _btn("עצור ייצוא", "danger", slot=self._stop_export)
        self.btn_stop_export.setEnabled(False)
        tl.addWidget(self.btn_stop_export)
        self.prog = QProgressBar(); self.prog.setFixedWidth(260); self.prog.setValue(0); self.prog.hide()
        tl.addWidget(self.prog)
        tl.addStretch(1)
        self.btn_log = QToolButton(); self.btn_log.setText("יומן פעולות"); self.btn_log.setCheckable(True)
        tl.addWidget(self.btn_log)
        lay.addWidget(tb)
        card = QFrame(); card.setObjectName("Card"); cl = QVBoxLayout(card); cl.setContentsMargins(1, 1, 1, 1)
        self.full = self._make_table(compact=False)
        cl.addWidget(self.full)
        lay.addWidget(card, 1)
        self.logbox = QPlainTextEdit(); self.logbox.setObjectName("Log"); self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(3000); self.logbox.setFixedHeight(150)
        self.logbox.hide()
        self.btn_log.toggled.connect(self.logbox.setVisible)
        lay.addWidget(self.logbox)
        return w

    def _build_alerts_page(self):
        w = self._page(); outer = QHBoxLayout(w); outer.setContentsMargins(12, 12, 12, 12)
        col = QWidget(); col.setMaximumWidth(1000)
        lay = QVBoxLayout(col); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)
        outer.addStretch(1); outer.addWidget(col, 10); outer.addStretch(1)
        head = QLabel("<span style='font-size:15pt; font-weight:700; color:#0f172a'>התראות</span><br>"
                      "<span style='color:#64748b'>הערכים מוצגים תמיד בטבלאות. כשערך עובר את הסף שנקבע כאן – "
                      "המשבצת (או כל השורה) נצבעת באדום. אין חלונות קופצים, והמדידה לא נעצרת.</span>")
        head.setWordWrap(True)
        lay.addWidget(head)
        body = QHBoxLayout(); body.setSpacing(12)
        left = QVBoxLayout(); left.setSpacing(12)
        aw = self.alert_w = {}
        g1 = QGroupBox("הפרש בין צד A לצד B")
        f1 = QFormLayout(g1); f1.setSpacing(10)
        aw["diff_on"] = QCheckBox("סמן באדום כשההפרש גדול מהסף")
        aw["diff_thr"] = QDoubleSpinBox(); aw["diff_thr"].setRange(0.01, 100); aw["diff_thr"].setDecimals(2)
        aw["diff_thr"].setSingleStep(0.1); aw["diff_thr"].setSuffix("  מ״מ")
        aw["diff_mode"] = QComboBox()
        aw["diff_mode"].addItem("Δa – הגידול מתחילת הניסוי (מומלץ)", "da")
        aw["diff_mode"].addItem("a – האורך מנקודת ההתחלה של כל צד", "a")
        aw["diff_mode"].setToolTip("Δa לא מושפע מכך שנקודות ההתחלה בשני הצדדים סומנו במקומות קצת שונים")
        f1.addRow(aw["diff_on"]); f1.addRow("סף:", aw["diff_thr"]); f1.addRow("השווה:", aw["diff_mode"])
        n1 = QLabel("פעיל רק כשמודדים שני צדדים."); n1.setObjectName("Muted"); f1.addRow(n1)
        left.addWidget(g1)
        g2 = QGroupBox("זווית החלק החדש של הסדק")
        f2 = QFormLayout(g2); f2.setSpacing(10)
        aw["ang_new_on"] = QCheckBox("סמן באדום כשהסדק החדש סוטה מהכיוון ההתחלתי ביותר מ-")
        aw["ang_new_thr"] = QDoubleSpinBox(); aw["ang_new_thr"].setRange(0.5, 90); aw["ang_new_thr"].setDecimals(1)
        aw["ang_new_thr"].setSuffix("  °")
        aw["ang_last_mm"] = QDoubleSpinBox(); aw["ang_last_mm"].setRange(0, 50); aw["ang_last_mm"].setDecimals(1)
        aw["ang_last_mm"].setSingleStep(0.5); aw["ang_last_mm"].setSuffix("  מ״מ"); aw["ang_last_mm"].setSpecialValueText("כל החלק החדש")
        aw["ang_last_mm"].setToolTip("0 = כל החלק שגדל מאז תחילת הניסוי.\n"
                                     "ערך > 0 = רק המ״מ האחרונים ליד הקצה – מגלה שבירה/פנייה חדשה מהר יותר.")
        f2.addRow(aw["ang_new_on"]); f2.addRow("סף:", aw["ang_new_thr"]); f2.addRow("חשב על:", aw["ang_last_mm"])
        n2 = QLabel("נמדדת רק כשהחלק החדש ארוך מ-0.5 מ״מ (לפני כן מוצג ריק)."); n2.setObjectName("Muted")
        f2.addRow(n2)
        left.addWidget(g2)
        g3 = QGroupBox("זווית כוללת (מנקודת ההתחלה עד הקצה)")
        f3 = QFormLayout(g3); f3.setSpacing(10)
        aw["ang_all_on"] = QCheckBox("סמן באדום כשהזווית הכוללת גדולה מ-")
        aw["ang_all_thr"] = QDoubleSpinBox(); aw["ang_all_thr"].setRange(0.5, 90); aw["ang_all_thr"].setDecimals(1)
        aw["ang_all_thr"].setSuffix("  °")
        f3.addRow(aw["ang_all_on"]); f3.addRow("סף:", aw["ang_all_thr"])
        left.addWidget(g3)
        g4 = QGroupBox("הצגה")
        f4 = QVBoxLayout(g4)
        aw["row_mode"] = QCheckBox("צבע את כל השורה (ולא רק את המשבצת שחורגת)")
        f4.addWidget(aw["row_mode"])
        left.addWidget(g4)
        left.addStretch(1)
        body.addLayout(left, 3)
        right = QFrame(); right.setObjectName("Card")
        rl = QVBoxLayout(right); rl.setContentsMargins(14, 12, 14, 12)
        t = QLabel("איך נמדדות הזוויות"); t.setObjectName("PanelTitle"); rl.addWidget(t)
        rl.addWidget(AngleSketch())
        expl = QLabel("הכיוון ההתחלתי = מנקודת ההתחלה לקצה שסימנת בהגדרה (הקו התכלת המקווקו בתמונות).<br>"
                      "<b>θ חדש</b> – כיוון קו שמותאם לחלק שגדל אחרי קצה תמונת הבסיס (או ל-X המ״מ האחרונים).<br>"
                      "<b>θ כולל</b> – כיוון הקו הישר מנקודת ההתחלה עד הקצה הנוכחי.<br>"
                      "+ = נגד כיוון השעון בתמונה, − = עם כיוון השעון.")
        expl.setWordWrap(True); expl.setObjectName("Muted"); rl.addWidget(expl)
        self.alert_summary = QLabel(""); self.alert_summary.setObjectName("Banner"); self.alert_summary.setWordWrap(True)
        rl.addWidget(self.alert_summary)
        rl.addStretch(1)
        body.addWidget(right, 2, Qt.AlignTop)
        lay.addLayout(body, 1)
        self._alerts_to_widgets()
        for key, wd in aw.items():
            sig = wd.toggled if isinstance(wd, QCheckBox) else (
                wd.currentIndexChanged if isinstance(wd, QComboBox) else wd.valueChanged)
            sig.connect(lambda *_, key=key: self._alert_widget_changed(key))
        return w

    def _alerts_to_widgets(self):
        aw, a = self.alert_w, self.alerts
        for key, wd in aw.items():
            wd.blockSignals(True)
            if isinstance(wd, QCheckBox):
                wd.setChecked(bool(a[key]))
            elif isinstance(wd, QComboBox):
                wd.setCurrentIndex(max(0, wd.findData(a[key])))
            else:
                wd.setValue(float(a[key]))
            wd.blockSignals(False)
        self._alerts_enable()

    def _alerts_enable(self):
        aw = self.alert_w
        aw["diff_thr"].setEnabled(aw["diff_on"].isChecked()); aw["diff_mode"].setEnabled(aw["diff_on"].isChecked())
        aw["ang_new_thr"].setEnabled(aw["ang_new_on"].isChecked())
        aw["ang_all_thr"].setEnabled(aw["ang_all_on"].isChecked())

    def _alert_widget_changed(self, key):
        wd = self.alert_w[key]
        old_last = self.alerts["ang_last_mm"]
        if isinstance(wd, QCheckBox):
            self.alerts[key] = wd.isChecked()
        elif isinstance(wd, QComboBox):
            self.alerts[key] = wd.currentData()
        else:
            self.alerts[key] = float(wd.value())
        self._alerts_enable()
        self._alerts_changed(recompute=(key == "ang_last_mm" and self.alerts["ang_last_mm"] != old_last))

    def _alerts_changed(self, recompute=False):
        self.settings.setValue("alerts", json.dumps(self.alerts))
        if recompute and self.R["k"]:
            self._full_refresh()
        self.model.headers_changed()
        self.model.all_changed()
        self._apply_columns()
        if 0 <= self.cur_row < len(self.R["k"]):
            self._update_diff_lbl(self.cur_row)
            for i in self._active():
                res = self.R["res"][self.cur_row][i]
                ang = self.R["ang"][self.cur_row][i][0]
                self.sv[i].set_angle(ang, self.alerts["ang_new_on"] and np.isfinite(ang)
                                     and abs(ang) > self.alerts["ang_new_thr"])
        self._update_alert_summary()
        self._update_stats()

    def _update_alert_summary(self):
        n = len(self.R["k"])
        if not n:
            self.alert_summary.setText("עדיין אין מדידות. הספים יחולו על כל התמונות – גם על מה שכבר נמדד.")
            return
        cnt = {C_DIFF: 0, C_THA: 0, C_THB: 0, C_THA0: 0, C_THB0: 0}
        rows = 0
        for r in range(n):
            fl = self.alert_cells(r)
            rows += bool(fl)
            for c in fl:
                cnt[c] += 1
        self.alert_summary.setText(
            f"בתוצאות הנוכחיות ({n} תמונות): <b>{rows}</b> שורות עם חריגה  ·  "
            f"הפרש A−B: {cnt[C_DIFF]}  ·  θ חדש: {cnt[C_THA] + cnt[C_THB]}  ·  θ כולל: {cnt[C_THA0] + cnt[C_THB0]}")

    def _diff(self, r):
        R = self.R
        if not self.use_b or r >= len(R["k"]):
            return np.nan
        aA, aB = R["aA"][r], R["aB"][r]
        if self.alerts["diff_mode"] == "a":
            return aA - aB
        return (aA - self.refA) - (aB - self.refB)

    def alert_cells(self, r):
        """Columns of row r whose value crosses an alert threshold."""
        a, out = self.alerts, set()
        if r >= len(self.R["k"]):
            return out
        if a["diff_on"] and self.use_b:
            v = self._diff(r)
            if np.isfinite(v) and abs(v) > a["diff_thr"]:
                out.add(C_DIFF)
        ang = self.R["ang"][r]
        for i, (cn, ct) in enumerate(((C_THA, C_THA0), (C_THB, C_THB0))):
            if i == 1 and not self.use_b:
                continue
            if a["ang_new_on"] and np.isfinite(ang[i][0]) and abs(ang[i][0]) > a["ang_new_thr"]:
                out.add(cn)
            if a["ang_all_on"] and np.isfinite(ang[i][1]) and abs(ang[i][1]) > a["ang_all_thr"]:
                out.add(ct)
        return out

    def _update_diff_lbl(self, r):
        if not self.use_b or r < 0 or r >= len(self.R["k"]):
            self.diff_lbl.hide(); return
        v = self._diff(r)
        if not np.isfinite(v):
            self.diff_lbl.hide(); return
        alert = C_DIFF in self.alert_cells(r)
        bg, fg = ("#fecaca", "#991b1b") if alert else PILL["none"]
        name = "a" if self.alerts["diff_mode"] == "a" else "Δa"
        self.diff_lbl.setStyleSheet(f"background:{bg}; color:{fg};")
        self.diff_lbl.setText(f"הפרש {name} A−B:  {_ltr(f'{v:+.3f}')} מ״מ"); self.diff_lbl.show()

    def change_run_sens(self, i, v):
        """Sensitivity change during a run: applies to the images that have not been measured yet."""
        if not (self.worker is not None and self.worker.isRunning()):
            return
        prep = self.sides[i]["prep"]
        old = prep.get("sens_now", 0)
        if v == old:
            return
        noise = prep.get("noise_level", 0.0)
        thr = prep.get("thr_eff", prep["P"]["contrast_thr"])
        prep["thr_eff"] = noise + (thr - noise) * cc.thr_scale({"sensitivity": v}) / cc.thr_scale({"sensitivity": old})
        prep["sens_now"] = v
        prep["fitted"] = False
        k_next = len(self.R["k"]) + 1
        self.sens_events.append((k_next, i, v))
        self._plot_dirty = True
        self.log(f"צד {SIDE_NAMES[i]}: רגישות {old:+d} → {v:+d}, חלה מתמונה {k_next} ואילך.")
        self.toast(f"צד {SIDE_NAMES[i]}: רגישות {v:+d} – מתמונה {k_next} ואילך")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _new_side():
        return dict(folder="", files=[], base=None, base_file="", calib_img=None, view="base",
                    cal=[], mm_per_px=np.nan, root=None, seed=None, roi=None, roi_user=False,
                    prep=None, msg="לא סומן")

    def _reset_results(self):
        if hasattr(self, "model"):
            self.model.beginResetModel()
        self.R = dict(k=[], file=[], fileB=[], aA=[], aB=[], a=[], da=[], status=[], res=[], orig=[], export=[],
                      ang=[], sens=[])
        self.a_ref = self.refA = self.refB = np.nan
        self.cur_row = -1
        if hasattr(self, "model"):
            self.model.endResetModel()

    def log(self, msg):
        line = time.strftime("%H:%M:%S  ") + msg
        if hasattr(self, "logbox"):
            self.logbox.appendPlainText(line)
        try:
            print(line)
        except Exception:
            pass

    def set_hint(self, msg, action=False):
        self.hint.setText(msg)
        self.hint.setProperty("kind", "action" if action else "")
        _repolish(self.hint)
        self.hint.setVisible(bool(msg))

    def set_setup_hint(self, msg, action=False):
        self.setup_hint.setText(msg)
        self.setup_hint.setProperty("kind", "action" if action else "")
        _repolish(self.setup_hint)

    def _active(self):
        return [0, 1] if self.use_b else [0]

    def _busy(self):
        return any(t is not None and t.isRunning() for t in (self.worker, self.remeasure))

    def _update_ready(self):
        ready = all(self.sides[i]["prep"] for i in self._active())
        self.btn_go.setEnabled(ready)
        self.nav_btns[PAGE_SETUP].setText("1   הגדרה  ✓" if ready else "1   הגדרה")
        for i in (0, 1):
            s = self.sides[i]
            self.panels[i].set_done(bool(s["folder"]) and s["base"] is not None, bool(np.isfinite(s["mm_per_px"])),
                                    bool(s["prep"]))
        return ready

    # ------------------------------------------------------------------ side setup
    def set_use_b(self, on):
        self.use_b = bool(on)
        p = self.panels[1]
        if p.use.isChecked() != self.use_b:
            p.use.setChecked(self.use_b)
        for c in p.controls():
            c.setEnabled(self.use_b)
        self.sv[1].setVisible(self.use_b)
        self.phone_arm()
        self._apply_columns()
        self.diff_lbl.setVisible(False)
        self._update_ready()

    def _apply_columns(self):
        for tv in (self.compact, self.full):
            for c in B_COLS:
                tv.setColumnHidden(c, not self.use_b)
        for c in (C_THA0, C_THB0):
            self.compact.setColumnHidden(c, (not self.alerts["ang_all_on"]) or (c == C_THB0 and not self.use_b))

    def pick_folder(self, i):
        start = self.sides[i]["folder"] or self.sides[0]["folder"] or \
            self.settings.value("last_folder", os.path.expanduser("~"))
        f = QFileDialog.getExistingDirectory(self, f"תיקיית תמונות לצד {SIDE_NAMES[i]}", start)
        if f:
            self.settings.setValue("last_folder", os.path.dirname(f))
            self.set_folder(i, f)

    def set_folder(self, i, folder, base_file=None):
        files = cc.list_images(folder)
        if not files:
            self.toast("אין תמונות (jpg/png/tif/bmp) בתיקייה הזו.", "warn"); return
        s = self.sides[i]
        s.update(folder=folder, files=files, view="base", prep=None)
        s["base_file"] = base_file if base_file and os.path.exists(base_file) else files[0]
        try:
            s["base"] = cc.read_gray(s["base_file"])
        except Exception as ex:
            QMessageBox.warning(self, APP_NAME, f"לא ניתן לקרוא את התמונה:\n{ex}"); return
        if not s["roi_user"]:
            s["roi"] = None
        self.panels[i].prev_spin.setRange(1, len(files)); self.panels[i].prev_spin.setValue(len(files))
        self.panels[i].prev_info.setText("")
        self.panels[i].lbl_folder.setText(
            f"<b>{len(files)}</b> תמונות · בסיס: {_ltr(os.path.basename(s['base_file']))}<br>"
            f"<span style='color:#94a3b8'>{_ltr(folder)}</span>")
        self.log(f"צד {SIDE_NAMES[i]}: {len(files)} תמונות ב-{folder}")
        if i == 0:
            self.phone_arm()
        self._prepare(i)
        self.show_side(i, "base")
        self.su.view.fit_all()
        if not s["cal"]:
            self.set_setup_hint("שלב 2: הקלד מרחק ידוע ולחץ \"סמן 2 קצוות\", ואז לחץ על שני קצות המרחק בתמונה. "
                                "גלגלת = זום, גרירה = הזזה.")

    def pick_calib_image(self, i):
        if i == 1 and self.b_phone:
            self.phone_capture("calib"); return
        start = self.sides[i]["folder"] or os.path.expanduser("~")
        f, _ = QFileDialog.getOpenFileName(self, "תמונה עם מרחק ידוע", start,
                                           "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)")
        if not f:
            return
        self.sides[i]["calib_img"] = cc.read_gray(f)
        self.show_side(i, "calib")
        self.su.view.fit_all()
        self.set_setup_hint("מוצגת תמונת הכיול. הקלד את המרחק הידוע ולחץ \"סמן 2 קצוות\".")

    def add_calib(self, i, p1, p2, mm):
        if np.hypot(p2[0] - p1[0], p2[1] - p1[1]) < 2:
            self.set_setup_hint("הנקודות קרובות מדי – נסה שוב.", True)
            return
        self.sides[i]["cal"].append((tuple(p1), tuple(p2), float(mm)))
        self._update_scale(i)

    def remove_calib(self, i):
        if self.sides[i]["cal"]:
            self.sides[i]["cal"].pop()
            self._update_scale(i)

    def _update_scale(self, i):
        s, p = self.sides[i], self.panels[i]
        p.cal_table.setRowCount(0)
        if not s["cal"]:
            s["mm_per_px"] = np.nan
            p.lbl_scale.setText("לא נקבע")
        else:
            pxmm = []
            for (a, b, mm) in s["cal"]:
                px = float(np.hypot(b[0] - a[0], b[1] - a[1]))
                pxmm.append(px / mm)
                r = p.cal_table.rowCount(); p.cal_table.insertRow(r)
                for c, val in enumerate((f"{px:.1f}", f"{mm:g}", f"{px / mm:.3f}")):
                    it = QTableWidgetItem(val); it.setTextAlignment(Qt.AlignCenter)
                    p.cal_table.setItem(r, c, it)
            pxmm = np.array(pxmm)
            s["mm_per_px"] = 1.0 / pxmm.mean()
            txt = f"{pxmm.mean():.3f} px/mm"
            if len(pxmm) > 1:
                txt += f"   (±{100 * pxmm.std(ddof=1) / pxmm.mean():.1f}%, {len(pxmm)} מרחקים)"
            p.lbl_scale.setText(txt)
        if not s["roi_user"]:
            s["roi"] = None
        self._prepare(i)
        self.redraw()

    def set_crack(self, i, start_pt, tip):
        s = self.sides[i]
        s["root"], s["seed"] = tuple(start_pt), tuple(tip)
        if not s["roi_user"]:
            s["roi"] = None
        self._prepare(i)
        self.show_side(i, "base")
        self._zoom_to_crack(i)

    def _params(self):
        P = cc.default_params()
        P["contrast_thr"] = self.thr.value()
        P["auto_threshold"] = self.thr_auto.isChecked()
        P["search_ahead_mm"] = self.ahead.value()
        P["use_prior"] = self.chk_prior.isChecked()
        P["check_tip"] = self.chk_check.isChecked()
        return P

    def _prepare(self, i):
        s = self.sides[i]
        s["prep"] = None
        ok = False
        if s["base"] is None:
            msg = "בחר תיקייה."
        elif not np.isfinite(s["mm_per_px"]):
            msg = "קבע קנה מידה (שלב 2)."
        elif s["seed"] is None:
            msg = "סמן נקודת התחלה וקצה סדק (שלב 3)."
        else:
            try:
                P = self._params(); P["sensitivity"] = self.panels[i].sens.value()
                d = dict(mm_per_px=s["mm_per_px"], seed=s["seed"], root=s["root"], ref=s["root"],
                         base=s["base"], P=P, roi=s["roi"] if s["roi_user"] else None,
                         base_file=s["base_file"])
                prep = cc.prepare_side(d)
                prep.update(index=i, folder=s["folder"], files=s["files"])
                prep["thr_eff0"] = prep.get("thr_eff")
                s["prep"], s["roi"] = prep, prep["roi"]
                a0 = prep["a0_px"] * prep["mm_per_px"]
                msg = f"מוכן · בתמונת הבסיס a = {a0:.3f} מ״מ (מנקודת ההתחלה עד קצה הסדק)"
                ok = True
            except ValueError as ex:
                msg = str(ex)
        s["msg"] = msg
        lbl = self.panels[i].lbl_crack
        lbl.setText(msg)
        lbl.setStyleSheet("color:#15803d; font-weight:600" if ok else "color:#b91c1c")
        self._update_ready()

    def _params_changed(self):
        if self._busy():
            self.toast("השינוי יחול במדידה הבאה (מדידה רצה כרגע).", "warn")
            return
        for i in (0, 1):
            if self.sides[i]["base"] is not None:
                self._prepare(i)
        self.redraw()

    def preview_image(self, i, k):
        """Measures image #k of side i with the current setup and shows it on the setup screen."""
        s = self.sides[i]
        if not s["prep"]:
            self.toast(s["msg"], "warn"); return
        k = int(np.clip(k, 0, len(s["files"]) - 1))
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            r = cc.measure_frame(s["files"][k], s["prep"], keep_image=True)
        except Exception as ex:
            QApplication.restoreOverrideCursor()
            self.toast(f"לא ניתן לקרוא את התמונה: {ex}", "error"); return
        QApplication.restoreOverrideCursor()
        self._goto(PAGE_SETUP)
        if self.setup_side != i or s.get("preview") is None:
            self.su.pair.fitted = False
        self.setup_side = i
        self._render(self.su.pair, i, r, r.img)
        s["preview"] = k
        a0 = s["prep"]["a0_px"] * s["prep"]["mm_per_px"]
        self.su.title.setText(f"צד {SIDE_NAMES[i]} · תצוגה מקדימה · תמונה {k + 1}")
        self.su.info.setText(f"{_ltr(os.path.basename(r.file))}  ·  a = {_fmt(r.a)} מ״מ  ·  Δa = {_fmt(r.da)}"
                             f"  ·  {status_he(r.status)}")
        self.panels[i].prev_info.setText(f"תמונה {k + 1}:  a = {_fmt(r.a)} מ״מ  (בבסיס {a0:.3f})  ·  "
                                         f"{status_he(r.status)}")

    def test_last(self, i):
        self.preview_image(i, len(self.sides[i]["files"]) - 1)

    def apply_sensitivity(self, i):
        s = self.sides[i]
        if self._busy():
            self.toast("השינוי יחול במדידה הבאה (מדידה רצה כרגע).", "warn"); return
        if s["base"] is None:
            return
        k = s.get("preview")
        self._prepare(i)
        if s["prep"] and k is not None:
            self.preview_image(i, k)
        else:
            self.show_side(i, "base")

    # ------------------------------------------------------------------ side B from the phone
    def set_b_phone(self, on):
        on = bool(on)
        if on == self.b_phone:
            return
        p = self.panels[1]
        if self._busy():
            self.toast("עצור קודם את המדידה.", "warn")
            (p.rb_phone if self.b_phone else p.rb_folder).setChecked(True); return
        self.b_phone = on
        p.btn_folder.setVisible(not on); p.lbl_folder.setVisible(not on); p.phone_box.setVisible(on)
        p.btn_cal_img.setText("צלם תמונת כיול" if on else "תמונת כיול אחרת...")
        self.sides[1] = self._new_side()             # another camera: folder, scale and marks start over
        p.lbl_folder.setText("לא נבחרה תיקייה"); p.lbl_bfolder.setText("לא נבחרה תיקייה")
        p.prev_spin.setRange(1, 1); p.prev_info.setText("")
        if on:
            self._phone().start()
            self.set_setup_hint("צד B מהטלפון: בחר תיקיית שמירה, הפעל \"תצוגה חיה\" לכיוון הטלפון, "
                                "\"כוון פוקוס ונעל\", ואז \"צלם תמונת בסיס\". הכיול והסימון – כרגיל.")
            self.log("צד B: מקור = טלפון (טריגר מצד A)")
        else:
            p.btn_plive.setChecked(False)
            if self.phone is not None:
                self.phone.stop()
            self.sb_phone.hide()
            self.log("צד B: מקור = תיקייה")
        self._update_scale(1)
        self.show_side(1, "base")

    def _phone(self):
        if self.phone is None:
            ph = self.phone = phone_b.PhoneB(self)
            ph.status.connect(self._phone_status)
            ph.alert.connect(self._phone_alert)
            ph.log.connect(self.log)
            ph.preview.connect(self._phone_preview)
            ph.setup_captured.connect(self._phone_setup_captured)
            ph.triggered.connect(self._phone_triggered)
            ph.focus_progress.connect(self.panels[1].lbl_pfocus.setText)
            ph.focus_done.connect(self._phone_focus_done)
            ph.changed.connect(self._phone_ui)
            self.phone_params()
        return self.phone

    def phone_params(self, *_):
        p = self.panels[1]
        if self.phone is not None:
            self.phone.timeout_s = p.ph_timeout.value()
            self.phone.burst = p.ph_burst.value()

    def phone_arm(self):
        """The trigger runs whenever side B is the phone and both folders are set (measuring or not),
        so every side-A image gets its side-B partner."""
        ph = self.phone
        if ph is None:
            return
        if self.b_phone and self.use_b and self.panels[1].chk_trig.isChecked() and \
                self.sides[0]["folder"] and self.sides[1]["folder"]:
            if not (ph.armed and ph.folder_a == self.sides[0]["folder"] and ph.folder_b == self.sides[1]["folder"]):
                ph.arm(self.sides[0]["folder"], self.sides[1]["folder"])
        else:
            ph.disarm()
        self._phone_ui()

    def pick_b_folder(self):
        a = self.sides[0]["folder"]
        start = self.sides[1]["folder"] or (os.path.dirname(a) if a else
                                            self.settings.value("last_folder", os.path.expanduser("~")))
        f = QFileDialog.getExistingDirectory(self, "תיקייה לשמירת תמונות צד B מהטלפון", start)
        if f:
            self.set_phone_folder(f)

    def set_phone_folder(self, folder):
        if self.sides[0]["folder"] and os.path.normcase(os.path.abspath(folder)) == \
                os.path.normcase(os.path.abspath(self.sides[0]["folder"])):
            self.toast("תיקיית צד B חייבת להיות שונה מתיקיית צד A.", "warn"); return
        s = self.sides[1]
        s["folder"], s["files"] = folder, cc.list_images(folder)
        self._phone().folder_b = folder
        self.panels[1].lbl_bfolder.setText(f"<b>{len(s['files'])}</b> תמונות כבר בתיקייה<br>"
                                           f"<span style='color:#94a3b8'>{_ltr(folder)}</span>")
        if s["files"]:
            self.panels[1].prev_spin.setRange(1, len(s["files"])); self.panels[1].prev_spin.setValue(len(s["files"]))
        self.log(f"צד B (טלפון): התמונות יישמרו ב-{folder}")
        self.phone_arm()
        self._prepare(1)

    def install_phone_app(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ok, msg = fc_link.install_app()
        finally:
            QApplication.restoreOverrideCursor()
        self.toast(msg, "ok" if ok else "error", 6000)
        self.log("התקנת אפליקציית הטלפון: " + msg)

    def _phone_alert(self, text):
        self.toast(text, "warn", 6000)
        self.log(text)

    def _phone_status(self, level, text):
        ph, p = self.phone, self.panels[1]
        col = {"ok": "#15803d", "warn": "#b45309", "err": "#b91c1c"}.get(level, "#334155")
        if level == "ok" and ph is not None and ph.hello:
            caps = ph.hello.get("caps", {})
            text = f"מחובר · {ph.hello.get('device', '')}"
            fmax = float(caps.get("min_focus_diopter") or 10.0)
            p.ph_focus.blockSignals(True); p.ph_focus.setMaximum(fmax)
            if "focus" in ph.cfg["camera"]:
                p.ph_focus.setValue(ph.cfg["camera"]["focus"])
            p.ph_focus.blockSignals(False)
        p.lbl_conn.setText(f"<span style='color:{col}; font-weight:600'>● {text}</span>")
        self._phone_ui()

    def _phone_ui(self):
        ph, p = self.phone, self.panels[1]
        if ph is None or not self.b_phone:
            return
        sm = ph.summary()
        if not self.use_b:
            arm = "<span style='color:#94a3b8'>הטריגר כבוי (צד B לא בשימוש)</span>"
        elif ph.armed:
            arm = "<span style='color:#15803d; font-weight:600'>● הטריגר פעיל</span>"
            if ph.trig is not None and ph.trig.error:
                arm += f" <span style='color:#b91c1c'>· שגיאה בתיקיית A: {ph.trig.error}</span>"
        elif not p.chk_trig.isChecked():
            arm = "<span style='color:#b45309'>הטריגר מושבת</span>"
        else:
            arm = "<span style='color:#94a3b8'>הטריגר ממתין לתיקיית צד A ולתיקיית שמירה</span>"
        txt = arm + f"<br>טריגרים: <b>{sm['n']}</b> · התקבלו <b>{sm['ok']}</b> · " + \
            (f"<b style='color:#b91c1c'>פספוסים {sm['missed']}</b>" if sm["missed"] else "פספוסים 0") + \
            (f" · בדרך {sm['pending']}" if sm["pending"] else "")
        if sm["ok"]:
            if np.isfinite(sm["exp_med"]):
                if np.isfinite(sm["sexp_med"]):
                    txt += (f"<br><b>מצילום A (לפי שם הקובץ) עד הצילום בטלפון: חציון {sm['sexp_med']:.2f} ש׳, "
                            f"מקס׳ {sm['sexp_max']:.2f}</b>")
                txt += (f"<br>מיצירת תמונת A עד הצילום בטלפון: חציון <b>{sm['exp_med']:.2f}</b> ש׳, "
                        f"מקס׳ <b>{sm['exp_max']:.2f}</b>"
                        f"<br>מסיום כתיבת A עד הצילום: חציון {sm['cexp_med']:.2f} ש׳, מקס׳ {sm['cexp_max']:.2f}")
            txt += f"<br>עד שהתמונה הגיעה למחשב: חציון {sm['recv_med']:.2f} ש׳, מקס׳ {sm['recv_max']:.2f}"
        p.lbl_trig.setText(txt)
        conn = ph.connected
        self.sb_phone.setVisible(self.use_b)
        self.sb_phone.setText(("טלפון ● מחובר" if conn else "טלפון ● מנותק") +
                              (f" · {sm['ok']}/{sm['n']}" if sm["n"] else "") +
                              (f" · פספוסים {sm['missed']}" if sm["missed"] else ""))
        self.sb_phone.setStyleSheet("color:#15803d; font-weight:600" if conn and not sm["missed"]
                                    else "color:#b91c1c; font-weight:600")
        p.btn_pfocus.setText("עצור כיוון" if ph.auto.running else
                             ("כוון פוקוס מחדש" if ph.focus_locked is not None else "כוון פוקוס ונעל"))

    def phone_live(self, on):
        ph = self.phone
        if ph is None:
            return
        ph.want_preview = bool(on)
        if on:
            self._goto(PAGE_SETUP)
            if self.tabs.currentIndex() != 1:
                self.tabs.setCurrentIndex(1)
            self.su.title.setText("צד B · תצוגה חיה מהטלפון")
            self.su.zoom.clear_image("")
            if not ph.connected:
                self.su.view.clear_image("ממתין לטלפון...")
        else:
            self.redraw()

    def _phone_preview(self, q):
        p = self.panels[1]
        if not (self.b_phone and p.btn_plive.isChecked() and self.pick is None and self.tabs.currentIndex() == 1
                and self.stack.currentIndex() == PAGE_SETUP):
            return
        v = self.su.view
        v.clear_overlays()
        v.set_ruler(None)                     # preview pixels are not calibrated
        v.set_qimage(q)
        x, y, w, h = self.phone.roi
        v.add_rect((x * q.width(), y * q.height(), w * q.width(), h * q.height()), "#22c55e")
        sh, br = self.phone.last_metrics
        self.su.info.setText(f"חדות {sh:.0f} · בהירות {br:.0f}/255   (המסגרת הירוקה = אזור הפוקוס)"
                             if np.isfinite(sh) else "")

    def _phone_roi_from_crack(self):
        """Focus / sharpness area of the phone: around the marked crack tip, else the image centre."""
        s = self.sides[1]
        if s["seed"] is None or s["base"] is None:
            return
        h, w = s["base"].shape
        cx, cy = s["seed"][0] / w, s["seed"][1] / h
        rw, rh = 0.3, 0.3
        self.phone.set_roi(float(np.clip(cx - rw / 2, 0, 1 - rw)), float(np.clip(cy - rh / 2, 0, 1 - rh)), rw, rh)

    def phone_focus(self):
        ph = self._phone()
        if ph.auto.running:
            ph.auto.cancel(); self.panels[1].lbl_pfocus.setText("הכיוון בוטל"); self._phone_ui(); return
        if not ph.connected or not ph.hello:
            self.toast("אין חיבור לטלפון.", "warn"); return
        self._phone_roi_from_crack()
        ph.start_focus()
        self._phone_ui()

    def _phone_focus_done(self, ok, msg, res):
        p = self.panels[1]
        p.lbl_pfocus.setText(msg)
        p.lbl_pfocus.setStyleSheet("" if ok else "color:#b91c1c")
        self.log("צד B: " + msg)
        if ok:
            p.ph_focus.blockSignals(True); p.ph_focus.setValue(res["focus"]); p.ph_focus.blockSignals(False)
            self._phone_focus_check()
            self.toast("הפוקוס ננעל.", "ok")
        else:
            self.toast(msg, "error", 6000)

    def phone_focus_manual(self, v):
        if self.phone is None:
            return
        self.phone.set_focus(v)
        self.phone.focus_locked = round(float(v), 3)
        self.panels[1].lbl_pfocus.setText(f"פוקוס ננעל ידנית על {v:.2f} D")
        self._phone_focus_check()
        self._phone_ui()

    def _phone_focus_check(self):
        """The scale (mm/px) is only valid for the focus it was calibrated at."""
        s, f = self.sides[1], self.phone.cfg["camera"].get("focus")
        old = [n for n, k in (("הבסיס", "focus_base"), ("הכיול", "focus_cal")) if s.get(k) is not None and
               f is not None and abs(s[k] - f) > 0.02]
        if old:
            self.toast(f"הפוקוס השתנה מאז צילום {' ו'.join(old)} – צלם מחדש (קנה המידה תלוי בפוקוס).", "warn", 7000)

    def phone_capture(self, purpose):
        ph = self._phone()
        if not self.sides[1]["folder"]:
            self.toast("בחר קודם תיקיית שמירה לצד B.", "warn"); return
        if not ph.connected:
            self.toast("אין חיבור לטלפון.", "warn"); return
        if ph.focus_locked is None:
            self.toast("הפוקוס עוד לא ננעל – מומלץ \"כוון פוקוס ונעל\" לפני הכיול.", "warn", 5000)
        if ph.capture_setup(purpose):
            self.set_setup_hint("מצלם בטלפון...")

    def _phone_setup_captured(self, purpose, path, msg):
        s, p = self.sides[1], self.panels[1]
        if not self.b_phone:
            return
        try:
            img = cc.read_gray(path)
        except Exception as ex:
            self.toast(f"לא ניתן לקרוא את התמונה מהטלפון: {ex}", "error"); return
        p.btn_plive.setChecked(False)
        focus = self.phone.cfg["camera"].get("focus")
        self.log(f"צד B: צולמה תמונת {'בסיס' if purpose == 'base' else 'כיול'} {img.shape[1]}x{img.shape[0]} "
                 f"(חדות {msg.get('sharp') or 0:.0f}, פוקוס {focus}) → {path}")
        self._goto(PAGE_SETUP)
        if purpose == "base":
            self._set_b_base(path, img)
            s["focus_base"] = focus
            self.phone.ref_sharp, self.phone.blurred = msg.get("sharp") or None, 0   # focus watch reference
            if s["seed"] is not None:
                self._zoom_to_crack(1)
            else:
                self.su.view.fit_all()
            self.set_setup_hint("תמונת הבסיס צולמה. שלב 2: הקלד מרחק ידוע ולחץ \"סמן 2 קצוות\" "
                                "(או \"צלם תמונת כיול\" לתמונה נפרדת עם סרגל)." if not s["cal"] else
                                "תמונת הבסיס צולמה. שלב 3: סמן נקודת התחלה וקצה סדק.")
        else:
            s["calib_img"] = img
            s["focus_cal"] = focus
            self.show_side(1, "calib")
            self.su.view.fit_all()
            self.set_setup_hint("מוצגת תמונת הכיול מהטלפון. הקלד את המרחק הידוע ולחץ \"סמן 2 קצוות\".")
        self._update_ready()

    def _set_b_base(self, path, img=None):
        s, p = self.sides[1], self.panels[1]
        s["base_file"] = path
        s["base"] = img if img is not None else cc.read_gray(path)
        s["view"] = "base"
        if not s["roi_user"]:
            s["roi"] = None
        p.lbl_bfolder.setText(f"בסיס: {_ltr(os.path.basename(path))}<br>"
                              f"<span style='color:#94a3b8'>{_ltr(s['folder'])}</span>")
        self._prepare(1)
        self.show_side(1, "base")

    def _phone_triggered(self, r):
        def d(a, b):
            return f"{b - a:.2f}" if a is not None and b is not None else "—"
        self.log(f"צד B: {os.path.basename(r['file_A'])} → {r['status']} · A→צילום {d(r['t_created'], r['t_exposure'])} ש׳"
                 f" · A→התקבלה {d(r['t_created'], r['t_recv'])} ש׳ · פקודה→התקבלה {d(r['t_sent'], r['t_recv'])} ש׳")

    def open_phone_log(self):
        p = self.phone.log_path() if self.phone else ""
        if not p or not os.path.exists(p):
            self.toast("עוד אין לוג טריגרים (הוא נוצר בתיקיית צד B בטריגר הראשון).", "info"); return
        os.startfile(p)

    # ------------------------------------------------------------------ picking / correcting
    def start_pick(self, i, mode):
        s = self.sides[i]
        if s["base"] is None and not (mode == "calib" and s["calib_img"] is not None):
            self.toast("צלם קודם תמונת בסיס מהטלפון." if (i == 1 and self.b_phone) else "בחר קודם תיקיית תמונות.",
                       "warn"); return
        if self._busy():
            self.toast("עצור קודם את המדידה.", "warn"); return
        if i == 1:
            self.panels[1].btn_plive.setChecked(False)
        self._cancel_pick()
        self._goto(PAGE_SETUP)
        if mode == "calib":
            self.show_side(i, s["view"])
            self.set_setup_hint("לחץ על הקצה הראשון של המרחק הידוע  ·  גלגלת = זום  ·  Esc = ביטול", True)
        elif mode == "crack":
            self.show_side(i, "base")
            self.set_setup_hint("1/2  לחץ על נקודת ההתחלה של המדידה – למשל קצה החריץ. "
                                "המדידה תתחיל בדיוק מהנקודה הזו.  ·  Esc = ביטול", True)
        else:
            self.show_side(i, "base")
            self.set_setup_hint("אזור יישור: לחץ על פינה אחת של אזור עם טקסטורה (רחוק מהסדק), ואז על הפינה הנגדית",
                                True)
        self.pick = dict(mode=mode, side=i, pts=[])
        self.su.pair.set_pick(True)

    def _cancel_pick(self):
        self.pick = None
        self.su.pair.set_pick(False)
        for v in self.sv:
            v.pair.set_pick(False)
            v.btn_fix.setChecked(False)

    def start_fix(self, i):
        if i == 1 and not self.use_b:
            return
        if self.cur_row < 0 or self.cur_row >= len(self.R["res"]) or self.R["res"][self.cur_row][i] is None:
            self.sv[i].btn_fix.setChecked(False)
            self.toast("קודם בחר תמונה שנמדדה.", "warn"); return
        if self.pick and self.pick.get("mode") == "fix" and self.pick["side"] == i:
            self._cancel_pick(); self.set_hint(""); return             # second press = cancel
        self._set_follow(False)          # stay on this image while the measurement continues
        self._cancel_pick()
        self.pick = dict(mode="fix", side=i, pts=[], row=self.cur_row)
        self.sv[i].pair.set_pick(True)
        self.sv[i].btn_fix.setChecked(True)
        self.set_hint(f"תיקון קצה · צד {SIDE_NAMES[i]} · תמונה {self.R['k'][self.cur_row]}:  "
                      f"לחץ על הקצה הנכון של הסדק (מומלץ בחלון ההגדלה). המדידה ממשיכה ברקע.  Esc = ביטול", True)

    def undo_fix(self, i):
        r = self.cur_row
        if r < 0 or r >= len(self.R["res"]):
            return
        if self.R["res"][r][i] is not self.R["orig"][r][i]:
            self.R["res"][r][i] = self.R["orig"][r][i]
            self._update_row(r, full=(r == 0))
            self.show_row(r)
            self.toast(f"התיקון בתמונה {self.R['k'][r]} (צד {SIDE_NAMES[i]}) בוטל.")
            self.log(f"תמונה {self.R['k'][r]}, צד {SIDE_NAMES[i]}: התיקון בוטל.")

    def apply_fix_to_all(self, i):
        r = self.cur_row
        if r < 0 or r >= len(self.R["res"]) or self.R["res"][r][i] is None or \
                self.R["res"][r][i].status != "manual":
            self.toast("קודם תקן את הקצה בתמונה הזו (\"תקן קצה\").", "warn"); return
        if self.remeasure and self.remeasure.isRunning():
            self.toast("מדידה חוזרת כבר רצה – המתן לסיומה.", "warn"); return
        prep = self.sides[i]["prep"]
        man, orig = self.R["res"][r][i], self.R["orig"][r][i]
        d = orig.d if orig.d is not None and np.all(np.isfinite(orig.d)) else (0.0, 0.0)
        R = self.R

        def file_of(row, R=R):
            if row >= len(R["orig"]) or R["orig"][row][i] is None:
                return None
            return R["orig"][row][i].file

        self._fix_info = (i, self.R["k"][r], prep.get("thr_eff", prep["P"]["contrast_thr"]), man.a)
        self.remeasure = RemeasureWorker(i, prep, (man.file, prep, d, man.a), file_of, lambda R=R: len(R["k"]),
                                         lambda: self.worker is not None and self.worker.isRunning())
        self.remeasure.thr_found.connect(self._on_thr_found)
        self.remeasure.row_done.connect(self._on_remeasured)
        self.remeasure.progress.connect(lambda n, t: self._progress(n, t, "מדידה חוזרת"))
        self.remeasure.finished.connect(self._on_remeasure_finished)
        self.sv[i].btn_all.setEnabled(False)
        self._set_state("remeasure", "מחפש סף שמתאים לתיקון")
        self.set_hint(f"צד {SIDE_NAMES[i]}: מחפש סף זיהוי שמתאים לתיקון... (ברקע – אפשר להמשיך לעבוד)")
        self.remeasure.start()

    def _on_thr_found(self, thr, a_got):
        i, k, old, a_man = self._fix_info
        if thr is None:
            self.set_hint("")
            QMessageBox.warning(self, APP_NAME,
                                "לא נמצא סף זיהוי שמביא את הקצה האוטומטי לנקודה שתיקנת "
                                "(כנראה שאין שם ראיה ברורה לסדק בתמונה).\n\nהתיקון נשאר רק בתמונה הזו.")
            return
        msg = (f"צד {SIDE_NAMES[i]}: סף הזיהוי הותאם לתיקון בתמונה {k} ({old:.3f} → {thr:.3f}; "
               f"a אוטומטי {a_got:.3f} מול תיקון {a_man:.3f}). מודד מחדש את כל התמונות ברקע...")
        self.log(msg); self.set_hint(msg)

    def _on_remeasured(self, row, i, res):
        if row >= len(self.R["res"]) or res.file != (self.R["orig"][row][i].file if self.R["orig"][row][i] else None):
            return
        self.R["orig"][row][i] = res
        if self.R["res"][row][i] is None or self.R["res"][row][i].status != "manual":
            self.R["res"][row][i] = res
        self.R["sens"][row][i] = "fit"
        self._update_row(row)
        if row == self.cur_row:
            self.show_row(row)

    def _on_remeasure_finished(self):
        i = self._fix_info[0]
        self._full_refresh()
        running = self.worker is not None and self.worker.isRunning()
        self._set_state(("live" if self.worker.live else "run") if running else "idle")
        self.log("המדידה החוזרת הסתיימה.")
        if self.hint.text().startswith(f"צד {SIDE_NAMES[i]}: "):
            self.set_hint("")
            self.toast(f"צד {SIDE_NAMES[i]}: כל התמונות עודכנו לפי התיקון.", "ok", 5000)

    def _on_fix_click(self, i, x, y):
        if not self.pick or self.pick["mode"] != "fix" or self.pick["side"] != i:
            return
        r = self.pick["row"]
        self._cancel_pick()
        res = self.R["orig"][r][i]
        new = cc.correct_tip(res, self.sides[i]["prep"], (x, y))
        self.R["res"][r][i] = new
        self._update_row(r, full=(r == 0))
        self.show_row(r)
        self.set_hint(f"תוקן · תמונה {self.R['k'][r]} · צד {SIDE_NAMES[i]}:  a = {new.a:.3f} מ״מ "
                      f"(אוטומטי {_fmt(res.a)}).   \"החל תיקון על כולם\" יתאים את שאר התמונות.   "
                      f"End = חזרה לתמונה האחרונה.")
        self.log(f"תמונה {self.R['k'][r]}, צד {SIDE_NAMES[i]}: קצה תוקן ידנית, a = {new.a:.3f} מ\"מ")

    def _on_setup_click(self, x, y):
        if not self.pick or self.pick["mode"] == "fix":
            return
        i, mode = self.pick["side"], self.pick["mode"]
        self.pick["pts"].append((x, y))
        self.su.view.add_marker(x, y, "+", "cyan" if mode != "crack" else C_REF, 8, 2)
        if len(self.pick["pts"]) < 2:
            self.set_setup_hint({"calib": "עכשיו לחץ על הקצה השני של המרחק",
                                 "crack": "2/2  עכשיו לחץ על קצה הסדק בתמונה הזו – התוכנה תעקוב אחריו בכל התמונות",
                                 "roi": "עכשיו לחץ על הפינה הנגדית"}[mode], True)
            return
        pts = self.pick["pts"]
        self._cancel_pick()
        if mode == "calib":
            self.add_calib(i, pts[0], pts[1], self.panels[i].known.value())
            self.set_setup_hint("המרחק נוסף. אפשר להוסיף עוד מרחק לממוצע מדויק יותר, או לעבור לשלב 3.")
        elif mode == "crack":
            self.set_crack(i, pts[0], pts[1])
            if self.sides[i]["prep"]:
                nxt = ("עכשיו הגדר את צד B, או " if (self.use_b and not self.sides[1 - i]["prep"]) else "")
                self.set_setup_hint(f"צד {SIDE_NAMES[i]} מוכן ✓   + צהוב = נקודת ההתחלה, עיגול אדום = קצה הסדק. "
                                    f"{nxt}לחץ \"המשך למדידה\".  מומלץ: בשלב 4 בדוק תמונה מאוחרת וכוון רגישות.")
            else:
                self.set_setup_hint(self.sides[i]["msg"], True)
        else:
            (x0, y0), (x1, y1) = pts
            s = self.sides[i]
            s["roi"] = (int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0)))
            s["roi_user"] = True
            self._prepare(i)
            self.show_side(i, "base")
            self.set_setup_hint("אזור היישור עודכן.")

    # ------------------------------------------------------------------ display
    def show_side(self, i, view="base"):
        s = self.sides[i]
        if view == "calib" and s["calib_img"] is not None:
            img = s["calib_img"]
        else:
            view, img = "base", s["base"]
        s["view"] = view
        s["preview"] = None
        self.setup_side = i
        if self.tabs.currentIndex() != i:
            self.tabs.blockSignals(True); self.tabs.setCurrentIndex(i); self.tabs.blockSignals(False)
        self.su.title.setText(f"צד {SIDE_NAMES[i]} · " + ("תמונת כיול" if view == "calib" else "תמונת בסיס"))
        if img is None:
            self.su.view.clear_image(f"בחר תיקיית תמונות לצד {SIDE_NAMES[i]}")
            self.su.zoom.clear_image(""); self.su.info.setText("")
            return
        self.su.info.setText(_ltr(os.path.basename(s["base_file"])) if view == "base" else "")
        self.su.view.set_image(img)
        self.su.view.set_ruler(s["mm_per_px"])
        self._draw_setup(i, img)

    def redraw(self):
        self.show_side(self.setup_side, self.sides[self.setup_side]["view"])

    def _draw_setup(self, i, img):
        s, v = self.sides[i], self.su.view
        v.clear_overlays()
        p = s["prep"]
        if s["view"] == "base":
            if s["roi"]:
                v.add_rect(s["roi"], "#d8c000")
            if p:
                i_r = int(np.argmin(np.linalg.norm(np.asarray(p["base_path"]) - np.asarray(p["ref"]), axis=1)))
                meas = np.vstack([np.asarray(p["ref"])[None, :], np.asarray(p["base_path"])[i_r:]])
                v.add_polyline(meas, C_MEAS, 2.0, alpha=170)
                v.add_marker(*p["ref"], "+", C_REF, 8, 2)
                v.add_marker(*p["base_tip_xy"], "circle", C_TIP, 13, 2)
                self.su.pair.show_zoom(img, p["base_tip_xy"], p["mm_per_px"],
                                       lambda z: z.add_marker(*p["base_tip_xy"], "ticks", C_TIP, 10, 2))
            else:
                self.su.zoom.clear_image("ההגדלה תופיע אחרי סימון קצה הסדק")
                if s["root"] is not None:
                    v.add_marker(*s["root"], "+", C_REF, 8, 2)
                    v.add_marker(*s["seed"], "+", "yellow", 7, 1.5)
        if s["cal"]:
            for a, b, _ in s["cal"]:
                v.add_polyline([a, b], "cyan", 1.5)
                v.add_marker(*a, "+", "cyan", 6, 1.5); v.add_marker(*b, "+", "cyan", 6, 1.5)

    def _render(self, pair, i, res, img):
        """Draws one measurement (image, measured line, start point, tip, zoom) into an ImagePair."""
        p = self.sides[i]["prep"]
        st = res.status
        col = C_MAN if st == "manual" else (C_UNC if st.startswith("uncertain") else C_TIP)
        pair.view.set_image(img)
        pair.view.set_ruler(p["mm_per_px"])
        pair.view.clear_overlays()
        pts = [np.asarray(x, float) for x in (res.ref_xy, res.tip_xy) if x is not None and np.all(np.isfinite(x))]
        if pts:
            pts = np.array(pts); k = 1.0 / p["mm_per_px"]
            lo, hi = pts.min(0) - 2.5 * k, pts.max(0) + 2.5 * k
            if not pair.fitted:
                pair.view.zoom_to(lo[0], lo[1], hi[0], hi[1]); pair.fitted = True
            else:
                pair.view.centerOn((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2)
        if res.ref_xy is not None and np.all(np.isfinite(res.ref_xy)) and pts is not None and len(pts) == 2:
            u = np.asarray(p["u"], float); r0 = np.asarray(res.ref_xy, float)
            reach = float(np.dot(np.asarray(res.tip_xy, float) - r0, u)) + 2.0 / p["mm_per_px"]
            pair.view.add_polyline([r0, r0 + u * reach], "#7dd3fc", 1.0, Qt.DashLine, z=5, alpha=150)
        if res.meas_xy is not None and len(res.meas_xy) > 1:
            pair.view.add_polyline(res.meas_xy, C_MEAS, 2.0, z=7, alpha=170)
        if res.ref_xy is not None and np.all(np.isfinite(res.ref_xy)):
            pair.view.add_marker(*res.ref_xy, "+", C_REF, 8, 2)
        has_tip = res.tip_xy is not None and np.all(np.isfinite(res.tip_xy))
        if has_tip:
            pair.view.add_marker(*res.tip_xy, "circle", col, 13, 2.2)
        pair.show_zoom(img, res.tip_xy, p["mm_per_px"],
                       (lambda z: z.add_marker(*res.tip_xy, "ticks", col, 10, 2)) if has_tip else (lambda z: None))

    def _show_result_in(self, i, res, img, label):
        sv = self.sv[i]
        if img is None:
            try:
                img = cc.load_frame(res.file, self.sides[i]["prep"])
            except Exception as ex:
                sv.view.clear_image(f"לא ניתן לקרוא את התמונה:\n{ex}")
                sv.set_metrics(np.nan, np.nan, "unreadable image", label); return
        self._render(sv.pair, i, res, img)
        sv.set_metrics(res.a, res.da, res.status, label)
        try:
            ang = cc.crack_angles(res, self.sides[i]["prep"], self.alerts["ang_last_mm"])[0] if res.ok else np.nan
        except Exception:
            ang = np.nan
        sv.set_angle(ang, self.alerts["ang_new_on"] and np.isfinite(ang) and abs(ang) > self.alerts["ang_new_thr"])
        sv.set_fix_enabled(True, res.status == "manual")

    def _preview_bases(self):
        """Measuring screen before the first run: show each side's base image with its setup."""
        for i in self._active():
            sv, s, p = self.sv[i], self.sides[i], self.sides[i]["prep"]
            if not p:
                sv.view.clear_image(f"צד {SIDE_NAMES[i]} עדיין לא מוכן – השלם את ההגדרה (מסך 1).")
                sv.zoom.clear_image(""); sv.set_metrics(np.nan, np.nan, "", ""); sv.set_fix_enabled(False)
                continue
            i_r = int(np.argmin(np.linalg.norm(np.asarray(p["base_path"]) - np.asarray(p["ref"]), axis=1)))
            meas = np.vstack([np.asarray(p["ref"])[None, :], np.asarray(p["base_path"])[i_r:]])
            a0 = p["a0_px"] * p["mm_per_px"]
            res = cc.FrameResult(s["base_file"], True, "ok", 0.0, 1.0, (0.0, 0.0), p["base_tip_xy"],
                                 meas_xy=meas, a=a0, ref_xy=p["ref"])
            sv.fitted = False
            self._render(sv.pair, i, res, s["base"])
            sv.set_metrics(a0, 0.0, "", f"תמונת בסיס · {_ltr(os.path.basename(s['base_file']))}")
            sv.set_fix_enabled(False)
        if self._update_ready():
            self.set_hint("הכל מוכן. לחץ \"מדוד את כל התמונות\" (F5), או \"מעקב חי\" בזמן ניסוי.")
        else:
            self.set_hint("ההגדרה לא הושלמה – חזור למסך \"הגדרה\".", True)

    def _zoom_to_crack(self, i):
        s = self.sides[i]
        if not np.isfinite(s["mm_per_px"]) or s["seed"] is None:
            return
        seed, root = np.array(s["seed"]), np.array(s["root"])
        u = (seed - root) / max(1e-9, np.linalg.norm(seed - root))
        k = 1.0 / s["mm_per_px"]
        pts = np.array([root - 2 * k * u, seed + 12 * k * u])
        lo, hi = pts.min(0) - 2 * k, pts.max(0) + 2 * k
        self.su.view.zoom_to(lo[0], lo[1], hi[0], hi[1])

    def _tab_changed(self, idx):
        if idx <= 1:
            self.show_side(idx, self.sides[idx]["view"])
            if self.sides[idx]["base"] is not None:
                if self.sides[idx]["seed"] is not None:
                    self._zoom_to_crack(idx)
                else:
                    self.su.view.fit_all()

    # ------------------------------------------------------------------ navigation
    def _set_follow(self, on):
        self.follow = bool(on)
        if self.chk_follow.isChecked() != self.follow:
            self.chk_follow.blockSignals(True); self.chk_follow.setChecked(self.follow)
            self.chk_follow.blockSignals(False)
        if self.follow:
            if self.pick and self.pick.get("mode") == "fix":
                self._cancel_pick(); self.set_hint("")
            if self.R["k"]:
                self.show_row(len(self.R["k"]) - 1)

    def _step(self, d):
        if not self.R["k"]:
            return
        self._set_follow(False)
        self.show_row(int(np.clip(self.cur_row + d, 0, len(self.R["k"]) - 1)))

    def _jump(self, r):
        if self.R["k"]:
            self._set_follow(False)
            self.show_row(int(np.clip(r, 0, len(self.R["k"]) - 1)))

    def _spin_changed(self, v):
        if self._prog_select or not self.R["k"]:
            return
        r = int(np.clip(v - 1, 0, len(self.R["k"]) - 1))
        if r != self.cur_row:
            self._jump(r)

    def _compact_row_changed(self, cur, prev):
        if self._prog_select or not cur.isValid():
            return
        if cur.row() != self.cur_row:
            self._jump(cur.row())

    def _next_check(self):
        n = len(self.R["k"])
        for j in range(1, n + 1):
            r = (self.cur_row + j) % n
            if _needs_check(self.R["status"][r]):
                self._jump(r)
                return
        self.toast("אין תמונות שסומנו לבדיקה.", "ok")

    def _open_row(self, row):
        if 0 <= row < len(self.R["k"]):
            self._goto(PAGE_MEASURE)
            self._jump(row)

    def _select_row_in_tables(self, row, scroll_full=False):
        self._prog_select = True
        try:
            idx = self.model.index(row, C_N)
            self.compact.setCurrentIndex(idx)
            self.compact.scrollTo(idx, QAbstractItemView.EnsureVisible)
            if scroll_full:
                self.full.setCurrentIndex(self.model.index(row, C_N))
                self.full.scrollTo(idx, QAbstractItemView.PositionAtCenter)
            self.spin_img.setValue(row + 1)
        finally:
            self._prog_select = False

    def show_row(self, row):
        if row < 0 or row >= len(self.R["k"]):
            return
        if self.pick and self.pick.get("mode") == "fix" and self.pick["row"] != row:
            self._cancel_pick(); self.set_hint("")
        self.cur_row = row
        self._select_row_in_tables(row, scroll_full=True)
        self._update_diff_lbl(row)
        for i in self._active():
            res = self.R["res"][row][i]
            if res is not None and self.sides[i]["prep"] is not None:
                self._show_result_in(i, res, None, f"תמונה {self.R['k'][row]} · {_ltr(os.path.basename(res.file))}")
        self._plot_dirty = True

    # ------------------------------------------------------------------ measuring
    def start_worker(self, live):
        if self._busy():
            self.toast("מדידה כבר רצה.", "warn"); return
        for i in self._active():
            if not self.sides[i]["prep"]:
                self.toast(f"צד {SIDE_NAMES[i]} לא מוכן: {self.sides[i]['msg']}", "warn")
                self._goto(PAGE_SETUP); return
        if self.R["k"]:
            b = QMessageBox.question(self, APP_NAME, "להתחיל מדידה חדשה?\n\nהתוצאות והתיקונים הנוכחיים יימחקו "
                                     "(אפשר לייצא אותם קודם במסך \"תוצאות וייצוא\").")
            if b != QMessageBox.Yes:
                return
        for i in self._active():
            prep = self.sides[i]["prep"]
            if prep.get("thr_eff0") is not None:   # fresh threshold (undo an earlier 'apply to all')
                prep["thr_eff"] = prep["thr_eff0"]
            else:
                prep.pop("thr_eff", None)
            prep["P"].update(use_prior=self.chk_prior.isChecked(), check_tip=self.chk_check.isChecked())
            prep["sens_now"] = self.panels[i].sens.value()
            prep["fitted"] = False
            sp = self.sv[i].sens
            sp.blockSignals(True); sp.setValue(prep["sens_now"]); sp.blockSignals(False); sp.setEnabled(True)
            self.sides[i]["files"] = cc.list_images(self.sides[i]["folder"])
            prep["files"] = self.sides[i]["files"]
            prep["b_state"] = self.phone.b_state if (i == 1 and self.b_phone and self.phone) else None
        self._goto(PAGE_MEASURE)
        self._cancel_pick()
        self._reset_results()
        self.sens_events = []
        for v in self.sv:
            v.fitted = False
        self._prog_select = True
        self.spin_img.setRange(0, 0)
        self._prog_select = False
        self.lbl_count.setText("/ 0")
        self._set_follow(True)
        self._total = 0 if live else min(len(self.sides[i]["files"]) for i in self._active()
                                         if not self.sides[i]["prep"].get("b_state"))
        self.worker = Worker([self.sides[i]["prep"] for i in self._active()], live, 0)
        self.worker.result.connect(self._on_result)
        self.worker.message.connect(self.log)
        self.worker.finished.connect(self._on_finished)
        self._t0 = time.time()
        self.worker.start()
        self.btn_run.setEnabled(False); self.btn_live.setEnabled(False); self.btn_stop.setEnabled(True)
        self._set_state("live" if live else "run")
        self.set_hint(("מעקב חי: כל תמונה חדשה שנשמרת בתיקייה תימדד, עד ללחיצה על \"עצור\"." +
                       (" צד B: התמונה מהטלפון לכל תמונת A." if self.b_phone and self.use_b else "")) if live else "")
        self.log("מעקב חי התחיל." if live else f"מודד {self._total} תמונות...")

    def stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self._set_state("stop")

    def _on_finished(self):
        self.btn_run.setEnabled(True); self.btn_live.setEnabled(True); self.btn_stop.setEnabled(False)
        for v in self.sv:
            v.sens.setEnabled(False)
        self._plot_dirty = True
        n = len(self.R["k"])
        n_chk = sum(_needs_check(st) for st in self.R["status"])
        dt = time.time() - self._t0
        self._set_state("idle", f"{n} תמונות ב-{dt:.0f} שניות")
        self.sb_rate.setText("")
        self.log(f"הסתיים: {n} תמונות ב-{dt:.0f} שניות. {n_chk} סומנו לבדיקה.")
        if self.hint.text().startswith("מעקב חי"):
            self.set_hint("")
        self.toast(f"המדידה הסתיימה · {n} תמונות" + (f" · {n_chk} לבדיקה (N)" if n_chk else ""),
                   "ok" if not n_chk else "info", 5000)

    def _row_values(self, r):
        res = self.R["res"][r]
        aA = res[0].a if res[0] is not None else np.nan
        aB = res[1].a if (self.use_b and res[1] is not None) else np.nan
        vals = [x for x in ((aA,) if not self.use_b else (aA, aB)) if np.isfinite(x)]
        a = float(np.mean(vals)) if vals else np.nan
        st = " | ".join(res[i].status if res[i] is not None else "" for i in self._active())
        ang = [(np.nan, np.nan), (np.nan, np.nan)]
        for i in self._active():
            p = self.sides[i]["prep"]
            if res[i] is not None and p is not None and res[i].ok:
                try:
                    ang[i] = cc.crack_angles(res[i], p, self.alerts["ang_last_mm"])
                except Exception:
                    pass
        return aA, aB, a, st, ang

    def _update_row(self, r, full=False):
        """Recompute one row (incremental, O(1)). full=True also re-references Δa (first row changed)."""
        aA, aB, a, st, ang = self._row_values(r)
        R = self.R
        R["aA"][r], R["aB"][r], R["a"][r], R["status"][r], R["ang"][r] = aA, aB, a, st, ang
        if full or not np.isfinite(self.refA):
            self.refA = next((x for x in R["aA"] if np.isfinite(x)), np.nan)
        if full or not np.isfinite(self.refB):
            self.refB = next((x for x in R["aB"] if np.isfinite(x)), np.nan)
        if full or not np.isfinite(self.a_ref):
            self.a_ref = next((x for x in R["a"] if np.isfinite(x)), np.nan)
            R["da"] = [x - self.a_ref if np.isfinite(x) else np.nan for x in R["a"]]
            self.model.all_changed()
        else:
            R["da"][r] = a - self.a_ref if np.isfinite(a) else np.nan
            self.model.row_changed(r)
        self._plot_dirty = True

    def _full_refresh(self):
        R = self.R
        for r in range(len(R["k"])):
            R["aA"][r], R["aB"][r], R["a"][r], R["status"][r], R["ang"][r] = self._row_values(r)
        self.a_ref = next((x for x in R["a"] if np.isfinite(x)), np.nan)
        self.refA = next((x for x in R["aA"] if np.isfinite(x)), np.nan)
        self.refB = next((x for x in R["aB"] if np.isfinite(x)), np.nan)
        R["da"] = [x - self.a_ref if np.isfinite(x) else np.nan for x in R["a"]]
        self.model.all_changed()
        self._plot_dirty = True
        if self.cur_row >= 0:
            self.show_row(self.cur_row)

    def _set_all_export(self, on):
        self.R["export"] = [on] * len(self.R["k"])
        self.model.all_changed()
        self._update_stats()

    def _select_flagged(self):
        self.R["export"] = [_kind(st) in ("bad", "unc", "man") or bool(self.alert_cells(r))
                            for r, st in enumerate(self.R["status"])]
        self.model.all_changed()
        self._update_stats()

    def _on_result(self, out):
        try:
            R = self.R
            n = len(R["k"])
            self.model.beginInsertRows(QModelIndex(), n, n)
            R["k"].append(out["k"] + 1); R["file"].append(out["file"][0]); R["fileB"].append(out["file"][1])
            R["res"].append(list(out["res"])); R["orig"].append(list(out["res"])); R["export"].append(True)
            for key in ("aA", "aB", "a", "da"):
                R[key].append(np.nan)
            R["status"].append("")
            R["ang"].append([(np.nan, np.nan), (np.nan, np.nan)])
            R["sens"].append(list(out.get("sens", [None, None])))
            self.model.endInsertRows()
            self._update_row(n)
            self._prog_select = True
            self.spin_img.setRange(1, n + 1)
            self._prog_select = False
            self.lbl_count.setText(f"/ {n + 1}")
            if self.follow:
                self.cur_row = n
                self._select_row_in_tables(n)
                self._update_diff_lbl(n)
                for i in self._active():
                    if out["img"][i] is not None:
                        self._show_result_in(i, out["res"][i], out["img"][i],
                                             f"תמונה {R['k'][-1]} · {_ltr(out['file'][i])}")
        except Exception:
            self.log("שגיאה בעדכון התצוגה: " + traceback.format_exc(limit=1).strip().splitlines()[-1])

    def _periodic(self):
        try:
            if self._plot_dirty and self.stack.currentIndex() == PAGE_MEASURE:
                self._flush_plot()
            n_chk = sum(_needs_check(st) for st in self.R["status"])
            self.btn_next_check.setText(f"  לבדיקה: {n_chk}")
            if self.worker is not None and self.worker.isRunning():
                n = len(self.R["k"]); dt = max(1e-6, time.time() - self._t0)
                rate = n / dt
                txt = f"{n}" + (f" / {self._total}" if self._total else "") + f" תמונות  ·  {rate * 60:.0f} לדקה"
                if self._total and rate > 0:
                    eta = (self._total - n) / rate
                    txt += f"  ·  נותרו כ-{int(eta // 60)}:{int(eta % 60):02d} דק'"
                self.sb_rate.setText(txt)
            if self.stack.currentIndex() == PAGE_RESULTS:
                self._update_stats()
        except Exception:
            pass

    def _update_stats(self):
        R = self.R
        last_a = next((x for x in reversed(R["a"]) if np.isfinite(x)), np.nan)
        last_da = next((x for x in reversed(R["da"]) if np.isfinite(x)), np.nan)
        self.stat["n"].setText(str(len(R["k"])))
        self.stat["a"].setText(_fmt(last_a, 2) or "—")
        self.stat["da"].setText(_fmt(last_da, 2) or "—")
        self.stat["man"].setText(str(sum("manual" in st for st in R["status"])))
        self.stat["chk"].setText(str(sum(_needs_check(st) for st in R["status"])))
        self.stat["exp"].setText(str(sum(R["export"])))
        n_alr = sum(1 for r in range(len(R["k"])) if self.alert_cells(r))
        self.stat["alr"].setText(str(n_alr))
        self.stat["alr"].setStyleSheet("color:#b91c1c" if n_alr else "")

    def _flush_plot(self):
        self._plot_dirty = False
        R, ax = self.R, self.ax
        ax.clear()
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#cbd5e1")
        if R["k"]:
            n = len(R["k"])
            ms = 3.5 if n < 300 else 2 if n < 2000 else 1.2
            ax.plot(R["k"], R["aA"], "o", color=COL_A_PLOT, ms=ms, label="A", rasterized=True)
            if self.use_b:
                ax.plot(R["k"], R["aB"], "o", color=COL_B_PLOT, ms=ms, label="B", rasterized=True)
            man = [j for j, st in enumerate(R["status"]) if "manual" in st]
            if man:
                ax.plot([R["k"][j] for j in man], [R["a"][j] for j in man], "x", color="#9333ea", ms=6,
                        label="manual")
            unc = [j for j, st in enumerate(R["status"]) if _needs_check(st)]
            if unc:
                ax.plot([R["k"][j] for j in unc], [R["a"][j] for j in unc], "o", mfc="none", color="#f59e0b",
                        ms=ms + 2.5, label="check")
            for k_ev, i_ev, v_ev in self.sens_events:
                ax.axvline(k_ev - 0.5, color="#a855f7", lw=0.8, ls=":")
                ax.annotate(f"S{SIDE_NAMES[i_ev]}{v_ev:+d}", (k_ev - 0.5, 1), xycoords=("data", "axes fraction"),
                            fontsize=6, color="#7e22ce", ha="left", va="top")
            if 0 <= self.cur_row < n:
                ax.axvline(R["k"][self.cur_row], color="#64748b", lw=1, ls="--")
            ax.legend(loc="upper left", fontsize=7, ncol=4, frameon=False)
        ax.set_xlabel("image #", fontsize=8, color="#475569"); ax.set_ylabel("a [mm]", fontsize=8, color="#475569")
        ax.tick_params(labelsize=7, colors="#475569"); ax.grid(alpha=0.6, color="#e2e8f0")
        self.canvas.draw_idle()

    def _progress(self, n, total, what):
        for pb in (self.prog, self.sb_prog):
            pb.setMaximum(max(1, total)); pb.setValue(n)
            pb.setFormat(f"{what}: {n}/{total}")
            pb.show()
        if n >= total:
            QTimer.singleShot(2500, self._hide_progress)

    def _hide_progress(self):
        if not ((self.exporter and self.exporter.isRunning()) or (self.remeasure and self.remeasure.isRunning())):
            self.prog.hide(); self.sb_prog.hide()

    # ------------------------------------------------------------------ export
    def _export_rows(self):
        rows = [r for r in range(len(self.R["k"])) if self.R["export"][r]]
        if not rows:
            self.toast("אין שורות מסומנות לייצוא (עמודת \"ייצוא\" בטבלה).", "warn")
        return rows

    def export_annotated(self):
        if not self.R.get("res"):
            self.toast("עדיין לא נמדד כלום.", "warn"); return
        if self.exporter and self.exporter.isRunning():
            self.toast("ייצוא כבר רץ.", "warn"); return
        rows = self._export_rows()
        if not rows:
            return
        start = self.sides[0]["folder"] or os.path.expanduser("~")
        sub = os.path.join(start, getattr(self, "marked_dirname", "CrackTracker_marked"))
        try:
            os.makedirs(sub, exist_ok=True)
        except OSError:
            sub = start
        out = QFileDialog.getExistingDirectory(self, "תיקייה לשמירת התמונות המסומנות", sub)
        if not out:
            return
        srcs = {os.path.normcase(os.path.abspath(self.sides[i]["folder"])) for i in self._active()}
        if os.path.normcase(os.path.abspath(out)) in srcs:
            QMessageBox.warning(self, APP_NAME, "אי אפשר לשמור לתיקיית התמונות עצמה – בהרצה הבאה התמונות "
                                "המסומנות היו נמדדות כאילו הן תמונות ניסוי.\n\n"
                                "בחר תיקייה אחרת (למשל CrackTracker_marked).")
            return
        self._start_export(rows, out)

    def _start_export(self, rows, out):
        jobs = []
        for row in rows:
            for i in self._active():
                res, p = self.R["res"][row][i], self.sides[i]["prep"]
                if res is None or p is None:
                    continue
                base = os.path.splitext(os.path.basename(res.file))[0]
                jobs.append((res, p, f"image #{self.R['k'][row]}  side {SIDE_NAMES[i]}",
                             f"{self.R['k'][row]:05d}_{SIDE_NAMES[i]}_{base}.png"))
        self._export_dir = out
        self.exporter = ExportWorker(jobs, out)
        self.exporter.progress.connect(lambda n, t: self._progress(n, t, "ייצוא תמונות"))
        self.exporter.done.connect(self._export_done)
        self.btn_stop_export.setEnabled(True)
        self.exporter.start()
        self.toast(f"מייצא {len(jobs)} תמונות ברקע – אפשר להמשיך לעבוד.")
        self.log(f"מייצא {len(jobs)} תמונות ל-{out}...")

    def _stop_export(self):
        if self.exporter and self.exporter.isRunning():
            self.exporter.stop()
            self.log("עוצר ייצוא...")

    def _export_done(self, n, err):
        self.btn_stop_export.setEnabled(False)
        self.log(f"נשמרו {n} תמונות מסומנות ב-{self._export_dir}" + (f"  (שגיאה אחרונה: {err})" if err else ""))
        self.toast(f"נשמרו {n} תמונות ב-{self._export_dir}", "ok" if not err else "warn", 6000)

    def export_csv(self):
        if not self.R["k"]:
            self.toast("עדיין לא נמדד כלום.", "warn"); return
        rows = self._export_rows()
        if not rows:
            return
        start = os.path.join(self.sides[0]["folder"] or os.path.expanduser("~"), "crack_results.csv")
        f, _ = QFileDialog.getSaveFileName(self, "שמירת תוצאות", start, "CSV (*.csv)")
        if not f:
            return
        self._write_csv(f, rows)

    def _write_csv(self, f, rows):
        R = self.R
        try:
            with open(f, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow(["image", "file_A", "file_B", "a_mm_sideA", "a_mm_sideB", "a_mm", "delta_a_mm",
                            "diff_AB_mm (" + ("a" if self.alerts["diff_mode"] == "a" else "delta_a") + ")",
                            "angle_new_A_deg", "angle_new_B_deg", "angle_total_A_deg", "angle_total_B_deg",
                            "sensitivity_A", "sensitivity_B", "alerts", "status"])
                names = {C_DIFF: "diff", C_THA: "angle_new_A", C_THB: "angle_new_B",
                         C_THA0: "angle_total_A", C_THB0: "angle_total_B"}
                for i in rows:
                    ang = R["ang"][i]
                    sens = [("" if x is None else x) for x in R["sens"][i]]
                    w.writerow([R["k"][i], R["file"][i], R["fileB"][i] if self.use_b else "", _num(R["aA"][i]), _num(R["aB"][i]),
                                _num(R["a"][i]), _num(R["da"][i]), _num(self._diff(i)),
                                _num(ang[0][0]), _num(ang[1][0]), _num(ang[0][1]), _num(ang[1][1]),
                                sens[0], sens[1] if self.use_b else "",
                                ";".join(names[c] for c in sorted(self.alert_cells(i))), R["status"][i]])
            self.log(f"נשמר: {f} ({len(rows)} שורות)")
            self.toast(f"הטבלה נשמרה · {len(rows)} שורות", "ok")
        except OSError as ex:
            QMessageBox.warning(self, APP_NAME, f"לא ניתן לשמור את הקובץ:\n{ex}")

    def save_setup(self):
        f, _ = QFileDialog.getSaveFileName(self, "שמירת הגדרות", "crack_tracker_setup.json", "JSON (*.json)")
        if not f:
            return
        data = dict(version=VERSION, use_b=self.use_b, thr=self.thr.value(),
                    thr_auto=self.thr_auto.isChecked(), ahead=self.ahead.value(),
                    prior=self.chk_prior.isChecked(), check=self.chk_check.isChecked(),
                    alerts=self.alerts,
                    sides=[dict(folder=s["folder"], base_file=s["base_file"], cal=s["cal"],
                                sens=self.panels[j].sens.value(),
                                root=s["root"], seed=s["seed"],
                                roi=s["roi"] if s["roi_user"] else None, roi_user=s["roi_user"],
                                focus_base=s.get("focus_base"), focus_cal=s.get("focus_cal"))
                           for j, s in enumerate(self.sides)])
        if self.b_phone and self.phone is not None:
            p = self.panels[1]
            data["b_phone"] = True
            data["phone"] = dict(camera=self.phone.cfg["camera"], roi=self.phone.roi, timeout=p.ph_timeout.value(),
                                 burst=p.ph_burst.value(), trigger=p.chk_trig.isChecked(), ref_sharp=self.phone.ref_sharp)
        with open(f, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        self.toast("ההגדרות נשמרו.", "ok")
        self.log(f"ההגדרות נשמרו: {f}")

    def load_setup(self):
        f, _ = QFileDialog.getOpenFileName(self, "טעינת הגדרות", "", "JSON (*.json)")
        if not f:
            return
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        wds = (self.thr, self.thr_auto, self.ahead, self.chk_prior, self.chk_check)
        for wdg in wds:
            wdg.blockSignals(True)
        self.thr.setValue(data.get("thr", 0.2)); self.thr_auto.setChecked(data.get("thr_auto", True))
        self.thr.setEnabled(not self.thr_auto.isChecked())
        self.ahead.setValue(data.get("ahead", 40))
        self.chk_prior.setChecked(data.get("prior", True)); self.chk_check.setChecked(data.get("check", True))
        for wdg in wds:
            wdg.blockSignals(False)
        if isinstance(data.get("alerts"), dict):
            self.alerts.update(data["alerts"])
            self._alerts_to_widgets()
            self._alerts_changed(recompute=False)
        self.set_use_b(data["use_b"])
        p1 = self.panels[1]
        (p1.rb_phone if data.get("b_phone") else p1.rb_folder).setChecked(True)
        if self.b_phone and isinstance(data.get("phone"), dict):
            ph, pd = self._phone(), data["phone"]
            for wdg, key in ((p1.ph_timeout, "timeout"), (p1.ph_burst, "burst"), (p1.chk_trig, "trigger")):
                if key in pd:
                    wdg.blockSignals(True)
                    (wdg.setChecked if wdg is p1.chk_trig else wdg.setValue)(pd[key])
                    wdg.blockSignals(False)
            self.phone_params()
            if pd.get("roi"):
                ph.set_roi(*pd["roi"])
            ph.ref_sharp = pd.get("ref_sharp")             # after set_roi (which clears it)
            ph.set_camera(pd.get("camera") or {})          # sent now, and again on every (re)connect
            if "focus" in ph.cfg["camera"]:
                ph.focus_locked = ph.cfg["camera"]["focus"]
                p1.ph_focus.blockSignals(True); p1.ph_focus.setValue(ph.focus_locked); p1.ph_focus.blockSignals(False)
                p1.lbl_pfocus.setText(f"פוקוס מקובץ ההגדרות: {ph.focus_locked:.2f} D (נעול)")
        for i, sd in enumerate(data["sides"]):
            s = self.sides[i] = self._new_side()
            s["cal"] = [(tuple(a), tuple(b), mm) for a, b, mm in sd["cal"]]
            s["root"] = tuple(sd["root"]) if sd["root"] else None
            s["seed"] = tuple(sd["seed"]) if sd["seed"] else None
            s["roi_user"] = sd.get("roi_user", False)
            s["roi"] = tuple(sd["roi"]) if sd.get("roi") else None
            sl = self.panels[i].sens
            sl.blockSignals(True); sl.setValue(int(sd.get("sens", 0))); sl.blockSignals(False)
            self.panels[i].set_sens_label(sl.value())
            if i == 1 and self.b_phone:
                s["focus_base"], s["focus_cal"] = sd.get("focus_base"), sd.get("focus_cal")
                if sd["folder"]:
                    self.set_phone_folder(sd["folder"])
                    if sd.get("base_file") and os.path.exists(sd["base_file"]):
                        self._set_b_base(sd["base_file"])
            elif sd["folder"] and os.path.isdir(sd["folder"]):
                self.set_folder(i, sd["folder"], sd.get("base_file"))
            self._update_scale(i)
        self.show_side(0, "base")
        self._zoom_to_crack(0)
        self.toast("ההגדרות נטענו.", "ok")
        self.log(f"ההגדרות נטענו: {f}")

    def show_help(self):
        d = QDialog(self); d.setWindowTitle("איך משתמשים"); d.resize(720, 640)
        lay = QVBoxLayout(d)
        tb = QTextBrowser(); tb.setHtml(HELP_HTML)
        lay.addWidget(tb)
        bb = QDialogButtonBox(QDialogButtonBox.Close); bb.rejected.connect(d.reject); lay.addWidget(bb)
        d.exec()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "toast_w") and self.toast_w.isVisible():
            self.toast_w.place()

    def closeEvent(self, e):
        if self.worker is not None and self.worker.isRunning():
            b = QMessageBox.question(self, APP_NAME, "מדידה רצה כרגע. לעצור ולסגור?")
            if b != QMessageBox.Yes:
                e.ignore(); return
        for t in (self.worker, self.remeasure, self.exporter):
            if t is not None and t.isRunning():
                t.stop(); t.wait(5000)
        if self.phone is not None:
            self.phone.stop()
        self.settings.setValue("geometry", self.saveGeometry())
        e.accept()


HELP_HTML = """
<div dir="rtl" style="font-family:'Segoe UI'; font-size:10.5pt; line-height:150%">
<h2 style="color:#0f172a">Crack Tracker – מדריך קצר</h2>
<h3 style="color:#1d4ed8">1 · הגדרה</h3>
<ol>
<li><b>תיקיית תמונות</b> לכל צד. התמונה הראשונה (לפי מספר) היא תמונת הבסיס.</li>
<li><b>קנה מידה:</b> הקלד מרחק ידוע ולחץ על שני קצותיו. אם הסרגל באינצ'ים,
אינץ' אחד = 25.4 מ״מ. ודא את המרחק המסומן בפועל.</li>
<li><b>נקודת התחלה וקצה:</b> לחיצה ראשונה – נקודת ההתחלה של המדידה (למשל קצה החריץ),
לחיצה שנייה – קצה הסדק בתמונת הבסיס.</li>
<li><b>רגישות הזיהוי</b> (לא חובה): כמה דק קו יכול להיות ועדיין להיחשב סדק. בחר תמונה מאוחרת
ב"תצוגה מקדימה" והזז את המחוון – רואים מיד איפה נמצא הקצה. 0 = ברירת המחדל.</li>
</ol>
<h3 style="color:#1d4ed8">צד B מהטלפון (7.0)</h3>
<ul>
<li>בצד B בחר <b>מקור: טלפון (טריגר מצד A)</b>. הטלפון מחובר בכבל USB עם ניפוי באגים; אפליקציית FatigueCam
נפתחת לבד (או "התקן אפליקציה"). שום דבר לא נשמר בטלפון.</li>
<li>הטלפון צריך להיות <b>פתוח (לא נעול)</b> כשהאפליקציה עולה – אחרת אנדרואיד חוסם את המצלמה (תופיע אזהרה).
אחרי שהיא עלתה אפשר לכבות את המסך. בכ-10 השניות הראשונות אחרי כיבוי המסך המצלמה עלולה לפספס –
לכבות את המסך <b>לפני</b> שהמכונה מתחילה.</li>
<li><b>תיקיית שמירה</b> – שם נשמרות תמונות B (בשם של תמונת A, למשל img_0012.tif → img_0012.jpg) ולוג הטריגרים.</li>
<li><b>תצוגה חיה</b> לכיוון הטלפון · <b>כוון פוקוס ונעל</b> – סריקה ונעילה לכל הריצה (ההגדלה לא משתנה) ·
<b>צלם תמונת בסיס</b> / <b>צלם תמונת כיול</b> – ואז הכיול והסימון כרגיל. אחרי שינוי פוקוס – לצלם כיול מחדש.</li>
<li>כל תמונה חדשה בתיקיית A (אחרי שהכתיבה שלה הסתיימה) שולחת פקודת צילום. אם התמונה לא הגיעה תוך הזמן שנקבע –
אזהרה, ובטבלה "אין תמונה מהטלפון" (הזיווג של שאר התמונות לא זז).</li>
<li><b>phone_trigger_log.csv</b> בתיקיית B: זמן יצירת תמונת A, סיום הכתיבה, שליחת הפקודה, רגע הצילום בטלפון
וקבלת התמונה, ופספוסים. A_complete_to_exposure_s = כמה זמן אחרי שתמונת A נכתבה הטלפון צילם (ו-A_created_to_exposure_s – מיצירת הקובץ) – לפי זה קובעים את זמן העצירה.</li>
</ul>
<h3 style="color:#1d4ed8">2 · מדידה</h3>
<ul>
<li><b>מדוד את כל התמונות</b> (F5) או <b>מעקב חי</b> בזמן ניסוי.</li>
<li>שני הצדדים מוצגים יחד, כל אחד עם a, Δa, סטטוס והגדלה של הקצה (גלגלת = זום).
למטה – התוצאות האחרונות והגרף.</li>
<li><b>עקוב אחרי התמונה האחרונה</b> (F): כשמסומן התצוגה עוברת לכל תמונה חדשה. בחירת תמונה
(← →, Home, מספר, לחיצה בטבלה) או "תקן קצה" מכבים אותו – והמדידה ממשיכה ברקע. End חוזר לאחרונה.</li>
<li><b>לבדיקה</b> (N): קופץ לתמונה הבאה שבה הקצה לא בטוח (כתום) או שהמדידה נכשלה (אדום).</li>
</ul>
<h3 style="color:#1d4ed8">תיקון</h3>
<ul>
<li><b>תקן קצה</b> (T לצד A) ואז לחיצה על הקצה הנכון – עדיף בחלון ההגדלה. Esc מבטל.</li>
<li><b>בטל תיקון</b> מחזיר את המדידה האוטומטית.</li>
<li><b>החל תיקון על כולם</b>: התוכנה מתאימה את סף הזיהוי כך שהקצה האוטומטי יגיע לנקודה שתיקנת,
ומודדת מחדש ברקע את כל התמונות של אותו צד (גם תמונות חדשות).</li>
</ul>
<h3 style="color:#1d4ed8">מה נמדד</h3>
<ul>
<li><b>a</b> – אורך הסדק מנקודת ההתחלה עד הקצה, לאורך קו הסדק. <b>Δa</b> – a פחות a של התמונה הראשונה.
בלי סינון – מדידה טהורה.</li>
<li>החיפוש מניח קודם שהסדק לפחות באורך התמונה הקודמת; אם אין שם סדק – יימצא קצר יותר.</li>
<li>כל קצה נבדק גם בסף נמוך/גבוה יותר; אם הוא זז יותר מ-0.8 מ״מ הוא מסומן "לא בטוח".</li>
</ul>
<h3 style="color:#1d4ed8">רגישות בזמן ריצה</h3>
<ul>
<li>בכרטיס של כל צד במסך המדידה יש "רגישות". בזמן ריצה אפשר לשנות אותה – השינוי חל רק על התמונות שעוד
לא נמדדו. השינוי נרשם ביומן, מסומן בגרף (קו סגול מנוקד) ונשמר ב-CSV לכל תמונה.</li>
</ul>
<h3 style="color:#1d4ed8">4 · התראות</h3>
<ul>
<li>בטבלאות תמיד מוצגים: הפרש בין צד A לצד B, זווית החלק החדש של הסדק, וזווית כוללת (מנקודת ההתחלה לקצה).
הזוויות ביחס לכיוון ההתחלתי של הסדק (הקו התכלת המקווקו בתמונה); + = נגד כיוון השעון בתמונה.</li>
<li>במסך "התראות" קובעים ספים. ערך שעובר את הסף נצבע באדום (המשבצת או כל השורה) – בלי חלונות קופצים.</li>
</ul>
<h3 style="color:#1d4ed8">3 · תוצאות וייצוא</h3>
<ul>
<li>הטבלה המלאה וסיכום. עמודת "ייצוא" קובעת אילו שורות ייוצאו; לחיצה כפולה פותחת את התמונה.</li>
<li>ייצוא CSV (Ctrl+Shift+E) ושמירת תמונות מסומנות (Ctrl+E) – ברקע, עם פס התקדמות ועצירה.</li>
</ul>
<p style="color:#64748b">קיצורים: Ctrl+1/2/3/4 מעבר בין מסכים · F11 מסך מלא · F1 עזרה</p>
</div>"""


def make_app(argv):
    if hasattr(Qt, "HighDpiScaleFactorRoundingPolicy"):
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setLayoutDirection(Qt.RightToLeft)
    app.setStyleSheet(QSS)
    return app


def main():
    app = make_app(sys.argv)
    w = MainWindow()
    if w.settings.value("geometry") is None:
        w.showMaximized()
    else:
        w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

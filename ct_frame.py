"""C(T) photos in the machine / from the phone (CT Tracker 2.0).

Every image is first turned so that the notch is on the RIGHT (crack grows to the left, the back edge
is on the left). On that upright image:
1. back edge (left) and the two side edges (top / bottom): straight lines, back half of the specimen
   only (the clevises hide the front half). Scale = side-to-side distance / measured height H.
2. notch slot: dark band ahead of the predicted tip; its centre line and width from cross profiles.
3. notch tip = the DEEPEST point of the slot opening (towards the back edge), searched only inside the
   slot band. On the slot centre line alone the end of a wire-cut notch often shows a bright 'tooth'
   between two lobes - stopping there puts the tip ~0.3-0.5 mm short (a0 too small).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_opening, gaussian_filter, label, map_coordinates, zoom

from ct_core import frame_geometry, line_intersect, refine_line
from ct_detect3 import _hough_lines

ROT_MACHINE = 1          # np.rot90 k for the machine camera: notch at the bottom -> on the right


def upright(img, k):
    return np.ascontiguousarray(np.rot90(img, k)) if k % 4 else img


# ----------------------------------------------------------------------------- edges
def _support(pol, p1, p2, n=60):
    """Edge strength of the right polarity along a segment (pol = magnitude where the polarity is
    right, else 0; normalised to ~1 for a strong edge)."""
    t = np.linspace(0.04, 0.96, n)
    p = p1[None, :] + t[:, None] * (p2 - p1)[None, :]
    d = (p2 - p1) / max(1e-9, np.linalg.norm(p2 - p1)); nrm = np.array([-d[1], d[0]])
    q = np.concatenate([p + o * nrm for o in (-1.5, 0.0, 1.5)])
    v = map_coordinates(pol, [q[:, 1], q[:, 0]], order=1, mode="constant").reshape(3, n)
    return v.max(0)


def _line_strength(pol, p0, d, H, W):
    """Total right-polarity edge strength along a whole (image-clipped) line."""
    t = np.arange(-2.0 * max(H, W), 2.0 * max(H, W), 2.0)
    q = p0[None, :] + t[:, None] * d[None, :]
    ok = (q[:, 0] >= 0) & (q[:, 0] <= W - 1) & (q[:, 1] >= 0) & (q[:, 1] <= H - 1)
    if not ok.any():
        return 0.0
    q = q[ok]
    return float(map_coordinates(pol, [q[:, 1], q[:, 0]], order=1).sum())


def _pts(theta, rho):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c * rho, s * rho]), np.array([-s, c])


def find_edges(img, H_mm=36.0, W125_mm=37.5, max_dim=800):
    """Back edge (left) and side edges (top, bottom) of the upright specimen.
    Returns dict(back, side1, side2 (each ((x,y),(x,y)) full-res), score) or None."""
    H0, W0 = img.shape
    f = min(1.0, max_dim / max(H0, W0))
    if f < 1:
        st = max(1, int(1 / f))                       # box-average first (fast), then resample
        h2, w2 = (H0 // st) * st, (W0 // st) * st
        sm = img[:h2, :w2].reshape(h2 // st, st, w2 // st, st).mean((1, 3))
        f = sm.shape[1] / W0
    else:
        sm = img
    g = gaussian_filter(sm.astype(float), 1.2)
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    lines = _hough_lines(gx, gy, mag, n_lines=40)
    thr = np.percentile(mag, 90)
    magn = np.minimum(mag / max(thr, 1e-9), 1.0)
    pol = dict(r=np.where(gx > 0, magn, 0.0), d=np.where(gy > 0, magn, 0.0), u=np.where(gy < 0, magn, 0.0))
    H, W = mag.shape
    vert, top, bot = [], [], []
    for th, rho, _ in lines:
        p0, d = _pts(th, rho)
        ang = np.degrees(np.arctan2(d[1], d[0])) % 180
        if min(ang, 180 - ang) > 75:          # direction ~vertical
            d = d if d[1] > 0 else -d
            vert.append((_line_strength(pol["r"], p0, d, H, W), p0, d))
        elif min(ang, 180 - ang) < 15:        # ~horizontal
            d = d if d[0] > 0 else -d
            top.append((_line_strength(pol["d"], p0, d, H, W), p0, d))
            bot.append((_line_strength(pol["u"], p0, d, H, W), p0, d))
    vert = [x[1:] for x in sorted(vert, key=lambda x: -x[0])[:8]]
    top = [x[1:] for x in sorted(top, key=lambda x: -x[0])[:8]]
    bot = [x[1:] for x in sorted(bot, key=lambda x: -x[0])[:8]]
    best = None
    for pb, db in vert:
        for pt, dt in top:
            for pbm, dbm in bot:
                P1 = line_intersect(pb, db, pt, dt); P2 = line_intersect(pb, db, pbm, dbm)
                if P1 is None or P2 is None or P2[1] - P1[1] < 0.12 * H:
                    continue
                if not (-0.02 * W < P1[0] < 0.9 * W and -0.02 * H < P1[1] and P2[1] < 1.02 * H):
                    continue
                k = (P2[1] - P1[1]) / H_mm                # px/mm (downscaled)
                L = 0.45 * W125_mm * k
                Q1, Q2 = P1 + dt * L, P2 + dbm * L
                if Q1[0] > 1.02 * W or Q2[0] > 1.02 * W:
                    continue
                # polarity: back edge dark -> bright going right, top dark -> bright going down,
                # bottom bright -> dark going down
                s_back = _support(pol["r"], P1, P2).mean()
                s_top = _support(pol["d"], P1, Q1).mean()
                s_bot = _support(pol["u"], P2, Q2).mean()
                if min(s_back, s_top, s_bot) < 0.35:
                    continue
                # real corners: the back edge stops there. A line inside the specimen (the notch plane,
                # a scratch) taken as a side leaves the back edge going on past the "corner"
                ext = 0.15 * (P2 - P1)
                s_ext = max(_support(pol["r"], P2 + 0.1 * ext, P2 + ext).mean(),
                            _support(pol["r"], P1 - ext, P1 - 0.1 * ext).mean())
                if s_ext > 0.5:
                    continue
                # inside brighter than outside (the specimen is lit, the background is not)
                c = (P1 + P2 + Q1 + Q2) / 4
                inside = map_coordinates(g, [[c[1]], [c[0]]], order=1)[0]
                outside = map_coordinates(g, [[c[1]], [max(0.0, P1[0] - 0.15 * (P2[1] - P1[1]))]], order=1)[0]
                if inside <= outside:
                    continue
                sc = s_back + s_top + s_bot
                if best is None or sc > best[0]:
                    best = (sc, P1, P2, Q1, Q2)
    if best is None:
        return None
    sc, P1, P2, Q1, Q2 = best
    P1, P2, Q1, Q2 = (np.asarray(p, float) / f for p in (P1, P2, Q1, Q2))
    srch = max(4.0, 1.5 / f)
    out = dict(score=float(sc))
    for key, a, b in (("back", P1, P2), ("side1", P1, Q1), ("side2", P2, Q2)):
        a2, b2 = a + 0.08 * (b - a), b - 0.08 * (b - a)
        q1, q2 = refine_line(img, a2, b2, search=srch)
        out[key] = (tuple(map(float, q1)), tuple(map(float, q2)))
    return out


# ----------------------------------------------------------------------------- notch
def _strip(img, origin, ax, nrm, s, t):
    S, T = np.meshgrid(s, t, indexing="ij")
    X = origin[0] + S * ax[0] + T * nrm[0]
    Y = origin[1] + S * ax[1] + T * nrm[1]
    return map_coordinates(img, [Y.ravel(), X.ravel()], order=1, mode="nearest").reshape(S.shape)


def find_slot(img, tip_pred, ax, k, lat_mm=3.0):
    """Notch slot ahead of the predicted tip. ax = crack direction (front -> back, i.e. into the
    specimen). Returns (centre point near the tip, unit axis of the slot, half width px, levels)."""
    ax = np.asarray(ax, float); nrm = np.array([-ax[1], ax[0]])
    tip_pred = np.asarray(tip_pred, float)
    t = np.arange(-lat_mm * k, lat_mm * k + 0.5, 1.0)
    cents, halfs, xs = [], [], []
    lo_l, hi_l = [], []
    for back_mm in np.arange(1.2, 4.01, 0.4):                    # cross profiles in front of the tip
        p0 = tip_pred - ax * back_mm * k
        prof = _strip(img, p0, ax, nrm, np.arange(-0.15 * k, 0.15 * k + 0.5, 1.0), t).mean(0)
        prof = gaussian_filter(prof, 1.0)
        lo, hi = np.percentile(prof, 5), np.percentile(prof, 90)
        if hi - lo < 0.08:
            continue
        dark = prof < (lo + hi) / 2
        lab, n = label(dark)
        if not n:
            continue
        # the dark run nearest to the predicted axis, at least 0.3 mm wide
        best = None
        for j in range(1, n + 1):
            idx = np.nonzero(lab == j)[0]
            w = (idx[-1] - idx[0] + 1)
            if w < 0.3 * k or w > 2.6 * k:
                continue
            c = 0.5 * (t[idx[0]] + t[idx[-1]])
            if best is None or abs(c) < abs(best[0]):
                best = (c, w / 2.0)
        if best is None:
            continue
        cents.append(best[0]); halfs.append(best[1]); xs.append(-back_mm * k)
        lo_l.append(lo); hi_l.append(hi)
    if len(cents) < 3:
        return None
    xs, cents = np.array(xs), np.array(cents)
    med = np.median(cents)
    keep = np.abs(cents - med) <= max(3.0, 0.25 * k)
    if keep.sum() < 3:
        return None
    A = np.column_stack([xs[keep], np.ones(keep.sum())])
    slope, icpt = np.linalg.lstsq(A, cents[keep], rcond=None)[0]
    if abs(slope) > np.tan(np.radians(8)):
        slope = 0.0; icpt = float(np.median(cents[keep]))
    u = ax + slope * nrm; u /= np.linalg.norm(u)
    nu = np.array([-u[1], u[0]])
    c0 = tip_pred + icpt * nrm
    return dict(p=c0, u=u, n=nu, half=float(np.median(np.array(halfs)[keep])),
                dark=float(np.median(lo_l)), bright=float(np.median(hi_l)))


def find_notch_tip(img, tip_pred, ax, k, search_mm=2.5):
    """Notch tip with a dark notch (lit specimen, dark behind it) or, failing that, a bright one
    (light behind the specimen shows through the notch): the bright case is the dark one inverted."""
    tip, slot, note = _notch_tip(img, tip_pred, ax, k, search_mm)
    tip2, slot2, _ = _notch_tip(1.0 - img, tip_pred, ax, k, search_mm)
    if tip2 is not None:
        tip2 = _through_rim(img, tip2, slot2["u"], k)
    if tip2 is None:
        return tip, slot, note
    if tip is None:
        return tip2, slot2, ""
    # both found: the notch lies on the specimen's mid-line - keep the tip nearer to it
    nrm = np.array([-ax[1], ax[0]], float) / max(1e-9, float(np.hypot(*ax)))
    off = lambda p: abs(float(np.dot(np.asarray(p, float) - np.asarray(tip_pred, float), nrm)))
    return (tip, slot, note) if off(tip) <= off(tip2) else (tip2, slot2, "")


def _through_rim(img, tip, u, k):
    """Bright notch: its end is seen through a dark rim (the notch wall). The notch root on the
    surface is where that rim meets the specimen face, a little further towards the back edge."""
    u = np.asarray(u, float)
    sv = np.arange(0.0, 1.0 * k, 0.5)
    prof = _strip(gaussian_filter(img, 1.0), np.asarray(tip, float), u, np.array([-u[1], u[0]]),
                  sv, np.arange(-1.0, 1.5, 1.0)).mean(1)
    i_min = int(np.argmin(prof[: max(2, int(0.5 * k / 0.5))]))
    spec = float(np.median(prof[int(0.5 * k / 0.5):]))
    if spec - prof[i_min] < 0.04:
        return tip
    thr = 0.5 * (prof[i_min] + spec)
    after = np.nonzero(prof[i_min:] > thr)[0]
    if not len(after):
        return tip
    i = i_min + int(after[0])
    fr = float(np.clip((thr - prof[i - 1]) / max(1e-9, prof[i] - prof[i - 1]), 0, 1))
    d = sv[i - 1] + fr * (sv[i] - sv[i - 1])
    p = np.asarray(tip, float) + d * u
    return (float(p[0]), float(p[1]))


def _notch_tip(img, tip_pred, ax, k, search_mm=2.5):
    """Deepest point of the notch opening along the slot (towards the back edge), only inside the
    slot band (+-half width + 0.15 mm). Returns (tip (x,y), slot dict, note) or (None, slot, note)."""
    slot = find_slot(img, tip_pred, ax, k)
    if slot is None:
        return None, None, "החריץ לא נמצא ליד המקום הצפוי"
    u, nu, c0 = slot["u"], slot["n"], slot["p"]
    half = slot["half"] + 0.15 * k
    s = np.arange(-4.0 * k, search_mm * k + 0.5, 1.0)          # along u (+ = towards the back edge)
    t = np.arange(-half, half + 0.5, 1.0)
    st = _strip(gaussian_filter(img, 0.8), c0, u, nu, s, t)
    thr = 0.5 * (slot["dark"] + slot["bright"])
    dark = st < thr
    dark = binary_opening(dark, np.ones((2, 2), bool))
    lab, n = label(dark)
    seeds = lab[(s < -1.2 * k) & (s > -3.5 * k)][:, np.abs(t) <= max(2.0, 0.3 * slot["half"])]
    ids = [j for j in np.unique(seeds) if j]
    if not ids:
        return None, slot, "לא נמצא חלק כהה של החריץ"
    comp = np.isin(lab, ids)
    rows = np.nonzero(comp.any(1))[0]
    i_deep = int(rows.max())                                    # deepest row (largest s)
    if s[i_deep] >= search_mm * k - 1:
        return None, slot, "החריץ ממשיך מעבר לטווח החיפוש"
    cols = np.nonzero(comp[i_deep])[0]
    j = int(cols[np.argmin(np.abs(t[cols] - np.mean(t[cols])))])
    # sub-pixel: where the profile along u at that column crosses the threshold
    col = st[:, j]
    s_tip = s[i_deep]
    if i_deep + 1 < len(s) and col[i_deep + 1] > col[i_deep]:
        fr = float(np.clip((thr - col[i_deep]) / (col[i_deep + 1] - col[i_deep]), 0, 1))
        s_tip = s[i_deep] + fr
    tip = c0 + s_tip * u + t[j] * nu
    return (float(tip[0]), float(tip[1])), slot, ""


# ----------------------------------------------------------------------------- whole specimen
def detect(img, H_mm, W_mm, W125_mm, an_mm, off_mm=0.0):
    """Upright image -> dict(back, side1, side2, tip, slot_u, frame (ct_core.frame_geometry), a0_mm,
    notes). Missing parts are left out."""
    out = dict(notes=[])
    e = find_edges(img, H_mm, W125_mm)
    if e is None:
        out["notes"].append("קצוות הדגם (אחורי + שני הצדדים) לא זוהו – סמן אותם ידנית.")
        return out
    out.update(back=e["back"], side1=e["side1"], side2=e["side2"], score=e["score"])
    fg = frame_geometry(e["back"], e["side1"], e["side2"], H_mm, W_mm)
    if fg is None:
        out["notes"].append("הקצוות שזוהו לא יוצרים מסגרת תקינה.")
        return out
    k = fg["pxmm"]; n_out = fg["n"]                  # n points out through the back edge
    ax = n_out                                       # crack direction: towards the back edge
    # predicted tip: W - a_n from the back edge, at the notch offset from the middle of the sides
    off = off_mm * k
    pred = fg["axis_back"] - n_out * (W_mm - an_mm) * k + fg["e1"] * off
    tip, slot, note = find_notch_tip(img, pred, ax, k)
    if tip is None:
        tip2, slot2, _ = find_notch_tip(img, fg["axis_back"] - n_out * (W_mm - an_mm) * k, ax, k)
        if tip2 is not None:
            tip, slot = tip2, slot2
    out["pred"] = tuple(map(float, pred))
    if tip is None:
        out["notes"].append(f"קצה החריץ לא זוהה ({note}) – סמן אותו ידנית.")
        out["frame"] = fg
        return out
    out["tip"] = tip
    out["slot_u"] = tuple(map(float, slot["u"]))
    out["slot_half_mm"] = slot["half"] / k
    fg = frame_geometry(e["back"], e["side1"], e["side2"], H_mm, W_mm, tip=tip, a0_mm=an_mm)
    out["frame"] = fg
    out["a0_mm"] = fg["a0_mm"]
    if abs(fg["a0_mm"] - an_mm) > 1.0:
        out["notes"].append(f"a0 בתמונה ({fg['a0_mm']:.2f} מ״מ) רחוק מ-a0 של הדגם ({an_mm:.2f}) – בדוק את קצה החריץ.")
    return out


def detect_any_orientation(img, H_mm, W_mm, W125_mm, an_mm, off_mm=0.0, ks=(0, 1, 2, 3)):
    """Phone photos (orientation unknown): tries every quarter turn, keeps the one where the specimen
    frame and the notch are found and a0 fits best. Returns (k, result) or (None, last result)."""
    best = None
    last = None
    for k in ks:
        r = detect(upright(img, k), H_mm, W_mm, W125_mm, an_mm, off_mm)
        last = r
        if "tip" not in r:
            continue
        sc = r.get("score", 0.0) - abs(r["a0_mm"] - an_mm)
        if best is None or sc > best[0]:
            best = (sc, k, r)
    if best is None:
        return None, last
    return best[1], best[2]


def back_edge_at(img, d, back, u, k, half_mm=1.5):
    """Back edge position in a measured frame (shift d relative to the base): the point where the
    line along the crack axis crosses the edge (dark outside -> bright specimen), sub-pixel.
    back: base-image back-edge point on the crack axis; u: crack direction (towards the back)."""
    u = np.asarray(u, float); p = np.asarray(back, float) + np.asarray(d, float)
    nrm = np.array([-u[1], u[0]])
    s = np.arange(-half_mm * k, half_mm * k + 0.25, 0.25)
    t = np.arange(-0.3 * k, 0.3 * k + 0.5, 1.0)
    prof = _strip(gaussian_filter(img, 0.8), p, u, nrm, s, t).mean(1)   # along u, outward = +s
    lo, hi = np.percentile(prof, 5), np.percentile(prof, 95)
    if hi - lo < 0.06:
        return None
    thr = 0.5 * (lo + hi)
    above = prof > thr
    # last bright sample before it turns dark going outward
    idx = np.nonzero(above[:-1] & ~above[1:])[0]
    if not len(idx):
        return None
    i = int(idx[np.argmin(np.abs(s[idx]))])
    fr = float((prof[i] - thr) / max(1e-9, prof[i] - prof[i + 1]))
    return p + (s[i] + fr * (s[i + 1] - s[i])) * u

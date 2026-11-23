"""C(T) specimen helpers: geometry from marks on a photo, load calibration (ASTM E647 K-scaling)
and E647 crack-length measurement spacing."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# ----------------------------------------------------------------------------- nominal specimen
# Illustrative portfolio values only. Not a laboratory drawing, manufacturing
# specification or approved test configuration. Replace with reviewed dimensions.
NOMINAL = dict(W=50.0, B=10.0, W125=62.5, H=60.0, D=12.5, D1=12.5, D2=12.5, S=27.5, an=10.0, off=0.0)
TOL = {key: (-0.5, 0.5) for key in NOMINAL}
DIM_NAMES = dict(H="גובה (בין הצדדים)", W125="רוחב (קצה קדמי ← קצה אחורי)", W="W (קו העומס ← קצה אחורי)",
                 an="a0 (קו העומס ← קצה החריץ)", D="קוטר חור", D1="קוטר חור 1", D2="קוטר חור 2",
                 S="מרחק בין מרכזי החורים", off="היסט החריץ ממרכז הדגם", B="B (עובי)")
DIM_ORDER = ("H", "W125", "W", "an", "D1", "D2", "S", "off", "B")
LBF_TO_N = 4.448222


# ----------------------------------------------------------------------------- fitting
def fit_circle(pts):
    """Least-squares circle (Kasa) -> (cx, cy, r)."""
    p = np.asarray(pts, float)
    if len(p) < 3:
        raise ValueError("need 3 points")
    A = np.column_stack([2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))])
    b = (p ** 2).sum(1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(cx), float(cy), float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))


def fit_line(pts):
    """Total-least-squares line -> (point, unit direction)."""
    p = np.asarray(pts, float)
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    return c, vt[0] / np.linalg.norm(vt[0])


def _grad_mag(img):
    g = gaussian_filter(img.astype(float), 1.2)
    gy, gx = np.gradient(g)
    return np.hypot(gx, gy)


def refine_circle(img, cx, cy, r, search=10.0, n=180):
    """Snap a circle to the strongest edge within +-search px of radius r (robust re-fit)."""
    H, W = img.shape; m = int(r + search) + 6
    x0 = int(max(0, cx - m)); x1 = int(min(W, cx + m + 1)); y0 = int(max(0, cy - m)); y1 = int(min(H, cy + m + 1))
    G = _grad_mag(img[y0:y1, x0:x1])
    cx -= x0; cy -= y0
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dr = np.arange(-search, search + 0.5, 0.5)
    pts = []
    for t in th:
        rr = r + dr
        xs = cx + rr * np.cos(t); ys = cy + rr * np.sin(t)
        v = map_coordinates(G, [ys, xs], order=1, mode="nearest")
        j = int(np.argmax(v))
        if v[j] > 0:
            pts.append((xs[j], ys[j]))
    pts = np.array(pts)
    if len(pts) < 10:
        return cx + x0, cy + y0, r
    for _ in range(3):
        c2 = fit_circle(pts)
        res = np.abs(np.hypot(pts[:, 0] - c2[0], pts[:, 1] - c2[1]) - c2[2])
        keep = res <= max(1.0, 2.5 * np.median(res))
        if keep.sum() < 10:
            break
        pts = pts[keep]
    cx2, cy2, r2 = fit_circle(pts)
    if abs(r2 - r) > search or np.hypot(cx2 - cx, cy2 - cy) > search:
        return cx + x0, cy + y0, r                      # refinement went somewhere else: keep the clicks
    return cx2 + x0, cy2 + y0, r2


def refine_line(img, p1, p2, search=8.0, n=60):
    """Snap a line (2 clicks on an edge) to the strongest edge within +-search px, robust fit."""
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    d = p2 - p1; L = np.linalg.norm(d)
    if L < 5:
        return p1, p2
    d /= L; nrm = np.array([-d[1], d[0]])
    # work on a crop around the line only (fast on 12-Mpixel photos)
    H, W = img.shape; m = int(search) + 6
    x0 = int(max(0, min(p1[0], p2[0]) - m)); x1 = int(min(W, max(p1[0], p2[0]) + m + 1))
    y0 = int(max(0, min(p1[1], p2[1]) - m)); y1 = int(min(H, max(p1[1], p2[1]) + m + 1))
    off0 = np.array([x0, y0], float)
    G = _grad_mag(img[y0:y1, x0:x1])
    p1 = p1 - off0; p2 = p2 - off0
    off = np.arange(-search, search + 0.5, 0.5)
    pts = []
    for t in np.linspace(0.05, 0.95, n):
        q = p1 + t * L * d
        xs = q[0] + off * nrm[0]; ys = q[1] + off * nrm[1]
        v = map_coordinates(G, [ys, xs], order=1, mode="nearest")
        j = int(np.argmax(v))
        pts.append((xs[j], ys[j]))
    pts = np.array(pts)
    for _ in range(3):
        c, u = fit_line(pts)
        res = np.abs((pts - c) @ np.array([-u[1], u[0]]))
        keep = res <= max(0.8, 2.5 * np.median(res))
        if keep.sum() < 8:
            break
        pts = pts[keep]
    c, u = fit_line(pts)
    if np.dot(u, d) < 0:
        u = -u
    q1 = c + np.dot(p1 - c, u) * u; q2 = c + np.dot(p2 - c, u) * u
    if np.linalg.norm(q1 - p1) > search * 1.5 or np.linalg.norm(q2 - p2) > search * 1.5:
        return p1 + off0, p2 + off0
    return q1 + off0, q2 + off0


def line_intersect(p, u, q, w):
    """Intersection of p + s*u with q + t*w."""
    A = np.array([[u[0], -w[0]], [u[1], -w[1]]], float)
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    s, _ = np.linalg.solve(A, np.asarray(q, float) - np.asarray(p, float))
    return np.asarray(p, float) + s * np.asarray(u, float)


# ----------------------------------------------------------------------------- geometry
def geometry_px(f):
    """Specimen dimensions in pixels from marked features.
    f: dict with optional 'hole1','hole2' (cx,cy,r), 'tip' (x,y), 'back','front','side1','side2' ((x,y),(x,y)).
    Returns dict of pixel dimensions + construction points (load line, crack axis, ...)."""
    out = {}
    h1, h2 = f.get("hole1"), f.get("hole2")
    tip = f.get("tip")
    if h1 and h2:
        c1 = np.array(h1[:2]); c2 = np.array(h2[:2])
        L = c2 - c1
        out["S"] = float(np.linalg.norm(L))
        out["D"] = float(h1[2] + h2[2])          # mean diameter = r1 + r2
        out["D1"] = float(2 * h1[2]); out["D2"] = float(2 * h2[2])
        lu = L / np.linalg.norm(L)
        M = (c1 + c2) / 2
        out["load_p"], out["load_u"] = M, lu
        ax = np.array([-lu[1], lu[0]])
        ref = None
        if tip is not None:
            if np.dot(np.asarray(tip) - M, ax) < 0:
                ax = -ax
            ref = M + np.dot(np.asarray(tip) - M, lu) * lu          # foot of perpendicular from the tip
            out["an"] = float(np.dot(np.asarray(tip) - M, ax))
            out["tip_offset"] = float(np.dot(np.asarray(tip) - M, lu))
        elif f.get("back"):
            b = np.asarray(f["back"][0], float)
            if np.dot(b - M, ax) < 0:
                ax = -ax
        out["axis_u"] = ax
        out["start"] = ref if ref is not None else M
        o = out["start"]
        if f.get("back"):
            b1, b2 = (np.asarray(x, float) for x in f["back"])
            X = line_intersect(o, ax, b1, b2 - b1)
            if X is not None:
                out["W"] = float(np.dot(X - o, ax)); out["back_pt"] = X
        if f.get("front"):
            q1, q2 = (np.asarray(x, float) for x in f["front"])
            Y = line_intersect(o, ax, q1, q2 - q1)
            if Y is not None:
                out["front_pt"] = Y
                if "back_pt" in out:
                    out["W125"] = float(np.linalg.norm(out["back_pt"] - Y))
    if f.get("side1") and f.get("side2"):
        a1, a2 = (np.asarray(x, float) for x in f["side1"])
        b1, b2 = (np.asarray(x, float) for x in f["side2"])
        mid = (a1 + a2) / 2
        dvec = (a2 - a1) / max(1e-9, np.linalg.norm(a2 - a1))
        nrm = np.array([-dvec[1], dvec[0]])
        e = (b2 - b1) / max(1e-9, np.linalg.norm(b2 - b1))
        Z = line_intersect(mid, nrm, b1, e)
        if Z is not None:
            out["H"] = float(np.linalg.norm(Z - mid)); out["side_pts"] = (mid, Z)
        if tip is not None and "load_u" in out:
            # lateral offset of the notch tip from the middle between the two side edges
            t0 = np.asarray(tip, float); lu = out["load_u"]
            Z1 = line_intersect(t0, lu, a1, dvec); Z2 = line_intersect(t0, lu, b1, e)
            if Z1 is not None and Z2 is not None:
                out["off"] = float(np.dot(t0 - (Z1 + Z2) / 2, lu))
    return out


def scale_from_caliper(gpx, caliper):
    """px/mm estimates from caliper-measured dimensions that are visible in the photo.
    caliper: {key: mean_mm}. Returns list of (key, px, mm, px_per_mm)."""
    est = []
    for k, mm in caliper.items():
        if k in gpx and k not in ("an", "B") and mm and mm > 0:
            est.append((k, gpx[k], mm, gpx[k] / mm))
    return est


# ----------------------------------------------------------------------------- loads (ASTM E647, C(T))
def f_alpha(a):
    a = np.asarray(a, float)
    return (2 + a) / (1 - a) ** 1.5 * (0.886 + 4.64 * a - 13.32 * a ** 2 + 14.72 * a ** 3 - 5.6 * a ** 4)


def K_from_P(P_N, a_mm, B_mm, W_mm):
    """K [MPa*sqrt(m)] for load P [N] (C(T))."""
    B = B_mm / 1000.0; W = W_mm / 1000.0
    return P_N / (B * np.sqrt(W)) * f_alpha(a_mm / W_mm) * 1e-6


def P_from_K(K, a_mm, B_mm, W_mm):
    B = B_mm / 1000.0; W = W_mm / 1000.0
    return K * B * np.sqrt(W) * 1e6 / f_alpha(a_mm / W_mm)


# The laboratory load schedule is intentionally not distributed.
DEFAULT_STAGES = []


def load_table(stages, W0, B0, W, B, lbf_to_n=LBF_TO_N, tol=0.02):
    """Rows: name, a, Pmax0 [lbf], Pmin0 [lbf], Pmax0 [N], Pmin0 [N], alpha0, f0, K0 (target Kmax),
    alpha, f, Pmax [N], Pmin [N], R, upper, lower, Kmin, dK."""
    rows = []
    for name, a, pmax_lb, pmin_lb in stages:
        pmax0 = pmax_lb * lbf_to_n; pmin0 = pmin_lb * lbf_to_n
        a0 = a / W0; f0 = float(f_alpha(a0))
        K0 = float(K_from_P(pmax0, a, B0, W0))
        R = pmin_lb / pmax_lb if pmax_lb else 0.1
        al = a / W; f = float(f_alpha(al))
        pmax = float(P_from_K(K0, a, B, W)); pmin = R * pmax
        rows.append(dict(name=name, a=a, pmax0_lb=pmax_lb, pmin0_lb=pmin_lb, pmax0=pmax0, pmin0=pmin0,
                         alpha0=a0, f0=f0, K0=K0, alpha=al, f=f, pmax=pmax, pmin=pmin, R=R,
                         upper=pmax * (1 + tol), lower=pmax * (1 - tol), dK=K0 * (1 - R)))
    return rows


# ----------------------------------------------------------------------------- E647 spacing
DEFAULT_SPACING = [(0.25, 1.2), (0.40, 0.6), (0.60, 0.3)]     # a/W from -> max delta-a [mm]


def max_spacing(aW, table=DEFAULT_SPACING):
    m = table[0][1]                       # below the first range: use the first (largest) spacing
    for lo, mx in table:
        if aW >= lo:
            m = mx
    return m


def select_points(a, W, min_inc=0.25, step=None, table=DEFAULT_SPACING):
    """Greedy E647 point selection on a crack-length series (time order).
    Keeps the first valid point, then each next point whose a has grown by >= the step
    (step = min_inc by default, never below min_inc and never above the allowed spacing).
    Returns (selected bool array, gap list [(i_prev, i_next, delta_a, allowed)])."""
    a = np.asarray(a, float)
    sel = np.zeros(len(a), bool)
    gaps = []
    valid = np.nonzero(np.isfinite(a))[0]
    if not len(valid):
        return sel, gaps
    last = valid[0]; sel[last] = True
    prev_valid = last
    for i in valid[1:]:
        allowed = max_spacing(a[prev_valid] / W, table)
        jump = a[i] - a[prev_valid]
        if jump > allowed + 1e-9:
            gaps.append((int(prev_valid), int(i), float(jump), float(allowed)))
        prev_valid = i
        allowed_sel = max_spacing(a[last] / W, table)
        st = min_inc if step is None else min(max(step, min_inc), allowed_sel)
        if a[i] - a[last] >= st - 1e-9:
            sel[i] = True; last = i
    # selected points must also respect the maximum spacing (min increment vs. max spacing conflict)
    idx = np.nonzero(sel)[0]
    have = {(g[0], g[1]) for g in gaps}
    for i, j in zip(idx[:-1], idx[1:]):
        allowed = max_spacing(a[i] / W, table)
        if a[j] - a[i] > allowed + 1e-9 and not any(g[0] >= i and g[1] <= j for g in gaps):
            gaps.append((int(i), int(j), float(a[j] - a[i]), float(allowed)))
    return sel, sorted(gaps)



def frame_geometry(back, side1, side2, H_mm, W_mm, tip=None, a0_mm=None):
    """Machine photo: scale from the specimen height (distance between the side edges), load line
    = back edge shifted by W into the specimen, crack axis perpendicular to the back edge through
    the notch tip (or the middle between the sides). Returns dict or None."""
    b1, b2 = (np.asarray(p, float) for p in back)
    e1 = (b2 - b1) / np.linalg.norm(b2 - b1)
    n = np.array([-e1[1], e1[0]])
    s1a, s1b = (np.asarray(p, float) for p in side1)
    s2a, s2b = (np.asarray(p, float) for p in side2)
    inner = (s1a + s1b + s2a + s2b) / 4
    if np.dot(inner - b1, n) > 0:
        n = -n                                   # n points out of the specimen through the back edge
    P1 = line_intersect(b1, e1, s1a, s1b - s1a); P2 = line_intersect(b1, e1, s2a, s2b - s2a)
    if P1 is None or P2 is None:
        return None
    d1 = (s1b - s1a) / np.linalg.norm(s1b - s1a); ns = np.array([-d1[1], d1[0]])
    H_px = abs(float(np.dot(P2 - P1, ns)))
    pxmm = H_px / H_mm
    Mb = (P1 + P2) / 2
    lat = float(np.dot(np.asarray(tip, float) - Mb, e1)) if tip is not None else 0.0
    axis_back = Mb + lat * e1
    start = axis_back - n * W_mm * pxmm
    out = dict(pxmm=pxmm, mmpp=1.0 / pxmm, e1=e1, n=n, start=start, axis_back=axis_back, H_px=H_px,
               lat_mm=lat / pxmm, P1=P1, P2=P2, load_p=start, load_u=e1)
    if tip is not None:
        out["a0_px"] = float(np.dot(np.asarray(tip, float) - start, n))
        out["a0_mm"] = out["a0_px"] / pxmm
    if a0_mm is not None:
        out["tip_pred"] = start + n * a0_mm * pxmm
    return out



def clevis_check(clevises, start, u, e1, pxmm, S_mm):
    """Load line from the clevises: each pin (= hole centre) lies on its clevis centre line (middle
    between the clevis top and bottom edges) at S/2 from the crack axis. With two clevises the load
    line goes through both pins; with one, it goes through the pin parallel to the given load line.
    clevises: [((top1, top2), (bot1, bot2)), ...]. Returns dict(offset_mm, pins, widths_px) or None;
    offset_mm = position of that load line on the crack axis relative to start (along u)."""
    start = np.asarray(start, float); u = np.asarray(u, float); e1 = np.asarray(e1, float)
    pins, rejected = [], []
    for top, bot in clevises:
        a1, a2 = (np.asarray(p, float) for p in top); b1, b2 = (np.asarray(p, float) for p in bot)
        # pair each top point with the nearest bottom point to get the centre line
        if np.linalg.norm(a1 - b1) + np.linalg.norm(a2 - b2) > np.linalg.norm(a1 - b2) + np.linalg.norm(a2 - b1):
            b1, b2 = b2, b1
        dt = (a2 - a1) / np.linalg.norm(a2 - a1); nt = np.array([-dt[1], dt[0]])
        db = (b2 - b1) / np.linalg.norm(b2 - b1)
        width = abs(float(np.dot(b1 - a1, nt)) + float(np.dot(b2 - a1, nt))) / 2
        ang = np.degrees(np.arccos(min(1.0, abs(float(np.dot(dt, db))))))
        if ang > 1.5 or not (0.85 * 18.0 < width / pxmm < 1.15 * 18.0):
            rejected.append(dict(angle_deg=float(ang), width_mm=width / pxmm))
            continue
        # centre line: points halfway between the two edge lines
        c1 = (a1 + (a1 + nt * np.dot(b1 - a1, nt))) / 2
        c2 = (a2 + (a2 + nt * np.dot(b2 - a2, nt))) / 2
        cdir = (c2 - c1) / np.linalg.norm(c2 - c1)
        mid = (c1 + c2) / 2
        sgn = 1.0 if np.dot(mid - start, e1) > 0 else -1.0
        q = start + sgn * S_mm / 2 * pxmm * e1
        P = line_intersect(q, u, c1, cdir)
        if P is not None:
            pins.append(dict(pin=P, width_mm=width / pxmm, sgn=sgn))
    if not pins:
        return dict(rejected=rejected) if rejected else None
    if len(pins) >= 2:
        P1, P2 = pins[0]["pin"], pins[1]["pin"]
        X = line_intersect(start, u, P1, P2 - P1)
    else:
        X = line_intersect(start, u, pins[0]["pin"], e1)
    if X is None:
        return None
    return dict(offset_mm=float(np.dot(X - start, u)) / pxmm, start=X, pins=[p["pin"] for p in pins],
                widths_mm=[p["width_mm"] for p in pins], rejected=rejected)

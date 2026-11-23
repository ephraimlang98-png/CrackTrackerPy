"""Calibration-photo detection v3: geometry first.
1. straight lines (Hough on colour gradients) -> the specimen quadrilateral (36 x 37.5 mm, square-ish)
2. front edge = the side with the notch gap and the holes next to it
3. holes searched where the C(T) geometry puts them, fitted on their rim
4. notch tip along the crack axis (the slot differs from the metal around it)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter, zoom

from ct_core import fit_circle, geometry_px, line_intersect, refine_line

NOM = dict(W=30.0, W125=37.5, H=36.0, D=7.5, S=16.5, an=6.0, front_to_load=7.5)


def _to_float_rgb(img):
    a = np.asarray(img, float)
    if a.max() > 1.5:
        a = a / 255.0
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return a


def _grad(rgb, sigma):
    gx = np.zeros(rgb.shape[:2]); gy = np.zeros(rgb.shape[:2]); best = np.zeros(rgb.shape[:2])
    for c in range(3):
        g = gaussian_filter(rgb[..., c], sigma)
        y, x = np.gradient(g)
        m = np.hypot(x, y)
        sel = m > best
        gx[sel], gy[sel], best[sel] = x[sel], y[sel], m[sel]
    return gx, gy, best


def _hough_lines(gx, gy, mag, n_lines=24):
    """Lines as (theta, rho) from strong edge pixels voting near their gradient direction."""
    H, W = mag.shape
    thr = np.percentile(mag, 92)
    ys, xs = np.nonzero(mag > thr)
    ang = np.arctan2(gy[ys, xs], gx[ys, xs]) % np.pi
    thetas = np.deg2rad(np.arange(0, 180, 0.5))
    diag = int(np.hypot(H, W)) + 1
    acc = np.zeros((len(thetas), 2 * diag), np.float32)
    w = mag[ys, xs]
    for d in np.deg2rad(np.arange(-3, 3.01, 0.5)):
        a = (ang + d) % np.pi
        ti = np.round(np.rad2deg(a) / 0.5).astype(int) % len(thetas)
        rho = np.round(xs * np.cos(thetas[ti]) + ys * np.sin(thetas[ti])).astype(int) + diag
        np.add.at(acc, (ti, rho), w)
    acc = gaussian_filter(acc, (1.0, 1.5))
    pk = (acc == maximum_filter(acc, size=(9, 15))) & (acc > 0.05 * acc.max())
    ti, ri = np.nonzero(pk)
    order = np.argsort(-acc[ti, ri])[:n_lines]
    return [(thetas[ti[j]], ri[j] - diag, float(acc[ti[j], ri[j]])) for j in order]


def _seg_support(mag, p1, p2, thr, n=120):
    t = np.linspace(0.05, 0.95, n)
    p = p1[None, :] + t[:, None] * (p2 - p1)[None, :]
    d = (p2 - p1) / max(1e-9, np.linalg.norm(p2 - p1)); nrm = np.array([-d[1], d[0]])
    best = np.zeros(n)
    for o in (-2, -1, 0, 1, 2):
        q = p + o * nrm
        best = np.maximum(best, map_coordinates(mag, [q[:, 1], q[:, 0]], order=1, mode="nearest"))
    return best > thr


def _line_pts(theta, rho):
    c, s = np.cos(theta), np.sin(theta)
    p0 = np.array([c * rho, s * rho]); d = np.array([-s, c])
    return p0, d


def find_quad(rgb, max_dim=1000):
    """The specimen outline: four lines forming a square-ish quadrilateral with strong edge support.
    Returns corners (4x2, full-res, ordered around) and the side lines, or None."""
    H0, W0 = rgb.shape[:2]
    f = min(1.0, max_dim / max(H0, W0))
    small = zoom(gaussian_filter(rgb, (0.5 / f, 0.5 / f, 0)), (f, f, 1), order=1) if f < 1 else rgb
    gx, gy, mag = _grad(small, 1.5)
    lines = _hough_lines(gx, gy, mag)
    thr = np.percentile(mag, 85)
    H, W = mag.shape
    best = None
    L = len(lines)
    pairs = []
    for i in range(L):
        for j in range(i + 1, L):
            ti, ri, _ = lines[i]; tj, rj, _ = lines[j]
            dt = abs((ti - tj + np.pi / 2) % np.pi - np.pi / 2)
            if dt < np.deg2rad(6) and 0.1 * min(H, W) < abs(ri - rj * np.cos(ti - tj)) < 0.95 * min(H, W):
                pairs.append((i, j))
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            i1, j1 = pairs[a]; i2, j2 = pairs[b]
            if len({i1, j1, i2, j2}) < 4:
                continue
            dt = abs((lines[i1][0] - lines[i2][0]) % np.pi - np.pi / 2)
            if dt > np.deg2rad(12):
                continue
            ls = [lines[i1], lines[i2], lines[j1], lines[j2]]
            P = [_line_pts(t, r) for t, r, _ in ls]
            corners = []
            ok = True
            for k in range(4):
                X = line_intersect(P[k][0], P[k][1], P[(k + 1) % 4][0], P[(k + 1) % 4][1])
                if X is None or not (-0.05 * W < X[0] < 1.05 * W and -0.05 * H < X[1] < 1.05 * H):
                    ok = False; break
                corners.append(X)
            if not ok:
                continue
            C = np.array(corners)
            sides = [np.linalg.norm(C[(k + 1) % 4] - C[k]) for k in range(4)]
            if min(sides) < 0.08 * min(H, W):
                continue
            ratio = (sides[0] + sides[2]) / (sides[1] + sides[3])
            if not 0.8 < ratio < 1.25:
                continue
            sup = [_seg_support(mag, C[k], C[(k + 1) % 4], thr).mean() for k in range(4)]
            # the front edge has the notch gap - allow one weaker side
            ss = sorted(sup)
            score = ss[1] + ss[2] + ss[3] + 0.5 * ss[0] - 2.0 * abs(np.log(ratio)) + 0.0005 * np.mean(sides)
            if best is None or score > best[0]:
                best = (score, C / f, sup)
    if best is None:
        return None
    return best[1], best[2]


def _circle_near(img_gray, c0, r0, search=0.35):
    """Hole rim near a predicted centre / radius: radial edge search, robust circle fit."""
    m = int((1 + search) * r0 + 12)
    x0 = int(max(0, c0[0] - m)); y0 = int(max(0, c0[1] - m))
    sub = img_gray[y0:int(c0[1] + m) + 1, x0:int(c0[0] + m) + 1]
    if sub.shape[0] < 8 or sub.shape[1] < 8:
        return None
    gy, gx = np.gradient(gaussian_filter(sub, max(1.0, r0 / 40)))
    th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    rr = np.arange((1 - search) * r0, (1 + search) * r0, 0.5)
    pts = []
    for t in th:
        xx = c0[0] - x0 + rr * np.cos(t); yy = c0[1] - y0 + rr * np.sin(t)
        rad = np.abs(map_coordinates(gx, [yy, xx], order=1) * np.cos(t) + map_coordinates(gy, [yy, xx], order=1) * np.sin(t))
        j = int(np.argmax(rad))
        if rad[j] > 0:
            pts.append((xx[j] + x0, yy[j] + y0))
    pts = np.array(pts)
    if len(pts) < 20:
        return None
    for _ in range(4):
        c = fit_circle(pts)
        res = np.abs(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) - c[2])
        keep = res <= max(1.0, 2.0 * np.median(res))
        if keep.sum() < 20:
            break
        pts = pts[keep]
    c = fit_circle(pts)
    res = np.abs(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) - c[2])
    return c, float(len(pts)) / len(th), float(np.median(res))


def _edge_metal(rgb_s, p1, p2, inside, k, n=80, min_span=0.04):
    """Edge of the metal face (not of its shadow / side face): along normals, the first point going
    outward where the colour is closer to the outside than to the metal. Robust line fit. None if weak."""
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    d = (p2 - p1) / np.linalg.norm(p2 - p1); nrm = np.array([-d[1], d[0]])
    if np.dot(np.asarray(inside, float) - (p1 + p2) / 2, nrm) > 0:
        nrm = -nrm                                          # nrm points outward
    o = np.arange(-3.5 * k, 3.5 * k, 0.25)
    pts = []
    for t in np.linspace(0.1, 0.9, n):
        c = p1 + t * (p2 - p1)
        P = c[None, :] + o[:, None] * nrm[None, :]
        prof = np.stack([map_coordinates(rgb_s[..., ch], [P[:, 1], P[:, 0]], order=1, mode="nearest")
                         for ch in range(rgb_s.shape[2])], 1)
        cin = np.median(prof[o < -2.6 * k], 0)
        cout = np.median(prof[o > 2.8 * k], 0)
        span = np.linalg.norm(cout - cin)
        if span < min_span:
            continue
        dist = np.linalg.norm(prof - cin, axis=1) / span
        j0 = int(np.searchsorted(o, -2.4 * k))
        above = np.nonzero(dist[j0:] > 0.5)[0]
        if not len(above) or j0 + above[0] == 0:
            continue
        j = j0 + above[0]
        f = float(np.clip((0.5 - dist[j - 1]) / max(1e-9, dist[j] - dist[j - 1]), 0, 1))
        pts.append(c + (o[j - 1] + f * (o[j] - o[j - 1])) * nrm)
    if len(pts) < n // 3:
        return None
    pts = np.array(pts)
    keep = np.ones(len(pts), bool)
    for _ in range(5):
        q = pts[keep]; m = q.mean(0)
        u = np.linalg.svd(q - m)[2][0]
        r = np.abs((pts - m) @ np.array([-u[1], u[0]]))
        keep = r <= max(0.7, 2.5 * np.median(r[keep]))
    if keep.sum() < n // 3:
        return None
    s = (pts[keep] - m) @ u
    a, b = m + s.min() * u, m + s.max() * u
    return (a, b) if np.dot(b - a, p2 - p1) > 0 else (b, a)


def find_tip_slot(gray, start, ax, k, a_lo=3.0, a_hi=10.0):
    """Notch tip: walking along the crack axis from the front, the narrow slot is darker / different
    from the metal beside it; the tip is where that difference ends. Lateral offset tracked."""
    ax = np.asarray(ax, float); nrm = np.array([-ax[1], ax[0]]); start = np.asarray(start, float)
    s = np.arange(a_lo, a_hi, 0.02)
    t = np.arange(-1.2, 1.2, 0.02)
    P = start[None, None, :] + (s[:, None, None] * k) * ax[None, None, :] + (t[None, :, None] * k) * nrm[None, None, :]
    strip = map_coordinates(gaussian_filter(gray, 0.7), [P[..., 1].ravel(), P[..., 0].ravel()], order=1,
                            mode="nearest").reshape(len(s), len(t))
    side = np.median(np.concatenate([strip[:, :15], strip[:, -15:]], 1), axis=1)
    dev = side[:, None] - strip                        # positive where darker than the surroundings
    sgn = 1.0 if np.median(dev[: len(s) // 4].max(1)) >= np.median((-dev[: len(s) // 4]).max(1)) else -1.0
    dev = sgn * dev
    cen = np.argmax(gaussian_filter(dev, (6, 2)), axis=1)
    val = gaussian_filter(dev, (2, 1))[np.arange(len(s)), cen]
    ref = np.median(val[: len(s) // 5])
    if ref <= 0.02:
        return None
    inside = val > 0.35 * ref
    run = int(0.5 / 0.02)
    j = 0
    while j + 1 < len(s) and (inside[j + 1] or inside[j + 1:j + run].mean() > 0.5):
        j += 1
    tip = start + s[j] * k * ax + t[cen[j]] * k * nrm
    return (float(tip[0]), float(tip[1]))


def is_color(img_rgb):
    """True for a colour photo (phone), False for a grey camera image saved as RGB."""
    a = _to_float_rgb(np.asarray(img_rgb)[::8, ::8])
    ss = a.sum(2) + 1e-6
    return float(max(np.std(a[..., 0] / ss), np.std(a[..., 2] / ss))) * 10.0 > 0.05


def detect_specimen_v3(img_rgb):
    """Returns (feats, notes): hole1, hole2, tip, back, front, side1, side2 (full-res px)."""
    rgb = _to_float_rgb(img_rgb)
    gray = rgb.mean(2)
    q = find_quad(rgb)
    if q is None:
        return {}, ["לא נמצא קו המתאר של הדגם – ודא שכל הדגם בתמונה ושהרקע שונה ממנו."]
    C, sup = q
    best = None
    for kf in range(4):                                 # which side is the front (notched) edge?
        f1, f2 = C[kf], C[(kf + 1) % 4]
        b1, b2 = C[(kf + 2) % 4], C[(kf + 3) % 4]
        fm, bm = (f1 + f2) / 2, (b1 + b2) / 2
        depth = np.linalg.norm(bm - fm)
        width = np.linalg.norm(f2 - f1)
        k = (depth / NOM["W125"] + width / NOM["H"]) / 2      # px/mm estimate
        ax = (bm - fm) / depth                                  # front -> back
        lu = (f2 - f1) / width
        load_c = fm + ax * NOM["front_to_load"] * k
        found = []
        for sg in (-1, 1):
            r = _circle_near(gray, load_c + sg * lu * NOM["S"] / 2 * k, NOM["D"] / 2 * k)
            if r is not None:
                found.append(r)
        if len(found) < 2:
            continue
        sc = sum(x[1] for x in found) - sum(x[2] for x in found) / (0.05 * k) - abs(found[0][0][2] - found[1][0][2]) / k
        if best is None or sc > best[0]:
            best = (sc, kf, found, k, ax, lu)
    if best is None:
        return {}, ["החורים לא נמצאו במקום שהגיאומטריה צופה – סמן אותם ידנית."]
    _, kf, found, k, ax, lu = best
    (c1, _, _), (c2, _, _) = found
    feats = {"hole1": tuple(map(float, c1)), "hole2": tuple(map(float, c2))}
    names = ["front", "side2", "back", "side1"]
    rgb_s = gaussian_filter(rgb, (1.0, 1.0, 0))
    ssum = rgb_s.sum(2) + 1e-6
    chroma = np.stack([rgb_s[..., 0] / ssum, rgb_s[..., 2] / ssum], -1) * 10.0   # brightness-free
    is_color = float(np.std(chroma[::4, ::4], axis=(0, 1)).max()) > 0.05
    cen = C.mean(0)
    for j in range(4):
        p1, p2 = C[(kf + j) % 4], C[(kf + j + 1) % 4]
        e = None
        if is_color:                     # colour differs metal / background: shadows and glints ignored
            e = _edge_metal(chroma, p1, p2, cen, k, min_span=0.15)
        if e is None:
            e = refine_line(gray, p1 + 0.08 * (p2 - p1), p2 - 0.08 * (p2 - p1), search=max(4.0, 0.3 * k))
        q1, q2 = e
        feats[names[j]] = (tuple(map(float, q1)), tuple(map(float, q2)))
    # notch tip along the axis through the middle of the holes (perpendicular to the load line)
    M = (np.array(c1[:2]) + np.array(c2[:2])) / 2
    lu2 = (np.array(c2[:2]) - np.array(c1[:2])); lu2 /= np.linalg.norm(lu2)
    ax2 = np.array([-lu2[1], lu2[0]])
    if np.dot(ax2, ax) < 0:
        ax2 = -ax2
    notes = []
    g = geometry_px(feats)
    if "H" in g:                         # the holes must fit the C(T) proportions, else they are not holes
        Hh = g["H"]
        if abs(g["S"] / Hh / (NOM["S"] / NOM["H"]) - 1) > 0.07 or abs(g["D"] / Hh / (NOM["D"] / NOM["H"]) - 1) > 0.15:
            return {}, ["החורים שנמצאו לא מתאימים לפרופורציות של דגם C(T)."]
    tip = None
    if not is_color:                     # grey (camera) images: the texture method is proven there
        from ct_detect2 import find_tip_v2
        tip = find_tip_v2(gray, M, ax2, k)
    if tip is None:
        tip = find_tip_slot(gray, M, ax2, k)
    if tip is not None and 3.0 <= np.dot(np.asarray(tip) - M, ax2) / k <= 10.0:
        feats["tip"] = tip
    else:
        notes.append("קצה החריץ לא זוהה – סמן אותו ידנית.")
    return feats, notes

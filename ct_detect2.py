"""Robust automatic detection for C(T) photos (v2).
Calibration photo: holes from several independent cues, then every edge searched only where the
C(T) geometry predicts it. Machine photo: specimen edges with automatic orientation."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (binary_fill_holes, binary_opening, find_objects, gaussian_filter,
                           gaussian_filter1d, label, map_coordinates, zoom)

from ct_core import frame_geometry, refine_circle, refine_line
from ct_detect import _otsu, _texture, detect_frame, find_holes, find_tip, find_tip_near


def _blob_candidates(img, max_dim=800):
    """Round blob candidates (cx, cy, r, circularity) in full-res px from three independent cues:
    smooth (texture-free) regions, bright regions and dark regions."""
    f, tex = _texture(img, max_dim)
    im = zoom(gaussian_filter(img, 0.5 / f), f, order=1) if f < 1 else img
    H, W = im.shape
    lt = np.log(tex + 1e-4)
    t_int = _otsu(im.ravel())
    masks = [binary_opening(lt < _otsu(lt), iterations=2),
             binary_opening(im > t_int, iterations=2),
             binary_opening(im < t_int, iterations=2)]
    out = []
    for mask in masks:
        lab, _ = label(mask)
        for kk, sl in enumerate(find_objects(lab), 1):
            if sl is None:
                continue
            comp = lab[sl] == kk
            a = int(comp.sum())
            if a < 0.0004 * H * W or a > 0.08 * H * W:
                continue
            ys, xs = np.nonzero(comp)
            ys = ys + sl[0].start; xs = xs + sl[1].start
            if ys.min() <= 1 or xs.min() <= 1 or ys.max() >= H - 2 or xs.max() >= W - 2:
                continue
            filled = binary_fill_holes(comp)
            hh, ww = comp.shape
            circ = filled.sum() / (np.pi * (max(hh, ww) / 2.0) ** 2)
            if circ < 0.55 or min(hh, ww) < 0.7 * max(hh, ww):
                continue
            out.append((xs.mean() / f, ys.mean() / f, np.sqrt(filled.sum() / np.pi) / f, float(circ)))
    return out


def _rim_score(img, cx, cy, r):
    """Fraction of the circle with a clear radial edge (a real hole has an edge all around)."""
    m = int(r * 1.3) + 4
    H, W = img.shape
    x0, y0 = int(max(0, cx - m)), int(max(0, cy - m))
    x1, y1 = int(min(W, cx + m)), int(min(H, cy + m))
    win = gaussian_filter(img[y0:y1, x0:x1], max(1.0, r / 60))
    gy, gx = np.gradient(win)
    th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    best = []
    for dr in np.linspace(-0.06 * r, 0.06 * r, 7):
        xx = cx - x0 + (r + dr) * np.cos(th); yy = cy - y0 + (r + dr) * np.sin(th)
        best.append(np.abs(map_coordinates(gx, [yy, xx], order=1) * np.cos(th) +
                           map_coordinates(gy, [yy, xx], order=1) * np.sin(th)))
    g = np.max(best, axis=0)
    ref = np.percentile(np.hypot(gx, gy), 90)
    return float(np.mean(g > 0.5 * ref)) if ref > 0 else 0.0


def find_holes_v2(img):
    cands = _blob_candidates(img)
    merged = []
    for c in sorted(cands, key=lambda c: -(c[3] + c[2] / max(1e-6, max(x[2] for x in cands)))):
        if all(np.hypot(c[0] - m[0], c[1] - m[1]) > max(c[2], m[2]) for m in merged):
            merged.append(c)
    merged = merged[:20]
    scored = [(c[0], c[1], c[2], c[3], _rim_score(img, c[0], c[1], c[2])) for c in merged]
    rmax = max((c[2] for c in scored), default=1.0)
    best, bs = None, -1e9
    for a in range(len(scored)):
        for b in range(a + 1, len(scored)):
            c1, c2 = scored[a], scored[b]
            r = (c1[2] + c2[2]) / 2
            dd = np.hypot(c1[0] - c2[0], c1[1] - c2[1]) / max(r, 1e-6)
            if abs(c1[2] - c2[2]) > 0.25 * r or not (3.6 < dd < 5.2) or r < 0.4 * rmax:
                continue
            # the loading holes are the largest round regions on the specimen
            sc = c1[4] + c2[4] + 0.5 * (c1[3] + c2[3]) - 0.6 * abs(dd - 4.4) + 2.0 * r / rmax
            if sc > bs:
                bs, best = sc, (c1, c2)
    if best is None:
        return None
    return [tuple(float(x) for x in _refine_hole(img, c[0], c[1], c[2]))
            for c in sorted(best, key=lambda c: (c[0], c[1]))]


def _edge_profile(img, origin, along, normal, t_range, s_range, step_t, skip=None):
    """Mean brightness profile across a predicted straight edge, then its derivative: a real edge
    is the same all along and adds up, random scratches cancel out."""
    s = np.arange(s_range[0], s_range[1], 0.5)
    acc = np.zeros(len(s)); n = 0
    for t in np.arange(t_range[0], t_range[1], step_t):
        if skip and any(a <= t <= b for a, b in skip):
            continue
        p = origin[None, :] + t * along[None, :] + s[:, None] * normal[None, :]
        acc += map_coordinates(img, [p[:, 1], p[:, 0]], order=1, mode="nearest")
        n += 1
    prof = gaussian_filter1d(acc / max(1, n), 2.0)
    return s, np.abs(np.gradient(prof))


def _pick_near(s, g, rel=0.4):
    """Edge position: among the clear peaks (>= rel * strongest) the one nearest the predicted place
    (s = 0); the specimen dimensions are known to a few tenths of a mm."""
    from scipy.signal import find_peaks
    pk, _ = find_peaks(g, height=rel * g.max())
    if not len(pk):
        return s[int(np.argmax(g))], float(g.max())
    j = pk[int(np.argmin(np.abs(s[pk])))]
    return s[j], float(g[j])


def _refine_hole(img, cx, cy, r):
    """Circle on the specimen FACE edge: the outermost strong edge around the blob (the hole wall
    seen inside the hole gives an inner edge that must not be taken)."""
    m = int(r * 1.25) + 6
    H, W = img.shape
    x0, y0 = int(max(0, cx - m)), int(max(0, cy - m))
    x1, y1 = int(min(W, cx + m)), int(min(H, cy + m))
    win = gaussian_filter(img[y0:y1, x0:x1], max(1.0, r / 80))
    gy, gx = np.gradient(win)
    G = np.hypot(gx, gy)
    th = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    rr = np.arange(0.8 * r, 1.2 * r, 0.5)
    pts = []
    for t in th:
        xx = cx - x0 + rr * np.cos(t); yy = cy - y0 + rr * np.sin(t)
        g = map_coordinates(G, [yy, xx], order=1)
        if g.max() <= 0:
            continue
        idx = np.nonzero(g >= 0.6 * g.max())[0]
        j = idx[-1]
        while j + 1 < len(g) and g[j + 1] >= g[j]:            # climb to the local peak
            j += 1
        pts.append((cx + rr[j] * np.cos(t), cy + rr[j] * np.sin(t)))
    pts = np.array(pts)
    if len(pts) < 20:
        return cx, cy, r
    from ct_core import fit_circle
    for _ in range(3):
        c = fit_circle(pts)
        res = np.abs(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) - c[2])
        keep = res <= max(1.0, 2.5 * np.median(res))
        if keep.sum() < 20:
            break
        pts = pts[keep]
    return fit_circle(pts)


def find_tip_v2(img, M, ax, k, k0=40.0):
    """Notch tip in a calibration photo. A strip along the crack axis is resampled at a fixed
    40 px/mm (independent of photo size and rotation); band-pass texture energy is low in the
    notch and high on the machined metal; the tip is where the metal texture starts for good."""
    from scipy.ndimage import uniform_filter
    M = np.asarray(M, float); ax = np.asarray(ax, float); nrm = np.array([-ax[1], ax[0]])
    f = k0 / k
    src = gaussian_filter(img, 0.5 / f) if f < 1 else img
    s_mm = np.arange(-1.0, 13.0, 1.0 / k0)
    t_mm = np.arange(-1.5, 1.5, 1.0 / k0)
    P = (M[None, None, :] + (s_mm[:, None, None] * k) * ax[None, None, :] +
         (t_mm[None, :, None] * k) * nrm[None, None, :])
    strip = map_coordinates(src, [P[..., 1].ravel(), P[..., 0].ravel()], order=1,
                            mode="nearest").reshape(len(s_mm), len(t_mm))
    B = gaussian_filter(strip, 0.8) - gaussian_filter(strip, 3.0)
    # texture energy on the axis only (a window long along the crack, 3 px across): near its end
    # the notch narrows and the converging walls must not count as metal before it really ends
    from scipy.ndimage import uniform_filter1d
    # follow the notch centre line: in each row the middle of the smooth (low-texture) band
    # around the previous centre; the notch need not lie exactly on the hole midline
    E2 = np.sqrt(gaussian_filter(B * B, (4.0, 1.5)))
    c0 = len(t_mm) // 2
    centers = np.full(len(s_mm), float(c0))
    cur = float(c0)
    lim = int(0.6 * k0)
    for j in np.nonzero((s_mm >= 1.0) & (s_mm <= 12.0))[0]:
        lo = int(max(0, cur - lim)); hi = int(min(len(t_mm), cur + lim + 1))
        row = E2[j, lo:hi]
        jj = lo + int(np.argmin(row))
        cur = 0.8 * cur + 0.2 * jj
        centers[j] = cur
    # use the centre line found inside the notch (before the tip region) as a straight line
    m = (s_mm >= 1.0) & (s_mm <= 4.0)
    cf = np.polyfit(s_mm[m], centers[m], 1)
    cl = np.clip(np.round(np.polyval(cf, s_mm)).astype(int), 1, len(t_mm) - 2)
    rows = np.arange(len(s_mm))
    e = np.mean([B[rows, cl + o] ** 2 for o in (-1, 0, 1)], axis=0)
    v = np.sqrt(uniform_filter1d(e, 9))
    v = np.convolve(v, np.ones(5) / 5, mode="same")
    lat_mm = np.polyval(cf, s_mm) / k0 - 1.5
    sel = lambda a, b: v[(s_mm >= a) & (s_mm <= b)]
    notch_lvl, metal_lvl = np.median(sel(1.0, 3.5)), np.median(sel(9.0, 12.0))
    if not metal_lvl > 1.3 * notch_lvl:
        return None
    thr = np.sqrt(notch_lvl * metal_lvl)
    run = int(1.0 * k0)
    for j in np.nonzero(s_mm >= 3.0)[0]:
        if j + run >= len(v):
            break
        if v[j] > thr and np.median(v[j:j + run]) > thr:
            sj = s_mm[j]
            if v[j - 1] < thr:
                sj = s_mm[j - 1] + (thr - v[j - 1]) / (v[j] - v[j - 1]) * (s_mm[j] - s_mm[j - 1])
            lt = float(np.interp(sj, s_mm, lat_mm))
            tip = M + sj * k * ax + lt * k * nrm
            return (float(tip[0]), float(tip[1]))
    return None


def detect_specimen_v2(img):
    """Holes -> scale + load line -> each edge searched near its predicted place -> notch tip."""
    notes = []
    holes = find_holes_v2(img)
    if not holes:
        return {}, ["לא נמצאו שני החורים – ודא שהם נראים בתמונה, או סמן אותם ידנית."]
    feats = {"hole1": holes[0], "hole2": holes[1]}
    c1, c2 = np.array(holes[0][:2]), np.array(holes[1][:2])
    S_px = float(np.linalg.norm(c2 - c1)); lu = (c2 - c1) / S_px; M = (c1 + c2) / 2
    k = S_px / 16.5
    nrm = np.array([-lu[1], lu[0]])
    best = None
    for ax in (nrm, -nrm):
        sb, gb = _edge_profile(img, M + 30 * k * ax, lu, ax, (-15 * k, 15 * k), (-3 * k, 3 * k), 0.25 * k)
        sf, gfr = _edge_profile(img, M - 7.5 * k * ax, lu, ax, (-16 * k, 16 * k), (-3 * k, 3 * k), 0.25 * k,
                                skip=[(-5 * k, 5 * k)])
        pb, vb = _pick_near(sb, gb); pf, vf = _pick_near(sf, gfr)
        sc = vb + vf
        if best is None or sc > best[0]:
            best = (sc, ax, pb, pf)
    _, ax, ob, of = best

    def line_at(center, along, half):
        q1, q2 = refine_line(img, center - along * half, center + along * half, search=max(6.0, 0.25 * k))
        return (tuple(map(float, q1)), tuple(map(float, q2)))
    feats["back"] = line_at(M + (30 * k + ob) * ax, lu, 13 * k)
    fc = M + (-7.5 * k + of) * ax
    fa = refine_line(img, fc - lu * 16 * k, fc - lu * 6 * k, search=max(6.0, 0.25 * k))
    fb = refine_line(img, fc + lu * 6 * k, fc + lu * 16 * k, search=max(6.0, 0.25 * k))
    feats["front"] = (tuple(map(float, fa[0])), tuple(map(float, fb[1])))
    for key, sgn in (("side1", -1), ("side2", 1)):
        ss, gs = _edge_profile(img, M + sgn * 18 * k * lu + 12 * k * ax, ax, lu, (-10 * k, 14 * k),
                               (-3 * k, 3 * k), 0.25 * k)
        o, _ = _pick_near(ss, gs)
        feats[key] = line_at(M + (sgn * 18 * k + o) * lu + 12 * k * ax, ax, 12 * k)

    def plausible(t):
        return t is not None and 3.0 <= float(np.dot(np.asarray(t, float) - M, ax)) / k <= 9.0
    tip = find_tip_v2(img, M, ax, k)
    if plausible(tip):
        feats["tip"] = tuple(map(float, tip))
    else:
        notes.append("קצה החריץ לא זוהה – סמן אותו ידנית.")
    return feats, notes


def detect_frame_auto(img, H_mm, W_mm, a0_mm, grow=None):
    """Machine photo with unknown orientation: try the four directions and keep the one where the
    specimen edges and the notch tip at the expected place are found most convincingly."""
    best = None
    for gdir in ([grow] if grow and grow != "auto" else ["up", "down", "left", "right"]):
        try:
            fr = detect_frame(img, gdir)
        except Exception:
            fr = {}
        if not fr:
            continue
        g = frame_geometry(fr["back"], fr["side1"], fr["side2"], H_mm, W_mm, a0_mm=a0_mm)
        if g is None:
            continue
        L = np.linalg.norm(np.subtract(fr["back"][1], fr["back"][0]))
        # physical plausibility: the back edge spans the specimen (length ~ its height between the
        # side edges) and the specimen fills a reasonable part of the photo
        frac = g["H_px"] / min(img.shape)
        score = 1.0 - 3.0 * abs(L / g["H_px"] - 1.0) - (3.0 if not 0.2 <= frac <= 1.2 else 0.0)
        tip = find_tip_near(img, g["tip_pred"], g["n"], g["mmpp"])
        if tip is not None:
            g2 = frame_geometry(fr["back"], fr["side1"], fr["side2"], H_mm, W_mm, tip=tip)
            score += 2.0 * max(0.0, 1.0 - abs(g2["a0_mm"] - a0_mm) / 1.5)
            fr["tip"] = tip
        fr["grow"] = gdir
        if best is None or score > best[0]:
            best = (score, fr)
    return best[1] if best else {}

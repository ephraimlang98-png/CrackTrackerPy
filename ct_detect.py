"""Automatic detection of a C(T) specimen in a photo: the two holes, the notch tip and the edges.
Machined / scratched metal has a strong fine texture; the holes (background or pins seen through
them) and the notch are smooth. The holes are found as two round smooth regions of equal size at
the C(T) spacing (S/r = 4.4), the notch tip where the smooth notch ends on the crack axis, and the
edges where the textured plate ends; every feature is then snapped to the exact edge."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (binary_closing, binary_fill_holes, binary_opening, find_objects, gaussian_filter,
                           label, map_coordinates, zoom)

from ct_core import fit_line, refine_circle, refine_line


def _texture(img, max_dim=800):
    H0, W0 = img.shape
    f = min(1.0, max_dim / max(H0, W0))
    im = zoom(gaussian_filter(img, 0.5 / f), f, order=1) if f < 1 else img     # anti-alias: noise averages out
    hp = np.abs(im - gaussian_filter(im, 1.5))
    return f, gaussian_filter(hp, 4.0)


def _otsu(v):
    h, e = np.histogram(v, 128); c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[int(np.argmax(w0 * w1 * (m0 - m1) ** 2))]


def _walk_out(mask, p, d, max_steps):
    """Steps from p along d while inside mask; returns the last inside point (or None)."""
    H, W = mask.shape
    last = None
    for t in range(int(max_steps)):
        q = p + t * d
        x, y = int(round(q[0])), int(round(q[1]))
        if not (0 <= x < W and 0 <= y < H):
            break
        if mask[y, x]:
            last = q
        elif last is not None:
            break
    return last


def find_holes(img):
    """Two round smooth regions of similar size, ~4.4 radii apart -> [(cx, cy, r), (cx, cy, r)] or None."""
    f, tex = _texture(img)
    lt = np.log(tex + 1e-4)
    thr = _otsu(lt)
    smooth = binary_opening(lt < thr, iterations=2)
    H, W = smooth.shape
    lab, _ = label(smooth)
    cands = []
    for k, sl in enumerate(find_objects(lab), 1):
        if sl is None:
            continue
        comp = lab[sl] == k
        if comp.sum() < 0.0005 * H * W:
            continue
        ys, xs = np.nonzero(comp)
        ys = ys + sl[0].start; xs = xs + sl[1].start
        if ys.min() <= 1 or xs.min() <= 1 or ys.max() >= H - 2 or xs.max() >= W - 2:
            continue
        filled = binary_fill_holes(comp)
        r_eq = np.sqrt(filled.sum() / np.pi)
        circ = filled.sum() / (np.pi * (max(comp.shape) / 2.0) ** 2)
        if circ < 0.6:
            continue
        cands.append((xs.mean(), ys.mean(), r_eq, circ))
    best, bs = None, -1e9
    for a in range(len(cands)):
        for b in range(a + 1, len(cands)):
            c1, c2 = cands[a], cands[b]
            r = (c1[2] + c2[2]) / 2
            dd = np.hypot(c1[0] - c2[0], c1[1] - c2[1]) / max(r, 1e-6)
            if abs(c1[2] - c2[2]) > 0.25 * r or not (3.4 < dd < 5.4):
                continue
            s = c1[3] + c2[3] - 0.5 * abs(dd - 4.4) + 0.002 * r
            if s > bs:
                bs, best = s, (c1, c2)
    if best is None:
        return None, (f, lt, thr, smooth)
    out = []
    for c in sorted(best, key=lambda c: (c[0], c[1])):
        cx, cy, r = c[0] / f, c[1] / f, c[2] / f
        out.append(tuple(float(x) for x in refine_circle(img, cx, cy, r, search=max(6.0, 0.12 * r))))
    return out, (f, lt, thr, smooth)


def _refine_tip(img, tip, ax, half_px):
    """Notch tip at full resolution: where the smooth notch ends and the textured metal begins."""
    s = np.arange(-half_px, half_px, 0.5)
    nrm = np.array([-ax[1], ax[0]])
    x0, y0 = int(tip[0]), int(tip[1]); R = int(half_px) + 12
    H, W = img.shape
    ya, yb, xa, xb = max(0, y0 - R), min(H, y0 + R), max(0, x0 - R), min(W, x0 + R)
    win = img[ya:yb, xa:xb]
    tx = gaussian_filter(np.abs(win - gaussian_filter(win, 1.0)), 2.0)
    vals = []
    for t in range(-2, 3):
        p = tip[None, :] + s[:, None] * ax[None, :] + t * nrm[None, :] - np.array([xa, ya])[None, :]
        vals.append(map_coordinates(tx, [p[:, 1], p[:, 0]], order=1, mode="nearest"))
    v = np.median(vals, axis=0)
    n3 = max(1, len(v) // 3)
    lo, hi = np.median(v[:n3]), np.median(v[-n3:])
    if hi <= lo:
        return tip
    cross = np.nonzero(v > (lo + hi) / 2)[0]
    return tip + s[cross[0]] * ax if cross.size else tip


def _plate(aux):
    f, lt, thr, smooth = aux
    plate = binary_closing(~smooth, iterations=2)
    lab, n = label(plate)
    if n:
        big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
        plate = binary_fill_holes(lab == big)
    return plate


def back_direction(hole1, hole2, aux):
    """Unit vector from the load line towards the back edge (the plate is much longer that way)."""
    f = aux[0]
    plate = _plate(aux)
    c1 = np.array(hole1[:2]); c2 = np.array(hole2[:2])
    S_px = float(np.linalg.norm(c2 - c1)); lu = (c2 - c1) / S_px; M = (c1 + c2) / 2
    ax = np.array([-lu[1], lu[0]])
    ext = []
    for sgn in (1, -1):
        e = []
        for t in (-0.3, -0.2, 0.2, 0.3):
            q = _walk_out(plate, (M + t * S_px * lu) * f, sgn * ax, 4 * S_px * f)
            e.append(0.0 if q is None else float(np.dot(q - (M + t * S_px * lu) * f, sgn * ax)))
        ext.append(np.median(e))
    return ax if ext[0] >= ext[1] else -ax


def find_tip(img, hole1, hole2, aux=None):
    """Notch tip at full resolution: walking from the load line towards the back edge along the
    crack axis, the smooth notch ends where the fine texture of the metal starts for good."""
    from scipy.ndimage import uniform_filter
    if aux is None:
        _, aux = find_holes(img)
    c1 = np.array(hole1[:2]); c2 = np.array(hole2[:2])
    S_px = float(np.linalg.norm(c2 - c1)); M = (c1 + c2) / 2
    ax = back_direction(hole1, hole2, aux)
    nrm = np.array([-ax[1], ax[0]])
    s = np.arange(0.0, 0.95 * S_px, 1.0)
    pts = M[None, :] + s[:, None] * ax[None, :]
    H, W = img.shape
    pad = 12
    x0 = int(max(0, pts[:, 0].min() - pad)); x1 = int(min(W, pts[:, 0].max() + pad + 1))
    y0 = int(max(0, pts[:, 1].min() - pad)); y1 = int(min(H, pts[:, 1].max() + pad + 1))
    win = img[y0:y1, x0:x1]
    hp = win - gaussian_filter(win, 1.0)
    sd = np.sqrt(uniform_filter(hp ** 2, 7))
    prof = []
    for o in (-2.0, 0.0, 2.0):
        q = pts + o * nrm - np.array([x0, y0])
        prof.append(map_coordinates(sd, [q[:, 1], q[:, 0]], order=1, mode="nearest"))
    v = np.median(prof, axis=0)
    v = np.convolve(v, np.ones(5) / 5, mode="same")
    notch_lvl = np.median(v[int(0.05 * S_px):int(0.22 * S_px)])
    metal_lvl = np.median(v[int(0.65 * S_px):int(0.9 * S_px)])
    if not metal_lvl > 1.3 * notch_lvl:
        return None, ax
    thr = np.sqrt(notch_lvl * metal_lvl)
    run = max(5, int(round(0.06 * S_px)))                  # metal for >= ~1 mm
    for j in range(int(0.05 * S_px), len(v) - run):
        if v[j] > thr and np.median(v[j:j + run]) > thr:
            tip = M + s[j] * ax
            return (float(tip[0]), float(tip[1])), ax
    return None, ax


def find_edges(img, hole1, hole2, ax, aux):
    """Back, front and the two side edges of the plate -> dict of ((x1,y1),(x2,y2))."""
    f = aux[0]
    c1 = np.array(hole1[:2]); c2 = np.array(hole2[:2])
    S_px = float(np.linalg.norm(c2 - c1)); lu = (c2 - c1) / S_px; M = (c1 + c2) / 2
    plate = _plate(aux)

    def edge_from(starts):
        hits = []
        for p0, d in starts:
            q = _walk_out(plate, p0 * f, d, 4 * S_px * f)
            if q is not None:
                hits.append(q / f)
        if len(hits) < 4:
            return None
        hits = np.array(hits)
        for _ in range(2):
            c, u = fit_line(hits)
            res = np.abs((hits - c) @ np.array([-u[1], u[0]]))
            keep = res <= max(3.0 / f, 2.5 * np.median(res))
            if keep.sum() < 4:
                break
            hits = hits[keep]
        c, u = fit_line(hits)
        proj = (hits - c) @ u
        q1, q2 = refine_line(img, c + proj.min() * u, c + proj.max() * u, search=max(8.0, 6.0 / f))
        return (tuple(map(float, q1)), tuple(map(float, q2)))

    out = {}
    out["back"] = edge_from([(M + t * S_px * lu + 0.6 * S_px * ax, ax) for t in np.linspace(-0.9, 0.9, 19)])
    out["front"] = edge_from([(M + t * S_px * lu, -ax) for t in
                              np.r_[np.linspace(-0.95, -0.3, 8), np.linspace(0.3, 0.95, 8)]])
    out["side1"] = edge_from([(M + s * S_px * ax, -lu) for s in np.linspace(0.4, 1.3, 10)])
    out["side2"] = edge_from([(M + s * S_px * ax, lu) for s in np.linspace(0.4, 1.3, 10)])
    return {k: v for k, v in out.items() if v is not None}


def detect_specimen(img, want_edges=True):
    """All features at once -> (feats, notes in Hebrew)."""
    notes = []
    holes, aux = find_holes(img)
    if not holes:
        return {}, ["לא נמצאו שני חורים עגולים – סמן אותם ידנית."]
    feats = {"hole1": holes[0], "hole2": holes[1]}
    tip, ax = find_tip(img, holes[0], holes[1], aux)
    if tip is None:
        notes.append("קצה החריץ לא זוהה – סמן אותו ידנית.")
    else:
        feats["tip"] = tip
    if want_edges:
        edges = find_edges(img, holes[0], holes[1], ax, aux)
        feats.update(edges)
        names = dict(back="אחורי", front="קדמי", side1="צד 1", side2="צד 2")
        for k in names:
            if k not in edges:
                notes.append(f"הקצה ה{names[k]} לא זוהה – סמן אותו ידנית.")
    return feats, notes


# ----------------------------------------------------------------------------- machine photo
_ROT = {"up": 0, "left": 1, "down": 2, "right": 3}     # np.rot90 turns needed so the crack grows 'up'


def _to_rot(p, k, shape):
    """Point (x, y) in the original image -> in np.rot90(img, k)."""
    x, y = p
    H, W = shape
    for _ in range(k % 4):
        x, y = y, W - 1 - x
        H, W = W, H
    return np.array([x, y], float)


def _from_rot(p, k, shape):
    """Point in np.rot90(img, k) -> original image (shape = original shape)."""
    shapes = [shape]
    for _ in range(k % 4):
        shapes.append((shapes[-1][1], shapes[-1][0]))
    x, y = p
    for j in range(k % 4, 0, -1):
        H, W = shapes[j - 1]                      # shape before this rotation step
        x, y = W - 1 - y, x
    return np.array([x, y], float)


def detect_frame(img, grow="up"):
    """Back edge and both side edges of the specimen in a machine photo (the clevises cover the
    holes and the front edge). grow = direction in which the crack grows in the photo.
    Returns dict(back, side1, side2) as point pairs, or {} with a note."""
    from scipy.ndimage import binary_opening
    k = _ROT.get(grow, 0)
    rimg = np.rot90(img, k)
    f = min(1.0, 1000.0 / max(rimg.shape))
    im = zoom(rimg, f, order=1) if f < 1 else rimg
    H, W = im.shape
    mask = binary_opening(im > _otsu(im.ravel()), iterations=2)
    if mask.mean() > 0.9 or mask.mean() < 0.02:
        mask = binary_opening(im < _otsu(im.ravel()), iterations=2)   # dark specimen on a bright background
    ytop = np.array([np.argmax(mask[:, x]) if mask[:, x].any() else H for x in range(W)], float)
    xs = np.arange(W)
    rng = np.random.default_rng(0)
    best = None
    for _ in range(600):
        i, j = sorted(rng.choice(W, 2, replace=False))
        if j - i < 20 or ytop[i] >= H or ytop[j] >= H:
            continue
        a = (ytop[j] - ytop[i]) / (j - i)
        if abs(a) > 0.3:
            continue
        inl = np.abs(ytop - (ytop[i] + a * (xs - i))) < 2.0
        runs = np.diff(np.r_[0, inl.astype(int), 0]); st = np.nonzero(runs == 1)[0]; en = np.nonzero(runs == -1)[0]
        m = int(np.argmax(en - st)); L = en[m] - st[m]
        score = L - 0.05 * float(ytop[st[m]:en[m]].mean())
        if best is None or score > best[0]:
            best = (score, st[m], en[m] - 1)
    if best is None or best[2] - best[1] < 0.15 * W:
        return {}
    _, x0, x1 = best
    a, b = np.polyfit(xs[x0:x1 + 1], ytop[x0:x1 + 1], 1)
    L = x1 - x0
    ymid = a * (x0 + x1) / 2 + b
    left, right = [], []
    for y in range(int(ymid + 0.05 * L), int(ymid + 0.40 * L)):
        if y >= H:
            break
        lo = max(0, int(x0 - 0.1 * L)); hi = min(W, int(x1 + 0.1 * L))
        seg = mask[y, lo:hi]
        if seg.any():
            left.append((lo + np.argmax(seg), y)); right.append((hi - 1 - np.argmax(seg[::-1]), y))
    if len(left) < 8:
        return {}

    def robust(pts, vertical):
        p = np.array(pts, float)
        for _ in range(3):
            if vertical:
                c = np.polyfit(p[:, 1], p[:, 0], 1); res = np.abs(np.polyval(c, p[:, 1]) - p[:, 0])
            else:
                c = np.polyfit(p[:, 0], p[:, 1], 1); res = np.abs(np.polyval(c, p[:, 0]) - p[:, 1])
            keep = res <= max(1.5, 2.5 * np.median(res))
            if keep.sum() < 6:
                break
            p = p[keep]
        if vertical:
            c = np.polyfit(p[:, 1], p[:, 0], 1)
            y1, y2 = p[:, 1].min(), p[:, 1].max()
            return np.array([np.polyval(c, y1), y1]), np.array([np.polyval(c, y2), y2])
        c = np.polyfit(p[:, 0], p[:, 1], 1)
        x1_, x2_ = p[:, 0].min(), p[:, 0].max()
        return np.array([x1_, np.polyval(c, x1_)]), np.array([x2_, np.polyval(c, x2_)])

    back = (np.array([x0, a * x0 + b]), np.array([x1, a * x1 + b]))
    s1 = robust(left, True); s2 = robust(right, True)
    out = {}
    for key, (p, q) in (("back", back), ("side1", s1), ("side2", s2)):
        p0 = _from_rot(p / f, k, img.shape); q0 = _from_rot(q / f, k, img.shape)
        r1, r2 = refine_line(img, p0, q0, search=max(8.0, 4.0 / f))
        out[key] = (tuple(map(float, r1)), tuple(map(float, r2)))
    return out


def find_tip_near(img, pred, u, mmpp, search_mm=3.0):
    """Notch tip close to a predicted position: walking along the crack axis u (towards the back
    edge) the notch is a narrow line that differs from the material beside it; the tip is where
    that difference ends. Returns (x, y) or None."""
    k = 1.0 / mmpp
    u = np.asarray(u, float) / np.linalg.norm(u); nrm = np.array([-u[1], u[0]])
    pred = np.asarray(pred, float)
    # centre the search on the notch: across the axis, 0.4-1.2 mm behind the predicted tip, the
    # notch is the narrow band that differs most from the material around it
    t = np.arange(-1.5 * k, 1.5 * k, 0.5)
    prof = []
    for back in np.linspace(0.4 * k, 1.2 * k, 9):
        p = pred[None, :] - back * u[None, :] + t[:, None] * nrm[None, :]
        prof.append(map_coordinates(img, [p[:, 1], p[:, 0]], order=1, mode="nearest"))
    prof = np.median(prof, axis=0)
    dev = np.abs(prof - np.median(prof))
    dev = np.convolve(dev, np.ones(max(3, int(0.1 * k)) | 1) / (max(3, int(0.1 * k)) | 1), mode="same")
    if dev.max() > 0.05:
        pred = pred + t[int(np.argmax(dev))] * nrm
    s = np.arange(-search_mm * k, search_mm * k, 0.5)
    def band(t0, t1):
        vals = []
        for t in np.linspace(t0, t1, max(2, int(abs(t1 - t0)) + 1)):
            p = pred[None, :] + s[:, None] * u[None, :] + t * nrm[None, :]
            vals.append(map_coordinates(img, [p[:, 1], p[:, 0]], order=1, mode="nearest"))
        return np.median(vals, axis=0)
    cen = band(-0.08 * k, 0.08 * k)
    cen = np.convolve(cen, np.ones(5) / 5, mode="same")
    n4 = max(5, len(cen) // 4)
    notch_lvl, mat_lvl = np.median(cen[:n4]), np.median(cen[-n4:])
    if abs(mat_lvl - notch_lvl) < 0.08:
        return None
    mid = (notch_lvl + mat_lvl) / 2
    is_mat = (cen - mid) * np.sign(mat_lvl - notch_lvl) > 0
    run = max(3, int(0.4 * k / 0.5))                    # material for >= 0.4 mm
    j = None
    for i in range(len(cen) - run):
        if is_mat[i] and is_mat[i:i + run].mean() > 0.9:
            j = i; break
    if j is None:
        return None
    # sub-pixel: interpolate the crossing of the mid level
    if j > 0 and cen[j] != cen[j - 1]:
        fr = (mid - cen[j - 1]) / (cen[j] - cen[j - 1])
        sj = s[j - 1] + float(np.clip(fr, 0, 1)) * (s[j] - s[j - 1])
    else:
        sj = s[j]
    tip = pred + sj * u
    return (float(tip[0]), float(tip[1]))

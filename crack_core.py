"""Measurement engine for Crack Tracker (v2).

Coordinates are 0-based pixel coordinates (x = column, y = row).

A "side" dict describes one camera / specimen face:
    mm_per_px   scale [mm/pixel]
    seed        (x, y) approximate crack tip clicked by the user
    root        (x, y) point on the crack behind the tip (gives the initial direction)
    roi         (x, y, w, h) registration rectangle, or None for automatic
    base        baseline image (2-D float32, 0..1)
    P           parameters (see default_params)
prepare_side() adds u, v, roi, thr_eff, base_len, base_tip_xy, base_path.

How a tip is found (the same in the baseline and in every frame):
  * the crack is TRACED in short straight segments, starting on the clicked line;
    after each segment the direction is re-fitted to the crack already found, so a
    slightly wrong click direction or a gently curving crack is followed;
  * inside each segment a narrow band around the line is searched for the darkest
    connected line; its contrast against the paint right beside it is the evidence;
  * the tip is where that contrast falls to the background level (threshold derived
    from the baseline image ahead of the crack), refined at fine resolution;
  * delta-a = length ALONG the traced crack line from the baseline tip to the tip.
"""
from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter1d

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def default_params() -> dict:
    return dict(
        band_half_mm=0.30,    # the crack line is searched within +-this of the segment line
        bg_gap_mm=0.12,       # background starts this far beside the traced line ...
        bg_width_mm=0.45,     # ... and is this wide (both sides)
        contrast_thr=0.20,    # manual threshold (used when auto_threshold is off)
        auto_threshold=True,  # threshold = paint "contrast" ahead of the baseline crack + margin
        auto_margin=0.05,
        auto_sigma=3.0,
        sensitivity=0,
        notch_start_mm=0.6,
        notch_min_mm=0.05,
        notch_min_contrast=0.06,
        notch_min_drop=0.04,  # notch mode: ... and lower its brightness by at least this (0..1 scale)  # notch mode: a crack must darken the surface at least this much
        bridge_run_mm=0.3,    # evidence after a bridged gap must continue this long    # notch mode: growth below this (1-2 px of tip noise) counts as no crack yet   # notch mode: crack evidence searched this far behind the notch tip        # -5 (only clear lines) .. +5 (also very faint lines); 0 = default
        path_penalty=0.20,     # cost of a sideways step of the traced line (per pixel row)
        smooth_mm=0.40,       # coarse smoothing of the contrast along the crack
        fine_mm=0.10,         # fine smoothing used to place the tip exactly
        max_gap_mm=0.25,      # tolerated weak stretch inside the crack
        seg_mm=1.0,           # tracking segment length
        fit_mm=1.5,           # direction re-fitted to this much crack behind the segment end
        max_turn_deg=12.0,    # max direction change per segment
        check_tip=True,       # extra check of every tip with a lower / higher threshold
        uncertain_mm=0.8,     # flag the tip as uncertain if it moves more than this
        use_prior=True,       # search first for a crack at least as long as in the previous image
        prior_margin_mm=0.2,  # ... up to a little before the previous tip
        bridge_mm=0.5,        # ... bridging at most this long a gap
        base_tip_mode="auto",  # "auto": baseline tip found automatically near the tip click
        click_tol_mm=0.5,     # click mode: max distance click <-> automatic tip to use the click
        base_beyond_mm=3.0,   # in the baseline the tip is searched at most this far past the click
        new_only=True,        # beyond the baseline crack, a line must be new (darker than baseline)
        old_crack_mm=0.25,    # distance from the baseline crack line that counts as 'old crack'
        search_back_mm=2.0,
        search_ahead_mm=40.0,
        roi_h_mm=6.0,         # height of the automatic registration area
        max_shift_px=0,       # registration search range [px]; 0 = automatic
        min_reg_corr=0.70,
        notch_diff="tight",   # notch mode: "tight" = this image vs the baseline at the SAME point (sub-pixel
                              # registration, light smoothing) - works on heavily scratched metal;
                              # "tolerant" = vs the darkest baseline pixel within +-2 px (older method)
        diff_sigma_px=0.6,    # tight mode: smoothing of both images before comparing (noise, sub-pixel shift)
        tight_min_drop=0.008,  # tight mode: minimum absolute brightness drop on the line (~3x the smoothed noise)
        crack_polarity="auto",  # tight mode: "dark" crack, "bright" crack (backlight through it), "auto"
        tight_margin=0.035,   # tight mode: threshold at least this above the measured noise level ...
        tight_min_contrast=0.045,  # ... and never below this (real pairs: noise max ~0.03)
        guard_confirm_n=2,    # growth guard: new growth must persist in this many following images (dust moves)
        guard_base_frac=0.7,  # growth guard: crack vs the crack-free start of the test, threshold x this
        guard_settle_n=6,     # growth guard: rows this far before a confirmed growth get their own value back
    )


# ----------------------------------------------------------------------------- files / images
def read_gray(path: str) -> np.ndarray:
    im = Image.open(path)
    a = np.asarray(im)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    elif a.dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
    elif np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float32) / float(np.iinfo(a.dtype).max)
    else:
        a = a.astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3]
        a = 0.2989 * a[..., 0] + 0.5870 * a[..., 1] + 0.1140 * a[..., 2]
    return np.ascontiguousarray(a, dtype=np.float32)


def natural_key(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name.lower())]


def list_images(folder: str) -> list[str]:
    try:
        names = [n for n in os.listdir(folder)
                 if os.path.splitext(n)[1].lower() in IMAGE_EXT
                 and os.path.isfile(os.path.join(folder, n))]
    except OSError:
        return []
    return [os.path.join(folder, n) for n in sorted(names, key=natural_key)]


def conform(img: np.ndarray, shape) -> np.ndarray:
    """Pad / crop img to `shape` (top-left anchored) so frames of slightly different size
    (e.g. hand-cropped screenshots) can still be registered."""
    if img.shape == tuple(shape):
        return img
    out = np.full(shape, float(np.median(img)), dtype=np.float32)
    h = min(shape[0], img.shape[0]); w = min(shape[1], img.shape[1])
    out[:h, :w] = img[:h, :w]
    return out


def load_frame(path: str, side: dict) -> np.ndarray:
    return conform(read_gray(path), side["base"].shape)


def _movmean(c: np.ndarray, win: int) -> np.ndarray:
    """Moving mean with shrinking endpoints (win odd). NaN samples (no evidence possible there, e.g.
    inside the notch) are left out of the mean instead of counting as zero; all-NaN windows give 0."""
    c = np.asarray(c, float)
    fin = np.isfinite(c)
    if win <= 1:
        return np.where(fin, c, 0.0)
    h = win // 2
    n = len(c)
    i = np.arange(n)
    lo = np.maximum(0, i - h); hi = np.minimum(n, i + h + 1)
    if fin.all():
        cs = np.concatenate([[0.0], np.cumsum(c, dtype=np.float64)])
        return (cs[hi] - cs[lo]) / (hi - lo)
    cs = np.concatenate([[0.0], np.cumsum(np.where(fin, c, 0.0), dtype=np.float64)])
    cn = np.concatenate([[0], np.cumsum(fin)])
    cnt = cn[hi] - cn[lo]
    return np.where(cnt > 0, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), 0.0)


# ----------------------------------------------------------------------------- registration
def default_roi(side: dict) -> tuple:
    """Intact region ahead of the tip, centred on the crack line, kept inside the image."""
    mmpp = side["mm_per_px"]
    u = side["u"]
    H, W = side["base"].shape
    w = min(20.0 / mmpp, 0.45 * W)
    h = min(side["P"].get("roi_h_mm", 6.0) / mmpp, 0.30 * H)
    c = np.asarray(side["seed"], float) + u * max(20.0 / mmpp, 0.6 * w)
    x = min(max(0.0, c[0] - w / 2), W - w - 1)
    y = min(max(0.0, c[1] - h / 2), H - h - 1)
    return (int(round(x)), int(round(y)), int(round(w)), int(round(h)))


def best_roi(side: dict):
    """Registration area with the most 2-D structure (corners / edges in both directions) that is
    on the specimen (side['spec_poly'], if given), not saturated, and away from the crack path
    ahead of the tip. Returns (x, y, w, h) or None."""
    from scipy.ndimage import zoom as _zoom, sobel
    img = side["base"]; mmpp = side["mm_per_px"]
    H, W = img.shape
    f = min(1.0, 1000.0 / max(H, W))
    im = _zoom(img, f, order=1) if f < 1 else img
    g = gaussian_filter(im, 1.0)
    gx, gy = sobel(g, 1), sobel(g, 0)
    Jxx = gaussian_filter(gx * gx, 2.0); Jyy = gaussian_filter(gy * gy, 2.0); Jxy = gaussian_filter(gx * gy, 2.0)
    tr = Jxx + Jyy; det = Jxx * Jyy - Jxy * Jxy
    lmin = np.maximum(tr / 2 - np.sqrt(np.maximum(tr * tr / 4 - det, 0)), 0)      # Shi-Tomasi
    sat = ((im > 0.985) | (im < 0.015)).astype(float)
    ii = np.pad(lmin, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    si = np.pad(sat, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    def rsum(I, x0, y0, x1, y1):
        return I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]
    k = f / mmpp                                         # downscaled px per mm
    rw, rh = int(round(8.0 * k)), int(round(6.0 * k))
    if rw < 8 or rh < 8:
        return None
    u = np.asarray(side["u"], float); seed = np.asarray(side["seed"], float) * f
    poly = side.get("spec_poly")
    if poly is not None:
        from matplotlib.path import Path as _P
        pp = _P(np.asarray(poly, float) * f)
    lim = side["P"].get("search_ahead_mm", 40.0) * k
    path = seed[None, :] + np.linspace(-0.5 * k, lim, 200)[:, None] * u[None, :]
    best = None
    step = max(2, int(round(0.75 * k)))
    Hs, Ws = im.shape
    for y0 in range(0, Hs - rh, step):
        for x0 in range(0, Ws - rw, step):
            x1, y1 = x0 + rw, y0 + rh
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if np.hypot(cx - seed[0], cy - seed[1]) > 25 * k:
                continue
            # keep away from the crack path (1.5 mm margin)
            dx = np.clip(path[:, 0], x0, x1) - path[:, 0]; dy = np.clip(path[:, 1], y0, y1) - path[:, 1]
            if np.min(np.hypot(dx, dy)) < 1.5 * k:
                continue
            if poly is not None:
                corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1), (cx, cy)]
                if not pp.contains_point((cx, cy)) or sum(pp.contains_point(c) for c in corners) < 3:
                    continue
            area = rw * rh
            sc = rsum(ii, x0, y0, x1, y1) / area * (1 - rsum(si, x0, y0, x1, y1) / area) ** 2
            if best is None or sc > best[0]:
                best = (sc, x0, y0)
    if best is None or best[0] <= 0:
        return None
    _, x0, y0 = best
    return (int(round(x0 / f)), int(round(y0 / f)), int(round(rw / f)), int(round(rh / f)))


def auto_max_shift(shape) -> int:
    return int(max(25, round(0.06 * max(shape))))


def register(base: np.ndarray, img: np.ndarray, roi, max_shift: int):
    """Translation of img relative to base: base(x,y) ~ img(x+dx, y+dy).
    Normalised cross-correlation of the ROI over +-max_shift px (FFT). Returns ((dx,dy), corr)."""
    from scipy.signal import fftconvolve
    H, W = base.shape
    m = int(max_shift)
    x1 = int(np.clip(roi[0], 0, W - 2)); y1 = int(np.clip(roi[1], 0, H - 2))
    x2 = int(np.clip(roi[0] + roi[2], x1 + 1, W - 1)); y2 = int(np.clip(roi[1] + roi[3], y1 + 1, H - 1))
    T = base[y1:y2 + 1, x1:x2 + 1].astype(np.float64)
    if T.size < 64:
        return (0.0, 0.0), 0.0
    T = T - T.mean(); tn = np.linalg.norm(T)
    if tn < 1e-9:
        return (0.0, 0.0), 0.0
    padded = np.pad(img.astype(np.float64), m, mode="edge")
    win = padded[y1:y2 + 1 + 2 * m, x1:x2 + 1 + 2 * m]
    num = fftconvolve(win, T[::-1, ::-1], mode="valid")
    ones = np.ones_like(T)
    s1 = fftconvolve(win, ones, mode="valid"); s2 = fftconvolve(win * win, ones, mode="valid")
    C = num / (np.sqrt(np.maximum(s2 - s1 * s1 / T.size, 1e-12)) * tn)
    iy, ix = np.unravel_index(np.argmax(C), C.shape)
    corr = float(C[iy, ix])

    def parab(a, b, c):
        den = a - 2 * b + c
        return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if abs(den) > 1e-12 else 0.0

    n0, n1 = C.shape
    sx = parab(C[iy, ix - 1], C[iy, ix], C[iy, ix + 1]) if 0 < ix < n1 - 1 else 0.0
    sy = parab(C[iy - 1, ix], C[iy, ix], C[iy + 1, ix]) if 0 < iy < n0 - 1 else 0.0
    return (ix - m + sx, iy - m + sy), corr


# ----------------------------------------------------------------------------- crack tracing
def _darkest_path(Z: np.ndarray, penalty: float) -> np.ndarray:
    """Viterbi: row per column minimising sum(Z) + penalty*|row step|, steps of <=1 row."""
    R, N = Z.shape
    cost = Z[:, 0].astype(np.float64).copy()
    back = np.zeros((R, N), dtype=np.int8)
    for j in range(1, N):
        up = np.concatenate([[np.inf], cost[:-1]]) + penalty
        dn = np.concatenate([cost[1:], [np.inf]]) + penalty
        stack = np.vstack([up, cost, dn])
        k = np.argmin(stack, axis=0)
        cost = Z[:, j] + stack[k, np.arange(R)]
        back[:, j] = k - 1
    rows = np.empty(N, dtype=np.int64)
    rows[-1] = int(np.argmin(cost))
    for j in range(N - 1, 0, -1):
        rows[j - 1] = rows[j] + back[rows[j], j]
    return rows


def _segment(img, origin, u, s0, s1, P, mmpp, gate=None):
    """Sample a straight strip along u from origin (s in [s0,s1] px), trace the crack line in
    the narrow band and return its contrast against the paint right beside it."""
    v = np.array([-u[1], u[0]])
    bh = P["band_half_mm"] / mmpp
    inner = P["bg_gap_mm"] / mmpp
    bgw = P["bg_width_mm"] / mmpp
    t_out = int(np.ceil(bh + inner + bgw)) + 1
    s = np.arange(int(np.floor(s0)), int(np.ceil(s1)) + 1, dtype=float)
    t = np.arange(-t_out, t_out + 1, dtype=float)
    S, T = np.meshgrid(s, t)
    X = origin[0] + S * u[0] + T * v[0]
    Y = origin[1] + S * u[1] + T * v[1]
    strip = map_coordinates(img, [Y.ravel(), X.ravel()], order=1, mode="constant",
                            cval=np.nan).reshape(S.shape)
    band = np.abs(t) <= bh
    tb = t[band]
    notch_gate = gate is not None and len(gate) > 4 and gate[4] is not None
    if notch_gate and P.get("notch_diff", "tight") == "tight":
        return _segment_tight(img, origin, u, v, s, t, S, T, strip, band, tb, P, mmpp, gate)
    bstrip = None
    if notch_gate:
        # machined notch (bare / scratched metal): follow the NEW darkening (this image relative to
        # the baseline) - old marks that are darker than a fresh crack must not pull the path away
        from scipy.ndimage import grey_erosion
        bd = gate[1]
        bstrip = map_coordinates(gate[0], [Y.ravel() - bd[1], X.ravel() - bd[0]], order=1,
                                 mode="nearest").reshape(S.shape)
        bero = grey_erosion(bstrip, size=(3, 3))          # tolerate a pixel of mis-registration
        ratio = strip / np.maximum(bero, 1e-4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rmed = np.nanmedian(ratio, axis=0)               # cancels a global exposure change
        Zb = ratio[band] / np.maximum(rmed, 1e-4)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            colmed = np.nanmedian(strip, axis=0)
        Zb = strip[band] / np.maximum(colmed, 1e-4)
    Zb[~np.isfinite(Zb)] = 1.0
    path = _darkest_path(uniform_filter1d(Zb, size=3, axis=1, mode="nearest"), P["path_penalty"])
    tp = tb[path]
    rel = np.abs(t[:, None] - tp[None, :])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bg = np.nanmedian(np.where((rel >= inner) & (rel <= inner + bgw), strip, np.nan), axis=0)
        dark = np.nanmin(np.where(rel <= 1.0, strip, np.nan), axis=0)
    c = 1.0 - dark / np.maximum(bg, 1e-4)
    c[~np.isfinite(c)] = 0.0
    xy = origin[None, :] + s[:, None] * u[None, :] + tp[:, None] * v[None, :]
    if gate is not None:
        # Away from the baseline crack, only NEW darkening counts: a line that was already
        # there in the baseline (paint mark, another ply) is not crack growth.
        base, d, tree, near = gate[:4]
        bx = xy[:, 0] - d[0]; by = xy[:, 1] - d[1]
        bvals = []
        for o in (-2.0, -1.0, 0.0, 1.0, 2.0):
            for ou in ((-2.0, 0.0, 2.0) if notch_gate else (0.0,)):   # notch mode: +-2 px along the
                # crack too - a sharp edge (notch end) that shifts by a pixel is not new darkening
                bvals.append(map_coordinates(base, [by + o * v[1] + ou * u[1], bx + o * v[0] + ou * u[0]],
                                             order=1, mode="nearest"))
        bline = np.min(bvals, axis=0)      # darkest baseline value there (old line -> no 'new' contrast)
        c_new = 1.0 - dark / np.maximum(bline, 1e-4)
        c_new[~np.isfinite(c_new)] = 0.0
        on_old = tree.query(np.column_stack([bx, by]))[0] <= near
        if len(gate) > 4 and gate[4] is not None:
            # machined notch: only the notch itself is 'old' - nothing ahead of its tip
            sx, sy, ux, uy = gate[4]
            on_old &= ((bx - sx) * ux + (by - sy) * uy) <= 0
        if notch_gate:
            # ahead of a machined notch: crack evidence = darkening relative to the baseline at the
            # same place, normalised by the brightness change of the surroundings (exposure). The
            # notch walls / lobes beside the crack start then do not matter.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                bgb = np.nanmedian(np.where((rel >= inner) & (rel <= inner + bgw), bstrip, np.nan), axis=0)
            rbg = np.clip(bg / np.maximum(bgb, 1e-4), 0.5, 2.0)
            rbg[~np.isfinite(rbg)] = 1.0
            c_n = 1.0 - (dark / np.maximum(bline, 1e-4)) / rbg
            c_n[~np.isfinite(c_n)] = 0.0
            # and a real absolute drop in brightness (noise on a dark edge gives a large ratio but
            # only a tiny drop; a crack in the surface darkens it by far more)
            drop = bline - dark / rbg
            c_n = np.where(drop >= P.get("notch_min_drop", 0.04), c_n, np.minimum(c_n, 0.0))
            # inside the notch (behind its tip) there is no evidence at all: through the notch one
            # sees the background, which does not move with the specimen, so it 'changes' after
            # registration - the crack can only start at the notch tip and grow forward
            sx, sy, ux, uy = gate[4]
            behind = ((bx - sx) * ux + (by - sy) * uy) <= 0
            nm = gate[5] if len(gate) > 5 else None
            if nm is not None:
                m, mx, my = nm
                xi = np.round(bx).astype(int) - mx; yi = np.round(by).astype(int) - my
                ok = (xi >= 0) & (yi >= 0) & (xi < m.shape[1]) & (yi < m.shape[0])
                inside = np.zeros(len(bx), bool)
                inside[ok] = m[yi[ok], xi[ok]]
                behind |= inside                       # inside the notch (incl. its lobes): no evidence
            c = np.where(behind, 0.0, c_n)
        else:
            # at least half of the line's darkness must be new (an old line that darkens a bit is not crack)
            c = np.where(on_old, c, np.minimum(c, 2.0 * c_new))
    return s, c, xy


def _segment_tight(img, origin, u, v, s, t, S, T, strip, band, tb, P, mmpp, gate):
    """Notch mode, crack evidence = darkening of THIS image relative to the baseline at the same
    point (registered sub-pixel), exposure-normalised by the surroundings. Both strips are lightly
    smoothed (sensor noise, residual sub-pixel shift). Old marks / scratches cancel exactly, so the
    method works on heavily scratched bare metal where a 'darkest pixel nearby' comparison hides
    the crack."""
    from scipy.ndimage import gaussian_filter
    base, d = gate[0], gate[1]
    X = origin[0] + S * u[0] + T * v[0]
    Y = origin[1] + S * u[1] + T * v[1]
    bstrip = map_coordinates(base, [Y.ravel() - d[1], X.ravel() - d[0]], order=1,
                             mode="nearest").reshape(S.shape)
    sig = float(P.get("diff_sigma_px", 0.6))
    fill = float(np.nanmedian(strip)) if np.isfinite(strip).any() else 0.5
    st = gaussian_filter(np.where(np.isfinite(strip), strip, fill), sig)
    bs = gaussian_filter(bstrip, sig)
    pol = P.get("crack_polarity", "auto")
    # dark crack (shadow in the face) = darkening, 1 - this/base; bright crack (light from behind through
    # an open crack) = brightening, 1 - base/this (bounded; a saturated face can not get brighter, so it
    # gives no evidence instead of noise). "auto": per segment, the one with more evidence - an open
    # crack near the notch can be bright and the tight part near its tip dark
    cands = []
    for kind in (("dark", "bright") if pol == "auto" else (pol,)):
        a, b = (st, bs) if kind == "dark" else (bs, st)
        tp, c = _tight_evidence(a, b, t, band, tb, P)
        xy = origin[None, :] + s[:, None] * u[None, :] + tp[:, None] * v[None, :]
        behind = _behind_notch(xy, d, gate)
        # NaN, not 0: the smoothing along the crack must not dilute a short new crack with the notch
        c = np.where(behind, np.nan, c)
        score = float(np.nansum(np.clip(c, 0.0, None))) if np.isfinite(c).any() else 0.0
        cands.append((score, c, xy))
    _, c, xy = max(cands, key=lambda q: q[0])
    return s, c, xy


def _tight_evidence(st, bs, t, band, tb, P):
    """Darkening of st relative to bs on the darkest line in the band: (line offsets, evidence)."""
    ratio = st / np.maximum(bs, 1e-4)
    rmed = np.median(ratio, axis=0)                        # exposure change of the surroundings
    rn = ratio / np.maximum(rmed, 1e-4)
    path = _darkest_path(uniform_filter1d(rn[band], size=3, axis=1, mode="nearest"), P["path_penalty"])
    tp = tb[path]
    rel = np.abs(t[:, None] - tp[None, :])
    near = np.where(rel <= 1.0, rn, np.inf)
    j = np.argmin(near, axis=0)
    cols = np.arange(rn.shape[1])
    c = 1.0 - rn[j, cols]
    drop = bs[j, cols] * c                                 # absolute brightness change on the line
    c = np.where(drop >= P.get("tight_min_drop", 0.008), c, np.minimum(c, 0.0))
    c[~np.isfinite(c)] = 0.0
    return tp, c


def _behind_notch(xy, d, gate):
    """Points behind the notch tip or inside the notch (incl. its lobes): no crack evidence there -
    through the notch one sees the background, which does not move with the specimen."""
    bx = xy[:, 0] - d[0]; by = xy[:, 1] - d[1]
    sx, sy, ux, uy = gate[4]
    behind = ((bx - sx) * ux + (by - sy) * uy) <= 0
    nm = gate[5] if len(gate) > 5 else None
    if nm is not None:
        m, mx, my = nm
        xi = np.round(bx).astype(int) - mx; yi = np.round(by).astype(int) - my
        ok = (xi >= 0) & (yi >= 0) & (xi < m.shape[1]) & (yi < m.shape[0])
        inside = np.zeros(len(bx), bool)
        inside[ok] = m[yi[ok], xi[ok]]
        behind |= inside
    return behind


def _walk(c_s, start, thr, max_gap):
    """Last index of the crack run starting at `start` (gaps <= max_gap allowed); -1 if none."""
    last, gap = -1, 0
    for k in range(start, len(c_s)):
        if c_s[k] >= thr:
            last, gap = k, 0
        else:
            gap += 1
            if gap > max_gap:
                break
    return last


def _fit_dir(pts, u_prev, max_turn_deg):
    """Principal direction of pts, oriented like u_prev, turn limited to max_turn_deg."""
    if len(pts) < 5:
        return u_prev
    q = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    un = vt[0]
    if un @ u_prev < 0:
        un = -un
    ang = np.arctan2(u_prev[0] * un[1] - u_prev[1] * un[0], u_prev @ un)
    lim = np.radians(max_turn_deg)
    ang = float(np.clip(ang, -lim, lim))
    ca, sa = np.cos(ang), np.sin(ang)
    return np.array([ca * u_prev[0] - sa * u_prev[1], sa * u_prev[0] + ca * u_prev[1]])


def _arclen(xy, mmpp):
    """Cumulative length along a (lightly smoothed) traced line, in pixels."""
    if len(xy) < 2:
        return np.zeros(len(xy))
    w = max(1, int(round(0.5 / mmpp))) | 1
    xs = _movmean(xy[:, 0], w); ys = _movmean(xy[:, 1], w)
    return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])


def _walk2(c_s, start, thr, max_gap, g, reach, max_bridge=10 ** 9, min_run=1):
    """Like _walk, but gaps are bridged (no stop) while the position g[k] is still short of
    `reach` (the crack length of the previous image). Evidence found after a bridged gap only
    counts once it continues for min_run samples (a single noisy spot beyond a gap is not crack -
    otherwise noise near the previous tip would make the crack creep longer image after image).
    Returns (last_good, broke, stop_index)."""
    last, gap, run, bridged = -1, 0, 0, False
    for k in range(start, len(c_s)):
        if c_s[k] >= thr:
            run += 1
            if not bridged or run >= min_run:
                last, bridged = k, False
            gap = 0
        else:
            if gap == 0 and run and bridged:
                run = 0                    # evidence after the bridge was too short: ignore it
            gap += 1
            if gap > max_gap:
                bridged = True; run = 0
            if gap > max_gap and (g[k] >= reach or gap > max_bridge):
                return last, True, k
    return last, False, len(c_s) - 1


def trace_crack(img: np.ndarray, side: dict, d=(0.0, 0.0), thr=None, limit_mm=None,
                min_reach_px=None) -> dict:
    """Trace the crack from the clicked root towards and past the clicked tip.
    min_reach_px: distance (along the crack direction, from the clicked tip) that the crack
    reached in the previous image - the search first assumes the crack is at least that long
    (gaps are bridged up to there); if there is no crack evidence that far it ends shorter.
    Returns dict(found, tip_xy, path_xy (root..tip), length (px), reach (px), ...)."""
    P = side["P"]; mmpp = side["mm_per_px"]; k = 1.0 / mmpp
    thr = side.get("thr_eff", P["contrast_thr"]) if thr is None else thr
    # bridge only up to a little before the previous tip, and never more than bridge_mm at once
    reach = -np.inf if min_reach_px is None else float(min_reach_px) - P.get("prior_margin_mm", 0.2) * k
    max_bridge = max(1, int(round(P.get("bridge_mm", 1.0) * k)))
    d = np.asarray(d, float)
    seed = np.asarray(side["seed"], float) + d
    root = np.asarray(side["root"], float) + d
    u = side["u"].copy()
    gate = None
    if P.get("new_only", True) and side.get("_tree") is not None and img is not side["base"]:
        gate = (side["base"], d, side["_tree"], P.get("old_crack_mm", 0.25) * k,
                (side["seed"][0], side["seed"][1], side["u"][0], side["u"][1]) if side.get("notch") else None,
                side.get("_notch_mask") if side.get("notch") else None)
    w = max(1, int(round(P["smooth_mm"] * k))); win = 2 * w + 1
    wf = max(1, int(round(P["fine_mm"] * k))); winf = 2 * wf + 1
    max_gap = max(1, int(round(P["max_gap_mm"] * k)))
    seg = P["seg_mm"] * k
    limit = P["search_ahead_mm"] * k if limit_mm is None else limit_mm * k

    # --- first segment: on the clicked line, from the root to just past the seed
    notch = bool(side.get("notch"))
    back = max(P["search_back_mm"] * k, float(np.linalg.norm(seed - root)))
    if notch:
        # the crack can only start at the notch tip: trace from just before it, not from the load line -
        # a line traced through the whole notch (background, light from behind) arrives at the tip
        # off the crack
        back = (P.get("notch_start_mm", 0.6) + 0.1) * k
    s, c, xy = _segment(img, seed, u, -back, min(0.3 * k, limit), P, mmpp, gate)
    tight = notch and gate is not None and P.get("notch_diff", "tight") == "tight"
    # a new crack right at the notch tip is short: a finer window there, so that 0.2 mm of crack is not
    # averaged away (the growth guard still demands it in the next image too)
    cs = _movmean(c, 2 * max(1, int(round(P.get("notch_smooth_mm", 0.2) * k))) + 1 if tight else win)
    start_c = np.nonzero((s <= 0) & (cs >= thr))[0]
    if notch:
        # machined notch: the crack can only start at the notch tip - look for crack evidence
        # only in the last notch_start_mm before the clicked tip (not in the wide notch itself)
        start_c = np.nonzero((s <= 0.3 * k) & (s >= -P.get("notch_start_mm", 0.6) * k) & (cs >= thr))[0]
        if not start_c.size and min_reach_px is not None and np.isfinite(min_reach_px) \
                and min_reach_px > 4 * P.get("notch_min_mm", 0.05) * k:
            # a crack was already found in the previous image and a crack does not heal: when its start
            # is faint in this one, start at the notch tip anyway and bridge up to the previous length
            start_c = np.nonzero(s >= 0)[0][:1]
        if not start_c.size:
            return _notch_only(side, root, seed, u, thr, mmpp)
    if not start_c.size:
        return dict(found=False, tip_xy=(np.nan, np.nan), path_xy=None, length=np.nan, thr=thr,
                    reach=np.nan)
    i0 = int(start_c[0])
    min_run = max(1, int(round(P.get("bridge_run_mm", 0.3) * k))) if side.get("notch") else 1
    last, broke, stop = _walk2(cs, i0, thr, max_gap, s, reach, max_bridge, min_run)
    chunks = [xy[i0:(stop if not broke else last) + 1]]
    good_n = last - i0 + 1                      # number of path points up to the last good one
    n_pts = len(chunks[0])
    good = (s, c, xy, last, s[last])           # segment holding the last good point, its g
    g_stop = s[stop if not broke else last]
    if side.get("u_tip") is not None:          # crack direction measured at the baseline tip
        u = np.asarray(side["u_tip"], float).copy()
    # --- continue segment by segment while the crack (or the bridge) runs to the segment end
    while not broke and g_stop < limit:
        allp = np.vstack(chunks)
        tip = allp[-1]
        near = allp[np.linalg.norm(allp - tip, axis=1) <= P["fit_mm"] * k]
        cands = [_fit_dir(near, u, P["max_turn_deg"]), u, side["u"]]
        best = None
        ov = 1.0 * k
        for ci, uc in enumerate(cands):
            if any(np.allclose(uc, cands[j]) for j in range(ci)):
                continue
            s2, c2, xy2 = _segment(img, tip, uc, -ov, seg, P, mmpp, gate)
            cs2 = _movmean(c2, win)
            j0 = int(np.searchsorted(s2, 0.0))
            g2 = g_stop + s2
            last2, broke2, stop2 = _walk2(cs2, max(0, j0 - wf), thr, max_gap, g2, reach, max_bridge, min_run)
            if last2 > j0 or (not broke2 and g2[-1] < reach):   # continues (or still bridging)
                best = (s2, c2, xy2, j0, last2, broke2, stop2, uc, g2)
                break
        if best is None:
            break
        s2, c2, xy2, j0, last2, broke, stop2, u, g2 = best
        end = stop2 if not broke else last2
        if end > j0:
            chunks.append(xy2[j0 + 1:end + 1]); n_pts += end - j0
        if last2 > j0:
            good_n = n_pts - (end - last2)
            good = (s2, c2, xy2, last2, g2[last2])
        g_stop = g2[end] if end > j0 else g_stop
        if end <= j0:
            break
    # --- tip = last point with crack evidence; refine it at fine resolution
    cur_s, cur_c, cur_xy, cur_last, g_tip = good
    pts = np.vstack(chunks)[:max(1, good_n)]
    cf = _movmean(cur_c, winf)
    lo = max(0, cur_last - 2 * w); hi = min(len(cf) - 1, cur_last + wf)
    t_tip = thr
    if tight:
        # the tip where the crack's own evidence falls to HALF its level along the last 0.4 mm (unbiased
        # for a smoothed step), not where it falls below the noise threshold (that cuts the crack short
        # by about half the smoothing window) - never below 0.6 x threshold
        near = cf[max(0, cur_last - int(round(0.4 * k))):cur_last + 1]
        near = near[near >= thr]
        if near.size >= 3:
            t_tip = max(0.5 * float(np.median(near)), 0.6 * thr)
    gidx = np.nonzero(cf[lo:hi + 1] >= t_tip)[0]
    tip_i = lo + int(gidx[-1]) if gidx.size else cur_last
    tip_xy = cur_xy[tip_i]
    if tip_i > cur_last:
        pts = np.vstack([pts, cur_xy[cur_last + 1:tip_i + 1]])
    elif tip_i < cur_last:
        drop = cur_last - tip_i
        pts = pts[:-drop] if drop < len(pts) else pts[:1]
    if tip_i + 1 < len(cf) and cf[tip_i] > cf[tip_i + 1]:
        f = float(np.clip((cf[tip_i] - t_tip) / (cf[tip_i] - cf[tip_i + 1]), 0, 1))
        tip_xy = cur_xy[tip_i] + f * (cur_xy[tip_i + 1] - cur_xy[tip_i])
        pts = np.vstack([pts[:-1], tip_xy]) if len(pts) else np.array([tip_xy])
    if len(pts) > 5:      # smooth the traced line (pixel jitter) but keep the exact tip
        ws = max(1, int(round(0.3 / mmpp))) | 1
        sm = np.column_stack([_movmean(pts[:, 0], ws), _movmean(pts[:, 1], ws)])
        sm[-1] = pts[-1]
        pts = sm
    if notch and float(np.dot(np.asarray(tip_xy, float) - seed, side["u"])) < P.get("notch_min_mm", 0.05) * k:
        return _notch_only(side, root, seed, u, thr, mmpp)      # nothing beyond the notch tip yet
    if notch:
        pts = np.vstack([_line_pts(root, seed), pts])            # notch (load line .. tip) + crack
    L = _arclen(pts, mmpp)
    H, W = img.shape
    at_edge = not (2 <= tip_xy[0] <= W - 3 and 2 <= tip_xy[1] <= H - 3)
    return dict(found=True, tip_xy=(float(tip_xy[0]), float(tip_xy[1])), path_xy=pts,
                cumlen=L, length=float(L[-1]), thr=thr, u_end=u, at_edge=at_edge,
                reach=float(g_tip + (tip_i - cur_last)),
                beyond=limit_mm is None and g_tip >= limit)


def _line_pts(a, b, step=2.0):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = max(2, int(np.linalg.norm(b - a) / step) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return a[None, :] + t * (b - a)[None, :]


def _notch_only(side, root, seed, u, thr, mmpp):
    """Trace result when no crack has grown from the machined notch yet: tip = notch tip."""
    pts = _line_pts(root, seed)
    L = _arclen(pts, mmpp)
    return dict(found=True, tip_xy=(float(seed[0]), float(seed[1])), path_xy=pts, cumlen=L,
                length=float(L[-1]), thr=thr, u_end=np.asarray(side["u"], float).copy(), at_edge=False,
                reach=0.0, beyond=False, notch_only=True)


def _notch_mask(side):
    """The notch itself in the baseline image (the dark slot ending at the marked tip, with its
    uneven 'lobes'), dilated by a few pixels. A crack can only appear outside it.
    Returns (mask, x0, y0) for a window around the tip, or None."""
    from scipy.ndimage import binary_dilation, label
    base = side["base"]; k = 1.0 / side["mm_per_px"]
    seed = np.asarray(side["seed"], float); u = np.asarray(side["u"], float)
    R = int(2.5 * k)
    H, W = base.shape
    x0, y0 = int(max(0, seed[0] - R)), int(max(0, seed[1] - R))
    x1, y1 = int(min(W, seed[0] + R + 1)), int(min(H, seed[1] + R + 1))
    patch = base[y0:y1, x0:x1]
    if patch.size < 100:
        return None
    def med_along(a0, a1):
        pts = seed[None, :] + np.linspace(a0, a1, 20)[:, None] * k * u[None, :]
        return float(np.median(map_coordinates(base, [pts[:, 1], pts[:, 0]], order=1, mode="nearest")))
    behind, ahead = med_along(-1.0, -0.2), med_along(0.2, 1.0)
    if abs(ahead - behind) < 0.15:
        return None                     # no distinct slot behind the tip (e.g. a tip marked in metal)
    lo, hi = min(behind, ahead), max(behind, ahead)
    dark = (patch < (lo + hi) / 2) if behind < ahead else (patch > (lo + hi) / 2)
    lab, n = label(dark)
    keep = np.zeros_like(dark)
    for t in np.linspace(0.1 * k, 1.0 * k, 10):        # components that contain the slot behind the tip
        p = seed - t * u
        xi, yi = int(round(p[0])) - x0, int(round(p[1])) - y0
        if 0 <= yi < lab.shape[0] and 0 <= xi < lab.shape[1] and lab[yi, xi]:
            keep |= lab == lab[yi, xi]
    if not keep.any() or keep.sum() > 0.5 * keep.size:
        return None                     # not a compact slot (scratch network etc.)
    return binary_dilation(keep, iterations=3), x0, y0


def _set_threshold(side, origin, u, img2=None, gate2=None):
    """Threshold = paint-texture 'line contrast' + margin, measured on lines parallel to the
    crack (+-1 and +-1.5 mm to either side), 1..8 mm ahead of `origin` along `u`. The two
    lines with the lowest level are pooled (a line hitting a ruler or an edge is ignored)."""
    k = 1.0 / side["mm_per_px"]
    v = np.array([-u[1], u[0]])
    lines = []
    for off in (-2.0, -1.5, -1.0, 1.0, 1.5, 2.0):
        if img2 is None:
            c = _noise_along(side["base"], side, origin + v * off * k, u, 1.0, 8.0)
        else:      # 'new darkening' noise between the baseline and a second crack-free image
            c = _noise_along(img2, side, origin + gate2[1] + v * off * k, u, 1.0, 8.0, gate2)
        if c.size > 20:
            lines.append((float(np.median(c)), c))
    if not lines:
        return
    lines.sort(key=lambda m: m[0])
    c = np.concatenate([m[1] for m in lines[:max(2, len(lines) // 2)]])
    med = float(np.median(c)); mad = 1.4826 * float(np.median(np.abs(c - med)))
    side["noise_level"] = med
    side["thr_eff"] = med + max(side["P"]["auto_margin"], side["P"]["auto_sigma"] * mad)
    P = side["P"]
    if img2 is not None and side.get("notch") and P.get("notch_diff", "tight") == "tight":
        # same-point difference: the noise is small and well measured (real frame pairs: MAD ~0.003,
        # max ~0.03) - the threshold follows it: well above the worst noise seen, not a fixed margin
        side["thr_eff"] = max(med + P.get("tight_margin", 0.035), med + 10.0 * mad,
                              1.3 * float(np.percentile(c, 99.9)))
        side["tight_noise"] = True


def notch_noise_threshold(side, img2, name="", finish=True):
    """Notch mode: threshold from the new-darkening noise between the baseline and a second
    crack-free image img2 (registered). Also used during a live run when the folder held only the
    baseline at setup time - without it the threshold comes from the scratches' own contrast, far
    too high on bare metal. Returns True when set."""
    from scipy.spatial import cKDTree
    ms = int(side["P"]["max_shift_px"]) or auto_max_shift(img2.shape)
    d2, corr2 = register(side["base"], img2, side["roi"], ms)
    if corr2 < side["P"]["min_reg_corr"]:
        return False
    side["base2"] = (img2, np.asarray(d2, float))      # second crack-free reference (growth guard)
    tree = cKDTree(_line_pts(side["root"], side["seed"]))
    k = 1.0 / side["mm_per_px"]
    _set_threshold(side, np.asarray(side["seed"], float), side["u"], img2,
                   (side["base"], np.asarray(d2, float), tree, side["P"].get("old_crack_mm", 0.25) * k,
                    (side["seed"][0], side["seed"][1], side["u"][0], side["u"][1])))
    side["noise_from"] = name or "next image"
    if finish:
        _notch_thr_finish(side)
    return True


def _notch_thr_finish(side):
    if "thr_eff" in side:
        # a very clean surface (or a saturated image) gives an almost-zero noise threshold:
        # keep a minimum contrast so that sensor noise can never look like a crack
        floor = side["P"].get("tight_min_contrast", 0.045) if side.get("tight_noise") else \
            side["P"].get("notch_min_contrast", 0.06)
        side["thr_eff"] = max(side["thr_eff"], floor, side.get("noise_level", 0.0) + floor)
    if "thr_eff" in side and thr_scale(side["P"]) != 1.0:
        noise = side.get("noise_level", 0.0)
        side["thr_eff"] = noise + (side["thr_eff"] - noise) * thr_scale(side["P"])
        # even at the highest sensitivity keep a minimum (sensor noise must never be a crack)
        side["thr_eff"] = max(side["thr_eff"], 0.5 * side["P"].get("notch_min_contrast", 0.06))


def thr_scale(P):
    """Sensitivity -> factor on the margin above the paint-texture level (each step ~19%)."""
    return float(2.0 ** (-float(P.get("sensitivity", 0)) / 4.0))


def _noise_along(img, side, origin, u, a_mm, b_mm, gate=None):
    """Smoothed line contrast of intact paint between a_mm and b_mm ahead of origin."""
    k = 1.0 / side["mm_per_px"]
    s, c, xy = _segment(img, origin, u, a_mm * k, b_mm * k, side["P"], side["mm_per_px"], gate)
    H, W = img.shape
    inside = (xy[:, 0] >= 2) & (xy[:, 0] <= W - 3) & (xy[:, 1] >= 2) & (xy[:, 1] <= H - 3)
    return _movmean(c, 2 * max(1, int(round(side["P"]["smooth_mm"] * k))) + 1)[inside]


def snap_to_dark(img, pt, r):
    """Darkest point (lightly smoothed) within radius r px of pt."""
    from scipy.ndimage import uniform_filter
    x, y = float(pt[0]), float(pt[1])
    r = max(2, int(round(r)))
    H, W = img.shape
    x0, x1 = max(0, int(x) - r), min(W, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(H, int(y) + r + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return (x, y)
    win = uniform_filter(img[y0:y1, x0:x1].astype(float), 3, mode="nearest")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    win[(xx - x) ** 2 + (yy - y) ** 2 > r * r] = np.inf
    iy, ix = np.unravel_index(np.argmin(win), win.shape)
    return (float(x0 + ix), float(y0 + iy))


def prepare_side(side: dict) -> dict:
    """Axis, ROI, threshold and the baseline tip. Raises ValueError with a readable message."""
    side = dict(side)
    side["P"] = {**default_params(), **(side.get("P") or {})}
    # snap both clicks onto the crack line (darkest point within ~0.35 mm)
    # the first click is the user's measurement START point (e.g. the notch tip): kept exactly
    side["ref"] = tuple(float(x) for x in (side.get("ref") or side["root"]))
    if side.get("notch"):
        # machined notch: start point (on the load line) and notch tip are taken exactly as set
        side["root"] = tuple(float(x) for x in side["ref"])
        side["seed"] = tuple(float(x) for x in side["seed"])
    else:
        snap = min(4.0, 0.2 / side["mm_per_px"])     # only fixes small click errors (pixels)
        side["root"] = snap_to_dark(side["base"], side["ref"], snap)
        side["seed"] = snap_to_dark(side["base"], side["seed"], snap)
    seed = np.asarray(side["seed"], float); root = np.asarray(side["root"], float)
    u = seed - root
    if np.linalg.norm(u) <= 3:
        raise ValueError("Click the crack point farther behind the tip.")
    side["u"] = u / np.linalg.norm(u)
    side["v"] = np.array([-side["u"][1], side["u"][0]])
    if not side.get("roi"):
        side["roi"] = (best_roi(side) if side.get("notch") else None) or default_roi(side)
    side.pop("thr_eff", None); side.pop("tight_noise", None)
    lim = side["P"]["base_beyond_mm"]          # trust the clicked tip in the baseline
    auto = side["P"].get("auto_threshold", True)
    if auto:
        _set_threshold(side, np.asarray(side["seed"], float), side["u"])
        nf = side.get("noise_file")
        if side.get("notch") and nf and os.path.exists(nf):
            # bare / scratched metal: pre-existing scratches never count (only NEW darkening does),
            # so the threshold is set from the new-darkening noise between two crack-free images
            try:
                notch_noise_threshold(side, conform(read_gray(nf), side["base"].shape), os.path.basename(nf),
                                      finish=False)
            except Exception:
                pass
    if side.get("notch"):
        # machined notch: the baseline tip IS the marked notch tip (no crack traced in the baseline)
        _notch_thr_finish(side)
        tr = _notch_only(side, np.asarray(side["root"], float), np.asarray(side["seed"], float),
                         side["u"], side.get("thr_eff", side["P"]["contrast_thr"]), side["mm_per_px"])
        side["base_len"] = tr["length"]; side["base_tip_xy"] = tr["tip_xy"]; side["click_note"] = ""
        side["base_path"] = tr["path_xy"]; side["u_tip"] = tr["u_end"]
        side["a0_px"] = _a_px(side, tr, (0.0, 0.0))
        side["_notch_mask"] = _notch_mask(side)
        from scipy.spatial import cKDTree
        side["_tree"] = cKDTree(np.asarray(tr["path_xy"], float))
        return side
    tr = trace_crack(side["base"], side, (0.0, 0.0), limit_mm=lim)
    for frac in (0.5, 0.25):                       # faint crack at the click: retry lower
        if tr["found"] or "thr_eff" not in side:
            break
        side["thr_eff"] = side["noise_level"] + frac * (side["thr_eff"] - side["noise_level"])
        tr = trace_crack(side["base"], side, (0.0, 0.0), limit_mm=lim)
    if not tr["found"]:
        raise ValueError("No crack line found near the clicked tip in the baseline image. "
                         "Click on the visible dark crack, or lower the threshold.")
    if auto and not side.get("notch"):
        # Re-estimate the threshold from the crack that was actually FOUND (its tip and its
        # direction) instead of from the clicks, and repeat until the tip settles. This makes
        # the reference tip independent of exactly where / in which direction the user clicked.
        for _ in range(4):
            prev = np.asarray(tr["tip_xy"], float)
            _set_threshold(side, prev, tr["u_end"])
            tr2 = trace_crack(side["base"], side, (0.0, 0.0), limit_mm=lim)
            if not tr2["found"]:
                break
            tr = tr2
            if np.linalg.norm(np.asarray(tr["tip_xy"], float) - prev) < 1.0:
                break
    if "thr_eff" in side and thr_scale(side["P"]) != 1.0:
        # user sensitivity: scale the margin above the paint-texture level (after the automatic
        # estimate has settled, so the estimate itself does not drift with the sensitivity)
        noise = side.get("noise_level", 0.0)
        side["thr_eff"] = noise + (side["thr_eff"] - noise) * thr_scale(side["P"])
        tr2 = trace_crack(side["base"], side, (0.0, 0.0), limit_mm=lim)
        if not tr2["found"]:
            raise ValueError("ברגישות הזו לא נמצא קו סדק ליד הקצה שסומן – הגדל את הרגישות.")
        tr = tr2
    side["base_len"] = tr["length"]
    side["base_tip_xy"] = tr["tip_xy"]
    side["click_note"] = ""
    if side["P"].get("base_tip_mode") == "click":
        # The click resolves a SMALL ambiguity of a faint baseline tip (e.g. two sides set up by
        # hand): if the clicked tip lies within click_tol_mm of the automatic tip (measured along
        # the crack), the click is the reference. If the automatic tip is much further away, the
        # same criterion is kept for every image (otherwise all later images would be offset).
        pts = np.asarray(tr["path_xy"], float); L = tr["cumlen"]
        cp = np.asarray(side["seed"], float)
        i = int(np.argmin(np.linalg.norm(pts - cp, axis=1)))
        extra = float(np.dot(cp - pts[-1], tr["u_end"])) if i == len(pts) - 1 else 0.0
        L_click = float(L[i] + max(0.0, extra))
        diff_mm = (tr["length"] - L_click) * side["mm_per_px"]
        if abs(diff_mm) <= side["P"].get("click_tol_mm", 0.5):
            side["base_len"] = L_click
            side["base_tip_xy"] = (float(cp[0]), float(cp[1]))
        else:
            side["click_note"] = (f"the crack is visible {diff_mm:+.2f} mm from the clicked tip; "
                                  f"the automatic tip is used as reference")
    side["base_path"] = tr["path_xy"]
    side["u_tip"] = tr["u_end"]
    side["a0_px"] = _a_px(side, tr, (0.0, 0.0))        # crack length from the start point, baseline
    from scipy.spatial import cKDTree
    side["_tree"] = cKDTree(np.asarray(tr["path_xy"], float))
    return side


def _ref_index(side, tr, d):
    """Index of the traced-crack point nearest to the user's start point."""
    ref = np.asarray(side["ref"], float) + np.asarray(d, float)
    pts = np.asarray(tr["path_xy"], float)
    return int(np.argmin(np.linalg.norm(pts - ref, axis=1))), ref, pts


def _a_px(side, tr, d):
    """Crack length in px from the user's start point: to the nearest point of the traced
    crack, then along the crack to the tip. With a_mode 'proj' (C(T), ASTM E647): the distance
    from the start point (on the load line) to the tip, projected on the crack axis."""
    if side.get("a_mode") == "proj":
        ref = np.asarray(side["ref"], float) + np.asarray(d, float)
        return float(np.dot(np.asarray(tr["tip_xy"], float) - ref, side["u"]))
    i, ref, pts = _ref_index(side, tr, d)
    return float(np.linalg.norm(pts[i] - ref) + tr["cumlen"][-1] - tr["cumlen"][i])


# ----------------------------------------------------------------------------- per frame
@dataclass
class FrameResult:
    file: str
    ok: bool
    status: str
    da: float
    corr: float
    d: tuple = (np.nan, np.nan)
    tip_xy: tuple = (np.nan, np.nan)
    img: np.ndarray | None = field(default=None, repr=False)
    path_xy: np.ndarray | None = field(default=None, repr=False)     # root .. tip
    meas_xy: np.ndarray | None = field(default=None, repr=False)     # start point .. tip
    a: float = np.nan                                                # crack length from start point [mm]
    ref_xy: tuple = (np.nan, np.nan)                                 # start point in this image
    reach: float = np.nan                                            # tip position (px) - prior for the next image
    unc: float = np.nan                                              # tip uncertainty [mm] (threshold check)


def measure_frame(path: str, side: dict, keep_image: bool = False, prev_reach=None) -> FrameResult:
    img = read_gray(path)
    resized = img.shape != side["base"].shape
    img = conform(img, side["base"].shape)
    P = side["P"]
    ms = int(P["max_shift_px"]) or auto_max_shift(img.shape)
    d, corr = register(side["base"], img, side["roi"], ms)
    bf0 = side.get("base_file")
    is_base = bool(bf0) and os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(bf0))
    if side.get("notch") and is_base:      # machined notch: the baseline image has no crack by definition
        tr = _notch_only(side, np.asarray(side["root"], float) + d, np.asarray(side["seed"], float) + d,
                         side["u"], side.get("thr_eff", P["contrast_thr"]), side["mm_per_px"])
    else:
        tr = trace_crack(img, side, d, min_reach_px=prev_reach)
    keep = img if keep_image else None
    if corr < P["min_reg_corr"]:
        return FrameResult(path, False, "low registration", np.nan, corr, d, tr["tip_xy"], keep,
                           tr["path_xy"])
    if not tr["found"]:
        return FrameResult(path, False, "crack not found", np.nan, corr, d, (np.nan, np.nan), keep)
    a_mm = _a_px(side, tr, d) * side["mm_per_px"]
    da = a_mm - side["a0_px"] * side["mm_per_px"]
    bf = side.get("base_file")
    if bf and os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(bf)):
        da = 0.0                   # the baseline image itself
    i_r, ref, pts = _ref_index(side, tr, d)
    meas = np.vstack([ref[None, :], pts[i_r:]])                        # start point -> tip
    # extra check: how much does the tip move if the threshold is a bit lower / higher?
    unc = np.nan
    if P.get("check_tip", True):
        thr0 = tr["thr"]; noise = side.get("noise_level", 0.0)
        alts = []
        for f in (0.75, 1.3):
            t2 = noise + f * (thr0 - noise)
            tr2 = trace_crack(img, side, d, thr=t2, min_reach_px=prev_reach)
            if tr2["found"]:
                alts.append(_a_px(side, tr2, d) * side["mm_per_px"])
        if alts:
            unc = float(max(abs(x - a_mm) for x in alts))
    status = "ok"
    if np.isfinite(unc) and unc > P.get("uncertain_mm", 0.3):
        status = f"uncertain (±{unc:.2f})"
    if tr.get("notch_only") and status == "ok":
        status = "ok (notch tip)"
    if tr["beyond"]:
        status = "tip beyond search range"
    elif tr["at_edge"]:
        status = "tip at image edge"
    elif resized and status == "ok":
        status = "ok (size differs)"
    return FrameResult(path, True, status, float(da), corr, d, tr["tip_xy"], keep, _thin(tr["path_xy"]),
                       _thin(meas), float(a_mm), (float(ref[0]), float(ref[1])), float(tr["reach"]), unc)


def _thin(pts, n=300):
    """Keep at most n points (endpoints kept), float32 - results of thousands of images stay small."""
    if pts is None:
        return None
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) <= n:
        return pts
    idx = np.unique(np.concatenate([np.linspace(0, len(pts) - 1, n).round().astype(int), [len(pts) - 1]]))
    return pts[idx]


def filter_series(raw, monotonic: bool = True, median3: bool = False) -> np.ndarray:
    """Running maximum (a crack cannot get shorter); optional causal median of 3 first."""
    raw = np.asarray(raw, float)
    f = raw.copy()
    good = np.isfinite(raw)
    if median3 and good.sum() >= 3:
        g = raw[good]
        f[good] = [np.median(g[max(0, i - 2):i + 1]) for i in range(len(g))]
    if monotonic:
        run = -np.inf
        for k in range(len(f)):
            if np.isfinite(f[k]):
                run = max(run, f[k])
            if np.isfinite(run):
                f[k] = run
    return f


# ----------------------------------------------------------------------------- drawing
def nice_step(px_per_mm: float, min_px: float) -> float:
    """Smallest 'nice' mm step (1-2-5 series) whose pixel length is >= min_px."""
    for e in range(-2, 4):
        for m in (1, 2, 5):
            st = m * 10.0 ** e
            if st * px_per_mm >= min_px:
                return st
    return 1000.0


def _lbl(a):
    a = round(float(a), 6)
    return f"{0.0 if a == 0 else a:g}"


def _draw_ruler(dr, size, ox, oy, sc, mmpp, font):
    """Subtle mm rulers along the top and left edges (0 = image left / top edge)."""
    W, H = size
    ppm = sc / mmpp                      # display px per mm
    minor = nice_step(ppm, 7); major = nice_step(ppm, 60)
    col = (255, 255, 0, 170)
    dr.rectangle([0, 0, W, 14], fill=(0, 0, 0, 70)); dr.rectangle([0, 0, 14, H], fill=(0, 0, 0, 70))
    for horizontal in (True, False):
        o_mm = (ox if horizontal else oy) * mmpp
        length = W if horizontal else H
        a = np.ceil(o_mm / minor) * minor
        while (a - o_mm) * ppm < length:
            p = (a - o_mm) * ppm
            isM = abs(a / major - round(a / major)) < 1e-6
            if horizontal:
                dr.line([(p, 0), (p, 12 if isM else 6)], fill=col, width=1)
                if isM and p > 18:
                    dr.text((p + 2, 1), _lbl(a), fill=col, font=font)
            else:
                dr.line([(0, p), (12 if isM else 6, p)], fill=col, width=1)
                if isM and p > 18:
                    dr.text((2, p + 1), _lbl(a), fill=col, font=font)
            a += minor


def render_annotated(img: np.ndarray, side: dict, res: FrameResult, label: str, crop=None,
                     ruler: bool = True, zoom: bool = True) -> Image.Image:
    """RGB image with the measurement drawn on it (for saving / reports).
    crop = (x0, y0, x1, y1) in image pixels, or None for the whole image."""
    from PIL import ImageDraw, ImageFont
    g = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(g).convert("RGB")
    ox = oy = 0
    if crop:
        x0, y0, x1, y1 = [int(round(v)) for v in crop]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(im.width, x1), min(im.height, y1)
        im = im.crop((x0, y0, x1, y1)); ox, oy = x0, y0
    sc = max(1.0, 1200.0 / im.width)
    if sc > 1:
        im = im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS)
    Pt = lambda x, y: ((x - ox + 0.5) * sc, (y - oy + 0.5) * sc)
    dr = ImageDraw.Draw(im, "RGBA")
    lw = max(2, im.width // 500)
    try:
        font = ImageFont.truetype("arial.ttf", max(13, im.width // 60))
        small = ImageFont.truetype("arial.ttf", max(10, im.width // 110))
    except OSError:
        font = small = ImageFont.load_default()
    if res.meas_xy is not None and len(res.meas_xy) > 1:
        dr.line([Pt(*p) for p in res.meas_xy], fill=(255, 150, 0, 170), width=max(2, lw))
    if res.ref_xy is not None and np.all(np.isfinite(res.ref_xy)):
        q = Pt(*res.ref_xy); r1 = max(6, im.width // 160)
        dr.line([(q[0] - r1, q[1]), (q[0] + r1, q[1])], fill=(255, 235, 0, 255), width=lw)
        dr.line([(q[0], q[1] - r1), (q[0], q[1] + r1)], fill=(255, 235, 0, 255), width=lw)
    if np.all(np.isfinite(res.tip_xy)):
        tp = Pt(*res.tip_xy)
        rr = max(10, int(0.35 / side["mm_per_px"] * sc))
        dr.ellipse([tp[0] - rr, tp[1] - rr, tp[0] + rr, tp[1] + rr], outline=(255, 30, 30, 255), width=lw)
    if ruler:
        _draw_ruler(dr, im.size, ox, oy, sc, side["mm_per_px"], small)
    tbx = dr.textbbox((0, 0), label, font=font)
    dr.rectangle([0, 0, tbx[2] + 16, tbx[3] + 12], fill=(0, 0, 0, 180))
    dr.text((8, 6), label, fill=(255, 255, 255, 255), font=font)
    if zoom and np.all(np.isfinite(res.tip_xy)):
        z = tip_zoom(img, res.tip_xy, side["mm_per_px"], int(np.clip(im.height, 360, 720)))
        out = Image.new("RGB", (im.width + 10 + z.width, max(im.height, z.height)), (40, 40, 40))
        out.paste(im, (0, 0)); out.paste(z, (im.width + 10, 0))
        return out
    return im


def tip_zoom(img: np.ndarray, tip_xy, mmpp: float, size: int = 360, half_mm: float = 0.75,
             max_mag: float = 3.0) -> Image.Image:
    """Magnified view of the raw image around the tip with nothing drawn over the crack itself:
    four short red ticks point at the tip from outside, plus a small scale bar."""
    from PIL import ImageDraw, ImageFont
    # never magnify more than max_mag x (beyond that you only see bigger pixels, not detail)
    h = max(int(round(half_mm / mmpp)), int(np.ceil(size / (2.0 * max_mag))), 8)
    x, y = tip_xy
    xs = np.arange(-h, h + 1) + x; ys = np.arange(-h, h + 1) + y
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    patch = map_coordinates(img, [Y.ravel(), X.ravel()], order=1, mode="nearest").reshape(Y.shape)
    # gentle contrast stretch (keeps the look of the original, no exaggerated texture)
    lo, hi = np.percentile(patch, 0.2), np.percentile(patch, 99.8)
    g = (np.clip((patch - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(g).resize((size, size), Image.BICUBIC).convert("RGB")
    dr = ImageDraw.Draw(im, "RGBA")
    c = size / 2.0; gap = size * 0.07; ln = size * 0.10; w = max(2, size // 180)
    red = (255, 40, 40, 255)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        dr.line([(c + dx * gap, c + dy * gap), (c + dx * (gap + ln), c + dy * (gap + ln))], fill=red, width=w)
    try:
        font = ImageFont.truetype("arial.ttf", max(12, size // 26))
    except OSError:
        font = ImageFont.load_default()
    ppm = size / (2 * h + 1) / mmpp                    # display px per mm
    bar = nice_step(ppm, size * 0.18)
    bx0 = size * 0.06; by0 = size - size * 0.07
    dr.rectangle([bx0 - 6, by0 - 26, bx0 + bar * ppm + 6, by0 + 8], fill=(0, 0, 0, 150))
    dr.line([(bx0, by0), (bx0 + bar * ppm, by0)], fill=(255, 235, 120, 255), width=3)
    dr.text((bx0, by0 - 24), f"{_lbl(bar)} mm", fill=(255, 235, 120, 255), font=font)
    dr.rectangle([0, 0, size, 26], fill=(0, 0, 0, 150))
    dr.text((6, 4), f"tip zoom  x{size / (2 * h + 1):.1f}", fill=(255, 255, 255, 255), font=font)
    return im


def correct_tip(res: FrameResult, side: dict, click_xy) -> FrameResult:
    """Manual correction: the user clicked the true tip in this image. The crack length is
    re-measured from the start point along the traced crack to the point nearest the click,
    then straight to the click (so a click a little beyond the traced line also works)."""
    from dataclasses import replace
    mmpp = side["mm_per_px"]
    cp = np.asarray(click_xy, float)
    d = np.asarray(res.d if res.d is not None and np.all(np.isfinite(res.d)) else (0.0, 0.0), float)
    ref = np.asarray(side["ref"], float) + d
    if res.path_xy is not None and len(res.path_xy) >= 2:
        pts = np.asarray(res.path_xy, float)
        L = _arclen(pts, mmpp)
        i_r = int(np.argmin(np.linalg.norm(pts - ref, axis=1)))
        j = int(np.argmin(np.linalg.norm(pts - cp, axis=1)))
        if j >= i_r:
            a_px = (np.linalg.norm(pts[i_r] - ref) + (L[j] - L[i_r]) + np.linalg.norm(cp - pts[j]))
            meas = np.vstack([ref[None, :], pts[i_r:j + 1], cp[None, :]])
        else:
            a_px = np.linalg.norm(cp - ref); meas = np.vstack([ref, cp])
    else:
        a_px = np.linalg.norm(cp - ref); meas = np.vstack([ref, cp])
    if side.get("a_mode") == "proj":
        a_px = float(np.dot(cp - ref, side["u"]))
    a_mm = float(a_px * mmpp)
    return replace(res, ok=True, status="manual", a=a_mm, da=a_mm - side["a0_px"] * mmpp,
                   tip_xy=(float(cp[0]), float(cp[1])), meas_xy=meas, ref_xy=(float(ref[0]), float(ref[1])))


def threshold_for_tip(path: str, side: dict, d, target_a_mm: float, tol_mm: float = 0.05):
    """Find the detection threshold at which the automatic tip of this image lands at the
    manually corrected crack length. Returns (thr, a_reached_mm) or (None, None)."""
    img = conform(read_gray(path), side["base"].shape)
    noise = side.get("noise_level", 0.0)
    mmpp = side["mm_per_px"]

    def a_at(t):
        tr = trace_crack(img, side, d, thr=t)
        return _a_px(side, tr, d) * mmpp if tr["found"] else np.nan

    lo, hi = noise + 0.005, 0.95                       # low thr -> longer crack, high -> shorter
    a_lo, a_hi = a_at(lo), a_at(hi)
    if not np.isfinite(a_lo) or a_lo < target_a_mm - 0.3:
        return None, None                              # the click lies beyond any crack evidence
    if np.isfinite(a_hi) and a_hi > target_a_mm + 0.3:
        return None, None
    best = (abs(a_lo - target_a_mm), lo, a_lo)
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        a = a_at(mid)
        if not np.isfinite(a):
            hi = mid; continue
        if abs(a - target_a_mm) < best[0]:
            best = (abs(a - target_a_mm), mid, a)
        if abs(a - target_a_mm) <= tol_mm:
            break
        if a > target_a_mm:
            lo = mid
        else:
            hi = mid
    return (best[1], best[2]) if best[0] <= 0.3 else (None, None)


# ----------------------------------------------------------------------------- crack direction
def _signed_angle_deg(vec, u):
    """Angle [deg] of vec relative to the crack axis u; + = counter-clockwise as seen on screen."""
    u = np.asarray(u, float); u = u / max(1e-12, np.linalg.norm(u))
    n = np.array([u[1], -u[0]])                   # u rotated 90 deg counter-clockwise on screen (y down)
    return float(np.degrees(np.arctan2(np.dot(vec, n), np.dot(vec, u))))


def crack_angles(res, side, last_mm=0.0, min_mm=0.5):
    """Direction of the crack relative to the initial crack axis (start click -> tip click).
    Returns (angle_new, angle_total) in degrees, NaN when not defined:
      angle_new   - direction of the crack part that grew beyond the baseline tip (a line fitted
                    through it); with last_mm > 0 only the last last_mm of it, next to the tip
      angle_total - direction from the start point straight to the tip."""
    if res is None or res.meas_xy is None or len(res.meas_xy) < 2 or res.tip_xy is None \
            or not np.all(np.isfinite(res.tip_xy)):
        return np.nan, np.nan
    mmpp = side["mm_per_px"]
    u = np.asarray(side["u"], float)
    pts = np.asarray(res.meas_xy, float)
    total = _signed_angle_deg(pts[-1] - pts[0], u)
    d = np.asarray(res.d if res.d is not None and np.all(np.isfinite(res.d)) else (0.0, 0.0), float)
    base_tip = np.asarray(side["base_tip_xy"], float) + d
    L = _arclen(pts, mmpp) * mmpp                       # mm along the measured line
    i0 = int(np.argmin(np.linalg.norm(pts - base_tip, axis=1)))
    if last_mm and last_mm > 0:
        i0 = max(i0, int(np.searchsorted(L, L[-1] - last_mm)))
    seg = pts[i0:]
    if len(seg) < 3 or L[-1] - L[i0] < min_mm:
        return np.nan, total
    c = seg - seg.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    dvec = vt[0]
    if np.dot(dvec, seg[-1] - seg[0]) < 0:
        dvec = -dvec
    return _signed_angle_deg(dvec, u), total

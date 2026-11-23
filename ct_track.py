"""Growth guard for C(T) runs (CT Tracker 2.0).

The tracer looks for the crack 'at least as long as in the previous image'. When the crack does not
grow for many images, a little noise at the tip used to be accepted as growth, the next image started
from there, and the crack crept longer and longer until the measurements failed.

Here growth is accepted only when it is (1) NEW and (2) PERSISTENT:
1. along the new part of the crack the surface got darker since the ANCHOR image (the last image with
   confirmed growth), exposure-normalised, continuously from the anchor tip;
2. the same new part is still darker than the anchor in the NEXT image. A crack does not heal; dust,
   reflections and noise change from image to image. Until the next image arrives the growth is
   provisional; if it is not confirmed the earlier row is corrected (revisions).
Otherwise the tip stays where it was ('ok (no growth)'). The anchor only moves when growth is
confirmed, so slow growth still adds up until it is visible. A crack can not get shorter: a shorter
result re-checks the confirmed crack in that image (still darker than the crack-free baseline up to the
old tip = the earlier measurements were right) and keeps the anchor tip - only when the crack is not
there in several images in a row is the anchor itself wrong and replaced."""
from __future__ import annotations

import copy
import os

import numpy as np
from scipy.ndimage import map_coordinates

import crack_core as cc

NO_GROWTH = "ok (no growth)"


def _densify(pts, step=1.0):
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return pts
    seg = np.hypot(*np.diff(pts, axis=0).T)
    L = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.arange(0.0, L[-1] + 1e-9, step)
    return np.column_stack([np.interp(t, L, pts[:, 0]), np.interp(t, L, pts[:, 1])])


def _sample(img, pts):
    return map_coordinates(img, [pts[:, 1], pts[:, 0]], order=1, mode="nearest")


def _bare(res):
    """Copy of a result without the image it may still carry (a kept copy must not hold 48 MB)."""
    r = copy.copy(res)
    if hasattr(r, "img"):
        r.img = None
    return r


def _d(res):
    return np.asarray(res.d if res.d is not None and np.all(np.isfinite(res.d)) else (0.0, 0.0), float)


class GrowthGuard:
    def __init__(self, prep, tol_mm=0.03, reanchor_n=5, reanchor_mm=0.2):
        self.p = prep
        self.anchor = None
        self.pending = None                      # provisional growth waiting for the next image
        self.tol = max(tol_mm, 1.5 * prep["mm_per_px"])
        self.reanchor_n, self.reanchor_mm = reanchor_n, reanchor_mm
        self.shorter = []                        # consecutive results clearly shorter than the anchor
        self.prev = None                         # (image, shift) of the previous image: a second reference
        self.held = []                           # (row, own result): rows shown below their own measurement

    # ------------------------------------------------------------------ anchor
    def _set_anchor(self, res, img, ref2=None):
        """ref2: a second reference without crack beyond this tip, with its own noise - the image before
        the anchor image (for the baseline: the first image after it, set in apply)."""
        d = _d(res)
        path = res.path_xy if res.path_xy is not None else res.meas_xy
        self.anchor = dict(img=img, d=d, a=float(res.a), da=float(res.da),
                           crack=float(res.da) > self.tol,          # a confirmed crack beyond the notch
                           tip_b=np.asarray(res.tip_xy, float) - d,
                           reach=float(res.reach) if np.isfinite(res.reach) else 0.0, ref2=ref2,
                           path_b=None if path is None or len(path) < 2 else np.asarray(path, float) - d)
        self.shorter = []

    def set_manual(self, res):
        """A manual tip correction of the newest image becomes the anchor."""
        try:
            img = cc.load_frame(res.file, self.p)
        except Exception:
            return
        seed = np.asarray(self.p["seed"], float)
        res.reach = float(np.dot(np.asarray(res.tip_xy, float) - (seed + _d(res)), self.p["u"]))
        self._set_anchor(res, img)
        self.pending = None
        self.held = []

    # ------------------------------------------------------------------ evidence
    def _new_points(self, path, d):
        """Points (1 px apart) of the crack part beyond the anchor tip, in the coordinates of an image
        whose shift is d, as offsets from the anchor tip; and their position along u. Or None."""
        u = np.asarray(self.p["u"], float)
        tA = self.anchor["tip_b"] + d
        path = np.asarray(path, float)
        s_all = (path - tA) @ u
        # the path is load line -> notch tip -> crack; keep only its final run beyond the anchor tip
        behind = np.nonzero(s_all <= 0)[0]
        i0 = int(behind[-1]) + 1 if len(behind) else 0
        if i0 >= len(path):
            return None
        q = _densify(np.vstack([tA[None, :], path[i0:]]), 1.0)
        if len(q) < 3:
            return None
        return q - tA, (q - tA) @ u

    def _walk(self, s, ev, gap_mm):
        k = 1.0 / self.p["mm_per_px"]
        max_gap = max(2, int(round(gap_mm * k)))
        start_gap = max(2, int(round(0.15 * k)))
        last, gap = -1, 0
        for j in range(len(ev)):
            if ev[j]:
                last, gap = j, 0
            else:
                gap += 1
                if gap > (start_gap if last < 0 else max_gap):
                    break
        return float(s[last]) if last >= 0 else 0.0

    def _new_length(self, path, d, img):
        """Length [px, along u] of the crack part beyond the anchor tip (path in the coordinates of img,
        whose shift is d) that is continuously darker in img than in the anchor image."""
        pts = self._new_points(path, d)
        if pts is None:
            return 0.0
        rel, s = pts
        ev = self._evidence(rel, d, img)
        return self._walk(s, ev, self.p["P"]["max_gap_mm"])

    def _tight_ev(self, rel, d, img, ref, ref_d, k, u, v, fac=1.0):
        """Per point: new darkening (or brightening) of img vs. the reference image ref (shift ref_d);
        fac scales the threshold."""
        P = self.p["P"]
        q = self.anchor["tip_b"] + d + rel
        shift = np.asarray(ref_d, float) - d          # coords of img -> coords of the reference
        ring = np.concatenate([q + o * k * v for o in (-0.5, -0.35, 0.35, 0.5)])
        ratio = _sample(img, ring) / np.maximum(_sample(ref, ring + shift), 1e-4)
        rbg = float(np.clip(np.median(ratio), 0.5, 2.0)) if ratio.size else 1.0
        thr = self.p.get("thr_eff", P["contrast_thr"]) * P.get("guard_thr_frac", 1.0) * fac

        def sm(im, pts):
            return np.mean([_sample(im, pts + a * u + b * v) for a in (-0.5, 0.0, 0.5) for b in (-0.5, 0.0, 0.5)],
                           axis=0)
        cur0 = np.array([sm(img, q + o * v) for o in (-1.0, 0.0, 1.0)])
        prv0 = np.array([sm(ref, q + shift + o * v) for o in (-1.0, 0.0, 1.0)])
        pol = P.get("crack_polarity", "auto")
        ev = np.zeros(cur0.shape[1], bool)
        for kind in (("dark", "bright") if pol == "auto" else (pol,)):
            # dark: 1 - this/ref; bright (light through an open crack): 1 - ref/this - bounded, a
            # saturated face can not get brighter (no evidence instead of noise)
            cur, prv, g = (cur0, prv0, rbg) if kind == "dark" else (prv0 * rbg, cur0, 1.0)
            r = cur / np.maximum(prv, 1e-4)
            j = np.argmin(r, axis=0); cols = np.arange(r.shape[1])
            new = 1.0 - r[j, cols] / g
            drop = prv[j, cols] * new
            ev |= (new >= max(thr, 0.03)) & (drop >= 0.75 * P.get("tight_min_drop", 0.008))
        return ev

    def _base_ev(self, rel, d, img, fac=1.0):
        """Per point: darker (or brighter) than the crack-free start of the test - against the baseline
        and, when there is one, the second crack-free image (a noise speck frozen into one of them is not
        in the other)."""
        k = 1.0 / self.p["mm_per_px"]
        u = np.asarray(self.p["u"], float); v = np.array([-u[1], u[0]])
        ev = self._tight_ev(rel, d, img, self.p["base"], (0.0, 0.0), k, u, v, fac)
        b2 = self.p.get("base2")
        if b2 is not None:
            ev &= self._tight_ev(rel, d, img, b2[0], b2[1], k, u, v, fac)
        return ev

    def _evidence(self, rel, d, img):
        """New darkening (vs. the anchor image) at the points anchor tip + rel of an image with shift d."""
        P = self.p["P"]; mmpp = self.p["mm_per_px"]; k = 1.0 / mmpp
        u = np.asarray(self.p["u"], float); v = np.array([-u[1], u[0]])
        A = self.anchor
        q = A["tip_b"] + d + rel
        shift = A["d"] - d                           # coords of img -> coords of the anchor image
        ring = np.concatenate([q + o * k * v for o in (-0.5, -0.35, 0.35, 0.5)])
        ratio = _sample(img, ring) / np.maximum(_sample(A["img"], ring + shift), 1e-4)
        rbg = float(np.clip(np.median(ratio), 0.5, 2.0)) if ratio.size else 1.0
        thr = 0.6 * self.p.get("thr_eff", P["contrast_thr"])
        w = max(1, int(round(P["fine_mm"] * k)))
        if P.get("notch_diff", "tight") == "tight":
            # the same point in both images (registered sub-pixel), lightly smoothed: on scratched metal
            # a 'darkest pixel nearby' comparison would hide the crack behind the old scratches.
            # Against TWO crack-free references with independent noise (the anchor image and the image
            # before it): every image is compared with the same reference, so a noise speck frozen into
            # that one reference looks like 'new darkening' in every later image - also in the image that
            # is supposed to confirm the growth. A real crack is new against both references.
            ev = self._tight_ev(rel, d, img, A["img"], A["d"], k, u, v)
            r2 = A.get("ref2")
            if r2 is not None:
                ev &= self._tight_ev(rel, d, img, r2[0], r2[1], k, u, v)
            # crack that was already there in the anchor image beyond its tip (the anchor was measured or
            # confirmed shorter than the crack - typical for a thin crack) is never 'new' against the
            # anchor and would stay stuck there for good: it also counts when it is darker than the
            # crack-free start of the test (two references, threshold x guard_base_frac) in this image
            # AND already in the anchor image. Never at the bare notch (anchor = the baseline): there
            # noise and changing light would pass the lower threshold.
            if A.get("crack"):
                fac = P.get("guard_base_frac", 0.7)
                ev |= self._base_ev(rel, d, img, fac) & self._base_ev(rel, A["d"], A["img"], fac)
            return cc._movmean(ev.astype(float), 2 * w + 1) >= 0.5
        cur = np.min([_sample(img, q + o * v) for o in (-1.0, 0.0, 1.0)], axis=0)
        prv = np.min([_sample(A["img"], q + shift + o * v + w * u)
                      for o in (-2.0, -1.0, 0.0, 1.0, 2.0) for w in (-2.0, 0.0, 2.0)], axis=0)
        ring = np.concatenate([q + o * k * v for o in (-0.5, -0.35, 0.35, 0.5)])
        ratio = _sample(img, ring) / np.maximum(_sample(A["img"], ring + shift), 1e-4)
        rbg = float(np.clip(np.median(ratio), 0.5, 2.0)) if ratio.size else 1.0
        new = 1.0 - (cur / np.maximum(prv, 1e-4)) / rbg
        drop = prv - cur / rbg
        thr = 0.6 * self.p.get("thr_eff", P["contrast_thr"])
        ev = (new >= max(thr, 0.03)) & (drop >= 0.75 * P.get("notch_min_drop", 0.04))
        w = max(1, int(round(P["fine_mm"] * k)))
        return cc._movmean(ev.astype(float), 2 * w + 1) >= 0.5

    def _still_there(self, res, img):
        """An image measures the crack clearly shorter than the confirmed tip: re-check the earlier
        measurements in THIS image. Fraction [0..1] of the confirmed crack between this tip and the anchor
        tip that is still darker (or brighter) than the crack-free baseline here, or None (can not check).
        A crack does not heal: if it is still there, the earlier measurements were right."""
        A = self.anchor
        if A.get("path_b") is None or self.p.get("base") is None or res.tip_xy is None \
                or not np.all(np.isfinite(res.tip_xy)):
            return None
        k = 1.0 / self.p["mm_per_px"]
        u = np.asarray(self.p["u"], float)
        d = _d(res)
        s_tip = float((np.asarray(res.tip_xy, float) - d - A["tip_b"]) @ u)      # < 0: behind the anchor tip
        q = _densify(A["path_b"], 1.0)
        s = (q - A["tip_b"]) @ u
        m = (s > s_tip) & (s <= 0)
        if m.sum() < 3:
            return None
        ev = self._base_ev(q[m] - A["tip_b"], d, img)
        w = max(1, int(round(self.p["P"]["fine_mm"] * k)))
        return float(np.mean(cc._movmean(ev.astype(float), 2 * w + 1) >= 0.5))

    # ------------------------------------------------------------------ results
    def _cut(self, res, tip, a, da, reach, status):
        u = np.asarray(self.p["u"], float)
        tip = np.asarray(tip, float)
        st = float(tip @ u)
        for key in ("path_xy", "meas_xy"):
            pts = getattr(res, key)
            if pts is None or len(pts) < 2:
                continue
            pts = np.asarray(pts, float)
            keep = pts[(pts @ u) < st]
            setattr(res, key, np.vstack([keep, tip[None, :]]).astype(np.float32) if len(keep) else
                    np.vstack([pts[:1], tip[None, :]]).astype(np.float32))
        if not hasattr(res, "raw_a"):
            res.raw_a = res.a
        res.tip_xy = (float(tip[0]), float(tip[1]))
        res.a, res.da, res.reach = float(a), float(da), float(reach)
        if status:
            res.status = status
        return res

    def _hold(self, res, why):
        A = self.anchor
        res = self._cut(res, A["tip_b"] + _d(res), A["a"], A["da"], A["reach"], NO_GROWTH)
        res.guard = why
        return res

    def _trim(self, res, acc_px):
        """Result with the tip moved back to acc_px (along u) beyond the anchor tip, on the traced line."""
        mmpp = self.p["mm_per_px"]
        A = self.anchor
        u = np.asarray(self.p["u"], float)
        tip = A["tip_b"] + _d(res) + u * acc_px
        if res.path_xy is not None and len(res.path_xy):
            q = _densify(res.path_xy, 1.0)
            j = int(np.argmin(np.abs((q - tip) @ u) + 0.01 * np.linalg.norm(q - tip, axis=1)))
            tip = q[j]
        a = float(np.dot(tip - np.asarray(res.ref_xy, float), u)) * mmpp
        return self._cut(res, tip, a, a - self.p["a0_px"] * mmpp, A["reach"] + acc_px, None)

    def _keep(self, row, raw):
        """A row shown below its own measurement (growth not confirmed yet): settled by a later confirmation."""
        if row is not None and raw is not None and np.isfinite(raw.a) and raw.a > self.anchor["a"] + self.tol:
            self.held.append((row, raw))
            lo = row - int(self.p["P"].get("guard_settle_n", 6)) - 4
            self.held = [h for h in self.held if h[0] >= lo]

    def _settle(self, upto):
        """Growth confirmed in row `upto`: the rows shortly before it that were shown below their own
        measurement get it back. A crack does not shrink and was at most this long then, so the row's own
        measurement - capped at the newly confirmed tip - is the best estimate (the previous value lagged
        behind fast growth). Only the last `guard_settle_n` images: older raw values are not trusted."""
        if upto is None:
            return []
        lo = upto - int(self.p["P"].get("guard_settle_n", 6))
        aC = self.anchor["a"]
        out, keep = [], []
        for row, raw in self.held:
            if row >= upto:
                keep.append((row, raw))
            elif row >= lo:
                if raw.a <= aC + self.tol:
                    r1 = copy.copy(raw)
                    r1.guard = f"עודכן בדיעבד: הצמיחה אושרה בתמונה מאוחרת יותר (a = {aC:.3f})"
                else:
                    r1 = self._hold(copy.copy(raw), f"עודכן בדיעבד לקצה שאושר אחריה (a = {aC:.3f})")
                out.append((row, r1))
        self.held = keep
        return out

    def _confirm(self, res, img):
        """The pending growth, checked on an image that came after it. A crack stays where it is, dust and
        reflections move: the new part must be darker than the references at the SAME places in the
        pending image and in the next `guard_confirm_n` images (with 3+ images, each point may be missed
        in one of them), continuously from the anchor tip. Returns (state, [(row, revised res), ...]); state "waiting" = not decided yet."""
        pd = self.pending
        mmpp = self.p["mm_per_px"]
        need = int(self.p["P"].get("guard_confirm_n", 2))
        r0 = pd["res"]
        pts = self._new_points(r0.path_xy, _d(r0))
        conf = 0.0
        seen = pd.get("seen", 1) + 1                     # images looked at: the pending one + the following
        pd["seen"] = seen
        total = need + 1
        # beyond a confirmed crack, of 3+ images one may miss a point (a crack near the threshold; moving
        # dust is in one image only). At the bare notch - the crack start, which raises the alert - every
        # image must show it: there changing light can darken the same spot in two of three images.
        req = total - 1 if total >= 3 and self.anchor.get("crack") else total
        if pts is not None:
            rel, s = pts
            cnt = pd.get("cnt")
            if cnt is None:
                cnt = self._evidence(rel, _d(r0), pd["img"]).astype(int)
            cnt = cnt + self._evidence(rel, _d(res), img)
            pd["cnt"] = cnt
            ev = cnt >= max(1, req - (total - seen))      # still reachable with the images to come
            conf = min(pd["acc"], self._walk(s, ev, 0.15))
        if conf * mmpp < self.tol:
            self.pending = None
            why = (f"צמיחה לא אושרה: בתמונה שאחריה אין התכהות במקום (נמדד {(r0.a - self.anchor['a']):+.3f} מ״מ)")
            revs = [(pd["row"], self._hold(copy.copy(r0), why))]
            revs += [(row, self._hold(copy.copy(rw), why)) for row, rw in pd.get("waits", [])]
            self._keep(pd["row"], pd.get("raw"))
            for row, rw in pd.get("waits", []):
                self._keep(row, rw)
            return "done", revs
        pd["acc"] = conf
        pd["n"] = seen - 1
        if pd["n"] < need:
            return "waiting", []
        self.pending = None
        revs = []
        if conf * mmpp < (r0.a - self.anchor["a"]) - self.tol:
            out = self._trim(copy.copy(r0), conf)
            out.guard = f"הצמיחה קוצרה ל-{conf * mmpp:.3f} מ״מ: רק החלק שנשאר כהה גם בתמונות שאחריה"
            self._set_anchor(out, pd["img"], pd.get("prev"))
            revs.append((pd["row"], out))
        else:
            self._set_anchor(r0, pd["img"], pd.get("prev"))
            if getattr(r0, "guard", "").startswith("ממתין"):
                r1 = copy.copy(r0); r1.guard = ""
                revs.append((pd["row"], r1))
        revs += self._settle(pd["row"])
        self._keep(pd["row"], pd.get("raw"))                # measured longer than confirmed: settled later
        for row, rw in pd.get("waits", []):               # images shown as 'waiting': the confirmed tip
            revs.append((row, self._hold(copy.copy(rw), "")))
            self._keep(row, rw)
        return "done", revs

    def finish(self):
        """No more images (end of the folder, run stopped): growth still waiting for confirmation is not
        confirmed - its rows go back to the confirmed tip. Returns [(row, revised res), ...]."""
        pd, self.pending = self.pending, None
        if pd is None:
            return []
        why = "צמיחה לא אושרה: אין עוד תמונות שיאשרו אותה (סוף הריצה)"
        revs = [(pd["row"], self._hold(copy.copy(pd["res"]), why))]
        revs += [(row, self._hold(copy.copy(rw), why)) for row, rw in pd.get("waits", [])]
        return [(row, r) for row, r in revs if row is not None]

    def apply(self, res, img, row=None):
        """Returns (result for this image, [(earlier row, revised result), ...])."""
        out = self._apply(res, img, row)
        if img is not None and res.d is not None and np.all(np.isfinite(res.d)):
            self.prev = (img, _d(res))
        return out

    def _ref2_from(self, img, res):
        """A held image (no growth beyond the anchor) is the second reference if there is none yet."""
        if self.anchor is not None and self.anchor.get("ref2") is None:
            self.anchor["ref2"] = (img, _d(res))

    def _apply(self, res, img, row=None):
        revisions = []
        if img is None or not res.ok or not np.isfinite(res.a) or res.d is None or not np.all(np.isfinite(res.d)):
            return res, revisions
        bf = self.p.get("base_file")
        if self.anchor is None or (bf and os.path.normcase(res.file) == os.path.normcase(bf)):
            self._set_anchor(res, img)
            self.pending = None
            self.held = []
            return res, revisions
        if self.pending is not None:
            raw = _bare(res)
            state, revs = self._confirm(res, img)
            revisions += revs
            if state == "waiting":
                # the pending growth is seen again here, one more image decides: this image shows the
                # pending tip for now and is revised with it
                pd = self.pending
                pd.setdefault("waits", []).append((row, raw))
                r0 = pd["res"]
                tip = np.asarray(r0.tip_xy, float) - _d(r0) + _d(res)
                out = self._cut(res, tip, r0.a, r0.da, r0.reach, None)
                out.guard = f"ממתין לאישור ({pd['n']}/{int(self.p['P'].get('guard_confirm_n', 2))})"
                return out, revisions
        A = self.anchor
        mmpp = self.p["mm_per_px"]
        ext = res.a - A["a"]
        if ext <= self.tol:
            # 'no crack beyond the notch' after a confirmed crack is a detection failure (a crack does not
            # heal), never a reason to move the anchor back
            notch_only = str(res.status).startswith("ok (notch")
            if ext < -self.reanchor_mm and not notch_only:
                frac = self._still_there(res, img)
                if frac is not None and frac >= 0.6:
                    # the confirmed crack is still visible up to the old tip: this image missed the tip
                    self.shorter = []
                    self._ref2_from(img, res)
                    return self._hold(res, f"נמדד קצר יותר (a = {res.a:.3f}), אבל הסדק עד הקצה הקודם עדיין "
                                           f"נראה בתמונה ({frac * 100:.0f}%) – המדידות הקודמות נכונות, "
                                           f"אין התקצרות"), revisions
                self.shorter.append(res.a)
                if len(self.shorter) >= self.reanchor_n and np.ptp(self.shorter[-self.reanchor_n:]) < 0.15:
                    raw, old = res.a, A["a"]
                    self._set_anchor(res, img, self.prev)
                    self.held = []
                    res.status = "uncertain (re-anchored)"
                    res.guard = (f"{self.reanchor_n} תמונות ברצף קצרות מהעוגן, והסדק בין {raw:.3f} ל-{old:.3f} "
                                 f"לא נראה בהן – המדידות הקודמות כנראה ארוכות מדי; העוגן עודכן ל-a = {raw:.3f}")
                    return res, revisions
            else:
                self.shorter = []
            self._ref2_from(img, res)
            return self._hold(res, "לא נמצאה צמיחה" if ext >= -self.tol else
                              f"קצר מהתמונה שבה הסדק אושר ({res.a:.3f} < {A['a']:.3f}) – סדק לא מתקצר"), revisions
        self.shorter = []
        acc = self._new_length(res.path_xy if res.path_xy is not None else res.meas_xy, _d(res), img)
        if acc * mmpp < self.tol:
            self._ref2_from(img, res)
            self._keep(row, _bare(res))
            return self._hold(res, f"הצמיחה ({ext:.3f} מ״מ) לא מאושרת: אין התכהות חדשה מעבר לקצה הקודם"), revisions
        raw_res = _bare(res)
        if acc * mmpp < ext - self.tol:
            raw = res.a
            res = self._trim(res, acc)
            res.guard = f"ממתין לאישור בתמונה הבאה · קוצר ל-{acc * mmpp:.3f} מ״מ (נמדד {raw - A['a']:.3f})"
        else:
            res.guard = "ממתין לאישור בתמונה הבאה"
        self.pending = dict(row=row, res=res, img=img, acc=acc, prev=self.prev, raw=raw_res)
        return res, revisions

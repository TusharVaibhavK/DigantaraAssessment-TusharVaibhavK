"""Step A3b: classify every detection as star (class 0) or streak (class 1).

Rules (thresholds in configs/params.yaml, reasoning in the write-up, Q2 c/d):
  streak          elong >= streak_elong, length >= streak_len_factor * L_star, even axial profile
  short/fat streak  elong >= short_elong, length >= short_len_factor * L_star, bright enough
                    for the shape to be trusted (peak SNR >= short_min_snr), even axial profile
  blend           elongated but with a dip along its axis -> split into stars (watershed)
  star            everything else that passed detection (>= 5 sigma, PSF-like, not a hot pixel)
L_star is the median weighted length of bright, round stars in the same frame.

Writes outputs/classified/<name>_instances.npz (uint32 instance map), <name>_objects.csv,
plus review sheets for the grey zone.
"""
import sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tqdm import tqdm

from detect import DET_DIR, crop_with_contour, review_sheet
from fits_io import OUT_DIR, list_frames, load_frame, short_name
from preprocess import PROC_DIR, load_params

import cv2

CLS_DIR = OUT_DIR / "classified"
STAR, STREAK = 0, 1


# ---------------------------------------------------------------- axial profile

def local_maps(raw, sl, mask_bg, psf_sigma):
    """Background-subtracted, PSF-smoothed crop and its noise, estimated from pixels outside objects."""
    crop = raw[sl].astype(np.float32)
    bg_pix = crop[mask_bg]
    bg = np.median(bg_pix) if bg_pix.size > 50 else np.median(crop)
    smooth = ndi.gaussian_filter(crop - bg, psf_sigma)
    s_bg = smooth[mask_bg] if bg_pix.size > 50 else smooth.ravel()
    noise = 1.4826 * np.median(np.abs(s_bg - np.median(s_bg))) + 1e-6
    return smooth, noise


def axial_profile(smooth, mask, trim):
    """Mean brightness along the object's major axis (PCA of its pixels), ends trimmed by `trim` px.
    Returns (profile, extent_px)."""
    ys, xs = np.nonzero(mask)
    w = np.clip(smooth[ys, xs], 0, None) + 1e-6
    cy, cx = np.average(ys, weights=w), np.average(xs, weights=w)
    cov = np.cov(np.vstack([ys - cy, xs - cx]), aweights=w)
    evals, evecs = np.linalg.eigh(cov)
    ax = evecs[:, 1]                      # major axis, (row, col)
    perp = np.array([-ax[1], ax[0]])
    t = (ys - cy) * ax[0] + (xs - cx) * ax[1]
    t0, t1 = t.min() + trim, t.max() - trim
    extent = float(t.max() - t.min())
    if t1 - t0 < 3:
        return None, extent
    ts = np.arange(t0, t1 + 1e-6, 1.0)
    samples = []
    for off in (-1, 0, 1):
        rr = cy + ts * ax[0] + off * perp[0]
        cc = cx + ts * ax[1] + off * perp[1]
        samples.append(ndi.map_coordinates(smooth, [rr, cc], order=1, mode="nearest"))
    return np.mean(samples, axis=0), extent


def evenness(profile):
    """p10 / p90 of the axial profile: ~0.5-1 for a streak, small for two stars joined by a faint bridge."""
    if profile is None or len(profile) < 3:
        return np.nan
    p10, p90 = np.percentile(profile, [10, 90])
    return float(np.clip(p10, 0, None) / p90) if p90 > 0 else np.nan


def split_blend(smooth, mask, noise, min_dist, seed_sigma):
    """Watershed on the smoothed image, one marker per significant local maximum."""
    peaks = peak_local_max(smooth, min_distance=min_dist, labels=mask.astype(int),
                           threshold_abs=seed_sigma * noise, exclude_border=False)
    if len(peaks) < 2:
        return None
    markers = np.zeros(mask.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return watershed(-smooth, markers, mask=mask)


# ---------------------------------------------------------------- per frame

def star_reference(objs):
    """Median weighted length of bright, unsaturated, round stars in this frame."""
    ref = objs[(objs.peak_snr > 20) & (objs.peak_snr < 200) & (objs.elong_w < 1.5) & ~objs.saturated]
    if len(ref) < 5:        # small crops only; every full frame has hundreds of reference stars
        return 9.0          # typical value measured on the 10 frames (8.6-9.5 px)
    return float(ref.length_w.median())


def classify_frame(raw, labels, objs, params, progress=None, L_star=None):
    """`L_star` can be supplied (e.g. the full-frame value when classifying a small crop)."""
    c = params["classify"]
    L_star = star_reference(objs) if L_star is None else L_star
    h, w = labels.shape
    objs = objs.copy()
    objs["edge"] = (objs.y0 == 0) | (objs.x0 == 0) | (objs.y1 == h) | (objs.x1 == w)
    objs["cls"] = STAR
    objs["reason"] = "star"
    objs["evenness"] = np.nan
    objs["extent"] = np.nan

    cand = objs[(objs.elong_w >= c["short_elong"]) & (objs.length_w >= c["short_len_factor"] * L_star)]
    slices = ndi.find_objects(labels)
    new_id = int(labels.max()) + 1
    new_rows, drop = [], []
    it = cand.itertuples()
    if progress is not None:
        it = progress(it, total=len(cand))
    for row in it:
        sl0 = slices[row.label - 1]
        pad = 6
        sl = (slice(max(sl0[0].start - pad, 0), min(sl0[0].stop + pad, h)),
              slice(max(sl0[1].start - pad, 0), min(sl0[1].stop + pad, w)))
        lab_crop = labels[sl]
        mask = lab_crop == row.label
        smooth, noise = local_maps(raw, sl, lab_crop == 0, c["psf_sigma_px"])
        profile, extent = axial_profile(smooth, mask, c["profile_trim_px"])
        ev = evenness(profile)
        objs.loc[row.Index, ["evenness", "extent"]] = [ev, extent]

        long_enough = row.length_w >= c["streak_len_factor"] * L_star and row.elong_w >= c["streak_elong"]
        short_fat = row.peak_snr >= c["short_min_snr"]
        if long_enough and ev >= c["min_evenness"]:
            objs.loc[row.Index, ["cls", "reason"]] = [STREAK, "streak"]
            continue
        if not long_enough and short_fat and ev >= c["short_min_evenness"]:
            objs.loc[row.Index, ["cls", "reason"]] = [STREAK, "short_streak"]
            continue
        if not ev >= (c["min_evenness"] if long_enough else c["short_min_evenness"]):
            parts = split_blend(smooth, mask, noise, c["blend_min_dist_px"], c["blend_seed_sigma"])
            if parts is not None:
                drop.append(row.Index)
                for k in range(1, parts.max() + 1):
                    part = parts == k
                    if part.sum() < params["detect"]["min_area"]:
                        continue
                    lab_crop[part] = new_id
                    ys, xs = np.nonzero(part)
                    new_rows.append({"label": new_id, "cls": STAR, "reason": "blend_split",
                                     "area": int(part.sum()),
                                     "cy": ys.mean() + sl[0].start, "cx": xs.mean() + sl[1].start,
                                     "peak_snr": row.peak_snr, "edge": row.edge})
                    new_id += 1
                # any leftover pixels of the original blend (too-small pieces) are released
                lab_crop[lab_crop == row.label] = 0
                continue
        objs.loc[row.Index, "reason"] = "elongated_star"
    objs = objs.drop(index=drop)
    if new_rows:
        objs = pd.concat([objs, pd.DataFrame(new_rows)], ignore_index=True)
    return labels, objs, L_star


def grey_zone(objs, c):
    """Cases closest to a decision boundary: shown on the review sheet."""
    near_ev = ((objs.evenness - c["min_evenness"]).abs() < 0.08) | \
              ((objs.evenness - c["short_min_evenness"]).abs() < 0.08)
    near_el = (objs.elong_w - c["streak_elong"]).abs() < 0.4
    return objs[(near_ev | near_el) & objs.evenness.notna()]


def main():
    CLS_DIR.mkdir(parents=True, exist_ok=True)
    params = load_params()
    c = params["classify"]
    summary, streak_crops, grey_crops = [], [], []
    frames = list_frames()
    for path in tqdm(frames, desc="frames", unit="frame", file=sys.stdout):
        name = short_name(path)
        raw, _ = load_frame(path)
        labels = np.load(DET_DIR / f"{name}_labels.npz")["labels"]
        objs = pd.read_csv(DET_DIR / f"{name}_objects.csv")
        prog = lambda it, total: tqdm(it, total=total, desc=f"  {name[:8]} candidates",
                                      leave=False, file=sys.stdout)
        labels, objs, L_star = classify_frame(raw, labels, objs, params, prog)
        np.savez_compressed(CLS_DIR / f"{name}_instances.npz", labels=labels)
        objs.to_csv(CLS_DIR / f"{name}_objects.csv", index=False)

        counts = objs.reason.value_counts().to_dict()
        summary.append({"frame": name, "L_star": round(L_star, 2),
                        "stars": int((objs.cls == STAR).sum()), "streaks": int((objs.cls == STREAK).sum()),
                        **counts})

        display = cv2.imread(str(PROC_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        for r in objs[objs.cls == STREAK].itertuples():
            streak_crops.append((crop_with_contour(display, labels, r),
                                 f"{name[:8]} {r.reason} e={r.elong_w:.1f} ev={r.evenness:.2f}"))
        for r in grey_zone(objs, c).head(8).itertuples():
            if not hasattr(r, "y0") or np.isnan(r.y0):
                continue
            grey_crops.append((crop_with_contour(display, labels, r),
                               f"{name[:8]} cls={r.cls} e={r.elong_w:.1f} ev={r.evenness:.2f}"))

    s = pd.DataFrame(summary).fillna(0)
    s.to_csv(CLS_DIR / "summary.csv", index=False)
    print(s.to_string(index=False))
    review_sheet(streak_crops, CLS_DIR / "streaks_review.png")
    review_sheet(grey_crops, CLS_DIR / "grey_zone_review.png")


if __name__ == "__main__":
    main()

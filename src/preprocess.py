"""Step A1b: pre-processing.

Every later threshold is expressed in units of the local noise, so the three intensity
regimes found during inspection (median 1, 1-3 and ~45 counts) are handled by one code path.

Run directly to process all frames and write:
  outputs/processed/<name>.png        full-size 8-bit display image (input for the tiles)
  outputs/processed/preprocess.csv    background, noise, hot pixels and PSF per frame
  outputs/processed/<name>_steps.png  raw / background-subtracted / SNR comparison crop
"""
import csv

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter

from fits_io import OUT_DIR, ROOT, list_frames, load_frame, short_name

PROC_DIR = OUT_DIR / "processed"


def load_params():
    with open(ROOT / "configs" / "params.yaml") as f:
        return yaml.safe_load(f)


def saturation_level(img):
    """12-bit frames clip at 4095, CAM_B frames at 65535; the darkest frames never saturate."""
    top = float(img.max())
    return top if top in (4095.0, 65535.0) else None


# ---------------------------------------------------------------- background / noise maps

def _clipped_mesh(img, box, clip, iters):
    """Sigma-clipped mean and std in box x box cells. Returns two (ny, nx) arrays."""
    h, w = img.shape
    ph, pw = -h % box, -w % box
    padded = np.pad(img, ((0, ph), (0, pw)), mode="reflect")
    ny, nx = padded.shape[0] // box, padded.shape[1] // box
    cells = padded.reshape(ny, box, nx, box).transpose(0, 2, 1, 3).reshape(ny, nx, -1)

    centre = np.median(cells, axis=2, keepdims=True)
    spread = cells.std(axis=2, keepdims=True)
    for _ in range(iters):
        keep = np.where(np.abs(cells - centre) <= clip * spread, cells, np.nan)
        centre = np.nanmean(keep, axis=2, keepdims=True)
        spread = np.nanstd(keep, axis=2, keepdims=True)
    return centre[..., 0], spread[..., 0]


def _to_full(mesh, shape, box, mesh_filter):
    """Median-filter the mesh (rejects cells dominated by a bright star) and upsample smoothly."""
    mesh = median_filter(mesh, size=mesh_filter, mode="nearest")
    h, w = shape
    full = cv2.resize(mesh.astype(np.float32), (mesh.shape[1] * box, mesh.shape[0] * box),
                      interpolation=cv2.INTER_LINEAR)
    return full[:h, :w]


def background_and_noise(img, p):
    box = p["mesh_size"]
    bg_mesh, sig_mesh = _clipped_mesh(img, box, p["clip_sigma"], p["clip_iters"])
    bg = _to_full(bg_mesh, img.shape, box, p["mesh_filter"])
    sig = _to_full(sig_mesh, img.shape, box, p["mesh_filter"])
    return bg, np.maximum(sig, 1e-3)


# ---------------------------------------------------------------- hot pixels

def remove_hot_pixels(resid, sigma, p):
    """Remove pixels sharper than the optics allow.

    With a PSF FWHM of ~3.6 px, the 3x3 median around a star's peak is ~0.7 of the peak.
    A hot/warm pixel or cosmic-ray hit stands far above its 3x3 median (> hot_pixel_sigma)
    while that median is below hot_sharp_ratio of the pixel. Such pixels are replaced by the
    local median. A 'bright pixel with all-dark neighbours' rule is not enough: warm pixels
    often have one warm partner, and smoothing turns the pair into a fake 5-sigma star."""
    local = median_filter(resid, size=3, mode="nearest")
    hot = ((resid - local) > p["hot_pixel_sigma"] * sigma) & (local < p["hot_sharp_ratio"] * resid)
    resid = np.where(hot, local, resid)
    return resid, hot


# ---------------------------------------------------------------- PSF

def measure_psf(resid, sigma, raw, sat, p):
    """Second-moment size of bright, isolated, unsaturated stars.
    Returns (median FWHM px, median elongation, number of stars used)."""
    snr = resid / sigma
    peaks = (snr == maximum_filter(snr, size=2 * p["psf_isolation_px"] + 1)) & (snr > p["psf_min_snr"])
    if sat is not None:
        peaks &= raw < 0.8 * sat
    r = p["psf_box"] // 2
    h, w = resid.shape
    ys, xs = np.nonzero(peaks)
    inside = (ys >= r) & (ys < h - r) & (xs >= r) & (xs < w - r)
    ys, xs = ys[inside], xs[inside]
    if len(ys) > 400:  # plenty for a median; keep the brightest-but-not-extreme middle of the sample
        order = np.argsort(snr[ys, xs])[len(ys) // 4: len(ys) // 4 + 400]
        ys, xs = ys[order], xs[order]

    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    fwhms, elongs = [], []
    for y, x in zip(ys, xs):
        stamp = np.clip(resid[y - r:y + r + 1, x - r:x + r + 1], 0, None)
        total = stamp.sum()
        if total <= 0:
            continue
        cy, cx = (stamp * yy).sum() / total, (stamp * xx).sum() / total
        myy = (stamp * (yy - cy) ** 2).sum() / total
        mxx = (stamp * (xx - cx) ** 2).sum() / total
        mxy = (stamp * (yy - cy) * (xx - cx)).sum() / total
        root = np.sqrt(((mxx - myy) / 2) ** 2 + mxy ** 2)
        l1, l2 = (mxx + myy) / 2 + root, (mxx + myy) / 2 - root
        if l2 <= 0:
            continue
        fwhms.append(2.3548 * np.sqrt((l1 + l2) / 2))
        elongs.append(np.sqrt(l1 / l2))
    if not fwhms:
        return 2.0, 1.0, 0  # fallback, flagged in the CSV by n=0
    return float(np.median(fwhms)), float(np.median(elongs)), len(fwhms)


# ---------------------------------------------------------------- display

def to_display(snr, p):
    """asinh stretch in noise units: every frame, whatever its raw level, looks alike."""
    soft, top = p["display_soft"], p["display_top_sigma"]
    x = np.clip(snr, -1, top)
    lo, hi = np.arcsinh(-1 / soft), np.arcsinh(top / soft)
    return ((np.arcsinh(x / soft) - lo) / (hi - lo) * 255).astype(np.uint8)


# ---------------------------------------------------------------- pipeline entry point

def preprocess(raw, params=None):
    """Return a dict with everything later steps need. `raw` is the float32 frame."""
    p = (params or load_params())["preprocess"]
    sat = saturation_level(raw)

    bg, sigma = background_and_noise(raw, p)
    resid = raw - bg
    resid, hot = remove_hot_pixels(resid, sigma, p)
    fwhm, psf_elong, n_psf = measure_psf(resid, sigma, raw, sat, p)

    # Matched filter: a Gaussian as wide as the PSF maximises SNR for point sources and
    # for the cross-section of a streak. Its noise is measured, not assumed, because the
    # raw noise is discrete and partly clipped at zero.
    psf_sigma = fwhm / 2.3548
    smooth = gaussian_filter(resid, psf_sigma)
    _, sigma_smooth = background_and_noise(smooth, p)

    return {
        "resid": resid,                        # background-subtracted, hot pixels removed
        "sigma": sigma,                        # per-pixel noise map of resid
        "snr": resid / sigma,                  # pixel-level SNR -> used for mask extent
        "snr_smooth": smooth / sigma_smooth,   # matched-filter SNR -> used for detection
        "display": to_display(resid / sigma, p),
        "saturated": raw >= sat if sat is not None else np.zeros(raw.shape, bool),
        "stats": {
            "bg_min": float(bg.min()), "bg_max": float(bg.max()),
            "sigma_median": float(np.median(sigma)),
            "sigma_smooth_median": float(np.median(sigma_smooth)),
            "hot_pixels": int(hot.sum()),
            "saturation": sat or "none",
            "psf_fwhm_px": round(fwhm, 2), "psf_elongation": round(psf_elong, 2), "psf_n_stars": n_psf,
        },
    }


def save_steps_figure(raw, out, name, path, size=256):
    """Raw vs processed on the same crop, for the write-up."""
    cy, cx = raw.shape[0] // 2, raw.shape[1] // 2
    sl = np.s_[cy - size // 2:cy + size // 2, cx - size // 2:cx + size // 2]
    raw_crop = raw[sl]
    lo, hi = np.percentile(raw_crop, [1, 99.5])
    panels = [(raw_crop, "raw (linear, 1-99.5%)", dict(vmin=lo, vmax=hi)),
              (out["display"][sl], "bg-subtracted, asinh", dict(vmin=0, vmax=255)),
              (out["snr_smooth"][sl], "matched-filter SNR", dict(vmin=-2, vmax=10))]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, (im, title, kw) in zip(axes, panels):
        ax.imshow(im, cmap="gray", **kw)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(name, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    params = load_params()
    rows = []
    for path in list_frames():
        name = short_name(path)
        raw, _ = load_frame(path)
        out = preprocess(raw, params)
        row = {"name": name, **out["stats"]}
        rows.append(row)
        print(row)
        cv2.imwrite(str(PROC_DIR / f"{name}.png"), out["display"])
        save_steps_figure(raw, out, name, PROC_DIR / f"{name}_steps.png")

    with open(PROC_DIR / "preprocess.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

"""Step A1a: inspect every raw frame before deciding on pre-processing.

Outputs (outputs/inspection/):
  stats.csv                 one row of statistics per frame
  <name>_preview.png        whole frame, 8x binned, asinh-stretched
  <name>_crop.png           central 1024x1024 at full resolution, asinh-stretched
  <name>_hist.png           histogram of the low end of the pixel values
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

from fits_io import OUT_DIR, list_frames, load_frame, short_name

INSPECT_DIR = OUT_DIR / "inspection"
BIN = 8            # preview binning factor
MESH = 256         # box size for the background-gradient check


def noise_sigma(img):
    """Robust noise estimate.

    MAD is 0 for frames whose median is 1 count, and 10-30% of pixels are clipped to 0,
    so the lower half of the histogram is unusable. Use the upper half instead:
    for a Gaussian, the 84th percentile sits 1 sigma above the median.
    """
    med = np.median(img)
    p84 = np.percentile(img, 84.13)
    return float(med), float(max(p84 - med, 0.5))  # floor at half a count (quantisation limit)


def background_mesh(img):
    """Median in MESH x MESH boxes -> reveals large-scale gradients / vignetting."""
    h, w = (img.shape[0] // MESH) * MESH, (img.shape[1] // MESH) * MESH
    boxes = img[:h, :w].reshape(h // MESH, MESH, w // MESH, MESH)
    return np.median(boxes, axis=(1, 3))


def hot_pixel_count(img, med, sigma):
    """Isolated bright pixels: far above noise while their 3x3 neighbourhood median is background."""
    thresh = med + 10 * sigma
    candidates = img > thresh
    if not candidates.any():
        return 0
    local = median_filter(img, size=3)
    return int((candidates & (local < med + 2 * sigma)).sum())


def bad_lines(img, med, sigma):
    """Rows/columns whose median deviates strongly from the frame median."""
    col = np.median(img, axis=0)
    row = np.median(img, axis=1)
    lim = 3 * sigma
    return int((np.abs(col - med) > lim).sum()), int((np.abs(row - med) > lim).sum())


def asinh_stretch(img, med, sigma, soft=3.0, top_sigma=60.0):
    """Map [median - 1 sigma, median + top_sigma*sigma] through asinh to 0-255."""
    x = (img - med) / sigma
    x = np.clip(x, -1, top_sigma)
    y = np.arcsinh(x / soft)
    y = (y - y.min()) / (np.arcsinh(top_sigma / soft) - np.arcsinh(-1 / soft))
    return (np.clip(y, 0, 1) * 255).astype(np.uint8)


def binned(img, f):
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    return img[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def save_hist(img, name, med, sigma, path):
    top = int(med + 30 * sigma) + 2
    vals = np.clip(img, 0, top).astype(np.int32).ravel()
    counts = np.bincount(vals, minlength=top + 1)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(np.arange(len(counts)), counts, width=1.0)
    ax.set_yscale("log")
    ax.set_xlabel(f"pixel value (counts; last bar = >= {top})")
    ax.set_ylabel("number of pixels")
    ax.set_title(f"{name}: median={med:.0f}, sigma~{sigma:.1f}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    INSPECT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in list_frames():
        name = short_name(path)
        img, hdr = load_frame(path)
        med, sigma = noise_sigma(img)
        mesh = background_mesh(img)
        bad_cols, bad_rows = bad_lines(img, med, sigma)

        row = {
            "name": name,
            "shape": f"{img.shape[1]}x{img.shape[0]}",
            "serial": hdr.get("SERIAL", ""),
            "min": int(img.min()),
            "max": int(img.max()),
            "median": med,
            "sigma_upper": round(sigma, 2),
            "pct_zero": round(100 * float((img == 0).mean()), 1),
            "n_saturated": int((img >= img.max()).sum()) if img.max() in (4095, 65535) else 0,
            "p99_9": float(np.percentile(img, 99.9)),
            "bg_mesh_min": float(mesh.min()),
            "bg_mesh_max": float(mesh.max()),
            "hot_pixels": hot_pixel_count(img, med, sigma),
            "bad_cols": bad_cols,
            "bad_rows": bad_rows,
        }
        rows.append(row)
        print(row)

        plt.imsave(INSPECT_DIR / f"{name}_preview.png",
                   asinh_stretch(binned(img, BIN), med, sigma / BIN), cmap="gray")
        cy, cx = img.shape[0] // 2, img.shape[1] // 2
        crop = img[cy - 512:cy + 512, cx - 512:cx + 512]
        plt.imsave(INSPECT_DIR / f"{name}_crop.png", asinh_stretch(crop, med, sigma), cmap="gray")
        save_hist(img, name, med, sigma, INSPECT_DIR / f"{name}_hist.png")

    with open(INSPECT_DIR / "stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {INSPECT_DIR / 'stats.csv'}")


if __name__ == "__main__":
    main()

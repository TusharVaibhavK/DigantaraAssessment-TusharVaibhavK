"""Step A3a: detection, pixel-level segmentation and shape measurement (full frame, before tiling).

For every frame writes:
  outputs/processed/<name>.png          8-bit display image (the source of the tiles)
  outputs/processed/<name>_steps.png    raw / background-subtracted / matched-filter comparison crop
  outputs/processed/preprocess.csv      background, noise, hot pixels and PSF per frame
  outputs/detections/<name>_labels.npz  uint32 label map, one id per object (0 = background)
  outputs/detections/<name>_objects.csv one row of shape / brightness features per object
and, over all frames:
  outputs/detections/shape_scatter.png  length vs elongation of every object
  outputs/detections/elongated_review.png  crops of the most elongated objects for visual review
"""
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
from skimage.morphology import convex_hull_image
from tqdm import tqdm

from fits_io import OUT_DIR, list_frames, load_frame, short_name
from preprocess import PROC_DIR, load_params, preprocess, save_steps_figure

DET_DIR = OUT_DIR / "detections"


# ---------------------------------------------------------------- segmentation

def hysteresis_labels(snr_smooth, p):
    """Connected regions above grow_sigma that contain at least one pixel above seed_sigma."""
    labels, n = ndi.label(snr_smooth > p["grow_sigma"], structure=np.ones((3, 3)))
    seeded = np.unique(labels[snr_smooth > p["seed_sigma"]])
    seeded = seeded[seeded > 0]
    areas = ndi.sum_labels(np.ones_like(labels, dtype=np.uint8), labels, seeded)
    seeded = seeded[areas >= p["min_area"]]
    lut = np.zeros(n + 1, np.uint32)
    lut[seeded] = np.arange(1, len(seeded) + 1, dtype=np.uint32)
    return lut[labels]


# ---------------------------------------------------------------- measurement

def measure(labels, pre):
    """Geometry from the mask, elongation from SNR-weighted moments, brightness from the SNR maps."""
    weight = np.clip(pre["snr_smooth"], 0, None)
    props = pd.DataFrame(regionprops_table(
        labels, intensity_image=weight,
        properties=("label", "area", "centroid", "bbox", "orientation",
                    "major_axis_length", "minor_axis_length", "solidity",
                    "moments_weighted_central")))
    m00 = props["moments_weighted_central-0-0"]
    mrr = props["moments_weighted_central-2-0"] / m00
    mcc = props["moments_weighted_central-0-2"] / m00
    mrc = props["moments_weighted_central-1-1"] / m00
    root = np.sqrt(((mrr - mcc) / 2) ** 2 + mrc ** 2)
    l1 = (mrr + mcc) / 2 + root
    l2 = np.maximum((mrr + mcc) / 2 - root, 0.05)
    props = props.drop(columns=[c for c in props if c.startswith("moments_weighted")])

    idx = props["label"].to_numpy()
    out = pd.DataFrame({
        "label": idx,
        "area": props["area"],
        "cy": props["centroid-0"], "cx": props["centroid-1"],
        "y0": props["bbox-0"], "x0": props["bbox-1"], "y1": props["bbox-2"], "x1": props["bbox-3"],
        "orientation": props["orientation"],   # radians, skimage convention (rows vs cols)
        "length": props["major_axis_length"],
        "width": props["minor_axis_length"],
        "solidity": props["solidity"],
        "length_w": 4 * np.sqrt(l1),            # brightness-weighted ellipse major axis
        "width_w": 4 * np.sqrt(l2),
        "elong_w": np.sqrt(l1 / l2),
        "peak_snr": ndi.maximum(pre["snr_smooth"], labels, idx),
        "int_snr": ndi.sum_labels(pre["snr"], labels, idx) / np.sqrt(props["area"]),
        "saturated": ndi.maximum(pre["saturated"], labels, idx).astype(bool),
    })
    return out


# ---------------------------------------------------------------- fragment linking

def link_fragments(labels, objs, p):
    """Merge pieces of one streak that the thresholds split: an elongated segment absorbs any
    object lying on its axis within link_gap_px of its end (and, if that object is itself
    elongated, pointing the same way)."""
    elong = objs[objs["elong_w"] >= p["link_min_elong"]]
    if elong.empty:
        return labels, 0, set()
    parent = {int(l): int(l) for l in objs["label"]}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree = cKDTree(objs[["cy", "cx"]].to_numpy())
    lab = objs["label"].to_numpy()
    ori = objs["orientation"].to_numpy()
    el = objs["elong_w"].to_numpy()
    pts = objs[["cy", "cx"]].to_numpy()
    merges = 0
    for i in np.nonzero(objs["elong_w"].to_numpy() >= p["link_min_elong"])[0]:
        axis = np.array([np.cos(ori[i]), np.sin(ori[i])])   # (row, col) direction of the major axis
        half = objs["length"].iat[i] / 2
        for j in tree.query_ball_point(pts[i], half + p["link_gap_px"]):
            if j == i:
                continue
            d = pts[j] - pts[i]
            along, perp = abs(d @ axis), abs(d[0] * axis[1] - d[1] * axis[0])
            if perp > p["link_perp_px"] or along < half:
                continue
            if el[j] >= p["link_min_elong"]:
                dang = abs((ori[i] - ori[j] + np.pi / 2) % np.pi - np.pi / 2)
                if np.degrees(dang) > p["link_angle_deg"]:
                    continue
            a, b = find(int(lab[i])), find(int(lab[j]))
            if a != b:
                parent[b] = a
                merges += 1
    if merges == 0:
        return labels, 0, set()
    lut = np.arange(labels.max() + 1, dtype=np.uint32)
    for l in parent:
        lut[l] = find(l)
    merged = lut[labels]
    groups = {find(l) for l in parent if find(l) != l}
    return merged, merges, groups


def bridge_gaps(labels, snr_smooth, groups, bridge_sigma):
    """Merged pieces are still separate islands. Inside the convex hull of each merged streak,
    add the background pixels that still carry signal, so the mask becomes one continuous streak."""
    slices = ndi.find_objects(labels)
    for g in groups:
        sl = slices[g - 1]
        if sl is None:
            continue
        mask = labels[sl] == g
        hull = convex_hull_image(mask)
        add = hull & (labels[sl] == 0) & (snr_smooth[sl] > bridge_sigma)
        labels[sl][add] = g
    return labels


# ---------------------------------------------------------------- review figures

def shape_scatter(all_objs, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(all_objs["length_w"], all_objs["elong_w"], c=np.log10(all_objs["peak_snr"]),
                    s=3, cmap="viridis", alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("brightness-weighted length (px)")
    ax.set_ylabel("elongation (major / minor)")
    fig.colorbar(sc, label="log10 peak matched-filter SNR")
    ax.set_title(f"{len(all_objs)} detections, all frames")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def review_sheet(crops, path, cols=8):
    if not crops:
        return
    rows = int(np.ceil(len(crops) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.0))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (img, title) in zip(np.atleast_1d(axes).ravel(), crops):
        ax.imshow(img)
        ax.set_title(title, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def crop_with_contour(display, labels, row, pad=24):
    h, w = display.shape
    y0, x0 = max(int(row.y0) - pad, 0), max(int(row.x0) - pad, 0)
    y1, x1 = min(int(row.y1) + pad, h), min(int(row.x1) + pad, w)
    rgb = cv2.cvtColor(display[y0:y1, x0:x1], cv2.COLOR_GRAY2RGB)
    mask = (labels[y0:y1, x0:x1] == row.label).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, contours, -1, (255, 60, 60), 1)
    return rgb


# ---------------------------------------------------------------- driver

def detect_frame(raw, params):
    pre = preprocess(raw, params)
    p = params["detect"]
    labels = hysteresis_labels(pre["snr_smooth"], p)
    objs = measure(labels, pre)
    merges, groups = 0, set()
    if p["link_enabled"]:
        labels, merges, groups = link_fragments(labels, objs, p)
    if merges:
        labels = bridge_gaps(labels, pre["snr_smooth"], groups, p["bridge_sigma"])
        objs = measure(labels, pre)
    return pre, labels, objs, merges


def main():
    DET_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    params = load_params()
    all_objs, crops, pre_stats = [], [], []
    for path in tqdm(list_frames(), desc="detect", unit="frame", file=sys.stdout):
        name = short_name(path)
        raw, _ = load_frame(path)
        pre, labels, objs, merges = detect_frame(raw, params)
        objs.insert(0, "frame", name)
        objs.to_csv(DET_DIR / f"{name}_objects.csv", index=False)
        np.savez_compressed(DET_DIR / f"{name}_labels.npz", labels=labels)
        cv2.imwrite(str(PROC_DIR / f"{name}.png"), pre["display"])
        save_steps_figure(raw, pre, name, PROC_DIR / f"{name}_steps.png")
        pre_stats.append({"name": name, **pre["stats"]})
        all_objs.append(objs)

        n_el = int(((objs.elong_w >= 3) & (objs.length_w >= 12)).sum())
        tqdm.write(f"  {name}: {len(objs)} objects, {n_el} with elongation>=3 & length>=12px, "
                   f"{int(objs.saturated.sum())} saturated, {pre['stats']['hot_pixels']} hot pixels removed")

        for row in objs.sort_values("elong_w", ascending=False).head(6).itertuples():
            crops.append((crop_with_contour(pre["display"], labels, row),
                          f"{name[:8]} e={row.elong_w:.1f} L={row.length_w:.0f}"))

    pd.DataFrame(pre_stats).to_csv(PROC_DIR / "preprocess.csv", index=False)
    all_objs = pd.concat(all_objs, ignore_index=True)
    all_objs.to_csv(DET_DIR / "all_objects.csv", index=False)
    shape_scatter(all_objs, DET_DIR / "shape_scatter.png")
    review_sheet(crops, DET_DIR / "elongated_review.png")


if __name__ == "__main__":
    main()

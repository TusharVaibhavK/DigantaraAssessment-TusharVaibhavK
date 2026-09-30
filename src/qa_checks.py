"""QA 2: automated sanity checks on the final annotations.

  counts.csv            per-frame objects per class, edge objects, blend splits
  shape_hist.png        elongation and length distributions per class
  collinear.csv/.png    faint 'stars' that sit in straight, evenly spaced rows of >= 3
                        (a streak broken into dots would look like this) -> for visual review
  dataset checks        tiles per frame, tile size, label syntax, empty label files
"""
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from classify import CLS_DIR, STAR, STREAK
from detect import review_sheet
from export_yolo import DS_DIR
from fits_io import OUT_DIR
from preprocess import PROC_DIR

QA_DIR = OUT_DIR / "qa"
ROW_RADIUS = 25        # px, max spacing between neighbouring dots of a broken streak
ROW_MAX_ANGLE = 12     # deg, max bend at the middle dot
ROW_SPACING_RATIO = 1.5
ROW_MAX_SNR = 15       # only faint dots; bright stars in a line are just stars


def collinear_rows(objs):
    """Middle dots of evenly spaced, nearly straight triplets of faint star-class objects."""
    s = objs[(objs.cls == STAR) & (objs.peak_snr < ROW_MAX_SNR)].reset_index(drop=True)
    pts = s[["cy", "cx"]].to_numpy()
    tree = cKDTree(pts)
    flagged = []
    for i, nb in enumerate(tree.query_ball_point(pts, ROW_RADIUS)):
        nb = [j for j in nb if j != i]
        found = False
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                va, vb = pts[nb[a]] - pts[i], pts[nb[b]] - pts[i]
                da, db = np.linalg.norm(va), np.linalg.norm(vb)
                cosang = va @ vb / (da * db)
                if cosang < -np.cos(np.radians(ROW_MAX_ANGLE)) and max(da, db) / min(da, db) < ROW_SPACING_RATIO:
                    found = True
                    break
            if found:
                break
        if found:
            flagged.append(i)
    return s.loc[flagged]


def dataset_checks():
    out = {}
    tiles = pd.read_csv(DS_DIR / "tiles.csv")
    out["tiles_per_frame"] = sorted(set(tiles.groupby("frame").size()))
    shapes, bad, empty, lines = set(), 0, 0, 0
    for t in tiles.itertuples():
        if t.row == 0 and t.col in (0, 9):
            shapes.add(cv2.imread(str(DS_DIR / "images" / t.split / f"{t.tile}.png"), cv2.IMREAD_GRAYSCALE).shape)
        text = (DS_DIR / "labels" / t.split / f"{t.tile}.txt").read_text().split("\n")
        rows = [r for r in text if r.strip()]
        empty += not rows
        for r in rows:
            lines += 1
            v = r.split()
            xy = np.array(v[1:], float)
            if v[0] not in ("0", "1") or len(xy) < 6 or len(xy) % 2 or xy.min() < 0 or xy.max() > 1:
                bad += 1
    out.update(tile_shapes=sorted(shapes), label_lines=lines, malformed_lines=bad, empty_label_files=empty)
    return out


def main():
    QA_DIR.mkdir(parents=True, exist_ok=True)
    counts, all_objs, row_crops, row_tables = [], [], [], []
    for p in tqdm(sorted(CLS_DIR.glob("*_objects.csv")), desc="qa frames", file=sys.stdout):
        name = p.name.replace("_objects.csv", "")
        objs = pd.read_csv(p)
        objs["frame"] = name
        all_objs.append(objs)
        rows = collinear_rows(objs)
        rows = rows.assign(frame=name)
        row_tables.append(rows)
        counts.append({"frame": name, "stars": int((objs.cls == STAR).sum()),
                       "streaks": int((objs.cls == STREAK).sum()),
                       "blend_splits": int((objs.reason == "blend_split").sum()),
                       "edge_objects": int(objs.edge.fillna(False).astype(bool).sum()),
                       "collinear_faint_dots": len(rows)})
        if len(rows):
            disp = cv2.imread(str(PROC_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            for r in rows.head(6).itertuples():
                y, x = int(r.cy), int(r.cx)
                crop = disp[max(y - 32, 0):y + 32, max(x - 32, 0):x + 32]
                row_crops.append((cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB), f"{name[:8]} snr={r.peak_snr:.1f}"))

    c = pd.DataFrame(counts)
    c.to_csv(QA_DIR / "counts.csv", index=False)
    print(c.to_string(index=False))
    pd.concat(row_tables).to_csv(QA_DIR / "collinear.csv", index=False)
    review_sheet(row_crops, QA_DIR / "collinear_review.png")

    a = pd.concat(all_objs, ignore_index=True)
    a = a[a.elong_w.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for cls, name, col in ((STAR, "star", "tab:blue"), (STREAK, "streak", "tab:red")):
        g = a[a.cls == cls]
        axes[0].hist(g.elong_w, bins=np.logspace(0, 1.1, 40), color=col, alpha=0.6, label=f"{name} (n={len(g)})",
                     density=True)
        axes[1].hist(g.length_w, bins=np.logspace(0.4, 2, 40), color=col, alpha=0.6, label=name, density=True)
    for ax, lab in zip(axes, ("elongation", "weighted length (px)")):
        ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(QA_DIR / "shape_hist.png", dpi=110)
    plt.close(fig)

    print("\ndataset:", dataset_checks())


if __name__ == "__main__":
    main()

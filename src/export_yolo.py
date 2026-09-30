"""Step A2 + A3c: cut every frame into 1024 x 1024 tiles and write YOLO (Ultralytics) segmentation labels.

Annotation happens on the full frame first, tiling second, so an object crossing a tile edge has one
consistent mask and class; each tile just receives the clipped part of it.

Output (outputs/yolo_dataset/):
  images/{train,val}/<frame>_y<y0>_x<x0>.png   8-bit display tile
  labels/{train,val}/<frame>_y<y0>_x<x0>.txt   one line per object: <class> x1 y1 x2 y2 ... (normalised)
  data.yaml                                    Ultralytics dataset file (0: star, 1: streak)
  tiles.csv                                    tile origins and per-class counts (used for repatching)
"""
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage as ndi
from tqdm import tqdm

from classify import CLS_DIR
from fits_io import OUT_DIR, list_frames, short_name
from preprocess import PROC_DIR, load_params
from tiling import tile_grid

DS_DIR = OUT_DIR / "yolo_dataset"
CLASS_NAMES = {0: "star", 1: "streak"}


def mask_to_polygons(mask, upsample, min_area, epsilon):
    """Polygons (in pixel-edge coordinates of `mask`) tracing the outer boundary of each piece.

    Contours are traced on a nearest-neighbour upsampled copy so the polygon follows pixel edges,
    not pixel centres; otherwise a 3x3 star would shrink to a 2x2 polygon."""
    if mask.sum() < min_area:
        return []
    big = cv2.resize(mask.astype(np.uint8), None, fx=upsample, fy=upsample, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area * upsample * upsample * 0.5:
            continue
        cnt = cv2.approxPolyDP(cnt, epsilon * upsample, True)
        if len(cnt) < 3:
            continue
        # continuous coordinates (pixel j spans [j, j+1)): a boundary sub-pixel centre is
        # (i + 0.5) / upsample, i.e. 1/(2*upsample) px inside the true pixel edge
        pts = (cnt[:, 0, :].astype(np.float64) + 0.5) / upsample
        polys.append(pts)
    return polys


def export_frame(name, display, instances, objs, split, cfg, tile):
    cls_of = dict(zip(objs["label"].astype(int), objs["cls"].astype(int)))
    slices = ndi.find_objects(instances)
    ids = np.array([i + 1 for i, s in enumerate(slices) if s is not None])
    boxes = np.array([[slices[i - 1][0].start, slices[i - 1][1].start,
                       slices[i - 1][0].stop, slices[i - 1][1].stop] for i in ids])
    h, w = instances.shape
    rows, missing = [], 0
    for r, c, y0, x0 in tile_grid(h, w, tile):
        stem = f"{name}_y{y0:04d}_x{x0:04d}"
        cv2.imwrite(str(DS_DIR / "images" / split / f"{stem}.png"), display[y0:y0 + tile, x0:x0 + tile])
        hit = (boxes[:, 0] < y0 + tile) & (boxes[:, 2] > y0) & (boxes[:, 1] < x0 + tile) & (boxes[:, 3] > x0)
        lines, counts = [], {0: 0, 1: 0}
        for lid, (by0, bx0, by1, bx1) in zip(ids[hit], boxes[hit]):
            if lid not in cls_of:
                missing += 1
                continue
            cy0, cx0 = max(by0, y0), max(bx0, x0)
            cy1, cx1 = min(by1, y0 + tile), min(bx1, x0 + tile)
            mask = instances[cy0:cy1, cx0:cx1] == lid
            for poly in mask_to_polygons(mask, cfg["upsample"], cfg["min_piece_px"], cfg["simplify_px"]):
                poly[:, 0] = (poly[:, 0] + cx0 - x0) / tile
                poly[:, 1] = (poly[:, 1] + cy0 - y0) / tile
                poly = np.clip(poly, 0.0, 1.0)
                k = cls_of[lid]
                lines.append(f"{k} " + " ".join(f"{v:.6f}" for v in poly.ravel()))
                counts[k] += 1
        (DS_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        rows.append({"frame": name, "split": split, "tile": stem, "row": r, "col": c, "y0": y0, "x0": x0,
                     "stars": counts[0], "streaks": counts[1]})
    return rows, missing


def main():
    params = load_params()
    cfg = params["export"]
    tile = cfg["tile_size"]
    for sub in ("images", "labels"):          # start clean so no stale tiles survive a re-run
        shutil.rmtree(DS_DIR / sub, ignore_errors=True)
    for split in ("train", "val"):
        (DS_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DS_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_rows = []
    for path in tqdm(list_frames(), desc="frames", unit="frame", file=sys.stdout):
        name = short_name(path)
        split = "val" if any(name.startswith(v) for v in cfg["val_frames"]) else "train"
        display = cv2.imread(str(PROC_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        instances = np.load(CLS_DIR / f"{name}_instances.npz")["labels"]
        objs = pd.read_csv(CLS_DIR / f"{name}_objects.csv")
        rows, missing = export_frame(name, display, instances, objs, split, cfg, tile)
        if missing:
            print(f"  warning: {name}: {missing} instance ids without a class")
        all_rows += rows

    tiles = pd.DataFrame(all_rows)
    tiles.to_csv(DS_DIR / "tiles.csv", index=False)
    # no 'path' key: Ultralytics then resolves train/val relative to this file, so the folder is portable
    (DS_DIR / "data.yaml").write_text(yaml.safe_dump({
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }, sort_keys=False))
    print(tiles.groupby("frame")[["stars", "streaks"]].sum().assign(tiles=tiles.groupby("frame").size()))
    print(f"total tiles {len(tiles)}, star polygons {tiles.stars.sum()}, streak polygons {tiles.streaks.sum()}")


if __name__ == "__main__":
    main()

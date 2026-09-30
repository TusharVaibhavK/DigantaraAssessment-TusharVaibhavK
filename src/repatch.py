"""Step A3d: rebuild full-size 9568 x 6380 frames from the exported tiles + YOLO labels only.

This is both the visual-inspection deliverable and an end-to-end check of the export: nothing
from the internal pipeline is used except tiles.csv (tile origins).

Output (outputs/repatched/):
  <name>_repatched.png  8-bit full frame stitched from the tiles
  <name>_classmask.png  0 = background, 1 = star, 2 = streak (from the polygons)
  <name>_overlay.jpg    full frame with stars (blue) and streaks (red) filled
  <name>_preview.jpg    same overlay, 4x smaller, for a quick look
  checks.csv            size / pixel / mask agreement per frame
"""
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from classify import CLS_DIR
from export_yolo import DS_DIR
from fits_io import OUT_DIR
from preprocess import PROC_DIR, load_params

REP_DIR = OUT_DIR / "repatched"
COLOURS = {1: (255, 90, 60), 2: (60, 60, 255)}      # BGR: star blue, streak red
ALPHA = 0.55


def read_polygons(path, tile):
    polys = []
    for line in path.read_text().splitlines():
        v = line.split()
        if v:
            polys.append((int(v[0]), np.array(v[1:], float).reshape(-1, 2) * tile))
    return polys


def fill(canvas, xy, value, x0, y0):
    """Continuous-coordinate polygon -> pixels whose centres lie inside (fillPoly centre convention)."""
    pts = np.round((xy + [x0, y0] - 0.5) * 16).astype(np.int32)
    cv2.fillPoly(canvas, [pts], value, shift=4)


def repatch_frame(name, tiles, tile):
    h = int(tiles.y0.max()) + tile
    w = int(tiles.x0.max()) + tile
    image = np.zeros((h, w), np.uint8)
    covered = np.zeros((h, w), bool)
    classmask = np.zeros((h, w), np.uint8)
    overlap_mismatch = 0
    for t in tiles.itertuples():
        img = cv2.imread(str(DS_DIR / "images" / t.split / f"{t.tile}.png"), cv2.IMREAD_GRAYSCALE)
        region = np.s_[t.y0:t.y0 + tile, t.x0:t.x0 + tile]
        seen = covered[region]
        overlap_mismatch += int((image[region][seen] != img[seen]).sum())
        image[region] = img
        covered[region] = True
        # streaks drawn after stars so a streak is never hidden by an overlapping star polygon
        polys = read_polygons(DS_DIR / "labels" / t.split / f"{t.tile}.txt", tile)
        for cls in (0, 1):
            for c, xy in polys:
                if c == cls:
                    fill(classmask, xy, cls + 1, t.x0, t.y0)
    return image, classmask, int((~covered).sum()), overlap_mismatch


def overlay(image, classmask):
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for value, colour in COLOURS.items():
        m = classmask == value
        rgb[m] = (rgb[m] * (1 - ALPHA) + np.array(colour) * ALPHA).astype(np.uint8)
    return rgb


def mask_agreement(name, classmask):
    """Pixel agreement between the polygon-derived mask and the pipeline's instance map + classes."""
    inst = np.load(CLS_DIR / f"{name}_instances.npz")["labels"]
    objs = pd.read_csv(CLS_DIR / f"{name}_objects.csv")
    lut = np.zeros(int(inst.max()) + 1, np.uint8)
    lut[objs.label.astype(int)] = objs.cls.astype(int) + 1
    truth = lut[inst]
    out = {}
    for value, cname in ((1, "star"), (2, "streak")):
        a, b = classmask == value, truth == value
        union = (a | b).sum()
        out[f"iou_{cname}"] = round(float((a & b).sum() / union), 4) if union else 1.0
    return out


def main():
    REP_DIR.mkdir(parents=True, exist_ok=True)
    tile = load_params()["export"]["tile_size"]
    tiles = pd.read_csv(DS_DIR / "tiles.csv")
    checks = []
    for name, group in tqdm(tiles.groupby("frame", sort=True), desc="repatching", unit="frame", file=sys.stdout):
        image, classmask, uncovered, mismatch = repatch_frame(name, group, tile)
        original = cv2.imread(str(PROC_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE)

        cv2.imwrite(str(REP_DIR / f"{name}_repatched.png"), image)
        cv2.imwrite(str(REP_DIR / f"{name}_classmask.png"), classmask)
        ov = overlay(image, classmask)
        cv2.imwrite(str(REP_DIR / f"{name}_overlay.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(REP_DIR / f"{name}_preview.jpg"),
                    cv2.resize(ov, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

        checks.append({
            "frame": name,
            "size": f"{image.shape[1]}x{image.shape[0]}",
            "size_matches": image.shape == original.shape,
            "uncovered_px": uncovered,
            "overlap_mismatch_px": mismatch,
            "pixels_identical": bool(np.array_equal(image, original)),
            "star_px": int((classmask == 1).sum()),
            "streak_px": int((classmask == 2).sum()),
            **mask_agreement(name, classmask),
        })
    df = pd.DataFrame(checks)
    df.to_csv(REP_DIR / "checks.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

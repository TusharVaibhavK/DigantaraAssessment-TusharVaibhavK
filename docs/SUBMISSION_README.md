# Digantara – AI/ML Data Annotation Intern assessment: submission guide

**Source code (GitHub):** <GITHUB_REPO_URL>
**Full archive (all tiles + full-size overlays, ~900 MB):** <FULL_ZIP_LINK>

This archive holds the written answer, the YOLO annotations and the visual-inspection outputs.
Everything was produced by the scripted pipeline in the repository (`python run_all.py`).

## Start here

1. `writeup/` – the written answers to Question 2 (a–d), max. 4 pages.
2. `full_size/previews/` – each full frame at 1/4 size with the masks drawn on (stars blue, streaks red).
   For a quick visual check open these first.
3. `full_size/overlays/` (full archive only) – the **repatched full-size frames (9568 x 6380)** with masks,
   stitched back from the exported 1024 x 1024 tiles and labels. Zoom in to see individual star masks.
4. `annotations/` – the YOLO Ultralytics segmentation annotations.

## Folder contents

| Path | What it is |
|---|---|
| `annotations/labels/{train,val}/*.txt` | one file per 1024 x 1024 tile; each line `<class> x1 y1 x2 y2 ...`, coordinates normalised to 0–1 |
| `annotations/images/{train,val}/*.png` | the 700 pre-processed 8-bit tiles (full archive only) |
| `annotations/data.yaml`, `classes.txt` | class ids: **0 = star (blob)**, **1 = streak** |
| `annotations/tiles.csv` | origin (x0, y0) of every tile in its full frame + per-class counts, used for repatching |
| `full_size/classmasks/*.png` | full-frame class mask rebuilt from the polygons: 0 background, 1 star, 2 streak (view with a contrast stretch) |
| `full_size/previews/*.jpg` | 1/4-size overlays |
| `qa/` | injection–recovery curve and table, per-frame counts, shape histograms, repatch checks, collinear-dot review |
| `figures/` | inspection statistics, raw vs pre-processed crops, streak and grey-zone review sheets, shape scatter |

## Naming

Tiles are named `<frame>_y<y0>_x<x0>`, where `(x0, y0)` is the tile's top-left pixel in the original frame.
UUID-named frames are shortened to their first block (e.g. `1a600998`); `CAM_B_*` frames keep their full name.
Tile origins are x = 0, 1024, …, 8192, **8544** and y = 0, 1024, …, 5120, **5356**: the last column and row are
shifted inward so every tile is exactly 1024 x 1024 with no padding (see Q2 b).

## Using the annotations

With the full archive extracted, the dataset loads directly in Ultralytics:

```python
from ultralytics import YOLO
YOLO("yolov8n-seg.pt").train(data="annotations/data.yaml", imgsz=1024)
```

Train/val split is by whole frame (val: `7030c2ac`, `CAM_B_20260815T154244_manual_f000276`) so no object appears in both.

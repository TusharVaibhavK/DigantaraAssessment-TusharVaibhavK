# SSA star / streak annotation pipeline

Semi-automatic pipeline that turns raw space-situational-awareness frames (FITS, 9568 x 6380, uint16)
into pixel-level instance masks for two classes, **star (blob)** and **streak (satellite / debris)**,
tiled to 1024 x 1024 and exported in **Ultralytics YOLO segmentation** format.

Built for the Digantara *AI/ML Data Annotation Intern* assessment. The written answers (Q2 a-d) are in the
submitted PDF; this repository holds the code that produced every annotation and figure.

## Results on the 10 assessment frames

| | |
|---|---|
| Tiles | 700 (70 per frame), all exactly 1024 x 1024, no padding |
| Annotations | ~80,000 stars, 62 streaks (97,340 star + 77 streak polygons after tiling) |
| Polygon fidelity | mask -> polygon -> mask IoU 0.9999 (stars), 1.000 (streaks) |
| Repatch check | all 10 frames rebuilt from tiles + labels: exact size, pixel-identical, 0 uncovered px |
| Star completeness | 50 % at peak = 2 sigma, 100 % from 3 sigma (injection test, all 3 intensity groups) |
| Streak completeness | >= 90 % classified correctly from ridge = 1.5 sigma for streaks >= 30 px |

## Pipeline

```
FITS ──► inspect ──► pre-process ──► detect + segment ──► classify ──► tile + YOLO export ──► repatch ──► QA
          stats      background,       matched filter,       star /        70 tiles/frame,       full-size   injection test,
          previews   noise map,        hysteresis masks,     streak /      pixel-edge polygons   overlays    sanity checks
                     hot pixels,       shape features        blend split
                     PSF, stretch
```

| Stage | Script | Key idea |
|---|---|---|
| Inspect | `src/inspect_fits.py` | Three intensity regimes (median 1 / 1-3 / ~45 counts); MAD noise = 0 on 5 frames |
| Pre-process | `src/preprocess.py` | Mesh background, sigma-clipped noise map, PSF-sharpness hot-pixel filter, matched filter, asinh display |
| Detect | `src/detect.py` | Seed at 5 sigma, grow at 3 sigma (hysteresis) on the matched-filter SNR map; weighted moments |
| Classify | `src/classify.py` | Elongation + length vs. the frame's own stars + evenness of the axial profile; watershed for blends |
| Tile + export | `src/tiling.py`, `src/export_yolo.py` | Last row/column shifted inward (no padding); contours traced at 8x for pixel-edge polygons |
| Repatch | `src/repatch.py` | Rebuilds full frames from the exported tiles + labels only (end-to-end check) |
| QA | `src/qa_checks.py`, `src/validate_injection.py` | Counts, shape histograms, collinear-dot scan, synthetic source recovery |

Every threshold lives in [`configs/params.yaml`](configs/params.yaml) with the measurement that justifies it.

## Setup

Python 3.12 (tested on Windows 11).

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Place the 10 raw frames in `Datasets_Assessment/` (not part of the repository).

## Run

```bash
python run_all.py                  # full pipeline, ~16 min, progress printed per stage
python run_all.py --from classify  # resume from a stage
python run_all.py --skip-injection # skip the ~1.5 min injection test
python package_submission.py       # build the submission archives in outputs/submission/
```

## Outputs (`outputs/`)

| Folder | Content |
|---|---|
| `inspection/` | per-frame statistics, previews, histograms |
| `processed/` | 8-bit display frames, pre-processing stats, raw vs processed figures |
| `detections/`, `classified/` | label maps, per-object feature tables, review sheets |
| `yolo_dataset/` | `images/`, `labels/` (train/val split by frame), `data.yaml`, `tiles.csv` |
| `repatched/` | full-size overlays (stars blue, streaks red), class masks, previews, checks |
| `qa/` | injection recall curve, counts, shape histograms, collinear-dot review |

## Main assumptions

- All 10 files (8 UUID-named, 2 `CAM_B_*`) are in scope; the `CAM_B` frames are a different capture pipeline
  (~40x higher background) and are handled by the same noise-normalised rules.
- Stars are unresolved point sources. In a 0.2 s static exposure they trail ~1.5 px (measured elongation 1.10-1.16),
  so any clearly elongated source is a moving object. Slow (e.g. GEO) objects look like stars in a single frame and
  are labelled star.
- The reference image and its annotation in the brief were used only for visual comparison, never as training data.
- Objects are annotated on the full frame before tiling; an object crossing a tile edge is clipped, and pieces
  smaller than 3 px are dropped.

# Star and streak annotation for SSA images

This is my solution for the Digantara AI/ML Data Annotation Intern assessment.

It takes raw telescope frames (FITS, 9568 x 6380, 16-bit) and turns them into a labelled dataset
for YOLO segmentation. Stars (blobs) are class 0 and satellites/debris (streaks) are class 1.
Everything is done by the scripts, nothing is labelled by hand.

## How to run

You need Python 3.12 (3.10+ should also work).

1. Clone the repo and set up a virtual environment:

   ```bash
   git clone https://github.com/TusharVaibhavK/DigantraAssessment-TusharVaibhavK.git
   cd DigantraAssessment-TusharVaibhavK
   python -m venv .venv
   .venv\Scripts\activate          # on Linux/Mac: source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Put the 10 FITS files in a folder called `Datasets_Assessment/` inside the repo
   (the data is not in the repo).

3. Run the whole pipeline:

   ```bash
   python run_all.py
   ```

   It takes around 15 minutes and prints progress for every stage. Everything it makes goes
   into `outputs/`. If you only want to redo part of it, use `python run_all.py --from classify`
   (or any other stage name), and `--skip-injection` skips the last test.

The YOLO dataset ends up in `outputs/yolo_dataset/` (images/train, images/test, labels/train,
labels/test and data.yaml) and can be loaded straight into Ultralytics.

## Approach

1. **Look at the data first.** All 10 frames are the same size, but the noise is very different
   between them (about 0.6 counts in the darkest ones, about 37 in the CAM_B ones). So nothing uses
   fixed numbers - every threshold is "x times the noise" of that frame.
2. **Clean each frame.** Remove the sky background, measure the noise, fix hot pixels (single pixels
   that are sharper than any real star can be), and smooth with a blur the size of a star so faint
   objects stand out.
3. **Find and outline objects.** Anything at least 5 times above the noise is a detection, and its
   mask is grown out to 3 times the noise to get the exact pixels.
4. **Star or streak.** Stars barely move in a 0.2 s exposure, so they are round. An object is a
   streak if it's clearly longer than the normal stars in that frame and evenly bright along its
   length. Long objects with a dip in the middle are two stars touching, and get split.
5. **Tile.** Each full frame is cut into 70 tiles of exactly 1024 x 1024. The last column and row
   are moved back so they end on the image edge, so there's no padding and nothing is lost.
   Annotation is done on the full frame before tiling, so objects on tile borders stay consistent.
6. **Export and check.** Masks are saved as YOLO polygons. Then the full frames are rebuilt from the
   tiles and label files to make sure they match the original, and fake stars/streaks are added
   into real frames to measure how faint the pipeline can go.

All the thresholds are in `configs/params.yaml`, each with a note on where the number came from.

## Results

| | |
|---|---|
| Tiles | 700 (70 per frame), all exactly 1024 x 1024 |
| Split | 8 frames train (560 tiles), 2 frames test (140 tiles) |
| Stars | about 80,500 (97,340 polygons after tiling) |
| Streaks | 62, all checked by eye (77 polygons after tiling) |
| Rebuild check | all 10 frames rebuilt from tiles + labels match the original exactly |
| Faint stars | fake stars 3x brighter than the noise found 100% of the time, 2x about 50-78% |
| Streaks | fake streaks 30 px or longer found and labelled right 90%+ of the time from 1.5x the noise |

## What's in the code

| File | What it does |
|---|---|
| `run_all.py` | runs every stage in order |
| `src/inspect_fits.py` | stats and previews of the raw frames |
| `src/preprocess.py` | background, noise, hot pixels, smoothing, display stretch |
| `src/detect.py` | finds objects and makes the pixel masks |
| `src/classify.py` | decides star or streak, splits touching stars |
| `src/tiling.py`, `src/export_yolo.py` | 1024 x 1024 tiles and YOLO label files |
| `src/repatch.py` | rebuilds the full frames from tiles + labels, with masks drawn on |
| `src/qa_checks.py`, `src/validate_injection.py` | sanity checks and the fake star/streak test |
| `configs/params.yaml` | all thresholds |

Optional: `app/app.py` is a small Streamlit app that runs the pipeline step by step on a crop so you
can see what each step does (`pip install -r requirements-app.txt`, then `streamlit run app/app.py`).

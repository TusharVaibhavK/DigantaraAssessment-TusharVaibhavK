"""Build the two submission archives from ./outputs.

    python package_submission.py

  outputs/submission/Digantara_Annotation_Submission.zip   compact (~40 MB): upload to the portal
  outputs/submission/Digantara_Annotation_Full.zip         everything (~900 MB): share via a drive link

Both archives share one layout; the full one adds the 700 tile images and the full-size overlays.
A write-up PDF placed in ./writeup/ is included automatically.
"""
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SUB = OUT / "submission"
STORED = {".png", ".jpg", ".jpeg", ".pdf", ".zip"}     # already compressed


def compact_files():
    """(archive path, source path) pairs shared by both archives."""
    ds, rep, qa = OUT / "yolo_dataset", OUT / "repatched", OUT / "qa"
    files = [("README.md", ROOT / "docs" / "SUBMISSION_README.md")]
    files += [(f"writeup/{p.name}", p) for p in sorted((ROOT / "writeup").glob("*.pdf"))]
    files += [("annotations/tiles.csv", ds / "tiles.csv")]
    files += [(f"annotations/{p.relative_to(ds).as_posix()}", p) for p in sorted((ds / "labels").rglob("*.txt"))]
    files += [(f"full_size/classmasks/{p.name}", p) for p in sorted(rep.glob("*_classmask.png"))]
    files += [(f"full_size/previews/{p.name}", p) for p in sorted(rep.glob("*_preview.jpg"))]
    files += [("qa/repatch_checks.csv", rep / "checks.csv")]
    files += [(f"qa/{p.name}", p) for p in sorted(qa.glob("*")) if not p.name.startswith("_")]
    files += [(f"figures/preprocessing/{p.name}", p) for p in sorted((OUT / "processed").glob("*_steps.png"))]
    files += [("figures/preprocessing/preprocess.csv", OUT / "processed" / "preprocess.csv"),
              ("figures/inspection/stats.csv", OUT / "inspection" / "stats.csv")]
    files += [(f"figures/inspection/{p.name}", p) for p in sorted((OUT / "inspection").glob("*_hist.png"))]
    files += [("figures/classification/streaks_review.png", OUT / "classified" / "streaks_review.png"),
              ("figures/classification/grey_zone_review.png", OUT / "classified" / "grey_zone_review.png"),
              ("figures/classification/summary.csv", OUT / "classified" / "summary.csv"),
              ("figures/detection/shape_scatter.png", OUT / "detections" / "shape_scatter.png")]
    return files


def full_extras():
    ds, rep = OUT / "yolo_dataset", OUT / "repatched"
    files = [(f"annotations/{p.relative_to(ds).as_posix()}", p) for p in sorted((ds / "images").rglob("*.png"))]
    files += [(f"full_size/overlays/{p.name}", p) for p in sorted(rep.glob("*_overlay.jpg"))]
    return files


def generated():
    """Small files written straight into the archive."""
    data_yaml = yaml.safe_dump({"train": "images/train", "val": "images/val",
                                "names": {0: "star", 1: "streak"}}, sort_keys=False)
    return {"annotations/data.yaml": data_yaml, "annotations/classes.txt": "star\nstreak\n"}


def write_zip(path, files):
    missing = [str(src) for _, src in files if not src.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs (run run_all.py first): {missing[:5]}")
    with zipfile.ZipFile(path, "w") as zf:
        for arc, text in generated().items():
            zf.writestr(arc, text, compress_type=zipfile.ZIP_DEFLATED)
        for arc, src in files:
            method = zipfile.ZIP_STORED if src.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
            zf.write(src, arc, compress_type=method)
    print(f"{path.name}: {len(files) + len(generated())} files, {path.stat().st_size / 1e6:.0f} MB")


def main():
    SUB.mkdir(parents=True, exist_ok=True)
    if not list((ROOT / "writeup").glob("*.pdf")):
        print("note: no PDF in ./writeup yet, archives will not contain the written answer")
    compact = compact_files()
    write_zip(SUB / "Digantara_Annotation_Submission.zip", compact)
    write_zip(SUB / "Digantara_Annotation_Full.zip", compact + full_extras())


if __name__ == "__main__":
    main()

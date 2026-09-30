"""Run the whole annotation pipeline end to end.

    python run_all.py                      # every stage
    python run_all.py --from classify      # resume from a stage (earlier outputs must exist)
    python run_all.py --skip-injection     # skip the ~1.5 min injection-recovery test

Stages: inspect -> detect (incl. pre-processing) -> classify -> export -> repatch -> qa -> injection
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import classify        # noqa: E402
import detect          # noqa: E402
import export_yolo     # noqa: E402
import inspect_fits    # noqa: E402
import qa_checks       # noqa: E402
import repatch         # noqa: E402
import validate_injection  # noqa: E402

STAGES = [
    ("inspect", "A1  inspect raw FITS frames", inspect_fits.main),
    ("detect", "A1/A3  pre-process, detect and segment", detect.main),
    ("classify", "A3  classify star / streak", classify.main),
    ("export", "A2/A3  tile to 1024x1024 + YOLO-seg labels", export_yolo.main),
    ("repatch", "rebuild full-size frames from tiles + labels", repatch.main),
    ("qa", "automated QA checks", qa_checks.main),
    ("injection", "injection-recovery test", validate_injection.main),
]


def main():
    names = [s[0] for s in STAGES]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", choices=names, default=names[0])
    ap.add_argument("--skip-injection", action="store_true")
    args = ap.parse_args()

    todo = STAGES[names.index(args.start):]
    if args.skip_injection:
        todo = [s for s in todo if s[0] != "injection"]
    t_all = time.time()
    for i, (key, title, fn) in enumerate(todo, 1):
        print(f"\n=== [{i}/{len(todo)}] {key}: {title} ===", flush=True)
        t0 = time.time()
        fn()
        print(f"--- {key} done in {time.time() - t0:.0f} s", flush=True)
    print(f"\nPipeline finished in {(time.time() - t_all) / 60:.1f} min. Outputs are in ./outputs")


if __name__ == "__main__":
    main()

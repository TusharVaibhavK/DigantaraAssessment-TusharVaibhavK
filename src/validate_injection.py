"""QA 1: injection-recovery test (completeness of detection + classification).

Synthetic stars (Gaussian PSF, sigma = measured PSF) and streaks (the same PSF swept along a line)
are added to a crop of a real frame, in raw counts and rounded to integers like the real data.
Brightness is given in units of that frame's per-pixel noise, so the three intensity groups are
directly comparable. The unchanged pipeline (preprocess -> detect -> classify) is then run.

  star recovered   : the injected centre falls inside an object classified as star
  streak recovered : the injected midpoint falls inside an object classified as streak
Amplitude 0 injections measure how often an empty position lands on a real object by chance.

Output: outputs/qa/injection_results.csv, outputs/qa/injection_recall.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from classify import STAR, STREAK, classify_frame
from detect import detect_frame
from fits_io import OUT_DIR, RAW_DIR, load_frame
from preprocess import background_and_noise, load_params

QA_DIR = OUT_DIR / "qa"
FRAMES = ["1a600998", "3e6beceb", "CAM_B_20260815T153133"]   # one per intensity group
CROP = 4096
GRID = 96                    # spacing between injection sites (px)
PSF_SIGMA = 1.53
STAR_AMPS = [0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]    # peak, in per-pixel noise sigma
STREAK_AMPS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]          # ridge peak, in per-pixel noise sigma
STREAK_LENGTHS = [15, 30, 60]
N_STAR, N_STREAK = 50, 12    # injections per (amplitude) and per (amplitude, length)


def add_psf_line(img, y, x, amp, length=0.0, angle=0.0):
    """Add a PSF (length 0) or a streak whose cross-section peak equals `amp`."""
    r = int(3 * PSF_SIGMA + length / 2 + 2)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    if length == 0:
        prof = np.exp(-(yy ** 2 + xx ** 2) / (2 * PSF_SIGMA ** 2))
    else:
        dy, dx = np.sin(angle), np.cos(angle)
        t = np.clip(yy * dy + xx * dx, -length / 2, length / 2)     # nearest point on the segment
        d2 = (yy - t * dy) ** 2 + (xx - t * dx) ** 2
        prof = np.exp(-d2 / (2 * PSF_SIGMA ** 2))
    img[y - r:y + r + 1, x - r:x + r + 1] += amp * prof


def build_plan(rng):
    plan = [("star", a, 0) for a in STAR_AMPS for _ in range(N_STAR)]
    plan += [("streak", a, L) for a in STREAK_AMPS for L in STREAK_LENGTHS for _ in range(N_STREAK)]
    sites = [(y, x) for y in range(GRID, CROP - GRID, GRID) for x in range(GRID, CROP - GRID, GRID)]
    order = rng.permutation(len(sites))[:len(plan)]
    rows = []
    for (kind, amp, L), k in zip(plan, order):
        y, x = sites[k]
        rows.append({"kind": kind, "amp": amp, "length": L, "y": y, "x": x,
                     "angle": float(rng.uniform(0, np.pi))})
    return pd.DataFrame(rows)


def run_frame(tag, params, rng):
    path = next(RAW_DIR.glob(tag + "*.fits"))
    raw, _ = load_frame(path)
    cy, cx = raw.shape[0] // 2, raw.shape[1] // 2
    crop = raw[cy - CROP // 2:cy + CROP // 2, cx - CROP // 2:cx + CROP // 2].astype(np.float64)
    _, sigma_map = background_and_noise(crop.astype(np.float32), params["preprocess"])
    sigma = float(np.median(sigma_map))

    plan = build_plan(rng)
    injected = crop.copy()
    for r in plan.itertuples():
        if r.amp > 0:
            add_psf_line(injected, r.y, r.x, r.amp * sigma, r.length, r.angle)
    top = 65535 if raw.max() > 4095 else 4095
    injected = np.clip(np.round(injected), 0, top).astype(np.float32)

    _, labels, objs, _ = detect_frame(injected, params)
    labels, objs, _ = classify_frame(injected, labels, objs, params)
    cls_of = dict(zip(objs.label.astype(int), objs.cls.astype(int)))
    hit = labels[plan.y.to_numpy(), plan.x.to_numpy()].astype(int)
    plan["detected"] = hit > 0
    plan["cls"] = [cls_of.get(h, -1) if h else -1 for h in hit]
    want = np.where(plan.kind == "star", STAR, STREAK)
    plan["correct"] = plan.detected & (plan.cls == want)
    plan["frame"] = tag
    plan["sigma_pix"] = sigma
    return plan


def plot(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    s = res[res.kind == "star"]
    for tag, g in s.groupby("frame"):
        r = g.groupby("amp").correct.mean()
        ax.plot(r.index, r.values, "o-", label=tag[:8])
    ax.axvline(1.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("injected star peak (x per-pixel noise sigma)")
    ax.set_ylabel("fraction recovered as star")
    ax.set_title("Stars")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.02)

    ax = axes[1]
    k = res[res.kind == "streak"]
    for L, g in k.groupby("length"):
        r = g.groupby("amp").correct.mean()
        d = g.groupby("amp").detected.mean()
        line, = ax.plot(r.index, r.values, "o-", label=f"L={L}px classified streak")
        ax.plot(d.index, d.values, "--", color=line.get_color(), alpha=0.5, label=f"L={L}px detected")
    ax.set_xlabel("injected streak ridge peak (x per-pixel noise sigma)")
    ax.set_ylabel("fraction")
    ax.set_title("Streaks (all 3 frames pooled)")
    ax.legend(fontsize=7)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    QA_DIR.mkdir(parents=True, exist_ok=True)
    params = load_params()
    rng = np.random.default_rng(42)
    results = [run_frame(tag, params, rng) for tag in tqdm(FRAMES, desc="injection frames", file=sys.stdout)]
    res = pd.concat(results, ignore_index=True)
    res.to_csv(QA_DIR / "injection_results.csv", index=False)
    plot(res, QA_DIR / "injection_recall.png")

    stars = res[res.kind == "star"].pivot_table(index="amp", columns="frame", values="correct", aggfunc="mean")
    print("\nSTARS: fraction recovered as star, by injected peak (sigma)\n", stars.round(2).to_string())
    streaks = res[res.kind == "streak"].pivot_table(index="amp", columns="length",
                                                    values=["detected", "correct"], aggfunc="mean")
    print("\nSTREAKS (3 frames pooled): detected / classified as streak\n", streaks.round(2).to_string())
    wrong = res[(res.kind == "streak") & res.detected & ~res.correct]
    print(f"\nstreaks detected but labelled star: {len(wrong)} (by length: {wrong.length.value_counts().to_dict()})")


if __name__ == "__main__":
    main()

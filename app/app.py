"""SSA annotation pipeline explorer.

    pip install -r requirements-app.txt
    streamlit run app/app.py

Runs the real pipeline functions from src/ on a crop of a real frame, step by step, and shows
every intermediate image with an explanation, a step slider, a cross-fade transition slider and
an animated walk-through.
"""
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from classify import STAR, STREAK, classify_frame          # noqa: E402
from detect import hysteresis_labels, measure              # noqa: E402
from export_yolo import mask_to_polygons                   # noqa: E402
from fits_io import OUT_DIR, list_frames, load_frame, short_name   # noqa: E402
from preprocess import (background_and_noise, load_params, measure_psf,   # noqa: E402
                        remove_hot_pixels, saturation_level, to_display)
from tiling import tile_grid                               # noqa: E402

VIEW = 720                               # displayed image size (px)
BLUE, RED, YELLOW, CYAN, ORANGE = (80, 140, 255), (235, 60, 60), (255, 210, 60), (70, 220, 240), (255, 150, 30)

st.set_page_config(page_title="SSA Pipeline Explorer", layout="wide")


# ============================================================== data

@st.cache_resource(max_entries=2, show_spinner="Reading FITS frame (122 MB)...")
def frame_data(path_str):
    raw, header = load_frame(path_str)
    return raw, dict(header)


@st.cache_data(show_spinner=False)
def targets(name):
    """Interesting crop centres from the pipeline outputs, if they exist."""
    p = OUT_DIR / "classified" / f"{name}_objects.csv"
    if not p.exists():
        return {}
    o = pd.read_csv(p)
    out = {}
    s = o[o.cls == STREAK].sort_values("peak_snr", ascending=False)
    if len(s):
        out["Brightest streak"] = (float(s.cy.iat[0]), float(s.cx.iat[0]))
    if len(s) > 1:
        out["Faintest streak"] = (float(s.cy.iat[-1]), float(s.cx.iat[-1]))
    b = o[o.reason == "blend_split"]
    if len(b):
        out["Blended star pair"] = (float(b.cy.iat[0]), float(b.cx.iat[0]))
    return out


@st.cache_data(show_spinner=False)
def frame_reference(name):
    """Full-frame PSF FWHM and typical star length from the pipeline outputs (None if not run yet).
    A crop holds too few bright stars to measure these reliably, the full pipeline uses whole frames."""
    fwhm = L_star = None
    p = OUT_DIR / "processed" / "preprocess.csv"
    if p.exists():
        r = pd.read_csv(p).set_index("name")
        if name in r.index:
            fwhm = float(r.loc[name, "psf_fwhm_px"])
    p = OUT_DIR / "classified" / "summary.csv"
    if p.exists():
        r = pd.read_csv(p).set_index("frame")
        if name in r.index:
            L_star = float(r.loc[name, "L_star"])
    return fwhm, L_star


@st.cache_data(show_spinner="Building frame overview...")
def frame_overview(name, shape):
    p = OUT_DIR / "repatched" / f"{name}_preview.jpg"
    if p.exists():
        return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    raw, _ = frame_data(str(next(f for f in list_frames() if short_name(f) == name)))
    small = cv2.resize(raw, (shape[1] // 4, shape[0] // 4), interpolation=cv2.INTER_AREA)
    lo, hi = np.percentile(small, [1, 99.7])
    g = (np.clip((small - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


# ============================================================== rendering helpers

def fit(img, interp=None):
    """Resize any 2-D/3-D image to VIEW x VIEW (nearest when enlarging so pixels stay visible)."""
    h = img.shape[0]
    if interp is None:
        interp = cv2.INTER_NEAREST if VIEW >= h else cv2.INTER_AREA
    return cv2.resize(img, (VIEW, VIEW), interpolation=interp)


def gray(arr, lo, hi):
    g = (np.clip((arr - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(fit(g), cv2.COLOR_GRAY2RGB)


def cmap(arr, name, lo=None, hi=None):
    lo = np.min(arr) if lo is None else lo
    hi = np.max(arr) if hi is None else hi
    norm = np.clip((arr - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = (matplotlib.colormaps[name](norm)[..., :3] * 255).astype(np.uint8)
    return fit(rgb, cv2.INTER_LINEAR)


def draw_contours(rgb, mask, colour, thickness=1):
    m = fit(mask.astype(np.uint8), cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, cnts, -1, colour, thickness)
    return rgb


def fill(rgb, mask, colour, alpha=0.55):
    m = fit(mask.astype(np.uint8), cv2.INTER_NEAREST).astype(bool)
    rgb[m] = (rgb[m] * (1 - alpha) + np.array(colour) * alpha).astype(np.uint8)
    return rgb


# ============================================================== the pipeline, one step at a time

def run_pipeline(raw_crop, params, on_step, ref_fwhm=None, ref_L_star=None):
    """Execute the pipeline on a crop; call on_step(step_dict) as soon as each step is ready.
    ref_fwhm / ref_L_star: full-frame values to use instead of measuring them on the small crop."""
    p, pd_ = params["preprocess"], params["detect"]
    n = raw_crop.shape[0]
    scale = VIEW / n

    # 1 raw
    lo, hi = np.percentile(raw_crop, [1, 99.5])
    on_step(dict(key="raw", title="Raw sensor data",
                 img=gray(raw_crop, lo, hi),
                 text="The FITS pixels as recorded (uint16 counts), shown with a plain linear stretch. "
                      "Most pixels hold only 0-3 counts, so stars barely rise above the grainy noise, and "
                      "the sky level is not flat.",
                 metrics={"min": f"{raw_crop.min():.0f}", "median": f"{np.median(raw_crop):.0f}",
                          "max": f"{raw_crop.max():.0f}"}))

    # 2 background model
    bg, sigma = background_and_noise(raw_crop, p)
    on_step(dict(key="bg", title="Background model",
                 img=cmap(bg, "viridis"),
                 text="The sky glow is estimated on a 128 px grid: in each cell a sigma-clipped mean ignores the "
                      "stars, a 3x3 median smooths the grid and it is upsampled. This is what gets subtracted.",
                 metrics={"background min": f"{bg.min():.2f}", "background max": f"{bg.max():.2f}"}))

    # 3 subtracted
    resid = raw_crop - bg
    on_step(dict(key="sub", title="Background subtracted",
                 img=gray(resid / sigma, -2, 6),
                 text="After subtraction the sky sits at zero everywhere, so the same threshold means the same "
                      "thing across the frame. Shown in units of the noise.",
                 metrics={"residual median": f"{np.median(resid):.3f}"}))

    # 4 noise map
    on_step(dict(key="noise", title="Noise map (sigma)",
                 img=cmap(sigma, "magma"),
                 text="The per-pixel noise, from a sigma-clipped standard deviation on the same grid. The usual "
                      "MAD estimate returns 0 on these integer, mostly-zero data; this estimator does not. "
                      "Every threshold that follows is 'k x sigma'.",
                 metrics={"sigma median": f"{np.median(sigma):.3f}", "sigma range": f"{sigma.min():.2f}-{sigma.max():.2f}"}))

    # 5 hot pixels
    cleaned, hot = remove_hot_pixels(resid, sigma, p)
    img = gray(resid / sigma, -1, 12)
    for y, x in zip(*np.nonzero(hot)):
        cv2.circle(img, (int((x + 0.5) * scale), int((y + 0.5) * scale)), max(4, int(scale * 2)), RED, 1)
    on_step(dict(key="hot", title="Hot / warm pixels found",
                 img=img,
                 text="A pixel is a defect if it is > 5 sigma above its 3x3 median while that median is < half "
                      "of it. Optics spread every real star over several pixels (FWHM ~3.7 px), so a star's "
                      "neighbours are ~0.8x its peak and can never trigger this. Circled: pixels to be replaced.",
                 metrics={"defective pixels": f"{int(hot.sum())}", "share": f"{100 * hot.mean():.3f} %"}))

    # 6 cleaned display (what the tiles look like)
    snr = cleaned / sigma
    disp = to_display(snr, p)
    on_step(dict(key="clean", title="Cleaned display image",
                 img=gray(disp, 0, 255),
                 text="Defects replaced by the local median, then an asinh stretch from -1 to 30 sigma to 8 bit. "
                      "Because it is in noise units, every frame looks alike. This image is what the 1024 x 1024 "
                      "tiles contain.",
                 metrics={"stretch": "-1 ... 30 sigma, asinh"}))

    # 7 matched filter
    sat = saturation_level(raw_crop)
    fwhm_crop, elong, n_psf = measure_psf(cleaned, sigma, raw_crop, sat, p)
    fwhm = ref_fwhm or fwhm_crop
    smooth = gaussian_filter(cleaned, fwhm / 2.3548)
    _, sigma_s = background_and_noise(smooth, p)
    snr_s = smooth / sigma_s
    on_step(dict(key="mf", title="Matched filter (SNR map)",
                 img=gray(snr_s, -2, 10),
                 text="The image is smoothed with a Gaussian as wide as a star (the measured PSF). This is the "
                      "optimal filter for point sources: noise drops ~5x while stars keep most of their signal. "
                      "The smoothed noise is re-measured, giving a true SNR map.",
                 metrics={"PSF FWHM used": f"{fwhm:.2f} px" + (" (full frame)" if ref_fwhm else ""),
                          "FWHM in this crop": f"{fwhm_crop:.2f} px ({n_psf} stars)"}))

    # 8 seeds
    seeds = snr_s > pd_["seed_sigma"]
    img = gray(snr_s, -2, 10)
    img = fill(img, seeds, ORANGE, 0.8)
    on_step(dict(key="seeds", title="Detection seeds (>= 5 sigma)",
                 img=img,
                 text="Pixels where the matched-filter SNR exceeds 5 sigma. With ~2 million independent noise "
                      "patches per frame, pure noise would give fewer than one such seed per frame.",
                 metrics={"seed pixels": f"{int(seeds.sum())}"}))

    # 9 masks
    pre = {"snr_smooth": snr_s, "snr": snr,
           "saturated": raw_crop >= sat if sat is not None else np.zeros(raw_crop.shape, bool)}
    labels = hysteresis_labels(snr_s, pd_)
    img = gray(disp, 0, 255)
    img = draw_contours(img, labels > 0, CYAN, 1)
    on_step(dict(key="masks", title="Pixel masks (hysteresis 5 / 3 sigma)",
                 img=img,
                 text="Each seed grows through connected pixels down to 3 sigma: a strict threshold decides "
                      "that an object exists, a looser one decides how far it extends. Objects under 5 px are "
                      "dropped.",
                 metrics={"objects": f"{int(labels.max() and len(np.unique(labels)) - 1)}"}))

    # 10 classification
    objs = measure(labels, pre) if labels.max() else pd.DataFrame(columns=["label"])
    if len(objs):
        labels, objs, L_star = classify_frame(raw_crop, labels, objs, params, L_star=ref_L_star)
    else:
        L_star = float("nan")
    cls_of = dict(zip(objs.label.astype(int), objs.cls.astype(int))) if len(objs) else {}
    reason_of = dict(zip(objs.label.astype(int), objs.reason)) if len(objs) else {}
    star_m = np.isin(labels, [l for l, c in cls_of.items() if c == STAR and reason_of[l] != "blend_split"])
    blend_m = np.isin(labels, [l for l, r in reason_of.items() if r == "blend_split"])
    streak_m = np.isin(labels, [l for l, c in cls_of.items() if c == STREAK])
    img = gray(disp, 0, 255)
    img = fill(img, star_m, BLUE)
    img = fill(img, blend_m, YELLOW)
    img = fill(img, streak_m, RED)
    n_streak = sum(c == STREAK for c in cls_of.values())
    on_step(dict(key="cls", title="Classified: star / streak",
                 img=img,
                 text="Blue = star, red = streak, yellow = stars split out of a blend. A streak must be "
                      "elongated (>= 3x), long (>= 2.5x this frame's typical star) and evenly bright along its "
                      "axis; a dip in the middle means two touching stars, which a watershed splits.",
                 metrics={"stars": f"{len(cls_of) - n_streak}", "streaks": f"{n_streak}",
                          "blend pieces": f"{int(sum(r == 'blend_split' for r in reason_of.values()))}",
                          "star length ref": f"{L_star:.1f} px"}))

    # 11 YOLO polygons
    cfg = params["export"]
    img = gray(disp, 0, 255)
    example, n_poly = "", 0
    for lid, c in cls_of.items():
        m = labels == lid
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        y0, x0 = ys.min(), xs.min()
        for poly in mask_to_polygons(m[y0:ys.max() + 1, x0:xs.max() + 1], cfg["upsample"],
                                     cfg["min_piece_px"], cfg["simplify_px"]):
            pts = ((poly + [x0, y0]) * scale).round().astype(np.int32)
            cv2.polylines(img, [pts], True, RED if c == STREAK else (120, 255, 120), 1)
            n_poly += 1
            if c == STREAK or not example:
                xy = (poly + [x0, y0]) / n
                example = f"{c} " + " ".join(f"{v:.4f}" for v in xy.ravel()[:8]) + " ..."
    on_step(dict(key="yolo", title="YOLO segmentation polygons",
                 img=img,
                 text="Each mask becomes a polygon traced along pixel edges (on an 8x upsampled mask, so a 3x3 "
                      "star is not shrunk to 2x2) and written as 'class x1 y1 x2 y2 ...' normalised to the tile. "
                      "Green = star (0), red = streak (1).",
                 metrics={"polygons": f"{n_poly}"},
                 code=example))


# ============================================================== UI

frames = {short_name(f): f for f in list_frames()}
if not frames:
    st.error("No FITS files found in Datasets_Assessment/.")
    st.stop()
params = load_params()

with st.sidebar:
    st.header("Input")
    name = st.selectbox("Frame", list(frames))
    raw, header = frame_data(str(frames[name]))
    H, W = raw.shape
    size = st.select_slider("Crop size (px)", [256, 384, 512, 768, 1024], value=512)
    options = list(targets(name)) + ["Frame centre", "Custom position"]
    where = st.radio("Crop on", options, key="where")
    if where == "Custom position":
        cx = st.slider("x centre", size // 2, W - size // 2, W // 2, step=16)
        cy = st.slider("y centre", size // 2, H - size // 2, H // 2, step=16)
    elif where == "Frame centre":
        cy, cx = H // 2, W // 2
    else:
        cy, cx = targets(name)[where]
    y0 = int(np.clip(cy - size // 2, 0, H - size))
    x0 = int(np.clip(cx - size // 2, 0, W - size))
    st.caption(f"Crop origin x={x0}, y={y0}  ·  {header.get('INSTRUME', '')} {header.get('SERIAL', '')}")
    st.header("Playback")
    slow = st.toggle("Step-through run (pause after each step)", value=True)
    delay = st.slider("Pause per step (s)", 0.2, 3.0, 0.8, 0.1, disabled=not slow)
    run = st.button("Run pipeline", type="primary", width="stretch")

st.title("SSA star / streak annotation: pipeline explorer")
tab_run, tab_frame = st.tabs(["Pipeline explorer", "Frame & tiles"])

with tab_frame:
    ov = frame_overview(name, raw.shape).copy()
    sx, sy = ov.shape[1] / W, ov.shape[0] / H
    for _, _, ty, tx in tile_grid(H, W):
        cv2.rectangle(ov, (int(tx * sx), int(ty * sy)), (int((tx + 1024) * sx), int((ty + 1024) * sy)), CYAN, 1)
    cv2.rectangle(ov, (int(x0 * sx), int(y0 * sy)), (int((x0 + size) * sx), int((y0 + size) * sy)), YELLOW, 3)
    st.image(ov, caption=f"{name}: 70 tiles of 1024 x 1024 (cyan; last column/row shifted inward to overlap) "
                         f"and the selected crop (yellow).", width="stretch")

with tab_run:
    key = (name, size, y0, x0)
    if run:
        steps = []
        status_col, image_col = st.columns([1, 2])
        with status_col:
            bar = st.progress(0.0, text="Starting...")
            log = st.empty()
        live = image_col.empty()
        n_steps = 11
        t_start = time.time()

        def on_step(step):
            steps.append(step)
            bar.progress(len(steps) / n_steps, text=f"Step {len(steps)}/{n_steps}: {step['title']}")
            log.markdown("\n".join(f"{'✅' if i < len(steps) else '⏳'} {i + 1}. {s}"
                                   for i, s in enumerate([s["title"] for s in steps] +
                                                         ["..."] * (n_steps - len(steps)))))
            live.image(step["img"], caption=f"{len(steps)}. {step['title']}", width=VIEW)
            if slow:
                time.sleep(delay)

        crop = raw[y0:y0 + size, x0:x0 + size].astype(np.float32)
        ref_fwhm, ref_L = frame_reference(name)
        run_pipeline(crop, params, on_step, ref_fwhm, ref_L)
        bar.progress(1.0, text=f"Done in {time.time() - t_start:.1f} s")
        st.session_state["result"] = {"key": key, "steps": steps}
        st.rerun()

    res = st.session_state.get("result")
    if not res:
        st.info("Choose a frame and crop in the sidebar, then press **Run pipeline**.")
        st.stop()
    if res["key"] != key:
        st.warning("The crop changed since the last run. Press **Run pipeline** to update.")
    steps = res["steps"]
    titles = [f"{i + 1}. {s['title']}" for i, s in enumerate(steps)]

    mode = st.radio("View", ["Step by step", "Transition slider"], horizontal=True, key="view")
    left, right = st.columns([3, 2])
    if mode == "Step by step":
        choice = left.select_slider("Step", titles, value=titles[0], key="step")
        i = titles.index(choice)
        left.image(steps[i]["img"], width=VIEW)
    else:
        t = left.slider("Position (drag to morph between steps)", 0.0, float(len(steps) - 1), 0.0, 0.05, key="morph")
        play = left.button("▶ Play all steps", key="play")
        frame_slot = left.empty()

        def blended(pos):
            a = int(np.floor(pos))
            b = min(a + 1, len(steps) - 1)
            w = pos - a
            return (steps[a]["img"] * (1 - w) + steps[b]["img"] * w).astype(np.uint8), a, b, w

        if play:
            for pos in np.arange(0, len(steps) - 1 + 1e-6, 0.05):
                img, a, b, w = blended(pos)
                frame_slot.image(img, caption=f"{titles[a]}  →  {titles[b]}  ({w:.0%})", width=VIEW)
                time.sleep(0.12 if abs(w) < 1e-6 or abs(w - 1) < 1e-6 else 0.04)
            i = len(steps) - 1
        else:
            img, a, b, w = blended(t)
            frame_slot.image(img, caption=f"{titles[a]}  →  {titles[b]}  ({w:.0%})", width=VIEW)
            i = int(round(t))

    with right:
        s = steps[i]
        st.subheader(titles[i])
        st.write(s["text"])
        cols = st.columns(min(len(s["metrics"]), 3) or 1)
        for k, (label, value) in enumerate(s["metrics"].items()):
            cols[k % len(cols)].metric(label, value)
        if s.get("code"):
            st.caption("Example label line (normalised to this crop):")
            st.code(s["code"], language=None)
        st.divider()
        st.caption("All steps")
        st.markdown("\n".join(f"{'**' if j == i else ''}{t}{'**' if j == i else ''}" for j, t in enumerate(titles)))

"""Shared helpers for reading the FITS frames."""
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "Datasets_Assessment"
OUT_DIR = ROOT / "outputs"


def list_frames(raw_dir=RAW_DIR):
    return sorted(Path(raw_dir).glob("*.fits"))


def short_name(path):
    """UUID names are long; the first block is unique enough for display."""
    stem = Path(path).stem
    return stem if stem.startswith("CAM_") else stem.split("-")[0]


def load_frame(path):
    """Return (image as float32, header). astropy applies BZERO/BSCALE, so values are true uint16 counts."""
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header.copy()
    return data, header

# src/tables.py
from __future__ import annotations

import numpy as np
import soundfile as sf

from .dsp_core import sanitize_cycle, spectral_morph


def make_table_from_frames(frames: list[np.ndarray], *, peak: float = 0.99) -> np.ndarray:
    """
    Takes list of single-cycle frames (shape [N]) and returns flattened table.
    """
    if not frames:
        raise ValueError("No frames provided.")
    N = frames[0].shape[0]
    for f in frames:
        if f.shape[0] != N:
            raise ValueError("All frames must have same length.")
    cleaned = [sanitize_cycle(f, peak=peak) for f in frames]
    return np.concatenate(cleaned, axis=0)


def make_morph_table(
    a: np.ndarray,
    b: np.ndarray,
    *,
    frames: int = 256,
    peak: float = 0.99,
    phase_source: str = "a",
    mag_curve: str = "linear",
) -> np.ndarray:
    """
    Build a wavetable by spectrally morphing between a and b.
    """
    a = sanitize_cycle(a, peak=peak)
    b = sanitize_cycle(b, peak=peak)

    out = []
    for i in range(frames):
        m = i / (frames - 1) if frames > 1 else 1.0
        x = spectral_morph(a, b, m, phase_source=phase_source, mag_curve=mag_curve)
        x = sanitize_cycle(x, peak=peak)
        out.append(x)

    return make_table_from_frames(out, peak=peak)


def write_wavetable_wav(
    path: str,
    table: np.ndarray,
    *,
    sample_rate: int = 44100,
    subtype: str = "FLOAT",
) -> None:
    """
    Writes the flattened wavetable as a mono WAV.
    """
    t = np.asarray(table, dtype=np.float32)
    sf.write(path, t, sample_rate, subtype=subtype)

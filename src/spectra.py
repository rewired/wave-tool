from __future__ import annotations

import numpy as np

from . import dsp_core


def _apply_taper(amps: np.ndarray, taper: str) -> np.ndarray:
    """Apply optional harmonic taper to reduce Gibbs ringing."""
    taper = taper.lower()
    if taper == "none":
        return amps
    if taper in {"lanczos", "sinc"}:
        window = dsp_core.lanczos_taper(len(amps) - 1)
        return amps * window
    raise ValueError(f"Unknown taper: {taper}")


def harmonic_spectrum(name: str, max_harm: int, *, taper: str = "lanczos") -> np.ndarray:
    """Return harmonic amplitude array for a canonical spectrum."""
    name = name.lower()
    max_harm = int(max_harm)
    if max_harm < 1:
        raise ValueError("max_harm must be >= 1")

    amps = np.zeros(max_harm + 1, dtype=np.float64)
    k = np.arange(1, max_harm + 1, dtype=np.float64)

    if name == "sine":
        amps[1] = 1.0
    elif name == "saw":
        amps[1:] = 1.0 / k
    elif name == "square":
        amps[1:] = (1.0 / k) * (k % 2)
    elif name == "triangle":
        odd = (k % 2) == 1
        idx = k[odd]
        signs = (-1.0) ** ((idx - 1) / 2)
        amps[1:][odd] = signs / (idx**2)
    else:
        raise ValueError(f"Unknown spectrum name: {name}")

    return _apply_taper(amps, taper)

# src/dsp_core.py
from __future__ import annotations

import numpy as np

EPS = 1e-12


# -----------------------------
# Basic signal hygiene
# -----------------------------
def remove_dc(x: np.ndarray) -> np.ndarray:
    """Remove DC offset."""
    x = np.asarray(x, dtype=np.float64)
    return x - np.mean(x)


def normalize_peak(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Normalize to a target peak amplitude."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(np.abs(x))
    if m < EPS:
        return x.copy()
    return x * (peak / m)


def sanitize_cycle(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Standard cleanup for a single-cycle frame."""
    x = remove_dc(x)
    x = normalize_peak(x, peak=peak)
    return x


# -----------------------------
# FFT helpers
# -----------------------------
def rfft(x: np.ndarray) -> np.ndarray:
    """Real FFT."""
    x = np.asarray(x, dtype=np.float64)
    return np.fft.rfft(x)


def irfft(X: np.ndarray, n: int) -> np.ndarray:
    """Inverse real FFT to time domain of length n."""
    return np.fft.irfft(X, n=n)


def mag_phase(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return magnitude and phase (angle)."""
    return np.abs(X), np.angle(X)


def polar(mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Construct complex spectrum from magnitude and phase."""
    return mag * np.exp(1j * phase)


# -----------------------------
# Spectral shaping
# -----------------------------
def harmonic_mask(n_bins: int, max_harm: int) -> np.ndarray:
    """
    Mask that keeps bins 0..max_harm (including DC) and zeros the rest.
    n_bins is len(rfft(frame)) i.e. N/2+1.
    """
    max_harm = int(max_harm)
    m = np.zeros(n_bins, dtype=np.float64)
    hi = min(max_harm, n_bins - 1)
    m[: hi + 1] = 1.0
    return m


def apply_harmonic_limit(frame: np.ndarray, max_harm: int) -> np.ndarray:
    """
    Hard harmonic culling in the spectral domain.
    max_harm counts FFT bins (harmonic numbers for a single-cycle).
    """
    x = np.asarray(frame, dtype=np.float64)
    N = x.shape[0]
    X = rfft(x)
    m = harmonic_mask(len(X), max_harm=max_harm)
    X2 = X * m
    y = irfft(X2, n=N)
    return y


# -----------------------------
# Morphing (the critical part)
# -----------------------------
def spectral_morph(
    a: np.ndarray,
    b: np.ndarray,
    m: float,
    *,
    phase_source: str = "a",
    mag_curve: str = "linear",
) -> np.ndarray:
    """
    Spectral morph between two single-cycle frames.
    - Interpolates magnitudes (not time-domain samples).
    - Uses a stable phase reference to keep the cycle coherent.

    phase_source:
      "a"   -> keep phase of a across morph (stable for families)
      "b"   -> keep phase of b
      "mix" -> interpolate phase (can cause phase wandering; use intentionally)

    mag_curve:
      "linear" -> straight interpolation
      "sqrt"   -> slightly more perceptually even (optional)
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Frame size mismatch: {a.shape} vs {b.shape}")

    N = a.shape[0]
    m = float(np.clip(m, 0.0, 1.0))

    A = rfft(a)
    B = rfft(b)

    magA, phA = mag_phase(A)
    magB, phB = mag_phase(B)

    if mag_curve == "linear":
        mag = (1.0 - m) * magA + m * magB
    elif mag_curve == "sqrt":
        # reduces perceived "jump" in some cases
        mag = np.sqrt((1.0 - m) * (magA**2) + m * (magB**2))
    else:
        raise ValueError(f"Unknown mag_curve: {mag_curve}")

    if phase_source == "a":
        phase = phA
    elif phase_source == "b":
        phase = phB
    elif phase_source == "mix":
        # NOTE: Phase interpolation can wrap; unwrap to reduce discontinuities
        phA_u = np.unwrap(phA)
        phB_u = np.unwrap(phB)
        phase = (1.0 - m) * phA_u + m * phB_u
    else:
        raise ValueError(f"Unknown phase_source: {phase_source}")

    C = polar(mag, phase)
    y = irfft(C, n=N)
    return y


# -----------------------------
# Diagnostics (optional but useful)
# -----------------------------
def spectrum_db(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (bin_index, magnitude_db) for a single-cycle frame.
    Useful for plotting/debugging.
    """
    x = np.asarray(frame, dtype=np.float64)
    X = rfft(x)
    mag = np.abs(X)
    mag_db = 20.0 * np.log10(np.maximum(mag, EPS))
    bins = np.arange(len(mag_db))
    return bins, mag_db

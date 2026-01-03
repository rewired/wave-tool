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


def align_max_negative_jump(x: np.ndarray) -> np.ndarray:
    """Rotate so the largest negative jump occurs at index 0."""
    x = np.asarray(x, dtype=np.float64)
    diffs = np.diff(x, append=x[0])
    idx = int(np.argmin(diffs))
    return np.roll(x, -(idx + 1))


def sanitize_cycle(
    x: np.ndarray,
    *,
    peak: float = 0.99,
    align_discontinuity: bool = False,
) -> np.ndarray:
    """Standard cleanup for a single-cycle frame."""
    x = remove_dc(x)
    if align_discontinuity:
        x = align_max_negative_jump(x)
    x = normalize_peak(x, peak=peak)
    return x


# -----------------------------
# Spectrum helpers
# -----------------------------

def clamp_max_harm(max_harm: int, frame_len: int) -> int:
    """Clamp max harmonic to a valid range for the frame length."""
    max_harm = int(max_harm)
    nyq = frame_len // 2 - 1
    return max(1, min(max_harm, nyq))


def lanczos_taper(max_harm: int) -> np.ndarray:
    """Lanczos (sinc) taper for harmonics 0..max_harm."""
    if max_harm < 1:
        return np.ones(1, dtype=np.float64)
    k = np.arange(0, max_harm + 1, dtype=np.float64)
    x = k / (max_harm + 1)
    return np.sinc(x)


def harmonic_spectrum(
    amps: np.ndarray,
    frame_len: int,
    *,
    phase: float = -np.pi / 2,
) -> np.ndarray:
    """Build a complex rFFT spectrum from harmonic amplitudes."""
    amps = np.asarray(amps, dtype=np.float64)
    n_bins = frame_len // 2 + 1
    spec = np.zeros(n_bins, dtype=np.complex128)
    max_harm = min(len(amps) - 1, n_bins - 1)
    if max_harm <= 0:
        return spec
    phasor = np.exp(1j * phase)
    spec[1 : max_harm + 1] = amps[1 : max_harm + 1] * phasor
    return spec


def harmonics_to_frame(
    amps: np.ndarray,
    frame_len: int,
    *,
    phase: float = -np.pi / 2,
) -> np.ndarray:
    """Synthesize a single-cycle frame from harmonic amplitudes."""
    spec = harmonic_spectrum(amps, frame_len, phase=phase)
    return np.fft.irfft(spec, n=frame_len)


# -----------------------------
# Diagnostics (optional)
# -----------------------------

def spectrum_db(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (bin_index, magnitude_db) for a single-cycle frame.
    Useful for plotting/debugging.
    """
    x = np.asarray(frame, dtype=np.float64)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    mag_db = 20.0 * np.log10(np.maximum(mag, EPS))
    bins = np.arange(len(mag_db))
    return bins, mag_db

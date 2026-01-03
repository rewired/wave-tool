# src/generators.py
from __future__ import annotations

import numpy as np
from .dsp_core import irfft, polar, sanitize_cycle, harmonic_mask

EPS = 1e-12


def _single_cycle_spectrum_from_harmonics(
    N: int,
    amps: np.ndarray,
    phases: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build a single-cycle waveform from harmonic amplitudes (and optional phases)
    via an RFFT spectrum.

    amps: array of length H+1 where amps[k] is amplitude of harmonic k
          (k=0 is DC).
    phases: array of same length; if None -> all zeros.
    """
    if phases is None:
        phases = np.zeros_like(amps, dtype=np.float64)

    # rfft bins are 0..N/2
    n_bins = N // 2 + 1
    H = min(len(amps) - 1, n_bins - 1)

    mag = np.zeros(n_bins, dtype=np.float64)
    ph = np.zeros(n_bins, dtype=np.float64)

    mag[: H + 1] = np.asarray(amps[: H + 1], dtype=np.float64)
    ph[: H + 1] = np.asarray(phases[: H + 1], dtype=np.float64)

    X = polar(mag, ph)
    x = irfft(X, n=N)
    return x


def sine(N: int, *, peak: float = 0.99) -> np.ndarray:
    t = np.arange(N, dtype=np.float64) / N
    x = np.sin(2.0 * np.pi * t)
    return sanitize_cycle(x, peak=peak)


def saw_additive(
    N: int,
    *,
    max_harm: int = 128,
    peak: float = 0.99,
    sign: int = -1,
) -> np.ndarray:
    """
    Bandlimited saw via sine-series harmonics:
      saw(t) ~ sum_{k=1..H} (1/k) * sin(2*pi*k*t)

    Implemented by setting phase = -pi/2 for each harmonic so the IFFT produces sine components.
    """
    n_bins = N // 2 + 1
    H = min(int(max_harm), n_bins - 1)

    amps = np.zeros(H + 1, dtype=np.float64)
    phases = np.zeros(H + 1, dtype=np.float64)

    for k in range(1, H + 1):
        amps[k] = 1.0 / k
        phases[k] = -np.pi / 2  # cos -> sin

    x = _single_cycle_spectrum_from_harmonics(N, amps, phases=phases)
    x = float(sign) * x
    return sanitize_cycle(x, peak=peak)


def square_additive(
    N: int,
    *,
    max_harm: int = 128,
    peak: float = 0.99,
) -> np.ndarray:
    """
    Bandlimited square: only odd harmonics, amp(k)=1/k.
    """
    n_bins = N // 2 + 1
    H = min(int(max_harm), n_bins - 1)

    amps = np.zeros(H + 1, dtype=np.float64)
    for k in range(1, H + 1):
        if k % 2 == 1:
            amps[k] = 1.0 / k

    x = _single_cycle_spectrum_from_harmonics(N, amps)
    return sanitize_cycle(x, peak=peak)


def triangle_additive(
    N: int,
    *,
    max_harm: int = 128,
    peak: float = 0.99,
) -> np.ndarray:
    """
    Bandlimited triangle: odd harmonics, amp(k)=1/k^2 with alternating sign.
    """
    n_bins = N // 2 + 1
    H = min(int(max_harm), n_bins - 1)

    amps = np.zeros(H + 1, dtype=np.float64)
    phases = np.zeros(H + 1, dtype=np.float64)

    # Alternating sign can be expressed as phase 0 or pi
    # (equivalent to multiplying by -1 in harmonic domain)
    alt = 1
    for k in range(1, H + 1):
        if k % 2 == 1:
            amps[k] = 1.0 / (k * k)
            if alt < 0:
                phases[k] = np.pi
            alt *= -1

    x = _single_cycle_spectrum_from_harmonics(N, amps, phases=phases)
    return sanitize_cycle(x, peak=peak)


def apply_max_harm(frame: np.ndarray, max_harm: int) -> np.ndarray:
    """
    Convenience: spectral hard limit. Useful if you do shaping later.
    """
    x = np.asarray(frame, dtype=np.float64)
    N = x.shape[0]
    X = np.fft.rfft(x)
    m = harmonic_mask(len(X), max_harm=max_harm)
    y = np.fft.irfft(X * m, n=N)
    return sanitize_cycle(y)

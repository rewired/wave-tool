from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from . import dsp_core
from .spectra import harmonic_spectrum


@dataclass(frozen=True)
class Keyframe:
    position: float
    shape: str


def _sorted_keyframes(keyframes: Iterable[tuple[float, str] | Keyframe]) -> list[Keyframe]:
    frames = [
        k if isinstance(k, Keyframe) else Keyframe(position=k[0], shape=k[1])
        for k in keyframes
    ]
    if len(frames) < 2:
        raise ValueError("At least two keyframes are required.")
    frames.sort(key=lambda k: k.position)
    for k in frames:
        if not 0.0 <= k.position <= 1.0:
            raise ValueError("Keyframe positions must be within [0, 1].")
    return frames


def _frame_positions(frames: int) -> np.ndarray:
    if frames < 1:
        raise ValueError("frames must be >= 1")
    if frames == 1:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(0.0, 1.0, frames, dtype=np.float64)


def _segment_index(positions: Sequence[float], p: float, start: int) -> int:
    idx = start
    last = len(positions) - 1
    while idx < last and p > positions[idx + 1]:
        idx += 1
    return idx


def build_wavetable(
    keyframes: Iterable[tuple[float, str] | Keyframe],
    *,
    frame_len: int = 2048,
    frames: int = 256,
    max_harm: int = 256,
    taper: str = "lanczos",
    align_discontinuity: bool = False,
    phase: float = -np.pi / 2,
) -> np.ndarray:
    """Build a single wavetable by interpolating keyframed spectra."""
    if frame_len < 2:
        raise ValueError("frame_len must be >= 2")

    sorted_keys = _sorted_keyframes(keyframes)
    max_harm = dsp_core.clamp_max_harm(max_harm, frame_len)

    key_positions = [k.position for k in sorted_keys]
    key_amps = [
        harmonic_spectrum(k.shape, max_harm, taper=taper) for k in sorted_keys
    ]

    table_frames = []
    seg_idx = 0
    positions = _frame_positions(frames)
    for p in positions:
        seg_idx = _segment_index(key_positions, float(p), seg_idx)
        if seg_idx >= len(sorted_keys) - 1:
            a = key_amps[-1]
            b = key_amps[-1]
            t = 0.0
        else:
            a = key_amps[seg_idx]
            b = key_amps[seg_idx + 1]
            a_pos = key_positions[seg_idx]
            b_pos = key_positions[seg_idx + 1]
            if b_pos <= a_pos:
                t = 0.0
            else:
                t = (p - a_pos) / (b_pos - a_pos)

        amps = (1.0 - t) * a + t * b
        frame = dsp_core.harmonics_to_frame(amps, frame_len, phase=phase)
        frame = dsp_core.sanitize_cycle(
            frame,
            align_discontinuity=align_discontinuity,
        )
        table_frames.append(frame)

    return np.concatenate(table_frames)

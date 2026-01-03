from __future__ import annotations

from pathlib import Path

import soundfile as sf

from .keyframes import build_wavetable


def _output_name(
    label: str, frame_len: int, frames: int, max_harm: int, taper: str
) -> str:
    return f"{label}_N{frame_len}_F{frames}_H{max_harm}_{taper}.wav"


def _write_wavetable(path: Path, data, sample_rate: int = 44100) -> None:
    path.parent.mkdir(exist_ok=True)
    sf.write(path, data, samplerate=sample_rate, subtype="FLOAT")


def main() -> None:
    frame_len = 2048
    frames = 256
    max_harm = 256
    taper = "lanczos"

    out_dir = Path(__file__).resolve().parents[1] / "out"

    tables = {
        "sine-saw": [(0.0, "sine"), (1.0, "saw")],
        "sine-saw-square": [(0.0, "sine"), (0.5, "saw"), (1.0, "square")],
    }

    for label, keyframes in tables.items():
        wavetable = build_wavetable(
            keyframes,
            frame_len=frame_len,
            frames=frames,
            max_harm=max_harm,
            taper=taper,
            align_discontinuity=False,
        )
        out_path = out_dir / _output_name(label, frame_len, frames, max_harm, taper)
        _write_wavetable(out_path, wavetable)
        print(out_path)


if __name__ == "__main__":
    main()

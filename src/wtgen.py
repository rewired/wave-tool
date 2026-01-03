# src/wtgen.py
from __future__ import annotations
from pathlib import Path

from .generators import sine, saw_additive
from .tables import make_morph_table, write_wavetable_wav


OUT_DIR = Path(__file__).resolve().parents[1] / "out"
OUT_DIR.mkdir(exist_ok=True)

def main():
    N = 2048         # samples per frame
    FRAMES = 256     # number of frames in table
    MAX_HARM = 128   # harmonic limit for the saw (bandlimit proxy)

    a = sine(N)
    b = saw_additive(N, max_harm=N, sign=-1)

    table = make_morph_table(
        a, b,
        frames=FRAMES,
        phase_source="b",   # <-- wichtig
        mag_curve="linear",
    )

    out_path = OUT_DIR / "SINE_TO_SAW_WT.wav"
    write_wavetable_wav(str(out_path), table, sample_rate=44100)
    print(out_path)


if __name__ == "__main__":
    main()

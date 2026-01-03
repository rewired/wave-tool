# wave-tool

Deterministic wavetable generator using harmonic-domain keyframes. Outputs are
single-cycle frames concatenated into a mono WAV written to `/out`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate # Linux/MacOS
.venv\Scripts\Activate.ps1 # Windows
pip install -r requirements.txt
```

## Generate example wavetables

```bash
python -m src.wt_gen
```

Outputs are written to `/out` and named like:

```
sine-saw_N2048_F256_H256_lanczos.wav
sine-saw-square_N2048_F256_H256_lanczos.wav
```

Where:
- `N` is the single-cycle frame length
- `F` is the frame count
- `H` is the max harmonic used
- The final token is the taper mode

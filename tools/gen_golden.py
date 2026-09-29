"""
Generates tests/golden/*.json — reference values produced by SciPy.

The browser build re-implements the analysis that SciPy provides on the Python
side (Welch's method, fractional-octave smoothing, the waveform envelope),
because there is no SciPy in a browser. These files are how the JavaScript
tests prove that re-implementation still agrees with the original.

Every signal is described declaratively (`signal` in each file) so the JS side
can rebuild the exact same samples without shipping megabytes of audio.

Run:   python tools/gen_golden.py
Check: python tools/gen_golden.py --check    (exit 1 if the files are stale)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scipy import signal as sp_signal  # noqa: E402

from eqcore import equalizer, presets  # noqa: E402

GOLDEN_DIR = ROOT / "tests" / "golden"

#: Deterministic, RNG-free so both languages can rebuild it bit for bit:
#: two tones plus a DC offset (which exercises Welch's per-segment detrend)
#: plus a slow ramp (which gives the envelope something asymmetric to track).
SIGNAL = {
    "fs": 44100,
    "samples": 220_500,          # 5 seconds
    "tones": [[1000.0, 0.5], [5000.0, 0.25]],
    "dc": 0.1,
    "ramp": 0.15,
}


def build_signal() -> np.ndarray:
    """Must stay in lockstep with buildSignal() in tests/js/golden.mjs."""
    n = SIGNAL["samples"]
    fs = SIGNAL["fs"]
    i = np.arange(n, dtype=np.float64)

    out = np.full(n, SIGNAL["dc"], dtype=np.float64)
    for freq, amp in SIGNAL["tones"]:
        out += amp * np.sin(2.0 * np.pi * freq * i / fs)
    out += SIGNAL["ramp"] * (i / n)
    return out


def probe_indices(freqs: np.ndarray, count: int = 28) -> list[int]:
    """Log-spaced probe bins, so low and high frequencies are equally covered."""
    targets = np.logspace(np.log10(20.0), np.log10(freqs[-1] * 0.98), count)
    seen: list[int] = []
    for target in targets:
        idx = int(np.argmin(np.abs(freqs - target)))
        if idx not in seen:
            seen.append(idx)
    return seen


def spectrum_golden(smooth_fraction: float) -> dict:
    x = build_signal()
    freqs, db = equalizer.spectrum(x, SIGNAL["fs"], smooth_fraction=smooth_fraction)
    idx = probe_indices(freqs)
    return {
        "signal": SIGNAL,
        "smooth_fraction": smooth_fraction,
        "bins": int(freqs.size),
        "nperseg": int(2 * (freqs.size - 1)),
        "probes": [
            {"bin": i, "freq": float(freqs[i]), "db": float(db[i])} for i in idx
        ],
    }


def envelope_golden() -> dict:
    x = build_signal()
    target_points = 512
    lo, hi = equalizer.waveform_envelope(x, target_points=target_points)
    idx = list(range(0, lo.size, max(1, lo.size // 40)))
    return {
        "signal": SIGNAL,
        "target_points": target_points,
        "size": int(lo.size),
        "probes": [
            {"i": i, "min": float(lo[i]), "max": float(hi[i])} for i in idx
        ],
    }


def gain_staging_golden() -> dict:
    cases = []
    for peak in (0.0, 0.25, 0.5, 0.8999, 0.9, 0.9001, 1.0, 1.1478, 2.5):
        if peak > equalizer.HEADROOM_PEAK and peak > 0.0:
            scale = equalizer.HEADROOM_PEAK / peak
            cases.append({
                "peak": peak,
                "scale": scale,
                "applied_gain_db": float(20.0 * np.log10(scale)),
                "clipped": True,
            })
        else:
            cases.append({
                "peak": peak, "scale": 1.0, "applied_gain_db": 0.0, "clipped": False,
            })
    return {"headroom_peak": equalizer.HEADROOM_PEAK, "cases": cases}


def response_golden() -> dict:
    """
    The magnitude response of the cascade, for the browser test.

    web/ does not implement biquads — it hands f0/Q/gain to Web Audio's native
    BiquadFilterNode. This file is what tests/browser/response.spec.mjs checks
    that node's getFrequencyResponse against.
    """
    freqs = np.logspace(np.log10(20.0), np.log10(20000.0), 48)
    cases = []
    for preset_name in presets.PRESET_NAMES:
        gains = presets.get(preset_name)
        for fs in (44100, 48000):
            sos = equalizer.build_sos(gains, fs)
            _, h = sp_signal.sosfreqz(sos, worN=freqs, fs=fs)
            cases.append({
                "preset": preset_name,
                "fs": fs,
                "gains": gains,
                "db": [float(v) for v in 20.0 * np.log10(np.abs(h))],
            })
    return {"freqs": [float(f) for f in freqs], "cases": cases}


FILES = {
    "spectrum_raw.json": lambda: spectrum_golden(0.0),
    "spectrum_smoothed.json": lambda: spectrum_golden(12.0),
    "envelope.json": envelope_golden,
    "gain_staging.json": gain_staging_golden,
    "response.json": response_golden,
}


def render(name: str) -> str:
    return json.dumps(FILES[name](), indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the committed files instead of writing them")
    args = parser.parse_args()

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    stale = []

    for name in FILES:
        path = GOLDEN_DIR / name
        rendered = render(name)
        if args.check:
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            if current != rendered:
                stale.append(name)
        else:
            path.write_text(rendered, encoding="utf-8")
            print(f"wrote tests/golden/{name}")

    if args.check:
        if stale:
            print(
                "STALE golden file(s): " + ", ".join(stale)
                + "\nRun: python tools/gen_golden.py",
                file=sys.stderr,
            )
            return 1
        print("golden files are up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

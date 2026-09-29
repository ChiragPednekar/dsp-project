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

--check compares numerically, not byte for byte. SciPy is not bit-reproducible
across platforms -- the same call returns values differing in the last few
digits on macOS/arm64 and Linux/x86_64 -- so a text comparison would fail in CI
for files that are perfectly current. The tolerance below is many orders of
magnitude tighter than any real implementation change.
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


#: Numbers this close are platform noise, not a change in the implementation.
#: A genuine edit to a band, preset or algorithm moves values by >= 1e-3.
RTOL = 1e-7
ATOL = 1e-9


def differences(expected, actual, path: str = "") -> list[str]:
    """
    Human-readable differences between two parsed golden structures.

    Structure must match exactly; floats need only agree to RTOL/ATOL.
    """
    where = path or "<root>"

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{where}: expected an object, found {type(actual).__name__}"]
        out = []
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                out.append(f"{where}.{key}: missing")
            elif key not in expected:
                out.append(f"{where}.{key}: unexpected")
            else:
                out += differences(expected[key], actual[key], f"{path}.{key}")
        return out

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{where}: expected a list, found {type(actual).__name__}"]
        if len(expected) != len(actual):
            return [f"{where}: length {len(actual)}, expected {len(expected)}"]
        out = []
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            out += differences(e, a, f"{path}[{i}]")
        return out

    if isinstance(expected, bool) or isinstance(actual, bool):
        return [] if expected == actual else [f"{where}: {actual!r} != {expected!r}"]

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        tolerance = ATOL + RTOL * abs(expected)
        delta = abs(float(expected) - float(actual))
        if delta > tolerance:
            return [f"{where}: {actual!r} != {expected!r} (off by {delta:.3e})"]
        return []

    return [] if expected == actual else [f"{where}: {actual!r} != {expected!r}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the committed files instead of writing them")
    args = parser.parse_args()

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    stale = []

    for name in FILES:
        path = GOLDEN_DIR / name

        if args.check:
            if not path.exists():
                stale.append(f"tests/golden/{name}: missing")
                continue
            committed = json.loads(path.read_text(encoding="utf-8"))
            diffs = differences(FILES[name](), committed)
            if diffs:
                shown = diffs[:5]
                if len(diffs) > len(shown):
                    shown.append(f"... and {len(diffs) - len(shown)} more")
                stale.append(
                    f"tests/golden/{name}:\n    " + "\n    ".join(shown)
                )
        else:
            path.write_text(render(name), encoding="utf-8")
            print(f"wrote tests/golden/{name}")

    if args.check:
        if stale:
            print(
                "Golden file(s) no longer match the Python implementation:\n\n"
                + "\n".join(stale)
                + "\n\nRun: python tools/gen_golden.py",
                file=sys.stderr,
            )
            return 1
        print(f"golden files agree with SciPy (rtol={RTOL:g}, atol={ATOL:g}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

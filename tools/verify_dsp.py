"""
Verifies that the filters really do what the EQ curve claims.

For each check we push a signal through the actual `equalizer.process` path
and measure the result, then compare against the analytically computed
frequency response. If the drawn curve were decorative rather than the real
response of the chain, these checks would fail.

Run:  .venv/bin/python tools/verify_dsp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eqcore import biquad, equalizer, presets

FS = 44100
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(f"{label}: {detail}")


def sine(freq: float, seconds: float = 2.0, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def steady_state_amplitude(x: np.ndarray) -> float:
    """RMS of the back half, so the filter's transient is excluded."""
    tail = x[len(x) // 2:]
    return float(np.sqrt(np.mean(np.square(tail))) * np.sqrt(2.0))


def response_at(gains: dict[str, float], freq: float) -> float:
    """The designed response in dB at one frequency."""
    sos = equalizer.build_sos(gains, FS)
    from scipy import signal

    _, h = signal.sosfreqz(sos, worN=np.array([freq]), fs=FS)
    return float(20 * np.log10(abs(h[0])))


print("\n=== 1. Measured gain matches designed gain (peaking bands) ===")
for band_key, test_freq, gain in [
    ("mid", 1000.0, 12.0),
    ("mid", 1000.0, -12.0),
    ("bass", 122.0, 9.0),
    ("high_mid", 2828.0, 6.0),
    ("presence", 4899.0, -8.0),
]:
    gains = equalizer.flat_gains()
    gains[band_key] = gain

    x = sine(test_freq)
    # headroom_peak=1e9 disables the safety normaliser so we measure raw gain.
    y = equalizer.process(x, FS, gains, headroom_peak=1e9).audio

    measured = 20 * np.log10(steady_state_amplitude(y) / steady_state_amplitude(x))
    designed = response_at(gains, test_freq)
    err = abs(measured - designed)
    check(
        f"{band_key} @ {test_freq:.0f} Hz, {gain:+.0f} dB",
        err < 0.15,
        f"measured {measured:+.2f} dB, designed {designed:+.2f} dB, err {err:.3f}",
    )

print("\n=== 2. Shelving bands reach their target gain in the shelf ===")
for band_key, test_freq, gain in [
    ("sub_bass", 25.0, 10.0),
    ("sub_bass", 25.0, -10.0),
    ("brilliance", 15000.0, 8.0),
]:
    gains = equalizer.flat_gains()
    gains[band_key] = gain
    x = sine(test_freq, seconds=4.0)
    y = equalizer.process(x, FS, gains, headroom_peak=1e9).audio
    measured = 20 * np.log10(steady_state_amplitude(y) / steady_state_amplitude(x))
    designed = response_at(gains, test_freq)
    check(
        f"{band_key} @ {test_freq:.0f} Hz, {gain:+.0f} dB",
        abs(measured - designed) < 0.3,
        f"measured {measured:+.2f} dB, designed {designed:+.2f} dB",
    )
    # A shelf must actually approach its nominal gain, not just "move a bit".
    check(
        f"{band_key} shelf reaches ~{gain:+.0f} dB",
        abs(designed - gain) < 1.5,
        f"designed {designed:+.2f} dB vs nominal {gain:+.2f} dB",
    )

print("\n=== 3. Bands are independent (a boost does not leak across the spectrum) ===")
gains = equalizer.flat_gains()
gains["bass"] = 12.0
for probe, limit in [(1000.0, 1.0), (5000.0, 0.5), (12000.0, 0.5)]:
    leak = abs(response_at(gains, probe))
    check(f"bass +12 dB leaks < {limit} dB at {probe:.0f} Hz",
          leak < limit, f"{leak:.3f} dB")

print("\n=== 4. Flat settings are a true pass-through ===")
x = sine(1000.0)
y = equalizer.process(x, FS, equalizer.flat_gains(), headroom_peak=1e9).audio
max_diff = float(np.max(np.abs(y - x)))
check("flat EQ is bit-transparent", max_diff < 1e-6, f"max sample diff {max_diff:.2e}")

print("\n=== 5. All sections are stable across the full gain range ===")
unstable = []
for band in equalizer.BANDS:
    for gain in np.arange(-12.0, 12.5, 0.5):
        for fs in (8000, 22050, 44100, 48000, 96000):
            sec = biquad.design(band.kind, band.f0, float(gain), band.q, fs)
            if not biquad.is_stable(sec):
                unstable.append((band.key, gain, fs))
check("every band stable at all gains / sample rates", not unstable,
      f"{len(unstable)} unstable")

print("\n=== 6. Gain staging prevents clipping ===")
loud = (0.95 * np.sin(2 * np.pi * 120 * np.arange(FS * 2) / FS)).astype(np.float32)
result = equalizer.process(loud, FS, presets.get("Bass Boost"))
peak = float(np.max(np.abs(result.audio)))
check("output peak within headroom", peak <= equalizer.HEADROOM_PEAK + 1e-6,
      f"peak {peak:.4f}")
check("attenuation was reported", result.clipped and result.applied_gain_db < 0,
      f"applied {result.applied_gain_db:.2f} dB")

print("\n=== 7. Stereo is filtered per channel, not collapsed ===")
left = sine(120.0)
right = sine(5000.0)
stereo = np.stack([left, right], axis=1)
gains = equalizer.flat_gains()
gains["bass"] = 12.0
out = equalizer.process(stereo, FS, gains, headroom_peak=1e9).audio
check("stereo shape preserved", out.shape == stereo.shape,
      f"{stereo.shape} -> {out.shape}")
l_gain = 20 * np.log10(steady_state_amplitude(out[:, 0]) / steady_state_amplitude(left))
r_gain = 20 * np.log10(steady_state_amplitude(out[:, 1]) / steady_state_amplitude(right))
check("left (120 Hz) boosted", l_gain > 10.0, f"{l_gain:+.2f} dB")
check("right (5 kHz) untouched", abs(r_gain) < 0.5, f"{r_gain:+.2f} dB")

print("\n=== 8. Presets move the response in the direction they claim ===")
for name, probe, expect_positive in [
    ("Bass Boost", 50.0, True),
    ("Vocal Boost", 3000.0, True),
    ("Treble Boost", 12000.0, True),
    ("Podcast / Speech", 40.0, False),
    ("Loudness (V-shape)", 1000.0, False),
]:
    db = response_at(presets.get(name), probe)
    ok = (db > 2.0) if expect_positive else (db < -2.0)
    check(f"{name} @ {probe:.0f} Hz", ok, f"{db:+.2f} dB")

print("\n=== 9. Spectrum analysis reflects the EQ change ===")
rng = np.random.default_rng(0)
noise = (0.1 * rng.standard_normal(FS * 3)).astype(np.float32)
gains = equalizer.flat_gains()
gains["presence"] = 12.0
eqd = equalizer.process(noise, FS, gains, headroom_peak=1e9).audio

f_before, s_before = equalizer.spectrum(noise, FS)
f_after, s_after = equalizer.spectrum(eqd, FS)
idx = int(np.argmin(np.abs(f_after - 4899.0)))
delta = s_after[idx] - s_before[idx]
check("measured spectrum shows the +12 dB presence lift", 9.0 < delta < 14.0,
      f"delta {delta:+.2f} dB at {f_after[idx]:.0f} Hz")

lo = int(np.argmin(np.abs(f_after - 300.0)))
check("spectrum unchanged at 300 Hz", abs(s_after[lo] - s_before[lo]) < 1.0,
      f"delta {s_after[lo] - s_before[lo]:+.2f} dB")

print("\n" + "=" * 62)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("ALL DSP CHECKS PASSED")

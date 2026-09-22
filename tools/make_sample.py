"""
Generates samples/demo.wav — a synthetic clip for exercising the EQ.

The clip deliberately has energy in every band the equaliser touches, so that
moving any one slider produces an audible and visible change:

  * a bass line an octave apart from a sustained pad
  * a mid-range melody
  * a hi-hat-ish noise burst pattern for the presence/brilliance bands
  * gentle pink-ish noise underneath so the spectrum plot has a floor to read

Run:  .venv/bin/python tools/make_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

FS = 44100
DURATION = 45.0
OUT = Path(__file__).resolve().parent.parent / "samples" / "demo.wav"


def repeats(pattern: list, step: float) -> int:
    """How many times to loop `pattern` (one note every `step` s) to fill DURATION."""
    return int(DURATION / (len(pattern) * step)) + 1


def adsr(n: int, attack=0.01, decay=0.1, sustain=0.7, release=0.2) -> np.ndarray:
    """A simple amplitude envelope so notes have transients to look at."""
    a, d, r = int(attack * FS), int(decay * FS), int(release * FS)
    a, d, r = min(a, n), min(d, max(n - a, 0)), min(r, max(n - a - d, 0))
    s = max(n - a - d - r, 0)
    return np.concatenate([
        np.linspace(0, 1, a, endpoint=False),
        np.linspace(1, sustain, d, endpoint=False),
        np.full(s, sustain),
        np.linspace(sustain, 0, r),
    ])[:n]


def note(freq: float, start: float, length: float, amp: float,
         harmonics: int = 4) -> tuple[int, np.ndarray]:
    """A harmonically rich tone — richer than a sine, so the EQ has something to grip."""
    n = int(length * FS)
    t = np.arange(n) / FS
    wave = np.zeros(n)
    for h in range(1, harmonics + 1):
        f = freq * h
        if f >= FS / 2:
            break
        wave += (1.0 / h) * np.sin(2 * np.pi * f * t)
    wave *= adsr(n) * amp
    return int(start * FS), wave


def main() -> int:
    total = int(DURATION * FS)
    mix = np.zeros(total)
    rng = np.random.default_rng(7)

    # Bass line — sits in the sub-bass and bass bands.
    bass_pattern = [55.0, 55.0, 73.42, 82.41, 55.0, 55.0, 98.0, 82.41]
    for i, f in enumerate(bass_pattern * repeats(bass_pattern, 0.5)):
        start = i * 0.5
        if start >= DURATION:
            break
        at, w = note(f, start, 0.45, 0.30, harmonics=6)
        mix[at:at + len(w)] += w[:max(0, total - at)][:len(mix[at:at + len(w)])]

    # Mid melody — low-mid through high-mid.
    melody = [440.0, 523.25, 659.25, 523.25, 587.33, 440.0, 392.0, 440.0]
    for i, f in enumerate(melody * repeats(melody, 0.5)):
        start = 0.25 + i * 0.5
        if start >= DURATION:
            break
        at, w = note(f, start, 0.4, 0.16, harmonics=5)
        end = min(at + len(w), total)
        mix[at:end] += w[:end - at]

    # Sustained pad for continuous mid energy.
    t = np.arange(total) / FS
    for f, a in [(220.0, 0.05), (277.18, 0.04), (329.63, 0.04)]:
        mix += a * np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t))

    # Hats — high-passed noise bursts feeding presence and brilliance.
    for i in range(int(DURATION * 4)):
        at = int(i * 0.25 * FS)
        n = int(0.06 * FS)
        if at + n > total:
            break
        burst = rng.standard_normal(n)
        # One-pole high-pass so the burst is bright rather than full-band.
        burst = np.diff(np.concatenate([[0.0], burst]))
        env = np.exp(-np.linspace(0, 9, n))
        mix[at:at + n] += burst * env * (0.10 if i % 4 == 0 else 0.05)

    # Low-level broadband bed so every band has a readable noise floor.
    mix += 0.006 * rng.standard_normal(total)

    # Fade the very ends to avoid a click on loop.
    fade = int(0.05 * FS)
    mix[:fade] *= np.linspace(0, 1, fade)
    mix[-fade:] *= np.linspace(1, 0, fade)

    peak = float(np.max(np.abs(mix)))
    mix = (mix / peak) * 0.72 if peak > 0 else mix

    # Stereo with a slight width, so per-channel filtering is exercised too.
    stereo = np.stack([mix, np.roll(mix, 180) * 0.97], axis=1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sf.write(OUT, stereo.astype(np.float32), FS, subtype="PCM_16")

    size = OUT.stat().st_size / 1_048_576
    print(f"wrote {OUT}  ({DURATION:.0f}s stereo @ {FS} Hz, {size:.2f} MB, "
          f"peak {np.max(np.abs(stereo)):.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

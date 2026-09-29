"""
The 7-band equaliser itself.

Scope note: each band's centre frequency and Q are fixed by
`shared/eq_spec.json`; only the per-band gain is adjustable. That makes this a
graphic EQ in the usual sense of the term. Exposing f0 and Q per band is the
natural next step -- `biquad.design` already takes both -- and is tracked in
the README.

Design: one biquad per band, cascaded in series as a bank of second-order
sections. The lowest band is a low shelf, the highest a high shelf, and the
five bands in between are peaking (bell) filters. That is the standard layout
for a graphic EQ — shelves at the extremes so that the very bottom and very
top of the spectrum are lifted as a whole rather than pushed into a bell that
would roll off again past its centre frequency.

Filtering runs through scipy.signal.sosfilt, which applies the sections one
after another in the direct-form-II-transposed structure. Cascading SOS rather
than convolving the coefficients into one high-order polynomial is what keeps
the filter numerically stable at low frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from . import biquad, spec

# Re-exported (PEP 484 explicit form): callers import it from here.
from .spec import q_from_octaves as q_from_octaves

#: Peak level the processed signal is limited to, in linear amplitude.
#: Leaves ~0.9 dB of headroom below full scale so that boosted material does
#: not clip when written to a 16-bit WAV.
HEADROOM_PEAK = spec.HEADROOM_PEAK

GAIN_MIN_DB = spec.GAIN_MIN_DB
GAIN_MAX_DB = spec.GAIN_MAX_DB


@dataclass(frozen=True)
class Band:
    """One EQ band: what kind of filter it is and where it sits."""

    key: str            # stable identifier used for session state
    name: str           # human label shown next to the slider
    kind: str           # "low_shelf" | "peaking" | "high_shelf"
    f0: float           # centre (peaking) or corner (shelf) frequency in Hz
    q: float            # Q factor / shelf slope
    span: str           # the nominal frequency range this band governs

    @property
    def slider_label(self) -> str:
        return f"{self.name}  ·  {self.span}"


# Built from shared/eq_spec.json rather than written out here, so the browser
# build (web/spec.js, generated from the same file) cannot drift away from it.
# The layout follows the classic audio-engineering split of the spectrum:
# peaking centres are the geometric mean of the band edges, and each bell's Q
# is derived from its width in octaves so the bells meet cleanly.
BANDS: tuple[Band, ...] = tuple(
    Band(
        key=entry["key"],
        name=entry["name"],
        kind=entry["kind"],
        f0=float(entry["f0"]),
        q=spec.band_q(entry),
        span=entry["span"],
    )
    for entry in spec.band_entries()
)

BAND_KEYS: tuple[str, ...] = tuple(b.key for b in BANDS)


def flat_gains() -> dict[str, float]:
    """A neutral setting: every band at 0 dB."""
    return {band.key: 0.0 for band in BANDS}


def build_sos(gains: dict[str, float], fs: float) -> np.ndarray:
    """
    Build the cascaded second-order-section matrix for the current gains.

    Bands sitting at (or within rounding distance of) 0 dB are skipped —
    a 0 dB biquad is mathematically the identity, so dropping it saves work
    and avoids accumulating pointless rounding error. If every band is flat we
    return a single pass-through section so callers always get a valid SOS.
    """
    sections: list[np.ndarray] = []

    for band in BANDS:
        gain_db = float(np.clip(gains.get(band.key, 0.0), GAIN_MIN_DB, GAIN_MAX_DB))
        if abs(gain_db) < 1e-3:
            continue

        section = biquad.design(band.kind, band.f0, gain_db, band.q, fs)
        if not biquad.is_stable(section):
            # Degenerate design — skip this band rather than emit a filter
            # that rings. In practice this never fires at 44.1/48 kHz.
            continue
        sections.append(section)

    if not sections:
        sections.append(np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))

    return np.vstack(sections)


def frequency_response(
    gains: dict[str, float],
    fs: float,
    n_points: int = 1024,
    f_min: float = 20.0,
    f_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The real, computed magnitude response of the current filter chain.

    This evaluates H(e^{jw}) for the actual cascaded SOS that will filter the
    audio — it is not a drawing of the slider positions. Returns
    (frequencies in Hz, magnitude in dB) on a log-spaced frequency grid.
    """
    if f_max is None:
        f_max = fs / 2.0
    f_max = min(f_max, fs / 2.0 * 0.999)

    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_points)
    sos = build_sos(gains, fs)
    _, h = signal.sosfreqz(sos, worN=freqs, fs=fs)

    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
    return freqs, magnitude_db


@dataclass
class ProcessResult:
    """Processed audio plus what the gain staging had to do to it."""

    audio: np.ndarray
    applied_gain_db: float      # attenuation applied to avoid clipping (<= 0)
    peak_before: float          # peak of the filtered signal pre-normalisation
    clipped: bool               # whether normalisation was needed at all


def process(
    audio: np.ndarray,
    fs: float,
    gains: dict[str, float],
    headroom_peak: float = HEADROOM_PEAK,
) -> ProcessResult:
    """
    Run the audio through the EQ chain.

    `audio` is float32/float64, shape (n,) for mono or (n, channels) for
    multichannel. Filtering is applied down the time axis, so each channel is
    filtered independently by the same chain.

    Gain staging: boosting bands raises the peak level, so after filtering we
    measure the peak and attenuate if it would clip. We only ever attenuate —
    never make up gain — so an A/B against the original stays honest about
    what the EQ did.
    """
    sos = build_sos(gains, fs)

    audio = np.asarray(audio)
    if audio.size == 0:
        # sosfilt cannot reshape a zero-length array. The GUI never gets here
        # (audio_io rejects empty files), but eqcore is usable on its own and
        # should hand back an empty result rather than raise.
        return ProcessResult(
            audio=audio.astype(np.float32),
            applied_gain_db=0.0,
            peak_before=0.0,
            clipped=False,
        )

    # sosfilt operates along `axis`; time is axis 0 in both the mono and the
    # (frames, channels) layout soundfile gives us.
    filtered = signal.sosfilt(sos, audio, axis=0)
    filtered = np.asarray(filtered, dtype=np.float64)

    peak = float(np.max(np.abs(filtered))) if filtered.size else 0.0

    if peak > headroom_peak and peak > 0.0:
        scale = headroom_peak / peak
        filtered = filtered * scale
        applied_db = 20.0 * np.log10(scale)
        clipped = True
    else:
        applied_db = 0.0
        clipped = False

    return ProcessResult(
        audio=filtered.astype(np.float32),
        applied_gain_db=applied_db,
        peak_before=peak,
        clipped=clipped,
    )


# --------------------------------------------------------------------------
# Analysis helpers used by the plots
# --------------------------------------------------------------------------


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse to a single channel for display/analysis purposes."""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def octave_smooth(freqs: np.ndarray, power: np.ndarray, fraction: float = 12.0):
    """
    Fractional-octave smoothing of a power spectrum.

    Each output point is the average power over a band `1/fraction` of an
    octave wide centred on that frequency — the same constant-Q smoothing a
    hardware audio analyser applies. Without it, a harmonically rich signal
    produces a forest of individual partials and the before/after comparison
    is unreadable; the EQ's effect is broad (the narrowest band here is 0.585
    octaves) so 1/12-octave smoothing leaves it fully intact.
    """
    if fraction <= 0 or freqs.size < 3:
        return power

    ratio = 2.0 ** (1.0 / (2.0 * fraction))
    # Cumulative sum turns each band average into two lookups.
    csum = np.concatenate([[0.0], np.cumsum(power)])
    lo = np.searchsorted(freqs, freqs / ratio, side="left")
    hi = np.searchsorted(freqs, freqs * ratio, side="right")
    hi = np.maximum(hi, lo + 1)
    return (csum[hi] - csum[lo]) / (hi - lo)


def _largest_power_of_two_at_most(n: int) -> int:
    """The largest power of two <= n (0 for n < 1)."""
    return 1 << (int(n).bit_length() - 1) if n >= 1 else 0


def spectrum(
    audio: np.ndarray,
    fs: float,
    n_fft: int = 8192,
    max_seconds: float = 30.0,
    smooth_fraction: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Averaged FFT magnitude spectrum, in dB, on a linear frequency grid.

    Uses Welch's method (overlapping windowed segments, averaged) rather than
    one giant FFT: a single FFT of a whole song is dominated by noise between
    bins, while averaging gives a curve you can actually read the EQ changes
    off. The result is then 1/12-octave smoothed for display; pass
    smooth_fraction=0 for the raw estimate. Returns (frequencies, dB).
    """
    mono = to_mono(np.asarray(audio, dtype=np.float64))

    # Long files do not make the estimate meaningfully better, only slower.
    limit = int(max_seconds * fs)
    if mono.size > limit:
        mono = mono[:limit]

    # Rounded down to a power of two, and never longer than the signal. The
    # browser port (web/dsp.js) has a radix-2 FFT and can only use power-of-two
    # segments; matching that here keeps the two spectra on the same bin grid
    # instead of quietly diverging on short clips.
    nperseg = _largest_power_of_two_at_most(
        min(int(min(n_fft, max(256, mono.size))), mono.size)
    )
    if nperseg < 2:
        return np.zeros(1), np.full(1, -200.0)
    freqs, psd = signal.welch(
        mono,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        window="hann",
        scaling="spectrum",
    )

    psd = octave_smooth(freqs, psd, smooth_fraction)
    magnitude_db = 10.0 * np.log10(np.maximum(psd, 1e-20))
    return freqs, magnitude_db


def waveform_envelope(
    audio: np.ndarray, target_points: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Min/max envelope for drawing a waveform without plotting millions of points.

    Splitting into `target_points` buckets and taking each bucket's min and max
    preserves the visual shape of transients, which naive decimation destroys.
    Returns (min per bucket, max per bucket).
    """
    mono = to_mono(np.asarray(audio, dtype=np.float64))
    if mono.size == 0:
        return np.zeros(1), np.zeros(1)

    if mono.size <= target_points:
        return mono, mono

    bucket = mono.size // target_points
    usable = bucket * target_points
    reshaped = mono[:usable].reshape(target_points, bucket)
    return reshaped.min(axis=1), reshaped.max(axis=1)

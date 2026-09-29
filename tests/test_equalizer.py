"""
The cascade, the filtering path, gain staging and the analysis helpers.

Where possible a test measures the *processed audio* and compares it against
the analytically computed response, so the drawn curve and the sound cannot
drift apart without a failure here.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from eqcore import equalizer
from tests.conftest import FS, gain_db, sine

NO_LIMIT = 1e9  # disables the safety normaliser so raw gain is measurable


def designed_db(gains: dict[str, float], freq: float, fs: float = FS) -> float:
    sos = equalizer.build_sos(gains, fs)
    _, h = signal.sosfreqz(sos, worN=np.array([freq]), fs=fs)
    return float(20 * np.log10(abs(h[0])))


def only(key: str, gain: float) -> dict[str, float]:
    gains = equalizer.flat_gains()
    gains[key] = gain
    return gains


# --------------------------------------------------------------------------
# Cascade construction
# --------------------------------------------------------------------------


def test_flat_gains_produce_a_single_pass_through_section():
    sos = equalizer.build_sos(equalizer.flat_gains(), FS)
    assert sos.shape == (1, 6)
    assert np.allclose(sos[0], [1, 0, 0, 1, 0, 0])


def test_flat_bands_are_dropped_from_the_cascade():
    """A 0 dB biquad is the identity, so keeping it is wasted arithmetic."""
    assert equalizer.build_sos(only("mid", 6.0), FS).shape[0] == 1
    gains = equalizer.flat_gains()
    gains["mid"] = 6.0
    gains["bass"] = -3.0
    assert equalizer.build_sos(gains, FS).shape[0] == 2

    every = dict.fromkeys(equalizer.BAND_KEYS, 4.0)
    assert equalizer.build_sos(every, FS).shape[0] == len(equalizer.BANDS)


def test_gains_outside_the_slider_range_are_clamped():
    huge = designed_db(only("mid", 99.0), 1000.0)
    maxed = designed_db(only("mid", equalizer.GAIN_MAX_DB), 1000.0)
    assert huge == pytest.approx(maxed, abs=1e-9)


def test_unknown_band_keys_are_ignored():
    sos = equalizer.build_sos({"not_a_band": 12.0}, FS)
    assert sos.shape == (1, 6)


# --------------------------------------------------------------------------
# Measured behaviour
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "freq", "gain"),
    [
        ("mid", 1000.0, 12.0),
        ("mid", 1000.0, -12.0),
        ("bass", 122.0, 9.0),
        ("high_mid", 2828.0, 6.0),
        ("presence", 4899.0, -8.0),
    ],
)
def test_measured_gain_matches_the_designed_curve(key, freq, gain):
    gains = only(key, gain)
    x = sine(freq)
    y = equalizer.process(x, FS, gains, headroom_peak=NO_LIMIT).audio
    assert gain_db(y, x) == pytest.approx(designed_db(gains, freq), abs=0.15)


@pytest.mark.parametrize(
    ("key", "freq", "gain"),
    [("sub_bass", 25.0, 10.0), ("sub_bass", 25.0, -10.0), ("brilliance", 15000.0, 8.0)],
)
def test_shelves_reach_their_nominal_gain(key, freq, gain):
    gains = only(key, gain)
    x = sine(freq, seconds=4.0)
    y = equalizer.process(x, FS, gains, headroom_peak=NO_LIMIT).audio
    assert gain_db(y, x) == pytest.approx(designed_db(gains, freq), abs=0.3)
    # Well inside the shelf, the response must actually approach the nominal
    # gain rather than merely moving in the right direction.
    assert designed_db(gains, freq) == pytest.approx(gain, abs=1.5)


@pytest.mark.parametrize(
    ("probe", "limit"), [(1000.0, 1.0), (5000.0, 0.5), (12000.0, 0.5)]
)
def test_a_boosted_band_does_not_leak_across_the_spectrum(probe, limit):
    assert abs(designed_db(only("bass", 12.0), probe)) < limit


def test_flat_processing_is_bit_transparent():
    x = sine(1000.0)
    y = equalizer.process(x, FS, equalizer.flat_gains(), headroom_peak=NO_LIMIT).audio
    assert np.max(np.abs(y - x)) < 1e-6


def test_stereo_channels_are_filtered_independently():
    left, right = sine(120.0), sine(5000.0)
    stereo = np.stack([left, right], axis=1)
    out = equalizer.process(stereo, FS, only("bass", 12.0), headroom_peak=NO_LIMIT).audio

    assert out.shape == stereo.shape
    assert gain_db(out[:, 0], left) > 10.0      # 120 Hz sits in the boosted band
    assert abs(gain_db(out[:, 1], right)) < 0.5  # 5 kHz is untouched


def test_output_dtype_is_float32():
    out = equalizer.process(sine(440.0), FS, only("mid", 3.0))
    assert out.audio.dtype == np.float32


# --------------------------------------------------------------------------
# Gain staging
# --------------------------------------------------------------------------


def test_boosting_a_hot_signal_is_attenuated_into_headroom():
    loud = (0.95 * np.sin(2 * np.pi * 120 * np.arange(FS * 2) / FS)).astype(np.float32)
    result = equalizer.process(loud, FS, only("bass", 12.0))

    assert result.clipped
    assert result.applied_gain_db < 0
    assert np.max(np.abs(result.audio)) <= equalizer.HEADROOM_PEAK + 1e-6


def test_quiet_material_is_left_alone():
    """Gain staging only ever attenuates — it must never add make-up gain."""
    quiet = (0.05 * np.sin(2 * np.pi * 440 * np.arange(FS) / FS)).astype(np.float32)
    result = equalizer.process(quiet, FS, only("mid", 6.0))

    assert not result.clipped
    assert result.applied_gain_db == 0.0


def test_reported_attenuation_matches_what_was_applied():
    loud = (0.98 * np.sin(2 * np.pi * 120 * np.arange(FS) / FS)).astype(np.float32)
    result = equalizer.process(loud, FS, only("bass", 12.0))

    expected = equalizer.HEADROOM_PEAK / result.peak_before
    assert 10 ** (result.applied_gain_db / 20) == pytest.approx(expected, rel=1e-6)


def test_empty_input_does_not_crash():
    result = equalizer.process(np.zeros(0, dtype=np.float32), FS, only("mid", 6.0))
    assert result.audio.size == 0
    assert result.peak_before == 0.0


# --------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------


def test_frequency_response_is_the_response_of_the_real_cascade():
    gains = only("presence", 7.0)
    freqs, mag = equalizer.frequency_response(gains, FS, n_points=64)

    sos = equalizer.build_sos(gains, FS)
    _, h = signal.sosfreqz(sos, worN=freqs, fs=FS)
    assert np.allclose(mag, 20 * np.log10(np.abs(h)), atol=1e-9)


def test_frequency_response_stays_below_nyquist():
    _, mag = equalizer.frequency_response(equalizer.flat_gains(), 8000)
    freqs, _ = equalizer.frequency_response(equalizer.flat_gains(), 8000)
    assert freqs[-1] < 4000
    assert np.all(np.isfinite(mag))


def test_to_mono_averages_channels():
    stereo = np.stack([np.ones(10), np.zeros(10)], axis=1)
    assert np.allclose(equalizer.to_mono(stereo), 0.5)
    mono = np.ones(10)
    assert np.allclose(equalizer.to_mono(mono), mono)


def test_spectrum_shows_the_boost_where_it_was_applied():
    rng = np.random.default_rng(0)
    noise = (0.1 * rng.standard_normal(FS * 3)).astype(np.float32)
    eqd = equalizer.process(
        noise, FS, only("presence", 12.0), headroom_peak=NO_LIMIT
    ).audio

    f_before, s_before = equalizer.spectrum(noise, FS)
    _, s_after = equalizer.spectrum(eqd, FS)

    at = int(np.argmin(np.abs(f_before - 4899.0)))
    assert 9.0 < s_after[at] - s_before[at] < 14.0

    away = int(np.argmin(np.abs(f_before - 300.0)))
    assert abs(s_after[away] - s_before[away]) < 1.0


def test_spectrum_segment_length_is_a_power_of_two():
    """
    The browser port has a radix-2 FFT, so both sides round the Welch segment
    down to a power of two. If this drifts the two spectra land on different
    bin grids.
    """
    for n in (500, 5000, 50_000):
        freqs, _ = equalizer.spectrum(np.zeros(n, dtype=np.float32), FS)
        nperseg = 2 * (freqs.size - 1)
        assert nperseg & (nperseg - 1) == 0, f"nperseg {nperseg} is not a power of two"
        assert nperseg <= n


def test_spectrum_handles_a_signal_shorter_than_one_segment():
    freqs, db = equalizer.spectrum(np.zeros(7, dtype=np.float32), FS)
    assert freqs.size == db.size >= 1
    assert np.all(np.isfinite(db))


def test_octave_smooth_preserves_a_flat_spectrum():
    freqs = np.linspace(1.0, 20000.0, 2048)
    power = np.full_like(freqs, 3.0)
    assert np.allclose(equalizer.octave_smooth(freqs, power, 12.0), 3.0)


def test_octave_smooth_is_a_no_op_when_disabled():
    freqs = np.linspace(1.0, 1000.0, 64)
    power = np.random.default_rng(1).random(64)
    assert np.allclose(equalizer.octave_smooth(freqs, power, 0.0), power)


def test_waveform_envelope_brackets_the_signal():
    x = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 20_000))
    lo, hi = equalizer.waveform_envelope(x, target_points=200)

    assert lo.size == hi.size == 200
    assert np.all(lo <= hi)
    assert np.max(hi) == pytest.approx(np.max(x), abs=1e-3)
    assert np.min(lo) == pytest.approx(np.min(x), abs=1e-3)


def test_waveform_envelope_passes_short_signals_through():
    x = np.array([0.1, -0.2, 0.3])
    lo, hi = equalizer.waveform_envelope(x, target_points=2000)
    assert np.allclose(lo, x) and np.allclose(hi, x)


def test_waveform_envelope_handles_empty_input():
    lo, hi = equalizer.waveform_envelope(np.zeros(0))
    assert lo.size == hi.size == 1

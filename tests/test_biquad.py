"""
Coefficient-level tests for the Audio EQ Cookbook designs.

These assert the defining mathematical properties of each filter type rather
than comparing against stored numbers, so they stay meaningful if the
implementation is rewritten.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from eqcore import biquad, equalizer

FS = 48000
IDENTITY = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])


def mag_db(section: np.ndarray, freq: float, fs: float = FS) -> float:
    """|H| in dB at one frequency, for a single second-order section."""
    _, h = signal.sosfreqz(section.reshape(1, 6), worN=np.array([freq]), fs=fs)
    return float(20 * np.log10(abs(h[0])))


@pytest.mark.parametrize("gain", [-12.0, -6.0, -0.5, 0.5, 6.0, 12.0])
def test_peaking_hits_its_gain_exactly_at_the_centre(gain):
    section = biquad.peaking(1000.0, gain, 1.0, FS)
    assert mag_db(section, 1000.0) == pytest.approx(gain, abs=1e-9)


@pytest.mark.parametrize("gain", [-12.0, 6.0, 12.0])
def test_peaking_leaves_dc_and_nyquist_untouched(gain):
    """A bell may not shift the ends of the spectrum — that is a shelf's job."""
    section = biquad.peaking(1000.0, gain, 1.0, FS)
    assert mag_db(section, 0.0) == pytest.approx(0.0, abs=1e-9)
    assert mag_db(section, FS / 2) == pytest.approx(0.0, abs=1e-9)


def test_peaking_cut_is_the_exact_inverse_of_the_boost():
    """RBJ's peaking cut is reciprocal, so boost * cut is a perfect bypass."""
    freqs = np.logspace(np.log10(20), np.log10(FS / 2 * 0.99), 256)
    boost = biquad.peaking(1000.0, 9.0, 1.2, FS)
    cut = biquad.peaking(1000.0, -9.0, 1.2, FS)

    _, h = signal.sosfreqz(np.vstack([boost, cut]), worN=freqs, fs=FS)
    assert np.allclose(np.abs(h), 1.0, atol=1e-9)


def test_higher_q_makes_a_narrower_bell():
    wide = biquad.peaking(1000.0, 12.0, 0.5, FS)
    narrow = biquad.peaking(1000.0, 12.0, 4.0, FS)
    # Both peak at 12 dB; an octave away the narrow one has fallen much further.
    assert mag_db(wide, 2000.0) > mag_db(narrow, 2000.0) + 3.0


@pytest.mark.parametrize("gain", [-10.0, 10.0])
def test_low_shelf_lifts_dc_and_releases_nyquist(gain):
    section = biquad.low_shelf(200.0, gain, 0.707, FS)
    assert mag_db(section, 0.0) == pytest.approx(gain, abs=1e-6)
    assert mag_db(section, FS / 2) == pytest.approx(0.0, abs=1e-6)
    # Half gain at the corner is the defining property of an RBJ shelf.
    assert mag_db(section, 200.0) == pytest.approx(gain / 2, abs=0.25)


@pytest.mark.parametrize("gain", [-10.0, 10.0])
def test_high_shelf_lifts_nyquist_and_releases_dc(gain):
    section = biquad.high_shelf(6000.0, gain, 0.707, FS)
    assert mag_db(section, FS / 2) == pytest.approx(gain, abs=1e-6)
    assert mag_db(section, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert mag_db(section, 6000.0) == pytest.approx(gain / 2, abs=0.25)


@pytest.mark.parametrize("kind", ["low_shelf", "peaking", "high_shelf"])
def test_zero_gain_is_a_pass_through(kind):
    """
    At 0 dB every design collapses to H(z) = 1 — the numerator and denominator
    come out equal. The coefficients are not [1,0,0,1,0,0]; they are some
    [b, a] with b == a, which is the same filter. `build_sos` relies on this
    when it drops flat bands from the cascade.
    """
    section = biquad.design(kind, 1000.0, 0.0, 0.707, FS)
    b, a = section[:3], section[3:]
    assert np.allclose(b, a, atol=1e-12)

    freqs = np.logspace(np.log10(20), np.log10(FS / 2 * 0.99), 256)
    _, h = signal.sosfreqz(section.reshape(1, 6), worN=freqs, fs=FS)
    assert np.allclose(np.abs(h), 1.0, atol=1e-12)


def test_design_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown filter kind"):
        biquad.design("bandpass", 1000.0, 3.0, 1.0, FS)


def test_centre_frequency_is_clamped_below_nyquist():
    """
    Prewarping blows up as f0 approaches fs/2, so the design clamps. The
    clamped filter must still be stable and finite rather than producing NaN.
    """
    section = biquad.design("peaking", 40000.0, 12.0, 1.0, FS)
    assert np.all(np.isfinite(section))
    assert biquad.is_stable(section)


def test_is_stable_rejects_poles_outside_the_unit_circle():
    assert biquad.is_stable(IDENTITY)
    # a2 >= 1 puts at least one pole on or outside the unit circle.
    assert not biquad.is_stable(np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.5]))
    assert not biquad.is_stable(np.array([1.0, 0.0, 0.0, 1.0, -3.0, 0.5]))


@pytest.mark.slow
def test_every_band_is_stable_across_gains_and_sample_rates():
    unstable = [
        (band.key, gain, fs)
        for band in equalizer.BANDS
        for gain in np.arange(equalizer.GAIN_MIN_DB, equalizer.GAIN_MAX_DB + 0.5, 0.5)
        for fs in (8000, 16000, 22050, 44100, 48000, 96000, 192000)
        if not biquad.is_stable(
            biquad.design(band.kind, band.f0, float(gain), band.q, fs)
        )
    ]
    assert unstable == []

"""The shared spec loads, validates, and is the only definition of the bands."""

from __future__ import annotations

import json

import pytest

from eqcore import equalizer, presets, spec


def test_spec_file_exists_and_parses():
    assert spec.SPEC_PATH.exists(), f"{spec.SPEC_PATH} is missing"
    json.loads(spec.SPEC_PATH.read_text(encoding="utf-8"))


def test_constants_are_sane():
    assert spec.GAIN_MIN_DB < 0 < spec.GAIN_MAX_DB
    assert 0.0 < spec.HEADROOM_PEAK <= 1.0
    assert spec.MAX_DURATION_SECONDS > 0


def test_q_from_octaves_matches_the_closed_form():
    # Q = sqrt(2^N) / (2^N - 1); one octave is the textbook reference point.
    assert spec.q_from_octaves(1.0) == pytest.approx(2 ** 0.5, rel=1e-12)
    # Wider band -> lower Q -> gentler bell.
    assert spec.q_from_octaves(2.0) < spec.q_from_octaves(1.0)
    assert spec.q_from_octaves(0.5) > spec.q_from_octaves(1.0)


def test_bands_are_built_from_the_spec_in_order():
    entries = spec.band_entries()
    assert len(equalizer.BANDS) == len(entries)
    for band, entry in zip(equalizer.BANDS, entries, strict=True):
        assert band.key == entry["key"]
        assert band.kind == entry["kind"]
        assert band.f0 == pytest.approx(entry["f0"])
        assert band.q == pytest.approx(spec.band_q(entry))


def test_band_centres_ascend():
    """A cascade whose bands jump around would be a spec typo."""
    f0s = [b.f0 for b in equalizer.BANDS]
    assert f0s == sorted(f0s)


def test_shelves_sit_at_the_extremes():
    assert equalizer.BANDS[0].kind == "low_shelf"
    assert equalizer.BANDS[-1].kind == "high_shelf"
    assert all(b.kind == "peaking" for b in equalizer.BANDS[1:-1])


def test_band_q_rejects_ambiguous_declarations():
    with pytest.raises(ValueError, match="exactly one of"):
        spec.band_q({"key": "x", "q": 0.7, "bandwidth_octaves": 1.0})
    with pytest.raises(ValueError, match="exactly one of"):
        spec.band_q({"key": "x"})


def test_every_preset_covers_every_band():
    for name in presets.PRESET_NAMES:
        assert set(presets.get(name)) == set(equalizer.BAND_KEYS)


def test_every_preset_has_a_note():
    for name in presets.PRESET_NAMES:
        assert presets.PRESET_NOTES.get(name), f"{name} has no note"


def test_preset_gains_are_inside_the_slider_range():
    for name in presets.PRESET_NAMES:
        for key, value in presets.get(name).items():
            assert equalizer.GAIN_MIN_DB <= value <= equalizer.GAIN_MAX_DB, (
                f"{name}.{key} = {value} is outside the slider range"
            )


def test_first_preset_is_flat_so_the_default_is_a_pass_through():
    first = presets.PRESET_NAMES[0]
    assert all(v == 0.0 for v in presets.get(first).values())


def test_get_returns_a_copy():
    a = presets.get("Bass Boost")
    a["bass"] = 99.0
    assert presets.get("Bass Boost")["bass"] != 99.0

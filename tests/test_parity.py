"""
Guards against the two implementations drifting apart.

The band table and presets live in shared/eq_spec.json; `web/spec.js` is
generated from it and committed. These tests fail if anyone edits one without
regenerating the other, which is the failure mode that used to be silent —
a preset changed in Python while the deployed site kept serving the old value.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

from eqcore import equalizer, presets, spec

WEB_AUDIO_KIND = {
    "low_shelf": "lowshelf",
    "peaking": "peaking",
    "high_shelf": "highshelf",
}


def run_tool(root, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(root / "tools" / args[0]), *args[1:]],
        capture_output=True, text=True, cwd=root, check=False,
    )


# --------------------------------------------------------------------------
# Generated files are up to date
# --------------------------------------------------------------------------


def test_web_spec_is_not_stale(root):
    result = run_tool(root, "gen_web_spec.py", "--check")
    assert result.returncode == 0, (
        f"web/spec.js does not match shared/eq_spec.json.\n"
        f"Run: python tools/gen_web_spec.py\n\n{result.stderr}"
    )


def test_golden_files_are_not_stale(root):
    result = run_tool(root, "gen_golden.py", "--check")
    assert result.returncode == 0, (
        f"tests/golden/*.json no longer match the Python implementation.\n"
        f"Run: python tools/gen_golden.py\n\n{result.stderr}"
    )


def test_generated_web_spec_carries_the_do_not_edit_banner(root):
    text = (root / "web" / "spec.js").read_text(encoding="utf-8")
    assert "GENERATED FILE — DO NOT EDIT" in text


def test_no_band_or_preset_data_is_hand_written_in_dsp_js(root):
    """
    web/dsp.js must take every one of these from spec.js. Re-introducing a
    literal band key or preset name here is how the copies came back last time.
    """
    text = (root / "web" / "dsp.js").read_text(encoding="utf-8")
    for needle in ("sub_bass", "brilliance", "Bass Boost", "qFromOctaves"):
        assert needle not in text, (
            f"web/dsp.js contains {needle!r} — band and preset data belong in "
            f"shared/eq_spec.json, not in this module"
        )


# --------------------------------------------------------------------------
# The two languages actually agree, value by value
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def web_spec(root):
    """web/spec.js, evaluated by Node and handed back as JSON."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; skipping the cross-language check")

    script = """
      import * as s from './web/spec.js';
      process.stdout.write(JSON.stringify({
        GAIN_MIN_DB: s.GAIN_MIN_DB,
        GAIN_MAX_DB: s.GAIN_MAX_DB,
        HEADROOM_PEAK: s.HEADROOM_PEAK,
        MAX_DURATION_SECONDS: s.MAX_DURATION_SECONDS,
        BANDS: s.BANDS,
        BAND_KEYS: s.BAND_KEYS,
        PRESETS: s.PRESETS,
        PRESET_NAMES: s.PRESET_NAMES,
        PRESET_NOTES: s.PRESET_NOTES,
      }));
    """
    result = subprocess.run(
        [node, "--input-type=module", "-e", script],
        capture_output=True, text=True, cwd=root, check=False,
    )
    assert result.returncode == 0, f"could not evaluate web/spec.js:\n{result.stderr}"
    return json.loads(result.stdout)


def test_limits_agree(web_spec):
    assert web_spec["GAIN_MIN_DB"] == equalizer.GAIN_MIN_DB
    assert web_spec["GAIN_MAX_DB"] == equalizer.GAIN_MAX_DB
    assert web_spec["HEADROOM_PEAK"] == equalizer.HEADROOM_PEAK


def test_the_browser_duration_cap_is_not_looser_than_the_python_one(web_spec):
    """
    The browser renders the whole clip into memory to export it, so its ceiling
    must be at or below the desktop app's — never above.
    """
    assert 0 < web_spec["MAX_DURATION_SECONDS"] <= spec.MAX_DURATION_SECONDS


def test_band_tables_agree(web_spec):
    assert web_spec["BAND_KEYS"] == list(equalizer.BAND_KEYS)

    for band, web in zip(equalizer.BANDS, web_spec["BANDS"], strict=True):
        assert web["key"] == band.key
        assert web["name"] == band.name
        assert web["kind"] == WEB_AUDIO_KIND[band.kind], (
            f"{band.key}: {band.kind} should map to "
            f"{WEB_AUDIO_KIND[band.kind]!r}, got {web['kind']!r}"
        )
        assert web["f0"] == pytest.approx(band.f0, rel=0, abs=0)
        assert web["q"] == pytest.approx(band.q, rel=0, abs=0)
        assert web["span"] == band.span


def test_preset_tables_agree(web_spec):
    assert web_spec["PRESET_NAMES"] == list(presets.PRESET_NAMES)

    for name in presets.PRESET_NAMES:
        py = presets.get(name)
        web = web_spec["PRESETS"][name]
        assert set(web) == set(py), f"{name}: different band coverage"
        for key, value in py.items():
            assert web[key] == value, f"{name}.{key}: python={value} js={web[key]}"


def test_preset_notes_agree(web_spec):
    for name in presets.PRESET_NAMES:
        assert web_spec["PRESET_NOTES"][name] == presets.PRESET_NOTES[name]

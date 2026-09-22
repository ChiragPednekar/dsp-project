"""
Built-in EQ presets.

Each preset is just a dict of band-key -> gain in dB, so selecting one is the
same operation as dragging the sliders there by hand. Anything a preset can
do, the user can do manually, and vice versa.
"""

from __future__ import annotations

from .equalizer import BAND_KEYS, flat_gains

#: Order matters — this is the order of the dropdown, and "Flat / Reset"
#: is first so it is the default selection.
PRESETS: dict[str, dict[str, float]] = {
    "Flat / Reset": flat_gains(),
    "Bass Boost": {
        "sub_bass": 8.0,
        "bass": 6.0,
        "low_mid": 1.5,
        "mid": 0.0,
        "high_mid": 0.0,
        "presence": 0.0,
        "brilliance": 1.5,
    },
    "Vocal Boost": {
        "sub_bass": -4.0,
        "bass": -2.0,
        "low_mid": -1.0,
        "mid": 3.5,
        "high_mid": 5.0,
        "presence": 4.0,
        "brilliance": 1.0,
    },
    "Treble Boost": {
        "sub_bass": 0.0,
        "bass": -1.0,
        "low_mid": -1.0,
        "mid": 0.0,
        "high_mid": 3.0,
        "presence": 5.0,
        "brilliance": 7.5,
    },
    "Loudness (V-shape)": {
        "sub_bass": 6.0,
        "bass": 4.5,
        "low_mid": -1.0,
        "mid": -3.0,
        "high_mid": -1.0,
        "presence": 3.5,
        "brilliance": 6.0,
    },
    "Podcast / Speech": {
        "sub_bass": -10.0,
        "bass": -5.0,
        "low_mid": -2.0,
        "mid": 2.5,
        "high_mid": 4.0,
        "presence": 3.0,
        "brilliance": -1.0,
    },
}

PRESET_NAMES: tuple[str, ...] = tuple(PRESETS)

#: Short explanation shown under the dropdown so the choice is legible.
PRESET_NOTES: dict[str, str] = {
    "Flat / Reset": "All bands at 0 dB — the filter chain becomes a pass-through.",
    "Bass Boost": "Low-shelf lift under 60 Hz plus a bell at 122 Hz for weight and punch.",
    "Vocal Boost": "Cuts rumble, lifts 1–5 kHz where speech intelligibility lives.",
    "Treble Boost": "High-shelf air above 8 kHz with a presence lift for detail.",
    "Loudness (V-shape)": "Boosts both extremes and scoops the mids — the classic smiley curve.",
    "Podcast / Speech": "Steep low-end cut to kill room rumble, forward upper mids.",
}


def get(name: str) -> dict[str, float]:
    """
    Return a copy of a preset's gains, filled out for every band.

    Copying matters: the caller writes these straight into widget state, and a
    shared dict would let one session's slider drag mutate the preset table.
    """
    base = flat_gains()
    base.update(PRESETS.get(name, {}))
    return {key: float(base.get(key, 0.0)) for key in BAND_KEYS}

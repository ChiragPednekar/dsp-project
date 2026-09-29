"""
Built-in EQ presets.

Each preset is just a dict of band-key -> gain in dB, so selecting one is the
same operation as dragging the sliders there by hand. Anything a preset can
do, the user can do manually, and vice versa.

The values live in `shared/eq_spec.json`, which the browser build is generated
from as well — edit them there, not here.
"""

from __future__ import annotations

from . import spec
from .equalizer import BAND_KEYS, flat_gains

_ENTRIES = spec.preset_entries()


def _filled(gains: dict[str, float]) -> dict[str, float]:
    """A preset's gains, expanded to cover every band."""
    base = flat_gains()
    base.update({key: float(value) for key, value in gains.items()})
    return {key: float(base[key]) for key in BAND_KEYS}


#: Order matters — this is the order of the dropdown, and "Flat / Reset"
#: is first so it is the default selection.
PRESETS: dict[str, dict[str, float]] = {
    entry["name"]: _filled(entry.get("gains", {})) for entry in _ENTRIES
}

PRESET_NAMES: tuple[str, ...] = tuple(PRESETS)

#: Short explanation shown under the dropdown so the choice is legible.
PRESET_NOTES: dict[str, str] = {
    entry["name"]: entry.get("note", "") for entry in _ENTRIES
}


def get(name: str) -> dict[str, float]:
    """
    Return a copy of a preset's gains, filled out for every band.

    Copying matters: the caller writes these straight into widget state, and a
    shared dict would let one session's slider drag mutate the preset table.
    """
    return dict(PRESETS.get(name, flat_gains()))

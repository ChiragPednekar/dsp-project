"""
Loads the shared EQ specification.

`shared/eq_spec.json` is the single source of truth for the band layout, the
presets and the limits. This module reads it for the Python side;
`tools/gen_web_spec.py` generates `web/spec.js` from the same file for the
browser side. Neither front end keeps its own copy of these numbers, so a
preset edited here cannot silently disagree with the deployed site.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

#: Repo-relative location of the spec. Resolved from this file so the app
#: works regardless of the working directory it was launched from.
SPEC_PATH = Path(__file__).resolve().parent.parent / "shared" / "eq_spec.json"


def q_from_octaves(bandwidth_octaves: float) -> float:
    """
    Convert a bandwidth in octaves to the Q a peaking biquad needs.

        Q = sqrt(2^N) / (2^N - 1)

    A wide band (2 octaves) gives a low Q and a gentle bell; a narrow band
    (half an octave) gives a high Q and a sharp one. Using this instead of
    hand-picked Q values is what makes the bands tile the spectrum evenly.
    """
    ratio = 2.0 ** float(bandwidth_octaves)
    return float(math.sqrt(ratio) / (ratio - 1.0))


def _load() -> dict[str, Any]:
    try:
        with SPEC_PATH.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:  # pragma: no cover - an install-layout problem
        raise RuntimeError(
            f"The EQ specification is missing at {SPEC_PATH}. It ships with the "
            "repository; eqcore cannot define its bands without it."
        ) from None


SPEC: dict[str, Any] = _load()

_CONSTANTS: dict[str, float] = SPEC["constants"]

GAIN_MIN_DB: float = float(_CONSTANTS["gain_min_db"])
GAIN_MAX_DB: float = float(_CONSTANTS["gain_max_db"])
HEADROOM_PEAK: float = float(_CONSTANTS["headroom_peak"])
MAX_DURATION_SECONDS: int = int(_CONSTANTS["max_duration_seconds"])

VALID_KINDS = ("low_shelf", "peaking", "high_shelf")


def band_q(entry: dict[str, Any]) -> float:
    """
    The Q for one spec entry.

    Shelves state `q` outright; bells state `bandwidth_octaves` and have their
    Q derived. Exactly one of the two must be present.
    """
    has_q = "q" in entry
    has_bw = "bandwidth_octaves" in entry

    if has_q == has_bw:
        raise ValueError(
            f"band {entry.get('key')!r} must declare exactly one of "
            f"'q' or 'bandwidth_octaves'"
        )
    return float(entry["q"]) if has_q else q_from_octaves(entry["bandwidth_octaves"])


def band_entries() -> list[dict[str, Any]]:
    """The raw band entries, validated."""
    entries = SPEC["bands"]
    seen: set[str] = set()

    for entry in entries:
        key = entry.get("key")
        if not key:
            raise ValueError("every band needs a 'key'")
        if key in seen:
            raise ValueError(f"duplicate band key {key!r}")
        seen.add(key)

        if entry["kind"] not in VALID_KINDS:
            raise ValueError(
                f"band {key!r} has kind {entry['kind']!r}; "
                f"expected one of {VALID_KINDS}"
            )
        if float(entry["f0"]) <= 0:
            raise ValueError(f"band {key!r} has a non-positive f0")
        band_q(entry)  # raises if the Q declaration is malformed

    return list(entries)


def preset_entries() -> list[dict[str, Any]]:
    """The raw preset entries, validated against the band keys."""
    keys = {entry["key"] for entry in SPEC["bands"]}
    entries = SPEC["presets"]

    for entry in entries:
        unknown = set(entry.get("gains", {})) - keys
        if unknown:
            raise ValueError(
                f"preset {entry['name']!r} sets unknown band(s): {sorted(unknown)}"
            )
    return list(entries)

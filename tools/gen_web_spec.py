"""
Generates web/spec.js from shared/eq_spec.json.

The browser build cannot read the JSON directly — Vercel serves only `web/`,
and an extra fetch would delay the first paint — so the band table, presets
and limits are baked into a generated module instead. Running this script is
the only supported way to change them.

Run:   python tools/gen_web_spec.py
Check: python tools/gen_web_spec.py --check     (exit 1 if web/spec.js is stale)

tests/test_spec_parity.py runs the --check form, so CI fails if someone edits
the spec without regenerating.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eqcore import spec  # noqa: E402

OUTPUT = ROOT / "web" / "spec.js"

#: Python kind names -> the BiquadFilterNode `type` strings they correspond to.
WEB_AUDIO_KIND = {
    "low_shelf": "lowshelf",
    "peaking": "peaking",
    "high_shelf": "highshelf",
}

HEADER = """\
/**
 * GENERATED FILE — DO NOT EDIT.
 *
 * Produced from shared/eq_spec.json by tools/gen_web_spec.py, which is the
 * single source of truth this and the Python package both read. Editing this
 * file by hand will be overwritten, and CI (tests/test_spec_parity.py) fails
 * if it is out of date with the JSON.
 *
 * `kind` values are Web Audio BiquadFilterNode types. For the two shelves Web
 * Audio ignores `.Q` and uses shelf slope S = 1, which works out to exactly
 * the alpha = sin(w0)/sqrt(2) that a Q of 0.707 gives in the Audio EQ
 * Cookbook formulas -- so the shelves match the Python implementation
 * coefficient for coefficient.
 */

"""


def js_number(value: float) -> str:
    """Shortest round-trip literal. Python and JS both use IEEE-754 doubles."""
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def js_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def render() -> str:
    out = [HEADER]

    out.append(f"export const GAIN_MIN_DB = {js_number(spec.GAIN_MIN_DB)};\n")
    out.append(f"export const GAIN_MAX_DB = {js_number(spec.GAIN_MAX_DB)};\n\n")

    out.append(
        "/** Peak the processed signal is limited to. "
        "~0.9 dB below full scale. */\n"
    )
    out.append(f"export const HEADROOM_PEAK = {js_number(spec.HEADROOM_PEAK)};\n\n")

    out.append(
        "/**\n"
        " * Duration ceiling for the browser, lower than the Python app's.\n"
        " * Exporting renders the whole clip into memory at once, so a long file\n"
        " * would cost hundreds of megabytes and crash the tab on a phone.\n"
        " */\n"
    )
    out.append(
        "export const MAX_DURATION_SECONDS = "
        f"{js_number(spec.SPEC['constants']['browser_max_duration_seconds'])};\n\n"
    )

    out.append("/** The seven bands, in cascade order. */\n")
    out.append("export const BANDS = [\n")
    for entry in spec.band_entries():
        out.append(
            "  {{ key: {key}, name: {name}, kind: {kind}, "
            "f0: {f0}, q: {q}, span: {span} }},\n".format(
                key=js_string(entry["key"]),
                name=js_string(entry["name"]),
                kind=js_string(WEB_AUDIO_KIND[entry["kind"]]),
                f0=js_number(entry["f0"]),
                q=js_number(spec.band_q(entry)),
                span=js_string(entry["span"]),
            )
        )
    out.append("];\n\n")

    out.append("export const BAND_KEYS = BANDS.map((b) => b.key);\n\n")

    out.append("/** A neutral setting: every band at 0 dB. */\n")
    out.append("export function flatGains() {\n")
    out.append("  return Object.fromEntries(BAND_KEYS.map((k) => [k, 0.0]));\n")
    out.append("}\n\n")

    presets = spec.preset_entries()
    keys = [entry["key"] for entry in spec.band_entries()]

    out.append("export const PRESETS = {\n")
    for entry in presets:
        gains = entry.get("gains", {})
        pairs = ", ".join(
            f"{key}: {js_number(gains.get(key, 0.0))}" for key in keys
        )
        out.append(f"  {js_string(entry['name'])}: {{ {pairs} }},\n")
    out.append("};\n\n")

    out.append("export const PRESET_NAMES = Object.keys(PRESETS);\n\n")

    out.append("export const PRESET_NOTES = {\n")
    for entry in presets:
        out.append(
            f"  {js_string(entry['name'])}: {js_string(entry.get('note', ''))},\n"
        )
    out.append("};\n")

    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed file matches the spec instead of writing it",
    )
    args = parser.parse_args()

    rendered = render()

    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current == rendered:
            print(f"{OUTPUT.relative_to(ROOT)} is up to date.")
            return 0
        print(
            f"{OUTPUT.relative_to(ROOT)} is STALE — it does not match "
            f"shared/eq_spec.json.\nRun: python tools/gen_web_spec.py",
            file=sys.stderr,
        )
        return 1

    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

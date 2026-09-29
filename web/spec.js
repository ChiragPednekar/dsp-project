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

export const GAIN_MIN_DB = -12;
export const GAIN_MAX_DB = 12;

/** Peak the processed signal is limited to. ~0.9 dB below full scale. */
export const HEADROOM_PEAK = 0.9;

/**
 * Duration ceiling for the browser, lower than the Python app's.
 * Exporting renders the whole clip into memory at once, so a long file
 * would cost hundreds of megabytes and crash the tab on a phone.
 */
export const MAX_DURATION_SECONDS = 420;

/** The seven bands, in cascade order. */
export const BANDS = [
  { key: 'sub_bass', name: 'Sub-bass', kind: 'lowshelf', f0: 60, q: 0.707, span: '20 – 60 Hz' },
  { key: 'bass', name: 'Bass', kind: 'peaking', f0: 122, q: 0.6441995201303671, span: '60 – 250 Hz' },
  { key: 'low_mid', name: 'Low-mid', kind: 'peaking', f0: 354, q: 1.4142135623730951, span: '250 – 500 Hz' },
  { key: 'mid', name: 'Mid', kind: 'peaking', f0: 1000, q: 0.6666666666666666, span: '500 Hz – 2 kHz' },
  { key: 'high_mid', name: 'High-mid', kind: 'peaking', f0: 2828, q: 1.4142135623730951, span: '2 – 4 kHz' },
  { key: 'presence', name: 'Presence', kind: 'peaking', f0: 4899, q: 2.4493305818946345, span: '4 – 6 kHz' },
  { key: 'brilliance', name: 'Brilliance', kind: 'highshelf', f0: 8000, q: 0.707, span: '6 – 20 kHz' },
];

export const BAND_KEYS = BANDS.map((b) => b.key);

/** A neutral setting: every band at 0 dB. */
export function flatGains() {
  return Object.fromEntries(BAND_KEYS.map((k) => [k, 0.0]));
}

export const PRESETS = {
  'Flat / Reset': { sub_bass: 0, bass: 0, low_mid: 0, mid: 0, high_mid: 0, presence: 0, brilliance: 0 },
  'Bass Boost': { sub_bass: 8, bass: 6, low_mid: 1.5, mid: 0, high_mid: 0, presence: 0, brilliance: 1.5 },
  'Vocal Boost': { sub_bass: -4, bass: -2, low_mid: -1, mid: 3.5, high_mid: 5, presence: 4, brilliance: 1 },
  'Treble Boost': { sub_bass: 0, bass: -1, low_mid: -1, mid: 0, high_mid: 3, presence: 5, brilliance: 7.5 },
  'Loudness (V-shape)': { sub_bass: 6, bass: 4.5, low_mid: -1, mid: -3, high_mid: -1, presence: 3.5, brilliance: 6 },
  'Podcast / Speech': { sub_bass: -10, bass: -5, low_mid: -2, mid: 2.5, high_mid: 4, presence: 3, brilliance: -1 },
};

export const PRESET_NAMES = Object.keys(PRESETS);

export const PRESET_NOTES = {
  'Flat / Reset': 'All bands at 0 dB — the filter chain becomes a pass-through.',
  'Bass Boost': 'Low-shelf lift under 60 Hz plus a bell at 122 Hz for weight and punch.',
  'Vocal Boost': 'Cuts rumble, lifts 1–5 kHz where speech intelligibility lives.',
  'Treble Boost': 'High-shelf air above 8 kHz with a presence lift for detail.',
  'Loudness (V-shape)': 'Boosts both extremes and scoops the mids — the classic smiley curve.',
  'Podcast / Speech': 'Steep low-end cut to kill room rumble, forward upper mids.',
};

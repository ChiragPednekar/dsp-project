/**
 * Shared helpers for the JavaScript tests.
 *
 * buildSignal() must stay in lockstep with build_signal() in
 * tools/gen_golden.py — the whole point is that both languages construct the
 * identical samples so the analysis can be compared directly.
 */

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN_DIR = join(HERE, '..', 'golden');

export function golden(name) {
  return JSON.parse(readFileSync(join(GOLDEN_DIR, name), 'utf8'));
}

/** Port of tools/gen_golden.py build_signal(). */
export function buildSignal(spec) {
  const { samples: n, fs, tones, dc, ramp } = spec;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let v = dc + ramp * (i / n);
    for (const [freq, amp] of tones) {
      v += amp * Math.sin((2.0 * Math.PI * freq * i) / fs);
    }
    out[i] = v;
  }
  return out;
}

/** Largest absolute difference between two same-length sequences. */
export function maxAbsDiff(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > worst) worst = d;
  }
  return worst;
}

/**
 * Synthesises the demo clip in the browser — a port of tools/make_sample.py.
 *
 * Generating it here rather than shipping samples/demo.wav keeps ~8 MB off the
 * deployment. The clip deliberately has energy in every band the equaliser
 * touches, so moving any one slider produces an audible and visible change:
 *
 *   * a bass line an octave apart from a sustained pad
 *   * a mid-range melody
 *   * hi-hat-ish noise bursts for the presence/brilliance bands
 *   * a low broadband bed so the spectrum plot has a floor to read
 */

const FS = 44100;
const DURATION = 45.0;

/** Deterministic PRNG so the demo clip is identical on every load. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller: standard normal from the uniform PRNG above. */
function makeGaussian(rand) {
  let spare = null;
  return function () {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    let s = 0;
    do {
      u = rand() * 2 - 1;
      v = rand() * 2 - 1;
      s = u * u + v * v;
    } while (s === 0 || s >= 1);
    const mul = Math.sqrt((-2.0 * Math.log(s)) / s);
    spare = v * mul;
    return u * mul;
  };
}

/** How many times to loop `pattern` (one note every `step` s) to fill DURATION. */
function repeats(patternLength, step) {
  return Math.floor(DURATION / (patternLength * step)) + 1;
}

/** A simple amplitude envelope so notes have transients to look at. */
function adsr(n, attack = 0.01, decay = 0.1, sustain = 0.7, release = 0.2) {
  const env = new Float64Array(n);
  let a = Math.min(Math.floor(attack * FS), n);
  let d = Math.min(Math.floor(decay * FS), Math.max(n - a, 0));
  let r = Math.min(Math.floor(release * FS), Math.max(n - a - d, 0));
  const s = Math.max(n - a - d - r, 0);

  let i = 0;
  for (let k = 0; k < a; k++, i++) env[i] = k / a;
  for (let k = 0; k < d; k++, i++) env[i] = 1 + (sustain - 1) * (k / d);
  for (let k = 0; k < s; k++, i++) env[i] = sustain;
  for (let k = 0; k < r; k++, i++) env[i] = sustain * (1 - k / Math.max(r - 1, 1));
  return env;
}

/** A harmonically rich tone — richer than a sine, so the EQ has something to grip. */
function note(freq, length, amp, harmonics = 4) {
  const n = Math.floor(length * FS);
  const wave = new Float64Array(n);
  for (let h = 1; h <= harmonics; h++) {
    const f = freq * h;
    if (f >= FS / 2) break;
    const w = (2 * Math.PI * f) / FS;
    const scale = 1.0 / h;
    for (let i = 0; i < n; i++) wave[i] += scale * Math.sin(w * i);
  }
  const env = adsr(n);
  for (let i = 0; i < n; i++) wave[i] *= env[i] * amp;
  return wave;
}

function addAt(mix, wave, atSample) {
  const end = Math.min(atSample + wave.length, mix.length);
  for (let i = atSample; i < end; i++) mix[i] += wave[i - atSample];
}

/**
 * Render the demo clip into an AudioBuffer on the supplied context.
 * Stereo, 45 s, 44.1 kHz, peak-normalised to 0.72.
 */
export function buildDemoBuffer(ctx) {
  const total = Math.floor(DURATION * FS);
  const mix = new Float64Array(total);

  const rand = mulberry32(7);
  const gauss = makeGaussian(rand);

  // Bass line — sits in the sub-bass and bass bands.
  const bassPattern = [55.0, 55.0, 73.42, 82.41, 55.0, 55.0, 98.0, 82.41];
  const bassCount = bassPattern.length * repeats(bassPattern.length, 0.5);
  for (let i = 0; i < bassCount; i++) {
    const start = i * 0.5;
    if (start >= DURATION) break;
    addAt(mix, note(bassPattern[i % bassPattern.length], 0.45, 0.30, 6),
          Math.floor(start * FS));
  }

  // Mid melody — low-mid through high-mid.
  const melody = [440.0, 523.25, 659.25, 523.25, 587.33, 440.0, 392.0, 440.0];
  const melodyCount = melody.length * repeats(melody.length, 0.5);
  for (let i = 0; i < melodyCount; i++) {
    const start = 0.25 + i * 0.5;
    if (start >= DURATION) break;
    addAt(mix, note(melody[i % melody.length], 0.4, 0.16, 5),
          Math.floor(start * FS));
  }

  // Sustained pad for continuous mid energy.
  const pad = [[220.0, 0.05], [277.18, 0.04], [329.63, 0.04]];
  for (const [f, a] of pad) {
    const w = (2 * Math.PI * f) / FS;
    const wLfo = (2 * Math.PI * 0.25) / FS;
    for (let i = 0; i < total; i++) {
      mix[i] += a * Math.sin(w * i) * (0.6 + 0.4 * Math.sin(wLfo * i));
    }
  }

  // Hats — differentiated (high-passed) noise bursts feeding presence and brilliance.
  const burstLen = Math.floor(0.06 * FS);
  for (let i = 0; i < Math.floor(DURATION * 4); i++) {
    const at = Math.floor(i * 0.25 * FS);
    if (at + burstLen > total) break;
    const amp = i % 4 === 0 ? 0.10 : 0.05;
    let prev = 0.0;
    for (let j = 0; j < burstLen; j++) {
      const cur = gauss();
      const hp = cur - prev;          // one-pole high-pass, so the burst is bright
      prev = cur;
      mix[at + j] += hp * Math.exp(-(9 * j) / burstLen) * amp;
    }
  }

  // Low-level broadband bed so every band has a readable noise floor.
  for (let i = 0; i < total; i++) mix[i] += 0.006 * gauss();

  // Fade the very ends to avoid a click on loop.
  const fade = Math.floor(0.05 * FS);
  for (let i = 0; i < fade; i++) {
    mix[i] *= i / fade;
    mix[total - 1 - i] *= i / fade;
  }

  let peak = 0;
  for (let i = 0; i < total; i++) peak = Math.max(peak, Math.abs(mix[i]));
  if (peak > 0) {
    const k = 0.72 / peak;
    for (let i = 0; i < total; i++) mix[i] *= k;
  }

  // Stereo with a slight width, so per-channel filtering is exercised too.
  const buffer = ctx.createBuffer(2, total, FS);
  const left = buffer.getChannelData(0);
  const right = buffer.getChannelData(1);
  const shift = 180;
  for (let i = 0; i < total; i++) {
    left[i] = mix[i];
    right[i] = mix[(i - shift + total) % total] * 0.97;
  }
  return buffer;
}

export const DEMO_NAME = 'demo.wav';

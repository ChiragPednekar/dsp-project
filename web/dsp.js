/**
 * DSP core — the browser counterpart of the Python `eqcore` package.
 *
 * The band table, Q derivation, preset values and gain staging below are a
 * direct port of eqcore/equalizer.py and eqcore/presets.py. The biquads
 * themselves are not reimplemented here: Web Audio's BiquadFilterNode already
 * implements the same Audio EQ Cookbook (Robert Bristow-Johnson) formulas that
 * eqcore/biquad.py spells out, so `app.js` builds the chain from native nodes
 * and this module supplies the parameters they are driven with.
 *
 * Analysis (Welch spectrum, fractional-octave smoothing, waveform envelope) is
 * ported here because the browser has no SciPy.
 */

// Note: `export ... from` below re-exports without binding these names
// locally, so anything this module *uses* must also be imported here.
import {
  BAND_KEYS,
  GAIN_MAX_DB,
  GAIN_MIN_DB,
  HEADROOM_PEAK,
  PRESETS,
  flatGains,
} from './spec.js';

// The band table, presets and limits are GENERATED from shared/eq_spec.json
// into web/spec.js — the same file eqcore reads. Nothing in this module may
// redefine them; they are re-exported so callers still import one module.
export {
  GAIN_MIN_DB,
  GAIN_MAX_DB,
  HEADROOM_PEAK,
  MAX_DURATION_SECONDS,
  BANDS,
  BAND_KEYS,
  flatGains,
  PRESETS,
  PRESET_NAMES,
  PRESET_NOTES,
} from './spec.js';

export function getPreset(name) {
  const base = flatGains();
  Object.assign(base, PRESETS[name] ?? {});
  return Object.fromEntries(BAND_KEYS.map((k) => [k, Number(base[k] ?? 0.0)]));
}

export function clampGain(db) {
  return Math.min(GAIN_MAX_DB, Math.max(GAIN_MIN_DB, db));
}

// ---------------------------------------------------------------------------
// FFT
// ---------------------------------------------------------------------------

/**
 * In-place iterative radix-2 Cooley-Tukey FFT.
 * `re` and `im` are Float64Array of the same power-of-two length.
 */
function fftInPlace(re, im) {
  const n = re.length;

  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1.0;
      let curIm = 0.0;
      const half = len >> 1;
      for (let k = 0; k < half; k++) {
        const uRe = re[i + k];
        const uIm = im[i + k];
        const vRe = re[i + k + half] * curRe - im[i + k + half] * curIm;
        const vIm = re[i + k + half] * curIm + im[i + k + half] * curRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + half] = uRe - vRe;
        im[i + k + half] = uIm - vIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}

function largestPowerOfTwoAtMost(n) {
  let p = 1;
  while (p * 2 <= n) p *= 2;
  return p;
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

/** Collapse to a single channel for display/analysis purposes. */
export function toMono(channels) {
  if (channels.length === 1) return channels[0];
  const n = channels[0].length;
  const out = new Float64Array(n);
  for (let c = 0; c < channels.length; c++) {
    const ch = channels[c];
    for (let i = 0; i < n; i++) out[i] += ch[i];
  }
  for (let i = 0; i < n; i++) out[i] /= channels.length;
  return out;
}

/**
 * Fractional-octave smoothing of a power spectrum.
 *
 * Each output point is the average power over a band `1/fraction` of an octave
 * wide centred on that frequency — the constant-Q smoothing a hardware audio
 * analyser applies. Without it a harmonically rich signal shows up as a forest
 * of partials and the before/after comparison is unreadable.
 */
export function octaveSmooth(freqs, power, fraction = 12.0) {
  if (fraction <= 0 || freqs.length < 3) return power;

  const ratio = 2.0 ** (1.0 / (2.0 * fraction));
  const n = power.length;

  // Cumulative sum turns each band average into two lookups.
  const csum = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) csum[i + 1] = csum[i] + power[i];

  const out = new Float64Array(n);
  let lo = 0;
  let hi = 0;
  for (let i = 0; i < n; i++) {
    const fLo = freqs[i] / ratio;
    const fHi = freqs[i] * ratio;
    // freqs is ascending, and so are the bounds, so the cursors only move forward.
    while (lo < n && freqs[lo] < fLo) lo++;
    if (hi < lo) hi = lo;
    while (hi < n && freqs[hi] <= fHi) hi++;
    const h = Math.max(hi, lo + 1);
    out[i] = (csum[Math.min(h, n)] - csum[lo]) / (Math.min(h, n) - lo);
  }
  return out;
}

/**
 * Averaged FFT magnitude spectrum in dB, on a linear frequency grid.
 *
 * Welch's method — overlapping Hann-windowed segments, detrended and averaged —
 * mirroring scipy.signal.welch(..., scaling='spectrum'). One giant FFT of a
 * whole song is dominated by noise between bins; averaging gives a curve you
 * can actually read the EQ changes off.
 *
 * Returns { freqs, db }.
 */
export function spectrum(mono, fs, {
  nFft = 8192,
  maxSeconds = 30.0,
  smoothFraction = 12.0,
} = {}) {
  // Long files do not make the estimate better, only slower.
  const limit = Math.floor(maxSeconds * fs);
  const sig = mono.length > limit ? mono.subarray(0, limit) : mono;

  // Power of two because the FFT below is radix-2, and never longer than the
  // signal — eqcore.equalizer.spectrum applies the same two rules, so both
  // implementations land on the same bin grid even for very short clips.
  let nperseg = Math.min(nFft, Math.max(256, sig.length));
  nperseg = largestPowerOfTwoAtMost(Math.min(nperseg, sig.length));
  if (nperseg < 2) {
    return { freqs: new Float64Array([0]), db: new Float64Array([-200]) };
  }

  const noverlap = nperseg >> 1;
  const step = nperseg - noverlap;
  const nBins = (nperseg >> 1) + 1;

  // Hann window, matching scipy's sym=False periodic window.
  const win = new Float64Array(nperseg);
  let winSum = 0.0;
  for (let i = 0; i < nperseg; i++) {
    win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nperseg);
    winSum += win[i];
  }
  const scale = 1.0 / (winSum * winSum);

  const acc = new Float64Array(nBins);
  const re = new Float64Array(nperseg);
  const im = new Float64Array(nperseg);

  let segments = 0;
  for (let start = 0; start + nperseg <= sig.length; start += step) {
    // Detrend 'constant': remove the segment mean before windowing.
    let mean = 0.0;
    for (let i = 0; i < nperseg; i++) mean += sig[start + i];
    mean /= nperseg;

    for (let i = 0; i < nperseg; i++) {
      re[i] = (sig[start + i] - mean) * win[i];
      im[i] = 0.0;
    }
    fftInPlace(re, im);

    for (let k = 0; k < nBins; k++) {
      let p = (re[k] * re[k] + im[k] * im[k]) * scale;
      // One-sided: fold in the negative frequencies, except DC and Nyquist.
      if (k > 0 && k < nperseg - k) p *= 2.0;
      acc[k] += p;
    }
    segments++;
  }

  if (segments === 0) {
    return { freqs: new Float64Array([0]), db: new Float64Array([-200]) };
  }
  for (let k = 0; k < nBins; k++) acc[k] /= segments;

  const freqs = new Float64Array(nBins);
  for (let k = 0; k < nBins; k++) freqs[k] = (k * fs) / nperseg;

  const smoothed = octaveSmooth(freqs, acc, smoothFraction);
  const db = new Float64Array(nBins);
  for (let k = 0; k < nBins; k++) db[k] = 10.0 * Math.log10(Math.max(smoothed[k], 1e-20));

  return { freqs, db };
}

/**
 * Min/max envelope for drawing a waveform without plotting millions of points.
 * Bucketing and keeping both extremes preserves transients that naive
 * decimation would erase. Returns { min, max }.
 */
export function waveformEnvelope(mono, targetPoints = 2000) {
  if (mono.length === 0) {
    return { min: new Float64Array(1), max: new Float64Array(1) };
  }
  if (mono.length <= targetPoints) {
    return { min: Float64Array.from(mono), max: Float64Array.from(mono) };
  }

  const bucket = Math.floor(mono.length / targetPoints);
  const min = new Float64Array(targetPoints);
  const max = new Float64Array(targetPoints);

  for (let i = 0; i < targetPoints; i++) {
    const start = i * bucket;
    let lo = mono[start];
    let hi = lo;
    for (let j = 1; j < bucket; j++) {
      const v = mono[start + j];
      if (v < lo) lo = v;
      else if (v > hi) hi = v;
    }
    min[i] = lo;
    max[i] = hi;
  }
  return { min, max };
}

/** Peak absolute sample across every channel. */
export function peakOf(channels) {
  let peak = 0.0;
  for (const ch of channels) {
    for (let i = 0; i < ch.length; i++) {
      const v = Math.abs(ch[i]);
      if (v > peak) peak = v;
    }
  }
  return peak;
}

/**
 * Gain staging, matching eqcore.equalizer.process.
 *
 * Boosting bands raises the peak, so we measure it and attenuate if it would
 * clip. We only ever attenuate — never make up gain — so an A/B against the
 * original stays honest about what the EQ did.
 *
 * Returns { scale, appliedGainDb, peakBefore, clipped }.
 */
export function gainStaging(peak, headroomPeak = HEADROOM_PEAK) {
  if (peak > headroomPeak && peak > 0.0) {
    const scale = headroomPeak / peak;
    return {
      scale,
      appliedGainDb: 20.0 * Math.log10(scale),
      peakBefore: peak,
      clipped: true,
    };
  }
  return { scale: 1.0, appliedGainDb: 0.0, peakBefore: peak, clipped: false };
}

// ---------------------------------------------------------------------------
// WAV encoding
// ---------------------------------------------------------------------------

/**
 * Encode planar float channels as a 16-bit PCM WAV, matching the subtype the
 * Python version writes. Returns a Blob.
 */
export function encodeWav(channels, sampleRate) {
  const numChannels = channels.length;
  const numFrames = channels[0].length;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const dataBytes = numFrames * blockAlign;

  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);            // PCM chunk size
  view.setUint16(20, 1, true);             // format = PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);            // bits per sample
  writeString(36, 'data');
  view.setUint32(40, dataBytes, true);

  let offset = 44;
  for (let i = 0; i < numFrames; i++) {
    for (let c = 0; c < numChannels; c++) {
      let s = channels[c][i];
      s = s < -1 ? -1 : s > 1 ? 1 : s;
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

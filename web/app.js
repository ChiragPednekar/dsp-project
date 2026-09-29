/**
 * 7-Band Parametric Audio Equalizer — browser front end.
 *
 * This module is presentation and wiring only. The band table, presets, gain
 * staging and analysis live in dsp.js; the biquads themselves are native
 * Web Audio nodes driven with those parameters.
 *
 * The filter chain is built once and the sliders write straight into
 * BiquadFilterNode.gain, so moving a slider changes what you are hearing on
 * the next audio block. The heavier work — re-rendering the clip offline to
 * measure the new peak and redraw the spectrum — is debounced behind that.
 */

import {
  BANDS, GAIN_MIN_DB, GAIN_MAX_DB, MAX_DURATION_SECONDS, PRESET_NAMES,
  PRESET_NOTES, flatGains, getPreset, clampGain, spectrum, waveformEnvelope,
  toMono, peakOf, gainStaging, encodeWav,
} from './dsp.js';
import { buildDemoBuffer, DEMO_NAME } from './demo.js';
import { drawEqCurve, drawSpectrum, drawWaveform } from './plots.js';

/** Seconds of audio used for the live spectrum and peak estimate. */
const ANALYSIS_SECONDS = 30.0;

const ACCEPTED = ['wav', 'mp3', 'flac', 'ogg', 'oga', 'aiff', 'aif', 'm4a', 'aac', 'webm'];

const state = {
  gains: flatGains(),
  buffer: null,
  name: '',
  analysis: null,      // { freqs, dbBefore, envBefore, duration }
  bypass: false,
  playing: false,
  startedAt: 0,        // ctx.currentTime when playback began
  offset: 0,           // position within the buffer at that moment
  appliedGainDb: 0,
  renderToken: 0,
  lastDraw: null,
};

let ctx = null;
let graph = null;      // { input, filters, eqGain, bypassGain, master }
let source = null;

const el = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Audio graph
// ---------------------------------------------------------------------------

function audioContext() {
  if (!ctx) {
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    graph = buildGraph(ctx, state.gains);
    graph.master.connect(ctx.destination);
    applyRouting(0);
  }
  return ctx;
}

/**
 * Build the cascade: one biquad per band, in series, exactly the layout
 * eqcore.equalizer.build_sos produces. Bands at 0 dB stay in the chain because
 * a 0 dB biquad is the identity — keeping them avoids rebuilding the graph
 * every time a slider passes through zero.
 */
function buildGraph(context, gains) {
  const input = context.createGain();
  const eqGain = context.createGain();
  const bypassGain = context.createGain();
  const master = context.createGain();

  const filters = BANDS.map((band) => {
    const f = context.createBiquadFilter();
    f.type = band.kind;
    f.frequency.value = band.f0;
    f.Q.value = band.q;
    f.gain.value = clampGain(gains[band.key] ?? 0);
    return f;
  });

  let node = input;
  for (const f of filters) {
    node.connect(f);
    node = f;
  }
  node.connect(eqGain);
  eqGain.connect(master);

  input.connect(bypassGain);
  bypassGain.connect(master);

  return { input, filters, eqGain, bypassGain, master };
}

/** Crossfade between the EQ path and the dry path, and apply the headroom trim. */
function applyRouting(ramp = 0.02) {
  if (!graph || !ctx) return;
  const now = ctx.currentTime;
  const eqLevel = state.bypass ? 0 : 10 ** (state.appliedGainDb / 20);
  const dryLevel = state.bypass ? 1 : 0;

  graph.eqGain.gain.setTargetAtTime(eqLevel, now, Math.max(ramp, 0.001));
  graph.bypassGain.gain.setTargetAtTime(dryLevel, now, Math.max(ramp, 0.001));
}

function pushGainsToFilters() {
  if (!graph || !ctx) return;
  const now = ctx.currentTime;
  BANDS.forEach((band, i) => {
    // A short ramp instead of a step so a dragged slider does not click.
    graph.filters[i].gain.setTargetAtTime(clampGain(state.gains[band.key]), now, 0.01);
  });
}

// ---------------------------------------------------------------------------
// Offline rendering
// ---------------------------------------------------------------------------

/**
 * Re-render the clip through an identical chain offline.
 * `seconds` limits the render; pass null for the whole clip.
 * Returns the rendered AudioBuffer (before headroom trimming).
 */
async function renderOffline(seconds) {
  const buf = state.buffer;
  const frames = seconds === null
    ? buf.length
    : Math.min(buf.length, Math.ceil(seconds * buf.sampleRate));

  const offline = new OfflineAudioContext(buf.numberOfChannels, frames, buf.sampleRate);
  const g = buildGraph(offline, state.gains);
  g.eqGain.gain.value = 1;
  g.bypassGain.gain.value = 0;
  g.master.connect(offline.destination);

  const src = offline.createBufferSource();
  src.buffer = buf;
  src.connect(g.input);
  src.start();

  return offline.startRendering();
}

function channelsOf(buffer) {
  const out = [];
  for (let c = 0; c < buffer.numberOfChannels; c++) out.push(buffer.getChannelData(c));
  return out;
}

/**
 * Recompute the spectrum-after and the headroom trim.
 *
 * The render is capped at ANALYSIS_SECONDS: a 30 s window is plenty for the
 * Welch estimate, and it keeps the slider responsive on long files. The
 * download path re-renders the whole clip and measures the true peak, so the
 * exported file is trimmed exactly, not from this estimate.
 */
async function refreshAnalysis() {
  if (!state.buffer || !state.analysis) return;
  const token = ++state.renderToken;

  setBusy(true);
  try {
    const rendered = await renderOffline(ANALYSIS_SECONDS);
    if (token !== state.renderToken) return;   // a newer change superseded this

    const chans = channelsOf(rendered);
    const staging = gainStaging(peakOf(chans));
    state.appliedGainDb = staging.appliedGainDb;
    applyRouting();

    const mono = toMono(chans);
    const after = spectrum(mono, rendered.sampleRate, { maxSeconds: ANALYSIS_SECONDS });

    // Trim the same way playback will, so the waveform shows what you hear.
    const trimmed = new Float64Array(mono.length);
    for (let i = 0; i < mono.length; i++) trimmed[i] = mono[i] * staging.scale;

    // Kept so a resize can repaint without re-rendering the audio offline.
    state.lastDraw = {
      spectrum: [after.freqs, state.analysis.dbBefore, after.db, rendered.sampleRate],
      wave: [state.analysis.envBefore, waveformEnvelope(trimmed), state.analysis.duration],
    };
    drawAnalysisPlots();

    renderHeadroom(staging);
  } catch (err) {
    console.error(err);
  } finally {
    if (token === state.renderToken) setBusy(false);
  }
}

/** Repaint the spectrum and waveform from the last render's data. */
function drawAnalysisPlots() {
  if (!state.lastDraw) return;
  drawSpectrum(el('spectrumCanvas'), ...state.lastDraw.spectrum);
  drawWaveform(el('waveCanvas'), ...state.lastDraw.wave);
}

let analysisTimer = null;
function scheduleAnalysis() {
  clearTimeout(analysisTimer);
  analysisTimer = setTimeout(refreshAnalysis, 180);
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

async function loadArrayBuffer(bytes, name) {
  const context = audioContext();
  let buffer;
  try {
    buffer = await context.decodeAudioData(bytes);
  } catch {
    throw new Error(
      `Could not decode "${name}". The browser handles WAV, MP3, FLAC, OGG, ` +
      `M4A and AAC; some browsers refuse certain FLAC and AIFF variants.`
    );
  }
  if (buffer.duration > MAX_DURATION_SECONDS) {
    throw new Error(
      `"${name}" is ${(buffer.duration / 60).toFixed(1)} minutes long; ` +
      `the limit is ${Math.round(MAX_DURATION_SECONDS / 60)} minutes — ` +
      `exporting renders the whole clip in memory at once, so the browser ` +
      `build caps this lower than the desktop app does.`
    );
  }
  setClip(buffer, name);
}

function setClip(buffer, name) {
  stopPlayback();
  state.buffer = buffer;
  state.name = name;

  const mono = toMono(channelsOf(buffer));
  const before = spectrum(mono, buffer.sampleRate, { maxSeconds: ANALYSIS_SECONDS });
  const window = mono.length > ANALYSIS_SECONDS * buffer.sampleRate
    ? mono.subarray(0, Math.floor(ANALYSIS_SECONDS * buffer.sampleRate))
    : mono;

  state.analysis = {
    freqs: before.freqs,
    dbBefore: before.db,
    envBefore: waveformEnvelope(window),
    duration: window.length / buffer.sampleRate,
  };

  el('empty').hidden = true;
  el('analysis').hidden = false;
  el('transport').hidden = false;
  renderClipInfo();
  refreshAnalysis();
}

function renderClipInfo() {
  const b = state.buffer;
  if (!b) return;
  const total = Math.round(b.duration);
  el('clipName').textContent = state.name;
  el('clipMeta').textContent =
    `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')} · ` +
    `${b.sampleRate.toLocaleString()} Hz · ` +
    `${b.numberOfChannels === 1 ? 'mono' : `${b.numberOfChannels} ch`}`;
  el('clipInfo').hidden = false;
}

function renderHeadroom(staging) {
  const node = el('headroom');
  if (staging.clipped) {
    node.textContent =
      `Output trimmed ${staging.appliedGainDb.toFixed(1)} dB — the boosted mix ` +
      `peaked at ${staging.peakBefore.toFixed(2)}, above the 0.90 headroom limit.`;
    node.hidden = false;
  } else {
    node.hidden = true;
  }
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------

function startPlayback(from = state.offset) {
  if (!state.buffer) return;
  const context = audioContext();
  if (context.state === 'suspended') context.resume();

  stopSource();
  source = context.createBufferSource();
  source.buffer = state.buffer;
  source.connect(graph.input);
  source.onended = () => {
    if (state.playing && source && !source._stoppedManually) {
      state.playing = false;
      state.offset = 0;
      renderTransport();
    }
  };

  const offset = from % state.buffer.duration;
  source.start(0, offset);
  state.startedAt = context.currentTime;
  state.offset = offset;
  state.playing = true;
  renderTransport();
  tickProgress();
}

function stopSource() {
  if (source) {
    source._stoppedManually = true;
    try { source.stop(); } catch { /* already stopped */ }
    source.disconnect();
    source = null;
  }
}

function pausePlayback() {
  if (!state.playing) return;
  state.offset = currentPosition();
  stopSource();
  state.playing = false;
  renderTransport();
}

function stopPlayback() {
  stopSource();
  state.playing = false;
  state.offset = 0;
  renderTransport();
}

function currentPosition() {
  if (!state.playing || !ctx || !state.buffer) return state.offset;
  return Math.min(state.buffer.duration,
                  state.offset + (ctx.currentTime - state.startedAt));
}

function tickProgress() {
  if (!state.playing) return;
  paintPosition(currentPosition());
  requestAnimationFrame(tickProgress);
}

/**
 * Paint the playhead everywhere it is shown.
 *
 * The bar carries role="slider", so it has to report a value as well as look
 * like one — a screen reader reads aria-valuetext, not the pixel width.
 */
function paintPosition(pos) {
  if (!state.buffer) return;
  const dur = state.buffer.duration;
  const fraction = dur > 0 ? Math.min(1, Math.max(0, pos / dur)) : 0;

  el('progressFill').style.width = `${fraction * 100}%`;
  el('position').textContent = timecode(pos);

  const bar = el('progress');
  bar.setAttribute('aria-valuenow', String(Math.round(fraction * 100)));
  bar.setAttribute('aria-valuetext', `${timecode(pos)} of ${timecode(dur)}`);
}

/** Move the playhead to an absolute time, keeping playback state. */
function seekTo(seconds) {
  if (!state.buffer) return;
  const to = Math.min(state.buffer.duration, Math.max(0, seconds));
  if (state.playing) {
    startPlayback(to);
  } else {
    state.offset = to;
    renderTransport();
  }
}

function timecode(t) {
  const s = Math.max(0, Math.floor(t));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function renderTransport() {
  el('playBtn').textContent = state.playing ? 'Pause' : 'Play';
  el('playBtn').setAttribute('aria-pressed', String(state.playing));
  if (state.buffer) {
    el('duration').textContent = timecode(state.buffer.duration);
    paintPosition(currentPosition());
  }
}

// ---------------------------------------------------------------------------
// EQ curve
// ---------------------------------------------------------------------------

/**
 * A second, silent copy of the chain used only for drawing the response.
 *
 * The playback filters are driven with setTargetAtTime so a dragged slider
 * does not click, which means their .gain.value lags the slider by a few
 * milliseconds — and does not move at all while the context is suspended.
 * getFrequencyResponse reads that live value, so measuring off the playback
 * nodes would draw a stale curve. These nodes take the gain directly instead.
 */
let measureFilters = null;

function measurementFilters(sampleRate) {
  if (!measureFilters || measureFilters.sampleRate !== sampleRate) {
    const offline = new OfflineAudioContext(1, 1, sampleRate);
    measureFilters = {
      sampleRate,
      nodes: BANDS.map((band) => {
        const f = offline.createBiquadFilter();
        f.type = band.kind;
        f.frequency.value = band.f0;
        f.Q.value = band.q;
        return f;
      }),
    };
  }
  for (let i = 0; i < BANDS.length; i++) {
    measureFilters.nodes[i].gain.value = clampGain(state.gains[BANDS[i].key]);
  }
  return measureFilters.nodes;
}

/**
 * The response of the cascade, read back out of biquads carrying the current
 * band gains. Cascaded sections multiply in magnitude, so they add in dB.
 */
function eqResponse(nPoints = 512) {
  const context = audioContext();
  const nyquist = context.sampleRate / 2;
  const fMin = 20;
  const fMax = Math.min(20000, nyquist * 0.999);

  const freqs = new Float32Array(nPoints);
  const logMin = Math.log10(fMin);
  const logMax = Math.log10(fMax);
  for (let i = 0; i < nPoints; i++) {
    freqs[i] = 10 ** (logMin + ((logMax - logMin) * i) / (nPoints - 1));
  }

  const total = new Float64Array(nPoints);
  const mag = new Float32Array(nPoints);
  const phase = new Float32Array(nPoints);

  for (const f of measurementFilters(context.sampleRate)) {
    f.getFrequencyResponse(freqs, mag, phase);
    for (let i = 0; i < nPoints; i++) {
      total[i] += 20 * Math.log10(Math.max(mag[i], 1e-12));
    }
  }
  return { freqs, magDb: total };
}

function redrawEqCurve() {
  audioContext();
  const { freqs, magDb } = eqResponse();
  drawEqCurve(el('eqCanvas'), freqs, magDb);
  el('eqCanvas').setAttribute('aria-label', describeCurve());
}

/**
 * A text equivalent of the response curve.
 *
 * The plot is a canvas, which is opaque to assistive technology, so the same
 * information has to exist as a sentence on the element's aria-label.
 */
function describeCurve() {
  const moved = BANDS
    .filter((b) => Math.abs(state.gains[b.key]) >= 0.05)
    .map((b) => `${b.name} ${state.gains[b.key] > 0 ? 'up' : 'down'} `
      + `${Math.abs(state.gains[b.key]).toFixed(1)} decibels`);

  if (moved.length === 0) {
    return 'Frequency response of the filter chain. All bands at 0 dB, '
      + 'so the chain is a pass-through.';
  }
  return `Frequency response of the filter chain. ${moved.join(', ')}.`;
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

function buildSliders() {
  const host = el('bands');
  host.innerHTML = '';

  for (const band of BANDS) {
    const row = document.createElement('div');
    row.className = 'band';

    const label = document.createElement('label');
    label.className = 'band-label';
    label.htmlFor = `slider_${band.key}`;
    label.innerHTML =
      `<span class="band-name">${band.name}</span>` +
      `<span class="band-span">${band.span}</span>`;

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.id = `slider_${band.key}`;
    slider.min = String(GAIN_MIN_DB);
    slider.max = String(GAIN_MAX_DB);
    slider.step = '0.5';
    slider.value = String(state.gains[band.key]);
    slider.setAttribute('aria-label', `${band.name}, ${band.span}, gain in decibels`);

    const readout = document.createElement('output');
    readout.className = 'band-value';
    readout.id = `value_${band.key}`;
    readout.textContent = formatDb(state.gains[band.key]);

    slider.addEventListener('input', () => {
      state.gains[band.key] = Number(slider.value);
      readout.textContent = formatDb(state.gains[band.key]);
      readout.classList.toggle('is-set', Math.abs(state.gains[band.key]) >= 0.05);
      el('preset').value = matchingPreset() ?? '';
      onGainsChanged();
    });
    // Double-click a slider to return that band to 0 dB.
    slider.addEventListener('dblclick', () => {
      slider.value = '0';
      slider.dispatchEvent(new Event('input'));
    });

    row.append(label, slider, readout);
    host.appendChild(row);
  }
}

function formatDb(db) {
  const v = Number(db);
  return `${v > 0 ? '+' : ''}${v.toFixed(1)} dB`;
}

/** The preset name whose gains match the sliders exactly, if any. */
function matchingPreset() {
  for (const name of PRESET_NAMES) {
    const p = getPreset(name);
    if (BANDS.every((b) => Math.abs(p[b.key] - state.gains[b.key]) < 1e-9)) return name;
  }
  return null;
}

function syncSliders() {
  for (const band of BANDS) {
    el(`slider_${band.key}`).value = String(state.gains[band.key]);
    const readout = el(`value_${band.key}`);
    readout.textContent = formatDb(state.gains[band.key]);
    readout.classList.toggle('is-set', Math.abs(state.gains[band.key]) >= 0.05);
  }
}

function onGainsChanged() {
  pushGainsToFilters();
  redrawEqCurve();
  if (state.buffer) scheduleAnalysis();
}

function applyPreset(name) {
  state.gains = getPreset(name);
  el('presetNote').textContent = PRESET_NOTES[name] ?? '';
  syncSliders();
  onGainsChanged();
}

function setBusy(busy) {
  el('busy').hidden = !busy;
}

function showError(message) {
  const node = el('error');
  node.textContent = message;
  node.hidden = false;
}

function clearError() {
  el('error').hidden = true;
}

// ---------------------------------------------------------------------------
// Download
// ---------------------------------------------------------------------------

async function downloadProcessed() {
  if (!state.buffer) return;
  const btn = el('downloadBtn');
  btn.disabled = true;
  btn.textContent = 'Rendering…';
  setBusy(true);

  try {
    // The whole clip this time, and the true peak — the exported file gets the
    // exact trim, not the estimate the live display runs on.
    const rendered = await renderOffline(null);
    const chans = channelsOf(rendered).map((c) => Float32Array.from(c));
    const staging = gainStaging(peakOf(chans));

    if (staging.scale !== 1) {
      for (const c of chans) {
        for (let i = 0; i < c.length; i++) c[i] *= staging.scale;
      }
    }

    const blob = encodeWav(chans, rendered.sampleRate);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const stem = state.name.replace(/\.[^.]+$/, '') || 'audio';
    a.href = url;
    a.download = `${stem}-eq.wav`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  } catch (err) {
    console.error(err);
    showError('Rendering the processed file failed. See the browser console.');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Download processed WAV';
    setBusy(false);
  }
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

function init() {
  // Presets.
  const preset = el('preset');
  for (const name of PRESET_NAMES) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    preset.appendChild(opt);
  }
  preset.value = PRESET_NAMES[0];
  el('presetNote').textContent = PRESET_NOTES[PRESET_NAMES[0]];
  preset.addEventListener('change', () => {
    if (preset.value) applyPreset(preset.value);
  });

  buildSliders();
  redrawEqCurve();

  // File input.
  const fileInput = el('file');
  fileInput.addEventListener('change', async () => {
    const file = fileInput.files?.[0];
    if (file) await handleFile(file);
    fileInput.value = '';
  });

  // Drag and drop onto the whole page.
  const dropZone = el('drop');
  ['dragenter', 'dragover'].forEach((type) => {
    document.addEventListener(type, (e) => {
      e.preventDefault();
      dropZone.classList.add('is-over');
    });
  });
  ['dragleave', 'drop'].forEach((type) => {
    document.addEventListener(type, (e) => {
      e.preventDefault();
      if (type === 'dragleave' && e.relatedTarget) return;
      dropZone.classList.remove('is-over');
    });
  });
  document.addEventListener('drop', async (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) await handleFile(file);
  });

  el('demoBtn').addEventListener('click', async () => {
    clearError();
    setBusy(true);
    el('demoBtn').disabled = true;
    try {
      // Yield first so the button's disabled state paints before we block.
      await new Promise((r) => setTimeout(r, 0));
      setClip(buildDemoBuffer(audioContext()), DEMO_NAME);
    } catch (err) {
      console.error(err);
      showError('Could not build the demo clip.');
    } finally {
      el('demoBtn').disabled = false;
      setBusy(false);
    }
  });

  el('playBtn').addEventListener('click', () => {
    if (state.playing) pausePlayback();
    else startPlayback();
  });
  el('stopBtn').addEventListener('click', stopPlayback);

  el('bypass').addEventListener('change', (e) => {
    state.bypass = e.target.checked;
    applyRouting();
    el('bypassLabel').textContent = state.bypass ? 'Hearing: original' : 'Hearing: equalised';
  });

  const progress = el('progress');

  progress.addEventListener('click', (e) => {
    if (!state.buffer) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const fraction = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    seekTo(fraction * state.buffer.duration);
  });

  // role="slider" promises keyboard operation; without this the bar is
  // focusable and announced but cannot actually be moved.
  progress.addEventListener('keydown', (e) => {
    if (!state.buffer) return;
    const pos = currentPosition();
    const step = e.shiftKey ? 1 : 5;
    const handlers = {
      ArrowLeft: () => pos - step,
      ArrowRight: () => pos + step,
      ArrowDown: () => pos - step,
      ArrowUp: () => pos + step,
      PageDown: () => pos - 30,
      PageUp: () => pos + 30,
      Home: () => 0,
      End: () => state.buffer.duration,
    };
    const next = handlers[e.key];
    if (!next) return;
    e.preventDefault();
    seekTo(next());
  });

  el('resetBtn').addEventListener('click', () => {
    el('preset').value = PRESET_NAMES[0];
    applyPreset(PRESET_NAMES[0]);
  });

  el('downloadBtn').addEventListener('click', downloadProcessed);

  // Keep the canvases sharp and correctly sized.
  //
  // A canvas drawn while the layout is still settling captures the wrong
  // getBoundingClientRect and paints at the wrong scale until something
  // redraws it. Watching the plot boxes themselves catches every cause of
  // that -- window resize, the sidebar reflowing, a late web font, the tab
  // being shown -- which a window 'resize' listener alone does not.
  let resizeTimer = null;
  const observer = new ResizeObserver(() => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      redrawEqCurve();
      drawAnalysisPlots();   // repaints from cached data; no offline render
    }, 100);
  });
  for (const node of document.querySelectorAll('.plot')) observer.observe(node);

  // Space bar toggles playback unless a control has focus.
  document.addEventListener('keydown', (e) => {
    if (e.code !== 'Space') return;
    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'BUTTON') return;
    e.preventDefault();
    if (state.playing) pausePlayback();
    else startPlayback();
  });
}

async function handleFile(file) {
  clearError();
  const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
  if (!ACCEPTED.includes(ext)) {
    showError(`"${file.name}" is not an audio format this page reads ` +
              `(${ACCEPTED.join(', ')}).`);
    return;
  }
  setBusy(true);
  try {
    await loadArrayBuffer(await file.arrayBuffer(), file.name);
  } catch (err) {
    showError(err.message);
  } finally {
    setBusy(false);
  }
}

init();

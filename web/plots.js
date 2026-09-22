/**
 * Canvas plotting — the browser counterpart of ui/plots.py.
 *
 * Three plots, drawn straight to a 2D canvas with no charting library so the
 * page has no third-party runtime dependency: the live filter response, the
 * before/after spectrum, and the before/after waveform. Styling follows the
 * same dark palette the Streamlit theme used.
 */

const INK = '#c9d1d9';
const MUTED = '#6e7681';
const GRID = '#21262d';
const ACCENT = '#00d4a0';
const BEFORE = '#8b949e';
const PANEL = '#0d1117';

const PAD = { top: 14, right: 14, bottom: 34, left: 48 };

/** Size the backing store to the element's CSS box at device resolution. */
function prepare(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));

  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = PANEL;
  ctx.fillRect(0, 0, w, h);

  return {
    ctx,
    w,
    h,
    plotW: Math.max(1, w - PAD.left - PAD.right),
    plotH: Math.max(1, h - PAD.top - PAD.bottom),
  };
}

function frame(ctx, w, h, plotW, plotH) {
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.strokeRect(PAD.left + 0.5, PAD.top + 0.5, plotW, plotH);
}

const FREQ_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

function freqLabel(f) {
  return f >= 1000 ? `${f / 1000}k` : String(f);
}

/** Log-frequency x mapping, plus vertical gridlines and tick labels. */
function logFreqAxis(ctx, plotW, plotH, fMin, fMax) {
  const logMin = Math.log10(fMin);
  const logMax = Math.log10(fMax);
  const xOf = (f) => PAD.left + ((Math.log10(f) - logMin) / (logMax - logMin)) * plotW;

  ctx.font = '10px ui-sans-serif, system-ui, -apple-system, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';

  for (const f of FREQ_TICKS) {
    if (f < fMin || f > fMax) continue;
    const x = xOf(f);
    ctx.strokeStyle = GRID;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, PAD.top);
    ctx.lineTo(x + 0.5, PAD.top + plotH);
    ctx.stroke();
    ctx.fillStyle = MUTED;
    ctx.fillText(freqLabel(f), x, PAD.top + plotH + 6);
  }

  ctx.fillStyle = MUTED;
  ctx.fillText('Frequency (Hz)', PAD.left + plotW / 2, PAD.top + plotH + 20);
  return xOf;
}

/** Linear y mapping in dB, plus horizontal gridlines and tick labels. */
function dbAxis(ctx, plotW, plotH, dbMin, dbMax, step, label) {
  const yOf = (db) => PAD.top + (1 - (db - dbMin) / (dbMax - dbMin)) * plotH;

  ctx.font = '10px ui-sans-serif, system-ui, -apple-system, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';

  const first = Math.ceil(dbMin / step) * step;
  for (let db = first; db <= dbMax + 1e-9; db += step) {
    const y = yOf(db);
    ctx.strokeStyle = Math.abs(db) < 1e-9 ? MUTED : GRID;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y + 0.5);
    ctx.lineTo(PAD.left + plotW, y + 0.5);
    ctx.stroke();
    ctx.fillStyle = MUTED;
    ctx.fillText(String(Math.round(db)), PAD.left - 6, y);
  }

  ctx.save();
  ctx.translate(11, PAD.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = MUTED;
  ctx.fillText(label, 0, 0);
  ctx.restore();

  return yOf;
}

function polyline(ctx, xs, ys, colour, width, clipTop, clipBottom) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(PAD.left, clipTop, xs.length ? 1e5 : 0, clipBottom - clipTop);
  ctx.restore();

  ctx.strokeStyle = colour;
  ctx.lineWidth = width;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < xs.length; i++) {
    const y = Math.min(clipBottom, Math.max(clipTop, ys[i]));
    if (!started) {
      ctx.moveTo(xs[i], y);
      started = true;
    } else {
      ctx.lineTo(xs[i], y);
    }
  }
  ctx.stroke();
}

/**
 * The live frequency response of the filter chain.
 * `magDb` comes from BiquadFilterNode.getFrequencyResponse — it is the real
 * response of the filters that process the audio, not a drawing of the sliders.
 */
export function drawEqCurve(canvas, freqs, magDb) {
  const { ctx, w, h, plotW, plotH } = prepare(canvas);
  const fMax = freqs[freqs.length - 1];
  const xOf = logFreqAxis(ctx, plotW, plotH, freqs[0], fMax);
  const yOf = dbAxis(ctx, plotW, plotH, -15, 15, 5, 'Gain (dB)');

  const xs = new Float64Array(freqs.length);
  const ys = new Float64Array(freqs.length);
  for (let i = 0; i < freqs.length; i++) {
    xs[i] = xOf(freqs[i]);
    ys[i] = yOf(magDb[i]);
  }

  // Fill between the curve and 0 dB so boosts and cuts read at a glance.
  const zero = yOf(0);
  ctx.save();
  ctx.beginPath();
  ctx.rect(PAD.left, PAD.top, plotW, plotH);
  ctx.clip();
  ctx.beginPath();
  ctx.moveTo(xs[0], zero);
  for (let i = 0; i < xs.length; i++) ctx.lineTo(xs[i], ys[i]);
  ctx.lineTo(xs[xs.length - 1], zero);
  ctx.closePath();
  ctx.fillStyle = 'rgba(0, 212, 160, 0.13)';
  ctx.fill();
  polyline(ctx, xs, ys, ACCENT, 2, PAD.top, PAD.top + plotH);
  ctx.restore();

  frame(ctx, w, h, plotW, plotH);
}

/** FFT magnitude spectrum of the audio before and after the EQ. */
export function drawSpectrum(canvas, freqs, dbBefore, dbAfter, fs) {
  const { ctx, w, h, plotW, plotH } = prepare(canvas);
  const fMax = Math.min(fs / 2, 20000);

  // Window the visible dB range around the loudest content so the curve fills
  // the axes instead of hugging the top on quiet material.
  let top = -Infinity;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] < 20 || freqs[i] > fMax) continue;
    if (dbBefore[i] > top) top = dbBefore[i];
    if (dbAfter && dbAfter[i] > top) top = dbAfter[i];
  }
  if (!Number.isFinite(top)) top = 0;
  const dbMax = Math.ceil((top + 6) / 10) * 10;
  const dbMin = dbMax - 100;

  const xOf = logFreqAxis(ctx, plotW, plotH, 20, fMax);
  const yOf = dbAxis(ctx, plotW, plotH, dbMin, dbMax, 20, 'Level (dB)');

  const build = (db) => {
    const xs = [];
    const ys = [];
    for (let i = 0; i < freqs.length; i++) {
      if (freqs[i] < 20 || freqs[i] > fMax) continue;
      xs.push(xOf(freqs[i]));
      ys.push(yOf(db[i]));
    }
    return { xs, ys };
  };

  ctx.save();
  ctx.beginPath();
  ctx.rect(PAD.left, PAD.top, plotW, plotH);
  ctx.clip();

  const before = build(dbBefore);
  polyline(ctx, before.xs, before.ys, BEFORE, 1.2, PAD.top, PAD.top + plotH);
  if (dbAfter) {
    const after = build(dbAfter);
    polyline(ctx, after.xs, after.ys, ACCENT, 1.6, PAD.top, PAD.top + plotH);
  }
  ctx.restore();

  legend(ctx, w, ['Original', 'Equalised'], [BEFORE, ACCENT]);
  frame(ctx, w, h, plotW, plotH);
}

/** Min/max waveform envelope, original above, processed below. */
export function drawWaveform(canvas, envBefore, envAfter, duration) {
  const { ctx, w, h, plotW, plotH } = prepare(canvas);

  const laneH = plotH / 2;
  const lanes = [
    { env: envBefore, colour: BEFORE, label: 'Original', top: PAD.top },
    { env: envAfter, colour: ACCENT, label: 'Equalised', top: PAD.top + laneH },
  ];

  ctx.font = '10px ui-sans-serif, system-ui, -apple-system, sans-serif';

  for (const lane of lanes) {
    if (!lane.env) continue;
    const mid = lane.top + laneH / 2;
    const half = (laneH / 2) * 0.88;

    ctx.strokeStyle = GRID;
    ctx.beginPath();
    ctx.moveTo(PAD.left, mid + 0.5);
    ctx.lineTo(PAD.left + plotW, mid + 0.5);
    ctx.stroke();

    const n = lane.env.min.length;
    ctx.strokeStyle = lane.colour;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = PAD.left + (i / Math.max(n - 1, 1)) * plotW;
      const yLo = mid - Math.max(-1, Math.min(1, lane.env.min[i])) * half;
      const yHi = mid - Math.max(-1, Math.min(1, lane.env.max[i])) * half;
      ctx.moveTo(x + 0.5, yLo);
      ctx.lineTo(x + 0.5, yHi);
    }
    ctx.stroke();

    ctx.fillStyle = MUTED;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(lane.label, PAD.left + 6, lane.top + 4);
  }

  // Time axis.
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const ticks = 6;
  for (let i = 0; i <= ticks; i++) {
    const x = PAD.left + (i / ticks) * plotW;
    const t = (i / ticks) * duration;
    ctx.fillStyle = MUTED;
    ctx.fillText(`${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`,
                 x, PAD.top + plotH + 6);
  }
  ctx.fillStyle = MUTED;
  ctx.fillText('Time', PAD.left + plotW / 2, PAD.top + plotH + 20);

  ctx.save();
  ctx.translate(11, PAD.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = MUTED;
  ctx.fillText('Amplitude', 0, 0);
  ctx.restore();

  frame(ctx, w, h, plotW, plotH);
}

function legend(ctx, w, labels, colours) {
  ctx.font = '10px ui-sans-serif, system-ui, -apple-system, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  let x = PAD.left + 8;
  const y = PAD.top + 10;
  for (let i = 0; i < labels.length; i++) {
    ctx.strokeStyle = colours[i];
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 14, y);
    ctx.stroke();
    ctx.fillStyle = INK;
    ctx.fillText(labels[i], x + 19, y + 0.5);
    x += 19 + ctx.measureText(labels[i]).width + 14;
  }
}

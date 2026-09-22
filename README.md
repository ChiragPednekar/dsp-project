# 7-Band Parametric Audio Equalizer

A working desktop audio equalizer built for a DSP course. Load a WAV or MP3,
drag seven band sliders, watch the filter's real frequency response redraw,
and A/B the result against the original while it plays.

The signal processing is genuine: each band is a second-order IIR biquad
designed with the Audio EQ Cookbook formulas, cascaded in series and applied
with `scipy.signal.sosfilt`. Nothing is faked with FFT-bin scaling, and the EQ
curve on screen is the computed response of the exact filter chain that
processes the audio.

---

## Run it

```bash
./run.sh
```

That creates the virtualenv, installs dependencies, generates the demo clip,
and opens the GUI at <http://localhost:8501>.

Manual equivalent:

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/streamlit run app.py
```

Verify the DSP independently at any time:

```bash
.venv/bin/python tools/verify_dsp.py
```

---

## The browser build (deployed on Vercel)

`web/` is the same equalizer as a static site, so it can be deployed anywhere
that serves files — including Vercel, which cannot host the Streamlit app
(Streamlit needs a long-running server holding a WebSocket per visitor;
Vercel runs static assets and short-lived serverless functions).

```bash
cd web && python3 -m http.server 8777    # then open http://localhost:8777
```

Deploy from the repo root:

```bash
npx vercel deploy --prod
```

`vercel.json` pins it to a no-build static deployment of `web/`, and
`.vercelignore` keeps the Python implementation out of the upload.

### It is the same DSP, not a lookalike

Web Audio's `BiquadFilterNode` implements the *same* Audio EQ Cookbook
formulas that `eqcore/biquad.py` spells out, so the browser build drives
native biquads with the band table from `eqcore/equalizer.py` rather than
reimplementing the coefficients. The two shelves line up as well: Web Audio
ignores `.Q` on a shelf and uses slope `S = 1`, and

    alpha = sin(w0)/2 · sqrt((A + 1/A)(1/S − 1) + 2)  with S = 1
          = sin(w0)·sqrt(2)/2
          = sin(w0)/(2Q)                              with Q = 0.707

which is exactly what `eqcore` passes. Measured against `scipy.signal.sosfreqz`
for the Loudness preset at 48 kHz, the two responses agree to **4×10⁻⁴ dB** —
float32 rounding in `getFrequencyResponse`, not an algorithmic difference.

What *is* ported by hand, because the browser has no SciPy, is the analysis:
Welch's method, fractional-octave smoothing and the waveform envelope, all in
`web/dsp.js`. Those agree with `scipy.signal.welch` to **5×10⁻⁵ dB** across a
140 dB range, DC bin included.

### Differences from the Python version

| | Python / Streamlit | Browser |
|---|---|---|
| Filtering | `scipy.signal.sosfilt`, whole file per change | native biquads, live on the audio thread |
| Decoding | libsndfile via `soundfile` | the browser's own decoder |
| Demo clip | `samples/demo.wav`, 7.9 MB in the repo | synthesised on load by `web/demo.js` |
| Headroom trim | exact peak of the filtered file | estimated from the first 30 s while you drag; exact on export |
| Export | WAV or MP3 | 16-bit PCM WAV |

Audio never leaves the tab — there is no upload and no server side.

---

## Using it

1. **Load audio** — *Upload an audio file* in the sidebar (WAV, MP3, FLAC, OGG,
   AIFF), or press **Load demo clip** for a synthetic 45-second full-spectrum
   test tone. A file that cannot be decoded produces a red message in the
   sidebar, never a console traceback.
2. **Pick a preset** or move the sliders. A preset drives the sliders, so
   anything it does you can also do by hand; the caption says *"Edited by
   hand"* once your values diverge from the preset.
3. **Play.** Play / Pause / Stop drive the transport. **A/B** switches between
   the original and the EQ'd audio *at the same instant of the track* — the
   playhead is handed from one to the other, so you hear the same bar both
   ways instead of restarting.
4. **Adjust while it plays.** Moving a slider re-renders the audio and playback
   resumes from where it was, so you hear the change in context.
5. **Export** the processed audio as WAV, FLAC, OGG or MP3.

---

## Why Streamlit rather than PyQt6

The brief allowed either. Streamlit was chosen because the whole app —
including playback and the A/B switch — could be driven and verified
end-to-end automatically, which is what "confirmed working" in this README is
based on. It also runs from one command on any OS with no Qt system packages.

The trade-off is that Streamlit re-runs the script on every widget change, so
playback is **near-real-time**: releasing a slider re-renders the audio (about
0.2 s for a 45-second stereo clip) and playback continues from the same
position. It is not sample-by-sample streaming while dragging.

The DSP layer (`eqcore/`) imports no GUI code, so it would drop into a Qt front
end unchanged.

---

## Project layout

```
app.py                  Streamlit GUI — widgets and layout only
eqcore/
  biquad.py             Audio EQ Cookbook coefficient formulas
  equalizer.py          Band table, SOS cascade, filtering, gain staging, analysis
  presets.py            Built-in presets
  audio_io.py           Decoding, encoding, and user-facing load errors
ui/
  plots.py              The three embedded matplotlib figures (Agg, never a popup)
  player.py             Transport component: Play/Pause/Stop + instant A/B
tools/
  make_sample.py        Generates samples/demo.wav
  verify_dsp.py         Numerical proof that the filters do what the curve says

web/                    The static browser build — what Vercel serves
  index.html            Markup; the band reference table is built from dsp.js
  dsp.js                Band table, presets, Welch spectrum, envelopes, WAV encoder
  demo.js               Port of tools/make_sample.py — builds the demo clip in-tab
  plots.js              The three canvas plots (no charting library)
  app.js                Audio graph, transport, and UI wiring
  styles.css            Dark theme carried over from .streamlit/config.toml

vercel.json             No-build static deployment of web/
.vercelignore           Keeps the Python implementation out of the upload
```

---

## DSP theory notes

### What a biquad is

A biquad is a second-order IIR filter — two poles and two zeros:

```
        b0 + b1 z⁻¹ + b2 z⁻²
H(z) = ──────────────────────
        a0 + a1 z⁻¹ + a2 z⁻²
```

which is the difference equation

```
y[n] = (b0/a0)·x[n] + (b1/a0)·x[n-1] + (b2/a0)·x[n-2]
                    − (a1/a0)·y[n-1] − (a2/a0)·y[n-2]
```

Five multiplies and four adds per sample, and it holds only two samples of
history — which is why every hardware EQ and audio plugin is built from these
rather than from long FIR kernels.

### What a *peaking* biquad does

A peaking (bell) filter boosts or cuts a band around a centre frequency `f₀`
while leaving DC and Nyquist untouched. It works by placing a conjugate pole
pair and a conjugate zero pair at the *same* angle `ω₀ = 2πf₀/fs` on the
z-plane but at slightly different radii:

* **Boost** — poles closer to the unit circle than the zeros. Near `ω₀` the
  denominator is small, so |H| rises.
* **Cut** — zeros closer than the poles, so |H| dips.
* **0 dB** — poles and zeros land on top of each other and cancel exactly. The
  filter becomes the identity, which is why `build_sos()` drops bands sitting
  at 0 dB instead of computing them.

Away from `ω₀` the pole and zero distances converge and the response returns to
unity — that locality is what makes the bands independent.

The gain enters the coefficients as `A = 10^(dB/40)`, the square root of the
linear amplitude gain, because it appears on both the numerator and the
denominator and the two contributions multiply.

`Q` controls the bell's width through `α = sin(ω₀)/(2Q)`. This project derives
each band's Q from the bandwidth it is supposed to cover:

```
Q = √(2^N) / (2^N − 1)          N = bandwidth in octaves
```

so the bells tile the spectrum evenly instead of overlapping arbitrarily.

### Why shelves at the ends

A bell centred at 40 Hz would roll back off below 40 Hz, doing nothing for the
lowest octave; the same happens above a bell at 15 kHz. So the outer bands are
**shelving** filters — a low shelf lifts *everything* below its corner and a
high shelf lifts *everything* above it. That is the standard graphic-EQ layout:
shelves at the extremes, bells in between.

### Band layout

| Band | Range | Filter | f₀ | Q | What lives there |
|---|---|---|---|---|---|
| Sub-bass | 20–60 Hz | Low shelf | 60 Hz | 0.707 | Felt more than heard; rumble, kick weight |
| Bass | 60–250 Hz | Peaking | 122 Hz | 0.64 | Bass guitar, kick body, warmth |
| Low-mid | 250–500 Hz | Peaking | 354 Hz | 1.41 | Boxiness / "mud" if overdone |
| Mid | 500 Hz–2 kHz | Peaking | 1 kHz | 0.67 | Where the ear is most sensitive; body of voices |
| High-mid | 2–4 kHz | Peaking | 2.83 kHz | 1.41 | Attack, consonants, intelligibility |
| Presence | 4–6 kHz | Peaking | 4.9 kHz | 2.45 | Clarity; harshness when overdone |
| Brilliance | 6–20 kHz | High shelf | 8 kHz | 0.707 | Air, cymbal shimmer, sibilance |

Peaking centres are the **geometric** mean of the band edges (√(f_lo·f_hi)),
not the arithmetic mean, because pitch is logarithmic — 1 kHz sits at the
centre of 500 Hz–2 kHz to the ear, though the arithmetic midpoint is 1.25 kHz.

### Cascading and stability

The bands are stacked as **second-order sections** and run through
`scipy.signal.sosfilt`, which applies them one after another. Convolving all
seven biquads into a single 14th-order polynomial would be mathematically
equivalent but numerically hopeless — the coefficients of a high-order
polynomial are extremely sensitive to rounding, and at 44.1 kHz the low-band
poles sit very close to z = 1. Keeping them as separate second-order stages
keeps every coefficient well conditioned. `biquad.is_stable()` additionally
checks each section's poles against the Jury criterion.

These filters are causal and minimum-phase, so — like every analogue EQ and
almost every plugin — they introduce frequency-dependent phase shift. Running
`sosfiltfilt` instead would cancel the phase at the cost of being non-causal.
Phase shift is the conventional choice and is what an EQ is expected to do.

### Gain staging

Boosting adds energy, so a +8 dB bass lift on a track already near full scale
will exceed ±1.0 and clip. After filtering, the peak is measured and the whole
signal is scaled down if it exceeds 0.90 (about 0.9 dB of headroom). The app
reports exactly how much attenuation it applied.

Only attenuation is ever applied — never make-up gain — so an A/B comparison
stays honest about what the EQ did rather than being loudness-matched behind
your back.

### Reading the spectrum plot

The spectrum is a **Welch** estimate: the signal is split into overlapping
Hann-windowed segments whose periodograms are averaged. A single FFT of a whole
track is dominated by variance between bins; averaging trades frequency
resolution for a readable curve. It is then smoothed to 1/12 octave — constant-Q
smoothing, as a hardware analyser does — because a harmonically rich signal
otherwise shows a forest of individual partials. The EQ's own bands are far
wider than 1/12 octave (the narrowest is 0.585 octaves), so the smoothing does
not hide the effect being measured.

---

## Verification

`tools/verify_dsp.py` pushes signals through the real `equalizer.process()`
path and measures the output rather than trusting the design maths. All checks
pass:

| Check | Result |
|---|---|
| Measured gain vs designed curve, 5 peaking cases | agree to **0.00 dB** |
| Shelving bands reach their nominal gain | within 0.35 dB |
| Band independence (bass +12 dB leaks elsewhere) | < 0.56 dB at 1 kHz, < 0.03 dB above 5 kHz |
| Flat settings are a true pass-through | bit-identical (max diff 0.0) |
| Stability across all gains × 5 sample rates | 0 unstable sections |
| Gain staging prevents clipping | peak held at 0.90, attenuation reported |
| Stereo filtered per channel | shape preserved; L +11.99 dB, R +0.02 dB |
| Presets move the response as described | all 5 directional checks pass |
| Measured spectrum reflects a +12 dB lift | +11.89 dB at 4.9 kHz, +0.01 dB at 300 Hz |

The first row is the important one: a 1 kHz sine through a +12 dB Mid band
comes out **+12.00 dB** louder, exactly what the on-screen curve predicts.

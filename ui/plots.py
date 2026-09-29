"""
The three plots embedded in the app window.

All figures are rendered with the Agg backend and handed to Streamlit as
in-page figures — nothing here ever opens an interactive matplotlib window.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from eqcore import equalizer

# Palette tuned for the dark app theme set in .streamlit/config.toml.
BG = "#0e1117"
PANEL = "#0e1117"
GRID = "#2a2f3a"
TEXT = "#c9d1d9"
MUTED = "#6e7681"
ORIGINAL = "#5b6b7f"
PROCESSED = "#00d4a0"
CURVE = "#ffb454"
ZERO = "#454c5a"

#: Ticks people actually read an audio spectrum by.
FREQ_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
FREQ_LABELS = ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"]


def _style(ax) -> None:
    """Apply the shared dark styling to one axes."""
    ax.set_facecolor(PANEL)
    ax.grid(True, which="major", color=GRID, linewidth=0.6, alpha=0.8)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def _new_figure(width: float, height: float) -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=110)
    fig.patch.set_facecolor(BG)
    _style(ax)
    return fig, ax


def _log_freq_axis(ax, fs: float, f_min: float = 20.0) -> None:
    f_max = min(20000.0, fs / 2.0)
    ax.set_xscale("log")
    ax.set_xlim(f_min, f_max)
    ticks = [f for f in FREQ_TICKS if f_min <= f <= f_max]
    labels = [FREQ_LABELS[FREQ_TICKS.index(f)] for f in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Frequency (Hz)", fontsize=9)


def eq_curve_figure(gains: dict[str, float], fs: float) -> Figure:
    """
    The filter chain's real frequency response.

    This calls equalizer.frequency_response, which evaluates H(e^{jw}) on the
    same cascaded SOS that filters the audio. Move a slider and this curve
    changes because the filter changed — it is a readout, not an illustration.
    """
    fig, ax = _new_figure(7.6, 2.9)

    freqs, magnitude = equalizer.frequency_response(gains, fs, n_points=2048)

    ax.axhline(0.0, color=ZERO, linewidth=1.0, linestyle="--", zorder=1)
    ax.plot(freqs, magnitude, color=CURVE, linewidth=2.2, zorder=3)
    ax.fill_between(freqs, 0.0, magnitude, color=CURVE, alpha=0.16, zorder=2)

    # Mark each band's centre so the sliders map visibly onto the curve.
    for band in equalizer.BANDS:
        gain = gains.get(band.key, 0.0)
        if band.f0 > fs / 2.0:
            continue
        ax.axvline(band.f0, color=GRID, linewidth=0.7, alpha=0.7, zorder=1)
        if abs(gain) > 0.05:
            ax.plot([band.f0], [gain], "o", color=CURVE, markersize=5,
                    markeredgecolor=BG, markeredgewidth=1.0, zorder=4)

    _log_freq_axis(ax, fs)
    ax.set_ylim(-16, 16)
    ax.set_yticks([-12, -6, 0, 6, 12])
    ax.set_ylabel("Gain (dB)", fontsize=9)
    ax.set_title("EQ curve — computed response of the biquad cascade",
                 fontsize=10, pad=8, loc="left")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return fig


def spectrum_figure(
    original: np.ndarray, processed: np.ndarray, fs: float
) -> Figure:
    """FFT magnitude spectrum of the audio before and after the EQ."""
    fig, ax = _new_figure(7.6, 2.9)

    f_before, s_before = equalizer.spectrum(original, fs)
    f_after, s_after = equalizer.spectrum(processed, fs)

    # Drop the DC bin: log(0) has no place on a log axis.
    ax.plot(f_before[1:], s_before[1:], color=ORIGINAL, linewidth=1.1,
            label="Original", zorder=2)
    ax.plot(f_after[1:], s_after[1:], color=PROCESSED, linewidth=1.3,
            label="EQ'd", zorder=3)

    _log_freq_axis(ax, fs)

    # Frame the y-range around the signal so the curves fill the panel.
    audible = s_before[(f_before >= 20) & (f_before <= min(20000, fs / 2))]
    if audible.size:
        top = float(np.max(audible)) + 8.0
        ax.set_ylim(top - 95.0, top)

    ax.set_ylabel("Magnitude (dB)", fontsize=9)
    ax.set_title("Frequency spectrum — before vs after", fontsize=10, pad=8, loc="left")
    legend = ax.legend(loc="upper right", fontsize=8, facecolor=PANEL,
                       edgecolor=GRID, framealpha=0.9)
    for text in legend.get_texts():
        text.set_color(TEXT)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return fig


def waveform_figure(
    original: np.ndarray, processed: np.ndarray, fs: float
) -> Figure:
    """
    Waveforms of the original and processed audio, stacked.

    Drawn as a min/max envelope per pixel column rather than every sample,
    so a five-minute file still renders instantly and transients stay visible.
    """
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 3.0), dpi=110, sharex=True)
    fig.patch.set_facecolor(BG)

    duration = original.shape[0] / float(fs)

    for ax, data, color, label in (
        (axes[0], original, ORIGINAL, "Original"),
        (axes[1], processed, PROCESSED, "EQ'd"),
    ):
        _style(ax)
        lo, hi = equalizer.waveform_envelope(data)
        t = np.linspace(0.0, duration, lo.size)
        ax.fill_between(t, lo, hi, color=color, linewidth=0.0)
        ax.set_xlim(0.0, max(duration, 1e-6))
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([-1, 0, 1])
        ax.set_ylabel(label, fontsize=8)

    axes[1].set_xlabel("Time (s)", fontsize=9)
    axes[0].set_title("Waveform", fontsize=10, pad=8, loc="left")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return fig

"""
Biquad filter design — Audio EQ Cookbook (Robert Bristow-Johnson) formulas.

Every filter here is a second-order IIR section ("biquad") of the form

    H(z) = (b0 + b1 z^-1 + b2 z^-2) / (a0 + a1 z^-1 + a2 z^-2)

which in the time domain is the difference equation

    y[n] = (b0/a0)x[n] + (b1/a0)x[n-1] + (b2/a0)x[n-2]
           - (a1/a0)y[n-1] - (a2/a0)y[n-2]

Coefficients are returned already normalised by a0 and packed in SciPy's
second-order-section layout: [b0, b1, b2, 1.0, a1, a2].

Nothing in this module touches the FFT. The gain shaping is done by placing
poles and zeros, which is what a real hardware/plugin equaliser does.
"""

from __future__ import annotations

import numpy as np

# A biquad is only well-behaved for centre frequencies strictly below Nyquist.
# Bilinear-transform prewarping blows up as f0 -> fs/2, so clamp a little short.
_NYQUIST_SAFETY = 0.45


def _prewarp(f0: float, fs: float) -> tuple[float, float, float]:
    """Return (w0, cos w0, sin w0) for a centre frequency clamped below Nyquist."""
    f0 = float(np.clip(f0, 1.0, fs * _NYQUIST_SAFETY))
    w0 = 2.0 * np.pi * f0 / fs
    return w0, np.cos(w0), np.sin(w0)


def peaking(f0: float, gain_db: float, q: float, fs: float) -> np.ndarray:
    """
    Peaking ("bell") EQ section.

    Boosts or cuts a band centred on f0 while leaving DC and Nyquist untouched.
    Q sets how wide the bell is: bandwidth (in octaves) narrows as Q rises.
    """
    a_gain = 10.0 ** (gain_db / 40.0)          # sqrt of linear amplitude gain
    _, cos_w0, sin_w0 = _prewarp(f0, fs)
    alpha = sin_w0 / (2.0 * q)

    b0 = 1.0 + alpha * a_gain
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a_gain
    a0 = 1.0 + alpha / a_gain
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a_gain

    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


def low_shelf(f0: float, gain_db: float, q: float, fs: float) -> np.ndarray:
    """
    Low-shelf section: everything below f0 is lifted/cut by gain_db,
    everything well above f0 is left alone.
    """
    a_gain = 10.0 ** (gain_db / 40.0)
    _, cos_w0, sin_w0 = _prewarp(f0, fs)
    alpha = sin_w0 / (2.0 * q)
    sqrt_a = np.sqrt(a_gain)
    two_sqrt_a_alpha = 2.0 * sqrt_a * alpha

    b0 = a_gain * ((a_gain + 1.0) - (a_gain - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = 2.0 * a_gain * ((a_gain - 1.0) - (a_gain + 1.0) * cos_w0)
    b2 = a_gain * ((a_gain + 1.0) - (a_gain - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (a_gain + 1.0) + (a_gain - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = -2.0 * ((a_gain - 1.0) + (a_gain + 1.0) * cos_w0)
    a2 = (a_gain + 1.0) + (a_gain - 1.0) * cos_w0 - two_sqrt_a_alpha

    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


def high_shelf(f0: float, gain_db: float, q: float, fs: float) -> np.ndarray:
    """
    High-shelf section: everything above f0 is lifted/cut by gain_db,
    everything well below f0 is left alone.
    """
    a_gain = 10.0 ** (gain_db / 40.0)
    _, cos_w0, sin_w0 = _prewarp(f0, fs)
    alpha = sin_w0 / (2.0 * q)
    sqrt_a = np.sqrt(a_gain)
    two_sqrt_a_alpha = 2.0 * sqrt_a * alpha

    b0 = a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = -2.0 * a_gain * ((a_gain - 1.0) + (a_gain + 1.0) * cos_w0)
    b2 = a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (a_gain + 1.0) - (a_gain - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = 2.0 * ((a_gain - 1.0) - (a_gain + 1.0) * cos_w0)
    a2 = (a_gain + 1.0) - (a_gain - 1.0) * cos_w0 - two_sqrt_a_alpha

    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


#: Dispatch table used by the Equalizer to build each band's section.
DESIGNERS = {
    "low_shelf": low_shelf,
    "peaking": peaking,
    "high_shelf": high_shelf,
}


def design(kind: str, f0: float, gain_db: float, q: float, fs: float) -> np.ndarray:
    """Design one section by name. `kind` must be a key of DESIGNERS."""
    try:
        return DESIGNERS[kind](f0, gain_db, q, fs)
    except KeyError:
        raise ValueError(
            f"unknown filter kind {kind!r}; expected one of {sorted(DESIGNERS)}"
        ) from None


def is_stable(sos_section: np.ndarray) -> bool:
    """
    True if the section's poles are strictly inside the unit circle.

    Used as a sanity check: a numerically degenerate design (extreme gain at a
    very low f0 with a low sample rate) could in principle produce a filter
    that rings forever, and we would rather fall back to bypass than emit noise.
    """
    a1, a2 = sos_section[4], sos_section[5]
    # Jury stability test for a 2nd-order denominator z^2 + a1 z + a2.
    return abs(a2) < 1.0 and abs(a1) < 1.0 + a2

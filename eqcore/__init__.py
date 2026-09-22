"""
eqcore — the DSP and audio-I/O layer of the equaliser.

Nothing in this package imports Streamlit; it is a plain library that the GUI
sits on top of, so the filters can be tested (and reused) without a front end.
"""

from . import audio_io, biquad, equalizer, presets

__all__ = ["audio_io", "biquad", "equalizer", "presets"]

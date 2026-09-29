"""
7-Band Parametric Audio Equalizer — GUI.

Launch with:  streamlit run app.py

This module is presentation only. Every filter decision lives in `eqcore`,
which knows nothing about Streamlit; this file wires widgets to it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from eqcore import audio_io, equalizer, presets
from eqcore.audio_io import AudioLoadError
from ui import player, plots

st.set_page_config(
    page_title="7-Band Audio Equalizer",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

UPLOAD_TYPES = ["wav", "mp3", "flac", "ogg", "aiff", "aif"]
DEMO_PATH = Path(__file__).parent / "samples" / "demo.wav"

#: Container used to feed the browser audio element. MP3 is ~17x smaller than
#: WAV, which matters because both the original and the processed audio are
#: inlined into the page as data URIs on every rerun.
PLAYBACK_FORMAT, PLAYBACK_MIME = ("MP3", "audio/mpeg")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def init_state() -> None:
    """Seed every session_state key the widgets below depend on."""
    for band in equalizer.BANDS:
        st.session_state.setdefault(f"gain_{band.key}", 0.0)
    st.session_state.setdefault("preset_select", presets.PRESET_NAMES[0])
    st.session_state.setdefault("clip", None)
    st.session_state.setdefault("clip_id", "")
    st.session_state.setdefault("load_error", "")
    st.session_state.setdefault("last_upload_id", "")


def current_gains() -> dict[str, float]:
    """The gains the sliders are currently set to."""
    return {b.key: float(st.session_state[f"gain_{b.key}"]) for b in equalizer.BANDS}


def apply_preset() -> None:
    """
    Preset dropdown callback.

    Runs before the sliders are re-instantiated on the next rerun, so writing
    into their session_state keys is what actually moves them on screen.
    """
    gains = presets.get(st.session_state["preset_select"])
    for key, value in gains.items():
        st.session_state[f"gain_{key}"] = float(value)


def reset_bands() -> None:
    st.session_state["preset_select"] = presets.PRESET_NAMES[0]
    for band in equalizer.BANDS:
        st.session_state[f"gain_{band.key}"] = 0.0


def set_clip(clip, clip_id: str) -> None:
    st.session_state["clip"] = clip
    st.session_state["clip_id"] = clip_id
    st.session_state["load_error"] = ""
    st.session_state.pop("render_cache", None)
    st.session_state.pop("source_cache", None)
    st.session_state.pop("export_cache", None)


# ---------------------------------------------------------------------------
# Caching
#
# Streamlit reruns this whole script on every widget change. Filtering and
# encoding a multi-megabyte clip each time would make the sliders feel dead,
# so both are memoised on the values they actually depend on.
# ---------------------------------------------------------------------------


def encoded_source(clip) -> bytes:
    """The unmodified clip, encoded once per loaded file."""
    cache = st.session_state.get("source_cache")
    if cache and cache["id"] == st.session_state["clip_id"]:
        return cache["bytes"]

    data = audio_io.encode(clip.samples, clip.sample_rate, PLAYBACK_FORMAT)
    st.session_state["source_cache"] = {"id": st.session_state["clip_id"], "bytes": data}
    return data


def render(clip, gains: dict[str, float]) -> dict:
    """
    Filter the clip and encode it for playback.

    Keyed on the clip plus the rounded gain vector, so dragging a slider back
    to a value you already heard is instant.
    """
    signature = (
        st.session_state["clip_id"],
        tuple(round(gains[k], 3) for k in equalizer.BAND_KEYS),
    )
    cache = st.session_state.get("render_cache")
    if cache and cache["signature"] == signature:
        return cache

    result = equalizer.process(clip.samples, clip.sample_rate, gains)
    payload = {
        "signature": signature,
        "result": result,
        "bytes": audio_io.encode(result.audio, clip.sample_rate, PLAYBACK_FORMAT),
    }
    st.session_state["render_cache"] = payload
    return payload


@st.cache_resource(show_spinner=False)
def export_formats() -> list[str]:
    """
    Containers this libsndfile build can actually write.

    The probe encodes a few samples in each candidate format to prove the
    encoder really exists, so it is far too expensive to repeat on every
    rerun — and the answer cannot change while the process is alive.
    """
    return audio_io.available_export_formats()


def exported(clip, rendered: dict, container: str) -> bytes:
    """
    The processed audio in `container`, encoded as rarely as possible.

    st.download_button needs the bytes up front — it cannot call back when the
    button is pressed — so something has to be encoded on every rerun. Two
    things keep that cheap:

      * when the export container is the one the transport already uses, its
        bytes are reused outright and nothing is encoded at all;
      * otherwise the result is memoised on the same signature `render` uses,
        so only an actual change to the audio or the format re-encodes.
    """
    if container == PLAYBACK_FORMAT:
        return rendered["bytes"]

    signature = (rendered["signature"], container)
    cache = st.session_state.get("export_cache")
    if cache and cache["signature"] == signature:
        return cache["bytes"]

    data = audio_io.encode(rendered["result"].audio, clip.sample_rate, container)
    st.session_state["export_cache"] = {"signature": signature, "bytes": data}
    return data


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def sidebar() -> None:
    sb = st.sidebar
    sb.markdown("### 🎛️ Equalizer")

    # --- Source -----------------------------------------------------------
    sb.markdown("#### 1 · Audio source")
    uploaded = sb.file_uploader(
        "Upload an audio file",
        type=UPLOAD_TYPES,
        help="WAV, MP3, FLAC, OGG or AIFF — up to 15 minutes.",
    )

    if uploaded is not None:
        # file_uploader hands us the same object on every rerun; only decode
        # when the actual file changes.
        upload_id = f"{uploaded.name}:{uploaded.size}"
        if upload_id != st.session_state["last_upload_id"]:
            st.session_state["last_upload_id"] = upload_id
            try:
                clip = audio_io.load(uploaded.getvalue(), uploaded.name)
            except AudioLoadError as exc:
                # Shown in the GUI, not raised into the console.
                st.session_state["load_error"] = str(exc)
                st.session_state["clip"] = None
            else:
                set_clip(clip, upload_id)

    if DEMO_PATH.exists() and sb.button(
        "Load demo clip",
        width="stretch",
        help="A synthetic full-spectrum test tone for trying the EQ.",
    ):
        try:
            clip = audio_io.load(DEMO_PATH.read_bytes(), DEMO_PATH.name)
        except AudioLoadError as exc:
            st.session_state["load_error"] = str(exc)
        else:
            st.session_state["last_upload_id"] = "demo"
            set_clip(clip, "demo")

    if st.session_state["load_error"]:
        sb.error(st.session_state["load_error"], icon="🚫")

    sb.divider()

    # --- Presets ----------------------------------------------------------
    sb.markdown("#### 2 · Preset")
    sb.selectbox(
        "Preset",
        presets.PRESET_NAMES,
        key="preset_select",
        on_change=apply_preset,
        label_visibility="collapsed",
    )

    selected = st.session_state["preset_select"]
    gains = current_gains()
    modified = any(
        abs(gains[k] - v) > 1e-6 for k, v in presets.get(selected).items()
    )
    note = presets.PRESET_NOTES.get(selected, "")
    sb.caption(f"{note}  \n**Edited by hand** — sliders no longer match this preset."
               if modified else note)

    sb.divider()

    # --- Bands ------------------------------------------------------------
    sb.markdown("#### 3 · Bands")
    sb.caption("Each slider sets one biquad's gain. −12 to +12 dB.")

    for band in equalizer.BANDS:
        kind = {"low_shelf": "low shelf", "peaking": "bell",
                "high_shelf": "high shelf"}[band.kind]
        sb.slider(
            band.slider_label,
            min_value=equalizer.GAIN_MIN_DB,
            max_value=equalizer.GAIN_MAX_DB,
            step=0.5,
            key=f"gain_{band.key}",
            format="%+.1f dB",
            help=f"{kind} @ {band.f0:.0f} Hz, Q = {band.q:.2f}",
        )

    sb.button("Reset all bands to 0 dB", on_click=reset_bands, width="stretch")


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------


def welcome() -> None:
    st.title("7-Band Parametric Audio Equalizer")
    st.caption(
        "Cascaded biquad filters — RBJ Audio EQ Cookbook — "
        "designed and applied in real time to your own audio."
    )
    st.info(
        "**Upload an audio file** in the sidebar to begin, or press "
        "**Load demo clip** to try the EQ on a synthetic full-spectrum tone.",
        icon="👈",
    )

    st.markdown("#### The filter chain")
    st.caption(
        "This is the live frequency response of the chain at the current "
        "slider positions. It updates as you move them, before any audio is loaded."
    )
    st.pyplot(plots.eq_curve_figure(current_gains(), 44100), width="stretch")

    with st.expander("What each band does"):
        rows = [
            {
                "Band": b.name,
                "Range": b.span,
                "Filter": {"low_shelf": "Low shelf", "peaking": "Peaking (bell)",
                           "high_shelf": "High shelf"}[b.kind],
                "f₀ (Hz)": f"{b.f0:,.0f}",
                "Q": f"{b.q:.2f}",
            }
            for b in equalizer.BANDS
        ]
        st.dataframe(rows, hide_index=True, width="stretch")


def main_panel(clip) -> None:
    gains_now = current_gains()
    rendered = render(clip, gains_now)
    result = rendered["result"]

    st.title("7-Band Parametric Audio Equalizer")

    # --- Clip summary -----------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("File", clip.name if len(clip.name) < 18 else clip.name[:15] + "…")
    c2.metric("Duration", clip.duration_str)
    c3.metric("Sample rate", f"{clip.sample_rate / 1000:g} kHz")
    c4.metric("Channels", "Stereo" if clip.channels == 2 else
              ("Mono" if clip.channels == 1 else f"{clip.channels} ch"))
    active = sum(1 for v in gains_now.values() if abs(v) > 1e-6)
    c5.metric("Active bands", f"{active} / {len(equalizer.BANDS)}")

    # --- Transport --------------------------------------------------------
    st.markdown("#### Playback")
    # The restore token identifies the *clip*, not the EQ settings. Editing a
    # band re-renders the component, and we want playback to pick up where it
    # left off; loading a different file must not restore the old playhead.
    components.html(
        player.render_html(
            original_bytes=encoded_source(clip),
            processed_bytes=rendered["bytes"],
            mime=PLAYBACK_MIME,
            token=hashlib.md5(st.session_state["clip_id"].encode()).hexdigest()[:16],
        ),
        height=player.PLAYER_HEIGHT,
    )

    if result.clipped:
        st.warning(
            f"Boosting pushed the peak to {result.peak_before:.2f} "
            f"(full scale is 1.00), so the output was attenuated by "
            f"{result.applied_gain_db:.1f} dB to stay clean. "
            "The EQ shape is unchanged — only the overall level moved.",
            icon="📉",
        )

    # --- Export -----------------------------------------------------------
    formats = export_formats()
    e1, e2 = st.columns([1, 3])
    export_format = e1.selectbox("Export format", formats, index=0)

    stem = Path(clip.name).stem or "audio"
    export_bytes = exported(clip, rendered, export_format)
    e2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    e2.download_button(
        f"⬇  Export EQ'd audio  ({len(export_bytes) / 1_048_576:.1f} MB)",
        data=export_bytes,
        file_name=f"{stem}_eq.{export_format.lower()}",
        mime=f"audio/{'mpeg' if export_format == 'MP3' else export_format.lower()}",
        width="stretch",
    )

    st.divider()

    # --- Visuals ----------------------------------------------------------
    left, right = st.columns(2)
    with left:
        st.pyplot(plots.eq_curve_figure(gains_now, clip.sample_rate), width="stretch")
    with right:
        st.pyplot(
            plots.spectrum_figure(clip.samples, result.audio, clip.sample_rate),
            width="stretch",
        )

    st.pyplot(
        plots.waveform_figure(clip.samples, result.audio, clip.sample_rate),
        width="stretch",
    )

    # --- What the chain is actually doing ---------------------------------
    with st.expander("Filter chain detail — the coefficients being applied"):
        sos = equalizer.build_sos(gains_now, clip.sample_rate)
        st.caption(
            f"{sos.shape[0]} second-order section(s) applied in series via "
            "`scipy.signal.sosfilt`. Bands at 0 dB are omitted because a 0 dB "
            "biquad is the identity filter."
        )
        rows = []
        for band in equalizer.BANDS:
            g = gains_now[band.key]
            rows.append({
                "Band": band.name,
                "Type": {"low_shelf": "Low shelf", "peaking": "Peaking",
                         "high_shelf": "High shelf"}[band.kind],
                "f₀ (Hz)": f"{band.f0:,.0f}",
                "Q": f"{band.q:.2f}",
                "Gain (dB)": f"{g:+.1f}",
                "In chain": "yes" if abs(g) > 1e-3 else "bypassed",
            })
        st.dataframe(rows, hide_index=True, width="stretch")

        st.caption("Cascaded SOS matrix — columns are [b0, b1, b2, a0, a1, a2]:")
        st.dataframe(
            [
                {
                    "b0": f"{r[0]: .6f}", "b1": f"{r[1]: .6f}", "b2": f"{r[2]: .6f}",
                    "a0": f"{r[3]: .6f}", "a1": f"{r[4]: .6f}", "a2": f"{r[5]: .6f}",
                }
                for r in sos
            ],
            hide_index=True,
            width="stretch",
        )

        peak_db = 20 * np.log10(max(result.peak_before, 1e-12))
        st.caption(
            f"Filtered peak before gain staging: {result.peak_before:.4f} "
            f"({peak_db:+.2f} dBFS) · "
            f"attenuation applied: {result.applied_gain_db:.2f} dB"
        )


def main() -> None:
    init_state()
    sidebar()
    clip = st.session_state["clip"]
    if clip is None:
        welcome()
    else:
        main_panel(clip)


main()

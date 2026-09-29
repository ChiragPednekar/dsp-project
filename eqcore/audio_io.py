"""
Audio file loading and encoding.

Reading and writing is done with libsndfile (via `soundfile`), which handles
WAV natively and MP3 from libsndfile 1.1 onwards. Every failure mode here
raises AudioLoadError with a message written for a user to read, because the
GUI shows that message directly instead of letting a traceback escape.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from . import spec

#: Refuse absurdly long files rather than melting the machine on a bad upload.
MAX_DURATION_SECONDS = spec.MAX_DURATION_SECONDS


class AudioLoadError(Exception):
    """Raised when an uploaded file cannot be decoded into usable audio."""


@dataclass
class AudioClip:
    """A decoded audio file, always float32 and always peak-sane."""

    samples: np.ndarray     # (frames,) mono or (frames, channels)
    sample_rate: int
    name: str
    subtype: str = "unknown"

    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else int(self.samples.shape[1])

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration(self) -> float:
        return self.frames / float(self.sample_rate)

    @property
    def duration_str(self) -> str:
        total = round(self.duration)
        return f"{total // 60}:{total % 60:02d}"


def _clean_decoder_message(exc: Exception) -> str:
    """
    Turn libsndfile's error text into something worth showing a user.

    It reports failures as `Error opening <_io.BytesIO object at 0x...>: Format
    not recognised.` — the buffer repr is noise, and the memory address changes
    every run, so strip the prefix and keep the reason.
    """
    text = str(exc).strip()
    marker = ">: "
    if text.startswith("Error opening") and marker in text:
        text = text.split(marker, 1)[1]
    return text.rstrip(".") or "unknown decoding error"


def _mp3_hint(name: str) -> str:
    """
    Suggest converting to WAV, but only when MP3 really is unsupported.

    libsndfile gained MP3 decoding in 1.1; on an older build every .mp3 fails
    with the same "format not recognised" as genuine corruption, and only then
    is "convert it to WAV" useful advice rather than a red herring.
    """
    if name.lower().endswith(".mp3") and "MP3" not in sf.available_formats():
        return (
            f" This build of libsndfile ({sf.__libsndfile_version__}) has no "
            f"MP3 support — converting the file to WAV will work."
        )
    return ""


def load(data: bytes, name: str = "audio") -> AudioClip:
    """
    Decode raw file bytes into an AudioClip.

    Always returns float32 in roughly [-1, 1]; libsndfile scales integer PCM
    for us. Raises AudioLoadError (never a bare soundfile error) so the GUI has
    exactly one exception type to catch and display.
    """
    if not data:
        raise AudioLoadError("That file is empty — there is no audio to load.")

    try:
        samples, sample_rate = sf.read(
            io.BytesIO(data), dtype="float32", always_2d=False
        )
    except Exception as exc:
        raise AudioLoadError(
            f"Could not decode “{name}” as audio — {_clean_decoder_message(exc)}. "
            f"Supported formats are WAV, FLAC, OGG and MP3.{_mp3_hint(name)}"
        ) from exc

    if samples.size == 0:
        raise AudioLoadError(f"“{name}” decoded to zero samples — the file has no audio.")

    if sample_rate <= 0:
        raise AudioLoadError(
            f"“{name}” reports an invalid sample rate ({sample_rate} Hz)."
        )

    duration = samples.shape[0] / float(sample_rate)
    if duration > MAX_DURATION_SECONDS:
        raise AudioLoadError(
            f"“{name}” is {duration / 60:.1f} minutes long. "
            f"Please use a clip under {MAX_DURATION_SECONDS // 60} minutes."
        )

    # Guard against non-finite samples, which would poison the filter state
    # and turn the whole output into NaN.
    if not np.all(np.isfinite(samples)):
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        subtype = sf.info(io.BytesIO(data)).subtype or "unknown"
    except Exception:
        subtype = "unknown"

    return AudioClip(
        samples=np.ascontiguousarray(samples, dtype=np.float32),
        sample_rate=int(sample_rate),
        name=name,
        subtype=subtype,
    )


def encode_wav(samples: np.ndarray, sample_rate: int, subtype: str = "PCM_16") -> bytes:
    """
    Encode samples to an in-memory WAV.

    Used for two things: feeding the browser audio element, and backing the
    export download. Samples are hard-clipped to [-1, 1] first — the EQ's gain
    staging should already have prevented overs, but an integer PCM encoder
    wraps around rather than saturating, so this is a cheap safety net.
    """
    audio = np.asarray(samples, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)

    buffer = io.BytesIO()
    sf.write(buffer, audio, int(sample_rate), format="WAV", subtype=subtype)
    return buffer.getvalue()


def encode(samples: np.ndarray, sample_rate: int, container: str) -> bytes:
    """
    Encode to a chosen container for export. Falls back to WAV if the
    requested format is not available in this libsndfile build.
    """
    audio = np.clip(
        np.nan_to_num(np.asarray(samples, dtype=np.float32)), -1.0, 1.0
    )
    container = container.upper()

    if container == "WAV":
        return encode_wav(audio, sample_rate)

    subtypes = {"FLAC": "PCM_16", "OGG": "VORBIS", "MP3": "MPEG_LAYER_III"}
    buffer = io.BytesIO()
    try:
        sf.write(
            buffer,
            audio,
            int(sample_rate),
            format=container,
            subtype=subtypes.get(container),
        )
        return buffer.getvalue()
    except Exception:
        return encode_wav(audio, sample_rate)


def available_export_formats() -> list[str]:
    """
    Export containers this libsndfile build can actually write.

    Presence in `available_formats()` is necessary but not sufficient — MP3 in
    particular is listed even when the build lacks a LAME encoder — so each
    candidate is proved by encoding a few samples for real.
    """
    listed = sf.available_formats()
    probe = np.zeros(64, dtype=np.float32)
    usable: list[str] = []

    for name in ("WAV", "FLAC", "OGG", "MP3"):
        if name not in listed:
            continue
        try:
            sf.write(
                io.BytesIO(), probe, 44100, format=name,
                subtype={"FLAC": "PCM_16", "OGG": "VORBIS",
                         "MP3": "MPEG_LAYER_III"}.get(name, "PCM_16"),
            )
        except Exception:
            continue
        usable.append(name)

    return usable or ["WAV"]

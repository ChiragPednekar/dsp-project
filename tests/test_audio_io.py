"""Decoding, encoding, and the user-facing error messages."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from eqcore import audio_io
from eqcore.audio_io import AudioClip, AudioLoadError

FS = 44100


def wav_bytes(samples: np.ndarray, fs: int = FS, subtype: str = "PCM_16") -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, fs, format="WAV", subtype=subtype)
    return buffer.getvalue()


def tone(seconds: float = 0.5, channels: int = 1, fs: int = FS) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    mono = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return mono if channels == 1 else np.stack([mono] * channels, axis=1)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_round_trip_preserves_the_samples():
    original = tone()
    clip = audio_io.load(wav_bytes(original), "tone.wav")

    assert clip.sample_rate == FS
    assert clip.channels == 1
    assert clip.frames == original.size
    # PCM_16 quantises, so compare within one LSB rather than exactly.
    assert np.max(np.abs(clip.samples - original)) < 2 ** -15


def test_stereo_shape_is_preserved():
    clip = audio_io.load(wav_bytes(tone(channels=2)), "stereo.wav")
    assert clip.channels == 2
    assert clip.samples.ndim == 2 and clip.samples.shape[1] == 2


def test_samples_are_float32():
    clip = audio_io.load(wav_bytes(tone()), "tone.wav")
    assert clip.samples.dtype == np.float32


def test_empty_file_is_rejected_with_a_readable_message():
    with pytest.raises(AudioLoadError, match="empty"):
        audio_io.load(b"", "nothing.wav")


def test_undecodable_bytes_are_rejected():
    with pytest.raises(AudioLoadError) as excinfo:
        audio_io.load(b"this is definitely not audio" * 40, "notes.txt")
    assert "notes.txt" in str(excinfo.value)


def test_the_error_message_does_not_leak_a_buffer_repr():
    """
    libsndfile reports failures as `Error opening <_io.BytesIO object at 0x..>`.
    That address changes every run and means nothing to a user.
    """
    with pytest.raises(AudioLoadError) as excinfo:
        audio_io.load(b"junk" * 100, "broken.wav")
    message = str(excinfo.value)
    assert "BytesIO" not in message
    assert "0x" not in message


def test_overlong_audio_is_rejected(monkeypatch):
    monkeypatch.setattr(audio_io, "MAX_DURATION_SECONDS", 1)
    with pytest.raises(AudioLoadError, match="minutes"):
        audio_io.load(wav_bytes(tone(seconds=2.0)), "long.wav")


def test_non_finite_samples_are_scrubbed():
    """NaN in the input would poison the filter state and blank the output."""
    poisoned = tone(seconds=0.1).copy()
    poisoned[10] = np.nan
    poisoned[20] = np.inf

    # float32 WAV round-trips the non-finite values that PCM would clamp away.
    clip = audio_io.load(wav_bytes(poisoned, subtype="FLOAT"), "poisoned.wav")
    assert np.all(np.isfinite(clip.samples))


# --------------------------------------------------------------------------
# AudioClip
# --------------------------------------------------------------------------


def test_clip_reports_duration_in_minutes_and_seconds():
    clip = AudioClip(
        samples=np.zeros(FS * 75, dtype=np.float32), sample_rate=FS, name="x"
    )
    assert clip.duration == pytest.approx(75.0)
    assert clip.duration_str == "1:15"


def test_clip_channel_count_for_mono_and_stereo():
    mono = AudioClip(samples=np.zeros(10, dtype=np.float32), sample_rate=FS, name="m")
    stereo = AudioClip(
        samples=np.zeros((10, 2), dtype=np.float32), sample_rate=FS, name="s"
    )
    assert mono.channels == 1
    assert stereo.channels == 2


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def test_encoded_wav_decodes_back():
    original = tone()
    decoded = audio_io.load(audio_io.encode_wav(original, FS), "out.wav")
    assert decoded.frames == original.size
    assert decoded.sample_rate == FS


def test_encoding_hard_clips_rather_than_wrapping():
    """
    Integer PCM wraps on overflow, which turns a loud peak into a nasty click.
    Anything out of range must saturate instead.
    """
    hot = np.array([2.0, -2.0, 0.0, 0.5], dtype=np.float32)
    decoded = audio_io.load(audio_io.encode_wav(hot, FS), "hot.wav")
    assert np.max(decoded.samples) <= 1.0
    assert np.min(decoded.samples) >= -1.0
    assert decoded.samples[0] > 0.9   # saturated positive, not wrapped negative
    assert decoded.samples[1] < -0.9


def test_encoding_scrubs_non_finite_samples():
    data = np.array([np.nan, np.inf, -np.inf, 0.25], dtype=np.float32)
    decoded = audio_io.load(audio_io.encode_wav(data, FS), "nan.wav")
    assert np.all(np.isfinite(decoded.samples))


def test_wav_is_always_an_available_export_format():
    formats = audio_io.available_export_formats()
    assert "WAV" in formats
    assert formats[0] == "WAV"


def test_every_advertised_format_actually_encodes():
    """
    available_formats() lists containers the build cannot always write — MP3
    is listed without a LAME encoder — so the probe must be a real encode.
    """
    data = tone(seconds=0.2)
    for fmt in audio_io.available_export_formats():
        payload = audio_io.encode(data, FS, fmt)
        assert payload, f"{fmt} produced no bytes"


def test_unknown_container_falls_back_to_wav():
    payload = audio_io.encode(tone(seconds=0.1), FS, "AIFF_NOT_SUPPORTED_HERE")
    assert payload[:4] == b"RIFF"

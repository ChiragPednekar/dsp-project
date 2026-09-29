"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

FS = 44100


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


def sine(freq: float, seconds: float = 2.0, amp: float = 0.25, fs: int = FS):
    """A plain test tone."""
    t = np.arange(int(seconds * fs)) / fs
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def steady_state_amplitude(x: np.ndarray) -> float:
    """
    Peak-equivalent amplitude of the back half of a signal.

    Measuring the tail excludes the filter's start-up transient, which would
    otherwise drag an RMS measurement away from the steady-state gain.
    """
    tail = x[len(x) // 2:]
    return float(np.sqrt(np.mean(np.square(tail))) * np.sqrt(2.0))


def gain_db(processed: np.ndarray, original: np.ndarray) -> float:
    """Measured gain, in dB, between two tones of the same frequency."""
    return 20.0 * np.log10(
        steady_state_amplitude(processed) / steady_state_amplitude(original)
    )

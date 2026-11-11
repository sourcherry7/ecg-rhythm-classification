"""Shared fixtures: simulated ECG recordings (no competition data needed) and the ``needs_data`` auto-skip."""

import numpy as np
import pytest

from ecg_rhythm.data import RAW_DIR, RAW_FILES

SAMPLING_RATE = 300


def pytest_collection_modifyitems(config, items):
    if all((RAW_DIR / name).exists() for name in RAW_FILES):
        return
    skip = pytest.mark.skip(reason="competition data not present under data/raw")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def ecg_signals() -> list[np.ndarray]:
    """Four 30 s simulated recordings at 300 Hz with different heart rates (NeuroKit2's ECGSYN model)."""
    nk = pytest.importorskip("neurokit2")
    signals = []
    for i, heart_rate in enumerate((60, 75, 90, 110)):
        signal = nk.ecg_simulate(
            duration=30, sampling_rate=SAMPLING_RATE, heart_rate=heart_rate, noise=0.05, random_state=i
        )
        signals.append(np.asarray(signal, dtype=np.float32))
    return signals

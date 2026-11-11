import numpy as np

from ecg_rhythm.config import PreprocessConfig
from ecg_rhythm.preprocessing import denoise, preprocess_signals, remove_baseline_drift, wavelet_denoise


def test_baseline_drift_is_removed(ecg_signals):
    x = ecg_signals[0].astype(float)
    t = np.arange(len(x)) / 300
    drifted = x + 2.0 * np.sin(2 * np.pi * 0.1 * t) + 5.0
    cleaned = remove_baseline_drift(drifted, sampling_rate=300)
    assert len(cleaned) == len(x)
    assert abs(np.mean(cleaned)) < 0.1 * abs(np.mean(drifted))


def test_wavelet_denoise_keeps_length_and_reduces_noise(ecg_signals):
    rng = np.random.default_rng(0)
    x = ecg_signals[1].astype(float)
    noisy = x + rng.normal(scale=0.2, size=len(x))
    cleaned = wavelet_denoise(noisy)
    assert len(cleaned) == len(x)
    assert np.mean((cleaned - x) ** 2) < np.mean((noisy - x) ** 2)


def test_full_chain_and_passthrough(ecg_signals):
    cfg_on, cfg_off = PreprocessConfig(denoise=True), PreprocessConfig(denoise=False)
    padded = np.concatenate([ecg_signals[2], np.full(10, np.nan, dtype=np.float32)])
    cleaned = denoise(padded, cfg_on)
    assert cleaned.shape == (len(ecg_signals[2]),) and np.isfinite(cleaned).all()
    untouched = preprocess_signals([ecg_signals[2]], cfg_off)
    np.testing.assert_array_equal(untouched[0], ecg_signals[2].astype(float))
    assert preprocess_signals([ecg_signals[2]], cfg_on, n_jobs=1)[0].shape == cleaned.shape

"""Optional signal cleaning.

This is the denoising the team added on the last weekend of the competition (median-filter baseline removal, a
0.5 Hz Butterworth high-pass, soft-threshold db4 wavelet denoising, then BioSPPy's ECG band-pass). The features behind
the selected submission were computed on the raw recordings, so ``PreprocessConfig.denoise`` defaults to ``False``
and this module is an ablation (preset ``denoised``).
"""

from __future__ import annotations

import numpy as np
import pywt
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt, medfilt

from .config import SAMPLING_RATE_HZ, PreprocessConfig


def wavelet_denoise(x: np.ndarray, wavelet: str = "db4") -> np.ndarray:
    """Universal soft threshold on all detail levels (noise level from the finest level's median absolute value)."""
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=None)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(x)))
    denoised = [coeffs[0]] + [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(denoised, wavelet=wavelet)[: len(x)]


def remove_baseline_drift(
    x: np.ndarray,
    sampling_rate: int = SAMPLING_RATE_HZ,
    order: int = 4,
    median_kernel: int = 301,
    cutoff_hz: float = 0.5,
) -> np.ndarray:
    """Subtract a running median (the slow baseline), then high-pass what is left."""
    baseline = medfilt(x, kernel_size=median_kernel)
    b, a = butter(order, cutoff_hz / (0.5 * sampling_rate), btype="high", analog=False)
    return filtfilt(b, a, x - baseline)


def biosppy_filter(x: np.ndarray, sampling_rate: int = SAMPLING_RATE_HZ) -> np.ndarray:
    """BioSPPy's standard ECG band-pass (FIR, 3 to 45 Hz)."""
    from biosppy.signals import ecg

    return np.asarray(ecg.ecg(signal=x, sampling_rate=sampling_rate, show=False)["filtered"], dtype=float)


def denoise(x: np.ndarray, cfg: PreprocessConfig, sampling_rate: int = SAMPLING_RATE_HZ) -> np.ndarray:
    """The team's full cleaning chain for one recording (NaN padding removed first)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x = remove_baseline_drift(x, sampling_rate, cfg.butter_order, cfg.median_kernel, cfg.highpass_hz)
    x = wavelet_denoise(x, cfg.wavelet)
    return biosppy_filter(x, sampling_rate)


def preprocess_signals(
    signals: list[np.ndarray], cfg: PreprocessConfig, sampling_rate: int = SAMPLING_RATE_HZ, n_jobs: int = 1
) -> list[np.ndarray]:
    """Apply ``denoise`` to every recording when ``cfg.denoise`` is set; otherwise return the recordings unchanged."""
    if not cfg.denoise:
        return [np.asarray(s, dtype=float) for s in signals]
    return list(Parallel(n_jobs=n_jobs)(delayed(denoise)(s, cfg, sampling_rate) for s in signals))

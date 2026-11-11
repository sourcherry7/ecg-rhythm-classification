"""Per-recording features and the mean heartbeat: a faithful port of the team's ``feature_extraction.py``.

Three 2024 quirks are kept on purpose and exposed as options (see ``FeatureConfig`` and the README caveats):

* NeuroKit2 and Welch are told the recordings are sampled at 250 Hz although they are 300 Hz
  (``FeatureConfig.sampling_rate``).
* A recording that NeuroKit2 cannot process gets a copy of the previous recording's features
  (``on_error="previous"``); about 2 % of the recordings are affected.
* The "QRS interval" features are computed from NeuroKit2's 0/1 R-peak indicator column rather than from the peak
  positions, so they carry almost no information. They are kept because the submitted model saw them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal as sps
from scipy import stats
from tqdm.auto import tqdm

from .config import FeatureConfig


def hr_features(x: np.ndarray, sampling_rate: int) -> pd.Series:
    """All features of the 2024 ``extract_hr_features`` for one recording (raises when NeuroKit2 fails on it)."""
    import neurokit2 as nk

    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    signals, _ = nk.ecg_process(x, sampling_rate=sampling_rate)
    hrv = pd.concat(
        [
            nk.hrv_time(signals, sampling_rate=sampling_rate),
            nk.hrv_frequency(signals, sampling_rate=sampling_rate),
            nk.hrv_nonlinear(signals, sampling_rate=sampling_rate),
        ],
        axis=1,
    ).iloc[0]

    statistical = {
        "mean": np.mean(x),
        "std": np.std(x),
        "skew": stats.skew(x),
        "kurtosis": stats.kurtosis(x),
        "rms": np.sqrt(np.mean(np.square(x))),
        "peak_to_peak": np.max(x) - np.min(x),
        "zero_crossings": float((x[:-1] * x[1:] < 0).sum()),
        "above_mean": (x > np.mean(x)).sum() / len(x),
    }
    freqs, psd = sps.welch(x, fs=sampling_rate)
    spectral = {
        "dominant_freq": freqs[np.argmax(psd)],
        "mean_freq": np.average(freqs, weights=psd),
        "median_freq": freqs[np.argwhere(np.cumsum(psd) >= 0.5 * np.sum(psd))[0][0]],
        "power_total": np.sum(psd),
        "power_ratio": np.sum(psd[freqs > 10]) / np.sum(psd[freqs <= 10]),
    }
    sampen = nk.entropy_sample(x)
    apen = nk.entropy_approximate(x)
    p = x / np.sum(x)
    entropy = {
        "sample_entropy": sampen[0] if isinstance(sampen, tuple) else sampen,
        "approximate_entropy": apen[0] if isinstance(apen, tuple) else apen,
        "shannon_entropy": -np.sum(np.square(p) * np.log2(np.square(p) + 1e-10)),
    }
    peaks = signals["ECG_R_Peaks"]  # the 0/1 indicator column, exactly as in 2024
    if len(peaks) > 1:
        intervals = np.diff(peaks)
        qrs = {
            "qrs_mean_interval": np.mean(intervals),
            "qrs_std_interval": np.std(intervals),
            "qrs_rmssd": np.sqrt(np.mean(np.square(np.diff(intervals)))),
            "qrs_density": len(peaks) / len(x),
        }
    else:
        qrs = {"qrs_mean_interval": 0.0, "qrs_std_interval": 0.0, "qrs_rmssd": 0.0, "qrs_density": 0.0}
    quality = {"signal_quality_index": np.mean(signals["ECG_Quality"]), "noise_ratio": np.std(x) / np.ptp(x)}
    parts = [hrv, pd.Series(statistical), pd.Series(spectral), pd.Series(qrs), pd.Series(quality), pd.Series(entropy)]
    return pd.concat(parts).astype(float)


def _run_parallel(tasks, n_jobs: int, total: int, desc: str, progress: bool):
    """Run joblib tasks, streaming results when joblib supports it (>= 1.3) and collecting them otherwise."""
    try:
        results = Parallel(n_jobs=n_jobs, return_as="generator")(tasks)
    except TypeError:  # older joblib (e.g. the 2024 environment): no streaming, same results
        results = Parallel(n_jobs=n_jobs)(tasks)
    return tqdm(results, total=total, disable=not progress, desc=desc)


def _safe_hr_features(index: int, x: np.ndarray, sampling_rate: int) -> tuple[int, pd.Series | None, str | None]:
    warnings.filterwarnings("ignore")
    try:
        return index, hr_features(x, sampling_rate), None
    except Exception as exc:  # noqa: BLE001 - any NeuroKit2/numpy failure counts as "could not process"
        return index, None, f"{type(exc).__name__}: {exc}"


@dataclass
class FeatureTable:
    """Feature rows aligned with the input recordings, plus which recordings could not be processed."""

    frame: pd.DataFrame
    failed: dict[int, str] = field(default_factory=dict)


def extract_features(
    signals: list[np.ndarray], cfg: FeatureConfig, n_jobs: int | None = None, progress: bool = True
) -> FeatureTable:
    """Run ``hr_features`` on every recording in parallel and apply the configured failure policy."""
    n_jobs = cfg.n_jobs if n_jobs is None else n_jobs
    tasks = (delayed(_safe_hr_features)(i, x, cfg.sampling_rate) for i, x in enumerate(signals))
    rows: list[pd.Series | None] = [None] * len(signals)
    failed: dict[int, str] = {}
    for index, row, error in _run_parallel(tasks, n_jobs, len(signals), "features", progress):
        rows[index] = row
        if error is not None:
            failed[index] = error
    # A DataFrame built from Series takes the union of their indexes in order of first appearance, which is how the
    # 2024 tables got their column order (the long-signal HRV_DFA_alpha2 columns come last).
    frame = pd.DataFrame([row if row is not None else pd.Series(dtype=float) for row in rows]).astype(float)
    frame.index = pd.RangeIndex(len(signals))
    if cfg.on_error == "previous":
        for index in sorted(failed):
            frame.iloc[index] = frame.iloc[index - 1] if index > 0 else 0.0
    elif cfg.on_error != "nan":
        raise ValueError(f"unknown on_error policy {cfg.on_error!r}")
    return FeatureTable(frame, failed)


def mean_wave(x: np.ndarray, sampling_rate: int, window_s: float = 0.5) -> np.ndarray:
    """Average of the fixed-length segments around the detected R-peaks (``2 * int(window_s * rate)`` samples)."""
    import neurokit2 as nk

    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    _, info = nk.ecg_peaks(x, sampling_rate=sampling_rate)
    window = int(window_s * sampling_rate)
    segments = [x[p - window : p + window] for p in info["ECG_R_Peaks"] if p - window >= 0 and p + window < len(x)]
    return np.mean(np.stack(segments), axis=0)  # raises ValueError when no complete segment exists, as in 2024


def _safe_mean_wave(index: int, x: np.ndarray, sampling_rate: int, window_s: float):
    warnings.filterwarnings("ignore")
    try:
        return index, mean_wave(x, sampling_rate, window_s), None
    except Exception as exc:  # noqa: BLE001
        return index, None, f"{type(exc).__name__}: {exc}"


def mean_waves(
    signals: list[np.ndarray], cfg: FeatureConfig, n_jobs: int | None = None, progress: bool = True
) -> tuple[np.ndarray, dict[int, str]]:
    """Mean heartbeat of every recording; rows of recordings without a usable beat are NaN and listed in the dict."""
    n_jobs = cfg.n_jobs if n_jobs is None else n_jobs
    length = 2 * int(cfg.mean_wave_window_s * cfg.mean_wave_sampling_rate)
    tasks = (
        delayed(_safe_mean_wave)(i, x, cfg.mean_wave_sampling_rate, cfg.mean_wave_window_s)
        for i, x in enumerate(signals)
    )
    waves = np.full((len(signals), length), np.nan)
    failed: dict[int, str] = {}
    for index, wave, error in _run_parallel(tasks, n_jobs, len(signals), "mean waves", progress):
        if error is None:
            waves[index] = wave
        else:
            failed[index] = error
    return waves, failed


def fill_wave_failures(waves: np.ndarray, failed: dict[int, str]) -> np.ndarray:
    """Replace a failed row by its successor's wave (the 2024 notebook swapped test recording 700 for 701 by hand)."""
    waves = waves.copy()
    for index in sorted(failed):
        candidates = [j for j in (index + 1, index - 1) if 0 <= j < len(waves) and j not in failed]
        if candidates:
            waves[index] = waves[candidates[0]]
    return waves


@dataclass
class FeatureCleaner:
    """The 2024 ``clean_features``: drop mostly-NaN columns, inf -> NaN, fill with the column mean, clip."""

    nan_column_threshold: float = 0.5
    clip: float = 1e15
    columns_: list[str] | None = None
    means_: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> FeatureCleaner:
        keep = frame.columns[frame.isna().mean() < self.nan_column_threshold]
        finite = frame[keep].replace([np.inf, -np.inf], np.nan)
        self.columns_ = list(keep)
        self.means_ = finite.mean()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.columns_ is None or self.means_ is None:
            raise RuntimeError("fit the cleaner first")
        finite = frame.reindex(columns=self.columns_).replace([np.inf, -np.inf], np.nan)
        return finite.fillna(self.means_).clip(-self.clip, self.clip).to_numpy(dtype=float)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


def clean_tables(
    train: pd.DataFrame, test: pd.DataFrame, cfg: FeatureConfig
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Clean both tables, per table as in 2024 (``clean_per_table``) or with a cleaner fitted on the training table."""
    cleaner = FeatureCleaner(cfg.nan_column_threshold, cfg.clip)
    X_train = cleaner.fit_transform(train)
    columns = list(cleaner.columns_ or [])
    if cfg.clean_per_table:
        test_cleaner = FeatureCleaner(cfg.nan_column_threshold, cfg.clip).fit(test)
        if test_cleaner.columns_ != columns:
            raise ValueError(
                "per-table cleaning would keep different columns for train and test; "
                "set features.clean_per_table=false to fit the cleaner on the training table only"
            )
        X_test = test_cleaner.transform(test)
    else:
        X_test = cleaner.transform(test)
    return X_train, X_test, columns

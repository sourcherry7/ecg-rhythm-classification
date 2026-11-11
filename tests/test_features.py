import numpy as np
import pandas as pd
import pytest

from ecg_rhythm.config import FeatureConfig
from ecg_rhythm.features import (
    FeatureCleaner,
    clean_tables,
    extract_features,
    fill_wave_failures,
    hr_features,
    mean_wave,
    mean_waves,
)

EXPECTED = {
    "HRV_MeanNN", "HRV_RMSSD", "HRV_LF", "HRV_SD1", "HRV_SampEn", "mean", "std", "skew", "dominant_freq",
    "power_ratio", "qrs_density", "signal_quality_index", "noise_ratio", "sample_entropy", "shannon_entropy",
}  # fmt: skip


def test_hr_features_has_the_2024_columns(ecg_signals):
    row = hr_features(ecg_signals[0], sampling_rate=250)
    assert EXPECTED <= set(row.index)
    assert row.dtype == float and np.isfinite(row["mean"]) and 0 < row["signal_quality_index"] <= 1
    assert row["qrs_density"] == pytest.approx(1.0)  # the 2024 quirk: computed from the 0/1 indicator column


def test_extract_features_failure_policies(ecg_signals):
    signals = [ecg_signals[0], np.zeros(12, dtype=np.float32), ecg_signals[1]]
    previous = extract_features(signals, FeatureConfig(on_error="previous", n_jobs=1), progress=False)
    assert list(previous.failed) == [1] and previous.frame.shape[0] == 3
    pd.testing.assert_series_equal(previous.frame.iloc[1], previous.frame.iloc[0], check_names=False)
    as_nan = extract_features(signals, FeatureConfig(on_error="nan", n_jobs=1), progress=False)
    assert as_nan.frame.iloc[1].isna().all() and not as_nan.frame.iloc[2].isna().all()
    assert list(as_nan.frame.columns) == list(previous.frame.columns)


def test_mean_wave_length_and_fallback(ecg_signals):
    assert len(mean_wave(ecg_signals[0], sampling_rate=250)) == 250
    assert len(mean_wave(ecg_signals[0], sampling_rate=300)) == 300
    with pytest.raises(Exception):  # noqa: B017 - NeuroKit2 raises different types; any failure is handled alike
        mean_wave(np.zeros(50), sampling_rate=250)
    waves, failed = mean_waves([ecg_signals[0], np.zeros(50), ecg_signals[1]], FeatureConfig(n_jobs=1), progress=False)
    assert list(failed) == [1] and np.isnan(waves[1]).all()
    filled = fill_wave_failures(waves, failed)
    np.testing.assert_array_equal(filled[1], waves[2])  # the successor, like test recording 700 -> 701 in 2024


def test_feature_cleaner_drops_fills_and_clips():
    frame = pd.DataFrame(
        {
            "mostly_nan": [np.nan, np.nan, np.nan, 1.0, 2.0],
            "with_inf": [1.0, np.inf, 3.0, -np.inf, 5.0],
            "huge": [1e20, 0.0, 0.0, 0.0, 0.0],
        }
    )
    cleaner = FeatureCleaner(nan_column_threshold=0.5, clip=1e15).fit(frame)
    assert cleaner.columns_ == ["with_inf", "huge"]
    cleaned = cleaner.transform(frame)
    assert cleaned.shape == (5, 2) and np.isfinite(cleaned).all()
    assert cleaned[1, 0] == pytest.approx(3.0) and cleaned[0, 1] == 1e15
    other = frame.assign(with_inf=[np.nan] * 5)
    with pytest.raises(ValueError):
        clean_tables(frame, other, FeatureConfig(clean_per_table=True))
    X_train, X_test, columns = clean_tables(frame, other, FeatureConfig(clean_per_table=False))
    assert columns == ["with_inf", "huge"] and X_test.shape == (5, 2)

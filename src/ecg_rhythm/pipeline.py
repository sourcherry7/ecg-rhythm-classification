"""End-to-end assembly: recordings -> (optional cleaning) -> feature tables + mean heartbeats -> matrix -> classifier.

The expensive part (``build_feature_tables``) is separated from the cheap part (``prepare`` + ``fit_predict``) so that
feature tables can be cached per configuration and reused by the scripts, and so that the team's own 2024 tables can
be plugged in as reproduction checkpoint A.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import PipelineConfig, config_hash
from .data import Signals
from .features import FeatureTable, clean_tables, extract_features, fill_wave_failures, mean_waves
from .preprocessing import preprocess_signals


@dataclass
class FeatureSet:
    """Everything the feature stage produces for one set of recordings."""

    ids: np.ndarray
    table: FeatureTable
    waves: np.ndarray | None  # NaN rows where no mean heartbeat could be computed
    wave_failures: dict[int, str] = field(default_factory=dict)


def build_feature_set(
    signals: Signals, cfg: PipelineConfig, n_jobs: int | None = None, progress: bool = True
) -> FeatureSet:
    cleaned = preprocess_signals(signals.signals, cfg.preprocess, n_jobs=n_jobs or cfg.features.n_jobs)
    table = extract_features(cleaned, cfg.features, n_jobs=n_jobs, progress=progress)
    waves, failures = (None, {})
    if cfg.features.mean_wave:
        waves, failures = mean_waves(cleaned, cfg.features, n_jobs=n_jobs, progress=progress)
    return FeatureSet(signals.ids, table, waves, failures)


@dataclass
class Prepared:
    """The matrices the classifier sees, plus what was dropped or patched on the way."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    train_ids: np.ndarray
    test_ids: np.ndarray
    feature_names: list[str]
    dropped_train_ids: list[int]  # training recordings without a mean heartbeat (2024: recording 1870)
    patched_test_ids: list[int]  # test recordings that got a neighbour's mean heartbeat (2024: recording 700)


def prepare(
    cfg: PipelineConfig,
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    y: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    train_waves: np.ndarray | None = None,
    test_waves: np.ndarray | None = None,
) -> Prepared:
    """Clean the feature tables and append the mean heartbeats (train rows without one are dropped, as in 2024)."""
    X_train, X_test, columns = clean_tables(train_table, test_table, cfg.features)
    names = list(columns)
    dropped: list[int] = []
    patched: list[int] = []
    keep = np.ones(len(X_train), dtype=bool)
    if cfg.features.mean_wave:
        if train_waves is None or test_waves is None:
            raise ValueError("mean_wave is enabled but no waves were given")
        train_failed = {int(i): "no heartbeat" for i in np.flatnonzero(np.isnan(train_waves).any(axis=1))}
        test_failed = {int(i): "no heartbeat" for i in np.flatnonzero(np.isnan(test_waves).any(axis=1))}
        keep[list(train_failed)] = False
        dropped = [int(train_ids[i]) for i in sorted(train_failed)]
        patched = [int(test_ids[i]) for i in sorted(test_failed)]
        X_train = np.hstack([X_train, train_waves])
        X_test = np.hstack([X_test, fill_wave_failures(test_waves, test_failed)])
        names += [f"wave_{i}" for i in range(train_waves.shape[1])]
    return Prepared(
        X_train=X_train[keep],
        y_train=np.asarray(y)[keep],
        X_test=X_test,
        train_ids=np.asarray(train_ids)[keep],
        test_ids=np.asarray(test_ids),
        feature_names=names,
        dropped_train_ids=dropped,
        patched_test_ids=patched,
    )


@dataclass
class PipelineResult:
    predictions: pd.DataFrame
    model: Pipeline
    prepared: Prepared
    config_hash: str
    elapsed_s: float


def fit_predict(cfg: PipelineConfig, prepared: Prepared) -> tuple[Pipeline, np.ndarray]:
    from .model import make_pipeline

    model = make_pipeline(cfg.model)
    model.fit(prepared.X_train, prepared.y_train)
    return model, model.predict(prepared.X_test)


def run_pipeline(cfg: PipelineConfig, prepared: Prepared) -> PipelineResult:
    start = time.perf_counter()
    model, y_pred = fit_predict(cfg, prepared)
    predictions = pd.DataFrame({"id": prepared.test_ids, "y": y_pred.astype(int)})
    return PipelineResult(predictions, model, prepared, config_hash(cfg), round(time.perf_counter() - start, 1))

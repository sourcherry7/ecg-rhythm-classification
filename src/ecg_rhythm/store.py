"""Feature-set folders on disk.

A feature folder holds ``features_train.csv`` and ``features_test.csv`` (one row per recording, in recording order,
no id column, exactly like the tables the team exchanged in 2024), optionally ``waves_train.npy`` / ``waves_test.npy``
(mean heartbeats, NaN rows where none could be computed) and ``metadata.json``. ``data/processed/legacy_2024`` holds
the team's own 2024 tables; ``scripts/extract_features.py`` writes ``data/processed/features_<config hash>``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PipelineConfig, config_hash
from .data import PROCESSED_DIR, Signals
from .features import mean_waves

TABLE_FILES = {"train": "features_train.csv", "test": "features_test.csv"}
WAVE_FILES = {"train": "waves_train.npy", "test": "waves_test.npy"}


def feature_dir_for(cfg: PipelineConfig, root: Path | str = PROCESSED_DIR) -> Path:
    """Where ``extract_features.py`` stores the tables of a configuration (the model section does not matter)."""
    extraction_cfg = cfg.with_overrides(model={}, cv={})
    return Path(root) / f"features_{config_hash(extraction_cfg)}"


def save_table(directory: Path | str, split: str, frame: pd.DataFrame) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TABLE_FILES[split]
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def save_waves(directory: Path | str, split: str, waves: np.ndarray) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / WAVE_FILES[split]
    np.save(path, waves)
    return path


def write_metadata(directory: Path | str, payload: dict[str, Any]) -> Path:
    path = Path(directory) / "metadata.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8", newline="\n")
    return path


def read_metadata(directory: Path | str) -> dict[str, Any]:
    path = Path(directory) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_tables(directory: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = Path(directory)
    train = pd.read_csv(directory / TABLE_FILES["train"])
    test = pd.read_csv(directory / TABLE_FILES["test"])
    if list(train.columns) != list(test.columns):
        raise ValueError(f"{directory}: train and test tables have different columns")
    return train, test


def load_waves(directory: Path | str) -> tuple[np.ndarray | None, np.ndarray | None]:
    directory = Path(directory)
    paths = [directory / WAVE_FILES[s] for s in ("train", "test")]
    if not all(p.exists() for p in paths):
        return None, None
    return np.load(paths[0]), np.load(paths[1])


def ensure_waves(
    directory: Path | str,
    cfg: PipelineConfig,
    train: Signals,
    test: Signals,
    n_jobs: int | None = None,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the mean heartbeats of a feature folder, computing and saving them first when they are missing."""
    train_waves, test_waves = load_waves(directory)
    if train_waves is None or test_waves is None:
        train_waves, train_failed = mean_waves(train.signals, cfg.features, n_jobs=n_jobs, progress=progress)
        test_waves, test_failed = mean_waves(test.signals, cfg.features, n_jobs=n_jobs, progress=progress)
        save_waves(directory, "train", train_waves)
        save_waves(directory, "test", test_waves)
        meta = read_metadata(directory)
        meta["waves"] = {
            "sampling_rate": cfg.features.mean_wave_sampling_rate,
            "window_s": cfg.features.mean_wave_window_s,
            "failed_train": {str(k): v for k, v in train_failed.items()},
            "failed_test": {str(k): v for k, v in test_failed.items()},
        }
        write_metadata(directory, meta)
    return train_waves, test_waves

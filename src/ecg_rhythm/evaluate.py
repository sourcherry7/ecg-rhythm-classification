"""Cross-validation the way the team ran it in 2024 (``cross_val_score(cv=5, scoring="f1_micro")``), with detail.

``cv=5`` on a classifier means ``StratifiedKFold(5)`` without shuffling, so the folds are deterministic. Besides the
per-fold micro-F1 the result keeps the per-class F1 and the confusion matrix of the out-of-fold predictions, which the
micro average (identical to accuracy here) hides.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from .config import CVConfig, PipelineConfig, config_hash
from .data import N_CLASSES
from .model import make_pipeline


@dataclass
class CVResult:
    name: str
    model_kind: str
    fold_scores: list[float]
    per_class_f1: list[float]
    confusion: list[list[int]]
    n_train: int
    n_features: int
    config_hash: str
    elapsed_s: float
    config: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def mean(self) -> float:
        return float(np.mean(self.fold_scores))

    @property
    def std(self) -> float:
        return float(np.std(self.fold_scores))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mean_f1"] = self.mean
        payload["std_f1"] = self.std
        return payload


def make_splitter(cfg: CVConfig) -> StratifiedKFold:
    return StratifiedKFold(cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.random_state if cfg.shuffle else None)


def cross_validate(
    cfg: PipelineConfig, X: np.ndarray, y: np.ndarray, name: str = "submitted", verbose: bool = True
) -> CVResult:
    """Fit the configured pipeline on every fold; report micro-F1 per fold plus out-of-fold per-class detail."""
    start = time.perf_counter()
    average = cfg.cv.scoring.removeprefix("f1_")
    out_of_fold = np.full(len(y), -1, dtype=int)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(make_splitter(cfg.cv).split(X, y)):
        pipeline = make_pipeline(cfg.model)
        pipeline.fit(X[train_idx], y[train_idx])
        pred = pipeline.predict(X[val_idx])
        out_of_fold[val_idx] = pred
        scores.append(float(f1_score(y[val_idx], pred, average=average)))
        if verbose:
            print(f"  fold {fold + 1}/{cfg.cv.n_splits}: {scores[-1]:.4f}", flush=True)
    labels = list(range(N_CLASSES))
    result = CVResult(
        name=name,
        model_kind=cfg.model.kind,
        fold_scores=scores,
        per_class_f1=[float(v) for v in f1_score(y, out_of_fold, average=None, labels=labels)],
        confusion=confusion_matrix(y, out_of_fold, labels=labels).tolist(),
        n_train=len(y),
        n_features=X.shape[1],
        config_hash=config_hash(cfg),
        elapsed_s=round(time.perf_counter() - start, 1),
        config=cfg.to_dict(),
    )
    if verbose:
        per_class = np.round(result.per_class_f1, 3)
        print(f"  {cfg.cv.scoring} {result.mean:.4f} +- {result.std:.4f}; per-class F1 {per_class}")
    return result


def summarize(results: list[CVResult]) -> pd.DataFrame:
    rows = [
        {
            "name": r.name,
            "model": r.model_kind,
            "mean_f1": round(r.mean, 4),
            "std_f1": round(r.std, 4),
            "f1_class0": round(r.per_class_f1[0], 3),
            "f1_class1": round(r.per_class_f1[1], 3),
            "f1_class2": round(r.per_class_f1[2], 3),
            "f1_class3": round(r.per_class_f1[3], 3),
            "n_train": r.n_train,
            "n_features": r.n_features,
            "minutes": round(r.elapsed_s / 60, 1),
            "config_hash": r.config_hash,
        }
        for r in results
    ]
    return pd.DataFrame(rows)

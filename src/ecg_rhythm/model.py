"""The submitted classifier: SimpleImputer(mean) -> StandardScaler -> XGBClassifier() with library defaults."""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig

_DEFAULTS = ModelConfig()


def make_classifier(cfg: ModelConfig = _DEFAULTS):
    params = dict(cfg.params)
    if cfg.random_state is not None:
        params.setdefault("random_state", cfg.random_state)
    params.setdefault("n_jobs", cfg.n_jobs)
    if cfg.kind == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(**params)
    if cfg.kind == "lgbm":
        from lightgbm import LGBMClassifier

        params.setdefault("verbose", -1)
        return LGBMClassifier(**params)
    raise ValueError(f"unknown model kind {cfg.kind!r}")


def make_pipeline(cfg: ModelConfig = _DEFAULTS) -> Pipeline:
    """Imputer and scaler in front of the classifier, exactly as in the 2024 notebook."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("classifier", make_classifier(cfg)),
        ]
    )

"""Pipeline configuration.

The dataclass defaults are the configuration behind the entry the team selected on Kaggle on 2024-12-02 (private
micro-F1 0.81031): NeuroKit2 features computed on the raw recordings with ``sampling_rate=250`` although the data is
sampled at 300 Hz (reproduced on purpose; the ``corrected_rate`` preset fixes it), the mean heartbeat around the
detected R-peaks, and a default ``XGBClassifier`` behind a mean imputer and a standard scaler. Every deviation from the
2024 code is an explicit field whose default keeps the submitted behaviour; ablations are named presets (``PRESETS``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

#: True sampling rate of the recordings (300 Hz, as stated by the competition).
SAMPLING_RATE_HZ = 300


@dataclass(frozen=True)
class PreprocessConfig:
    """Signal cleaning before feature extraction. The submitted features were computed on the raw signals."""

    denoise: bool = False  # True = median-filter baseline removal + 0.5 Hz high-pass + db4 wavelet + BioSPPy filter
    wavelet: str = "db4"
    median_kernel: int = 301
    highpass_hz: float = 0.5
    butter_order: int = 4


@dataclass(frozen=True)
class FeatureConfig:
    """Per-recording features (NeuroKit2 HRV, statistics, spectrum, entropies) and the mean heartbeat."""

    sampling_rate: int = 250  # what the 2024 code passed to NeuroKit2 and Welch; the data is 300 Hz
    on_error: str = "previous"  # "previous" = copy the previous recording's features (2024) | "nan"
    mean_wave: bool = True
    mean_wave_sampling_rate: int = 250  # 0.5 s window at this rate -> 2 * 125 = 250 samples
    mean_wave_window_s: float = 0.5
    nan_column_threshold: float = 0.5  # columns with at least this fraction of NaN are dropped
    clip: float = 1e15
    clean_per_table: bool = True  # 2024: train and test tables cleaned independently (own columns, own means)
    n_jobs: int = -1


@dataclass(frozen=True)
class ModelConfig:
    """Classifier behind ``SimpleImputer(mean)`` and ``StandardScaler``; empty params = library defaults (2024)."""

    kind: str = "xgb"  # "xgb" (submitted) | "lgbm"
    params: dict[str, Any] = field(default_factory=dict)
    random_state: int | None = None  # None = the library default, as in 2024
    n_jobs: int = -1


@dataclass(frozen=True)
class CVConfig:
    """``cross_val_score(..., cv=5, scoring="f1_micro")`` of 2024: StratifiedKFold without shuffling."""

    n_splits: int = 5
    shuffle: bool = False
    random_state: int = 42
    scoring: str = "f1_micro"


@dataclass(frozen=True)
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cv: CVConfig = field(default_factory=CVConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        return config_hash(self)

    def with_overrides(self, **sections: dict[str, Any]) -> PipelineConfig:
        """Return a copy with per-section overrides, e.g. ``cfg.with_overrides(features={"sampling_rate": 300})``."""
        updates = {}
        for name, values in sections.items():
            if name not in {f.name for f in dataclasses.fields(self)}:
                raise KeyError(f"unknown config section {name!r}")
            section = getattr(self, name)
            known = {f.name for f in dataclasses.fields(section)}
            unknown = set(values) - known
            if unknown:
                raise KeyError(f"unknown field(s) {sorted(unknown)} in section {name!r}")
            updates[name] = dataclasses.replace(section, **values)
        return dataclasses.replace(self, **updates)

    @classmethod
    def from_preset(cls, name: str) -> PipelineConfig:
        if name not in PRESETS:
            raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
        return cls().with_overrides(**PRESETS[name])

    def apply_cli_overrides(self, overrides: list[str]) -> PipelineConfig:
        """Apply ``section.field=value`` strings (values parsed as JSON, falling back to str)."""
        sections: dict[str, dict[str, Any]] = {}
        for item in overrides:
            key, sep, raw = item.partition("=")
            section, dot, name = key.partition(".")
            if not (section and dot and name and sep):
                raise ValueError(f"override must look like section.field=value, got {item!r}")
            try:
                value: Any = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            if isinstance(value, list):
                value = tuple(value)
            sections.setdefault(section, {})[name] = value
        return self.with_overrides(**sections)


#: Named configurations (README "what we tried"). Empty overrides = the submitted configuration.
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "submitted": {},
    "corrected_rate": {"features": {"sampling_rate": SAMPLING_RATE_HZ, "mean_wave_sampling_rate": SAMPLING_RATE_HZ}},
    "corrected_features_only": {"features": {"sampling_rate": SAMPLING_RATE_HZ}},
    "denoised": {"preprocess": {"denoise": True}},
    "no_mean_wave": {"features": {"mean_wave": False}},
    "nan_on_error": {"features": {"on_error": "nan"}},
    "lgbm": {"model": {"kind": "lgbm"}},
}


JOB_FIELDS = frozenset({"n_jobs"})


def _without_job_fields(section: Any) -> Any:
    if isinstance(section, dict):
        return {k: _without_job_fields(v) for k, v in section.items() if k not in JOB_FIELDS}
    return section


def config_hash(cfg: PipelineConfig) -> str:
    """Short, stable identifier of a configuration (parallelism fields excluded: they never change the result)."""
    payload = json.dumps(_without_job_fields(cfg.to_dict()), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]

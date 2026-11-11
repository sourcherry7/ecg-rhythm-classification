import pytest

from ecg_rhythm.config import PRESETS, PipelineConfig, config_hash


def test_defaults_are_the_submitted_configuration():
    cfg = PipelineConfig()
    assert cfg.features.sampling_rate == 250  # the 2024 mismatch, kept on purpose
    assert cfg.features.mean_wave and cfg.features.on_error == "previous"
    assert cfg.model.kind == "xgb" and cfg.model.params == {}
    assert not cfg.preprocess.denoise
    assert cfg.cv.n_splits == 5 and not cfg.cv.shuffle


def test_hash_ignores_parallelism_but_not_settings():
    base = PipelineConfig()
    assert config_hash(base) == config_hash(base.with_overrides(features={"n_jobs": 2}, model={"n_jobs": 1}))
    assert config_hash(base) != config_hash(base.with_overrides(features={"sampling_rate": 300}))


def test_presets_and_cli_overrides():
    for name in PRESETS:
        PipelineConfig.from_preset(name)
    corrected = PipelineConfig.from_preset("corrected_rate")
    assert corrected.features.sampling_rate == corrected.features.mean_wave_sampling_rate == 300
    cli = PipelineConfig().apply_cli_overrides(["model.kind=lgbm", "features.mean_wave=false", "cv.n_splits=3"])
    assert cli.model.kind == "lgbm" and cli.features.mean_wave is False and cli.cv.n_splits == 3
    with pytest.raises(KeyError):
        PipelineConfig().with_overrides(features={"no_such_field": 1})
    with pytest.raises(ValueError):
        PipelineConfig().apply_cli_overrides(["not-an-override"])

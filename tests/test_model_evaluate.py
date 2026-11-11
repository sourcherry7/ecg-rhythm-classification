import numpy as np
import pytest

from ecg_rhythm.config import ModelConfig, PipelineConfig
from ecg_rhythm.evaluate import cross_validate, summarize
from ecg_rhythm.model import make_pipeline
from ecg_rhythm.pipeline import prepare, run_pipeline


def _separable(n_per_class=30, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(loc=3 * k, scale=0.5, size=(n_per_class, 4)) for k in range(4)])
    y = np.repeat(np.arange(4), n_per_class)
    return X, y


@pytest.mark.parametrize("kind", ["xgb", "lgbm"])
def test_pipeline_fits_and_predicts_labels(kind):
    X, y = _separable()
    model = make_pipeline(ModelConfig(kind=kind, n_jobs=1)).fit(X, y)
    pred = model.predict(X)
    assert set(np.unique(pred)) <= {0, 1, 2, 3} and (pred == y).mean() > 0.95


def test_cross_validate_reports_folds_and_classes():
    X, y = _separable()
    cfg = PipelineConfig().with_overrides(model={"n_jobs": 1})
    result = cross_validate(cfg, X, y, name="synthetic", verbose=False)
    assert len(result.fold_scores) == 5 and len(result.per_class_f1) == 4
    assert np.array(result.confusion).sum() == len(y) and result.mean > 0.9
    table = summarize([result])
    assert table.loc[0, "name"] == "synthetic" and table.loc[0, "n_features"] == 4


def test_prepare_drops_train_rows_without_heartbeat_and_patches_test_rows():
    import pandas as pd

    rng = np.random.default_rng(1)
    train = pd.DataFrame(rng.normal(size=(8, 3)), columns=["a", "b", "c"])
    test = pd.DataFrame(rng.normal(size=(4, 3)), columns=["a", "b", "c"])
    train_waves = rng.normal(size=(8, 5))
    train_waves[2] = np.nan
    test_waves = rng.normal(size=(4, 5))
    test_waves[0] = np.nan
    cfg = PipelineConfig().with_overrides(model={"n_jobs": 1})
    prepared = prepare(
        cfg, train, test, np.array([0, 1, 2, 3, 0, 1, 2, 3]), np.arange(8), np.arange(100, 104), train_waves, test_waves
    )
    assert prepared.X_train.shape == (7, 8) and prepared.dropped_train_ids == [2]
    assert prepared.X_test.shape == (4, 8) and prepared.patched_test_ids == [100]
    np.testing.assert_array_equal(prepared.X_test[0, 3:], test_waves[1])
    assert prepared.feature_names == ["a", "b", "c"] + [f"wave_{i}" for i in range(5)]
    result = run_pipeline(cfg, prepared)
    assert list(result.predictions.columns) == ["id", "y"] and len(result.predictions) == 4

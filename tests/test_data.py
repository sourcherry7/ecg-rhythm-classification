import numpy as np
import pandas as pd
import pytest

from ecg_rhythm.data import load_raw, load_submission, save_submission, verify_raw


def _write_padded_csv(path, rows, with_target):
    width = max(len(r) for r in rows)
    records = []
    for i, row in enumerate(rows):
        padded = list(row) + [np.nan] * (width - len(row))
        record = {"id": i, **({"y": i % 4} if with_target else {}), **{f"x{j}": v for j, v in enumerate(padded)}}
        records.append(record)
    pd.DataFrame(records).to_csv(path, index=False)


def test_load_raw_strips_padding_and_caches(tmp_path):
    rng = np.random.default_rng(0)
    train_rows = [rng.normal(size=n).astype(np.float32) for n in (5, 9, 3, 7)]
    test_rows = [rng.normal(size=n).astype(np.float32) for n in (4, 6)]
    _write_padded_csv(tmp_path / "train.csv", train_rows, with_target=True)
    _write_padded_csv(tmp_path / "test.csv", test_rows, with_target=False)
    cache = tmp_path / "cache" / "signals.npz"

    train, test = load_raw(tmp_path, cache_path=cache)
    assert cache.exists()
    assert list(train.lengths) == [5, 9, 3, 7] and list(test.lengths) == [4, 6]
    assert list(train.y) == [0, 1, 2, 3] and test.y is None
    np.testing.assert_allclose(train.signals[1], train_rows[1], rtol=1e-6)

    cached_train, cached_test = load_raw(tmp_path / "does-not-exist", cache_path=cache)
    for a, b in zip(train.signals, cached_train.signals, strict=True):
        np.testing.assert_array_equal(a, b)
    assert np.array_equal(cached_test.ids, test.ids)


def test_submission_round_trip(tmp_path):
    path = save_submission(np.arange(5), np.array([0, 1, 2, 3, 0]), tmp_path / "sub.csv")
    frame = load_submission(path)
    assert list(frame.columns) == ["id", "y"] and frame["y"].tolist() == [0, 1, 2, 3, 0]
    assert path.read_bytes().startswith(b"id,y\n0,0\n")


@pytest.mark.needs_data
def test_real_data_integrity_and_shapes():
    assert set(verify_raw().values()) == {"ok"}
    train, test = load_raw()
    assert len(train) == 5117 and len(test) == 3411
    assert train.lengths.max() == 17807 and test.y is None
    assert np.bincount(train.y).tolist() == [3030, 443, 1474, 170]

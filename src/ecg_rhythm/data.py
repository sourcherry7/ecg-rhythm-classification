"""Paths, raw-file integrity, recording loader with a compact cache, and submission files.

The competition data (``train.csv`` with 5117 recordings, ``test.csv`` with 3411, ``sample_submission.csv``) is private
and never committed; see ``data/README.md``. Each row of the CSVs is one ECG recording of up to 17 807 samples at
300 Hz, right-padded with NaN. Reading the 290 MB training CSV takes about 90 s, so the recordings are cached once as
a flat float32 array with offsets (``data/processed/signals.npz``) and load in well under a second afterwards.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
#: The feature tables the team extracted in November 2024 (reproduction checkpoint A).
LEGACY_FEATURES_DIR = PROCESSED_DIR / "legacy_2024"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SIGNAL_CACHE = PROCESSED_DIR / "signals.npz"

COMPETITION = "ethz-aml-project-2"
RAW_FILES = ("train.csv", "test.csv", "sample_submission.csv")
RAW_SHA256 = {
    "train.csv": "d501e869a1ae517b8311e2d0236ee1b0ee15a34bde298c612ff3988b324336d3",
    "test.csv": "c23377c519109e023ffc0dafa4f637040f2e7eb7a9020ecdc8d3c80d7e943a9f",
    "sample_submission.csv": "cbba780b235212592541f5f3fef2c191a3a1ef41c05ca66028ea2b9bf17ba5fc",
}
LEGACY_FEATURE_FILES = ("features_train.csv", "features_test.csv")
N_CLASSES = 4


@dataclass
class Signals:
    """A set of recordings: ``signals[i]`` is a float32 array without the NaN padding, ``ids[i]`` its id."""

    ids: np.ndarray
    signals: list[np.ndarray]
    y: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.signals)

    @property
    def lengths(self) -> np.ndarray:
        return np.array([len(s) for s in self.signals], dtype=np.int64)


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw(raw_dir: Path | str = RAW_DIR) -> dict[str, str]:
    """Return ``{file: "ok" | "missing" | "mismatch"}`` for the three competition files."""
    raw_dir = Path(raw_dir)
    status = {}
    for name in RAW_FILES:
        path = raw_dir / name
        if not path.exists():
            status[name] = "missing"
        else:
            status[name] = "ok" if sha256_file(path) == RAW_SHA256[name] else "mismatch"
    return status


def copy_local(src_dir: Path | str, dest: Path | str = RAW_DIR, files: tuple[str, ...] = RAW_FILES) -> list[Path]:
    """Copy the competition files from a local folder (e.g. the original download) into ``data/raw``."""
    src_dir, dest = Path(src_dir), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in files:
        source = src_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        copied.append(Path(shutil.copy2(source, dest / name)))
    return copied


def _read_ragged(csv: Path, has_target: bool) -> Signals:
    frame = pd.read_csv(csv, index_col="id", dtype=np.float32)
    ids = frame.index.to_numpy().astype(np.int64)
    y = frame.pop("y").to_numpy().astype(np.int64) if has_target else None
    values = frame.to_numpy(dtype=np.float32)
    signals = [row[~np.isnan(row)] for row in values]
    return Signals(ids, signals, y)


def _pack(prefix: str, data: Signals) -> dict[str, np.ndarray]:
    offsets = np.concatenate([[0], np.cumsum(data.lengths)]).astype(np.int64)
    packed = {f"{prefix}_flat": np.concatenate(data.signals), f"{prefix}_offsets": offsets, f"{prefix}_ids": data.ids}
    if data.y is not None:
        packed[f"{prefix}_y"] = data.y
    return packed


def _unpack(prefix: str, store: np.lib.npyio.NpzFile) -> Signals:
    flat, offsets = store[f"{prefix}_flat"], store[f"{prefix}_offsets"]
    signals = [flat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
    y = store[f"{prefix}_y"] if f"{prefix}_y" in store.files else None
    return Signals(store[f"{prefix}_ids"], signals, y)


def load_raw(
    raw_dir: Path | str = RAW_DIR, cache_path: Path | str | None = SIGNAL_CACHE, use_cache: bool = True
) -> tuple[Signals, Signals]:
    """Load the training and test recordings (``train.y`` holds the labels 0..3), through the cache when possible."""
    cache_path = Path(cache_path) if cache_path else None
    if use_cache and cache_path and cache_path.exists():
        with np.load(cache_path) as store:
            return _unpack("train", store), _unpack("test", store)
    raw_dir = Path(raw_dir)
    train = _read_ragged(raw_dir / "train.csv", has_target=True)
    test = _read_ragged(raw_dir / "test.csv", has_target=False)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **_pack("train", train), **_pack("test", test))
    return train, test


def load_legacy_features(directory: Path | str = LEGACY_FEATURES_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The 113-column feature tables the team extracted on 2024-11-28 (raw signals, ``sampling_rate=250``)."""
    directory = Path(directory)
    train = pd.read_csv(directory / "features_train.csv")
    test = pd.read_csv(directory / "features_test.csv")
    if list(train.columns) != list(test.columns):
        raise ValueError("legacy feature tables have different columns")
    return train, test


def save_submission(ids: np.ndarray, y_pred: np.ndarray, path: Path | str) -> Path:
    """Write ``id,y`` with integer labels, the format of ``sample_submission.csv``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"id": np.asarray(ids, dtype=np.int64), "y": np.asarray(y_pred, dtype=np.int64)})
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def load_submission(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "y"]:
        raise ValueError(f"{path}: expected columns id,y, found {list(frame.columns)}")
    return frame

"""Fetch the competition files into data/raw and check their hashes.

    python scripts/download_data.py --kaggle                 # needs the repo-local Kaggle token (see data/README.md)
    python scripts/download_data.py --from-local <folder with train.csv, test.csv, sample_submission.csv>
    python scripts/download_data.py --from-local <folder> --legacy-features <folder with the 2024 feature tables>

The ``--legacy-features`` folder is copied to data/processed/legacy_2024: the feature tables the team extracted in
November 2024, which reproduction checkpoint A reuses.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from ecg_rhythm.data import LEGACY_FEATURE_FILES, LEGACY_FEATURES_DIR, RAW_DIR, copy_local, verify_raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--kaggle", action="store_true", help="download with the Kaggle API")
    source.add_argument("--from-local", type=Path, metavar="DIR", help="copy the three CSV files from a folder")
    parser.add_argument("--legacy-features", type=Path, metavar="DIR", help="copy the 2024 feature tables too")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    if args.kaggle:
        from ecg_rhythm.kaggle_local import download_competition_data

        for path in download_competition_data(args.raw_dir):
            print(f"downloaded {path}")
    else:
        for path in copy_local(args.from_local, args.raw_dir):
            print(f"copied {path}")
    status = verify_raw(args.raw_dir)
    for name, state in status.items():
        print(f"{name}: {state}")
    if args.legacy_features:
        LEGACY_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        for name in LEGACY_FEATURE_FILES:
            print(f"copied {shutil.copy2(args.legacy_features / name, LEGACY_FEATURES_DIR / name)}")
    return 0 if set(status.values()) == {"ok"} else 1


if __name__ == "__main__":
    sys.exit(main())

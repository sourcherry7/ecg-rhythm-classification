"""One command for the whole reproduction.

    python scripts/reproduce.py --from-local <folder with the CSVs> --legacy-features <folder with the 2024 tables>
    python scripts/reproduce.py --kaggle --from-raw            # extract the features yourself (about an hour)
    python scripts/reproduce.py --from-raw --preset corrected_rate

Runs download_data -> [extract_features] -> evaluate -> train_predict with this interpreter.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from ecg_rhythm.config import PipelineConfig
from ecg_rhythm.data import LEGACY_FEATURES_DIR, RESULTS_DIR
from ecg_rhythm.store import feature_dir_for

SCRIPTS = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(SCRIPTS / script), *args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--kaggle", action="store_true")
    source.add_argument("--from-local", type=Path, metavar="DIR")
    parser.add_argument("--legacy-features", type=Path, metavar="DIR")
    parser.add_argument("--from-raw", action="store_true", help="extract the features instead of using the 2024 tables")
    parser.add_argument("--preset", default="submitted")
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    if args.kaggle or args.from_local:
        download = ["--kaggle"] if args.kaggle else ["--from-local", str(args.from_local)]
        if args.legacy_features:
            download += ["--legacy-features", str(args.legacy_features)]
        run("download_data.py", *download)
    if args.from_raw:
        extract = ["--preset", args.preset]
        if args.n_jobs:
            extract += ["--n-jobs", str(args.n_jobs)]
        run("extract_features.py", *extract)
        features = feature_dir_for(PipelineConfig.from_preset(args.preset))
        out = RESULTS_DIR / f"submission_{args.preset}_from_raw.csv"
    else:
        features = LEGACY_FEATURES_DIR
        out = RESULTS_DIR / ("submission.csv" if args.preset == "submitted" else f"submission_{args.preset}.csv")
    if not args.skip_cv:
        run("evaluate.py", "--features", str(features), "--preset", args.preset)
    run("train_predict.py", "--features", str(features), "--preset", args.preset, "--out", str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

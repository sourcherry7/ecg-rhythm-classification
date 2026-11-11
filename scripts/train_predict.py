"""Fit the classifier on all training recordings of a feature folder and write a submission file.

    python scripts/train_predict.py                                              # checkpoint A: 2024 tables
    python scripts/train_predict.py --features data/processed/features_<hash> --out results/submission_v2.csv
    python scripts/train_predict.py --compare-to results/submission.csv          # label agreement with a reference

Next to the submission, ``<name>.json`` records the configuration, library versions, the recordings dropped or
patched on the way, the predicted class counts and, if requested, the agreement with the reference file.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from ecg_rhythm.config import PipelineConfig
from ecg_rhythm.data import LEGACY_FEATURES_DIR, RESULTS_DIR, load_raw, load_submission, save_submission, sha256_file
from ecg_rhythm.pipeline import prepare, run_pipeline
from ecg_rhythm.store import ensure_waves, load_tables


def compare(predictions, reference_path: Path) -> dict:
    reference = load_submission(reference_path)
    merged = predictions.merge(reference, on="id", suffixes=("", "_ref"))
    same = merged["y"] == merged["y_ref"]
    per_class = {int(c): round(float(same[merged["y_ref"] == c].mean()), 4) for c in sorted(merged["y_ref"].unique())}
    return {
        "reference": str(reference_path),
        "n_common_ids": int(len(merged)),
        "agreement": round(float(same.mean()), 4),
        "agreement_per_reference_class": per_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=LEGACY_FEATURES_DIR)
    parser.add_argument("--preset", default="submitted")
    parser.add_argument("--set", action="append", default=[], metavar="section.field=value")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "submission.csv")
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    cfg = PipelineConfig.from_preset(args.preset).apply_cli_overrides(args.set)
    train_table, test_table = load_tables(args.features)
    train, test = load_raw()
    waves = (None, None)
    if cfg.features.mean_wave:
        waves = ensure_waves(args.features, cfg, train, test, n_jobs=args.n_jobs)
    prepared = prepare(cfg, train_table, test_table, train.y, train.ids, test.ids, *waves)
    print(f"fitting {cfg.model.kind} (config {cfg.hash()}) on {prepared.X_train.shape} from {args.features.name} ...")
    result = run_pipeline(cfg, prepared)
    path = save_submission(result.predictions["id"], result.predictions["y"], args.out)
    (args.out.with_suffix("")).parent.mkdir(parents=True, exist_ok=True)
    features_path = args.out.with_name(args.out.stem + "_features.txt")
    features_path.write_text("\n".join(prepared.feature_names) + "\n", encoding="utf-8", newline="\n")

    import neurokit2
    import sklearn
    import xgboost

    counts = result.predictions["y"].value_counts().sort_index()
    metadata = {
        "submission": path.name,
        "sha256": sha256_file(path),
        "config_hash": result.config_hash,
        "preset": args.preset,
        "overrides": args.set,
        "features": args.features.name,
        "n_train_rows": int(prepared.X_train.shape[0]),
        "n_features": int(prepared.X_train.shape[1]),
        "dropped_train_ids": prepared.dropped_train_ids,
        "patched_test_ids": prepared.patched_test_ids,
        "predicted_class_counts": {int(k): int(v) for k, v in counts.items()},
        "elapsed_s": result.elapsed_s,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "neurokit2": neurokit2.__version__,
            "numpy": np.__version__,
        },
        "config": cfg.to_dict(),
    }
    if args.compare_to:
        metadata["comparison"] = compare(result.predictions, args.compare_to)
        print("comparison:", json.dumps(metadata["comparison"], indent=2))
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")
    class_counts = {int(k): int(v) for k, v in counts.items()}
    print(f"wrote {path} ({len(result.predictions)} rows, classes {class_counts}, {result.elapsed_s:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

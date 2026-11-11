"""Cross-validate the classifier on a feature folder (fast: about a minute per configuration).

    python scripts/evaluate.py                                       # checkpoint A: the team's 2024 tables
    python scripts/evaluate.py --features data/processed/features_<hash> --preset submitted corrected_rate
    python scripts/evaluate.py --preset no_mean_wave lgbm --set cv.shuffle=true

Records are appended to results/cv_scores.json (fold scores, per-class F1, confusion matrix, configuration).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from ecg_rhythm.config import PipelineConfig
from ecg_rhythm.data import LEGACY_FEATURES_DIR, RESULTS_DIR, load_raw
from ecg_rhythm.evaluate import cross_validate, summarize
from ecg_rhythm.pipeline import prepare
from ecg_rhythm.store import ensure_waves, load_tables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=LEGACY_FEATURES_DIR, help="feature folder")
    parser.add_argument("--preset", nargs="+", default=["submitted"])
    parser.add_argument("--set", action="append", default=[], metavar="section.field=value")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "cv_scores.json")
    parser.add_argument("--n-jobs", type=int, default=None, help="workers for the mean heartbeats, if missing")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    train_table, test_table = load_tables(args.features)
    train, test = load_raw()
    results = []
    for name in args.preset:
        cfg = PipelineConfig.from_preset(name).apply_cli_overrides(args.set)
        waves = (None, None)
        if cfg.features.mean_wave:
            waves = ensure_waves(args.features, cfg, train, test, n_jobs=args.n_jobs, progress=not args.quiet)
        prepared = prepare(cfg, train_table, test_table, train.y, train.ids, test.ids, *waves)
        print(
            f"== {name} on {args.features.name}: {prepared.X_train.shape[0]} rows x {prepared.X_train.shape[1]} columns"
        )
        result = cross_validate(cfg, prepared.X_train, prepared.y_train, name=name, verbose=not args.quiet)
        results.append(result)
        record = result.to_dict()
        record["features"] = args.features.name
        record["dropped_train_ids"] = prepared.dropped_train_ids
        existing = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else []
        existing.append(record)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(existing, indent=2), encoding="utf-8", newline="\n")
    print(summarize(results).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

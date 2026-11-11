"""Compute the feature tables and mean heartbeats of a configuration for all recordings (the slow step).

    python scripts/extract_features.py                              # submitted configuration (250 Hz setting)
    python scripts/extract_features.py --preset corrected_rate      # the same features at the true 300 Hz
    python scripts/extract_features.py --preset denoised --n-jobs 8

About 3 to 4 s of CPU per recording (NeuroKit2 plus two quadratic-time entropies): roughly one hour for the 8528
recordings with 10 workers. The training table is saved before the test table starts, so a crash costs at most half.
Output folder: data/processed/features_<hash of the preprocessing and feature sections>.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ecg_rhythm.config import PipelineConfig
from ecg_rhythm.data import load_raw
from ecg_rhythm.pipeline import build_feature_set
from ecg_rhythm.store import feature_dir_for, read_metadata, save_table, save_waves, write_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="submitted")
    parser.add_argument("--set", action="append", default=[], metavar="section.field=value")
    parser.add_argument("--out", type=Path, help="output folder (default: data/processed/features_<hash>)")
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--force", action="store_true", help="recompute even when the folder is complete")
    args = parser.parse_args()

    cfg = PipelineConfig.from_preset(args.preset).apply_cli_overrides(args.set)
    out = args.out or feature_dir_for(cfg)
    print(f"configuration {cfg.hash()} -> {out}", flush=True)
    done = {s for s in ("train", "test") if (out / f"features_{s}.csv").exists()} if not args.force else set()
    if done == {"train", "test"}:
        print("already complete; use --force to recompute")
        return 0

    train, test = load_raw()
    meta = read_metadata(out)
    meta.update({"config": cfg.to_dict(), "config_hash": cfg.hash(), "preset": args.preset, "overrides": args.set})
    for split, signals in (("train", train), ("test", test)):
        if split in done:
            print(f"{split}: already present, skipping", flush=True)
            continue
        start = time.perf_counter()
        feature_set = build_feature_set(signals, cfg, n_jobs=args.n_jobs)
        save_table(out, split, feature_set.table.frame)
        if feature_set.waves is not None:
            save_waves(out, split, feature_set.waves)
        meta[split] = {
            "n_recordings": len(signals),
            "n_columns": feature_set.table.frame.shape[1],
            "failed_features": {str(k): v for k, v in feature_set.table.failed.items()},
            "failed_waves": {str(k): v for k, v in feature_set.wave_failures.items()},
            "elapsed_s": round(time.perf_counter() - start, 1),
        }
        print(
            f"{split}: {len(signals)} recordings, {feature_set.table.frame.shape[1]} columns, "
            f"{len(feature_set.table.failed)} feature failures, {len(feature_set.wave_failures)} wave failures, "
            f"{(time.perf_counter() - start) / 60:.1f} min",
            flush=True,
        )
        write_metadata(out, meta)
    import neurokit2
    import sklearn

    meta["versions"] = {
        "python": platform.python_version(),
        "neurokit2": neurokit2.__version__,
        "scikit_learn": sklearn.__version__,
    }
    meta["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["n_jobs"] = args.n_jobs
    write_metadata(out, meta)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

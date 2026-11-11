"""Upload a late submission to Kaggle and log it in results/late_submissions.md.

Late submissions are scored on the real private test set but do not change the official leaderboard.
Nothing is uploaded without --yes. Use --status to list the team's submissions and their scores, and --fill
(a minute after an upload) to copy the scores of pending rows into the log.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from ecg_rhythm.data import COMPETITION, RESULTS_DIR, load_submission, sha256_file
from ecg_rhythm.kaggle_local import fill_pending_scores, list_submissions, submit

LOG = RESULTS_DIR / "late_submissions.md"
HEADER = (
    "# Late submissions\n\nPost-competition uploads used to verify the reproduction. "
    "They do not affect the official rank (62nd of 131, private micro-F1 0.81031). The sha256 column holds the first "
    "12 hex digits of the hash of the file as stored in this repository (LF line endings).\n\n"
    "| when (UTC) | file | sha256 | message | public | private |\n|---|---|---|---|---|---|\n"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", nargs="?", type=Path, default=RESULTS_DIR / "submission.csv")
    parser.add_argument("-m", "--message", default="")
    parser.add_argument("--yes", action="store_true", help="actually upload (otherwise a dry run)")
    parser.add_argument("--status", action="store_true", help="only list the submissions and scores")
    parser.add_argument("--fill", action="store_true", help="fill pending scores in the log from Kaggle")
    parser.add_argument("--competition", default=COMPETITION)
    args = parser.parse_args()

    if args.status:
        table = list_submissions(args.competition)
        print(table.head(15).to_string(index=False))
        return 0
    if args.fill:
        if not LOG.exists():
            print(f"{LOG} does not exist; nothing to fill")
            return 0
        text, filled = fill_pending_scores(LOG.read_text(encoding="utf-8"), list_submissions(args.competition))
        LOG.write_text(text, encoding="utf-8", newline="\n")
        print(f"filled {filled} pending row(s) in {LOG}")
        return 0

    frame = load_submission(args.file)
    digest = sha256_file(args.file)
    print(f"{args.file}: {len(frame)} rows, sha256 {digest[:16]}...")
    if not args.yes:
        print("dry run; add --yes to upload this file as a late submission")
        return 0
    response = submit(args.file, args.message or f"late submission {args.file.name}", args.competition)
    print(response.strip())
    stamp = datetime.now(timezone.utc).isoformat(timespec="minutes")
    if not LOG.exists():
        LOG.write_text(HEADER, encoding="utf-8", newline="\n")
    with LOG.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"| {stamp} | {args.file.name} | {digest[:12]} | {args.message} | pending | pending |\n")
    print(f"logged to {LOG}; run `scripts/submit.py --fill` in a minute to copy the scores into the log")
    return 0


if __name__ == "__main__":
    sys.exit(main())

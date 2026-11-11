"""Kaggle access with a token that lives inside the repo.

The token is expected in ``<repo>/.kaggle/`` (gitignored), in one of the two forms Kaggle issues:
the newer access token (``KGAT_...``) saved as ``access_token``, or the classic ``kaggle.json``. Nothing is
written to the user profile: the folder is handed to the client through environment variables
(``KAGGLE_API_TOKEN`` or ``KAGGLE_CONFIG_DIR``). All calls go through the ``kaggle`` command-line client
installed in the virtual environment.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from .data import COMPETITION, RAW_DIR, REPO_ROOT

KAGGLE_DIR = REPO_ROOT / ".kaggle"
TOKEN_HELP = (
    "Kaggle token not found. Create one on kaggle.com (Settings, API, Create New Token) and save it in {folder} "
    "(gitignored): either the KGAT_... access token as a file named access_token, or the downloaded kaggle.json."
)


def configure_kaggle_env(config_dir: Path | str = KAGGLE_DIR, require_token: bool = True) -> Path:
    """Point the Kaggle client at the repo-local token (must happen before ``import kaggle``).

    ``access_token`` (newer Kaggle tokens) is exported as ``KAGGLE_API_TOKEN``; the client additionally insists on a
    username/key pair in its configuration, which is satisfied with ``KAGGLE_USERNAME``/``KAGGLE_KEY`` (the bearer
    token is what authenticates). A classic ``kaggle.json`` is used through ``KAGGLE_CONFIG_DIR``.
    """
    config_dir = Path(config_dir)
    access_token = config_dir / "access_token"
    kaggle_json = config_dir / "kaggle.json"
    if access_token.exists():
        token = access_token.read_text(encoding="utf-8").strip()
        os.environ["KAGGLE_API_TOKEN"] = token
        os.environ.setdefault("KAGGLE_USERNAME", "token")
        os.environ.setdefault("KAGGLE_KEY", token)
        return access_token
    if kaggle_json.exists():
        os.environ["KAGGLE_CONFIG_DIR"] = str(config_dir)
        return kaggle_json
    if require_token:
        raise FileNotFoundError(TOKEN_HELP.format(folder=config_dir))
    return access_token


def kaggle_executable() -> str:
    """The ``kaggle`` CLI of the current interpreter's environment (falls back to PATH)."""
    scripts_dir = Path(sys.executable).parent
    for name in ("kaggle.exe", "kaggle"):
        candidate = scripts_dir / name
        if candidate.exists():
            return str(candidate)
    found = shutil.which("kaggle")
    if found:
        return found
    raise FileNotFoundError("kaggle CLI not found; install the project with the [kaggle] extra")


def run_kaggle(*args: str, config_dir: Path | str = KAGGLE_DIR) -> subprocess.CompletedProcess:
    configure_kaggle_env(config_dir)
    return subprocess.run(
        [kaggle_executable(), *args], check=True, text=True, capture_output=True, env=os.environ.copy()
    )


def _csv_from_cli(stdout: str) -> pd.DataFrame:
    """The CLI may print warnings before the CSV header; keep the lines from the header on."""
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(("fileName", "ref,", "teamId", "teamName")):
            return pd.read_csv(io.StringIO("\n".join(lines[i:])))
    raise ValueError(f"no CSV table found in Kaggle output:\n{stdout}")


def download_competition_data(dest: Path | str = RAW_DIR, competition: str = COMPETITION) -> list[Path]:
    """Download and unzip the competition files into ``dest`` (needs the token and competition access)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    run_kaggle("competitions", "download", "-c", competition, "-p", str(dest))
    extracted: list[Path] = []
    for archive in sorted(dest.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
            extracted.extend(dest / name for name in zf.namelist())
        archive.unlink()
    return extracted


def list_submissions(competition: str = COMPETITION) -> pd.DataFrame:
    """The team's submissions with their public and private scores."""
    return _csv_from_cli(run_kaggle("competitions", "submissions", "-c", competition, "-v").stdout)


def submit(file: Path | str, message: str, competition: str = COMPETITION) -> str:
    """Upload a late submission; returns the confirmation text printed by Kaggle."""
    return run_kaggle("competitions", "submit", "-c", competition, "-f", str(file), "-m", message).stdout


def leaderboard(competition: str = COMPETITION) -> pd.DataFrame:
    return _csv_from_cli(run_kaggle("competitions", "leaderboard", "-c", competition, "-s", "-v").stdout)


def fill_pending_scores(log_text: str, submissions: pd.DataFrame) -> tuple[str, int]:
    """Fill the ``pending`` public/private cells of ``results/late_submissions.md`` from Kaggle's submission list.

    Rows are matched on the file name; the newest completed Kaggle entry with that name wins. Returns the new
    text and the number of rows filled.
    """
    needed = {"fileName", "status", "publicScore", "privateScore"}
    if submissions.empty or not needed.issubset(submissions.columns):
        return log_text, 0
    done = submissions[submissions["status"].astype(str).str.contains("COMPLETE", na=False)]
    if "date" in done.columns:
        done = done.sort_values("date", kind="stable")
    done = done.drop_duplicates(subset="fileName", keep="last")
    scores = {row.fileName: (row.publicScore, row.privateScore) for row in done.itertuples()}
    filled = 0
    out = []
    for line in log_text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[4] == "pending" and cells[1] in scores:
            public, private = scores[cells[1]]
            if pd.notna(public) and pd.notna(private):
                cells[4], cells[5] = f"{float(public):.5f}", f"{float(private):.5f}"
                line = "| " + " | ".join(cells) + " |"
                filled += 1
        out.append(line)
    return "\n".join(out) + "\n", filled

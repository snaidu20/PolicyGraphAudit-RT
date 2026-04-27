"""
load_play_dsl.py — Play Data Safety Label ingestion for M3.

Public API
----------
load_play_dsl(appId, max_apps) -> pd.DataFrame
    Read data/raw/play_data_safety/sample_5000.json (JSONL), optionally
    filtering to a single app or top-N apps by realInstalls.

list_unique_apps(min_rows) -> list[str]
    Return appIds with at least min_rows rows in the sample.

Run as __main__ for a quick stat print.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution (same pattern as M2 vocab.py)
# ---------------------------------------------------------------------------

def _resolve_data_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, here.parent.parent, here.parent.parent.parent]:
        candidate = parent / "data"
        if candidate.is_dir():
            return candidate
        candidate = parent.parent / "data"
        if candidate.is_dir():
            return candidate
    return Path("data")


_SAMPLE_PATH = _resolve_data_root() / "raw" / "play_data_safety" / "sample_5000.json"

# Columns from schema.json
_EXPECTED_COLS = [
    "appId", "appTitle", "dataType", "category", "type", "optional",
    "purpose", "allPurposes", "installs", "realInstalls", "genreId",
    "developer", "developerWebsite", "privacyPolicy",
]


def _load_raw(path: Path) -> pd.DataFrame:
    """Read JSONL file; each line is a JSON object."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed JSON line: %s", exc)
    df = pd.DataFrame(rows)
    # Coerce realInstalls to numeric
    if "realInstalls" in df.columns:
        df["realInstalls"] = pd.to_numeric(df["realInstalls"], errors="coerce").fillna(0)
    return df


def load_play_dsl(
    appId: Optional[str] = None,
    max_apps: Optional[int] = None,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load the Play Data Safety Label JSONL sample into a DataFrame.

    Parameters
    ----------
    appId : str, optional
        If given, return only rows for this package name.
    max_apps : int, optional
        If given, keep only the top-N apps by realInstalls (highest install
        counts first).  Applied after appId filter if both are given.
    path : Path, optional
        Override the default sample file location.

    Returns
    -------
    pd.DataFrame
        One row per (app, dataType, purpose) declaration.  Columns match
        data/raw/play_data_safety/schema.json.
    """
    sample_path = path or _SAMPLE_PATH
    if not sample_path.exists():
        raise FileNotFoundError(f"Play DSL sample not found at {sample_path}")

    df = _load_raw(sample_path)
    log.info("Loaded %d rows from %s", len(df), sample_path)

    if appId is not None:
        df = df[df["appId"] == appId].copy()
        log.info("Filtered to appId=%s → %d rows", appId, len(df))

    if max_apps is not None:
        # Rank apps by their highest realInstalls value
        app_installs = (
            df.groupby("appId")["realInstalls"].max().nlargest(max_apps)
        )
        df = df[df["appId"].isin(app_installs.index)].copy()
        log.info("Filtered to top %d apps by installs → %d rows", max_apps, len(df))

    return df.reset_index(drop=True)


def list_unique_apps(min_rows: int = 3, path: Optional[Path] = None) -> list[str]:
    """
    Return sorted list of appIds with at least min_rows rows.

    This filters out trivial or near-empty apps (e.g. those that only
    declared 'No data collected').
    """
    df = load_play_dsl(path=path)
    counts = df.groupby("appId").size()
    qualifying = counts[counts >= min_rows].index.tolist()
    return sorted(qualifying)


# ---------------------------------------------------------------------------
# CLI — quick stats
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_play_dsl()
    n_rows = len(df)
    n_apps = df["appId"].nunique()

    top5 = (
        df.groupby("appId")["realInstalls"]
        .max()
        .nlargest(5)
        .reset_index()
    )

    print(f"Loaded {n_rows} rows, {n_apps} unique appIds")
    print(f"Top 5 by realInstalls:")
    for _, row in top5.iterrows():
        print(f"  {row['appId']:50s}  {int(row['realInstalls']):>12,}")

    qual = list_unique_apps(min_rows=3)
    print(f"\nApps with ≥3 rows: {len(qual)}")

"""Generic CSV persistence and seed-aggregation for evaluate_cell()'s output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_results(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_results(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def summarize(df: pd.DataFrame, metric_columns: list[str], group_columns: list[str]) -> pd.DataFrame:
    """Aggregate raw per-seed rows into mean +/- std per grid cell - the form
    typically used for confidence-interval plots."""
    grouped = df.groupby(group_columns)[metric_columns]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_std"))
    return summary.reset_index()

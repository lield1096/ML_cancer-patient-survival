"""Loading, cleaning and validating the raw dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "dead"
POSITIVE_LABEL = "Dead"

# Raw CSV header -> clean snake_case name (also fixes the "Reginol" typo in the source data).
COLUMN_MAP = {
    "Status": "status",
    "Age": "age",
    "Regional Node Examined": "regional_nodes_examined",
    "Reginol Node Positive": "regional_nodes_positive",
    "Survival Months": "survival_months",
    "Tumor Size": "tumor_size",
}

BASE_FEATURES = ["age", "tumor_size", "regional_nodes_examined", "regional_nodes_positive"]
LEAKY_FEATURE = "survival_months"


class DataValidationError(ValueError):
    """Raised when the input data does not match the expected schema."""


def feature_columns(include_survival_months: bool = False) -> list[str]:
    return BASE_FEATURES + ([LEAKY_FEATURE] if include_survival_months else [])


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the CSV, normalise column names, validate it and drop unusable rows."""
    df = pd.read_csv(path)
    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise DataValidationError(f"missing columns: {sorted(missing)}")
    df = df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and unlabeled rows, validate labels and value ranges, add binary target.

    Missing *feature* values are left in place on purpose: they are imputed inside the
    model pipeline so imputation statistics are learned from training folds only.
    """
    n_start = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=["status"])

    bad_labels = set(df["status"].unique()) - {"Alive", "Dead"}
    if bad_labels:
        raise DataValidationError(f"unexpected status labels: {sorted(bad_labels)}")

    numeric = [c for c in df.columns if c != "status"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    if (df[numeric] < 0).any().any():
        raise DataValidationError("numeric features must be non-negative")

    df = df.assign(**{TARGET: (df["status"] == POSITIVE_LABEL).astype(int)}).reset_index(drop=True)
    logger.info("Loaded %d rows (%d dropped as duplicate/unlabeled)", len(df), n_start - len(df))
    return df

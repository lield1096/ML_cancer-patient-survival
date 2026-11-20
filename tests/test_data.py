import pandas as pd
import pytest

from survival_predictor.data import (
    BASE_FEATURES,
    DataValidationError,
    feature_columns,
    load_dataset,
)
from tests.conftest import make_raw_df


def test_load_renames_columns_and_adds_target(raw_csv):
    df = load_dataset(raw_csv)
    assert "regional_nodes_positive" in df.columns  # typo "Reginol" fixed
    assert set(df["dead"].unique()) <= {0, 1}
    assert (df["dead"] == (df["status"] == "Dead")).all()


def test_drops_duplicates_and_unlabeled_rows(tmp_path):
    raw = make_raw_df(50)
    raw = pd.concat([raw, raw.iloc[[0]]])  # duplicate
    raw.loc[len(raw)] = [None, 50, 10, 1, 20, 15]  # unlabeled
    path = tmp_path / "d.csv"
    raw.to_csv(path, index=False)
    assert len(load_dataset(path)) == 50


def test_missing_feature_values_are_kept_for_pipeline_imputation(tmp_path):
    raw = make_raw_df(50)
    raw.loc[0, "Tumor Size"] = None
    path = tmp_path / "d.csv"
    raw.to_csv(path, index=False)
    assert load_dataset(path)["tumor_size"].isna().sum() == 1


def test_missing_column_raises(tmp_path):
    path = tmp_path / "d.csv"
    make_raw_df(10).drop(columns="Age").to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="Age"):
        load_dataset(path)


def test_unknown_label_raises(tmp_path):
    raw = make_raw_df(10)
    raw.loc[0, "Status"] = "Unknown"
    path = tmp_path / "d.csv"
    raw.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="Unknown"):
        load_dataset(path)


def test_negative_values_raise(tmp_path):
    raw = make_raw_df(10)
    raw.loc[0, "Age"] = -1
    path = tmp_path / "d.csv"
    raw.to_csv(path, index=False)
    with pytest.raises(DataValidationError):
        load_dataset(path)


def test_survival_months_excluded_by_default():
    assert feature_columns() == BASE_FEATURES
    assert "survival_months" in feature_columns(include_survival_months=True)

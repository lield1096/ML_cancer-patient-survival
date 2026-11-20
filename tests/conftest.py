from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from survival_predictor.config import load_config
from survival_predictor.train import train


def make_raw_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Small synthetic dataset in the raw CSV format; risk rises with tumor size and nodes."""
    rng = np.random.default_rng(seed)
    examined = rng.integers(1, 40, n)
    positive = np.minimum(rng.poisson(3, n), examined)
    tumor = rng.gamma(2.5, 12, n).round()
    risk = 0.03 * tumor + 0.25 * positive - 3
    dead = rng.random(n) < 1 / (1 + np.exp(-risk))
    return pd.DataFrame(
        {
            "Status": np.where(dead, "Dead", "Alive"),
            "Age": rng.integers(30, 70, n),
            "Regional Node Examined": examined,
            "Reginol Node Positive": positive,
            "Survival Months": rng.integers(1, 107, n),
            "Tumor Size": tumor,
        }
    )


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    path = tmp_path / "data.csv"
    make_raw_df().to_csv(path, index=False)
    return path


@pytest.fixture
def config_file(tmp_path: Path, raw_csv: Path) -> Path:
    cfg = {
        "data_path": str(raw_csv),
        "model_path": str(tmp_path / "model.joblib"),
        "reports_dir": str(tmp_path / "reports"),
        "random_state": 0,
        "test_size": 0.25,
        "cv_folds": 3,
        "models": {
            "logistic_regression": {"model__C": [0.1, 1.0]},
            "xgboost": {"model__n_estimators": [20], "model__max_depth": [2]},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture
def trained_model_path(config_file: Path) -> Path:
    return train(load_config(config_file)).model_path

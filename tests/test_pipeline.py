import json

import numpy as np
import pytest

from survival_predictor.config import load_config
from survival_predictor.data import TARGET, feature_columns, load_dataset
from survival_predictor.evaluate import compute_metrics
from survival_predictor.pipeline import MODEL_REGISTRY, build_pipeline
from survival_predictor.train import train


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_every_registered_model_fits_and_predicts_probabilities(raw_csv, name):
    df = load_dataset(raw_csv)
    X, y = df[feature_columns()], df[TARGET]
    proba = build_pipeline(name, random_state=0).fit(X, y).predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown model"):
        build_pipeline("svm")


def test_smote_runs_only_during_fit_not_predict(raw_csv):
    """SMOTE must never change the number of rows at prediction time (no test-set leakage)."""
    df = load_dataset(raw_csv)
    X, y = df[feature_columns()], df[TARGET]
    pipe = build_pipeline("logistic_regression", random_state=0).fit(X, y)
    assert len(pipe.predict(X)) == len(X)
    # imblearn exposes SMOTE as a sampler, which is skipped by predict/transform.
    assert pipe.named_steps["smote"].sampling_strategy_[1] > 0


def test_pipeline_imputes_missing_values(raw_csv):
    df = load_dataset(raw_csv)
    X, y = df[feature_columns()].copy(), df[TARGET]
    X.iloc[:5, 0] = np.nan
    pipe = build_pipeline("logistic_regression", random_state=0).fit(X, y)
    assert not np.isnan(pipe.predict_proba(X)).any()


def test_compute_metrics_perfect_and_keys():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == 1.0 and m["recall_dead"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_train_end_to_end_writes_artifacts(config_file):
    cfg = load_config(config_file)
    result = train(cfg)
    assert result.best_model in {"logistic_regression", "xgboost"}
    assert cfg.model_path.exists()
    report = json.loads((cfg.reports_dir / "metrics.json").read_text())
    assert report["features"] == feature_columns()
    assert 0 <= report["test_metrics"]["roc_auc"] <= 1
    assert (cfg.reports_dir / "figures" / "roc_curve.png").exists()


def test_training_is_reproducible(config_file, tmp_path):
    a = train(load_config(config_file)).test_metrics
    b = train(load_config(config_file, model_path=str(tmp_path / "m2.joblib"))).test_metrics
    assert a == b


def test_invalid_config_rejected(config_file):
    with pytest.raises(ValueError, match="test_size"):
        load_config(config_file, test_size=1.5)

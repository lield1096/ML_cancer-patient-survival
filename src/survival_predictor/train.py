"""Training: stratified split -> CV model selection + tuning -> single held-out evaluation."""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from . import __version__
from .config import Config
from .data import TARGET, feature_columns, load_dataset
from .evaluate import compute_metrics, save_figures
from .pipeline import build_pipeline

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    best_model: str
    cv_results: dict[str, dict[str, Any]]
    test_metrics: dict[str, Any]
    model_path: Path


def train(config: Config) -> TrainResult:
    df = load_dataset(config.data_path)
    features = feature_columns(config.include_survival_months)
    X, y = df[features], df[TARGET]

    # The test set is carved out first and touched exactly once, at the very end.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=config.random_state
    )
    logger.info(
        "Train: %d rows (%.1f%% dead) | Test: %d rows",
        len(X_train),
        100 * y_train.mean(),
        len(X_test),
    )

    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    searches: dict[str, GridSearchCV] = {}
    cv_results: dict[str, dict[str, Any]] = {}

    for name, grid in config.models.items():
        logger.info("Tuning %s over %d candidates...", name, _grid_size(grid))
        search = GridSearchCV(
            build_pipeline(name, config.random_state),
            param_grid=grid,
            scoring=config.scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        searches[name] = search
        std = search.cv_results_["std_test_score"][search.best_index_]
        cv_results[name] = {
            f"cv_{config.scoring}_mean": round(float(search.best_score_), 4),
            f"cv_{config.scoring}_std": round(float(std), 4),
            "best_params": _jsonable(search.best_params_),
        }
        logger.info("  %s: CV %s = %.4f ± %.4f", name, config.scoring, search.best_score_, std)

    best_name = max(searches, key=lambda n: searches[n].best_score_)
    best_pipeline = searches[best_name].best_estimator_
    logger.info("Selected model: %s", best_name)

    test_metrics = compute_metrics(y_test.to_numpy(), best_pipeline.predict_proba(X_test)[:, 1])
    logger.info(
        "Held-out test metrics: %s",
        {k: v for k, v in test_metrics.items() if k != "confusion_matrix"},
    )

    artifact = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "features": features,
        "test_metrics": test_metrics,
        "metadata": {
            "package_version": __version__,
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "random_state": config.random_state,
            "n_train": len(X_train),
        },
    }
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, config.model_path)
    logger.info("Saved model to %s", config.model_path)

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "selected_model": best_name,
        "features": features,
        "scoring": config.scoring,
        "cross_validation": cv_results,
        "test_metrics": test_metrics,
        "metadata": artifact["metadata"],
    }
    (config.reports_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    save_figures(best_pipeline, X_test, y_test, config.reports_dir / "figures", best_name)

    return TrainResult(best_name, cv_results, test_metrics, config.model_path)


def _grid_size(grid: dict[str, list[Any]]) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def _jsonable(params: dict[str, Any]) -> dict[str, Any]:
    return {k.removeprefix("model__"): v for k, v in params.items()}

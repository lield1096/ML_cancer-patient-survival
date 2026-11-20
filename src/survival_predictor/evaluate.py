"""Evaluation metrics suited to an imbalanced binary problem."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Return a JSON-serialisable dict of metrics. Positive class (1) = patient died."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision_dead": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_dead": round(float(recall_score(y_true, y_pred)), 4),
        "f1_dead": round(float(f1_score(y_true, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": threshold,
        "n_samples": int(len(y_true)),
    }


def save_figures(model, X_test, y_test, out_dir: Path, title: str) -> list[Path]:
    """Save ROC curve and confusion matrix plots for the final model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=title)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_title("ROC curve (held-out test set)")
    paths.append(out_dir / "roc_curve.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["Alive", "Dead"], cmap="Blues", ax=ax
    )
    ax.set_title("Confusion matrix (held-out test set)")
    paths.append(out_dir / "confusion_matrix.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)
    return paths

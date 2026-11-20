"""Quantify how much the original notebook's methodology inflated its results.

Compares, on the same data and model (XGBoost):
  1. Original approach  - SMOTE on the full training data *before* the train/test split,
                          so the test set contains synthetic rows derived from training rows.
  2. Leak-free + survival_months - correct split, SMOTE inside the pipeline, but still
                          using follow-up time as a feature.
  3. Leak-free (final)  - correct split, only features known at diagnosis.

Usage: python scripts/leakage_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from survival_predictor.data import TARGET, feature_columns, load_dataset
from survival_predictor.pipeline import build_pipeline

SEED = 42


def scores(y, proba) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(y, proba >= 0.5), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
    }


def main(data_path: str = "data/raw/cancer_patient_survival.csv") -> dict:
    df = load_dataset(data_path)
    results = {}

    # 1. Original: oversample everything, then split the oversampled data.
    X, y = df[feature_columns(True)], df[TARGET]
    X_res, y_res = SMOTE(random_state=SEED).fit_resample(X, y)
    Xtr, Xte, ytr, yte = train_test_split(X_res, y_res, test_size=0.2, random_state=SEED)
    model = XGBClassifier(random_state=SEED, verbosity=0).fit(Xtr, ytr)
    results["original_smote_before_split"] = scores(yte, model.predict_proba(Xte)[:, 1])

    # 2 & 3. Correct: split first on real data, SMOTE only inside the training pipeline.
    for name, include in [("leak_free_with_survival_months", True), ("leak_free_final", False)]:
        X, y = df[feature_columns(include)], df[TARGET]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        pipe = build_pipeline("xgboost", SEED).fit(Xtr, ytr)
        results[name] = scores(yte, pipe.predict_proba(Xte)[:, 1])

    results["majority_class_baseline_accuracy"] = round(1 - df[TARGET].mean(), 4)
    return results


if __name__ == "__main__":
    out = main(*sys.argv[1:])
    print(json.dumps(out, indent=2))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/leakage_comparison.json").write_text(json.dumps(out, indent=2))

"""Model registry and pipeline construction.

Every preprocessing step (imputation, scaling, SMOTE) lives *inside* one imblearn Pipeline.
That guarantees each step is fitted only on training data - during cross-validation, on the
training folds only - and that SMOTE is never applied when predicting. The original notebook
oversampled before splitting, so synthetic copies of training rows ended up in the test set.
"""

from __future__ import annotations

from collections.abc import Callable

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ModelFactory = Callable[[int], ClassifierMixin]

MODEL_REGISTRY: dict[str, ModelFactory] = {
    "logistic_regression": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "random_forest": lambda seed: RandomForestClassifier(random_state=seed, n_jobs=-1),
    "xgboost": lambda seed: XGBClassifier(
        random_state=seed, eval_metric="logloss", n_jobs=-1, verbosity=0
    ),
}


def build_pipeline(model_name: str, random_state: int = 42, use_smote: bool = True) -> Pipeline:
    """Create an unfitted pipeline: impute -> scale -> (SMOTE) -> classifier."""
    try:
        model = MODEL_REGISTRY[model_name](random_state)
    except KeyError:
        raise ValueError(
            f"unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}"
        ) from None

    steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("model", model))
    return Pipeline(steps)

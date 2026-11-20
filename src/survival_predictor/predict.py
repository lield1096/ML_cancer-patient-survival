"""Loading a trained model artifact and making predictions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .data import DataValidationError


@dataclass
class Prediction:
    probability_dead: float
    predicted_status: str


class SurvivalModel:
    """Thin wrapper around the saved pipeline that validates inputs before predicting."""

    def __init__(self, artifact: Mapping[str, Any], threshold: float = 0.5) -> None:
        self.pipeline = artifact["pipeline"]
        self.features: list[str] = list(artifact["features"])
        self.model_name: str = artifact["model_name"]
        self.test_metrics: dict[str, Any] = artifact.get("test_metrics", {})
        self.metadata: dict[str, Any] = artifact.get("metadata", {})
        self.threshold = threshold

    @classmethod
    def load(cls, path: str | Path, threshold: float = 0.5) -> SurvivalModel:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"model not found at {path}. Train one first: `survival-predictor train`"
            )
        return cls(joblib.load(path), threshold=threshold)

    def predict(self, records: Iterable[Mapping[str, Any]]) -> list[Prediction]:
        df = pd.DataFrame(list(records))
        if df.empty:
            return []
        missing = set(self.features) - set(df.columns)
        if missing:
            raise DataValidationError(f"missing features: {sorted(missing)}")
        proba = self.pipeline.predict_proba(df[self.features])[:, 1]
        return [
            Prediction(round(float(p), 4), "Dead" if p >= self.threshold else "Alive")
            for p in proba
        ]

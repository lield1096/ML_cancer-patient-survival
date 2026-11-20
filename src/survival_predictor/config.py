"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    data_path: Path
    model_path: Path
    reports_dir: Path
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    include_survival_months: bool = False
    scoring: str = "roc_auc"
    models: dict[str, dict[str, list[Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}")
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.cv_folds}")
        if not self.models:
            raise ValueError("config must define at least one model")


def load_config(path: str | Path, **overrides: Any) -> Config:
    """Load a YAML config. Relative paths inside it resolve against the current directory."""
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    for key in ("data_path", "model_path", "reports_dir"):
        raw[key] = Path(raw[key])
    return Config(**raw)

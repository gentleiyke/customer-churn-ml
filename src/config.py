"""Project configuration for the churn training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class Settings:
    # Reproducibility
    random_state: int = 42

    # Data / artifacts
    data_path: Path = Path("Customer-Churn.csv")
    artifact_dir: Path = Path("artifacts")

    # Train / Val / Test
    split: Tuple[float, float, float] = (0.60, 0.20, 0.20)

    # Cross-validation
    cv_folds: int = 5

    # Threshold sweep
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_step: float = 0.01

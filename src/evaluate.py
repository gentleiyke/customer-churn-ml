"""Model evaluation, threshold selection, and reporting utilities."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any, Iterable, Tuple
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)


def compute_prob_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def compute_threshold_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def sweep_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    t_min: float = 0.05,
    t_max: float = 0.95,
    t_step: float = 0.01,
) -> Tuple[pd.DataFrame, float]:
    thresholds = np.arange(t_min, t_max + 1e-12, t_step)
    rows = []
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        m = compute_threshold_metrics(y_true, y_proba, float(t))
        rows.append({k: v for k, v in m.items() if k in {"threshold", "precision", "recall", "f1"}})
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
    df = pd.DataFrame(rows)
    return df, best_t


def pr_curve_table(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # thresholds has length n-1; pad for alignment
    thresholds = np.concatenate([thresholds, [np.nan]])
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    return pd.DataFrame({
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

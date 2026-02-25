"""Data loading and feature engineering utilities."""

from __future__ import annotations

from typing import Tuple, List, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COL = "Churn"
ID_COLS = ("customerID",)


def load_churn_data(path: str | Path) -> pd.DataFrame:
    """Load churn CSV and apply deterministic type fixes/cleaning"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing required target column '{TARGET_COL}'. Columns: {list(df.columns)}")

    # Drop identifiers if present
    drop_cols = [c for c in ID_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Fix TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = (
            df["TotalCharges"]
            .astype(str)
            .str.strip()
            .replace({"": np.nan})
        )
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        if "tenure" in df.columns:
            # tenure may be int/float/object; coerce safely
            tenure_num = pd.to_numeric(df["tenure"], errors="coerce")
            df.loc[tenure_num.fillna(-1) == 0, "TotalCharges"] = 0.0

    # Encode target to 0/1
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    service_cols = [
        c for c in [
            "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        if c in out.columns
    ]
    if service_cols:
        out["TotalServices"] = (out[service_cols] == "Yes").sum(axis=1).astype(int)

    return out


def make_splits(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    split: Tuple[float, float, float] = (0.60, 0.20, 0.20),
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Train/Val/Test split with stratification."""
    train_size, val_size, test_size = split
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(f"Split must sum to 1.0. Got: {split}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split off test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Then split train vs val from remaining
    # val fraction of trainval:
    val_frac_of_trainval = val_size / (train_size + val_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_frac_of_trainval,
        stratify=y_trainval,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

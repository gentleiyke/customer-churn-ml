"""Pipeline construction utilities (preprocess + model)."""

from __future__ import annotations

from typing import List, Dict, Tuple
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier


def infer_feature_groups(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer numeric and categorical columns based on pandas dtypes."""
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocess(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Build a leakage-safe preprocessing transformer."""
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocess


def build_model_candidates(random_state: int = 42) -> Dict[str, object]:
    """Return candidate classifiers to compare."""
    return {
        "logreg_balanced": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
        "svc_rbf": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
        "hgb": HistGradientBoostingClassifier(
            random_state=random_state
        ),
    }


def build_pipeline(preprocess: ColumnTransformer, clf: object) -> Pipeline:
    return Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", clf),
    ])

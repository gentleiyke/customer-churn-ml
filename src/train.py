"""Training entrypoint: compare models, choose best on validation, select threshold, save artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, average_precision_score

from config import Settings
from data import load_churn_data, add_features, make_splits, TARGET_COL
from pipeline import infer_feature_groups, build_preprocess, build_model_candidates, build_pipeline
from evaluate import compute_prob_metrics, compute_threshold_metrics, sweep_thresholds, pr_curve_table


def evaluate_candidate(name: str, pipe, X_train, y_train, cv_folds: int, random_state: int) -> Dict[str, Any]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
    out = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=None, return_train_score=False)
    return {
        "model": name,
        "cv_roc_auc_mean": float(np.mean(out["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(out["test_roc_auc"])),
        "cv_pr_auc_mean": float(np.mean(out["test_pr_auc"])),
        "cv_pr_auc_std": float(np.std(out["test_pr_auc"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None, help="Path to churn CSV")
    parser.add_argument("--artifact-dir", type=str, default=None, help="Directory to write artifacts")
    args = parser.parse_args()

    settings = Settings(
        data_path=Path(args.data_path) if args.data_path else Settings().data_path,
        artifact_dir=Path(args.artifact_dir) if args.artifact_dir else Settings().artifact_dir,
    )

    settings.artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load + deterministic cleaning
    df = load_churn_data(settings.data_path)
    df = add_features(df)

    # 2) Split
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(
        df,
        target_col=TARGET_COL,
        split=settings.split,
        random_state=settings.random_state,
    )

    # 3) Preprocess
    num_cols, cat_cols = infer_feature_groups(X_train)
    preprocess = build_preprocess(num_cols, cat_cols)

    # 4) Compare model candidates via CV on TRAIN
    candidates = build_model_candidates(random_state=settings.random_state)
    cv_rows = []
    for name, clf in candidates.items():
        pipe = build_pipeline(preprocess, clf)
        cv_rows.append(evaluate_candidate(name, pipe, X_train, y_train, settings.cv_folds, settings.random_state))
    cv_df = pd.DataFrame(cv_rows).sort_values(by="cv_pr_auc_mean", ascending=False)
    cv_df.to_csv(settings.artifact_dir / "cv_results.csv", index=False)

    # 5) Select best model by validation PR-AUC (fit on TRAIN, eval on VAL)
    val_rows = []
    best_name = None
    best_pr = -1.0
    best_pipe = None

    for name, clf in candidates.items():
        pipe = build_pipeline(preprocess, clf)
        pipe.fit(X_train, y_train)
        val_proba = pipe.predict_proba(X_val)[:, 1]
        m = compute_prob_metrics(y_val.to_numpy(), val_proba)
        val_rows.append({"model": name, **m})
        if m["pr_auc"] > best_pr:
            best_pr = m["pr_auc"]
            best_name = name
            best_pipe = pipe

    val_df = pd.DataFrame(val_rows).sort_values(by="pr_auc", ascending=False)
    val_df.to_csv(settings.artifact_dir / "val_results.csv", index=False)

    assert best_pipe is not None and best_name is not None

    # 6) Threshold selection on VAL (optimize F1)
    val_proba = best_pipe.predict_proba(X_val)[:, 1]
    sweep_df, best_threshold = sweep_thresholds(
        y_val.to_numpy(), val_proba,
        t_min=settings.threshold_min,
        t_max=settings.threshold_max,
        t_step=settings.threshold_step,
    )
    sweep_df.to_csv(settings.artifact_dir / "threshold_sweep.csv", index=False)

    pr_df = pr_curve_table(y_val.to_numpy(), val_proba)
    pr_df.to_csv(settings.artifact_dir / "pr_curve.csv", index=False)

    # 7) Refit best model on TRAIN+VAL, evaluate once on TEST
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)

    final_pipe = build_pipeline(preprocess, candidates[best_name])
    final_pipe.fit(X_trainval, y_trainval)

    test_proba = final_pipe.predict_proba(X_test)[:, 1]
    test_prob_metrics = compute_prob_metrics(y_test.to_numpy(), test_proba)
    test_thresh_metrics = compute_threshold_metrics(y_test.to_numpy(), test_proba, best_threshold)

    metrics = {
        "model_name": best_name,
        "threshold": float(best_threshold),
        "test": {
            **test_prob_metrics,
            "precision": test_thresh_metrics["precision"],
            "recall": test_thresh_metrics["recall"],
            "f1": test_thresh_metrics["f1"],
            "confusion_matrix": test_thresh_metrics["confusion_matrix"],
        },
    }

    # 8) Save artifacts for deployment
    joblib.dump(final_pipe, settings.artifact_dir / "churn_model.joblib")

    schema = {
        "feature_names": list(X_train.columns),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "dtypes": {c: str(X_train[c].dtype) for c in X_train.columns},
        "target": TARGET_COL,
    }
    (settings.artifact_dir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (settings.artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved artifacts to:", settings.artifact_dir.resolve())
    print("Best model:", best_name)
    print("Best threshold (VAL F1):", best_threshold)
    print("TEST metrics:", metrics["test"])


if __name__ == "__main__":
    main()

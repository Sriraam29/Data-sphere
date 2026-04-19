"""
ML Task Detection & Baseline Model
- Detects classification vs regression
- Trains RandomForest + LightGBM baseline
- Computes feature importance and SHAP values
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from logger import TransformationLogger

_CLASSIFICATION_MAX_UNIQUE = 15
_MAX_ROWS_SHAP = 20_000
_MAX_ROWS_MODEL = 100_000


def detect_task_and_train(
    df: pd.DataFrame,
    target_col: str,
    logger: TransformationLogger,
) -> dict:
    """
    Returns ml_report dict with task type, feature importances, SHAP values.
    """
    report: dict = {}

    if target_col not in df.columns:
        return {"error": "target column not found"}

    y = df[target_col].dropna()
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])

    # Align
    common_idx = y.index.intersection(X.index)
    y = y.loc[common_idx]
    X = X.loc[common_idx].fillna(0)

    if len(y) < 20 or X.shape[1] == 0:
        return {"error": "insufficient data for ML analysis"}

    # ── Detect task type ──────────────────────────────────────────────
    n_unique = y.nunique()
    if n_unique <= _CLASSIFICATION_MAX_UNIQUE and not pd.api.types.is_float_dtype(y):
        task = "classification"
        y_encoded = pd.factorize(y)[0]
    else:
        task = "regression"
        y_encoded = y.values.astype(float)

    report["task"] = task
    report["n_unique_target"] = int(n_unique)
    report["n_features"] = X.shape[1]
    report["n_samples"] = len(y)

    logger.log("ml_task_detection", f"Detected task: {task}", reason=f"n_unique={n_unique}")

    # ── Subsample if large ────────────────────────────────────────────
    if len(y) > _MAX_ROWS_MODEL:
        idx = np.random.RandomState(42).choice(len(y), _MAX_ROWS_MODEL, replace=False)
        X_fit = X.iloc[idx]
        y_fit = y_encoded[idx]
    else:
        X_fit = X
        y_fit = y_encoded

    feature_names = list(X.columns)

    # ── Try LightGBM first, fall back to RandomForest ─────────────────
    importances: dict[str, float] = {}
    shap_values_df: pd.DataFrame | None = None
    model_used = "none"
    model_score = None

    try:
        import lightgbm as lgb
        from sklearn.model_selection import cross_val_score

        if task == "classification":
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
            scoring = "roc_auc" if len(np.unique(y_fit)) == 2 else "accuracy"
        else:
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
            scoring = "r2"

        model.fit(X_fit, y_fit)
        model_used = "LightGBM"

        try:
            scores = cross_val_score(model, X_fit, y_fit, cv=3, scoring=scoring, n_jobs=-1)
            model_score = round(float(scores.mean()), 4)
        except Exception:
            model_score = None

        imp = model.feature_importances_
        importances = {f: round(float(v), 6) for f, v in zip(feature_names, imp)}

    except ImportError:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        if task == "classification":
            model = RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
            )
            scoring = "accuracy"
        else:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
            )
            scoring = "r2"

        model.fit(X_fit, y_fit)
        model_used = "RandomForest"

        try:
            scores = cross_val_score(model, X_fit, y_fit, cv=3, scoring=scoring, n_jobs=-1)
            model_score = round(float(scores.mean()), 4)
        except Exception:
            model_score = None

        imp = model.feature_importances_
        importances = {f: round(float(v), 6) for f, v in zip(feature_names, imp)}

    report["model_used"] = model_used
    report["model_score"] = model_score
    report["scoring_metric"] = scoring if "scoring" in dir() else "unknown"
    report["feature_importances"] = dict(
        sorted(importances.items(), key=lambda x: x[1], reverse=True)
    )

    # ── SHAP values ───────────────────────────────────────────────────
    try:
        import shap

        shap_n = min(len(X), _MAX_ROWS_SHAP)
        X_shap = X.sample(n=shap_n, random_state=42) if len(X) > shap_n else X

        if model_used == "LightGBM":
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.TreeExplainer(model)

        sv = explainer.shap_values(X_shap)
        if isinstance(sv, list):
            # Multi-class: use mean absolute across classes
            sv = np.mean(np.abs(np.array(sv)), axis=0)

        mean_abs_shap = np.abs(sv).mean(axis=0)
        shap_dict = {
            f: round(float(v), 6) for f, v in zip(feature_names, mean_abs_shap)
        }
        report["shap_values"] = dict(
            sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
        )

    except Exception as e:
        report["shap_values"] = {}
        report["shap_error"] = str(e)

    logger.log(
        "ml_task_detection",
        "baseline_trained",
        reason=f"model={model_used}, score={model_score}",
    )
    return report

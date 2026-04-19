"""
IntelliML - Utility Module
Data preprocessing, task detection, pipeline construction, export utilities.
"""

from __future__ import annotations
import io
import json
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold


# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    n_numeric: int
    n_categorical: int
    missing_pct: float
    numeric_cols: list[str]
    categorical_cols: list[str]
    target_col: Optional[str] = None
    task_type: Optional[str] = None   # 'binary', 'multiclass', 'regression'
    class_balance: Optional[dict] = None
    feature_correlations: Optional[dict] = None


@dataclass
class ModelResult:
    name: str
    cv_scores: list[float]
    cv_mean: float
    cv_std: float
    train_score: float
    val_score: float
    overfit_gap: float
    training_time: float
    final_score: float
    high_variance: bool = False
    overfitting: bool = False
    params: dict = field(default_factory=dict)
    primary_metric: str = "accuracy"


@dataclass
class SupervisedResults:
    task_type: str
    primary_metric: str
    models: list[ModelResult]
    best_model_name: str
    best_estimator: Any = None
    feature_importance: Optional[pd.DataFrame] = None
    shap_values: Optional[Any] = None
    shap_explainer: Optional[Any] = None
    X_val: Optional[pd.DataFrame] = None
    y_val: Optional[pd.Series] = None
    y_pred: Optional[np.ndarray] = None
    y_pred_proba: Optional[np.ndarray] = None
    stat_equivalent: bool = False
    ttest_pvalue: Optional[float] = None
    label_encoder: Optional[LabelEncoder] = None


@dataclass
class ClusterResult:
    algorithm: str
    k: int
    labels: np.ndarray
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    stability_ari: float
    final_score: float
    params: dict = field(default_factory=dict)


@dataclass
class UnsupervisedResults:
    best_algorithm: str
    best_k: int
    best_labels: np.ndarray
    k_sweep: list[ClusterResult]
    cluster_profiles: pd.DataFrame
    anomaly_labels: np.ndarray       # IsolationForest
    anomaly_scores: np.ndarray
    lof_labels: np.ndarray           # LocalOutlierFactor
    pca_2d: np.ndarray
    tsne_2d: Optional[np.ndarray] = None
    umap_2d: Optional[np.ndarray] = None
    explained_variance_ratio: Optional[np.ndarray] = None


# ─────────────────────────────────────────────
#  TASK DETECTION
# ─────────────────────────────────────────────

def detect_task(series: pd.Series) -> str:
    """Detect ML task type from target column."""
    n_unique = series.nunique()
    dtype = series.dtype

    if dtype == object or n_unique <= 2:
        return "binary" if n_unique <= 2 else "multiclass"
    if n_unique <= 20:
        return "multiclass"
    return "regression"


def auto_select_metric(task: str) -> tuple[str, list[str]]:
    """Return (primary_metric, secondary_metrics) for a task."""
    metrics = {
        "binary":     ("roc_auc",  ["f1", "precision", "recall", "accuracy"]),
        "multiclass": ("f1_macro", ["accuracy", "f1_weighted"]),
        "regression": ("r2",       ["neg_rmse", "neg_mae"]),
    }
    return metrics.get(task, ("accuracy", []))


# ─────────────────────────────────────────────
#  DATA PROFILING
# ─────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame, target_col: Optional[str] = None) -> DataProfile:
    numeric_cols   = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    # Remove target from feature lists
    if target_col:
        numeric_cols   = [c for c in numeric_cols   if c != target_col]
        categorical_cols = [c for c in categorical_cols if c != target_col]

    missing_pct = df.isnull().mean().mean() * 100

    class_balance = None
    task_type = None
    if target_col and target_col in df.columns:
        task_type = detect_task(df[target_col])
        if task_type in ("binary", "multiclass"):
            vc = df[target_col].value_counts(normalize=True)
            class_balance = vc.to_dict()

    # Top correlations with target
    feature_correlations = None
    if target_col and target_col in df.columns:
        num_df = df[numeric_cols + [target_col]].copy()
        if task_type == "regression":
            corr = num_df.corr()[target_col].drop(target_col, errors="ignore")
            feature_correlations = corr.abs().sort_values(ascending=False).head(15).to_dict()

    return DataProfile(
        n_rows=len(df),
        n_cols=len(df.columns),
        n_numeric=len(numeric_cols),
        n_categorical=len(categorical_cols),
        missing_pct=missing_pct,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        target_col=target_col,
        task_type=task_type,
        class_balance=class_balance,
        feature_correlations=feature_correlations,
    )


# ─────────────────────────────────────────────
#  PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    strategy: str = "median",
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for mixed data."""
    transformers = []

    if numeric_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=strategy)),
            ("scaler",  StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_supervised_data(
    df: pd.DataFrame,
    target_col: str,
    profile: DataProfile,
) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer, Optional[LabelEncoder]]:
    """Split into X/y, build preprocessor, encode target if needed."""
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    label_enc = None
    if profile.task_type in ("binary", "multiclass"):
        if y.dtype == object or not pd.api.types.is_integer_dtype(y):
            label_enc = LabelEncoder()
            y = pd.Series(label_enc.fit_transform(y), index=y.index, name=y.name)

    preprocessor = build_preprocessor(profile.numeric_cols, profile.categorical_cols)
    return X, y, preprocessor, label_enc


def prepare_unsupervised_data(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Standardize numeric features, encode categoricals, return matrix."""
    numeric_cols   = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X = preprocessor.fit_transform(df)

    # Remove low-variance features
    selector = VarianceThreshold(threshold=0.01)
    try:
        X = selector.fit_transform(X)
    except Exception:
        pass

    feature_names = numeric_cols + categorical_cols
    return X, feature_names


# ─────────────────────────────────────────────
#  EXPORT UTILITIES
# ─────────────────────────────────────────────

def model_to_bytes(estimator) -> bytes:
    buf = io.BytesIO()
    pickle.dump(estimator, buf)
    return buf.getvalue()


def results_to_leaderboard_csv(models: list[ModelResult]) -> bytes:
    rows = []
    for m in models:
        rows.append({
            "rank":          models.index(m) + 1,
            "model":         m.name,
            "cv_mean":       round(m.cv_mean,  4),
            "cv_std":        round(m.cv_std,   4),
            "train_score":   round(m.train_score, 4),
            "val_score":     round(m.val_score,   4),
            "overfit_gap":   round(m.overfit_gap, 4),
            "final_score":   round(m.final_score, 4),
            "training_time": round(m.training_time, 3),
            "high_variance": m.high_variance,
            "overfitting":   m.overfitting,
            "primary_metric": m.primary_metric,
        })
    return pd.DataFrame(rows).to_csv(index=False).encode()


def results_to_json(models: list[ModelResult]) -> bytes:
    data = []
    for m in models:
        d = {
            "model": m.name,
            "cv_scores": [round(s, 4) for s in m.cv_scores],
            "cv_mean":   round(m.cv_mean,  4),
            "cv_std":    round(m.cv_std,   4),
            "train_score": round(m.train_score, 4),
            "val_score":   round(m.val_score,   4),
            "overfit_gap": round(m.overfit_gap, 4),
            "final_score": round(m.final_score, 4),
            "training_time": round(m.training_time, 3),
        }
        data.append(d)
    return json.dumps(data, indent=2).encode()


# ─────────────────────────────────────────────
#  SCORING HELPERS
# ─────────────────────────────────────────────

def compute_final_score(
    cv_mean: float,
    cv_std: float,
    overfit_gap: float,
    lambda1: float = 0.30,
    lambda2: float = 0.40,
) -> float:
    """IntelliML ranking formula: FinalScore = CV − λ₁·Var − λ₂·OverfitGap"""
    return cv_mean - lambda1 * cv_std - lambda2 * overfit_gap


def scoring_fn_for_task(task: str) -> str:
    """sklearn scorer string for cross_validate."""
    return {
        "binary":     "roc_auc",
        "multiclass": "f1_macro",
        "regression": "r2",
    }.get(task, "accuracy")


METRIC_DISPLAY = {
    "roc_auc":  "ROC-AUC",
    "f1_macro": "Macro F1",
    "r2":       "R² Score",
    "accuracy": "Accuracy",
}

def metric_label(metric: str) -> str:
    return METRIC_DISPLAY.get(metric, metric.replace("_", " ").title())

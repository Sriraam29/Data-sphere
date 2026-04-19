"""
IntelliML - Supervised Learning Engine
Parallel cross-validation, overfitting detection, statistical comparison,
hyperparameter tuning, and SHAP explainability.
"""

from __future__ import annotations
import time
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_validate,
    RandomizedSearchCV, train_test_split,
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve, average_precision_score,
)
from scipy.stats import ttest_rel
import xgboost as xgb
import lightgbm as lgb

from .utils import (
    DataProfile, ModelResult, SupervisedResults,
    compute_final_score, scoring_fn_for_task,
    build_preprocessor, prepare_supervised_data,
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ─────────────────────────────────────────────
#  MODEL REGISTRY
# ─────────────────────────────────────────────

CLASSIFICATION_MODELS: dict[str, Any] = {
    "Logistic Regression":     LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Random Forest":           RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting":       GradientBoostingClassifier(n_estimators=150, random_state=RANDOM_STATE),
    "XGBoost":                 xgb.XGBClassifier(n_estimators=150, random_state=RANDOM_STATE,
                                                  eval_metric="logloss", verbosity=0, n_jobs=-1),
    "LightGBM":                lgb.LGBMClassifier(n_estimators=150, random_state=RANDOM_STATE,
                                                   verbose=-1, n_jobs=-1),
    "Extra Trees":             ExtraTreesClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
    "SVM":                     SVC(probability=True, random_state=RANDOM_STATE),
    "KNN":                     KNeighborsClassifier(n_jobs=-1),
    "Decision Tree":           DecisionTreeClassifier(random_state=RANDOM_STATE),
}

REGRESSION_MODELS: dict[str, Any] = {
    "Ridge Regression":        Ridge(random_state=RANDOM_STATE),
    "Lasso Regression":        Lasso(random_state=RANDOM_STATE),
    "ElasticNet":              ElasticNet(random_state=RANDOM_STATE),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting Reg.":  GradientBoostingRegressor(n_estimators=150, random_state=RANDOM_STATE),
    "XGBoost Regressor":       xgb.XGBRegressor(n_estimators=150, random_state=RANDOM_STATE,
                                                  verbosity=0, n_jobs=-1),
    "LightGBM Regressor":      lgb.LGBMRegressor(n_estimators=150, random_state=RANDOM_STATE,
                                                   verbose=-1, n_jobs=-1),
    "Extra Trees Regressor":   ExtraTreesRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
    "SVR":                     SVR(),
    "KNN Regressor":           KNeighborsRegressor(n_jobs=-1),
}

# Hyperparameter search spaces (RandomizedSearchCV)
PARAM_GRIDS: dict[str, dict] = {
    "Random Forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.5],
    },
    "Random Forest Regressor": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
    },
    "XGBoost": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
    },
    "XGBoost Regressor": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.85, 1.0],
    },
    "LightGBM": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [-1, 5, 10],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__num_leaves": [31, 63, 127],
    },
    "LightGBM Regressor": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [-1, 5, 10],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__num_leaves": [31, 63, 127],
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__subsample": [0.8, 1.0],
    },
    "Gradient Boosting Reg.": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__subsample": [0.8, 1.0],
    },
}

# ─────────────────────────────────────────────
#  SINGLE MODEL TRAINING
# ─────────────────────────────────────────────

def _train_single_model(
    name: str,
    estimator: Any,
    preprocessor,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    cv_folds: int,
    scoring: str,
    lambda1: float,
    lambda2: float,
    progress_cb: Optional[Callable] = None,
) -> ModelResult:
    """Train a single model with cross-validation. Thread-safe."""
    t0 = time.perf_counter()

    # Build full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    # CV strategy
    if task in ("binary", "multiclass"):
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    # Scoring  — handle sign-flipped sklearn metrics
    neg_scorers = {"neg_rmse": "neg_root_mean_squared_error",
                   "neg_mae":  "neg_mean_absolute_error"}
    sk_scoring = neg_scorers.get(scoring, scoring)

    cv_result = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=sk_scoring,
        return_train_score=True,
        n_jobs=1,
    )

    raw_test  = cv_result["test_score"]
    raw_train = cv_result["train_score"]

    # Flip negative scorers back to positive
    if sk_scoring.startswith("neg_"):
        raw_test  = -raw_test
        raw_train = -raw_train

    cv_mean  = float(np.mean(raw_test))
    cv_std   = float(np.std(raw_test))
    train_sc = float(np.mean(raw_train))
    val_sc   = cv_mean
    gap      = abs(train_sc - val_sc)
    elapsed  = time.perf_counter() - t0

    final = compute_final_score(cv_mean, cv_std, gap, lambda1, lambda2)

    if progress_cb:
        progress_cb(name)

    return ModelResult(
        name=name,
        cv_scores=raw_test.tolist(),
        cv_mean=cv_mean,
        cv_std=cv_std,
        train_score=train_sc,
        val_score=val_sc,
        overfit_gap=gap,
        training_time=elapsed,
        final_score=final,
        high_variance=cv_std > 0.04,
        overfitting=gap > 0.10,
        primary_metric=scoring,
    )


# ─────────────────────────────────────────────
#  PARALLEL CROSS-VALIDATION ENGINE
# ─────────────────────────────────────────────

def run_parallel_cv(
    X: pd.DataFrame,
    y: pd.Series,
    profile: DataProfile,
    cv_folds: int = 5,
    lambda1: float = 0.30,
    lambda2: float = 0.40,
    max_workers: int = 4,
    progress_cb: Optional[Callable] = None,
) -> list[ModelResult]:
    """
    Evaluate all models in parallel using ThreadPoolExecutor.
    Returns list sorted by final_score descending.
    """
    task    = profile.task_type
    scoring = scoring_fn_for_task(task)
    models  = CLASSIFICATION_MODELS if task in ("binary", "multiclass") else REGRESSION_MODELS
    preprocessor = build_preprocessor(profile.numeric_cols, profile.categorical_cols)

    results: list[ModelResult] = []
    futures = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for name, estimator in models.items():
            # Each thread gets its own clone of preprocessor
            from sklearn.base import clone as sk_clone
            fut = executor.submit(
                _train_single_model,
                name, sk_clone(estimator), sk_clone(preprocessor),
                X, y, task, cv_folds, scoring,
                lambda1, lambda2, progress_cb,
            )
            futures[fut] = name

        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                name = futures[fut]
                # Append a placeholder result on failure
                results.append(ModelResult(
                    name=name, cv_scores=[], cv_mean=0.0, cv_std=1.0,
                    train_score=0.0, val_score=0.0, overfit_gap=1.0,
                    training_time=0.0, final_score=-1.0,
                    high_variance=True, overfitting=True,
                    primary_metric=scoring,
                ))

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results


# ─────────────────────────────────────────────
#  STATISTICAL COMPARISON
# ─────────────────────────────────────────────

def statistical_comparison(
    r1: ModelResult,
    r2: ModelResult,
    alpha: float = 0.05,
) -> tuple[float, bool]:
    """
    Paired t-test between CV fold scores of top 2 models.
    Returns (p_value, is_equivalent).
    """
    if len(r1.cv_scores) < 3 or len(r2.cv_scores) < 3:
        return 1.0, True
    s1 = np.array(r1.cv_scores[:len(r2.cv_scores)])
    s2 = np.array(r2.cv_scores[:len(r1.cv_scores)])
    try:
        _, pval = ttest_rel(s1, s2)
        return float(pval), pval > alpha
    except Exception:
        return 1.0, True


# ─────────────────────────────────────────────
#  HYPERPARAMETER TUNING
# ─────────────────────────────────────────────

def tune_best_model(
    name: str,
    estimator: Any,
    preprocessor,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    cv_folds: int = 3,
    n_iter: int = 20,
) -> tuple[Any, dict]:
    """
    RandomizedSearchCV on best model if param grid available.
    Returns (tuned_pipeline, best_params).
    """
    if name not in PARAM_GRIDS:
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X, y)
        return pipeline, {}

    scoring = scoring_fn_for_task(task)
    neg_map = {"neg_rmse": "neg_root_mean_squared_error", "neg_mae": "neg_mean_absolute_error"}
    sk_scoring = neg_map.get(scoring, scoring)

    cv = (StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
          if task in ("binary", "multiclass")
          else KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE))

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_GRIDS[name],
        n_iter=n_iter,
        cv=cv,
        scoring=sk_scoring,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


# ─────────────────────────────────────────────
#  FINAL MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    profile: DataProfile,
    best_name: str,
    tune: bool = False,
    tune_iters: int = 20,
) -> tuple[Pipeline, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Train best model on train split, evaluate on held-out test set.
    Returns (pipeline, y_test, y_pred, y_pred_proba, X_test).
    """
    from sklearn.base import clone as sk_clone

    task      = profile.task_type
    model_map = CLASSIFICATION_MODELS if task in ("binary", "multiclass") else REGRESSION_MODELS
    estimator = sk_clone(model_map[best_name])
    preprocessor = build_preprocessor(profile.numeric_cols, profile.categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE,
        stratify=y if task in ("binary", "multiclass") else None,
    )

    if tune and best_name in PARAM_GRIDS:
        pipeline, _ = tune_best_model(
            best_name, estimator, preprocessor, X_train, y_train, task, tune_iters
        )
    else:
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = None
    if task in ("binary", "multiclass"):
        try:
            y_pred_proba = pipeline.predict_proba(X_test)
        except Exception:
            pass

    return pipeline, X_test, y_test, y_pred, y_pred_proba


# ─────────────────────────────────────────────
#  FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def extract_feature_importance(
    pipeline: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract feature importances from final estimator."""
    model = pipeline.named_steps["model"]
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef).flatten() if coef.ndim > 1 else np.abs(coef)

    if importances is None or len(importances) == 0:
        return pd.DataFrame()

    # Align length with feature names
    min_len = min(len(importances), len(feature_names))
    df = pd.DataFrame({
        "feature":    feature_names[:min_len],
        "importance": importances[:min_len],
    }).sort_values("importance", ascending=False)
    df["importance_pct"] = df["importance"] / df["importance"].sum() * 100
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
#  SHAP EXPLAINABILITY
# ─────────────────────────────────────────────

def compute_shap(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    profile: DataProfile,
    max_samples: int = 100,
):
    """
    Compute SHAP values for the final model.
    Returns (shap_values, explainer, X_transformed).
    """
    try:
        import shap
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        X_sample = X_val.sample(min(max_samples, len(X_val)), random_state=RANDOM_STATE)
        X_tr = preprocessor.transform(X_sample)

        # Choose explainer
        if hasattr(model, "predict_proba") and hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                shap.sample(X_tr, 30),
            )

        shap_vals = explainer.shap_values(X_tr)
        return shap_vals, explainer, X_tr, X_sample

    except Exception:
        return None, None, None, None


# ─────────────────────────────────────────────
#  METRICS COMPUTATION
# ─────────────────────────────────────────────

def compute_classification_metrics(
    y_true, y_pred, y_pred_proba, task: str, label_encoder=None
) -> dict:
    labels = label_encoder.classes_ if label_encoder else None
    cm  = confusion_matrix(y_true, y_pred)
    cr  = classification_report(y_true, y_pred, output_dict=True, target_names=labels)

    result = {"confusion_matrix": cm, "classification_report": cr}

    # ROC curve (binary)
    if task == "binary" and y_pred_proba is not None:
        proba = y_pred_proba[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        result["roc"] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
        result["prc"] = {"precision": precision, "recall": recall, "ap": ap}

    return result


def compute_regression_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape,
            "residuals": (y_pred - y_true).tolist()}


# ─────────────────────────────────────────────
#  OPTUNA TUNING (ADVANCED)
# ─────────────────────────────────────────────

def optuna_tune(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    profile: DataProfile,
    n_trials: int = 30,
    progress_cb: Optional[Callable] = None,
) -> tuple[Any, float, dict]:
    """
    Optuna-based hyperparameter optimization for XGBoost and LightGBM.
    Returns (best_model_instance, best_score, best_params).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return None, 0.0, {}

    task    = profile.task_type
    scoring = scoring_fn_for_task(task)

    if task in ("binary", "multiclass"):
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    preprocessor = build_preprocessor(profile.numeric_cols, profile.categorical_cols)

    def objective(trial):
        if "XGBoost" in name:
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
                "max_depth":        trial.suggest_int("max_depth", 2, 10),
                "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": -1,
                "eval_metric": "logloss" if task != "regression" else "rmse",
            }
            if task == "regression":
                model = xgb.XGBRegressor(**params)
            elif task == "multiclass":
                model = xgb.XGBClassifier(**params, num_class=y.nunique())
            else:
                model = xgb.XGBClassifier(**params)
        elif "LightGBM" in name:
            params = {
                "n_estimators":  trial.suggest_int("n_estimators", 50, 400),
                "max_depth":     trial.suggest_int("max_depth", 2, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "num_leaves":    trial.suggest_int("num_leaves", 20, 300),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha":     trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":    trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": RANDOM_STATE, "verbose": -1, "n_jobs": -1,
            }
            model = lgb.LGBMRegressor(**params) if task == "regression" else lgb.LGBMClassifier(**params)
        else:
            return 0.0

        neg_map = {"neg_rmse": "neg_root_mean_squared_error", "neg_mae": "neg_mean_absolute_error"}
        sk_scoring = neg_map.get(scoring, scoring)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=sk_scoring, n_jobs=1)
        score = np.mean(scores["test_score"])
        return -score if sk_scoring.startswith("neg_") else score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score  = study.best_value

    return None, best_score, best_params

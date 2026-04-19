"""
Advanced Missing Value Intelligence
Decision rules based on skewness, correlation, missingness patterns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from logger import TransformationLogger

_DROP_MISSING_THRESHOLD = 0.60
_CORR_IMPUTE_THRESHOLD = 0.50


def _is_time_indexed(df: pd.DataFrame) -> bool:
    return pd.api.types.is_datetime64_any_dtype(df.index)


def handle_missing_values(
    df: pd.DataFrame,
    schema: dict,
    target_col: str | None,
    logger: TransformationLogger,
) -> pd.DataFrame:
    df = df.copy()
    n_rows = len(df)

    # --- Collect numeric cols for correlation-based imputation ---------------
    numeric_cols = [
        c for c, m in schema.items() if m["inferred_type"] == "numeric" and c in df.columns
    ]

    # Compute missingness-target correlation if target exists
    miss_target_corr: dict[str, float] = {}
    if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        for col in df.columns:
            miss_indicator = df[col].isna().astype(float)
            if miss_indicator.sum() > 5:
                corr = abs(miss_indicator.corr(df[target_col]))
                if not np.isnan(corr):
                    miss_target_corr[col] = float(corr)

    # Numeric correlation matrix for predictive imputation decisions
    numeric_df = df[numeric_cols].copy() if numeric_cols else pd.DataFrame()
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr().abs()
    else:
        corr_matrix = pd.DataFrame()

    dropped_cols: list[str] = []

    for col in list(df.columns):
        if col not in schema:
            continue
        col_type = schema[col]["inferred_type"]
        missing_rate = df[col].isna().mean()

        if missing_rate == 0.0:
            continue

        # ── Rule 1: Drop if >60% missing AND low variance ─────────────────
        if missing_rate >= _DROP_MISSING_THRESHOLD:
            if col_type == "numeric":
                non_null = df[col].dropna()
                variance = float(non_null.var()) if len(non_null) > 1 else 0.0
                if variance < 1e-6 or non_null.nunique() <= 2:
                    df = df.drop(columns=[col])
                    dropped_cols.append(col)
                    logger.warn(
                        "missing_values",
                        "drop_column",
                        column=col,
                        reason=f"missing={missing_rate:.1%}, variance={variance:.4f}",
                    )
                    continue
            else:
                # Non-numeric: drop if very high missing AND not correlated w/ target
                target_corr = miss_target_corr.get(col, 0.0)
                if target_corr < 0.15:
                    df = df.drop(columns=[col])
                    dropped_cols.append(col)
                    logger.warn(
                        "missing_values",
                        "drop_column",
                        column=col,
                        reason=f"missing={missing_rate:.1%}, miss-target corr={target_corr:.3f}",
                    )
                    continue

        # ── Rule 2: Preserve missingness as signal ────────────────────────
        target_signal = miss_target_corr.get(col, 0.0) > 0.15
        if target_signal:
            indicator_col = f"{col}__was_missing"
            df[indicator_col] = df[col].isna().astype(np.int8)
            logger.log(
                "missing_values",
                "add_missing_indicator",
                column=col,
                reason=f"miss-target corr={miss_target_corr[col]:.3f} — preserving signal",
            )

        # ── Rule 3: Numeric imputation ────────────────────────────────────
        if col_type == "numeric":
            non_null = df[col].dropna()
            if len(non_null) == 0:
                df[col] = df[col].fillna(0)
                continue
            skewness = float(non_null.skew())

            # Check corr with other numeric features
            best_corr = 0.0
            if col in corr_matrix.columns:
                others = corr_matrix[col].drop(index=col, errors="ignore")
                best_corr = float(others.max()) if len(others) > 0 else 0.0

            if _is_time_indexed(df):
                df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
                strategy = "forward_fill"
            elif best_corr >= _CORR_IMPUTE_THRESHOLD and n_rows <= 50_000:
                # Predictive imputation via IterativeImputer
                try:
                    imp = IterativeImputer(max_iter=5, random_state=42, n_nearest_features=5)
                    sub = df[numeric_cols].copy()
                    df[numeric_cols] = imp.fit_transform(sub)
                    strategy = "iterative_imputer"
                except Exception:
                    df[col] = df[col].fillna(non_null.median())
                    strategy = "median_fallback"
            elif abs(skewness) > 2.0:
                df[col] = df[col].fillna(non_null.median())
                strategy = "median"
            else:
                df[col] = df[col].fillna(non_null.mean())
                strategy = "mean"

            logger.log(
                "missing_values",
                f"impute_{strategy}",
                column=col,
                before=f"missing={missing_rate:.2%}",
                after="0%",
                reason=f"skew={skewness:.2f}, best_corr={best_corr:.2f}",
            )

        # ── Rule 4: Categorical / ordinal imputation ───────────────────────
        elif col_type in ("categorical", "ordinal", "boolean", "text"):
            non_null = df[col].dropna()
            if len(non_null) == 0:
                df[col] = df[col].fillna("Unknown")
                continue
            mode_val = non_null.mode()
            if len(mode_val) == 0:
                df[col] = df[col].fillna("Missing")
                strategy = "missing_category"
            else:
                mode_val = mode_val.iloc[0]
                mode_freq = (non_null == mode_val).mean()
                if mode_freq >= 0.50:
                    df[col] = df[col].fillna(mode_val)
                    strategy = f"mode({mode_val})"
                else:
                    df[col] = df[col].fillna("Missing")
                    strategy = "missing_category"

            logger.log(
                "missing_values",
                f"impute_{strategy}",
                column=col,
                before=f"missing={missing_rate:.2%}",
                after="0%",
            )

        elif col_type == "datetime":
            df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
            logger.log("missing_values", "impute_ffill", column=col)

    return df

"""
Automatic Feature Selection
1. Variance threshold removal
2. High-correlation filter (>0.95 pairwise)
3. Mutual information ranking
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from logger import TransformationLogger

_VARIANCE_THRESHOLD = 0.001
_CORR_THRESHOLD = 0.95
_MI_TOP_K_FRACTION = 0.90  # Keep top 90% by MI score


def select_features(
    df: pd.DataFrame,
    target_col: str | None,
    logger: TransformationLogger,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns (selected_df, feature_selection_report).
    """
    df = df.copy()
    report: dict = {
        "removed_variance": [],
        "removed_correlation": [],
        "mi_scores": {},
        "kept": [],
    }

    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        return df, report

    # Only work on numeric columns for selection
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return df, report

    X = df[num_cols].copy()

    # ── Step 1: Variance threshold ────────────────────────────────────
    try:
        sel = VarianceThreshold(threshold=_VARIANCE_THRESHOLD)
        sel.fit(X.fillna(0))
        low_var_mask = sel.get_support()
        low_var_removed = [c for c, keep in zip(num_cols, low_var_mask) if not keep]
        if low_var_removed:
            df = df.drop(columns=low_var_removed)
            X = X.drop(columns=low_var_removed)
            num_cols = [c for c in num_cols if c not in low_var_removed]
            report["removed_variance"] = low_var_removed
            for col in low_var_removed:
                logger.log(
                    "feature_selection",
                    "remove_low_variance",
                    column=col,
                    reason=f"variance < {_VARIANCE_THRESHOLD}",
                )
    except Exception:
        pass

    # ── Step 2: High-correlation filter ──────────────────────────────
    if len(num_cols) > 1:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop_corr = [
            col
            for col in upper.columns
            if (upper[col] > _CORR_THRESHOLD).any()
        ]
        if to_drop_corr:
            df = df.drop(columns=to_drop_corr, errors="ignore")
            X = X.drop(columns=to_drop_corr, errors="ignore")
            num_cols = [c for c in num_cols if c not in to_drop_corr]
            report["removed_correlation"] = to_drop_corr
            for col in to_drop_corr:
                logger.log(
                    "feature_selection",
                    "remove_high_correlation",
                    column=col,
                    reason=f"corr > {_CORR_THRESHOLD} with another feature",
                    severity="warning",
                )

    # ── Step 3: Mutual information ────────────────────────────────────
    if target_col and target_col in df.columns and num_cols:
        y = df[target_col].dropna()
        X_aligned = X.reindex(y.index).fillna(0)
        common_cols = list(X_aligned.columns)
        if len(common_cols) > 0 and len(y) > 10:
            try:
                if y.nunique() <= 15:
                    mi_scores = mutual_info_classif(
                        X_aligned[common_cols], y, random_state=42, n_neighbors=3
                    )
                else:
                    mi_scores = mutual_info_regression(
                        X_aligned[common_cols], y, random_state=42, n_neighbors=3
                    )
                mi_dict = dict(zip(common_cols, mi_scores))
                report["mi_scores"] = {k: round(float(v), 5) for k, v in mi_dict.items()}

                # Optionally remove near-zero MI features
                threshold = 0.0
                zero_mi = [c for c, s in mi_dict.items() if s <= threshold]
                if zero_mi and len(zero_mi) < len(common_cols) * 0.5:
                    df = df.drop(columns=zero_mi, errors="ignore")
                    for col in zero_mi:
                        logger.log(
                            "feature_selection",
                            "remove_zero_mi",
                            column=col,
                            reason="mutual information = 0 with target",
                        )
            except Exception:
                pass

    report["kept"] = [c for c in df.columns if c != target_col]
    return df, report

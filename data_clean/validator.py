"""
Data Quality Validator
Post-cleaning checks to guarantee dataset integrity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from logger import TransformationLogger


def validate(df: pd.DataFrame, logger: TransformationLogger) -> dict:
    """
    Runs all post-cleaning checks. Returns validation_report.
    """
    checks: dict[str, dict] = {}

    # ── 1. Missing values ──────────────────────────────────────────────
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0].to_dict()
    checks["no_missing_values"] = {
        "passed": len(missing_cols) == 0,
        "failed_cols": missing_cols,
    }
    if missing_cols:
        logger.warn(
            "validation",
            "residual_missing_values",
            reason=f"Columns with remaining NaN: {list(missing_cols.keys())}",
        )

    # ── 2. Infinite values ─────────────────────────────────────────────
    numeric_df = df.select_dtypes(include=[np.number])
    inf_mask = np.isinf(numeric_df).sum()
    inf_cols = inf_mask[inf_mask > 0].to_dict()
    checks["no_infinite_values"] = {
        "passed": len(inf_cols) == 0,
        "failed_cols": inf_cols,
    }
    if inf_cols:
        logger.warn("validation", "infinite_values_found", reason=str(inf_cols))
        # Auto-fix: replace inf with nan then median
        for col in inf_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median())

    # ── 3. Constant columns ────────────────────────────────────────────
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    checks["no_constant_columns"] = {
        "passed": len(const_cols) == 0,
        "failed_cols": const_cols,
    }
    if const_cols:
        logger.warn(
            "validation", "constant_columns", reason=f"Constant cols: {const_cols}"
        )

    # ── 4. Duplicate rows ──────────────────────────────────────────────
    n_dupes = int(df.duplicated().sum())
    checks["no_duplicate_rows"] = {
        "passed": n_dupes == 0,
        "n_duplicates": n_dupes,
    }
    if n_dupes > 0:
        logger.log(
            "validation",
            "duplicate_rows_found",
            reason=f"{n_dupes} duplicate rows",
            severity="warning",
        )

    # ── 5. Valid dtypes (no remaining objects in numeric context) ──────
    object_cols = list(df.select_dtypes(include="object").columns)
    checks["dtype_integrity"] = {
        "passed": True,  # Informational only
        "remaining_object_cols": object_cols,
    }

    # ── 6. Shape report ────────────────────────────────────────────────
    checks["shape"] = {
        "rows": len(df),
        "columns": len(df.columns),
        "passed": True,
    }

    all_passed = all(v.get("passed", True) for v in checks.values())
    checks["overall_passed"] = all_passed

    logger.log(
        "validation",
        "validation_complete",
        reason=f"all_passed={all_passed}",
    )

    return checks, df

"""
Data Drift Detection
Compare distribution of reference vs new dataset.
KS test for numeric, Chi-square for categorical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from logger import TransformationLogger

_DRIFT_P_THRESHOLD = 0.05


def detect_drift(
    df_ref: pd.DataFrame,
    df_new: pd.DataFrame,
    schema: dict,
    logger: TransformationLogger,
) -> dict:
    """
    Compare distributions column by column.
    Returns drift_report dict.
    """
    report: dict[str, dict] = {}
    common_cols = [c for c in df_ref.columns if c in df_new.columns and c in schema]

    for col in common_cols:
        col_type = schema[col]["inferred_type"]
        ref_series = df_ref[col].dropna()
        new_series = df_new[col].dropna()

        if len(ref_series) < 10 or len(new_series) < 10:
            continue

        result: dict = {"column": col, "type": col_type}

        try:
            if col_type == "numeric":
                stat, p_value = stats.ks_2samp(
                    ref_series.values.astype(float),
                    new_series.values.astype(float),
                )
                result.update(
                    {
                        "test": "KS",
                        "statistic": round(float(stat), 5),
                        "p_value": round(float(p_value), 5),
                        "drifted": p_value < _DRIFT_P_THRESHOLD,
                        "ref_mean": round(float(ref_series.mean()), 4),
                        "new_mean": round(float(new_series.mean()), 4),
                        "ref_std": round(float(ref_series.std()), 4),
                        "new_std": round(float(new_series.std()), 4),
                    }
                )
            elif col_type in ("categorical", "ordinal", "boolean"):
                ref_counts = ref_series.astype(str).value_counts()
                new_counts = new_series.astype(str).value_counts()
                all_cats = set(ref_counts.index) | set(new_counts.index)
                ref_vec = np.array([ref_counts.get(c, 0) for c in all_cats])
                new_vec = np.array([new_counts.get(c, 0) for c in all_cats])
                # Add small smoothing to avoid zero cells
                ref_vec = ref_vec + 1
                new_vec = new_vec + 1
                chi2, p_value = stats.chisquare(
                    new_vec / new_vec.sum() * len(new_series),
                    f_exp=ref_vec / ref_vec.sum() * len(new_series),
                )
                result.update(
                    {
                        "test": "chi2",
                        "statistic": round(float(chi2), 5),
                        "p_value": round(float(p_value), 5),
                        "drifted": p_value < _DRIFT_P_THRESHOLD,
                    }
                )
            else:
                continue

        except Exception as e:
            result["error"] = str(e)
            result["drifted"] = False

        report[col] = result

        if result.get("drifted", False):
            logger.warn(
                "drift_detection",
                "distribution_drift_detected",
                column=col,
                reason=f"p={result.get('p_value')}, test={result.get('test')}",
            )

    n_drifted = sum(1 for v in report.values() if v.get("drifted", False))
    report["__summary__"] = {
        "total_columns_tested": len(report) - 1,
        "drifted_columns": n_drifted,
        "drift_rate": round(n_drifted / max(len(report) - 1, 1), 4),
    }

    return report

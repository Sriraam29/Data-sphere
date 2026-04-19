"""
Bias Detection
- For categorical sensitive attributes, check target imbalance across groups
- Compute group mean differences
- Flag potential fairness issues
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from logger import TransformationLogger

_SENSITIVE_PATTERNS = [
    "gender", "sex", "race", "ethnicity", "age", "religion",
    "nationality", "disability", "income", "education", "marital",
]
_BIAS_THRESHOLD = 0.10  # 10% group mean difference


def detect_bias(
    df_original: pd.DataFrame,
    schema: dict,
    target_col: str | None,
    logger: TransformationLogger,
) -> dict:
    """
    Run on the ORIGINAL (pre-encoded) dataframe for interpretable group names.
    Returns bias_report dict.
    """
    report: dict[str, dict] = {}

    if not target_col or target_col not in df_original.columns:
        return report

    target = df_original[target_col]
    is_numeric_target = pd.api.types.is_numeric_dtype(target)

    # Find sensitive-looking columns
    sensitive_cols = []
    for col in df_original.columns:
        if col == target_col:
            continue
        col_lower = col.lower()
        if any(pat in col_lower for pat in _SENSITIVE_PATTERNS):
            if schema.get(col, {}).get("inferred_type") in ("categorical", "boolean", "ordinal"):
                sensitive_cols.append(col)

    for col in sensitive_cols:
        groups = df_original[col].dropna().unique()
        if len(groups) < 2 or len(groups) > 20:
            continue

        group_stats: dict[str, dict] = {}
        overall_mean = float(target.mean()) if is_numeric_target else None

        for group in groups:
            mask = df_original[col] == group
            group_target = target[mask]
            if len(group_target) < 5:
                continue

            if is_numeric_target:
                gmean = float(group_target.mean())
                gstd = float(group_target.std())
                group_stats[str(group)] = {
                    "n": int(len(group_target)),
                    "mean": round(gmean, 4),
                    "std": round(gstd, 4),
                    "diff_from_overall": round(abs(gmean - overall_mean), 4)
                    if overall_mean is not None
                    else None,
                }
            else:
                # Classification: check positive rate per group
                if target.nunique() == 2:
                    pos_class = sorted(target.unique())[-1]
                    pos_rate = float((group_target == pos_class).mean())
                    group_stats[str(group)] = {
                        "n": int(len(group_target)),
                        "positive_rate": round(pos_rate, 4),
                    }

        if not group_stats:
            continue

        # Compute max disparity
        if is_numeric_target:
            means = [v["mean"] for v in group_stats.values() if "mean" in v]
            if not means:
                continue
            disparity = max(means) - min(means)
            is_biased = disparity > _BIAS_THRESHOLD * (abs(overall_mean) + 1e-8) if overall_mean else False
        else:
            rates = [v.get("positive_rate", 0) for v in group_stats.values()]
            disparity = max(rates) - min(rates) if rates else 0.0
            is_biased = disparity > _BIAS_THRESHOLD

        # Statistical test (ANOVA for numeric, Chi-square for categorical)
        p_value = None
        try:
            groups_data = [
                target[df_original[col] == g].dropna().values
                for g in df_original[col].dropna().unique()
                if len(target[df_original[col] == g].dropna()) >= 5
            ]
            if len(groups_data) >= 2:
                if is_numeric_target:
                    _, p_value = stats.f_oneway(*groups_data)
                else:
                    contingency = pd.crosstab(df_original[col], target)
                    _, p_value, _, _ = stats.chi2_contingency(contingency)
                p_value = float(p_value)
        except Exception:
            pass

        report[col] = {
            "is_sensitive_attribute": True,
            "is_biased": is_biased,
            "disparity": round(disparity, 4),
            "p_value": round(p_value, 5) if p_value is not None else None,
            "statistically_significant": p_value is not None and p_value < 0.05,
            "group_stats": group_stats,
        }

        if is_biased:
            logger.warn(
                "bias_detection",
                "potential_bias_detected",
                column=col,
                reason=f"disparity={disparity:.3f}, p={p_value}",
            )
        else:
            logger.log(
                "bias_detection",
                "no_significant_bias",
                column=col,
                reason=f"disparity={disparity:.3f}",
            )

    return report

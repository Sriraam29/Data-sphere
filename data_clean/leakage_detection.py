"""
Data Leakage Detection
- Features with correlation > 0.98 to target
- Post-event timestamp detection
- Encoded target leakage
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from logger import TransformationLogger

_LEAKAGE_CORR_THRESHOLD = 0.98


def detect_leakage(
    df: pd.DataFrame,
    schema: dict,
    target_col: str | None,
    logger: TransformationLogger,
) -> dict:
    """
    Returns leakage report dict.
    Does NOT remove columns — only reports (user decides).
    """
    report: dict[str, dict] = {}

    if not target_col or target_col not in df.columns:
        return report

    target = df[target_col]
    if not pd.api.types.is_numeric_dtype(target):
        # encode for correlation
        target = pd.factorize(target)[0]
        target = pd.Series(target, index=df.index)

    for col in df.columns:
        if col == target_col:
            continue

        flags: list[str] = []

        # ── Flag 1: Schema-level heuristic risk ──────────────────────
        if schema.get(col, {}).get("is_leakage_risk", False):
            flags.append("name_heuristic")

        # ── Flag 2: Numerical correlation to target ───────────────────
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            corr = abs(float(series.fillna(series.median()).corr(target)))
            if corr >= _LEAKAGE_CORR_THRESHOLD:
                flags.append(f"corr_to_target={corr:.4f}")
                logger.critical(
                    "leakage_detection",
                    "high_target_correlation",
                    column=col,
                    reason=f"corr={corr:.4f} >= {_LEAKAGE_CORR_THRESHOLD}",
                )
        else:
            corr = None

        # ── Flag 3: Datetime after target ─────────────────────────────
        if schema.get(col, {}).get("inferred_type") == "datetime":
            flags.append("datetime_feature_check_manually")

        if flags:
            report[col] = {
                "flags": flags,
                "correlation_to_target": round(corr, 4) if isinstance(corr, float) else None,
            }
            if "corr_to_target" not in " ".join(flags):
                logger.warn(
                    "leakage_detection",
                    "potential_leakage",
                    column=col,
                    reason=str(flags),
                )

    return report

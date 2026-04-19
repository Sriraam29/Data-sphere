"""
Memory Optimization
- Downcast numeric columns to smallest fitting type
- Convert low-cardinality object cols to category
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from logger import TransformationLogger


def optimize_memory(
    df: pd.DataFrame,
    schema: dict,
    logger: TransformationLogger,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns (optimized_df, memory_report).
    """
    before_mb = df.memory_usage(deep=True).sum() / 1024 ** 2

    df = df.copy()

    for col in df.columns:
        series = df[col]

        # ── Integer downcasting ──────────────────────────────────────
        if pd.api.types.is_integer_dtype(series):
            col_min = series.min()
            col_max = series.max()
            for dtype in [np.int8, np.int16, np.int32, np.int64]:
                info = np.iinfo(dtype)
                if col_min >= info.min and col_max <= info.max:
                    df[col] = series.astype(dtype)
                    break

        # ── Float downcasting ─────────────────────────────────────────
        elif pd.api.types.is_float_dtype(series):
            df[col] = pd.to_numeric(series, downcast="float")

        # ── Object → category ─────────────────────────────────────────
        elif series.dtype == object:
            cardinality_ratio = series.nunique() / max(len(series), 1)
            if cardinality_ratio < 0.50:
                df[col] = series.astype("category")

    after_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    reduction_pct = (1 - after_mb / before_mb) * 100 if before_mb > 0 else 0.0

    report = {
        "before_mb": round(float(before_mb), 3),
        "after_mb": round(float(after_mb), 3),
        "reduction_pct": round(float(reduction_pct), 2),
    }

    logger.log(
        "memory_optimization",
        "downcast_complete",
        reason=f"{before_mb:.2f}MB → {after_mb:.2f}MB ({reduction_pct:.1f}% reduction)",
    )
    return df, report

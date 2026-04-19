"""
Advanced Feature Scaling
- Normal distribution → StandardScaler
- Skewed → RobustScaler
- Bounded [0,1] range → MinMaxScaler
- Binary, frequency-encoded, target → skip
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from logger import TransformationLogger

_SKEW_THRESHOLD = 1.0
_BOUNDED_RANGE = (0.0, 1.0)


def _detect_scaler(series: pd.Series) -> str:
    clean = series.dropna()
    if len(clean) < 5:
        return "none"
    try:
        # Convert to plain numpy float to avoid MaskedArray issues
        clean = pd.Series(clean.values.astype(float))
        skewness = abs(float(clean.skew()))
        col_min = float(clean.min())
        col_max = float(clean.max())
    except Exception:
        return "standard"
    # Already bounded [0,1]
    if col_min >= 0.0 and col_max <= 1.0 and (col_max - col_min) <= 1.0:
        return "minmax"
    if skewness > _SKEW_THRESHOLD:
        return "robust"
    return "standard"


def scale_features(
    df: pd.DataFrame,
    schema: dict,
    target_col: str | None,
    encoding_map: dict,
    logger: TransformationLogger,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns (scaled_df, scaler_map).
    scaler_map: {col: fitted_scaler}
    """
    df = df.copy()
    scaler_map: dict[str, object] = {}

    # Identify one-hot-generated columns — skip scaling
    ohe_cols: set[str] = set()
    for col, enc in encoding_map.items():
        if enc.get("method") == "onehot":
            ohe_cols.update(enc.get("new_cols", []))

    # Identify frequency-encoded cols
    freq_encoded_cols = {
        col for col, enc in encoding_map.items() if enc.get("method") == "frequency"
    }

    for col in df.columns:
        if col == target_col:
            continue
        if col in ohe_cols:
            continue
        if col in freq_encoded_cols:
            continue
        if col not in df.columns:
            continue

        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        # Convert category dtype to float to avoid MaskedArray issues
        if hasattr(series, 'cat'):
            try:
                series = series.astype(float)
                df[col] = series
            except Exception:
                continue

        # Skip binary columns
        unique_vals = set(series.dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0, -1, -1.0}):
            continue

        # Check schema for boolean
        if col in schema and schema[col]["inferred_type"] == "boolean":
            continue

        scaler_type = _detect_scaler(series)
        if scaler_type == "none":
            continue

        values = series.values.reshape(-1, 1)

        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            scaler = MinMaxScaler()

        try:
            df[col] = scaler.fit_transform(values).ravel().astype(np.float32)
            scaler_map[col] = scaler
            logger.log(
                "scaling",
                f"scale_{scaler_type}",
                column=col,
                reason=f"scaler={scaler_type}",
            )
        except Exception as e:
            logger.warn("scaling", "scale_failed", column=col, reason=str(e))

    return df, scaler_map

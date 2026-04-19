"""
Advanced Outlier Handler
- Distribution testing (Shapiro-Wilk, skewness, kurtosis)
- Z-score capping for normal distributions
- IQR method for non-normal
- Winsorization for heavy-tailed
- Warns if >10% flagged
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.mstats import winsorize
from logger import TransformationLogger

_SHAPIRO_MAX_N = 5_000  # scipy limit for Shapiro
_OUTLIER_WARN_THRESHOLD = 0.10
_ZSCORE_THRESHOLD = 3.0
_IQR_MULTIPLIER = 1.5


def _test_normality(series: pd.Series) -> tuple[bool, float, float]:
    """Returns (is_normal, skewness, kurtosis)."""
    clean = series.dropna()
    if len(clean) < 8:
        return False, float(clean.skew()), float(clean.kurt())
    skewness = float(clean.skew())
    kurt = float(clean.kurt())
    n = min(len(clean), _SHAPIRO_MAX_N)
    sample = clean.sample(n=n, random_state=42) if len(clean) > n else clean
    try:
        _, p_value = stats.shapiro(sample)
        is_normal = p_value > 0.05 and abs(skewness) < 1.0
    except Exception:
        is_normal = abs(skewness) < 1.0
    return is_normal, skewness, kurt


def _cap_zscore(series: pd.Series, threshold: float = _ZSCORE_THRESHOLD) -> pd.Series:
    mean = series.mean()
    std = series.std()
    if std == 0:
        return series
    lower = mean - threshold * std
    upper = mean + threshold * std
    return series.clip(lower=lower, upper=upper)


def _cap_iqr(series: pd.Series, multiplier: float = _IQR_MULTIPLIER) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return series.clip(lower=lower, upper=upper)


def _winsorize_series(series: pd.Series, limits=(0.01, 0.01)) -> pd.Series:
    arr = winsorize(series.values, limits=limits)
    return pd.Series(arr, index=series.index, name=series.name)


def handle_outliers(
    df: pd.DataFrame,
    schema: dict,
    logger: TransformationLogger,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns cleaned df and outlier_report dict.
    """
    df = df.copy()
    outlier_report: dict[str, dict] = {}

    numeric_cols = [
        c
        for c, m in schema.items()
        if m["inferred_type"] == "numeric"
        and c in df.columns
        and not m.get("is_id_like", False)
    ]

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 20:
            continue

        is_normal, skewness, kurt = _test_normality(series)

        # ── Decide method ─────────────────────────────────────────────
        is_heavy_tailed = abs(kurt) > 7
        if is_heavy_tailed:
            method = "winsorize"
        elif is_normal:
            method = "zscore"
        else:
            method = "iqr"

        # ── Count outliers before capping ─────────────────────────────
        if method == "zscore":
            z = np.abs(stats.zscore(series))
            n_outliers = int((z > _ZSCORE_THRESHOLD).sum())
        elif method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            n_outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        else:
            # winsorize flags top/bottom 1%
            n_outliers = int(0.02 * len(series))

        outlier_rate = n_outliers / max(len(series), 1)

        if outlier_rate > _OUTLIER_WARN_THRESHOLD:
            logger.warn(
                "outlier_handling",
                f"high_outlier_rate_{method}",
                column=col,
                reason=f"{outlier_rate:.1%} flagged — anomaly warning",
            )

        # ── Apply method ──────────────────────────────────────────────
        before_mean = float(df[col].mean())
        before_std = float(df[col].std())

        if method == "zscore":
            df[col] = _cap_zscore(df[col].fillna(series.mean()))
        elif method == "iqr":
            df[col] = _cap_iqr(df[col].fillna(series.median()))
        else:
            filled = df[col].fillna(series.median())
            df[col] = _winsorize_series(filled)

        after_mean = float(df[col].mean())
        after_std = float(df[col].std())

        outlier_report[col] = {
            "method": method,
            "n_outliers": n_outliers,
            "outlier_rate": round(outlier_rate, 4),
            "is_normal": is_normal,
            "skewness": round(skewness, 3),
            "kurtosis": round(kurt, 3),
            "before_mean": round(before_mean, 4),
            "before_std": round(before_std, 4),
            "after_mean": round(after_mean, 4),
            "after_std": round(after_std, 4),
        }

        logger.log(
            "outlier_handling",
            f"cap_{method}",
            column=col,
            before=f"mean={before_mean:.3f}, std={before_std:.3f}",
            after=f"mean={after_mean:.3f}, std={after_std:.3f}",
            reason=f"normal={is_normal}, skew={skewness:.2f}, kurt={kurt:.2f}",
        )

    return df, outlier_report

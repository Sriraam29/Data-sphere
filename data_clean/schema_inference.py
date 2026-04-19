"""
Schema Inference Engine
Detects true column types using conversion rates, cardinality, entropy,
string length analysis, and column-name heuristics.
"""

from __future__ import annotations

import re
import math
import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy
from logger import TransformationLogger

# ── ID heuristics ────────────────────────────────────────────────────────────
_ID_NAME_PATTERNS = re.compile(
    r"(^id$|_id$|^id_|\bidentif|\bkey$|\bindex$|\buuid|\bguid)",
    re.IGNORECASE,
)
_DATE_NAME_PATTERNS = re.compile(
    r"(date|time|timestamp|created|updated|modified|year|month|day|hour|dt$)",
    re.IGNORECASE,
)
_TARGET_NAME_PATTERNS = re.compile(
    r"(target|label|y$|outcome|output|result|class$|churn|fraud|default|survived|response)",
    re.IGNORECASE,
)

# ── Ordinal word sets ─────────────────────────────────────────────────────────
_ORDINAL_SETS = [
    {"low", "medium", "high"},
    {"low", "med", "high"},
    {"small", "medium", "large"},
    {"small", "med", "large"},
    {"very low", "low", "medium", "high", "very high"},
    {"poor", "fair", "good", "very good", "excellent"},
    {"never", "rarely", "sometimes", "often", "always"},
    {"strongly disagree", "disagree", "neutral", "agree", "strongly agree"},
    {"none", "low", "moderate", "high", "critical"},
    {"bronze", "silver", "gold", "platinum"},
    {"beginner", "intermediate", "advanced", "expert"},
    {"junior", "mid", "senior", "lead", "principal"},
    {"cold", "warm", "hot"},
    {"mild", "moderate", "severe"},
]
_ORDINAL_MAPS: list[dict[str, int]] = []
for _s in _ORDINAL_SETS:
    _items = sorted(_s)
    _ORDINAL_MAPS.append({v: i for i, v in enumerate(_items)})


def _try_numeric(series: pd.Series) -> tuple[bool, float]:
    """Try to coerce series to numeric; return (success, success_rate)."""
    coerced = pd.to_numeric(series.dropna().astype(str).str.replace(",", ""), errors="coerce")
    rate = coerced.notna().mean()
    return rate >= 0.80, float(rate)


def _try_datetime(series: pd.Series) -> tuple[bool, float]:
    """Try to parse series as datetime."""
    try:
        coerced = pd.to_datetime(
            series.dropna().astype(str), infer_format=True, errors="coerce"
        )
        rate = coerced.notna().mean()
        return rate >= 0.80, float(rate)
    except Exception:
        return False, 0.0


def _column_entropy(series: pd.Series) -> float:
    """Normalised Shannon entropy [0, 1]."""
    counts = series.value_counts(normalize=True, dropna=True)
    if len(counts) <= 1:
        return 0.0
    raw = float(sp_entropy(counts.values))
    max_ent = math.log(len(counts))
    return raw / max_ent if max_ent > 0 else 0.0


def _detect_ordinal(series: pd.Series) -> tuple[bool, dict[str, int] | None]:
    """Check if categorical values map to a known ordinal scale."""
    vals = set(series.dropna().astype(str).str.lower().str.strip().unique())
    for mapping in _ORDINAL_MAPS:
        if vals.issubset(set(mapping.keys())):
            return True, mapping
    return False, None


def _is_boolean(series: pd.Series) -> bool:
    vals = set(series.dropna().astype(str).str.lower().str.strip().unique())
    bool_sets = [
        {"true", "false"},
        {"yes", "no"},
        {"y", "n"},
        {"0", "1"},
        {"t", "f"},
        {"on", "off"},
    ]
    return any(vals.issubset(bs) for bs in bool_sets)


def infer_schema(df: pd.DataFrame, logger: TransformationLogger) -> dict:
    """
    Returns schema dict:
    {
      col: {
        "inferred_type": str,
        "original_dtype": str,
        "cardinality": int,
        "cardinality_ratio": float,
        "missing_rate": float,
        "entropy": float,
        "is_id_like": bool,
        "is_leakage_risk": bool,
        "is_potential_target": bool,
        "ordinal_map": dict | None,
        "numeric_conversion_rate": float,
      }
    }
    """
    n = len(df)
    schema: dict[str, dict] = {}

    for col in df.columns:
        s = df[col]
        orig_dtype = str(s.dtype)
        missing_rate = float(s.isna().mean())
        cardinality = int(s.nunique(dropna=True))
        cardinality_ratio = cardinality / max(n, 1)

        # ── Step 1: start with pandas dtype ──────────────────────────
        if pd.api.types.is_bool_dtype(s):
            inferred = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            inferred = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            # might still be binary / ID
            inferred = "numeric"
        else:
            inferred = "categorical"  # default; refine below

        # ── Step 2: attempt numeric coercion for object cols ─────────
        num_conv_rate = 0.0
        if inferred == "categorical":
            is_num, num_conv_rate = _try_numeric(s)
            if is_num:
                inferred = "numeric"

        # ── Step 3: attempt datetime for object cols ──────────────────
        if inferred == "categorical" and _DATE_NAME_PATTERNS.search(col):
            is_dt, _ = _try_datetime(s)
            if is_dt:
                inferred = "datetime"

        # ── Step 4: boolean check ─────────────────────────────────────
        if inferred == "categorical" and _is_boolean(s):
            inferred = "boolean"

        # ── Step 5: ordinal detection ─────────────────────────────────
        ordinal_map = None
        is_ordinal = False
        if inferred == "categorical":
            is_ordinal, ordinal_map = _detect_ordinal(s)
            if is_ordinal:
                inferred = "ordinal"

        # ── Step 6: text vs categorical ───────────────────────────────
        if inferred == "categorical":
            avg_len = float(
                s.dropna().astype(str).str.len().mean() if s.notna().any() else 0
            )
            if avg_len > 50 and cardinality_ratio > 0.5:
                inferred = "text"

        # ── Step 7: numeric binary → boolean ─────────────────────────
        if inferred == "numeric":
            unique_vals = set(s.dropna().unique())
            if unique_vals.issubset({0, 1, 0.0, 1.0}):
                inferred = "boolean"

        # ── Entropy ──────────────────────────────────────────────────
        ent = _column_entropy(s.astype(str)) if inferred not in ("numeric",) else 0.0

        # ── ID-like detection ─────────────────────────────────────────
        is_id_like = bool(_ID_NAME_PATTERNS.search(col)) or (
            cardinality_ratio > 0.95 and inferred in ("categorical", "numeric", "text")
        )

        # ── Potential target ──────────────────────────────────────────
        is_target = bool(_TARGET_NAME_PATTERNS.search(col))

        # ── Leakage risk heuristic ────────────────────────────────────
        _LEAK_PATTERNS = re.compile(
            r"(future|post|after|result|outcome|label|target|score_final|leak)",
            re.IGNORECASE,
        )
        is_leakage_risk = bool(_LEAK_PATTERNS.search(col)) and not is_target

        schema[col] = {
            "inferred_type": inferred,
            "original_dtype": orig_dtype,
            "cardinality": cardinality,
            "cardinality_ratio": cardinality_ratio,
            "missing_rate": missing_rate,
            "entropy": round(ent, 4),
            "is_id_like": is_id_like,
            "is_leakage_risk": is_leakage_risk,
            "is_potential_target": is_target,
            "ordinal_map": ordinal_map,
            "numeric_conversion_rate": round(num_conv_rate, 4),
            "avg_str_len": round(
                float(s.dropna().astype(str).str.len().mean()) if s.notna().any() else 0.0, 2
            ),
        }

        logger.log(
            "schema_inference",
            f"Inferred type: {inferred}",
            column=col,
            reason=(
                f"cardinality_ratio={cardinality_ratio:.3f}, "
                f"missing={missing_rate:.2%}, "
                f"dtype={orig_dtype}"
            ),
        )

    return schema

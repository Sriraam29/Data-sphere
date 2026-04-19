"""
High-Intelligence Encoding Strategy
- Low cardinality → OneHotEncoder
- Medium → Frequency encoding
- High + target available → Target encoding (k-fold safe)
- Ordinal → mapped integer
- Boolean → 0/1
- Leakage prevention via k-fold simulation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from logger import TransformationLogger

_LOW_CARD_MAX = 6
_HIGH_CARD_MIN = 20


def encode_features(
    df: pd.DataFrame,
    schema: dict,
    target_col: str | None,
    logger: TransformationLogger,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns (encoded_df, encoding_map).
    encoding_map stores enough info to replicate on new data.
    """
    df = df.copy()
    encoding_map: dict[str, dict] = {}

    # Gather target values for target encoding
    target_series: pd.Series | None = None
    if target_col and target_col in df.columns:
        if pd.api.types.is_numeric_dtype(df[target_col]):
            target_series = df[target_col].copy()

    for col in list(df.columns):
        if col not in schema or col == target_col:
            continue

        col_type = schema[col]["inferred_type"]
        cardinality = schema[col]["cardinality"]

        # ── Boolean → 0/1 ─────────────────────────────────────────────
        if col_type == "boolean":
            mapping = {
                "true": 1, "false": 0, "yes": 1, "no": 0,
                "y": 1, "n": 0, "1": 1, "0": 0,
                "t": 1, "f": 0, "on": 1, "off": 0,
                True: 1, False: 0, 1: 1, 0: 0,
            }
            df[col] = (
                df[col].astype(str).str.lower().str.strip().map(mapping).fillna(0).astype(np.int8)
            )
            encoding_map[col] = {"method": "boolean_map", "mapping": mapping}
            logger.log("encoding", "boolean_to_int", column=col)
            continue

        # ── Ordinal → integer ──────────────────────────────────────────
        if col_type == "ordinal" and schema[col].get("ordinal_map"):
            ord_map = schema[col]["ordinal_map"]
            df[col] = (
                df[col].astype(str).str.lower().str.strip()
                .map(ord_map)
                .fillna(-1)
                .astype(np.int8)
            )
            encoding_map[col] = {"method": "ordinal_map", "mapping": ord_map}
            logger.log("encoding", "ordinal_to_int", column=col, reason=str(ord_map))
            continue

        # ── Categorical encoding ───────────────────────────────────────
        if col_type in ("categorical", "ordinal", "text"):
            str_col = df[col].astype(str).str.strip()

            if cardinality <= _LOW_CARD_MAX:
                # OneHot
                dummies = pd.get_dummies(str_col, prefix=col, drop_first=False, dtype=np.int8)
                # drop first to avoid dummy trap
                if dummies.shape[1] > 1:
                    dummies = dummies.iloc[:, 1:]
                df = df.drop(columns=[col])
                df = pd.concat([df, dummies], axis=1)
                encoding_map[col] = {
                    "method": "onehot",
                    "categories": list(str_col.unique()),
                    "new_cols": list(dummies.columns),
                }
                logger.log(
                    "encoding",
                    "onehot",
                    column=col,
                    reason=f"cardinality={cardinality} ≤ {_LOW_CARD_MAX}",
                )

            elif cardinality >= _HIGH_CARD_MIN and target_series is not None:
                # K-fold target encoding (prevents leakage)
                n_splits = 5
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                global_mean = float(target_series.mean())
                encoded = np.full(len(df), global_mean)
                cat_values = str_col.values

                for train_idx, val_idx in kf.split(df):
                    train_cats = cat_values[train_idx]
                    train_targets = target_series.values[train_idx]
                    # compute mean target per category on train fold
                    fold_map: dict[str, float] = {}
                    for cat, tgt in zip(train_cats, train_targets):
                        if cat not in fold_map:
                            fold_map[cat] = []  # type: ignore
                        fold_map[cat].append(tgt)  # type: ignore
                    fold_map = {k: float(np.mean(v)) for k, v in fold_map.items()}
                    for i in val_idx:
                        encoded[i] = fold_map.get(cat_values[i], global_mean)

                df[col] = encoded.astype(np.float32)
                encoding_map[col] = {
                    "method": "target_encoding_kfold",
                    "global_mean": global_mean,
                }
                logger.log(
                    "encoding",
                    "target_encoding_kfold",
                    column=col,
                    reason=f"high cardinality={cardinality}, target available",
                )

            else:
                # Frequency encoding
                freq_map = str_col.value_counts(normalize=True).to_dict()
                df[col] = str_col.map(freq_map).fillna(0.0).astype(np.float32)
                encoding_map[col] = {"method": "frequency", "freq_map": freq_map}
                logger.log(
                    "encoding",
                    "frequency_encoding",
                    column=col,
                    reason=f"medium cardinality={cardinality}",
                )

    return df, encoding_map

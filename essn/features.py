from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ID_COL = "Node_ID"
TARGET_COL = "Label_Target"


def _encode_labels(y_raw: pd.Series) -> tuple[np.ndarray, list[int]]:
    y_num = pd.to_numeric(y_raw, errors="coerce")
    if y_num.isna().any():
        bad = int(y_num.isna().sum())
        raise ValueError(f"{TARGET_COL} has {bad} non-numeric/missing values")

    y_int = y_num.astype(np.int64)
    label_values = sorted(pd.unique(y_int).tolist())
    if len(label_values) != 7:
        raise ValueError(
            f"{TARGET_COL} must contain exactly 7 classes; got {len(label_values)}: {label_values[:20]}"
        )
    mapping = {v: i for i, v in enumerate(label_values)}
    y_idx = y_int.map(mapping).to_numpy(dtype=np.int64)
    return y_idx, [int(v) for v in label_values]


def _safe_log1p(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return pd.Series(
        np.log1p(np.clip(x.to_numpy(dtype=np.float64), 0.0, None)), index=x.index
    )


def build_features(
    df: pd.DataFrame, *, is_train: bool
) -> tuple[
    pd.DataFrame, np.ndarray | None, np.ndarray, list[str], list[str], list[int] | None
]:
    if ID_COL not in df.columns:
        raise ValueError(f"Missing required id column: {ID_COL}")

    y = None
    label_values: list[int] | None = None
    if is_train:
        if TARGET_COL not in df.columns:
            raise ValueError(f"Missing required target column: {TARGET_COL}")
        y, label_values = _encode_labels(df[TARGET_COL])

    id_values = df[ID_COL].astype(str).to_numpy()

    x = df.drop(columns=[c for c in [ID_COL, TARGET_COL] if c in df.columns]).copy()

    expected_cat = ["Group_A", "Group_B"]
    cat_features = [c for c in expected_cat if c in x.columns]

    if "Attr_02" in x.columns:
        angle_rad = np.deg2rad(pd.to_numeric(x["Attr_02"], errors="coerce").to_numpy())
        x["Attr_02_sin"] = np.sin(angle_rad)
        x["Attr_02_cos"] = np.cos(angle_rad)

    log_cols = [
        "Attr_04",
        "Attr_06",
        "Engineered_Dist_H",
        "Engineered_Density",
        "Engineered_Log_Dist",
    ]
    for c in log_cols:
        if c in x.columns:
            x[f"log1p_{c}"] = _safe_log1p(x[c])

    if {"Attr_01", "Attr_03"}.issubset(x.columns):
        a1 = pd.to_numeric(x["Attr_01"], errors="coerce")
        a3 = pd.to_numeric(x["Attr_03"], errors="coerce")
        x["Attr_01_x_Attr_03"] = (a1 * a3).to_numpy()
        x["Attr_03_sq"] = (a3 * a3).to_numpy()

    if {"Engineered_Flow_X", "Engineered_Momentum"}.issubset(x.columns):
        fx = pd.to_numeric(x["Engineered_Flow_X"], errors="coerce")
        mo = pd.to_numeric(x["Engineered_Momentum"], errors="coerce")
        x["FlowX_x_Momentum"] = (fx * mo).to_numpy()
        x["abs_Momentum"] = np.abs(mo.to_numpy(dtype=np.float64))

    if {"Engineered_Dist_H", "Attr_04"}.issubset(x.columns):
        dh = pd.to_numeric(x["Engineered_Dist_H"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        d4 = pd.to_numeric(x["Attr_04"], errors="coerce").to_numpy(dtype=np.float64)
        x["DistH_over_Attr04"] = dh / (np.abs(d4) + 1.0)

    for c in x.columns:
        if c in cat_features:
            continue
        x[c] = pd.to_numeric(x[c], errors="coerce").astype(np.float32)

    # Missingness indicators (often helpful for thresholdy/tabular tasks).
    for c in list(x.columns):
        if c in cat_features:
            continue
        x[f"isna_{c}"] = x[c].isna().astype(np.int8)

    for c in cat_features:
        x[c] = pd.to_numeric(x[c], errors="coerce").astype("Int64").astype("category")

    feature_names = list(x.columns)
    return x, y, id_values, feature_names, cat_features, label_values

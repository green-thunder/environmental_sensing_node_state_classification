from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def train_fold_catboost(
    *,
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_va: pd.DataFrame,
    y_va: np.ndarray,
    x_test: pd.DataFrame,
    categorical_features: list[str],
    seed: int,
    task_type: str = "GPU",
    iterations: int = 20_000,
    learning_rate: float = 0.03,
    depth: int = 8,
    l2_leaf_reg: float = 3.0,
    subsample: float = 0.8,
    colsample_bylevel: float = 0.8,
    early_stopping_rounds: int = 500,
) -> tuple[object, np.ndarray, np.ndarray]:
    from catboost import CatBoostClassifier
    from sklearn.utils.class_weight import compute_class_weight

    cat_features_idx = [int(x_tr.columns.get_loc(c)) for c in categorical_features]

    classes = np.unique(y_tr)
    class_w = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    class_weights = {int(c): float(w) for c, w in zip(classes, class_w)}

    params: dict[str, Any] = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": int(iterations),
        "learning_rate": float(learning_rate),
        "depth": int(depth),
        "l2_leaf_reg": float(l2_leaf_reg),
        "subsample": float(subsample),
        "colsample_bylevel": float(colsample_bylevel),
        "random_seed": int(seed),
        "allow_writing_files": False,
        "class_weights": class_weights,
        "od_type": "Iter",
        "od_wait": int(early_stopping_rounds),
        "verbose": 250,
        "task_type": str(task_type).upper(),
    }

    try:
        model = CatBoostClassifier(**params)
        model.fit(
            x_tr,
            y_tr,
            cat_features=cat_features_idx,
            eval_set=(x_va, y_va),
            use_best_model=True,
        )
    except Exception as e:
        if str(task_type).upper() == "GPU":
            print(
                "CatBoost GPU training failed; retrying on CPU.\n"
                f"Reason: {type(e).__name__}: {e}"
            )
            params["task_type"] = "CPU"
            model = CatBoostClassifier(**params)
            model.fit(
                x_tr,
                y_tr,
                cat_features=cat_features_idx,
                eval_set=(x_va, y_va),
                use_best_model=True,
            )
        else:
            raise

    va_proba = model.predict_proba(x_va)
    te_proba = model.predict_proba(x_test)
    return model, va_proba, te_proba

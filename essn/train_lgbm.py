from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from essn.metrics import macro_f1


def train_fold_lgbm(
    *,
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_va: pd.DataFrame,
    y_va: np.ndarray,
    x_test: pd.DataFrame,
    categorical_features: list[str],
    num_classes: int,
    seed: int,
    learning_rate: float,
    num_boost_round: int,
    early_stopping_rounds: int,
    num_leaves: int = 255,
    min_data_in_leaf: int = 80,
    feature_fraction: float = 0.8,
    bagging_fraction: float = 0.8,
    lambda_l2: float = 2.0,
    device_type: str = "cpu",
) -> tuple[object, np.ndarray, np.ndarray, int]:
    import lightgbm as lgb
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.arange(num_classes, dtype=np.int64)
    class_w = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    weight_tr = class_w[y_tr]
    weight_va = class_w[y_va]

    dtrain = lgb.Dataset(
        x_tr,
        label=y_tr,
        weight=weight_tr,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        x_va,
        label=y_va,
        weight=weight_va,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    params: dict[str, Any] = {
        "objective": "multiclass",
        "num_class": num_classes,
        "learning_rate": learning_rate,
        "num_leaves": int(num_leaves),
        "min_data_in_leaf": int(min_data_in_leaf),
        "feature_fraction": float(feature_fraction),
        "bagging_fraction": float(bagging_fraction),
        "bagging_freq": 1,
        "lambda_l2": float(lambda_l2),
        "lambda_l1": 0.0,
        "max_depth": -1,
        "max_bin": 255,
        "min_gain_to_split": 0.0,
        "extra_trees": True,
        "verbosity": -1,
        "seed": seed,
        "force_col_wise": True,
    }
    if str(device_type).lower() in {"gpu", "cuda"}:
        params["device_type"] = str(device_type).lower()
        params["gpu_use_dp"] = False

    def feval_macro_f1(y_pred: np.ndarray, dataset: lgb.Dataset):
        y_true = dataset.get_label().astype(np.int64)
        y_pred = y_pred.reshape(num_classes, -1).T
        y_hat = y_pred.argmax(axis=1)
        return "macro_f1", macro_f1(y_true, y_hat, num_classes=num_classes), True

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=250),
    ]

    try:
        booster = lgb.train(
            params=params,
            train_set=dtrain,
            valid_sets=[dvalid],
            num_boost_round=num_boost_round,
            feval=feval_macro_f1,
            callbacks=callbacks,
        )
    except Exception as e:
        if str(device_type).lower() in {"gpu", "cuda"}:
            print(
                f"LightGBM {device_type!r} training failed; retrying on CPU.\n"
                f"Reason: {type(e).__name__}: {e}"
            )
            params.pop("device_type", None)
            params.pop("gpu_use_dp", None)
            booster = lgb.train(
                params=params,
                train_set=dtrain,
                valid_sets=[dvalid],
                num_boost_round=num_boost_round,
                feval=feval_macro_f1,
                callbacks=callbacks,
            )
        else:
            raise

    best_iter = int(booster.best_iteration or num_boost_round)
    va_proba = booster.predict(x_va, num_iteration=best_iter)
    te_proba = booster.predict(x_test, num_iteration=best_iter)
    return booster, va_proba, te_proba, best_iter

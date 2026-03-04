from __future__ import annotations

import numpy as np
import pandas as pd


def train_fold_hgb(
    *,
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_va: pd.DataFrame,
    y_va: np.ndarray,
    x_test: pd.DataFrame,
    categorical_features: list[str],
    seed: int,
    learning_rate: float = 0.05,
    max_iter: int = 2500,
    max_leaf_nodes: int = 127,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1e-4,
) -> tuple[object, np.ndarray, np.ndarray]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.utils.class_weight import compute_class_weight

    cat_mask = np.array([c in set(categorical_features) for c in x_tr.columns], dtype=bool)

    classes = np.unique(y_tr)
    class_w = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    w_map = {int(c): float(w) for c, w in zip(classes, class_w)}
    w_tr = np.asarray([w_map[int(c)] for c in y_tr], dtype=np.float32)

    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=None,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        max_bins=255,
        categorical_features=cat_mask,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=80,
        max_iter=max_iter,
        random_state=seed,
    )

    model.fit(x_tr, y_tr, sample_weight=w_tr)
    va_proba = model.predict_proba(x_va)
    te_proba = model.predict_proba(x_test)
    return model, va_proba, te_proba

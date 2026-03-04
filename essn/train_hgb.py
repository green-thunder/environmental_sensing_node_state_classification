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
) -> tuple[object, np.ndarray, np.ndarray]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    cat_mask = np.array([c in set(categorical_features) for c in x_tr.columns], dtype=bool)

    model = HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_depth=None,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1e-3,
        max_bins=255,
        categorical_features=cat_mask,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=seed,
    )

    model.fit(x_tr, y_tr)
    va_proba = model.predict_proba(x_va)
    te_proba = model.predict_proba(x_test)
    return model, va_proba, te_proba


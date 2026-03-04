from __future__ import annotations

import numpy as np


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int) -> float:
    from sklearn.metrics import f1_score

    labels = list(range(num_classes))
    return float(f1_score(y_true, y_pred, labels=labels, average="macro"))


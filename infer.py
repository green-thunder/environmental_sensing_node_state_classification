from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from essn.features import build_features
from essn.serialization import load_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=Path("data"))
    p.add_argument("--artifacts_dir", type=Path, default=Path("artifacts"))
    p.add_argument("--out", type=Path, default=Path("submission.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta = load_json(args.artifacts_dir / "meta.json")

    test_path = args.data_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}")

    test_df = pd.read_csv(test_path)
    x_test, _, id_test, _, _, _ = build_features(test_df, is_train=False)

    num_classes = int(meta["num_classes"])
    n_splits = int(meta["n_splits"])
    label_values = meta.get("label_values")
    if not isinstance(label_values, list) or len(label_values) != num_classes:
        raise ValueError("meta.json missing/invalid label_values")

    proba = np.zeros((x_test.shape[0], num_classes), dtype=np.float64)
    if meta["model"] == "lgbm":
        import lightgbm as lgb

        for fold in range(n_splits):
            booster = lgb.Booster(model_file=str(args.artifacts_dir / f"lgbm_fold{fold}.txt"))
            proba += booster.predict(x_test) / n_splits
    else:
        from joblib import load

        for fold in range(n_splits):
            model = load(args.artifacts_dir / f"hgb_fold{fold}.joblib")
            proba += model.predict_proba(x_test) / n_splits

    pred = np.take(np.asarray(label_values, dtype=np.int64), proba.argmax(axis=1))
    submission = pd.DataFrame(
        {"Node_ID": id_test, "Label_Target_Predicted": pred.astype(np.int64)}
    )
    submission.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

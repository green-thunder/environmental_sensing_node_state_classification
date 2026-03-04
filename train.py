from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from essn.cv import make_splitter
from essn.features import build_features
from essn.metrics import macro_f1
from essn.serialization import save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=Path("data"))
    p.add_argument("--artifacts_dir", type=Path, default=Path("artifacts"))
    p.add_argument("--model", choices=["lgbm", "hgb"], default="lgbm")
    p.add_argument("--cv", choices=["stratified", "group_ab"], default="stratified")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_boost_round", type=int, default=50_000)
    p.add_argument("--early_stopping_rounds", type=int, default=500)
    p.add_argument("--learning_rate", type=float, default=0.03)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir: Path = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    x_train, y_train, id_train, feature_names, cat_features, label_values = (
        build_features(train_df, is_train=True)
    )
    x_test, _, id_test, _, _, _ = build_features(test_df, is_train=False)

    if label_values is None:
        raise RuntimeError("label_values missing for training set")
    if label_values == list(range(7)):
        print(
            "Note: train labels appear to be 0..6; submissions will follow the same label IDs."
        )

    chosen_model = args.model
    if chosen_model == "lgbm":
        try:
            import lightgbm  # noqa: F401
        except Exception as e:  # pragma: no cover
            chosen_model = "hgb"
            print(
                "LightGBM import failed; falling back to sklearn HistGradientBoostingClassifier.\n"
                f"Reason: {type(e).__name__}: {e}\n"
                "Fix on macOS (one option): install OpenMP runtime `libomp`.\n"
                "- Homebrew: `brew install libomp`\n"
                "- Conda: `conda install -c conda-forge libomp`\n"
                "Then reinstall/repair lightgbm if needed."
            )

    splitter = make_splitter(
        cv=args.cv,
        n_splits=args.n_splits,
        seed=args.seed,
        groups_df=train_df if args.cv == "group_ab" else None,
    )

    num_classes = int(len(label_values))
    oof_pred = np.full_like(y_train, fill_value=-1)
    test_proba = np.zeros((x_test.shape[0], num_classes), dtype=np.float64)

    meta = {
        "model": chosen_model,
        "cv": args.cv,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "num_classes": num_classes,
        "label_values": label_values,
        "feature_names": feature_names,
        "categorical_features": cat_features,
    }
    save_json(artifacts_dir / "meta.json", meta)

    fold_scores: list[float] = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x_train, y_train)):
        x_tr, y_tr = x_train.iloc[tr_idx], y_train[tr_idx]
        x_va, y_va = x_train.iloc[va_idx], y_train[va_idx]

        if chosen_model == "lgbm":
            from essn.train_lgbm import train_fold_lgbm

            booster, va_proba, te_proba, best_iter = train_fold_lgbm(
                x_tr=x_tr,
                y_tr=y_tr,
                x_va=x_va,
                y_va=y_va,
                x_test=x_test,
                categorical_features=cat_features,
                num_classes=num_classes,
                seed=args.seed + fold,
                learning_rate=args.learning_rate,
                num_boost_round=args.num_boost_round,
                early_stopping_rounds=args.early_stopping_rounds,
            )
            booster.save_model(str(artifacts_dir / f"lgbm_fold{fold}.txt"))
            fold_meta = {"fold": fold, "best_iteration": int(best_iter)}
            save_json(artifacts_dir / f"lgbm_fold{fold}.json", fold_meta)
        else:
            from essn.train_hgb import train_fold_hgb

            model, va_proba, te_proba = train_fold_hgb(
                x_tr=x_tr,
                y_tr=y_tr,
                x_va=x_va,
                y_va=y_va,
                x_test=x_test,
                categorical_features=cat_features,
                seed=args.seed + fold,
            )
            from joblib import dump

            dump(model, artifacts_dir / f"hgb_fold{fold}.joblib")

        oof_pred[va_idx] = va_proba.argmax(axis=1)
        test_proba += te_proba / args.n_splits

        fold_f1 = macro_f1(y_va, va_proba.argmax(axis=1), num_classes=num_classes)
        fold_scores.append(float(fold_f1))
        print(f"[fold {fold}] macro_f1={fold_f1:.6f}")

    oof_score = macro_f1(y_train, oof_pred, num_classes=num_classes)
    print(f"OOF macro_f1={oof_score:.6f} (fold mean={np.mean(fold_scores):.6f})")

    oof_out = pd.DataFrame(
        {
            "Node_ID": id_train,
            "Label_Target_true": np.take(label_values, y_train),
            "Label_Target_pred": np.take(label_values, oof_pred),
        }
    )
    oof_out.to_csv(artifacts_dir / "oof.csv", index=False)

    test_pred = np.take(label_values, test_proba.argmax(axis=1))
    submission = pd.DataFrame(
        {"Node_ID": id_test, "Label_Target_Predicted": test_pred.astype(np.int64)}
    )
    submission.to_csv(artifacts_dir / "submission.csv", index=False)
    print(f"Wrote {artifacts_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()

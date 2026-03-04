# Environmental Sensing Node State Classification

This repo is designed to be run from a single **Google Colab GPU notebook**: `notebooks/gpu_finetune_catboost_optuna.ipynb`.

The notebook is **self-contained** (no `essn.*` imports) and performs:
1) feature engineering
2) Optuna hyperparameter tuning (CatBoost)
3) 5-fold stratified CV training (CatBoost GPU)
4) `submission.csv` export

## Setup

Put the competition files here:

- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv` (optional)

## Colab GPU (recommended)

Open and run: `notebooks/gpu_finetune_catboost_optuna.ipynb`

In Colab:
- Runtime → Change runtime type → **GPU**
- Upload your `data/` folder into the workspace so the notebook can read `data/train.csv` and `data/test.csv`
- Run all cells top-to-bottom

## Methods used

- **Features** (implemented inside the notebook)
  - categorical handling for `Group_A` / `Group_B`
  - trig transform for `Attr_02` (sin/cos)
  - `log1p` transforms for several heavy-tailed columns
  - interaction features + missingness indicators
- **Validation**
  - stratified K-fold CV
- **Model**
  - CatBoost multiclass with `task_type=GPU` (falls back to CPU if no CUDA GPU is available)
- **Metric**
  - macro F1 on CV folds + OOF
- **Tuning**
  - Optuna (TPE) tunes key CatBoost hyperparameters on 3-fold CV, then retrains 5-fold CV with best params

## Results

Results are visible directly in the notebook output cells after you run it (Optuna best params, per-fold macro-F1 prints, and final OOF macro-F1 summary). They are also saved to `artifacts/meta.json`.

After the final training cell finishes, outputs are also written to disk:
- `artifacts/meta.json` → includes `oof_macro_f1`, per-fold `fold_macro_f1`, and the selected `best_params`
- `artifacts/submission.csv` → ready to upload

To re-print the stored score in Colab (optional):

```bash
python3 -c "import json; print(json.load(open('artifacts/meta.json'))['oof_macro_f1'])"
```

### Example run (March 4, 2026)

From a Colab GPU run of the notebook:

- Best params (Optuna):
  - `iterations`: `18000`
  - `learning_rate`: `0.056650220734838656`
  - `depth`: `10`
  - `l2_leaf_reg`: `5.325036946404286`
  - `subsample`: `0.8682180570419546`
  - `colsample_bylevel`: `0.9703199093466347`
- CV macro-F1:
  - Fold macro-F1 examples printed in the notebook: `0.786778`, `0.781472`, `0.752230`, `0.806693`
  - Final: `OOF macro_f1=0.775843` (fold mean `0.777859`)

Your exact score may differ depending on dataset version/shuffle, GPU type, and library versions; use the notebook output cells and `artifacts/meta.json` as the source of truth for your run.

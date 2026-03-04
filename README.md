# Environmental Sensing Node State Classification

This repo contains a strong, competition-style tabular ML pipeline for the dataset described in `description.md`.

## Setup

Put the competition files here:

- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv` (optional)

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## Train (CV + submission)

LightGBM (recommended):

```bash
python3 train.py --data_dir data --model lgbm --n_splits 5 --cv stratified --seed 42
```

Fallback without LightGBM (sklearn HistGradientBoosting):

```bash
python3 train.py --data_dir data --model hgb --n_splits 5 --cv stratified --seed 42
```

Artifacts are written to `artifacts/` (fold models, OOF predictions, and `submission.csv`).

## Inference (from saved artifacts)

```bash
python3 infer.py --data_dir data --artifacts_dir artifacts --out submission.csv
```


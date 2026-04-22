# CSDS 412 PD Progression Baseline Models

This repository contains the baseline modeling stage for Parkinson's disease progression prediction using the AMP-PD Kaggle dataset. The current stage focuses on a serial, reproducible training pipeline with support for multiple baseline regressors and consistent experiment tracking.

## Current Stage

Implemented so far:

- Baseline model registry in `src/pd_progression/models.py`
- Supported models:
  - `ridge`
  - `lasso`
  - `random_forest`
  - `xgboost`
- Patient-level and time-aware validation splits
- CLI entrypoint for running baseline experiments
- Saved run artifacts:
  - `config.json`
  - `metrics.json`
  - `predictions.csv`
  - `model.pkl`
- Unit and smoke tests for model creation, CLI config loading, and pipeline execution

Not implemented yet:

- Parallel processing
- Runtime benchmarking
- Feature importance analysis
- Full hyperparameter tuning workflow

## Repository Layout

- `src/pd_progression/` - package code for loading, splitting, modeling, and tracking runs
- `scripts/` - executable wrappers for training and evaluation
- `data/cleaned/` - prepared dataset used for baseline runs
- `data/raw/` - raw AMP-PD competition files
- `results/` - saved run outputs
- `tests/` - pytest suite for the baseline stage

## Requirements

- Python 3.10+
- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`

Install the project in editable mode:

```bash
python -m pip install -e .
```

If you only want to run the tests, install the same dependencies first so `xgboost` is available when needed.

## Data

The baseline pipeline expects a prepared CSV file. The repo includes a derived dataset at:

```text
data/cleaned/final_dataset.csv
```

This file is the output of the feature engineering pipeline and contains the clinical target along with numeric features used for baseline training.

Some rows in the prepared dataset still contain missing values. Before running the baseline models, clean the dataset with:

```bash
python3 scripts/clean_nan_data.py \
  --input data/cleaned/final_dataset.csv \
  --output data/cleaned/final_dataset_no_nan.csv
```

The cleaned output removes rows where the target is missing and imputes remaining NaNs so the scikit-learn baselines can train without errors.

## Running a Baseline Experiment

Use the training wrapper script:

```bash
python scripts/train_baseline.py \
  --data data/cleaned/final_dataset_no_nan.csv \
  --model ridge \
  --target updrs_1 \
  --split patient \
  --output-dir results
```

### Supported models

- `ridge`
- `lasso`
- `random_forest` or `rf`
- `xgboost`, `xgb`, or `xg_boost`

### Validation splits

- `patient`
- `time_aware`

## Passing Hyperparameters

You can pass model-specific hyperparameters from the CLI with `--model-params` as a JSON object:

```bash
python scripts/train_baseline.py \
  --data data/cleaned/final_dataset_no_nan.csv \
  --model random_forest \
  --model-params '{"n_estimators": 200, "max_depth": 8}'
```

Examples:

- Ridge: `{"alpha": 1.0}`
- Lasso: `{"alpha": 0.05, "max_iter": 5000}`
- Random Forest: `{"n_estimators": 300, "max_depth": 12}`
- XGBoost: `{"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6}`

You can also supply a JSON config file with the same keys used by `BaselineRunConfig`.

## Config File Example

```json
{
  "model_name": "xgboost",
  "target_column": "updrs_1",
  "split_strategy": "time_aware",
  "random_seed": 42,
  "validation_fraction": 0.2,
  "data_path": "data/cleaned/final_dataset_no_nan.csv",
  "output_dir": "results",
  "model_params": {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05
  }
}
```

Run it with:

```bash
python scripts/train_baseline.py --config path/to/config.json
```

## Output Artifacts

Each training run writes to a timestamped directory under `results/runs/`. The run directory contains:

- `config.json` - resolved run configuration, including selected feature columns
- `metrics.json` - regression metrics for the validation split
- `predictions.csv` - row-level predictions for the validation fold
- `model.pkl` - serialized fitted model wrapper

## Metrics

The baseline pipeline reports:

- `rmse`
- `mae`
- `smape`
- `r2`

## Tests

Run the full test suite:

```bash
pytest tests -q
```

Individual test files can also be run directly in an IDE or with Python:

```bash
python tests/test_models.py
python tests/test_cli.py
python tests/test_pipeline.py
```

## Notes on the Current Phase

This stage is intentionally serial. The next phase will add parallel processing and runtime benchmarking once the baseline model workflow is complete and stable.

# CSDS 412 PD Progression Baseline Models

This repository contains the baseline modeling and parallel-preprocessing stages for Parkinson's disease progression prediction using the AMP-PD Kaggle dataset. The pipeline covers data cleaning, longitudinal feature construction, baseline regressors, patient-level / time-aware validation, and opt-in parallel per-patient preprocessing.

## Current Stage

Implemented so far:

- Data cleaning and validation for clinical, protein, and peptide tables
- Feature engineering pipeline with protein/peptide aggregation and per-patient longitudinal features (lag, delta, rolling mean/std, visit gap, missingness indicators)
- Baseline model registry in `src/pd_progression/models.py` with supported models:
  - `linear_regression`
  - `ridge`
  - `lasso`
  - `random_forest`
  - `xgboost`
- `StandardScaler` wrapping for scale-sensitive linear models (`linear_regression`, `ridge`, `lasso`)
- Patient-level and time-aware validation splits
- Automatic NaN handling in the training pipeline (drops rows with missing target or features, excludes all `updrs_*` columns from features to prevent target leakage)
- Opt-in parallel per-patient preprocessing via a shared `parallel_groupby_apply` helper and an `n_jobs` knob on both the feature-engineering pipeline and the baseline trainer
- CLI entrypoints for building the cleaned dataset and running baseline experiments
- Saved run artifacts:
  - `config.json`
  - `metrics.json` (with `rmse`, `mae`, `smape`, `r2`, `n_train`, `n_val`, `n_dropped_missing`)
  - `predictions.csv`
  - `model.pkl`
- Unit and smoke tests for model creation, CLI config loading, pipeline execution, patient-level splits, parallel groupby semantics, and feature-engineering parallel invariants

Not implemented yet:

- Per-stage benchmark harness and scaling study
- SLURM / HPC job scripts
- Systematic hyperparameter tuning workflow
- Feature importance analysis
- Singularity container

## Repository Layout

- `src/pd_progression/` - package code for loading, preprocessing, feature engineering, modeling, splits, tracking, and parallelism
- `scripts/` - executable wrappers for building the training dataset, training baselines, and evaluating saved predictions
- `data/cleaned/` - prepared dataset used for baseline runs
- `data/raw/` - raw AMP-PD competition files
- `results/` - saved run outputs (per-run timestamped directories)
- `tests/` - pytest suite

## Requirements

- Python 3.10+
- `numpy`
- `pandas`
- `scikit-learn` (pulls in `joblib`)
- `xgboost`

Install the project in editable mode:

```bash
python -m pip install -e .
```

`pytest` is needed only for running the test suite:

```bash
python -m pip install pytest
```

## Data

The baseline pipeline expects a prepared CSV file. The repo includes a derived dataset at:

```text
data/cleaned/final_dataset.csv
```

This file is the output of the feature engineering pipeline and contains the clinical targets along with numeric features used for baseline training.

### Rebuilding the prepared dataset

Regenerate `data/cleaned/final_dataset.csv` from the raw AMP-PD tables with:

```bash
python scripts/build_training_data.py \
  --raw-dir data/raw/amp-parkinsons-disease-progression-prediction \
  --output data/cleaned/final_dataset.csv
```

The per-patient longitudinal feature step is parallelizable via `--n-jobs`:

```bash
python scripts/build_training_data.py --n-jobs 4
```

Output is byte-identical regardless of worker count; `--n-jobs` only affects wall time.

### Missing values

The training pipeline handles missing values automatically:

- Rows where the selected target column is `NaN` are dropped before splitting.
- All `updrs_*` columns are excluded from the feature matrix so training on `updrs_1` cannot leak `updrs_2/3/4` (and avoids the `~40%` `NaN` rate on `updrs_4`).
- Linear models (`linear_regression`, `ridge`, `lasso`) are wrapped in a `StandardScaler` pipeline to prevent the ill-conditioned-matrix warnings that appear on unscaled multi-scale features.

The number of dropped rows is reported in each run's `metrics.json` as `n_dropped_missing`.

The legacy helper `scripts/clean_nan_data.py` (which writes a separately cleaned CSV) is still available but is no longer required for baseline runs.

## Running a Baseline Experiment

Use the training wrapper script:

```bash
python scripts/train_baseline.py \
  --data data/cleaned/final_dataset.csv \
  --model ridge \
  --target updrs_1 \
  --split patient \
  --output-dir results
```

### Supported models

- `linear_regression`, `linear`, or `ols`
- `ridge`
- `lasso`
- `random_forest` or `rf`
- `xgboost`, `xgb`, or `xg_boost`

### Validation splits

- `patient`
- `time_aware`

### Split behavior

- `patient` split keeps all visits from a given patient on the same side of the train/validation boundary.
- `time_aware` split keeps earlier visits for training and holds out later visits for validation within each patient.
- Both strategies use `validation_fraction` to determine how much data is reserved for validation.

## Parallel Preprocessing and Training

The package exposes a single `--n-jobs` knob that both the dataset builder and the baseline trainer accept. It controls:

- Per-patient longitudinal feature construction (`add_longitudinal_features`), via the shared `parallel_groupby_apply` helper in `src/pd_progression/parallel.py`.
- Any baseline model that supports `n_jobs` natively (currently `random_forest` and `xgboost`). Linear models are unaffected.

Example:

```bash
python scripts/build_training_data.py --n-jobs 8
python scripts/train_baseline.py --data data/cleaned/final_dataset.csv --model random_forest --n-jobs 8
```

`n_jobs = 1` preserves the original single-threaded behavior exactly; the parallel path uses a serial-equivalent fast path when `n_jobs <= 1`, so there is no joblib overhead in that case.

## Passing Hyperparameters

You can pass model-specific hyperparameters from the CLI with `--model-params` as a JSON object:

```bash
python scripts/train_baseline.py \
  --data data/cleaned/final_dataset.csv \
  --model random_forest \
  --model-params '{"n_estimators": 200, "max_depth": 8}'
```

Examples:

- Ridge: `{"alpha": 1.0}`
- Lasso: `{"alpha": 0.05, "max_iter": 5000}`
- Random Forest: `{"n_estimators": 300, "max_depth": 12}`
- XGBoost: `{"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6}`

You can also supply a JSON config file with the same keys used by `BaselineRunConfig`, including the new top-level `n_jobs` field.

## Config File Example

```json
{
  "model_name": "random_forest",
  "target_column": "updrs_1",
  "split_strategy": "time_aware",
  "random_seed": 42,
  "validation_fraction": 0.2,
  "data_path": "data/cleaned/final_dataset.csv",
  "output_dir": "results",
  "n_jobs": 4,
  "model_params": {
    "n_estimators": 400,
    "max_depth": 12
  }
}
```

Run it with:

```bash
python scripts/train_baseline.py --config path/to/config.json
```

## Output Artifacts

Each training run writes to a timestamped directory under `results/runs/`. The run directory contains:

- `config.json` - resolved run configuration, including selected feature columns and `n_jobs`
- `metrics.json` - regression metrics for the validation split, plus `n_train`, `n_val`, and `n_dropped_missing`
- `predictions.csv` - row-level predictions for the validation fold
- `model.pkl` - serialized fitted model wrapper

## Metrics

The baseline pipeline reports:

- `rmse`
- `mae`
- `smape`
- `r2`

Plus the sample-size counters:

- `n_train`
- `n_val`
- `n_dropped_missing`

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
python tests/test_splits.py
python tests/test_parallel.py
python tests/test_feature_engineering.py
```

The parallel correctness invariants (`test_parallel.py`, `test_feature_engineering.py`) verify that `n_jobs=1` and `n_jobs=N` produce byte-identical outputs, which is the guarantee the benchmark harness will rely on.

## Notes on the Current Phase

The baseline-model stage (Phase 3) is complete: all five models train, feature scaling is applied where needed, and NaN rows are dropped before fitting. The parallel-preprocessing stage (Phase 5) has landed the `n_jobs` knob, the shared `parallel_groupby_apply` helper, and the parallelized longitudinal feature step. The remaining Phase 5 work is the per-stage benchmark harness, SLURM job scripts, the strong-scaling / data-size sweep on HPC, and the plots that summarize the runtime comparison in the final report.

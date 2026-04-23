# CSDS 412 PD Progression Baseline Models

This repository contains the baseline modeling and parallel-preprocessing stages for Parkinson's disease progression prediction using the AMP-PD Kaggle dataset. The pipeline covers data cleaning, longitudinal feature construction, baseline regressors, patient-level / time-aware validation, and opt-in parallel per-patient preprocessing.

## Current Stage

Implemented so far:

- Data cleaning and validation for clinical, protein, and peptide tables
- Feature engineering pipeline with protein/peptide aggregation and per-patient longitudinal features (lag, delta, rolling mean/std, visit gap, missingness indicators)
- Supported models:
  - `linear_regression`
  - `ridge`
  - `lasso`
  - `random_forest`
  - `xgboost`
- `StandardScaler` wrapping for scale-sensitive linear models (`linear_regression`, `ridge`, `lasso`)
- Patient-level and time-aware validation splits
- Automatic NaN handling in the training pipeline (drops rows with missing target or features, excludes all `updrs_*` columns from features to prevent target leakage)
- Opt-in parallel per-patient preprocessing via a shared `parallel_groupby_apply` helper and an `n_jobs` knob on both the feature-engineering pipeline and the baseline trainer
- Benchmark harness for serial vs parallel dataset building and training runs
- Feature-importance export for trained linear, random-forest, and XGBoost models
- HPC assets:
  - pinned dependency file (`requirements-hpc.txt`)
  - Apptainer/Singularity recipe (`containers/apptainer.def`)
  - SLURM scripts for build, train, benchmark, and feature-importance jobs
- CLI entrypoints for running baseline experiments
- Saved run artifacts:
  - `config.json`
  - `metrics.json`
  - `predictions.csv`
  - `model.pkl`
- Unit and smoke tests for model creation, CLI config loading, pipeline execution, patient-level splits, parallel groupby semantics, and feature-engineering parallel invariants

Not implemented yet:

- Full hyperparameter tuning workflow
- Checked-in benchmark results / trained model artifacts
- Final report plots summarizing runtime scaling and feature importance

## Repository Layout

- `src/pd_progression/` - package code for loading, preprocessing, feature engineering, modeling, splits, tracking, and parallelism
- `scripts/` - executable wrappers for building the training dataset, training baselines, and evaluating saved predictions
- `hpc/` - SLURM batch scripts plus a virtualenv bootstrap helper
- `containers/` - Apptainer/Singularity recipe for portable cluster execution
- `data/cleaned/` - prepared dataset used for baseline runs
- `data/raw/` - raw AMP-PD competition files
- `results/` - saved run outputs
- `tests/` - pytest suite

## Requirements

- Python 3.10+
- `numpy<2`
- `pandas`
- `scipy`
- `scikit-learn`
- `joblib`
- `xgboost`

Install the project in editable mode:

```bash
python -m pip install -e .
```

`pytest` is needed only for running the test suite:

```bash
python -m pip install pytest
```

For a fresh Linux or HPC environment, use the pinned dependency file instead:

```bash
python -m pip install -r requirements-hpc.txt
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


### Missing values

The training pipeline handles missing values automatically:

- Rows where the selected target column is `NaN` are dropped before splitting.
- All `updrs_*` columns are excluded from the feature matrix so training on `updrs_1` cannot leak `updrs_2/3/4`.
- Linear models (`linear_regression`, `ridge`, `lasso`) are wrapped in a `StandardScaler` pipeline to prevent the warnings that appear on unscaled multi-scale features.

The number of dropped rows is reported in each run's `metrics.json` as `n_dropped_missing`.

The helper `scripts/clean_nan_data.py` is still available but is no longer required for baseline runs.

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

- Per-patient longitudinal feature construction (`add_longitudinal_features`).
- Any baseline model that supports `n_jobs` natively (currently `random_forest` and `xgboost`).

Example:

```bash
python scripts/build_training_data.py --n-jobs 8
python scripts/train_baseline.py --data data/cleaned/final_dataset.csv --model random_forest --n-jobs 8
```

`n_jobs = 1` preserves the original single-threaded behavior exactly. The parallel path uses a serial-equivalent fast path when `n_jobs <= 1`, so there is no joblib overhead in that case.

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

You can also supply a JSON config file with the same keys used by `BaselineRunConfig`.

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

- `config.json` - resolved run configuration, including selected feature columns
- `metrics.json` - regression metrics for the validation split
- `predictions.csv` - row-level predictions for the validation fold
- `model.pkl` - serialized fitted model wrapper

Feature-importance export adds:

- `feature_importance.csv` - ranked features extracted from `coef_` or `feature_importances_`

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

## Feature Importance

Export feature importances from a completed run:

```bash
python scripts/export_feature_importance.py \
  --run-dir results/runs/<timestamp_model_split_seed> \
  --top-k 20
```

The script reads `config.json` to recover the resolved feature order, loads `model.pkl`, and writes `feature_importance.csv` into the same run directory by default.

For linear models, the ranking uses absolute coefficient magnitude. For `random_forest` and `xgboost`, it uses native `feature_importances_`.

## Benchmarking

Benchmark dataset construction and model training across multiple worker counts:

```bash
python scripts/benchmark_pipeline.py \
  --raw-dir data/raw/amp-parkinsons-disease-progression-prediction \
  --data data/cleaned/final_dataset.csv \
  --models random_forest,xgboost \
  --targets updrs_1,updrs_2 \
  --n-jobs-values 1,2,4,8 \
  --repeat 3 \
  --output results/benchmarks/benchmark_results.csv
```

The benchmark output is a flat CSV with one row per stage / worker-count / repeat. Build-stage rows cover the end-to-end raw-table preprocessing flow. Training-stage rows include runtime plus validation metrics for each completed run.

## HPC Usage

Create a cluster-friendly virtual environment from the pinned dependency set:

```bash
bash hpc/setup_venv.sh
```

Submit common jobs with SLURM:

```bash
sbatch hpc/build_dataset.sbatch
sbatch hpc/train_baseline.sbatch
sbatch hpc/benchmark_pipeline.sbatch
sbatch hpc/export_feature_importance.sbatch
```

Each SLURM script accepts environment-variable overrides. Examples:

```bash
MODEL=xgboost TARGET=updrs_2 N_JOBS=8 sbatch hpc/train_baseline.sbatch
N_JOBS_VALUES=1,2,4,8 MODELS=random_forest,xgboost sbatch hpc/benchmark_pipeline.sbatch
RUN_DIR=$PWD/results/runs/<run_name> sbatch hpc/export_feature_importance.sbatch
```

For containerized execution, build an Apptainer image from `containers/apptainer.def` and run the same scripts inside the container.

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
python tests/test_metrics.py
python tests/test_interpretability.py
```

## Notes on the Current Phase

The baseline-model stage (Phase 3) is complete: all five models train, feature scaling is applied where needed, and NaN rows are dropped before fitting. The parallel-preprocessing / operations stage now includes the `n_jobs` knob, the shared `parallel_groupby_apply` helper, the benchmark harness, feature-importance export, and SLURM / Apptainer assets for HPC execution. The remaining work is to run the actual experiments, collect benchmark outputs, perform systematic tuning, and turn those outputs into final report plots and analysis.

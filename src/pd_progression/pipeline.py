from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import BaselineRunConfig
from .metrics import compute_regression_metrics, metrics_table
from .models import build_model
from .splits import make_patient_level_split, make_time_aware_split
from .tracking import ExperimentTracker

IDENTIFIER_COLUMNS = {"patient_id", "visit_id", "visit_month"}


CURRENT_TARGET_PREFIXES: tuple[str, ...] = ("updrs_",)
FUTURE_TARGET_PREFIXES: tuple[str, ...] = ("future_updrs_",)
ENGINEERED_CLINICAL_FEATURE_PREFIXES: tuple[str, ...] = ("current_updrs_", "prior_updrs_")


@dataclass
class TrainingResult:
    metrics: dict[str, float]
    predictions: pd.DataFrame
    feature_columns: list[str]
    run_dir: Path


def _is_current_target_like(column: str) -> bool:
    return any(column.startswith(prefix) for prefix in CURRENT_TARGET_PREFIXES)


def _is_future_target_like(column: str) -> bool:
    return any(column.startswith(prefix) for prefix in FUTURE_TARGET_PREFIXES)


def _is_engineered_clinical_feature(column: str) -> bool:
    return any(column.startswith(prefix) for prefix in ENGINEERED_CLINICAL_FEATURE_PREFIXES)


def _should_exclude_feature_column(column: str, config: BaselineRunConfig) -> bool:
    if column in set(config.id_columns) | {config.target_column}:
        return True

    # Never allow next-visit targets into the feature matrix.
    if _is_future_target_like(column):
        return True

    # Raw same-visit UPDRS labels should never enter the feature matrix.
    if _is_current_target_like(column):
        return True

    # The engineered current/prior clinical features are only valid for the
    # future-forecasting setup, not for the original same-visit baselines.
    if _is_engineered_clinical_feature(column):
        return not config.target_column.startswith("future_updrs_")

    return False


def select_feature_columns(df: pd.DataFrame, config: BaselineRunConfig) -> list[str]:
    if config.feature_columns:
        return [column for column in config.feature_columns if column in df.columns]

    feature_columns = []
    for column in df.columns:
        if _should_exclude_feature_column(column, config):
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            feature_columns.append(column)
    return feature_columns


def prepare_xy(df: pd.DataFrame, config: BaselineRunConfig) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_columns = select_feature_columns(df, config)
    if not feature_columns:
        raise ValueError("No usable feature columns found")
    if config.target_column not in df.columns:
        raise ValueError(f"Target column not found: {config.target_column}")
    return df[feature_columns], df[config.target_column], feature_columns


def _drop_missing_rows(df: pd.DataFrame, config: BaselineRunConfig) -> pd.DataFrame:
    if config.target_column not in df.columns:
        raise ValueError(f"Target column not found: {config.target_column}")
    feature_columns = select_feature_columns(df, config)
    subset = [config.target_column] + feature_columns
    cleaned = df.dropna(subset=subset).reset_index(drop=True)
    return cleaned


def _split_dataframe(df: pd.DataFrame, config: BaselineRunConfig):
    if config.split_strategy == "patient":
        return make_patient_level_split(
            df,
            group_column=config.group_column,
            validation_fraction=config.validation_fraction,
            random_seed=config.random_seed,
        )
    if config.split_strategy == "time_aware":
        return make_time_aware_split(
            df,
            group_column=config.group_column,
            time_column=config.time_column,
            validation_fraction=config.validation_fraction,
            random_seed=config.random_seed,
        )
    raise ValueError(f"Unknown split strategy: {config.split_strategy}")


def run_training_pipeline(df: pd.DataFrame, config: BaselineRunConfig) -> TrainingResult:
    original_rows = len(df)
    df = _drop_missing_rows(df, config)
    dropped_rows = original_rows - len(df)
    if df.empty:
        raise ValueError(
            f"No rows remain after dropping NaN target/features for '{config.target_column}'"
        )

    split = _split_dataframe(df, config)
    X, y, feature_columns = prepare_xy(df, config)
    model_params = dict(config.model_params)
    effective_n_jobs = int(model_params.pop("n_jobs", config.n_jobs))
    model = build_model(
        config.model_name,
        random_seed=config.random_seed,
        n_jobs=effective_n_jobs,
        **model_params,
    )
    resolved_config = replace(config, feature_columns=tuple(feature_columns))

    X_train = X.loc[list(split.train_index)]
    y_train = y.loc[list(split.train_index)]
    X_val = X.loc[list(split.validation_index)]
    y_val = y.loc[list(split.validation_index)]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    metrics = compute_regression_metrics(y_val, y_pred)
    metrics["n_train"] = int(len(X_train))
    metrics["n_val"] = int(len(X_val))
    metrics["n_dropped_missing"] = int(dropped_rows)
    prediction_frame = pd.DataFrame(
        {
            "row_index": list(split.validation_index),
            "y_true": y_val.values,
            "y_pred": y_pred,
        }
    )

    tracker = ExperimentTracker.create(
        config.output_dir,
        model_name=config.model_name,
        target_column=config.target_column,
        split_strategy=config.split_strategy,
        random_seed=config.random_seed,
    )
    tracker.ensure()
    tracker.save_config(resolved_config.to_dict())
    tracker.save_metrics(metrics)
    tracker.save_predictions(prediction_frame)
    tracker.save_model(model)
    return TrainingResult(metrics=metrics, predictions=prediction_frame, feature_columns=feature_columns, run_dir=tracker.run_dir)


def summarize_runs(results: dict[str, TrainingResult]) -> pd.DataFrame:
    return metrics_table({name: result.metrics for name, result in results.items()})

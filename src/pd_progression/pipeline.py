from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class TrainingResult:
    metrics: dict[str, float]
    predictions: pd.DataFrame
    feature_columns: list[str]
    run_dir: Path


def select_feature_columns(df: pd.DataFrame, config: BaselineRunConfig) -> list[str]:
    if config.feature_columns:
        return [column for column in config.feature_columns if column in df.columns]

    excluded = set(config.id_columns) | {config.target_column}
    feature_columns = []
    for column in df.columns:
        if column in excluded:
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
    split = _split_dataframe(df, config)
    X, y, feature_columns = prepare_xy(df, config)
    model = build_model(config.model_name, random_seed=config.random_seed, **config.model_params)

    X_train = X.loc[list(split.train_index)]
    y_train = y.loc[list(split.train_index)]
    X_val = X.loc[list(split.validation_index)]
    y_val = y.loc[list(split.validation_index)]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    metrics = compute_regression_metrics(y_val, y_pred)
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
        split_strategy=config.split_strategy,
        random_seed=config.random_seed,
    )
    tracker.ensure()
    tracker.save_config(config.to_dict())
    tracker.save_metrics(metrics)
    tracker.save_predictions(prediction_frame)
    tracker.save_model(model)
    return TrainingResult(metrics=metrics, predictions=prediction_frame, feature_columns=feature_columns, run_dir=tracker.run_dir)


def summarize_runs(results: dict[str, TrainingResult]) -> pd.DataFrame:
    return metrics_table({name: result.metrics for name, result in results.items()})


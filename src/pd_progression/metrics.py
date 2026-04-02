from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def smape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    denominator = np.abs(y_true_arr) + np.abs(y_pred_arr)
    numerator = np.abs(y_pred_arr - y_true_arr)
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(2.0 * numerator[mask] / denominator[mask]))


def r2(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(r2_score(y_true, y_pred))


def compute_regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def metrics_table(metrics_by_run: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for run_name, metrics in metrics_by_run.items():
        row = {"run_name": run_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


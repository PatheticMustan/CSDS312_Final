from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("Metric inputs must contain at least one value")
    return arr


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = _as_float_array(y_true)
    y_pred_arr = _as_float_array(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(math.sqrt(np.mean(np.square(y_true_arr - y_pred_arr))))


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = _as_float_array(y_true)
    y_pred_arr = _as_float_array(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def smape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = _as_float_array(y_true)
    y_pred_arr = _as_float_array(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    denominator = np.abs(y_true_arr) + np.abs(y_pred_arr)
    numerator = np.abs(y_pred_arr - y_true_arr)
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(2.0 * numerator[mask] / denominator[mask]))


def r2(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = _as_float_array(y_true)
    y_pred_arr = _as_float_array(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true_arr.size < 2:
        return 0.0

    residual_sum = float(np.sum(np.square(y_true_arr - y_pred_arr)))
    centered = y_true_arr - float(np.mean(y_true_arr))
    total_sum = float(np.sum(np.square(centered)))
    if total_sum == 0.0:
        return 1.0 if residual_sum == 0.0 else 0.0
    return float(1.0 - residual_sum / total_sum)


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

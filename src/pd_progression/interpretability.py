from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def _resolve_core_estimator(model: Any) -> Any:
    estimator = getattr(model, "estimator", model)
    named_steps = getattr(estimator, "named_steps", None)
    if named_steps and "model" in named_steps:
        return named_steps["model"]
    return estimator


def extract_feature_importance(
    model: Any,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    feature_names = list(feature_columns)
    estimator = _resolve_core_estimator(model)

    if hasattr(estimator, "feature_importances_"):
        raw_values = np.asarray(estimator.feature_importances_, dtype=float)
        importance = raw_values.copy()
        method = "feature_importances_"
    elif hasattr(estimator, "coef_"):
        raw_values = np.asarray(estimator.coef_, dtype=float)
        if raw_values.ndim > 1:
            raw_values = np.mean(raw_values, axis=0)
        importance = np.abs(raw_values)
        method = "abs_coef"
    else:
        raise ValueError(
            "Model does not expose feature importance via feature_importances_ or coef_."
        )

    if raw_values.shape[0] != len(feature_names):
        raise ValueError(
            "Feature importance length does not match the configured feature columns."
        )

    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
            "raw_value": raw_values,
            "method": method,
        }
    )
    frame = frame.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1, dtype=int)
    return frame[["rank", "feature", "importance", "raw_value", "method"]]

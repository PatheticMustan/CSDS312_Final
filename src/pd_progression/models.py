from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge


def _default_random_state(params: dict[str, Any], random_seed: int) -> dict[str, Any]:
    if "random_state" in params:
        return params
    updated = dict(params)
    updated["random_state"] = random_seed
    return updated


@dataclass
class SklearnRegressorWrapper:
    estimator_name: str
    estimator: Any
    params: dict[str, Any] = field(default_factory=dict)

    def fit(self, X, y) -> "SklearnRegressorWrapper":
        self.estimator.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.estimator.predict(X)


def build_model(model_name: str, random_seed: int = 42, **params: Any) -> SklearnRegressorWrapper:
    model_name = model_name.lower()
    if model_name in {"linear_regression", "linear", "ols"}:
        estimator = LinearRegression(**params)
        return SklearnRegressorWrapper("linear_regression", estimator, params)
    if model_name == "ridge":
        estimator = Ridge(**params)
        return SklearnRegressorWrapper("ridge", estimator, params)
    if model_name == "lasso":
        estimator = Lasso(**params)
        return SklearnRegressorWrapper("lasso", estimator, params)
    if model_name in {"random_forest", "rf"}:
        merged = _default_random_state(params, random_seed)
        estimator = RandomForestRegressor(**merged)
        return SklearnRegressorWrapper("random_forest", estimator, merged)
    raise ValueError(f"Unsupported model name: {model_name}")


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge


MODEL_REGISTRY: dict[str, tuple[str, Any, bool]] = {
    "linear_regression": ("linear_regression", LinearRegression, False),
    "linear": ("linear_regression", LinearRegression, False),
    "ols": ("linear_regression", LinearRegression, False),
    "ridge": ("ridge", Ridge, False),
    "lasso": ("lasso", Lasso, False),
    "random_forest": ("random_forest", RandomForestRegressor, True),
    "rf": ("random_forest", RandomForestRegressor, True),
}


def _normalize_model_name(model_name: str) -> str:
    return model_name.strip().lower().replace("-", "_").replace(" ", "_")


def _with_random_state(params: dict[str, Any], random_seed: int) -> dict[str, Any]:
    if "random_state" in params:
        return dict(params)
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
    normalized_name = _normalize_model_name(model_name)

    if normalized_name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(set(MODEL_REGISTRY)))
        raise ValueError(f"Unsupported model name: {normalized_name}. Supported models: {supported}")

    estimator_name, estimator_cls, needs_random_state = MODEL_REGISTRY[normalized_name]
    fitted_params = _with_random_state(params, random_seed) if needs_random_state else dict(params)
    estimator = estimator_cls(**fitted_params)
    return SklearnRegressorWrapper(estimator_name, estimator, fitted_params)

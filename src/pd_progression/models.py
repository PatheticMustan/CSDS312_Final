from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


def _wrap_with_scaler(estimator: Any) -> Pipeline:
    # Scale features before fitting scale-sensitive linear models.
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def _build_linear_regression(params: dict[str, Any], _: int) -> SklearnRegressorWrapper:
    estimator = _wrap_with_scaler(LinearRegression(**params))
    return SklearnRegressorWrapper("linear_regression", estimator, dict(params))


def _build_ridge(params: dict[str, Any], _: int) -> SklearnRegressorWrapper:
    estimator = _wrap_with_scaler(Ridge(**params))
    return SklearnRegressorWrapper("ridge", estimator, dict(params))


def _build_lasso(params: dict[str, Any], _: int) -> SklearnRegressorWrapper:
    estimator = _wrap_with_scaler(Lasso(**params))
    return SklearnRegressorWrapper("lasso", estimator, dict(params))


def _build_random_forest(params: dict[str, Any], random_seed: int) -> SklearnRegressorWrapper:
    fitted_params = _with_random_state(params, random_seed)
    estimator = RandomForestRegressor(**fitted_params)
    return SklearnRegressorWrapper("random_forest", estimator, fitted_params)


def _build_xgboost(params: dict[str, Any], random_seed: int) -> SklearnRegressorWrapper:
    if XGBRegressor is None:
        raise ImportError(
            "xgboost is not installed. Install the project dependencies to use the xgboost baseline."
        )

    fitted_params = dict(params)
    fitted_params.setdefault("objective", "reg:squarederror")
    fitted_params = _with_random_state(fitted_params, random_seed)
    estimator = XGBRegressor(**fitted_params)
    return SklearnRegressorWrapper("xgboost", estimator, fitted_params)


# Estimators that accept an `n_jobs` parameterfor parallel-aware models.
_N_JOBS_AWARE_MODELS: frozenset[str] = frozenset({"random_forest", "xgboost"})


MODEL_REGISTRY: dict[str, tuple[str, Any]] = {
    "linear_regression": ("linear_regression", _build_linear_regression),
    "linear": ("linear_regression", _build_linear_regression),
    "ols": ("linear_regression", _build_linear_regression),
    "ridge": ("ridge", _build_ridge),
    "lasso": ("lasso", _build_lasso),
    "random_forest": ("random_forest", _build_random_forest),
    "rf": ("random_forest", _build_random_forest),
    "xgboost": ("xgboost", _build_xgboost),
    "xgb": ("xgboost", _build_xgboost),
    "xg_boost": ("xgboost", _build_xgboost),
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


def build_model(
    model_name: str,
    random_seed: int = 42,
    n_jobs: int = 1,
    **params: Any,
) -> SklearnRegressorWrapper:
    normalized_name = _normalize_model_name(model_name)

    if normalized_name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(set(MODEL_REGISTRY)))
        raise ValueError(f"Unsupported model name: {normalized_name}. Supported models: {supported}")

    estimator_name, builder = MODEL_REGISTRY[normalized_name]
    effective_params = dict(params)
    if estimator_name in _N_JOBS_AWARE_MODELS:
        effective_params.setdefault("n_jobs", n_jobs)
    return builder(effective_params, random_seed)

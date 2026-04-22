from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Keep the test runnable both under pytest and as a standalone file in an IDE.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pd_progression.models import SklearnRegressorWrapper, build_model


def test_build_model_resolves_linear_aliases() -> None:
    wrapper = build_model("  Linear-Regression  ", fit_intercept=False)

    assert isinstance(wrapper, SklearnRegressorWrapper)
    assert wrapper.estimator_name == "linear_regression"
    assert isinstance(wrapper.estimator, Pipeline)
    assert isinstance(wrapper.estimator.named_steps["scaler"], StandardScaler)
    inner = wrapper.estimator.named_steps["model"]
    assert isinstance(inner, LinearRegression)
    assert wrapper.params == {"fit_intercept": False}
    assert inner.fit_intercept is False


def test_build_model_supports_ridge_and_lasso() -> None:
    ridge = build_model("ridge", alpha=2.5)
    lasso = build_model("LASSO", alpha=0.1, max_iter=5000)

    assert isinstance(ridge.estimator, Pipeline)
    ridge_inner = ridge.estimator.named_steps["model"]
    assert isinstance(ridge_inner, Ridge)
    assert ridge.estimator_name == "ridge"
    assert ridge.params == {"alpha": 2.5}
    assert ridge_inner.alpha == 2.5

    assert isinstance(lasso.estimator, Pipeline)
    lasso_inner = lasso.estimator.named_steps["model"]
    assert isinstance(lasso_inner, Lasso)
    assert lasso.estimator_name == "lasso"
    assert lasso.params == {"alpha": 0.1, "max_iter": 5000}
    assert lasso_inner.alpha == 0.1
    assert lasso_inner.max_iter == 5000


def test_build_model_injects_random_state_for_random_forest() -> None:
    wrapper = build_model("rf", random_seed=123, n_estimators=10)

    assert isinstance(wrapper.estimator, RandomForestRegressor)
    assert wrapper.estimator_name == "random_forest"
    assert wrapper.params["n_estimators"] == 10
    assert wrapper.params["random_state"] == 123
    assert wrapper.estimator.random_state == 123


def test_build_model_preserves_explicit_random_state() -> None:
    wrapper = build_model("random_forest", random_seed=123, random_state=77)

    assert wrapper.params["random_state"] == 77
    assert wrapper.estimator.random_state == 77


def test_build_model_injects_n_jobs_for_random_forest() -> None:
    wrapper = build_model("random_forest", n_jobs=4)

    assert wrapper.params["n_jobs"] == 4
    assert wrapper.estimator.n_jobs == 4


def test_build_model_does_not_pass_n_jobs_to_linear_models() -> None:
    wrapper = build_model("ridge", n_jobs=8, alpha=1.0)

    assert "n_jobs" not in wrapper.params


def test_build_model_supports_xgboost() -> None:
    pytest.importorskip("xgboost")

    wrapper = build_model("XGBoost", random_seed=99, n_estimators=25, max_depth=4)

    assert wrapper.estimator_name == "xgboost"
    assert wrapper.params["n_estimators"] == 25
    assert wrapper.params["max_depth"] == 4
    assert wrapper.params["objective"] == "reg:squarederror"
    assert wrapper.params["random_state"] == 99
    assert wrapper.estimator.get_params()["random_state"] == 99


def test_build_model_rejects_unknown_model_names() -> None:
    with pytest.raises(ValueError, match=r"Unsupported model name: not_a_model"):
        build_model("not-a-model")

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pd_progression.interpretability import extract_feature_importance


class _DummyTree:
    feature_importances_ = [0.1, 0.7, 0.2]


class _DummyLinear:
    coef_ = [-2.0, 0.5, 0.0]


class _DummyPipeline:
    named_steps = {"model": _DummyLinear()}


class _DummyWrapper:
    def __init__(self, estimator) -> None:
        self.estimator = estimator


def test_extract_feature_importance_from_tree_model() -> None:
    frame = extract_feature_importance(
        _DummyWrapper(_DummyTree()),
        ["feature_a", "feature_b", "feature_c"],
    )

    assert list(frame["feature"]) == ["feature_b", "feature_c", "feature_a"]
    assert list(frame["rank"]) == [1, 2, 3]
    assert set(frame["method"]) == {"feature_importances_"}


def test_extract_feature_importance_from_linear_pipeline_uses_absolute_coefficients() -> None:
    frame = extract_feature_importance(
        _DummyWrapper(_DummyPipeline()),
        ["feature_a", "feature_b", "feature_c"],
    )

    assert list(frame["feature"]) == ["feature_a", "feature_b", "feature_c"]
    assert list(frame["importance"]) == pytest.approx([2.0, 0.5, 0.0])
    assert list(frame["raw_value"]) == pytest.approx([-2.0, 0.5, 0.0])
    assert set(frame["method"]) == {"abs_coef"}

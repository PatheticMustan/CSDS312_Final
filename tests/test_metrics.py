from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pd_progression.metrics import compute_regression_metrics, r2


def test_compute_regression_metrics_matches_known_values() -> None:
    metrics = compute_regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])

    assert metrics["rmse"] == pytest.approx(math.sqrt(4.0 / 3.0))
    assert metrics["mae"] == pytest.approx(2.0 / 3.0)
    assert metrics["smape"] == pytest.approx((0.0 + 0.0 + 0.5) / 3.0)
    assert metrics["r2"] == pytest.approx(-1.0)


def test_r2_handles_constant_targets() -> None:
    assert r2([4.0, 4.0, 4.0], [4.0, 4.0, 4.0]) == pytest.approx(1.0)
    assert r2([4.0, 4.0, 4.0], [3.0, 4.0, 5.0]) == pytest.approx(0.0)

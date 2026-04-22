from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Keep the test runnable both under pytest and as a standalone file in an IDE.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pd_progression.parallel import parallel_groupby_apply


# Must be a top-level function so loky workers can pickle it.
def _patient_rolling_mean(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("visit_month").copy()
    group["value_rolling2"] = (
        group["value"].rolling(window=2, min_periods=1).mean()
    )
    return group


def _make_frame(n_patients: int = 6, visits_per_patient: int = 5) -> pd.DataFrame:
    rows = []
    for patient_id in range(n_patients):
        for visit in range(visits_per_patient):
            rows.append(
                {
                    "patient_id": patient_id,
                    "visit_month": visit * 6,
                    "value": float(patient_id * 10 + visit),
                }
            )
    return pd.DataFrame(rows)


def test_parallel_matches_serial_output() -> None:
    df = _make_frame()

    serial = parallel_groupby_apply(df, "patient_id", _patient_rolling_mean, n_jobs=1)
    parallel = parallel_groupby_apply(df, "patient_id", _patient_rolling_mean, n_jobs=2)

    pd.testing.assert_frame_equal(
        serial.reset_index(drop=True),
        parallel.reset_index(drop=True),
    )


def test_parallel_preserves_group_order_and_shape() -> None:
    df = _make_frame(n_patients=4, visits_per_patient=3)

    result = parallel_groupby_apply(df, "patient_id", _patient_rolling_mean, n_jobs=2)

    assert len(result) == len(df)
    assert list(result["patient_id"].unique()) == [0, 1, 2, 3]
    assert "value_rolling2" in result.columns


def test_empty_frame_returns_empty() -> None:
    df = pd.DataFrame({"patient_id": [], "visit_month": [], "value": []})

    result = parallel_groupby_apply(df, "patient_id", _patient_rolling_mean, n_jobs=2)

    assert result.empty


def test_missing_group_column_raises() -> None:
    df = _make_frame()

    with pytest.raises(KeyError, match="Group column not found"):
        parallel_groupby_apply(df, "not_a_column", _patient_rolling_mean, n_jobs=1)

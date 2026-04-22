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

from pd_progression.splits import make_patient_level_split, make_time_aware_split


def test_patient_level_split_keeps_patients_disjoint() -> None:
    df = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2, 3, 3],
            "visit_id": [11, 12, 21, 22, 31, 32],
            "visit_month": [0, 6, 0, 6, 0, 6],
            "feature_a": [1, 2, 3, 4, 5, 6],
        }
    )

    split = make_patient_level_split(df, validation_fraction=1 / 3, random_seed=42)

    train_patients = set(df.loc[list(split.train_index), "patient_id"])
    validation_patients = set(df.loc[list(split.validation_index), "patient_id"])

    assert train_patients.isdisjoint(validation_patients)
    assert split.strategy == "patient"
    assert split.metadata["group_column"] == "patient_id"
    assert split.metadata["validation_fraction"] == pytest.approx(1 / 3)
    assert split.metadata["train_group_count"] + split.metadata["validation_group_count"] == 3


def test_time_aware_split_holds_out_later_visits_per_patient() -> None:
    df = pd.DataFrame(
        {
            "patient_id": [1, 1, 1, 2, 2],
            "visit_id": [11, 12, 13, 21, 22],
            "visit_month": [0, 6, 12, 0, 6],
            "feature_a": [1, 2, 3, 4, 5],
        }
    )

    split = make_time_aware_split(df, validation_fraction=1 / 3, random_seed=7)

    train = df.loc[list(split.train_index)].sort_values(["patient_id", "visit_month"])
    validation = df.loc[list(split.validation_index)].sort_values(["patient_id", "visit_month"])

    assert split.strategy == "time_aware"
    assert split.metadata["group_column"] == "patient_id"
    assert split.metadata["time_column"] == "visit_month"
    assert train.groupby("patient_id").size().to_dict() == {1: 2, 2: 1}
    assert validation.groupby("patient_id").size().to_dict() == {1: 1, 2: 1}
    assert validation.loc[validation["patient_id"] == 1, "visit_month"].iloc[0] == 12
    assert validation.loc[validation["patient_id"] == 2, "visit_month"].iloc[0] == 6

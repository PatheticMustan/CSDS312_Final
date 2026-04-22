from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

# Keep the test runnable both under pytest and as a standalone file in an IDE.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pd_progression.feature_engineering import (
    LONGITUDINAL_BASE_COLUMNS,
    add_longitudinal_features,
    build_training_dataset,
)


def _make_merged_frame(n_patients: int = 6, visits_per_patient: int = 5) -> pd.DataFrame:
    """Synthesize a frame resembling the post-aggregation merged table."""
    rows = []
    for patient_id in range(n_patients):
        for visit in range(visits_per_patient):
            rows.append(
                {
                    "patient_id": patient_id,
                    "visit_id": f"{patient_id}_{visit}",
                    "visit_month": visit * 6,
                }
            )
    df = pd.DataFrame(rows)
    # Fill the longitudinal base columns with deterministic per-patient values.
    for i, col in enumerate(LONGITUDINAL_BASE_COLUMNS):
        df[col] = df["patient_id"] * (i + 1) + df["visit_month"].astype(float)
    return df


def _make_raw_tables(n_patients: int = 4, visits_per_patient: int = 3):
    clinical_rows = []
    protein_rows = []
    peptide_rows = []
    for patient_id in range(n_patients):
        for visit in range(visits_per_patient):
            visit_id = f"{patient_id}_{visit}"
            clinical_rows.append(
                {
                    "visit_id": visit_id,
                    "patient_id": patient_id,
                    "visit_month": visit * 6,
                    "updrs_1": 10.0 + patient_id + visit,
                    "updrs_2": 5.0 + patient_id,
                    "updrs_3": 20.0 + visit,
                    "updrs_4": 1.0 * visit,
                }
            )
            for uniprot_idx in range(3):
                protein_rows.append(
                    {
                        "visit_id": visit_id,
                        "patient_id": patient_id,
                        "visit_month": visit * 6,
                        "UniProt": f"P{uniprot_idx}",
                        "NPX": float(patient_id + uniprot_idx + visit),
                    }
                )
                peptide_rows.append(
                    {
                        "visit_id": visit_id,
                        "patient_id": patient_id,
                        "visit_month": visit * 6,
                        "UniProt": f"P{uniprot_idx}",
                        "Peptide": f"pep_{uniprot_idx}",
                        "PeptideAbundance": float(100 * (patient_id + 1) + uniprot_idx + visit),
                    }
                )
    return (
        pd.DataFrame(clinical_rows),
        pd.DataFrame(protein_rows),
        pd.DataFrame(peptide_rows),
    )


def test_add_longitudinal_features_parallel_matches_serial() -> None:
    df = _make_merged_frame(n_patients=8, visits_per_patient=6)

    serial = add_longitudinal_features(df, n_jobs=1)
    parallel = add_longitudinal_features(df, n_jobs=2)

    # Sort both by the canonical ordering because the serial path returns rows
    # in input order within each group; parallel must match that up to concat.
    sort_cols = ["patient_id", "visit_month", "visit_id"]
    pd.testing.assert_frame_equal(
        serial.sort_values(sort_cols).reset_index(drop=True),
        parallel.sort_values(sort_cols).reset_index(drop=True),
    )


def test_add_longitudinal_features_produces_expected_schema() -> None:
    df = _make_merged_frame(n_patients=3, visits_per_patient=4)

    out = add_longitudinal_features(df, n_jobs=1)

    for col in LONGITUDINAL_BASE_COLUMNS:
        assert f"{col}_lag1" in out.columns
        assert f"{col}_delta" in out.columns
        assert f"{col}_rolling3_mean" in out.columns
        assert f"{col}_rolling3_std" in out.columns
        assert f"{col}_lag1_missing" in out.columns
        assert f"{col}_delta_missing" in out.columns

    assert "visit_gap" in out.columns
    assert "visit_gap_missing" in out.columns
    assert "visit_number" in out.columns
    assert "has_prior_visit" in out.columns

    # No NaNs should survive the post-join fills.
    engineered_cols = [c for c in out.columns if c not in {"patient_id", "visit_id"}]
    assert not out[engineered_cols].isna().any().any()


def test_build_training_dataset_parallel_matches_serial() -> None:
    clinical, proteins, peptides = _make_raw_tables(n_patients=5, visits_per_patient=4)

    serial = build_training_dataset(clinical, proteins, peptides, n_jobs=1)
    parallel = build_training_dataset(clinical, proteins, peptides, n_jobs=2)

    pd.testing.assert_frame_equal(serial, parallel)


def test_add_longitudinal_features_lag_and_delta_values() -> None:
    # Single patient, simple ramp; verify lag / delta / rolling semantics
    # survived the refactor correctly.
    df = pd.DataFrame(
        {
            "patient_id": [1, 1, 1],
            "visit_id": ["1_0", "1_1", "1_2"],
            "visit_month": [0, 6, 12],
        }
    )
    for col in LONGITUDINAL_BASE_COLUMNS:
        df[col] = [1.0, 2.0, 3.0]

    out = add_longitudinal_features(df, n_jobs=1).sort_values("visit_month").reset_index(drop=True)

    sample_col = LONGITUDINAL_BASE_COLUMNS[0]
    assert list(out[f"{sample_col}_lag1"]) == [0.0, 1.0, 2.0]
    assert list(out[f"{sample_col}_lag1_missing"]) == [1, 0, 0]
    assert list(out[f"{sample_col}_delta"]) == [0.0, 1.0, 1.0]
    assert list(out[f"{sample_col}_delta_missing"]) == [1, 0, 0]
    assert list(out["visit_number"]) == [0, 1, 2]
    assert list(out["has_prior_visit"]) == [0, 1, 1]
    assert list(out["visit_gap"]) == [0.0, 6.0, 6.0]
    assert list(out["visit_gap_missing"]) == [1, 0, 0]

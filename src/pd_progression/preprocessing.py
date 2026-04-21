from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .io import load_amp_pd_raw_tables, validate_required_columns


CLINICAL_COLUMNS = (
    "visit_id",
    "patient_id",
    "visit_month",
    "updrs_1",
    "updrs_2",
    "updrs_3",
    "updrs_4",
    "upd23b_clinical_state_on_medication",
)

PROTEIN_COLUMNS = (
    "visit_id",
    "visit_month",
    "patient_id",
    "UniProt",
    "NPX",
)

PEPTIDE_COLUMNS = (
    "visit_id",
    "visit_month",
    "patient_id",
    "UniProt",
    "Peptide",
    "PeptideAbundance",
)


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_clinical(clinical: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(clinical, CLINICAL_COLUMNS[:3], "clinical")

    df = clinical.copy()
    df = df.drop_duplicates()

    numeric_cols = ["patient_id", "visit_month", "updrs_1", "updrs_2", "updrs_3", "updrs_4"]
    df = _coerce_numeric(df, numeric_cols)

    if "upd23b_clinical_state_on_medication" in df.columns:
        df["medication_on"] = (
            df["upd23b_clinical_state_on_medication"]
            .fillna("Unknown")
            .replace({"On": "On", "Off": "Off"})
        )
        df["medication_on_flag"] = (df["medication_on"] == "On").astype(int)
        df["medication_off_flag"] = (df["medication_on"] == "Off").astype(int)
        df["medication_unknown_flag"] = (df["medication_on"] == "Unknown").astype(int)

    return df.sort_values(["patient_id", "visit_month", "visit_id"]).reset_index(drop=True)


def clean_proteins(proteins: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(proteins, PROTEIN_COLUMNS, "proteins")

    df = proteins.copy()
    df = df.drop_duplicates()
    df = _coerce_numeric(df, ["patient_id", "visit_month", "NPX"])

    df = df.dropna(subset=["visit_id", "patient_id", "visit_month", "UniProt"])
    return df.sort_values(["patient_id", "visit_month", "visit_id", "UniProt"]).reset_index(drop=True)


def clean_peptides(peptides: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(peptides, PEPTIDE_COLUMNS, "peptides")

    df = peptides.copy()
    df = df.drop_duplicates()
    df = _coerce_numeric(df, ["patient_id", "visit_month", "PeptideAbundance"])

    df = df.dropna(subset=["visit_id", "patient_id", "visit_month", "UniProt", "Peptide"])
    return df.sort_values(["patient_id", "visit_month", "visit_id", "UniProt", "Peptide"]).reset_index(drop=True)


def load_and_clean_raw_tables(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    tables = load_amp_pd_raw_tables(raw_dir)

    clinical = clean_clinical(tables["clinical"])
    proteins = clean_proteins(tables["proteins"])
    peptides = clean_peptides(tables["peptides"])

    if "supplemental_clinical" in tables:
        supplemental = clean_clinical(tables["supplemental_clinical"])
    else:
        supplemental = pd.DataFrame(columns=clinical.columns)

    return {
        "clinical": clinical,
        "supplemental_clinical": supplemental,
        "proteins": proteins,
        "peptides": peptides,
    }

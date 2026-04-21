from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_std(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    value = series.std()
    return 0.0 if pd.isna(value) else float(value)


def aggregate_proteins(proteins: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        proteins.groupby(["patient_id", "visit_id", "visit_month"], as_index=False)
        .agg(
            protein_n_unique=("UniProt", "nunique"),
            protein_n_rows=("UniProt", "size"),
            protein_npx_mean=("NPX", "mean"),
            protein_npx_std=("NPX", _safe_std),
            protein_npx_min=("NPX", "min"),
            protein_npx_max=("NPX", "max"),
            protein_npx_median=("NPX", "median"),
        )
    )
    return grouped


def aggregate_peptides(peptides: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        peptides.groupby(["patient_id", "visit_id", "visit_month"], as_index=False)
        .agg(
            peptide_n_unique=("Peptide", "nunique"),
            peptide_n_rows=("Peptide", "size"),
            peptide_protein_n_unique=("UniProt", "nunique"),
            peptide_abundance_mean=("PeptideAbundance", "mean"),
            peptide_abundance_std=("PeptideAbundance", _safe_std),
            peptide_abundance_min=("PeptideAbundance", "min"),
            peptide_abundance_max=("PeptideAbundance", "max"),
            peptide_abundance_median=("PeptideAbundance", "median"),
        )
    )
    return grouped


def add_longitudinal_features(
    df: pd.DataFrame,
    group_col: str = "patient_id",
    time_col: str = "visit_month",
) -> pd.DataFrame:
    df = df.sort_values([group_col, time_col, "visit_id"]).copy()

    numeric_base_cols = [
        "protein_n_unique",
        "protein_n_rows",
        "protein_npx_mean",
        "protein_npx_std",
        "protein_npx_min",
        "protein_npx_max",
        "protein_npx_median",
        "peptide_n_unique",
        "peptide_n_rows",
        "peptide_protein_n_unique",
        "peptide_abundance_mean",
        "peptide_abundance_std",
        "peptide_abundance_min",
        "peptide_abundance_max",
        "peptide_abundance_median",
    ]

    for col in numeric_base_cols:
        if col not in df.columns:
            continue

        df[f"{col}_lag1"] = df.groupby(group_col)[col].shift(1)
        df[f"{col}_delta"] = df[col] - df[f"{col}_lag1"]
        df[f"{col}_rolling3_mean"] = (
            df.groupby(group_col)[col]
            .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
        )
        df[f"{col}_rolling3_std"] = (
            df.groupby(group_col)[col]
            .transform(lambda s: s.rolling(window=3, min_periods=2).std())
            .fillna(0.0)
        )

    df["visit_gap"] = df.groupby(group_col)[time_col].diff()
    df["visit_number"] = df.groupby(group_col).cumcount()
    df["has_prior_visit"] = (df["visit_number"] > 0).astype(int)

    for col in numeric_base_cols:
        lag_col = f"{col}_lag1"
        if lag_col in df.columns:
            df[f"{lag_col}_missing"] = df[lag_col].isna().astype(int)
            df[lag_col] = df[lag_col].fillna(0.0)

        delta_col = f"{col}_delta"
        if delta_col in df.columns:
            df[f"{delta_col}_missing"] = df[delta_col].isna().astype(int)
            df[delta_col] = df[delta_col].fillna(0.0)

    df["visit_gap_missing"] = df["visit_gap"].isna().astype(int)
    df["visit_gap"] = df["visit_gap"].fillna(0.0)

    return df


def build_training_dataset(
    clinical: pd.DataFrame,
    proteins: pd.DataFrame,
    peptides: pd.DataFrame,
) -> pd.DataFrame:
    protein_features = aggregate_proteins(proteins)
    peptide_features = aggregate_peptides(peptides)

    merged = clinical.merge(
        protein_features,
        on=["patient_id", "visit_id", "visit_month"],
        how="left",
    ).merge(
        peptide_features,
        on=["patient_id", "visit_id", "visit_month"],
        how="left",
    )

    count_cols = [
        "protein_n_unique",
        "protein_n_rows",
        "peptide_n_unique",
        "peptide_n_rows",
        "peptide_protein_n_unique",
    ]
    summary_cols = [
        "protein_npx_mean",
        "protein_npx_std",
        "protein_npx_min",
        "protein_npx_max",
        "protein_npx_median",
        "peptide_abundance_mean",
        "peptide_abundance_std",
        "peptide_abundance_min",
        "peptide_abundance_max",
        "peptide_abundance_median",
    ]

    for col in count_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    for col in summary_cols:
        if col in merged.columns:
            merged[f"{col}_missing"] = merged[col].isna().astype(int)
            merged[col] = merged[col].fillna(0.0)

    merged = add_longitudinal_features(merged)
    return merged.sort_values(["patient_id", "visit_month", "visit_id"]).reset_index(drop=True)

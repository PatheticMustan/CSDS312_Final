from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from .parallel import parallel_groupby_apply


LONGITUDINAL_BASE_COLUMNS: tuple[str, ...] = (
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
)

CLINICAL_TARGET_COLUMNS: tuple[str, ...] = (
    "updrs_1",
    "updrs_2",
    "updrs_3",
    "updrs_4",
)


def _safe_std(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    value = series.std()
    return 0.0 if pd.isna(value) else float(value)


def _compute_longitudinal_for_patient(
    group_df: pd.DataFrame,
    base_columns: tuple[str, ...],
    time_col: str,
) -> pd.DataFrame:

    group_df = group_df.sort_values([time_col, "visit_id"]).copy()

    for col in base_columns:
        if col not in group_df.columns:
            continue

        series = group_df[col]
        group_df[f"{col}_lag1"] = series.shift(1)
        group_df[f"{col}_delta"] = series - group_df[f"{col}_lag1"]
        group_df[f"{col}_rolling3_mean"] = (
            series.rolling(window=3, min_periods=1).mean()
        )
        group_df[f"{col}_rolling3_std"] = (
            series.rolling(window=3, min_periods=2).std().fillna(0.0)
        )

    group_df["visit_gap"] = group_df[time_col].diff()
    group_df["visit_number"] = np.arange(len(group_df), dtype=np.int64)
    group_df["has_prior_visit"] = (group_df["visit_number"] > 0).astype(int)
    return group_df


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
    n_jobs: int = 1,
) -> pd.DataFrame:
    df = df.sort_values([group_col, time_col, "visit_id"]).copy()

    worker = partial(
        _compute_longitudinal_for_patient,
        base_columns=LONGITUDINAL_BASE_COLUMNS,
        time_col=time_col,
    )
    df = parallel_groupby_apply(df, group_col, worker, n_jobs=n_jobs)


    missing_indicators: dict[str, pd.Series] = {}
    for col in LONGITUDINAL_BASE_COLUMNS:
        lag_col = f"{col}_lag1"
        if lag_col in df.columns:
            missing_indicators[f"{lag_col}_missing"] = df[lag_col].isna().astype(int)
            df[lag_col] = df[lag_col].fillna(0.0)

        delta_col = f"{col}_delta"
        if delta_col in df.columns:
            missing_indicators[f"{delta_col}_missing"] = df[delta_col].isna().astype(int)
            df[delta_col] = df[delta_col].fillna(0.0)

    missing_indicators["visit_gap_missing"] = df["visit_gap"].isna().astype(int)
    df["visit_gap"] = df["visit_gap"].fillna(0.0)

    df = pd.concat(
        [df, pd.DataFrame(missing_indicators, index=df.index)],
        axis=1,
    )

    return df


def add_clinical_history_and_future_targets(
    df: pd.DataFrame,
    group_col: str = "patient_id",
    time_col: str = "visit_month",
) -> pd.DataFrame:
    df = df.sort_values([group_col, time_col, "visit_id"]).copy()
    grouped = df.groupby(group_col, sort=False)

    for col in CLINICAL_TARGET_COLUMNS:
        if col not in df.columns:
            continue

        current_col = f"current_{col}"
        df[current_col] = df[col]
        df[f"{current_col}_missing"] = df[current_col].isna().astype(int)
        df[current_col] = df[current_col].fillna(0.0)

        prior_col = f"prior_{col}"
        df[prior_col] = grouped[col].shift(1)
        df[f"{prior_col}_missing"] = df[prior_col].isna().astype(int)
        df[prior_col] = df[prior_col].fillna(0.0)

        future_col = f"future_{col}"
        df[future_col] = grouped[col].shift(-1)

    return df


def build_training_dataset(
    clinical: pd.DataFrame,
    proteins: pd.DataFrame,
    peptides: pd.DataFrame,
    n_jobs: int = 1,
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

    merged = add_longitudinal_features(merged, n_jobs=n_jobs)
    merged = add_clinical_history_and_future_targets(merged)
    return merged.sort_values(["patient_id", "visit_month", "visit_id"]).reset_index(drop=True)

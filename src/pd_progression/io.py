from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

RAW_FILENAME_MAP = {
    "clinical": "train_clinical_data.csv",
    "supplemental_clinical": "supplemental_clinical_data.csv",
    "proteins": "train_proteins.csv",
    "peptides": "train_peptides.csv",
}

RAW_REQUIRED_COLUMNS = {
    "clinical": ("visit_id", "patient_id", "visit_month"),
    "supplemental_clinical": ("visit_id", "patient_id", "visit_month"),
    "proteins": ("visit_id", "patient_id", "visit_month"),
    "peptides": ("visit_id", "patient_id", "visit_month"),
}


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    table_name: str,
) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def load_table_by_name(name: str, raw_dir: str | Path) -> pd.DataFrame:
    if name not in RAW_FILENAME_MAP:
        raise KeyError(f"Unknown raw table name: {name}")
    df = load_csv(Path(raw_dir) / RAW_FILENAME_MAP[name])
    validate_required_columns(df, RAW_REQUIRED_COLUMNS[name], name)
    return df


def load_amp_pd_raw_tables(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    return {name: load_table_by_name(name, raw_dir) for name in RAW_FILENAME_MAP}


def raw_schema_summary(raw_dir: str | Path) -> pd.DataFrame:
    rows = []
    for name, filename in RAW_FILENAME_MAP.items():
        df = pd.read_csv(Path(raw_dir) / filename, nrows=0)
        rows.append(
            {
                "table": name,
                "filename": filename,
                "columns": list(df.columns),
                "column_count": len(df.columns),
            }
        )
    return pd.DataFrame(rows)


#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from typing import Any

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean NaN values from the prepared AMP-PD dataset")
    parser.add_argument(
        "--input",
        default="data/cleaned/final_dataset.csv",
        help="Path to the prepared dataset CSV",
    )
    parser.add_argument(
        "--output",
        default="data/cleaned/final_dataset_no_nan.csv",
        help="Path to write the NaN-cleaned dataset",
    )
    parser.add_argument(
        "--target-column",
        default="updrs_1",
        help="Target column to drop when missing before imputing features",
    )
    return parser


def _fill_object_column(series: pd.Series) -> pd.Series:
    mode = series.mode(dropna=True)
    if not mode.empty:
        fill_value: Any = mode.iloc[0]
    else:
        fill_value = "Unknown"
    return series.fillna(fill_value)


def clean_nan_values(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, dict[str, int]]:
    cleaned = df.copy()
    stats = {
        "rows_dropped_target_missing": 0,
        "numeric_values_filled": 0,
        "categorical_values_filled": 0,
    }

    if target_column in cleaned.columns:
        target_missing = cleaned[target_column].isna()
        stats["rows_dropped_target_missing"] = int(target_missing.sum())
        cleaned = cleaned.loc[~target_missing].copy()

    for column in cleaned.columns:
        missing_count = int(cleaned[column].isna().sum())
        if missing_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(cleaned[column]):
            fill_value = cleaned[column].median()
            if pd.isna(fill_value):
                fill_value = 0.0
            cleaned[column] = cleaned[column].fillna(fill_value)
            stats["numeric_values_filled"] += missing_count
        else:
            cleaned[column] = _fill_object_column(cleaned[column])
            stats["categorical_values_filled"] += missing_count

    return cleaned, stats


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    cleaned_df, stats = clean_nan_values(df, target_column=args.target_column)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    print(f"Saved cleaned dataset to: {output_path}")
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Rows dropped because '{args.target_column}' was missing: {stats['rows_dropped_target_missing']}")
    print(f"Numeric NaNs filled: {stats['numeric_values_filled']}")
    print(f"Categorical NaNs filled: {stats['categorical_values_filled']}")
    print(f"Remaining NaNs: {int(cleaned_df.isna().sum().sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

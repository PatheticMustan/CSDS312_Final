#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from pd_progression.feature_engineering import build_training_dataset
from pd_progression.preprocessing import load_and_clean_raw_tables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build cleaned training dataset for AMP-PD project")
    parser.add_argument(
        "--raw-dir",
        default="data/raw/amp-parkinsons-disease-progression-prediction",
        help="Directory containing raw AMP-PD csv files",
    )
    parser.add_argument(
        "--output",
        default="data/cleaned/final_dataset.csv",
        help="Output path for cleaned training data",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker count for per-patient longitudinal feature construction",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    tables = load_and_clean_raw_tables(args.raw_dir)
    final_df = build_training_dataset(
        clinical=tables["clinical"],
        proteins=tables["proteins"],
        peptides=tables["peptides"],
        n_jobs=args.n_jobs,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"Saved prepared dataset to: {output_path}")
    print(f"Shape: {final_df.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

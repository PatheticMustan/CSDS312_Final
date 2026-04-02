from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import BaselineRunConfig
from .pipeline import run_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline training scaffold for Parkinson's progression prediction")
    parser.add_argument("--data", required=True, help="Path to a prepared CSV file")
    parser.add_argument("--config", help="Path to a JSON config file")
    parser.add_argument("--model", default="ridge", help="Model name")
    parser.add_argument("--target", default="updrs_1", help="Target column")
    parser.add_argument("--split", default="patient", help="Split strategy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    return parser


def load_config(args: argparse.Namespace) -> BaselineRunConfig:
    if args.config:
        payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
        return BaselineRunConfig.from_mapping(payload)
    return BaselineRunConfig(
        model_name=args.model,
        target_column=args.target,
        split_strategy=args.split,
        random_seed=args.seed,
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args)
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Prepared data file not found: {data_path}")

    import pandas as pd

    df = pd.read_csv(data_path)
    result = run_training_pipeline(df, config)
    print(result.run_dir)
    print(json.dumps(result.metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


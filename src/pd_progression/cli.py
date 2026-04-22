from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import BaselineRunConfig
from .pipeline import run_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline training scaffold for Parkinson's progression prediction")
    parser.add_argument("--data", help="Path to a prepared CSV file")
    parser.add_argument("--config", help="Path to a JSON config file")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--target", help="Target column")
    parser.add_argument("--split", help="Split strategy")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--model-params", help="JSON object with model hyperparameters")
    parser.add_argument("--output-dir", help="Output directory")
    return parser


def _parse_model_params(raw_value: str | None) -> dict[str, Any]:
    if raw_value is None:
        return {}
    payload = json.loads(raw_value)
    if not isinstance(payload, dict):
        raise ValueError("--model-params must decode to a JSON object")
    return payload


def load_config(args: argparse.Namespace) -> BaselineRunConfig:
    payload: dict[str, Any] = {}
    if args.config:
        payload.update(json.loads(Path(args.config).read_text(encoding="utf-8")))

    if args.data is not None:
        payload["data_path"] = args.data
    if args.output_dir is not None:
        payload["output_dir"] = args.output_dir
    if args.model is not None:
        payload["model_name"] = args.model
    if args.target is not None:
        payload["target_column"] = args.target
    if args.split is not None:
        payload["split_strategy"] = args.split
    if args.seed is not None:
        payload["random_seed"] = args.seed

    model_params = dict(payload.get("model_params") or {})
    model_params.update(_parse_model_params(args.model_params))
    if model_params:
        payload["model_params"] = model_params

    payload.setdefault("model_name", "ridge")
    payload.setdefault("target_column", "updrs_1")
    payload.setdefault("split_strategy", "patient")
    payload.setdefault("random_seed", 42)
    payload.setdefault("output_dir", "results")

    return BaselineRunConfig.from_mapping(payload)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args)
    data_path = config.data_path
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

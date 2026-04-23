#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
import pickle

from pd_progression.interpretability import extract_feature_importance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export feature importance from a saved baseline run"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing config.json and model.pkl",
    )
    parser.add_argument(
        "--output",
        help="Output CSV path. Defaults to <run-dir>/feature_importance.csv",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top-ranked features to print to stdout",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)

    config_path = run_dir / "config.json"
    model_path = run_dir / "model.pkl"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    feature_columns = list(config.get("feature_columns") or [])
    if not feature_columns:
        raise ValueError("Run config does not contain resolved feature_columns")

    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    importance = extract_feature_importance(model, feature_columns)
    output_path = Path(args.output) if args.output else run_dir / "feature_importance.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)

    print(f"Saved feature importances to: {output_path}")
    print(importance.head(args.top_k).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

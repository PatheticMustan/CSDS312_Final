#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import pandas as pd

from pd_progression.metrics import compute_regression_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved predictions")
    parser.add_argument("--predictions", required=True, help="CSV with y_true and y_pred columns")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    df = pd.read_csv(args.predictions)
    metrics = compute_regression_metrics(df["y_true"], df["y_pred"])
    print(pd.DataFrame([metrics]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


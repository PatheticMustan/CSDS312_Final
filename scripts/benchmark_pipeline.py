#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd

from pd_progression.feature_engineering import build_training_dataset
from pd_progression.preprocessing import load_and_clean_raw_tables


@dataclass
class BenchmarkRecord:
    stage: str
    n_jobs: int
    repeat_index: int
    seconds: float
    status: str
    model_name: str = ""
    target_column: str = ""
    split_strategy: str = ""
    n_rows: int = 0
    n_columns: int = 0
    rmse: float | None = None
    mae: float | None = None
    smape: float | None = None
    r2: float | None = None
    n_train: int | None = None
    n_val: int | None = None
    n_dropped_missing: int | None = None
    run_dir: str = ""
    error: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark dataset building and baseline training runtime"
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/amp-parkinsons-disease-progression-prediction",
        help="Raw AMP-PD directory used for build-stage benchmarking",
    )
    parser.add_argument(
        "--data",
        default="data/cleaned/final_dataset.csv",
        help="Prepared dataset used for training-stage benchmarking",
    )
    parser.add_argument(
        "--output",
        help="Output CSV path for benchmark rows. Defaults to a timestamped file under results/benchmarks/",
    )
    parser.add_argument(
        "--n-jobs-values",
        default="1,2,4",
        help="Comma-separated worker counts to benchmark",
    )
    parser.add_argument(
        "--models",
        default="random_forest,xgboost",
        help="Comma-separated model names to benchmark in the training stage",
    )
    parser.add_argument(
        "--targets",
        default="updrs_1",
        help="Comma-separated target columns to benchmark in the training stage",
    )
    parser.add_argument(
        "--split",
        default="patient",
        help="Split strategy for training benchmarks",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Repetitions per configuration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training benchmarks",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Validation fraction for training benchmarks",
    )
    parser.add_argument(
        "--model-params",
        default="{}",
        help="JSON object of model hyperparameters applied to every training run",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip dataset-building benchmarks",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training benchmarks",
    )
    parser.add_argument(
        "--keep-runs",
        action="store_true",
        help="Preserve per-run training artifacts under results/benchmarks/runs",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Raise immediately on the first failed benchmark run",
    )
    return parser


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    values = [int(item) for item in _parse_csv_list(raw)]
    if not values:
        raise ValueError("At least one n_jobs value is required")
    return values


def _benchmark_build(raw_dir: str | Path, n_jobs_values: list[int], repeat: int) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for n_jobs in n_jobs_values:
        for repeat_index in range(1, repeat + 1):
            started = time.perf_counter()
            dataset = None
            try:
                tables = load_and_clean_raw_tables(raw_dir)
                dataset = build_training_dataset(
                    clinical=tables["clinical"],
                    proteins=tables["proteins"],
                    peptides=tables["peptides"],
                    n_jobs=n_jobs,
                )
                elapsed = time.perf_counter() - started
                records.append(
                    BenchmarkRecord(
                        stage="build_dataset",
                        n_jobs=n_jobs,
                        repeat_index=repeat_index,
                        seconds=elapsed,
                        status="ok",
                        n_rows=int(len(dataset)),
                        n_columns=int(len(dataset.columns)),
                    )
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                records.append(
                    BenchmarkRecord(
                        stage="build_dataset",
                        n_jobs=n_jobs,
                        repeat_index=repeat_index,
                        seconds=elapsed,
                        status="error",
                        n_rows=int(len(dataset)) if dataset is not None else 0,
                        n_columns=int(len(dataset.columns)) if dataset is not None else 0,
                        error=str(exc),
                    )
                )
    return records


def _benchmark_train(
    data_path: str | Path,
    output_root: Path,
    n_jobs_values: list[int],
    model_names: list[str],
    target_columns: list[str],
    split_strategy: str,
    repeat: int,
    random_seed: int,
    validation_fraction: float,
    model_params: dict,
    keep_runs: bool,
    fail_fast: bool,
) -> list[BenchmarkRecord]:
    from pd_progression.config import BaselineRunConfig
    from pd_progression.pipeline import run_training_pipeline

    df = pd.read_csv(data_path)
    records: list[BenchmarkRecord] = []

    for model_name in model_names:
        for target_column in target_columns:
            for n_jobs in n_jobs_values:
                for repeat_index in range(1, repeat + 1):
                    started = time.perf_counter()
                    try:
                        if keep_runs:
                            benchmark_output_dir = output_root / "runs"
                            benchmark_output_dir.mkdir(parents=True, exist_ok=True)
                            config = BaselineRunConfig(
                                model_name=model_name,
                                target_column=target_column,
                                split_strategy=split_strategy,
                                random_seed=random_seed,
                                validation_fraction=validation_fraction,
                                data_path=Path(data_path),
                                output_dir=benchmark_output_dir,
                                model_params=dict(model_params),
                                n_jobs=n_jobs,
                            )
                            result = run_training_pipeline(df, config)
                        else:
                            with tempfile.TemporaryDirectory(prefix="pd_benchmark_") as temp_dir:
                                config = BaselineRunConfig(
                                    model_name=model_name,
                                    target_column=target_column,
                                    split_strategy=split_strategy,
                                    random_seed=random_seed,
                                    validation_fraction=validation_fraction,
                                    data_path=Path(data_path),
                                    output_dir=Path(temp_dir),
                                    model_params=dict(model_params),
                                    n_jobs=n_jobs,
                                )
                                result = run_training_pipeline(df, config)

                        elapsed = time.perf_counter() - started
                        records.append(
                            BenchmarkRecord(
                                stage="train_model",
                                n_jobs=n_jobs,
                                repeat_index=repeat_index,
                                seconds=elapsed,
                                status="ok",
                                model_name=model_name,
                                target_column=target_column,
                                split_strategy=split_strategy,
                                n_rows=int(len(df)),
                                n_columns=int(len(df.columns)),
                                rmse=result.metrics.get("rmse"),
                                mae=result.metrics.get("mae"),
                                smape=result.metrics.get("smape"),
                                r2=result.metrics.get("r2"),
                                n_train=result.metrics.get("n_train"),
                                n_val=result.metrics.get("n_val"),
                                n_dropped_missing=result.metrics.get("n_dropped_missing"),
                                run_dir=str(result.run_dir) if keep_runs else "",
                            )
                        )
                    except Exception as exc:
                        elapsed = time.perf_counter() - started
                        records.append(
                            BenchmarkRecord(
                                stage="train_model",
                                n_jobs=n_jobs,
                                repeat_index=repeat_index,
                                seconds=elapsed,
                                status="error",
                                model_name=model_name,
                                target_column=target_column,
                                split_strategy=split_strategy,
                                n_rows=int(len(df)),
                                n_columns=int(len(df.columns)),
                                error=str(exc),
                            )
                        )
                        if fail_fast:
                            raise
    return records


def main() -> int:
    args = build_parser().parse_args()
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = Path("results/benchmarks") / f"{args.split}_{timestamp}_benchmark_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_jobs_values = _parse_int_csv(args.n_jobs_values)
    model_names = _parse_csv_list(args.models)
    target_columns = _parse_csv_list(args.targets)
    model_params = json.loads(args.model_params)
    if not isinstance(model_params, dict):
        raise ValueError("--model-params must decode to a JSON object")

    records: list[BenchmarkRecord] = []
    if not args.skip_build:
        records.extend(_benchmark_build(args.raw_dir, n_jobs_values, args.repeat))
    if not args.skip_train:
        records.extend(
            _benchmark_train(
                data_path=args.data,
                output_root=output_path.parent,
                n_jobs_values=n_jobs_values,
                model_names=model_names,
                target_columns=target_columns,
                split_strategy=args.split,
                repeat=args.repeat,
                random_seed=args.seed,
                validation_fraction=args.validation_fraction,
                model_params=model_params,
                keep_runs=args.keep_runs,
                fail_fast=args.fail_fast,
            )
        )

    frame = pd.DataFrame([asdict(record) for record in records])
    frame.to_csv(output_path, index=False)
    print(f"Saved benchmark results to: {output_path}")
    if not frame.empty:
        print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pd_progression.interpretability import extract_feature_importance


MODEL_ORDER = ["linear_regression", "ridge", "lasso", "random_forest", "xgboost"]
MODEL_LABELS = {
    "linear_regression": "Linear",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MODEL_SHORT = {
    "linear_regression": "Linear",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "random_forest": "RF",
    "xgboost": "XGB",
}
TARGET_LABELS = {
    "updrs_1": "UPDRS I",
    "updrs_2": "UPDRS II",
    "updrs_3": "UPDRS III",
    "updrs_4": "UPDRS IV",
    "future_updrs_1": "Future UPDRS I",
    "future_updrs_2": "Future UPDRS II",
    "future_updrs_3": "Future UPDRS III",
    "future_updrs_4": "Future UPDRS IV",
}
SPLIT_ORDER = ["patient", "time_aware"]
SPLIT_LABELS = {
    "patient": "Patient-Level Split",
    "time_aware": "Time-Aware Split",
}

PALETTE = {
    "patient": "#1f5aa6",
    "time_aware": "#d57a1f",
    "random_forest": "#1b7f79",
    "xgboost": "#c64f3a",
    "build_dataset": "#7f8c8d",
}

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "figure.titlesize": 16,
        "font.size": 10,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Generate presentation/report figures from experiment outputs")
    parser.add_argument(
        "--summary",
        default="results/summary_metrics.csv",
        help="CSV containing one row per training run",
    )
    parser.add_argument(
        "--runs-dir",
        default="results/runs",
        help="Directory containing per-run artifacts",
    )
    parser.add_argument(
        "--benchmarks-dir",
        default="results/benchmarks",
        help="Directory containing benchmark CSV files",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/figures",
        help="Directory where figures will be written",
    )
    return parser


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def target_sort_key(target: str) -> tuple[int, int, str]:
    match = re.search(r"(\d+)$", target)
    target_number = int(match.group(1)) if match else 99
    future_flag = 1 if target.startswith("future_") else 0
    return (future_flag, target_number, target)


def ordered_targets(summary_df: pd.DataFrame) -> list[str]:
    targets = sorted(summary_df["target"].dropna().unique().tolist(), key=target_sort_key)
    return targets


def target_label(target: str) -> str:
    return TARGET_LABELS.get(target, target.replace("_", " ").title())


def latest_benchmark_files(benchmarks_dir: Path) -> dict[str, Path]:
    latest: dict[str, Path] = {}
    for split in SPLIT_ORDER:
        matches = [
            path
            for path in benchmarks_dir.glob(f"{split}*.csv")
            if "benchmark" in path.name
        ]
        if not matches:
            continue

        def sort_key(path: Path) -> tuple[int, float, str]:
            match = re.search(r"_(\d+)\.csv$", path.name)
            return (
                int(match.group(1)) if match else -1,
                path.stat().st_mtime,
                path.name,
            )

        latest[split] = max(matches, key=sort_key)
    return latest


def prettify_feature_name(name: str) -> str:
    replacements = [
        ("current_updrs", "Current UPDRS"),
        ("prior_updrs", "Prior UPDRS"),
        ("medication_unknown_flag", "Medication unknown flag"),
        ("medication_on_flag", "Medication on flag"),
        ("medication_off_flag", "Medication off flag"),
        ("protein_npx", "Protein NPX"),
        ("peptide_abundance", "Peptide abundance"),
        ("peptide_protein_n_unique", "Peptide protein count"),
        ("protein_n_unique", "Protein count"),
        ("protein_n_rows", "Protein rows"),
        ("peptide_n_unique", "Peptide count"),
        ("peptide_n_rows", "Peptide rows"),
        ("rolling3", "3-visit rolling"),
        ("lag1", "lag-1"),
        ("delta", "delta"),
        ("visit_gap", "Visit gap"),
        ("visit_number", "Visit number"),
        ("has_prior_visit", "Has prior visit"),
        ("std", "std"),
        ("mean", "mean"),
        ("min", "min"),
        ("max", "max"),
        ("median", "median"),
        ("missing", "missing"),
    ]
    label = name
    for src, dst in replacements:
        label = label.replace(src, dst)
    label = label.replace("_", " ")
    label = re.sub(r"\s+", " ", label).strip()
    return label.title()


def plot_r2_heatmaps(summary_df: pd.DataFrame, output_dir: Path) -> None:
    target_order = ordered_targets(summary_df)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "performance",
        ["#b23a48", "#f4efe7", "#2a9d8f"],
    )
    vmin = float(summary_df["r2"].min())
    vmax = float(summary_df["r2"].max())
    if vmin < 0.0 < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        if vmin == vmax:
            eps = 1e-6 if vmin == 0.0 else abs(vmin) * 1e-6
            vmin -= eps
            vmax += eps
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, split in zip(axes, SPLIT_ORDER):
        pivot = (
            summary_df[summary_df["split"] == split]
            .pivot(index="model", columns="target", values="r2")
            .reindex(index=MODEL_ORDER, columns=target_order)
        )

        image = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(SPLIT_LABELS[split])
        ax.set_xticks(range(len(target_order)), [target_label(t) for t in target_order])
        ax.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER])

        for row_idx, model in enumerate(MODEL_ORDER):
            for col_idx, target in enumerate(target_order):
                value = float(pivot.loc[model, target])
                text_color = "#111111" if abs(value) < 0.25 else "white"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    fontweight="bold",
                )

    cbar = fig.colorbar(image, ax=axes, shrink=0.88)
    cbar.set_label(r"$R^2$ (higher is better)")
    fig.suptitle("Model Performance Across Targets", y=1.03)
    save_figure(fig, output_dir, "01_r2_heatmaps")


def plot_best_r2_by_target(summary_df: pd.DataFrame, output_dir: Path) -> None:
    target_order = ordered_targets(summary_df)
    best = (
        summary_df.sort_values(["target", "split", "r2", "rmse"], ascending=[True, True, False, True])
        .groupby(["target", "split"], as_index=False)
        .first()
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    x = np.arange(len(target_order))
    width = 0.34

    for idx, split in enumerate(SPLIT_ORDER):
        split_df = (
            best[best["split"] == split]
            .set_index("target")
            .reindex(target_order)
            .reset_index()
        )
        offset = (-width / 2) if idx == 0 else (width / 2)
        bars = ax.bar(
            x + offset,
            split_df["r2"],
            width=width,
            color=PALETTE[split],
            label=SPLIT_LABELS[split],
            alpha=0.92,
        )
        for bar, (_, row) in zip(bars, split_df.iterrows()):
            value = float(row["r2"])
            model_label = MODEL_SHORT[row["model"]]
            y = value + 0.02 if value >= 0 else value - 0.03
            va = "bottom" if value >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{model_label}\n{value:.2f}",
                ha="center",
                va=va,
                fontsize=9,
                fontweight="bold",
            )

    ax.axhline(0.0, color="#666666", linewidth=1)
    ax.set_xticks(x, [target_label(t) for t in target_order])
    ax.set_ylabel(r"Best $R^2$")
    ax.set_title("Best Model by Target and Validation Strategy")
    ax.legend(ncols=2, loc="upper left")
    save_figure(fig, output_dir, "02_best_r2_by_target_split")


def plot_prediction_scatter(summary_df: pd.DataFrame, runs_dir: Path, output_dir: Path) -> None:
    chosen_rows = []
    for split in SPLIT_ORDER:
        row = (
            summary_df[summary_df["split"] == split]
            .sort_values(["r2", "rmse"], ascending=[False, True])
            .iloc[0]
        )
        chosen_rows.append(row)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    for ax, row in zip(axes, chosen_rows):
        predictions = pd.read_csv(runs_dir / row["run_name"] / "predictions.csv")
        min_value = min(predictions["y_true"].min(), predictions["y_pred"].min())
        max_value = max(predictions["y_true"].max(), predictions["y_pred"].max())
        pad = 0.05 * (max_value - min_value)
        low = min_value - pad
        high = max_value + pad

        split = row["split"]
        ax.scatter(
            predictions["y_true"],
            predictions["y_pred"],
            s=26,
            alpha=0.55,
            color=PALETTE[split],
            edgecolors="none",
        )
        ax.plot([low, high], [low, high], linestyle="--", linewidth=1.25, color="#444444")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("Observed score")
        ax.set_ylabel("Predicted score")
        ax.set_title(
            f"{SPLIT_LABELS[split]}\n{MODEL_LABELS[row['model']]} on {target_label(row['target'])}"
        )
        ax.text(
            0.03,
            0.97,
            f"RMSE {row['rmse']:.2f}\nMAE {row['mae']:.2f}\n$R^2$ {row['r2']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#dddddd"},
        )

    fig.suptitle("Observed vs Predicted Scores for the Strongest Runs", y=1.03)
    save_figure(fig, output_dir, "03_prediction_scatter_best_runs")


def plot_benchmark_scaling(benchmarks_dir: Path, output_dir: Path) -> None:
    benchmark_files = latest_benchmark_files(benchmarks_dir)
    if not benchmark_files:
        print(f"No benchmark CSV files found under: {benchmarks_dir}. Skipping benchmark figure.")
        return
    frames = [pd.read_csv(path) for path in benchmark_files.values()]
    benchmark_df = pd.concat(frames, ignore_index=True)
    benchmark_df = benchmark_df[benchmark_df["status"] == "ok"].copy()

    build_df = (
        benchmark_df[benchmark_df["stage"] == "build_dataset"]
        .groupby("n_jobs")["seconds"]
        .agg(mean_seconds="mean", std_seconds="std")
        .reset_index()
    )

    train_df = benchmark_df[benchmark_df["stage"] == "train_model"].copy()
    train_df["model_name"] = train_df["model_name"].astype(str)
    train_summary = (
        train_df.groupby(["model_name", "split_strategy", "n_jobs"])["seconds"]
        .agg(mean_seconds="mean", std_seconds="std")
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(
        build_df["n_jobs"],
        build_df["mean_seconds"],
        marker="o",
        linewidth=2.4,
        color=PALETTE["build_dataset"],
    )
    ax.fill_between(
        build_df["n_jobs"],
        build_df["mean_seconds"] - build_df["std_seconds"].fillna(0.0),
        build_df["mean_seconds"] + build_df["std_seconds"].fillna(0.0),
        color=PALETTE["build_dataset"],
        alpha=0.15,
    )
    speedup = build_df.loc[build_df["n_jobs"] == 1, "mean_seconds"].iloc[0] / build_df.loc[
        build_df["n_jobs"] == 8, "mean_seconds"
    ].iloc[0]
    ax.set_title(f"Preprocessing\n{speedup:.1f}x faster at 8 jobs")
    ax.set_xlabel("n_jobs")
    ax.set_ylabel("Wall time (s)")
    ax.set_xticks([1, 2, 4, 8])

    for ax, model_name in zip(axes[1:], ["random_forest", "xgboost"]):
        model_df = train_summary[train_summary["model_name"] == model_name]
        for split in SPLIT_ORDER:
            split_df = model_df[model_df["split_strategy"] == split]
            ax.plot(
                split_df["n_jobs"],
                split_df["mean_seconds"],
                marker="o",
                linewidth=2.2,
                color=PALETTE[split],
                label=SPLIT_LABELS[split],
            )
            ax.fill_between(
                split_df["n_jobs"],
                split_df["mean_seconds"] - split_df["std_seconds"].fillna(0.0),
                split_df["mean_seconds"] + split_df["std_seconds"].fillna(0.0),
                color=PALETTE[split],
                alpha=0.12,
            )
        ax.set_title(MODEL_LABELS[model_name])
        ax.set_xlabel("n_jobs")
        ax.set_xticks([1, 2, 4, 8])
        ax.set_ylabel("Wall time (s)")

    axes[2].legend(loc="upper right")
    fig.suptitle("Parallel Scaling on Markov CPU Nodes", y=1.03)
    save_figure(fig, output_dir, "04_benchmark_scaling")


def plot_feature_importance(summary_df: pd.DataFrame, runs_dir: Path, output_dir: Path) -> None:
    best_row = summary_df.sort_values(["r2", "rmse"], ascending=[False, True]).iloc[0]
    run_dir = runs_dir / best_row["run_name"]
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with (run_dir / "model.pkl").open("rb") as handle:
            model = pickle.load(handle)

    importance = extract_feature_importance(model, config["feature_columns"]).head(12).copy()
    importance = importance.iloc[::-1]
    importance["feature_label"] = importance["feature"].map(prettify_feature_name)

    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    ax.barh(
        importance["feature_label"],
        importance["importance"],
        color=PALETTE["random_forest"],
        alpha=0.92,
    )
    ax.set_xlabel("Importance")
    ax.set_title(
        "Top Features in the Best-Performing Run\n"
        f"{MODEL_LABELS[best_row['model']]} on {target_label(best_row['target'])} ({SPLIT_LABELS[best_row['split']]})"
    )
    ax.text(
        0.98,
        0.03,
        f"RMSE {best_row['rmse']:.2f}   MAE {best_row['mae']:.2f}   $R^2$ {best_row['r2']:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#dddddd"},
    )
    save_figure(fig, output_dir, "05_feature_importance_best_run")
def plot_xgboost_feature_importance(summary_df: pd.DataFrame, runs_dir: Path, output_dir: Path) -> None:
    # Filter only XGBoost runs
    xgb_df = summary_df[summary_df["model"] == "xgboost"].copy()

    # Pick best XGBoost run
    best_row = xgb_df.sort_values(["r2", "rmse"], ascending=[False, True]).iloc[0]

    run_dir = runs_dir / best_row["run_name"]
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with (run_dir / "model.pkl").open("rb") as handle:
            model = pickle.load(handle)

    importance = extract_feature_importance(model, config["feature_columns"]).head(12).copy()
    importance = importance.iloc[::-1]
    importance["feature_label"] = importance["feature"].map(prettify_feature_name)

    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    ax.barh(
        importance["feature_label"],
        importance["importance"],
        color=PALETTE["xgboost"],
        alpha=0.92,
    )

    ax.set_xlabel("Importance")
    ax.set_title(
        "Top Features in XGBoost Run\n"
        f"{target_label(best_row['target'])} ({SPLIT_LABELS[best_row['split']]})"
    )

    ax.text(
        0.98,
        0.03,
        f"RMSE {best_row['rmse']:.2f}   MAE {best_row['mae']:.2f}   $R^2$ {best_row['r2']:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )

    save_figure(fig, output_dir, "06_xgboost_feature_importance")

def write_manifest(summary_df: pd.DataFrame, benchmarks_dir: Path, output_dir: Path) -> None:
    best = (
        summary_df.sort_values(["target", "split", "r2", "rmse"], ascending=[True, True, False, True])
        .groupby(["target", "split"], as_index=False)
        .first()
    )
    best_lines = []
    for _, row in best.iterrows():
        best_lines.append(
            f"{target_label(row['target'])} / {SPLIT_LABELS[row['split']]}: "
            f"{MODEL_LABELS[row['model']]} (R^2={row['r2']:.3f}, RMSE={row['rmse']:.3f})"
        )

    benchmark_files = latest_benchmark_files(benchmarks_dir)
    lines = [
        "# Figure Guide",
        "",
        "Generated by `scripts/generate_report_figures.py`.",
        "",
        "## Suggested Slide Figures",
        "",
        "- `01_r2_heatmaps.png`: use for the main model-comparison slide.",
        "- `02_best_r2_by_target_split.png`: use for the key takeaway slide about which split/model wins.",
        "- `03_prediction_scatter_best_runs.png`: use for the qualitative fit slide.",
        "- `04_benchmark_scaling.png`: use for the HPC / parallelism slide.",
        "- `05_feature_importance_best_run.png`: use for interpretation and discussion.",
        "",
        "## Best Runs",
        "",
        *best_lines,
        "",
        "## Benchmark Files Used",
        "",
    ]
    for split, path in benchmark_files.items():
        lines.append(f"- {SPLIT_LABELS[split]}: `{path.name}`")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    summary_path = Path(args.summary)
    runs_dir = Path(args.runs_dir)
    benchmarks_dir = Path(args.benchmarks_dir)
    output_dir = Path(args.output_dir)

    summary_df = pd.read_csv(summary_path)

    plot_r2_heatmaps(summary_df, output_dir)
    plot_best_r2_by_target(summary_df, output_dir)
    plot_prediction_scatter(summary_df, runs_dir, output_dir)
    plot_benchmark_scaling(benchmarks_dir, output_dir)
    plot_feature_importance(summary_df, runs_dir, output_dir)
    plot_xgboost_feature_importance(summary_df, runs_dir, output_dir)
    write_manifest(summary_df, benchmarks_dir, output_dir)

    print(f"Saved figures to: {output_dir}")
    for path in sorted(output_dir.glob("*")):
        if path.is_file():
            print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
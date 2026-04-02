from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataSplit:
    train_index: tuple[int, ...]
    validation_index: tuple[int, ...]
    strategy: str
    random_seed: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["train_index"] = list(self.train_index)
        payload["validation_index"] = list(self.validation_index)
        return payload


def _unique_ordered(values: pd.Series) -> list[Any]:
    seen = []
    seen_set = set()
    for value in values.tolist():
        if pd.isna(value) or value in seen_set:
            continue
        seen.append(value)
        seen_set.add(value)
    return seen


def make_patient_level_split(
    df: pd.DataFrame,
    group_column: str = "patient_id",
    validation_fraction: float = 0.2,
    random_seed: int = 42,
) -> DataSplit:
    if group_column not in df.columns:
        raise ValueError(f"Missing group column: {group_column}")
    groups = np.array(_unique_ordered(df[group_column]))
    if len(groups) == 0:
        raise ValueError("Cannot split an empty dataframe")

    rng = np.random.default_rng(random_seed)
    rng.shuffle(groups)

    validation_size = max(1, int(round(len(groups) * validation_fraction)))
    validation_groups = set(groups[:validation_size].tolist())
    validation_mask = df[group_column].isin(validation_groups)

    train_index = tuple(df.index[~validation_mask].tolist())
    validation_index = tuple(df.index[validation_mask].tolist())
    return DataSplit(
        train_index=train_index,
        validation_index=validation_index,
        strategy="patient",
        random_seed=random_seed,
        metadata={
            "group_column": group_column,
            "validation_fraction": validation_fraction,
            "train_group_count": len(set(df.loc[list(train_index), group_column])),
            "validation_group_count": len(validation_groups),
        },
    )


def make_time_aware_split(
    df: pd.DataFrame,
    group_column: str = "patient_id",
    time_column: str = "visit_month",
    validation_fraction: float = 0.2,
    random_seed: int = 42,
) -> DataSplit:
    if group_column not in df.columns:
        raise ValueError(f"Missing group column: {group_column}")
    if time_column not in df.columns:
        raise ValueError(f"Missing time column: {time_column}")

    train_index: list[int] = []
    validation_index: list[int] = []

    ordered = df.reset_index().rename(columns={"index": "_row_index"})
    ordered = ordered.sort_values([group_column, time_column, "_row_index"], kind="mergesort")
    for _, group_df in ordered.groupby(group_column, sort=False):
        if len(group_df) < 2:
            train_index.extend(group_df["_row_index"].tolist())
            continue
        validation_count = max(1, int(round(len(group_df) * validation_fraction)))
        validation_rows = group_df.tail(validation_count)
        train_rows = group_df.iloc[: len(group_df) - validation_count]
        train_index.extend(train_rows["_row_index"].tolist())
        validation_index.extend(validation_rows["_row_index"].tolist())

    return DataSplit(
        train_index=tuple(train_index),
        validation_index=tuple(validation_index),
        strategy="time_aware",
        random_seed=random_seed,
        metadata={
            "group_column": group_column,
            "time_column": time_column,
            "validation_fraction": validation_fraction,
        },
    )


def save_split_metadata(split: DataSplit, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(str(split.to_dict()), encoding="utf-8")
    return output_path

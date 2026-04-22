"""Shared helpers for patient-level parallel preprocessing."""

from __future__ import annotations

from typing import Callable

import pandas as pd
from joblib import Parallel, delayed

GroupFunc = Callable[[pd.DataFrame], pd.DataFrame]


def parallel_groupby_apply(
    df: pd.DataFrame,
    group_col: str,
    func: GroupFunc,
    n_jobs: int = 1,
    backend: str = "loky",
) -> pd.DataFrame:

    if group_col not in df.columns:
        raise KeyError(f"Group column not found: {group_col}")

    groups = [group for _, group in df.groupby(group_col, sort=True)]
    if not groups:
        return df.iloc[0:0].copy()

    if n_jobs is None or n_jobs <= 1:
        parts = [func(group) for group in groups]
    else:
        parts = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(func)(group) for group in groups
        )

    return pd.concat(parts, axis=0)

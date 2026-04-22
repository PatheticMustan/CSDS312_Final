"""Shared helpers for patient-level parallel preprocessing (Phase 5).

The current pipeline does most of its heavy per-patient work through
``pandas.DataFrame.groupby`` with Python-level lambdas (rolling windows, lag /
delta features, etc.). Those paths are single-threaded even though the work is
embarrassingly parallel across patients. This module exposes a small primitive
that other modules (``feature_engineering``) can use to opt into process-based
parallelism via joblib without each call site reimplementing the pattern.
"""

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
    """Apply ``func`` to each ``group_col`` group, optionally in parallel.

    Parameters
    ----------
    df:
        Input frame. Must contain ``group_col``.
    group_col:
        Column used to partition the frame before applying ``func``.
    func:
        Per-group transformation returning a DataFrame. It must be a module-
        level, picklable function (closures/lambdas break the loky backend).
    n_jobs:
        Worker count. ``<= 1`` runs serially, skipping joblib overhead. This
        keeps the serial and parallel code paths identical on small datasets
        where worker spin-up would dominate.
    backend:
        joblib backend. Defaults to ``"loky"`` (process-based) because pandas
        and Python-level per-group work are GIL-bound; a thread backend would
        not actually parallelise the heavy parts.

    Returns
    -------
    pd.DataFrame
        Concatenation of the per-group outputs, with group keys processed in
        ascending order so serial and parallel runs yield byte-identical rows.
        The original index from each group is preserved.
    """
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

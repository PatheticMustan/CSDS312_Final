"""Parkinson's progression prediction scaffolding.

Keep package import light-weight so preprocessing-only workflows remain usable
even when the full model stack is not installed yet.
"""

from .config import BaselineRunConfig
from .io import load_amp_pd_raw_tables, load_table_by_name, validate_required_columns
from .parallel import parallel_groupby_apply
from .splits import DataSplit, make_patient_level_split, make_time_aware_split

__all__ = [
    "BaselineRunConfig",
    "DataSplit",
    "build_model",
    "compute_regression_metrics",
    "load_amp_pd_raw_tables",
    "load_table_by_name",
    "make_patient_level_split",
    "make_time_aware_split",
    "parallel_groupby_apply",
    "validate_required_columns",
]


def __getattr__(name: str):
    if name == "compute_regression_metrics":
        from .metrics import compute_regression_metrics

        return compute_regression_metrics
    if name == "build_model":
        from .models import build_model

        return build_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

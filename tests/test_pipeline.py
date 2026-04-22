from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Keep the test runnable both under pytest and as a standalone file in an IDE.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pd_progression.config import BaselineRunConfig
from pd_progression.pipeline import run_training_pipeline


@pytest.mark.parametrize("split_strategy", ["patient", "time_aware"])
def test_run_training_pipeline_persists_resolved_config_and_artifacts(tmp_path: Path, split_strategy: str) -> None:
    df = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2],
            "visit_id": [101, 102, 201, 202],
            "visit_month": [0, 6, 0, 6],
            "feature_a": [0.2, 0.4, 0.6, 0.8],
            "feature_b": [1.0, 1.2, 1.4, 1.6],
            "updrs_1": [10.0, 11.0, 12.0, 13.0],
        }
    )

    config = BaselineRunConfig(
        model_name="ridge",
        target_column="updrs_1",
        split_strategy=split_strategy,
        random_seed=7,
        validation_fraction=0.5,
        data_path=tmp_path / "input.csv",
        output_dir=tmp_path / "runs",
        model_params={"alpha": 1.5},
    )

    result = run_training_pipeline(df, config)

    assert result.feature_columns == ["feature_a", "feature_b"]
    assert not result.predictions.empty
    assert result.run_dir.exists()
    assert (result.run_dir / "config.json").exists()
    assert (result.run_dir / "metrics.json").exists()
    assert (result.run_dir / "predictions.csv").exists()
    assert (result.run_dir / "model.pkl").exists()

    saved_config = json.loads((result.run_dir / "config.json").read_text(encoding="utf-8"))
    assert saved_config["feature_columns"] == ["feature_a", "feature_b"]
    assert saved_config["model_name"] == "ridge"
    assert saved_config["model_params"] == {"alpha": 1.5}

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

# Keep the test runnable both under pytest and as a standalone file in an IDE.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pd_progression.cli import load_config


def test_load_config_merges_config_file_and_cli_model_params(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name": "random_forest",
                "target_column": "updrs_1",
                "data_path": "data/cleaned/final_dataset.csv",
                "output_dir": "results/base",
                "split_strategy": "time_aware",
                "random_seed": 17,
                "model_params": {"n_estimators": 50},
                "feature_columns": ["feature_a", "feature_b"],
            }
        ),
        encoding="utf-8",
    )

    args = Namespace(
        config=str(config_path),
        data=None,
        model=None,
        target=None,
        split=None,
        seed=None,
        model_params='{"max_depth": 4, "n_estimators": 75}',
        output_dir=None,
        n_jobs=None,
    )

    config = load_config(args)

    assert config.model_name == "random_forest"
    assert config.target_column == "updrs_1"
    assert config.data_path == Path("data/cleaned/final_dataset.csv")
    assert config.output_dir == Path("results/base")
    assert config.split_strategy == "time_aware"
    assert config.random_seed == 17
    assert config.feature_columns == ("feature_a", "feature_b")
    assert config.model_params == {"n_estimators": 75, "max_depth": 4}
    assert config.n_jobs == 1


def test_load_config_picks_up_n_jobs_from_cli(tmp_path: Path) -> None:
    args = Namespace(
        config=None,
        data="data/cleaned/final_dataset.csv",
        model="random_forest",
        target="updrs_1",
        split=None,
        seed=None,
        model_params=None,
        output_dir=None,
        n_jobs=8,
    )

    config = load_config(args)

    assert config.n_jobs == 8
    assert config.model_name == "random_forest"

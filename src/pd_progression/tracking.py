from __future__ import annotations

import json
import re
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("_") or "unknown"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


@dataclass
class ExperimentTracker:
    base_dir: Path
    run_name: str

    @classmethod
    def create(
        cls,
        base_dir: str | Path,
        model_name: str,
        target_column: str,
        split_strategy: str,
        random_seed: int,
    ) -> "ExperimentTracker":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_name = (
            f"{timestamp}_{_slugify(model_name)}_{_slugify(target_column)}_"
            f"{_slugify(split_strategy)}_seed{random_seed}"
        )
        return cls(Path(base_dir), run_name)

    @property
    def run_dir(self) -> Path:
        return self.base_dir / "runs" / self.run_name

    def ensure(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def save_json(self, filename: str, payload: dict[str, Any]) -> Path:
        self.ensure()
        output = self.run_dir / filename
        output.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return output

    def save_metrics(self, metrics: dict[str, float]) -> Path:
        return self.save_json("metrics.json", metrics)

    def save_predictions(self, predictions: pd.DataFrame) -> Path:
        self.ensure()
        output = self.run_dir / "predictions.csv"
        predictions.to_csv(output, index=False)
        return output

    def save_model(self, model: Any) -> Path:
        self.ensure()
        output = self.run_dir / "model.pkl"
        with output.open("wb") as handle:
            pickle.dump(model, handle)
        return output

    def save_config(self, config: dict[str, Any]) -> Path:
        return self.save_json("config.json", config)

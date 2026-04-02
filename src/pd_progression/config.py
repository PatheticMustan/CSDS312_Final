from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class BaselineRunConfig:
    model_name: str
    target_column: str
    split_strategy: str = "patient"
    random_seed: int = 42
    validation_fraction: float = 0.2
    data_path: Path = Path("data/cleaned")
    output_dir: Path = Path("results")
    feature_columns: tuple[str, ...] = ()
    id_columns: tuple[str, ...] = ("patient_id", "visit_id", "visit_month")
    group_column: str = "patient_id"
    time_column: str = "visit_month"
    model_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BaselineRunConfig":
        values = dict(payload)
        if "data_path" in values and values["data_path"] is not None:
            values["data_path"] = Path(values["data_path"])
        if "output_dir" in values and values["output_dir"] is not None:
            values["output_dir"] = Path(values["output_dir"])
        if "feature_columns" in values and values["feature_columns"] is not None:
            values["feature_columns"] = tuple(values["feature_columns"])
        if "id_columns" in values and values["id_columns"] is not None:
            values["id_columns"] = tuple(values["id_columns"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_path"] = str(self.data_path)
        payload["output_dir"] = str(self.output_dir)
        payload["feature_columns"] = list(self.feature_columns)
        payload["id_columns"] = list(self.id_columns)
        return payload


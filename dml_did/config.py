"""Study configuration: one YAML file per empirical application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

VALID_APPROACHES = ("wg_approx", "fd_exact", "cre_general", "cre_normal")


@dataclass
class StudyConfig:
    name: str
    data_path: Path
    y_col: str
    d_col: str
    id_col: str
    time_col: str
    controls: list[str]
    description: str = ""
    learners: list[str] = field(
        default_factory=lambda: [
            "random_forest",
            "neural_network",
            "gradient_boosting",
            "elastic_net",
            "ridge",
        ]
    )
    approach: str = "wg_approx"
    n_folds: int = 5
    n_rep: int = 1
    seed: int = 123
    twfe_benchmark: bool = True
    true_effect: float | None = None  # only known for synthetic data; drawn on the plot

    def __post_init__(self):
        self.data_path = Path(self.data_path)
        if not self.data_path.is_absolute():
            self.data_path = REPO_ROOT / self.data_path
        if self.approach not in VALID_APPROACHES:
            raise ValueError(f"approach must be one of {VALID_APPROACHES}, got {self.approach!r}")


def load_config(path: str | Path) -> StudyConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return StudyConfig(**raw)

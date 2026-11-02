"""Double/debiased machine learning for difference-in-differences on firm-year panels."""

from .config import StudyConfig, load_config
from .data import build_dml_data, load_panel, prepare_panel
from .estimate import run_study
from .learners import LEARNERS

__all__ = [
    "LEARNERS",
    "StudyConfig",
    "build_dml_data",
    "load_config",
    "load_panel",
    "prepare_panel",
    "run_study",
]

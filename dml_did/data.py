"""Load, clean and validate a firm-year panel, then wrap it for DoubleML."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from doubleml import DoubleMLPanelData

from .config import StudyConfig


def load_panel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".dta":
        return pd.read_stata(path, convert_categoricals=False)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suffix} (use .dta, .csv or .parquet)")


def prepare_panel(df: pd.DataFrame, cfg: StudyConfig) -> tuple[pd.DataFrame, list[str]]:
    """Return the cleaned panel and the covariate list (controls + year dummies).

    Year dummies enter X so that, after the within-firm transformation done by
    DoubleMLPLPR, the model absorbs both firm and year fixed effects.
    """
    cols = [cfg.y_col, cfg.d_col, cfg.id_col, cfg.time_col] + cfg.controls
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in {cfg.data_path.name}: {missing}")

    data = df[cols].copy()
    for col in [cfg.y_col, cfg.d_col, cfg.time_col] + cfg.controls:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # DoubleML needs a numeric unit id; map string ids (e.g. stock codes) to integers.
    if not pd.api.types.is_numeric_dtype(data[cfg.id_col]):
        data[cfg.id_col] = pd.factorize(data[cfg.id_col])[0] + 1

    data = data.sort_values([cfg.id_col, cfg.time_col]).reset_index(drop=True)

    if data.duplicated([cfg.id_col, cfg.time_col]).any():
        raise ValueError(f"({cfg.id_col}, {cfg.time_col}) does not uniquely identify observations.")
    if not set(data[cfg.d_col].unique()).issubset({0, 1}):
        raise ValueError(f"Treatment {cfg.d_col!r} must be a 0/1 DID indicator (Treat x Post).")

    year_dummies = pd.get_dummies(
        data[cfg.time_col].astype("category"),
        prefix="year",
        drop_first=True,
        dtype=float,
    )
    data = pd.concat([data, year_dummies], axis=1)
    x_cols = cfg.controls + year_dummies.columns.tolist()
    return data, x_cols


def build_dml_data(data: pd.DataFrame, x_cols: list[str], cfg: StudyConfig) -> DoubleMLPanelData:
    # static_panel=True also sets the unit id as the cluster variable, so
    # cross-fitting folds are split by firm and standard errors are firm-clustered.
    return DoubleMLPanelData(
        data=data,
        y_col=cfg.y_col,
        d_cols=cfg.d_col,
        t_col=cfg.time_col,
        id_col=cfg.id_col,
        x_cols=x_cols,
        static_panel=True,
    )


def describe_panel(data: pd.DataFrame, cfg: StudyConfig) -> dict:
    return {
        "observations": len(data),
        "firms": data[cfg.id_col].nunique(),
        "years": f"{int(data[cfg.time_col].min())}-{int(data[cfg.time_col].max())}",
        "treated_share": round(float(data[cfg.d_col].mean()), 3),
    }

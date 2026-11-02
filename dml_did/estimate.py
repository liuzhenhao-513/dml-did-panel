"""Fit panel DML with each learner, plus a conventional TWFE benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from doubleml import DoubleMLPLPR

from .config import StudyConfig
from .data import build_dml_data, describe_panel, load_panel, prepare_panel
from .learners import LEARNERS


def _row(method: str, coef, se, t, p, lo, hi) -> dict:
    return {
        "method": method,
        "coef": float(coef),
        "se": float(se),
        "t": float(t),
        "p_value": float(p),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def fit_dml(dml_data, learner_key: str, cfg: StudyConfig) -> tuple[dict, DoubleMLPLPR]:
    label, factory = LEARNERS[learner_key]

    # DoubleML draws its sample splits from numpy's global RNG.
    np.random.seed(cfg.seed)
    model = DoubleMLPLPR(
        obj_dml_data=dml_data,
        ml_l=factory(cfg.seed),
        ml_m=factory(cfg.seed),
        n_folds=cfg.n_folds,
        n_rep=cfg.n_rep,
        score="partialling out",
        approach=cfg.approach,
    )
    model.fit()

    ci = model.confint(level=0.95)
    row = _row(
        f"DML - {label}",
        model.coef[0],
        model.se[0],
        model.t_stat[0],
        model.pval[0],
        ci.iloc[0, 0],
        ci.iloc[0, 1],
    )
    return row, model


def fit_twfe_ols(data: pd.DataFrame, x_cols: list[str], cfg: StudyConfig) -> dict:
    """Linear two-way FE regression, analogous to `xtreg y did X i.year, fe vce(cluster id)`.

    Firm effects are swept out by within-firm demeaning; year effects enter
    as dummies. Standard errors are clustered by firm.
    """
    cols = [cfg.y_col, cfg.d_col] + x_cols
    demeaned = data[cols] - data.groupby(cfg.id_col)[cols].transform("mean")

    fit = sm.OLS(demeaned[cfg.y_col], demeaned[[cfg.d_col] + x_cols]).fit(
        cov_type="cluster",
        cov_kwds={"groups": data[cfg.id_col]},
    )
    lo, hi = fit.conf_int().loc[cfg.d_col]
    return _row(
        "TWFE OLS (benchmark)",
        fit.params[cfg.d_col],
        fit.bse[cfg.d_col],
        fit.tvalues[cfg.d_col],
        fit.pvalues[cfg.d_col],
        lo,
        hi,
    )


def run_study(cfg: StudyConfig, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    raw = load_panel(cfg.data_path)
    data, x_cols = prepare_panel(raw, cfg)
    info = describe_panel(data, cfg)
    if verbose:
        print(f"[{cfg.name}] {info}")

    dml_data = build_dml_data(data, x_cols, cfg)

    rows = []
    for key in cfg.learners:
        row, model = fit_dml(dml_data, key, cfg)
        rows.append(row)
        if verbose:
            print(f"\n{'=' * 60}\n{row['method']}\n{'=' * 60}")
            print(model.summary)

    if cfg.twfe_benchmark:
        rows.append(fit_twfe_ols(data, x_cols, cfg))

    return pd.DataFrame(rows), info

import numpy as np
import pandas as pd
import pytest

from dml_did import StudyConfig, build_dml_data, prepare_panel
from dml_did.estimate import fit_dml, fit_twfe_ols
from dml_did.synthetic import make_did_panel

THETA = 0.5


def _cfg(**overrides) -> StudyConfig:
    params = dict(
        name="test",
        data_path="data/unused.csv",
        y_col="y",
        d_col="did",
        id_col="firm_id",
        time_col="year",
        controls=["x1", "x2", "x3", "x4", "x5"],
        learners=["ridge"],
    )
    params.update(overrides)
    return StudyConfig(**params)


@pytest.fixture(scope="module")
def panel():
    return make_did_panel(n_firms=400, theta=THETA, seed=1)


def test_prepare_panel_adds_year_dummies(panel):
    cfg = _cfg()
    data, x_cols = prepare_panel(panel, cfg)
    n_years = panel["year"].nunique()
    assert len(x_cols) == len(cfg.controls) + n_years - 1
    assert data[x_cols].notna().all().all()


def test_prepare_panel_maps_string_ids(panel):
    df = panel.assign(firm_id="F" + panel["firm_id"].astype(str))
    data, _ = prepare_panel(df, _cfg())
    assert pd.api.types.is_numeric_dtype(data["firm_id"])
    assert data["firm_id"].nunique() == panel["firm_id"].nunique()


def test_prepare_panel_rejects_duplicates(panel):
    df = pd.concat([panel, panel.head(1)])
    with pytest.raises(ValueError, match="uniquely identify"):
        prepare_panel(df, _cfg())


def test_prepare_panel_rejects_non_binary_treatment(panel):
    df = panel.assign(did=panel["did"] * 2)
    with pytest.raises(ValueError, match="0/1"):
        prepare_panel(df, _cfg())


def test_dml_recovers_true_effect(panel):
    cfg = _cfg()
    data, x_cols = prepare_panel(panel, cfg)
    row, _ = fit_dml(build_dml_data(data, x_cols, cfg), "ridge", cfg)
    assert row["ci_low"] < THETA < row["ci_high"]
    assert abs(row["coef"] - THETA) < 3 * row["se"]


def test_dml_is_reproducible(panel):
    cfg = _cfg()
    data, x_cols = prepare_panel(panel, cfg)
    dml_data = build_dml_data(data, x_cols, cfg)
    a, _ = fit_dml(dml_data, "ridge", cfg)
    b, _ = fit_dml(dml_data, "ridge", cfg)
    assert np.isclose(a["coef"], b["coef"])


def test_twfe_benchmark_recovers_true_effect(panel):
    cfg = _cfg()
    data, x_cols = prepare_panel(panel, cfg)
    row = fit_twfe_ols(data, x_cols, cfg)
    assert row["ci_low"] < THETA < row["ci_high"]

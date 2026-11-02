"""Synthetic staggered-adoption DID panel with a known treatment effect.

Design (theta is the true effect of `did` on `y`):

* firms with a high latent characteristic z are more likely to be treated,
  and adopt in a random year between the 4th and 9th period;
* covariate x1 = z * trend, so treated firms drift differently over time -
  parallel trends hold only *conditional on* x1;
* the outcome depends on x1, x2, x3 non-linearly, plus firm and year effects.

This is a sanity check for the pipeline: every estimator's 95% CI should
usually cover theta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_did_panel(
    n_firms: int = 2000,
    n_years: int = 12,
    theta: float = 0.5,
    first_year: int = 2010,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_firms * n_years
    firm = np.repeat(np.arange(n_firms), n_years)
    t = np.tile(np.arange(n_years), n_firms)

    alpha = rng.normal(size=n_firms)  # firm fixed effect
    z = rng.normal(size=n_firms)  # latent driver of selection into treatment

    treated = rng.random(n_firms) < 1 / (1 + np.exp(-1.5 * z))
    adopt = rng.integers(3, 9, size=n_firms)
    did = (treated[firm] & (t >= adopt[firm])).astype(float)

    x1 = z[firm] * t / (n_years - 1) + 0.5 * rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.uniform(-1, 1, size=n)
    x4 = rng.normal(size=n)  # irrelevant covariate
    x5 = rng.binomial(1, 0.4, size=n).astype(float)

    g = 0.8 * x1 + 0.5 * np.tanh(2 * x1) + 0.5 * np.sin(np.pi * x2) + 0.8 * x3 * (x2 > 0) + 0.3 * x5
    y = theta * did + g + alpha[firm] + 0.05 * t + rng.normal(size=n)

    return pd.DataFrame(
        {
            "firm_id": firm + 1,
            "year": first_year + t,
            "did": did,
            "y": y,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "x5": x5,
        }
    )

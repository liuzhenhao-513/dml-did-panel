"""Nuisance learners for E[Y|X] and E[D|X].

Each entry is a factory so every model gets fresh, unfitted estimators. Linear
and neural learners are wrapped with a scaler; tree ensembles do not need one.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def random_forest(seed: int = 123):
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
    )


def neural_network(seed: int = 123):
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=3000,
            early_stopping=True,
            random_state=seed,
        ),
    )


def gradient_boosting(seed: int = 123):
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=seed,
    )


def elastic_net(seed: int = 123):
    return make_pipeline(
        StandardScaler(),
        ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0],
            cv=5,
            max_iter=10000,
            n_jobs=-1,
            random_state=seed,
        ),
    )


def ridge(seed: int = 123):
    return make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]),
    )


LEARNERS = {
    "random_forest": ("Random Forest", random_forest),
    "neural_network": ("Neural Network", neural_network),
    "gradient_boosting": ("Gradient Boosting", gradient_boosting),
    "elastic_net": ("Elastic Net", elastic_net),
    "ridge": ("Ridge", ridge),
}

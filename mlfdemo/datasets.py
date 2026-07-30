"""Small teaching datasets - no downloads, no dependencies, no surprises.

Every function returns `(X, y)` where `X` is a list of rows (each a list of floats) and `y`
is a list of targets, which is the shape Assignments 1 and 2 use.
"""

from __future__ import annotations

import math
import random

# The twelve flats used from session 1 onwards. Columns: area_sqm, dist_metro_km,
# built_year. Target: monthly rent in euros. Also shipped as
# course-materials-f2026/datasets/housing-mini.csv.
_HOUSING = [
    (32, 0.3, 1965, 540), (45, 0.9, 1998, 510), (52, 0.4, 1972, 640),
    (60, 1.6, 2010, 545), (68, 0.7, 1930, 720), (75, 2.1, 1985, 620),
    (80, 1.1, 2015, 770), (95, 0.5, 1955, 860), (38, 1.8, 1978, 420),
    (55, 0.6, 2004, 640), (110, 1.4, 1926, 930), (48, 2.6, 1968, 400),
]

HOUSING_FEATURES = ("area_sqm", "dist_metro_km", "built_year")


def housing(features: tuple[str, ...] = ("area_sqm", "dist_metro_km")):
    """The twelve-flat rent extract.

    Args:
        features: which columns to return, in order. Any subset of `HOUSING_FEATURES`.

    Returns:
        (X, y) - X is a list of 12 rows, y is a list of 12 rents in euros.
    """
    unknown = [f for f in features if f not in HOUSING_FEATURES]
    if unknown:
        raise ValueError(f"unknown feature(s) {unknown}; choose from {HOUSING_FEATURES}")
    idx = [HOUSING_FEATURES.index(f) for f in features]
    X = [[float(row[i]) for i in idx] for row in _HOUSING]
    y = [float(row[-1]) for row in _HOUSING]
    return X, y


def classification(n: int = 200, seed: int = 2026, noise: float = 1.0):
    """A synthetic two-feature binary problem, generated from a logistic model.

    The true coefficients are (intercept 0.3, x1 1.2, x2 -1.0) - fit it and see how close
    you get, and how that depends on `n`.

    Args:
        n: number of rows.
        seed: passed to `random.Random`, so results are reproducible.
        noise: multiplies the label noise; 0 gives a (nearly) separable problem.

    Returns:
        (X, y) with y in {0, 1}.
    """
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        x1, x2 = rng.uniform(-2, 2), rng.uniform(-2, 2)
        score = (0.3 + 1.2 * x1 - 1.0 * x2) / max(noise, 1e-9)
        p = 1 / (1 + math.exp(-max(min(score, 500), -500)))
        X.append([x1, x2])
        y.append(1 if rng.random() < p else 0)
    return X, y


def train_test_split(X, y, test_fraction: float = 0.3, seed: int = 2026):
    """A reproducible random split. Decide it before you look at the target."""
    rng = random.Random(seed)
    order = list(range(len(y)))
    rng.shuffle(order)
    cut = len(y) - max(1, round(test_fraction * len(y)))
    tr, te = order[:cut], order[cut:]
    return ([X[i] for i in tr], [y[i] for i in tr],
            [X[i] for i in te], [y[i] for i in te])

"""Ridge regression on standardised features (the linear benchmark for the FNN)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class RidgeModel:
    alpha: float
    scaler: StandardScaler
    model: Ridge

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X.to_numpy()))


def fit_ridge(X: pd.DataFrame, y: pd.Series, alpha: float) -> RidgeModel:
    scaler = StandardScaler().fit(X.to_numpy())
    model = Ridge(alpha=alpha).fit(scaler.transform(X.to_numpy()), y.to_numpy())
    return RidgeModel(alpha=alpha, scaler=scaler, model=model)


def tune_ridge(X_tr, y_tr, X_val, y_val, alphas) -> tuple[float, pd.DataFrame]:
    """Validation MSE for each alpha; returns the best alpha and the full table."""
    rows = []
    for a in alphas:
        m = fit_ridge(X_tr, y_tr, a)
        rows.append({"alpha": a, "val_mse": float(np.mean((y_val.to_numpy() - m.predict(X_val)) ** 2))})
    table = pd.DataFrame(rows)
    best = float(table.loc[table["val_mse"].idxmin(), "alpha"])
    return best, table

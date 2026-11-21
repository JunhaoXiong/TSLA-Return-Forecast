"""Chronological train / validation / test split and train-only scaling.

No shuffling anywhere. Boundaries are by row count on the usable dataset:
train = first floor(0.7 n) rows, validation = next rows up to floor(0.8 n), test = the rest.

Boundary note: the last training row's target is the return on the first validation day.
That is not look-ahead: when the first validation forecast is made (at that day's close),
that return has already been observed. Because the horizon is one day, no embargo gap is
needed; a multi-day horizon would need a gap of horizon - 1 rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, part in (("train", self.train), ("val", self.val), ("test", self.test)):
            rows.append({"set": name, "start": part.index.min().date(), "end": part.index.max().date(),
                         "rows": len(part)})
        return pd.DataFrame(rows)


def chronological_split(data: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.1) -> Split:
    if not data.index.is_monotonic_increasing:
        raise ValueError("Data must be sorted by date before splitting.")
    n = len(data)
    a = int(np.floor(train_frac * n))
    b = int(np.floor((train_frac + val_frac) * n))
    if not 0 < a < b < n:
        raise ValueError(f"Split produced an empty set (n={n}, a={a}, b={b}).")
    return Split(train=data.iloc[:a], val=data.iloc[a:b], test=data.iloc[b:])


def fit_scaler(train_X: pd.DataFrame) -> StandardScaler:
    """Fit on the training rows only; apply the same transform to val/test."""
    return StandardScaler().fit(train_X.to_numpy())


def transform(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(scaler.transform(X.to_numpy()), index=X.index, columns=X.columns)

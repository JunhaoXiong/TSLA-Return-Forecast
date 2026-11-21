"""Parameter-free benchmarks every model must beat.

* ``zero``: forecast r_{t+1} = 0, i.e. the random walk C_{t+1} = C_t.
* ``train_mean``: forecast r_{t+1} = mean of the training targets (a constant drift).
  The mean is computed on the training set only and then frozen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def zero_forecast(part: pd.DataFrame) -> np.ndarray:
    return np.zeros(len(part))


def train_mean_forecast(train: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    return np.full(len(part), float(train["target"].mean()))

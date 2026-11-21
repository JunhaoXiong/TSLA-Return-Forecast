"""Sequence inputs for recurrent models.

The FNN sees 20 lagged returns flattened into one vector. The LSTM instead reads a window of
L days in time order, with a small set of features per day. Row t of the dataset (forecast
made at the close of day t) gets the window of days t-L+1 .. t, so the window ends on the
forecast day and never reaches past it.

A window for an early validation or test row reaches back into earlier sets. Those are
*features* of past days, known when the forecast is made, so this is not leakage; targets of
earlier rows are never part of an input window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import log_returns

DAILY_FEATURES = ("ret", "hl_range", "co_return", "log_vol_ratio20", "spy_ret", "qqq_ret",
                  "vix_level", "vix_change")


def daily_features(prices: pd.DataFrame, target: str = "TSLA", volume_window: int = 20) -> pd.DataFrame:
    """One row per day, eight features, each using data of that day and earlier only."""
    o, h, l, c, v = (prices[f"{target}_{f}"] for f in ("Open", "High", "Low", "Close", "Volume"))
    vx = prices["VIX_Close"]
    return pd.DataFrame({
        "ret": log_returns(c),
        "hl_range": (h - l) / c,
        "co_return": (c - o) / o,
        "log_vol_ratio20": np.log(v / v.rolling(volume_window, min_periods=volume_window).mean()),
        "spy_ret": log_returns(prices["SPY_Close"]),
        "qqq_ret": log_returns(prices["QQQ_Close"]),
        "vix_level": vx / 100.0,
        "vix_change": np.log(vx / vx.shift(1)),
    }, index=prices.index)


def window_positions(daily: pd.DataFrame, dates: pd.DatetimeIndex, L: int) -> np.ndarray:
    """Integer position of each date in ``daily``; -1 if its L-day window is incomplete."""
    pos = daily.index.get_indexer(dates)
    if (pos < 0).any():
        raise KeyError("Some dates are missing from the daily feature frame.")
    valid = ~daily.isna().any(axis=1).to_numpy()
    # first position from which every later row is valid
    first_valid = int(np.argmax(valid))
    if not valid[first_valid:].all():
        raise ValueError("Daily features have gaps after the warm-up period.")
    return np.where(pos - L + 1 >= first_valid, pos, -1)


def make_windows(values: np.ndarray, positions: np.ndarray, L: int) -> np.ndarray:
    """Stack windows values[p-L+1 : p+1] -> array (n, L, n_features)."""
    if (positions < L - 1).any():
        raise ValueError("Window would start before the first row.")
    idx = positions[:, None] + np.arange(-L + 1, 1)[None, :]
    return values[idx]

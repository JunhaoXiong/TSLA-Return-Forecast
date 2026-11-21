"""Feature and target construction.

Timing convention (the most important rule in this project)
------------------------------------------------------------
A row dated t is a forecast made **at the close of day t**.

* Every feature in row t uses only data observed on or before day t.
* The target in row t is the *next* day's log return, y_t = ln(C_{t+1} / C_t).
* The matching price forecast is C_t * exp(y_hat_t), a forecast of C_{t+1}.

All features are log returns or ratios, so they do not depend on the split-adjusted price
scale. Rolling windows use only past values (pandas ``rolling`` is trailing by default) and
nothing is centred or back-filled. ``tests/test_features.py`` checks this by perturbing the
future and asserting that earlier rows do not change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def build_features(prices: pd.DataFrame, target: str = "TSLA", market_closes: tuple[str, ...] = ("SPY", "QQQ"),
                   vix: str | None = "VIX", return_lags: int = 20, rv_windows: tuple[int, ...] = (5, 20),
                   volume_window: int = 20, market_return_lags: int = 5) -> pd.DataFrame:
    """Feature matrix indexed by date; rows lacking enough history are left as NaN (dropped later)."""
    o, h, l, c, v = (prices[f"{target}_{f}"] for f in ("Open", "High", "Low", "Close", "Volume"))
    r = log_returns(c)
    feats: dict[str, pd.Series] = {}

    # 1) TSLA own log returns: ret_lag1 = r_t (return ending today), ..., ret_lagK = r_{t-K+1}
    for k in range(1, return_lags + 1):
        feats[f"ret_lag{k}"] = r.shift(k - 1)

    # 2) Realized volatility, annualised: sqrt(252 * mean(r^2)) over the trailing window
    for w in rv_windows:
        feats[f"rv{w}"] = np.sqrt(252.0 * (r ** 2).rolling(w, min_periods=w).mean())

    # 3) Volume relative to its trailing average (log), so split rescaling of volume cancels
    feats[f"log_vol_ratio{volume_window}"] = np.log(v / v.rolling(volume_window, min_periods=volume_window).mean())

    # 4) Intraday shape of day t
    feats["hl_range"] = (h - l) / c
    feats["co_return"] = (c - o) / o

    # 5) Market returns
    for m in market_closes:
        rm = log_returns(prices[f"{m}_Close"])
        for k in range(1, market_return_lags + 1):
            feats[f"{m.lower()}_ret_lag{k}"] = rm.shift(k - 1)

    # 6) VIX level (in decimal) and its daily log change
    if vix is not None:
        vx = prices[f"{vix}_Close"]
        feats["vix_level"] = vx / 100.0
        feats["vix_change"] = np.log(vx / vx.shift(1))

    return pd.DataFrame(feats, index=prices.index)


def build_target(close: pd.Series) -> pd.Series:
    """y_t = ln(C_{t+1} / C_t); NaN on the last date (no next day)."""
    return np.log(close.shift(-1) / close).rename("target")


def build_dataset(prices: pd.DataFrame, cfg) -> pd.DataFrame:
    """Features + target + the current close (needed to turn return forecasts into prices).

    Only rows where every feature and the target exist are kept. With the default config the
    first usable date is the 21st trading day (20 lags need 21 closes) and the last is the
    second-to-last day.
    """
    market = tuple(t.replace("^", "") for t in cfg.market_tickers if t != "^VIX")
    vix = "VIX" if "^VIX" in cfg.market_tickers else None
    X = build_features(prices, target=cfg.target_ticker, market_closes=market, vix=vix,
                       return_lags=cfg.return_lags, rv_windows=cfg.rv_windows,
                       volume_window=cfg.volume_window, market_return_lags=cfg.market_return_lags)
    close = prices[f"{cfg.target_ticker}_Close"]
    data = X.assign(close_t=close, close_next=close.shift(-1), target=build_target(close))
    return data.dropna(how="any")


def feature_columns(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if c not in ("close_t", "close_next", "target")]

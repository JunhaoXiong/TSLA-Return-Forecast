"""Forecast evaluation metrics.

Return metrics (the ones that measure real predictive power)
* ``oos_r2_vs_zero`` = 1 - SSE(model) / SSE(zero forecast). Zero return is the random walk,
  so a positive value means the model beats "tomorrow's price = today's price".
* ``hit_rate`` = share of days where sign(forecast) == sign(actual). Days with a zero
  forecast are excluded; an all-zero forecast gives NaN.

Price metrics (kept only for comparison with the old notebook)
* price forecast = C_t * exp(y_hat); RMSE, MAE and R^2 against C_{t+1}.
  R^2 on price levels is high even for the random walk because prices are persistent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def oos_r2_vs_zero(y: np.ndarray, y_hat: np.ndarray) -> float:
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    return float(1.0 - np.sum((y - y_hat) ** 2) / np.sum(y ** 2))


def hit_rate(y: np.ndarray, y_hat: np.ndarray) -> float:
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    mask = y_hat != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(y[mask]) == np.sign(y_hat[mask])))


def r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    return float(1.0 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))


def evaluate(part: pd.DataFrame, y_hat: np.ndarray) -> dict:
    y = part["target"].to_numpy()
    price_hat = part["close_t"].to_numpy() * np.exp(np.asarray(y_hat, float))
    price = part["close_next"].to_numpy()
    err = price - price_hat
    return {
        "ret_rmse": float(np.sqrt(np.mean((y - y_hat) ** 2))),
        "ret_oos_r2_vs_zero": oos_r2_vs_zero(y, y_hat),
        "hit_rate": hit_rate(y, y_hat),
        "price_rmse": float(np.sqrt(np.mean(err ** 2))),
        "price_mae": float(np.mean(np.abs(err))),
        "price_r2": r2(price, price_hat),
    }

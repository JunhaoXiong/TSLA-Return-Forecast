"""Forecast-comparison tests.

Conventions: every test is one-sided with H1 = "the model is better than the benchmark",
so a small p-value is evidence FOR the model. The benchmark is the zero-return forecast
(random walk) unless stated otherwise.

* Diebold-Mariano (1995) on squared errors, with a Newey-West long-run variance and the
  Harvey-Leybourne-Newbold (1997) small-sample correction (t distribution, T-1 df).
* Clark-West (2007): the right test when the benchmark is nested in the model, as the zero
  forecast is (set every coefficient to 0). Plain DM is undersized in that case because the
  model pays an estimation-noise penalty even when it contains real signal.
* Pesaran-Timmermann (1992): is the share of correct signs above what independent
  guessing with the same up/down frequencies would give?
* Holm (1979) step-down adjustment for testing several models at once.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def newey_west_lrv(x: np.ndarray, lag: int) -> float:
    """Long-run variance of a series (Bartlett kernel). lag=0 gives the plain variance (ddof=0)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    v = float(np.dot(x, x) / n)
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        v += 2.0 * w * float(np.dot(x[k:], x[:-k]) / n)
    return v


def default_lag(n: int) -> int:
    return int(np.floor(n ** (1.0 / 3.0)))


def diebold_mariano(y, f_model, f_bench, lag: int | None = None, h: int = 1) -> dict:
    y, fm, fb = (np.asarray(a, float) for a in (y, f_model, f_bench))
    d = (y - fb) ** 2 - (y - fm) ** 2          # > 0 when the model has the smaller loss
    n = len(d)
    lag = default_lag(n) if lag is None else lag
    lrv = newey_west_lrv(d, lag)
    stat = float(np.mean(d) / np.sqrt(lrv / n)) if lrv > 0 else float("nan")
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat_hln = stat * hln
    return {"dm_stat": stat_hln, "dm_p": float(1 - stats.t.cdf(stat_hln, df=n - 1)),
            "mean_loss_diff": float(np.mean(d)), "lag": lag}


def clark_west(y, f_model, f_bench, lag: int | None = None) -> dict:
    y, fm, fb = (np.asarray(a, float) for a in (y, f_model, f_bench))
    f_adj = (y - fb) ** 2 - ((y - fm) ** 2 - (fb - fm) ** 2)
    n = len(f_adj)
    lag = default_lag(n) if lag is None else lag
    lrv = newey_west_lrv(f_adj, lag)
    stat = float(np.mean(f_adj) / np.sqrt(lrv / n)) if lrv > 0 else float("nan")
    return {"cw_stat": stat, "cw_p": float(1 - stats.norm.cdf(stat)), "lag": lag}


def pesaran_timmermann(y, f) -> dict:
    """Days with a zero forecast are dropped. Undefined (NaN) if the forecast never changes sign."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    m = f != 0
    y, f = y[m], f[m]
    n = len(y)
    if n == 0:
        return {"pt_stat": float("nan"), "pt_p": float("nan"), "hit_rate": float("nan"), "n": 0}
    up_y, up_f = (y > 0).astype(float), (f > 0).astype(float)
    p_hat = float(np.mean(up_y == up_f))
    py, px = float(up_y.mean()), float(up_f.mean())
    p_star = py * px + (1 - py) * (1 - px)
    var_p = p_star * (1 - p_star) / n
    var_pstar = ((2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n
                 + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    denom = var_p - var_pstar
    if px in (0.0, 1.0) or denom <= 0:
        return {"pt_stat": float("nan"), "pt_p": float("nan"), "hit_rate": p_hat, "n": n}
    stat = (p_hat - p_star) / np.sqrt(denom)
    return {"pt_stat": float(stat), "pt_p": float(1 - stats.norm.cdf(stat)), "hit_rate": p_hat, "n": n}


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (NaN entries are left as NaN)."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out

"""Data-quality checks, with the split-adjustment check as the main one.

If a data source forgot to adjust for a split, the split day would show a mechanical
return of 1/ratio - 1 (-80% for TSLA's 1:5 in 2020, -66.7% for 1:3 in 2022). We flag:

* any day whose simple return is beyond ``extreme_return_abs`` (default 40%), and
* any day within ``split_ratio_tolerance`` of a known split's mechanical return.

Flags are for manual review; ``check_known_splits`` is the hard assertion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    rows: int
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    n_missing: int
    non_positive_prices: int
    high_low_violations: int
    flagged_days: pd.DataFrame
    split_day_returns: pd.DataFrame
    largest_moves: pd.DataFrame

    @property
    def passed(self) -> bool:
        return (self.n_missing == 0 and self.non_positive_prices == 0
                and self.high_low_violations == 0 and self.flagged_days.empty
                and bool(self.split_day_returns["ok"].all()))


def simple_returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def flag_suspicious_returns(close: pd.Series, known_splits: dict[str, float],
                            extreme_abs: float, tol: float) -> pd.DataFrame:
    r = simple_returns(close).dropna()
    mechanical = [1.0 / ratio - 1.0 for ratio in known_splits.values()]
    near_split = np.zeros(len(r), dtype=bool)
    for m in mechanical:
        near_split |= (r - m).abs().to_numpy() < tol
    extreme = r.abs().to_numpy() > extreme_abs
    mask = near_split | extreme
    out = pd.DataFrame({"return": r[mask]})
    out["reason"] = np.where(extreme[mask], "extreme", "near split ratio")
    return out


def check_known_splits(close: pd.Series, known_splits: dict[str, float],
                       extreme_abs: float, tol: float) -> pd.DataFrame:
    """Return on each known split date; ``ok`` = looks like a normal (adjusted) day."""
    r = simple_returns(close)
    rows = []
    for date, ratio in known_splits.items():
        ts = pd.Timestamp(date)
        if ts not in r.index:
            rows.append({"date": date, "ratio": ratio, "return": np.nan, "ok": False})
            continue
        val = float(r.loc[ts])
        ok = abs(val) <= extreme_abs and abs(val - (1.0 / ratio - 1.0)) >= tol
        rows.append({"date": date, "ratio": ratio, "return": val, "ok": ok})
    return pd.DataFrame(rows)


def run_quality_checks(prices: pd.DataFrame, target: str, known_splits: dict[str, float],
                       extreme_abs: float, tol: float) -> QualityReport:
    close = prices[f"{target}_Close"]
    price_cols = [c for c in prices.columns if not c.endswith("_Volume")]
    hl = ((prices[f"{target}_High"] < prices[[f"{target}_Open", f"{target}_Close", f"{target}_Low"]].max(axis=1) - 1e-6)
          | (prices[f"{target}_Low"] > prices[[f"{target}_Open", f"{target}_Close", f"{target}_High"]].min(axis=1) + 1e-6))
    r = simple_returns(close).dropna()
    largest = r.reindex(r.abs().sort_values(ascending=False).index[:10]).to_frame("return")
    return QualityReport(
        rows=len(prices),
        first_date=prices.index.min(),
        last_date=prices.index.max(),
        n_missing=int(prices.isna().sum().sum()),
        non_positive_prices=int((prices[price_cols] <= 0).sum().sum()),
        high_low_violations=int(hl.sum()),
        flagged_days=flag_suspicious_returns(close, known_splits, extreme_abs, tol),
        split_day_returns=check_known_splits(close, known_splits, extreme_abs, tol),
        largest_moves=largest,
    )

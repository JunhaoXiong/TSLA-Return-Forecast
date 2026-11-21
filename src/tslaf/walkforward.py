"""Expanding-window walk-forward over the test period.

The test period is cut into blocks of ``step`` trading days. Before each block, every model
is refit on all dataset rows dated before the block's first day, then forecasts the block.
Hyperparameters (Ridge alpha, network shape, epoch count) are the ones chosen in Phases 2-3
using data up to 2022-11-22, before the test period starts, so reusing them here looks at
nothing from the future.

Timing check: the last fit row is dated t-1 and its target is the return on day t, the first
day of the block. The first forecast of the block is made at the close of day t, when that
return is already known. So the fit never uses a return the forecast could not have seen.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def blocks(index: pd.DatetimeIndex, step: int) -> list[tuple[int, int]]:
    """(start, stop) integer positions into ``index`` for consecutive blocks."""
    return [(s, min(s + step, len(index))) for s in range(0, len(index), step)]


def walk_forward(data: pd.DataFrame, test_index: pd.DatetimeIndex, step: int,
                 models: dict[str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]]) -> pd.DataFrame:
    """Return a frame of forecasts (one column per model) over ``test_index``.

    Each model is a function (fit_rows, predict_rows) -> forecasts. ``fit_rows`` are all rows of
    ``data`` dated strictly before the block's first date.
    """
    out = pd.DataFrame(index=test_index, columns=list(models), dtype=float)
    out["refit_block"] = -1
    for b, (s, e) in enumerate(blocks(test_index, step)):
        block_dates = test_index[s:e]
        fit_rows = data.loc[data.index < block_dates[0]]
        pred_rows = data.loc[block_dates]
        for name, fn in models.items():
            out.loc[block_dates, name] = fn(fit_rows, pred_rows)
        out.loc[block_dates, "refit_block"] = b
    return out

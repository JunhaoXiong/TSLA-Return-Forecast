import numpy as np
import pandas as pd

from tslaf.walkforward import blocks, walk_forward


def test_blocks_cover_index_without_overlap():
    idx = pd.bdate_range("2020-01-01", periods=130)
    b = blocks(idx, 63)
    assert b == [(0, 63), (63, 126), (126, 130)]


def test_fit_rows_strictly_before_block():
    idx = pd.bdate_range("2020-01-01", periods=300)
    data = pd.DataFrame({"target": np.arange(300.0)}, index=idx)
    test_idx = idx[200:]
    seen = []

    def spy(fit, pred):
        seen.append((fit.index.max(), pred.index.min(), len(fit)))
        return np.full(len(pred), fit["target"].mean())

    out = walk_forward(data, test_idx, 40, {"m": spy})
    for last_fit, first_pred, n in seen:
        assert last_fit < first_pred
    assert seen[0][2] == 200 and seen[1][2] == 240  # expanding window
    assert out["m"].notna().all() and out["refit_block"].max() == 2
    # the first block's forecast is the mean of rows 0..199
    assert np.isclose(out["m"].iloc[0], np.arange(200.0).mean())

import numpy as np
import pandas as pd

from tslaf.features import build_dataset, build_features, build_target, feature_columns


def test_return_lag_timing(prices):
    X = build_features(prices)
    c = prices["TSLA_Close"]
    t = 50
    assert np.isclose(X["ret_lag1"].iloc[t], np.log(c.iloc[t] / c.iloc[t - 1]))
    assert np.isclose(X["ret_lag5"].iloc[t], np.log(c.iloc[t - 4] / c.iloc[t - 5]))
    assert np.isclose(X["spy_ret_lag1"].iloc[t],
                      np.log(prices["SPY_Close"].iloc[t] / prices["SPY_Close"].iloc[t - 1]))


def test_target_is_next_day_return(prices):
    y = build_target(prices["TSLA_Close"])
    c = prices["TSLA_Close"]
    assert np.isclose(y.iloc[10], np.log(c.iloc[11] / c.iloc[10]))
    assert np.isnan(y.iloc[-1])


def test_rv_uses_trailing_window(prices):
    X = build_features(prices)
    r = np.log(prices["TSLA_Close"] / prices["TSLA_Close"].shift(1))
    t = 40
    expected = np.sqrt(252 * np.mean(r.iloc[t - 4:t + 1] ** 2))
    assert np.isclose(X["rv5"].iloc[t], expected)


def test_no_look_ahead(prices, cfg):
    """Changing data after day T must not change any feature on or before T,
    nor any target before T (target at T legitimately uses day T+1)."""
    base = build_dataset(prices, cfg)
    T = prices.index[200]
    shocked = prices.copy()
    after = shocked.index > T
    rng = np.random.default_rng(1)
    for c in shocked.columns:
        shocked.loc[after, c] *= rng.uniform(0.5, 1.5, after.sum())
    alt = build_dataset(shocked, cfg)
    cols = feature_columns(base)
    pd.testing.assert_frame_equal(base.loc[:T, cols], alt.loc[:T, cols])
    before = base.index < T
    np.testing.assert_allclose(base.loc[before, "target"], alt.loc[alt.index < T, "target"])


def test_scale_invariance(prices, cfg):
    """A constant price rescaling (what split adjustment does) leaves features and target unchanged."""
    scaled = prices.copy()
    for c in ("TSLA_Open", "TSLA_High", "TSLA_Low", "TSLA_Close"):
        scaled[c] /= 15.0
    scaled["TSLA_Volume"] *= 15.0
    a, b = build_dataset(prices, cfg), build_dataset(scaled, cfg)
    cols = feature_columns(a) + ["target"]
    pd.testing.assert_frame_equal(a[cols], b[cols], rtol=1e-9)


def test_dataset_drops_warmup_and_last_day(prices, cfg):
    data = build_dataset(prices, cfg)
    assert data.index[0] == prices.index[cfg.return_lags]
    assert data.index[-1] == prices.index[-2]
    assert not data.isna().any().any()
    assert len(feature_columns(data)) == 37

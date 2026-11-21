import numpy as np
import pandas as pd
import pytest

from tslaf.features import build_dataset
from tslaf.models.legacy import LegacyConfig, fit_legacy
from tslaf.models.lstm import LSTMConfig, config_grid, fit_lstm
from tslaf.sequences import daily_features, make_windows, window_positions
from tslaf.split import chronological_split

FAST = LSTMConfig(window=10, hidden=8, dropout=0.0, max_epochs=5, patience=3)


def test_daily_feature_timing(prices):
    d = daily_features(prices)
    c = prices["TSLA_Close"]
    assert np.isclose(d["ret"].iloc[30], np.log(c.iloc[30] / c.iloc[29]))
    assert list(d.columns) == ["ret", "hl_range", "co_return", "log_vol_ratio20", "spy_ret", "qqq_ret",
                               "vix_level", "vix_change"]


def test_window_ends_on_forecast_day(prices):
    d = daily_features(prices)
    vals = np.arange(len(d) * 2, dtype=float).reshape(len(d), 2)
    pos = np.array([30, 40])
    W = make_windows(vals, pos, 5)
    assert W.shape == (2, 5, 2)
    np.testing.assert_array_equal(W[0, -1], vals[30])  # last step = day t itself
    np.testing.assert_array_equal(W[0, 0], vals[26])
    with pytest.raises(ValueError):
        make_windows(vals, np.array([2]), 5)


def test_window_positions_mark_warmup(prices):
    d = daily_features(prices)
    pos = window_positions(d, d.index[[19, 20, 40]], 5)
    # vol ratio needs 20 rows, so the first valid day is position 19; a 5-day window needs position >= 23
    assert pos[0] == -1 and pos[1] == -1 and pos[2] == 40


def test_lstm_windows_do_not_use_future(prices, cfg):
    """Perturbing prices after day T must not change the window (hence the forecast) for day T."""
    data = build_dataset(prices, cfg)
    sp = chronological_split(data)
    m = fit_lstm(daily_features(prices), sp.train, FAST, seed=0, val_part=sp.val)
    T = sp.test.index[5]
    shocked = prices.copy()
    after = shocked.index > T
    shocked.loc[after] *= 1.7
    a = m.predict(daily_features(prices), sp.test.loc[:T])
    b = m.predict(daily_features(shocked), sp.test.loc[:T])
    np.testing.assert_allclose(a, b)


def test_lstm_scaler_uses_fit_period_only(prices, cfg):
    data = build_dataset(prices, cfg)
    sp = chronological_split(data)
    d = daily_features(prices)
    m = fit_lstm(d, sp.train, FAST, seed=0, val_part=sp.val)
    np.testing.assert_allclose(m.scaler.mean_, d.loc[:sp.train.index.max()].dropna().mean().to_numpy())


def test_lstm_deterministic_and_epochs(prices, cfg):
    data = build_dataset(prices, cfg)
    sp = chronological_split(data)
    d = daily_features(prices)
    a = fit_lstm(d, sp.train, FAST, seed=1, epochs=3)
    b = fit_lstm(d, sp.train, FAST, seed=1, epochs=3)
    assert a.epochs_trained == 3
    np.testing.assert_allclose(a.predict(d, sp.test), b.predict(d, sp.test))
    with pytest.raises(ValueError):
        fit_lstm(d, sp.train, FAST, seed=0)


def test_lstm_learns_sequence_signal():
    """Sanity check: the target is the sum of the last 3 values of one input; an LSTM can learn that."""
    rng = np.random.default_rng(0)
    n = 1400
    idx = pd.bdate_range("2010-01-01", periods=n)
    daily = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=idx)
    target = daily["a"].rolling(3).sum().shift(0)
    data = pd.DataFrame({"target": target}, index=idx).iloc[20:]
    tr, va, te = data.iloc[:900], data.iloc[900:1100], data.iloc[1100:]
    cfgl = LSTMConfig(window=10, hidden=16, dropout=0.0, max_epochs=60, patience=10, learning_rate=3e-3)
    m = fit_lstm(daily, tr, cfgl, seed=0, val_part=va)
    p = m.predict(daily, te)
    y = te["target"].to_numpy()
    assert 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2) > 0.8


def test_lstm_grid(cfg):
    assert len(config_grid(cfg.lstm)) == 8


def test_legacy_scaler_and_output(prices, cfg):
    data = build_dataset(prices, cfg)
    sp = chronological_split(data)
    close = prices["TSLA_Close"]
    lc = LegacyConfig(window=20, units=8, layers=2, max_epochs=3, patience=2)
    m = fit_legacy(close, sp.train, lc, seed=0, val_part=sp.val)
    assert np.isclose(m.scaler.data_max_[0], close.loc[:sp.train.index.max()].max())
    p = m.predict_price(close, sp.test)
    assert p.shape == (len(sp.test),) and np.all(np.isfinite(p))

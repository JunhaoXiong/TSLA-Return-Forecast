import numpy as np
import pandas as pd
import pytest


def make_prices(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV for TSLA plus SPY/QQQ/VIX closes on business days."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=n, name="Date")
    r = rng.normal(0.001, 0.03, n)
    close = 50 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.01, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, n))
    vol = rng.uniform(1e6, 5e6, n)
    spy = 200 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    qqq = 150 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n)))
    vix = 15 + 5 * np.abs(np.sin(np.arange(n) / 20)) + rng.uniform(0, 2, n)
    return pd.DataFrame({"TSLA_Open": open_, "TSLA_High": high, "TSLA_Low": low, "TSLA_Close": close,
                         "TSLA_Volume": vol, "SPY_Close": spy, "QQQ_Close": qqq, "VIX_Close": vix}, index=idx)


@pytest.fixture
def prices():
    return make_prices()


@pytest.fixture
def cfg():
    from tslaf.config import load_config
    return load_config()

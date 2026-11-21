import numpy as np
import pandas as pd

from tslaf.baselines import train_mean_forecast, zero_forecast
from tslaf.metrics import evaluate, hit_rate, oos_r2_vs_zero, r2


def test_oos_r2_reference_points():
    y = np.array([0.01, -0.02, 0.03])
    assert oos_r2_vs_zero(y, np.zeros(3)) == 0.0
    assert oos_r2_vs_zero(y, y) == 1.0
    assert oos_r2_vs_zero(y, -y) < 0


def test_hit_rate():
    y = np.array([0.01, -0.02, 0.03, -0.01])
    assert hit_rate(y, np.array([1, -1, -1, 0.0])) == 2 / 3
    assert np.isnan(hit_rate(y, np.zeros(4)))


def test_r2_mean_forecast_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert np.isclose(r2(y, np.full(3, 2.0)), 0.0)


def test_evaluate_price_conversion():
    part = pd.DataFrame({"target": [np.log(1.1), np.log(0.9)], "close_t": [100.0, 110.0],
                         "close_next": [110.0, 99.0]})
    perfect = evaluate(part, part["target"].to_numpy())
    assert np.isclose(perfect["price_rmse"], 0) and np.isclose(perfect["ret_oos_r2_vs_zero"], 1)
    rw = evaluate(part, zero_forecast(part))
    assert np.isclose(rw["price_mae"], (10 + 11) / 2)


def test_train_mean_is_frozen():
    train = pd.DataFrame({"target": [0.01, 0.03]})
    test = pd.DataFrame({"target": [5.0, 5.0, 5.0]})
    np.testing.assert_allclose(train_mean_forecast(train, test), 0.02)

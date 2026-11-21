import numpy as np
import pandas as pd
import pytest

from tslaf.models.fnn import FNNConfig, config_grid, fit_fnn, tune_fnn
from tslaf.models.ridge import fit_ridge, tune_ridge


def _linear_data(n=600, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    y = pd.Series(0.5 * X["x0"] - 0.3 * X["x1"] + noise * rng.normal(size=n))
    return X, y


SMALL = FNNConfig(hidden=(16, 8), dropout=0.0, weight_decay=0.0, max_epochs=200, patience=15)


def test_ridge_recovers_linear_signal():
    X, y = _linear_data()
    m = fit_ridge(X.iloc[:400], y.iloc[:400], alpha=1.0)
    pred = m.predict(X.iloc[400:])
    r2 = 1 - np.sum((y.iloc[400:] - pred) ** 2) / np.sum((y.iloc[400:] - y.iloc[400:].mean()) ** 2)
    assert r2 > 0.9


def test_ridge_tuning_picks_lowest_val_mse():
    X, y = _linear_data()
    best, table = tune_ridge(X.iloc[:400], y.iloc[:400], X.iloc[400:], y.iloc[400:], [0.01, 1, 1e6])
    assert best == table.loc[table["val_mse"].idxmin(), "alpha"]
    assert best != 1e6  # huge shrinkage destroys a real signal


def test_fnn_learns_nonlinear_signal():
    """Sanity check that the training loop works: y = x0^2 is learnable, a linear model cannot fit it."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(1500, 3)), columns=["a", "b", "c"])
    y = pd.Series(X["a"] ** 2 + 0.05 * rng.normal(size=1500))
    m = fit_fnn(X.iloc[:1000], y.iloc[:1000], SMALL, seed=0, X_val=X.iloc[1000:1250], y_val=y.iloc[1000:1250])
    pred = m.predict(X.iloc[1250:])
    yt = y.iloc[1250:]
    assert 1 - np.sum((yt - pred) ** 2) / np.sum((yt - yt.mean()) ** 2) > 0.8
    lin = fit_ridge(X.iloc[:1000], y.iloc[:1000], 1.0).predict(X.iloc[1250:])
    assert 1 - np.sum((yt - lin) ** 2) / np.sum((yt - yt.mean()) ** 2) < 0.1


def test_fnn_is_deterministic_per_seed():
    X, y = _linear_data(n=300)
    a = fit_fnn(X, y, SMALL, seed=3, epochs=5).predict(X)
    b = fit_fnn(X, y, SMALL, seed=3, epochs=5).predict(X)
    c = fit_fnn(X, y, SMALL, seed=4, epochs=5).predict(X)
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


def test_early_stopping_keeps_best_epoch_weights():
    X, y = _linear_data(n=500)
    m = fit_fnn(X.iloc[:350], y.iloc[:350], SMALL, seed=0, X_val=X.iloc[350:], y_val=y.iloc[350:])
    val_mse = float(np.mean((y.iloc[350:].to_numpy() - m.predict(X.iloc[350:])) ** 2))
    assert np.isclose(val_mse, min(m.history), rtol=1e-5)
    assert m.history[m.best_epoch - 1] == min(m.history)
    assert m.epochs_trained <= SMALL.max_epochs


def test_fnn_without_val_needs_epochs():
    X, y = _linear_data(n=100)
    with pytest.raises(ValueError):
        fit_fnn(X, y, SMALL, seed=0)
    m = fit_fnn(X, y, SMALL, seed=0, epochs=7)
    assert m.epochs_trained == 7


def test_fit_does_not_touch_prediction_rows():
    """The scaler and the weights depend only on the fit rows: changing the rows we
    predict on must not change the fitted model."""
    X, y = _linear_data(n=400)
    m1 = fit_fnn(X.iloc[:300], y.iloc[:300], SMALL, seed=0, epochs=5)
    X2 = X.copy()
    X2.iloc[300:] *= 100
    m2 = fit_fnn(X2.iloc[:300], y.iloc[:300], SMALL, seed=0, epochs=5)
    np.testing.assert_allclose(m1.scaler.mean_, m2.scaler.mean_)
    np.testing.assert_allclose(m1.predict(X.iloc[:300]), m2.predict(X.iloc[:300]))


def test_config_grid_and_tuning(cfg):
    grid = config_grid(cfg.fnn)
    assert len(grid) == 8
    X, y = _linear_data(n=300)
    small_grid = [FNNConfig(hidden=(8,), max_epochs=20, patience=5), FNNConfig(hidden=(16,), max_epochs=20, patience=5)]
    best, epochs, summary = tune_fnn(X.iloc[:200], y.iloc[:200], X.iloc[200:], y.iloc[200:], small_grid, [0, 1])
    assert len(summary) == 2 and 1 <= epochs <= 20
    assert best in small_grid

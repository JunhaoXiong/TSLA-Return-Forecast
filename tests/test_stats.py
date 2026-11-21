import numpy as np

from tslaf.stats import clark_west, diebold_mariano, holm, newey_west_lrv, pesaran_timmermann


def test_newey_west_lag0_is_variance():
    x = np.random.default_rng(0).normal(size=500)
    assert np.isclose(newey_west_lrv(x, 0), np.var(x))


def test_newey_west_positive_autocorrelation_increases_lrv():
    rng = np.random.default_rng(1)
    e = rng.normal(size=3000)
    x = np.convolve(e, np.ones(5), mode="valid")  # MA(4): strong positive autocorrelation
    assert newey_west_lrv(x, 8) > 2 * newey_west_lrv(x, 0)


def test_dm_detects_better_model_and_not_worse():
    rng = np.random.default_rng(2)
    n = 1000
    signal = rng.normal(size=n)
    y = 0.5 * signal + rng.normal(size=n)
    good = diebold_mariano(y, 0.5 * signal, np.zeros(n))
    assert good["dm_p"] < 0.01
    bad = diebold_mariano(y, -0.5 * signal, np.zeros(n))
    assert bad["dm_p"] > 0.99


def test_dm_identical_forecasts_not_significant():
    rng = np.random.default_rng(3)
    y = rng.normal(size=500)
    f = rng.normal(size=500) * 0.01
    r = diebold_mariano(y, f, f + 1e-12)
    assert abs(r["mean_loss_diff"]) < 1e-9


def test_clark_west_finds_weak_signal_that_dm_misses_more_often():
    """Weak true signal plus estimation noise: CW should reject at least as often as DM."""
    rng = np.random.default_rng(4)
    rej_cw = rej_dm = 0
    for _ in range(200):
        n = 800
        x = rng.normal(size=n)
        y = 0.08 * x + rng.normal(size=n)
        f = (0.08 + rng.normal(scale=0.06)) * x + rng.normal(scale=0.05, size=n)  # noisy estimate
        rej_cw += clark_west(y, f, np.zeros(n))["cw_p"] < 0.05
        rej_dm += diebold_mariano(y, f, np.zeros(n))["dm_p"] < 0.05
    assert rej_cw >= rej_dm


def test_clark_west_null_size_reasonable():
    rng = np.random.default_rng(5)
    rej = 0
    for _ in range(300):
        n = 500
        y = rng.normal(size=n)
        f = rng.normal(scale=0.05, size=n)  # pure noise forecast, nested benchmark is true
        rej += clark_west(y, f, np.zeros(n))["cw_p"] < 0.05
    assert rej / 300 < 0.12


def test_pesaran_timmermann():
    rng = np.random.default_rng(6)
    n = 1000
    s = rng.normal(size=n)
    y = s + rng.normal(size=n)
    assert pesaran_timmermann(y, s)["pt_p"] < 0.001
    assert pesaran_timmermann(y, rng.normal(size=n))["pt_p"] > 0.01
    const = pesaran_timmermann(y, np.full(n, 0.001))
    assert np.isnan(const["pt_stat"])  # never changes sign: test undefined
    assert np.isnan(pesaran_timmermann(y, np.zeros(n))["pt_p"])


def test_holm_known_values():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    out = holm([0.2, np.nan, 0.01])
    assert np.isnan(out[1]) and np.isclose(out[2], 0.02) and np.isclose(out[0], 0.2)

import numpy as np
import pytest

from tslaf.features import build_dataset, feature_columns
from tslaf.split import chronological_split, fit_scaler, transform


def test_sizes_order_and_no_overlap(prices, cfg):
    data = build_dataset(prices, cfg)
    sp = chronological_split(data, 0.7, 0.1)
    n = len(data)
    assert len(sp.train) == int(np.floor(0.7 * n))
    assert len(sp.train) + len(sp.val) == int(np.floor(0.8 * n))
    assert len(sp.train) + len(sp.val) + len(sp.test) == n
    assert sp.train.index.max() < sp.val.index.min()
    assert sp.val.index.max() < sp.test.index.min()
    # contiguous: nothing skipped or duplicated
    joined = sp.train.index.append(sp.val.index).append(sp.test.index)
    assert (joined == data.index).all()


def test_unsorted_rejected(prices, cfg):
    data = build_dataset(prices, cfg).iloc[::-1]
    with pytest.raises(ValueError):
        chronological_split(data)


def test_scaler_uses_train_only(prices, cfg):
    data = build_dataset(prices, cfg)
    sp = chronological_split(data)
    cols = feature_columns(data)
    sc = fit_scaler(sp.train[cols])
    np.testing.assert_allclose(sc.mean_, sp.train[cols].mean().to_numpy())
    z = transform(sc, sp.train[cols])
    np.testing.assert_allclose(z.mean().to_numpy(), 0, atol=1e-10)
    # refitting on a different val/test must not change the train-fitted scaler
    altered = data.copy()
    altered.loc[sp.test.index, cols] *= 3
    sc2 = fit_scaler(chronological_split(altered).train[cols])
    np.testing.assert_allclose(sc.mean_, sc2.mean_)

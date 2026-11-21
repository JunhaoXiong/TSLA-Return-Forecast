import pytest

from tslaf.data import clean_name, load_prices, save_prices


def test_roundtrip_and_hash(tmp_path, prices):
    path = tmp_path / "p.csv"
    meta = save_prices(prices, path, {"source": "test"})
    assert meta["rows"] == len(prices)
    back = load_prices(path)
    assert list(back.columns) == list(prices.columns)
    assert (back.index == prices.index).all()
    assert abs(back["TSLA_Close"] - prices["TSLA_Close"]).max() < 1e-5


def test_hash_mismatch_raises(tmp_path, prices):
    path = tmp_path / "p.csv"
    save_prices(prices, path, {"source": "test"})
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_prices(path)


def test_clean_name():
    assert clean_name("^VIX") == "VIX"
    assert clean_name("SPY") == "SPY"

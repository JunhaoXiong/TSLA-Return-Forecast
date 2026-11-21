import pandas as pd

from tslaf.quality import check_known_splits, flag_suspicious_returns, run_quality_checks


def _unadjust(prices, date, ratio):
    """Simulate a data source that forgot a split: prices before `date` are ratio x higher."""
    p = prices.copy()
    before = p.index < pd.Timestamp(date)
    for c in ("TSLA_Open", "TSLA_High", "TSLA_Low", "TSLA_Close"):
        p.loc[before, c] *= ratio
    return p


def test_clean_synthetic_data_passes(prices):
    splits = {str(prices.index[150].date()): 5.0}
    qr = run_quality_checks(prices, "TSLA", splits, 0.40, 0.03)
    assert qr.passed
    assert qr.flagged_days.empty
    assert len(qr.largest_moves) == 10


def test_unadjusted_split_is_caught(prices):
    date = str(prices.index[150].date())
    bad = _unadjust(prices, date, 5.0)
    flags = flag_suspicious_returns(bad["TSLA_Close"], {date: 5.0}, 0.40, 0.03)
    assert pd.Timestamp(date) in flags.index
    table = check_known_splits(bad["TSLA_Close"], {date: 5.0}, 0.40, 0.03)
    assert not table["ok"].iloc[0]
    assert not run_quality_checks(bad, "TSLA", {date: 5.0}, 0.40, 0.03).passed


def test_one_for_three_split_is_caught(prices):
    date = str(prices.index[100].date())
    bad = _unadjust(prices, date, 3.0)
    # -66.7% is above the 40% extreme threshold too, but the ratio rule must catch it alone
    flags = flag_suspicious_returns(bad["TSLA_Close"], {date: 3.0}, 0.99, 0.03)
    assert flags.loc[pd.Timestamp(date), "reason"] == "near split ratio"


def test_missing_split_date_is_not_ok(prices):
    table = check_known_splits(prices["TSLA_Close"], {"1999-01-01": 2.0}, 0.40, 0.03)
    assert not table["ok"].iloc[0]


def test_high_low_violation_counted(prices):
    bad = prices.copy()
    bad.iloc[10, bad.columns.get_loc("TSLA_High")] = bad["TSLA_Low"].iloc[10] * 0.5
    qr = run_quality_checks(bad, "TSLA", {}, 0.40, 0.03)
    assert qr.high_low_violations >= 1
    assert not qr.passed

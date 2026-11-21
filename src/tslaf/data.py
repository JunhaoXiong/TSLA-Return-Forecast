"""Download, snapshot and load the price data.

Design notes
------------
* All analysis reads a CSV snapshot, never the live API. Yahoo rewrites the whole
  adjusted history after every new split, so a fresh download after a future split
  would rescale every past price. The snapshot plus its SHA-256 keeps results reproducible.
* ``auto_adjust=True`` divides all pre-split Open/High/Low/Close by the cumulative split
  factor, and Yahoo multiplies pre-split Volume by it. Every feature we build is a ratio or
  a log return, so the absolute price scale never enters the model anyway.
* yfinance treats ``end`` as exclusive; our config ``end`` is inclusive, so we add one day.
* Column naming: ``TSLA_Open``, ``TSLA_High``, ``TSLA_Low``, ``TSLA_Close``, ``TSLA_Volume``,
  then ``<TICKER>_Close`` for each market series, with ``^`` dropped (``^VIX`` -> ``VIX``).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OHLCV = ("Open", "High", "Low", "Close", "Volume")


def clean_name(ticker: str) -> str:
    return ticker.replace("^", "")


def download_prices(
    target: str, market: tuple[str, ...], start: str, end_inclusive: str, auto_adjust: bool = True
) -> pd.DataFrame:
    """Download TSLA OHLCV plus market closes into one wide frame (inner-joined on dates)."""
    import yfinance as yf  # lazy import: tests never touch the network

    end_exclusive = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tickers = [target, *market]
    raw = yf.download(tickers, start=start, end=end_exclusive, auto_adjust=auto_adjust,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for {tickers}; check network access to Yahoo Finance.")

    cols = {f"{clean_name(target)}_{f}": raw[f][target] for f in OHLCV}
    for t in market:
        cols[f"{clean_name(t)}_Close"] = raw["Close"][t]
    frame = pd.DataFrame(cols)
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    frame.index.name = "Date"
    # No forward-fill: a date missing for any series is dropped, and the count is reported.
    return frame.dropna(how="any")


def download_splits(target: str) -> dict[str, float]:
    """Split history as Yahoo reports it ({date: ratio}); stored in the snapshot metadata."""
    import yfinance as yf

    splits = yf.Ticker(target).splits
    return {pd.Timestamp(d).strftime("%Y-%m-%d"): float(v) for d, v in splits.items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".meta.json")


def save_prices(prices: pd.DataFrame, csv_path: str | Path, source_info: dict) -> dict:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(csv_path, float_format="%.6f")
    meta = {
        **source_info,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(prices)),
        "columns": list(prices.columns),
        "first_date": prices.index.min().strftime("%Y-%m-%d"),
        "last_date": prices.index.max().strftime("%Y-%m-%d"),
        "sha256": _sha256(csv_path),
    }
    metadata_path(csv_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_metadata(csv_path: str | Path) -> dict:
    return json.loads(metadata_path(Path(csv_path)).read_text(encoding="utf-8"))


def load_prices(csv_path: str | Path, verify_hash: bool = True) -> pd.DataFrame:
    csv_path = Path(csv_path)
    meta_file = metadata_path(csv_path)
    if verify_hash and meta_file.exists():
        expected = json.loads(meta_file.read_text(encoding="utf-8"))["sha256"]
        if _sha256(csv_path) != expected:
            raise ValueError(f"{csv_path} hash mismatch: file changed since it was downloaded.")
    return pd.read_csv(csv_path, index_col="Date", parse_dates=["Date"]).sort_index()


def get_prices(cfg, refresh: bool = False) -> pd.DataFrame:
    """Prices from the snapshot; download only if it is absent or ``refresh``."""
    path = Path(cfg.cache_path)
    if refresh or not path.exists():
        import yfinance as yf

        prices = download_prices(cfg.target_ticker, cfg.market_tickers, cfg.start, cfg.end, cfg.auto_adjust)
        save_prices(prices, path, {
            "source": "yfinance",
            "yfinance_version": getattr(yf, "__version__", "unknown"),
            "auto_adjust": cfg.auto_adjust,
            "requested_start": cfg.start,
            "requested_end_inclusive": cfg.end,
            "yahoo_splits": download_splits(cfg.target_ticker),
        })
    return load_prices(path)

"""Load the project configuration (config.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    target_ticker: str
    market_tickers: tuple[str, ...]
    start: str
    end: str
    auto_adjust: bool
    cache_path: Path
    extreme_return_abs: float
    split_ratio_tolerance: float
    known_splits: dict[str, float]
    return_lags: int
    rv_windows: tuple[int, ...]
    volume_window: int
    market_return_lags: int
    train_frac: float
    val_frac: float
    ridge_alphas: tuple[float, ...] = ()
    fnn: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False)


def load_config(path: str | Path = PROJECT_ROOT / "config.yaml") -> Config:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    d, q, f, s = raw["data"], raw["quality"], raw["features"], raw["split"]

    train_frac, val_frac = float(s["train_frac"]), float(s["val_frac"])
    if not (0 < train_frac and 0 < val_frac and train_frac + val_frac < 1):
        raise ValueError(f"Invalid split fractions: train={train_frac}, val={val_frac}")

    cache = Path(d["cache_path"])
    if not cache.is_absolute():
        cache = path.parent / cache

    return Config(
        target_ticker=str(d["target_ticker"]),
        market_tickers=tuple(str(t) for t in d["market_tickers"]),
        start=str(d["start"]),
        end=str(d["end"]),
        auto_adjust=bool(d["auto_adjust"]),
        cache_path=cache,
        extreme_return_abs=float(q["extreme_return_abs"]),
        split_ratio_tolerance=float(q["split_ratio_tolerance"]),
        known_splits={str(k): float(v) for k, v in q["known_splits"].items()},
        return_lags=int(f["return_lags"]),
        rv_windows=tuple(int(w) for w in f["rv_windows"]),
        volume_window=int(f["volume_window"]),
        market_return_lags=int(f["market_return_lags"]),
        train_frac=train_frac,
        val_frac=val_frac,
        ridge_alphas=tuple(float(a) for a in raw.get("ridge", {}).get("alpha_grid", [])),
        fnn=dict(raw.get("fnn", {})),
        raw=raw,
    )

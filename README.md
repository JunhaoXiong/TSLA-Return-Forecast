# TSLA Return Forecast

Next-day Tesla (TSLA) return forecasting with neural networks (FNN, then LSTM), evaluated
under a strict no-look-ahead protocol against random-walk benchmarks. Results are reported
as they come out, including when the networks do not beat the benchmarks.

## Why returns, not prices

A model trained on price levels can show a high R^2 just by copying yesterday's price.
On this project's test period the random walk ("tomorrow = today") already has a price R^2
of 0.987. So the target is the next-day log return, y_t = ln(C_{t+1}/C_t), and a price forecast is
C_t * exp(y_hat). The key metric is out-of-sample R^2 against the zero-return forecast.

## Setup

- Data: yfinance, TSLA OHLCV plus SPY, QQQ, ^VIX closes, 2010-06-29 (IPO) to 2025-12-31,
  split-adjusted (TSLA splits 1:5 on 2020-08-31 and 1:3 on 2022-08-25; no dividends).
  A CSV snapshot with SHA-256 lives in `data/raw/`.
- Features (37, all known at the close of day t): 20 lagged TSLA log returns, 5/20-day realized vol,
  volume vs its 20-day average, high-low range, close-open return, 5 lags of SPY and QQQ returns,
  VIX level and change.
- Split: chronological 70/10/20 (train / validation / test), scaler fitted on train only.

## Roadmap

| Phase | Content | Status |
| --- | --- | --- |
| 1 | Data snapshot, split checks, features, chronological split, zero / train-mean baselines | done |
| 2 | Ridge regression and FNN (PyTorch), tuned on validation, refit on 80% | this PR |
| 3 | LSTM on the same pipeline | |
| 4 | Diebold-Mariano tests, multiple seeds, walk-forward robustness | |
| 5 | Final report | |

## Run

```bash
pip install -r requirements.txt
python scripts/run_phase1.py      # uses the snapshot; --refresh re-downloads
python scripts/run_phase2.py      # Ridge + FNN, about 40 s on one CPU thread
python -m pytest
```

Reports: [Phase 1](reports/phase1_report.md), [Phase 2](reports/phase2_report.md).

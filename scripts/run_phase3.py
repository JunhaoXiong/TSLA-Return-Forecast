"""Phase 3: LSTM on daily feature sequences, plus a clean replication of the old notebook design.

Usage:  python scripts/run_phase3.py
Writes reports/phase3_report.md and reports/phase3_*.csv. Takes several minutes on CPU.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tslaf.baselines import zero_forecast  # noqa: E402
from tslaf.config import load_config  # noqa: E402
from tslaf.data import get_prices  # noqa: E402
from tslaf.features import build_dataset  # noqa: E402
from tslaf.metrics import evaluate  # noqa: E402
from tslaf.models.legacy import LegacyConfig, fit_legacy  # noqa: E402
from tslaf.models.lstm import config_grid, fit_lstm, tune_lstm  # noqa: E402
from tslaf.sequences import daily_features  # noqa: E402
from tslaf.split import chronological_split  # noqa: E402


def fmt_table(df: pd.DataFrame) -> str:
    out = df.copy()
    for c in ("ret_rmse", "ret_oos_r2_vs_zero", "hit_rate", "price_r2"):
        if c in out:
            out[c] = out[c].map(lambda x: "n/a" if pd.isna(x) else f"{x:.4f}")
    for c in ("price_rmse", "price_mae"):
        if c in out:
            out[c] = out[c].map(lambda x: f"{x:.3f}")
    return out.to_markdown(index=False)


def main() -> None:
    t0 = time.time()
    torch.set_num_threads(1)
    cfg = load_config()
    prices = get_prices(cfg)
    data = build_dataset(prices, cfg)
    sp = chronological_split(data, cfg.train_frac, cfg.val_frac)
    fit80 = pd.concat([sp.train, sp.val])
    daily = daily_features(prices, cfg.target_ticker, cfg.volume_window)
    close = prices[f"{cfg.target_ticker}_Close"]
    out = ROOT / "reports"
    zero_val_mse = float(np.mean(sp.val["target"] ** 2))

    # ---------------- LSTM ----------------
    seeds = [int(s) for s in cfg.lstm["seeds"]]
    grid = config_grid(cfg.lstm)
    best, epochs, tune_tab = tune_lstm(daily, sp.train, sp.val, grid, seeds)
    tune_tab["val_mse_vs_zero"] = tune_tab["val_mse_mean"] / zero_val_mse
    tune_tab.to_csv(out / "phase3_lstm_tuning.csv", index=False, float_format="%.8f")
    t_lstm_tune = time.time() - t0

    val_preds = [fit_lstm(daily, sp.train, best, s, val_part=sp.val).predict(daily, sp.val) for s in seeds]
    test_preds = [fit_lstm(daily, fit80, best, s, epochs=epochs).predict(daily, sp.test) for s in seeds]
    seed_test = pd.DataFrame([{"seed": s, **evaluate(sp.test, p)} for s, p in zip(seeds, test_preds)])
    ens = np.mean(test_preds, axis=0)

    # ---------------- Legacy replication ----------------
    lcfg = LegacyConfig.from_dict(cfg.legacy)
    lseeds = [int(s) for s in cfg.legacy["seeds"]]
    leg_val = [fit_legacy(close, sp.train, lcfg, s, val_part=sp.val) for s in lseeds]
    leg_epochs = max(1, int(round(np.median([m.best_epoch for m in leg_val]))))
    leg_val_prices = [m.predict_price(close, sp.val) for m in leg_val]
    leg_test = [fit_legacy(close, fit80, lcfg, s, epochs=leg_epochs).predict_price(close, sp.test) for s in lseeds]
    leg_test_price = np.mean(leg_test, axis=0)

    def implied(part, price_hat):
        return np.log(np.asarray(price_hat) / part["close_t"].to_numpy())

    # ---------------- Tables ----------------
    val_rows = [{"model": "zero (random walk)", **evaluate(sp.val, zero_forecast(sp.val))},
                {"model": "LSTM, mean of 5 seeds",
                 **pd.DataFrame([evaluate(sp.val, p) for p in val_preds]).mean().to_dict()},
                {"model": "LSTM, 5-seed ensemble", **evaluate(sp.val, np.mean(val_preds, axis=0))},
                {"model": "legacy price LSTM, ensemble", **evaluate(sp.val, implied(sp.val, np.mean(leg_val_prices, axis=0)))}]
    val_res = pd.DataFrame(val_rows)

    test_rows = [{"model": "LSTM, mean of 5 seeds", **seed_test.drop(columns="seed").mean().to_dict()},
                 {"model": "LSTM, 5-seed ensemble", **evaluate(sp.test, ens)},
                 {"model": f"legacy price LSTM, {len(lseeds)}-seed ensemble",
                  **evaluate(sp.test, implied(sp.test, leg_test_price))}]
    p2 = pd.read_csv(out / "phase2_test.csv")
    p2["model"] = p2["model"] + " [Phase 2]"
    test_res = pd.concat([p2, pd.DataFrame(test_rows)], ignore_index=True)

    leg_seed = pd.DataFrame([{"seed": s, **evaluate(sp.test, implied(sp.test, p))} for s, p in zip(lseeds, leg_test)])

    val_res.to_csv(out / "phase3_validation.csv", index=False, float_format="%.6f")
    test_res.to_csv(out / "phase3_test.csv", index=False, float_format="%.6f")
    seed_test.to_csv(out / "phase3_lstm_test_by_seed.csv", index=False, float_format="%.6f")
    leg_seed.to_csv(out / "phase3_legacy_test_by_seed.csv", index=False, float_format="%.6f")
    sp.test[["close_t", "close_next", "target"]].assign(lstm_ensemble=ens, legacy_price=leg_test_price) \
        .to_csv(out / "phase3_test_predictions.csv", float_format="%.8f")

    fit_max = float(close.loc[:fit80.index.max()].max())
    above = float((sp.test["close_next"] > fit_max).mean())
    leg_err = leg_test_price - sp.test["close_next"].to_numpy()
    sd = seed_test["ret_oos_r2_vs_zero"]

    lines = [
        "# Phase 3 report: LSTM and a clean replication of the old notebook",
        "",
        "Generated by `scripts/run_phase3.py`. All numbers are script output. Same split, same protocol as Phase 2: "
        "tune on validation, refit on the first 80%, evaluate the test set once.",
        "",
        "## LSTM tuning",
        "",
        f"Inputs: windows of L days x {daily.shape[1]} daily features ({', '.join(daily.columns)}). "
        f"Chosen: window={best.window}, hidden={best.hidden}, dropout={best.dropout}; "
        f"final epochs = {epochs} (median best epoch over seeds {seeds}).",
        "",
        tune_tab.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Validation results (fitted on train only)",
        "",
        fmt_table(val_res),
        "",
        "## Test results, all models so far (refit on 80%, evaluated once)",
        "",
        "Rows marked [Phase 2] are read from `reports/phase2_test.csv`; same test dates.",
        "",
        fmt_table(test_res),
        "",
        f"LSTM test OOS R^2 across seeds: mean {sd.mean():.4f}, sd {sd.std():.4f}, "
        f"min {sd.min():.4f}, max {sd.max():.4f}. LSTM ensemble forecast std {ens.std():.5f} "
        f"vs actual {sp.test['target'].std():.5f}.",
        "",
        "## Legacy design: price-level LSTM",
        "",
        f"60-day window of closes, {lcfg.layers} x LSTM({lcfg.units}), dropout {lcfg.dropout}, MinMax fitted on closes "
        f"up to {fit80.index.max().date()}; final epochs {leg_epochs} (median best epoch over seeds {lseeds}).",
        "",
        f"- Highest close in the fit period: {fit_max:.2f}. Share of test days whose next close is above it: {above:.1%}. "
        "A MinMax-scaled network has never seen targets above 1 on those days.",
        f"- Mean price error on test (forecast minus actual): {leg_err.mean():+.3f}; "
        f"mean absolute error {np.abs(leg_err).mean():.3f}.",
        "",
        "Per-seed test results (price metrics are the ones the old notebook reported):",
        "",
        fmt_table(leg_seed),
        "",
        "Compare `price_r2` and `price_rmse` with the random-walk row above: the random walk uses no model at all.",
        "",
        "Significance testing (Diebold-Mariano vs the random walk) is Phase 4.",
        "",
        f"Runtime: LSTM tuning {t_lstm_tune:.0f} s, total {time.time() - t0:.0f} s.",
        "",
    ]
    (out / "phase3_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(val_res.to_string(index=False))
    print(test_res.to_string(index=False))
    print(best, epochs, "legacy epochs", leg_epochs, "above", above, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

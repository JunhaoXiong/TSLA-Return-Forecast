"""Phase 4: significance tests, seed distribution, walk-forward over the test period.

Usage:  python scripts/run_phase4.py   (needs the Phase 2 and 3 outputs in reports/)
Writes reports/phase4_report.md and reports/phase4_*.csv.
"""

from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tslaf.config import load_config  # noqa: E402
from tslaf.data import get_prices  # noqa: E402
from tslaf.features import build_dataset, feature_columns  # noqa: E402
from tslaf.metrics import oos_r2_vs_zero  # noqa: E402
from tslaf.models.fnn import FNNConfig, fit_fnn  # noqa: E402
from tslaf.models.lstm import LSTMConfig, fit_lstm  # noqa: E402
from tslaf.models.ridge import fit_ridge  # noqa: E402
from tslaf.sequences import daily_features  # noqa: E402
from tslaf.split import chronological_split  # noqa: E402
from tslaf.stats import clark_west, diebold_mariano, holm, pesaran_timmermann  # noqa: E402
from tslaf.walkforward import walk_forward  # noqa: E402


def chosen_fnn(cfg) -> tuple[FNNConfig, int]:
    t = pd.read_csv(ROOT / "reports" / "phase2_fnn_tuning.csv")
    b = t.loc[t["val_mse_mean"].idxmin()]
    f = cfg.fnn
    c = FNNConfig(hidden=tuple(ast.literal_eval(b["hidden"])), dropout=float(b["dropout"]),
                  weight_decay=float(b["weight_decay"]), learning_rate=float(f["learning_rate"]),
                  batch_size=int(f["batch_size"]), max_epochs=int(f["max_epochs"]), patience=int(f["patience"]))
    return c, max(1, int(round(b["best_epoch_median"])))


def chosen_lstm(cfg) -> tuple[LSTMConfig, int]:
    t = pd.read_csv(ROOT / "reports" / "phase3_lstm_tuning.csv")
    b = t.loc[t["val_mse_mean"].idxmin()]
    l = cfg.lstm
    c = LSTMConfig(window=int(b["window"]), hidden=int(b["hidden"]), dropout=float(b["dropout"]),
                   weight_decay=float(l["weight_decay"]), learning_rate=float(l["learning_rate"]),
                   batch_size=int(l["batch_size"]), max_epochs=int(l["max_epochs"]), patience=int(l["patience"]))
    return c, max(1, int(round(b["best_epoch_median"])))


def chosen_alpha() -> float:
    t = pd.read_csv(ROOT / "reports" / "phase2_ridge_tuning.csv")
    return float(t.loc[t["val_mse"].idxmin(), "alpha"])


def test_table(y: np.ndarray, forecasts: dict[str, np.ndarray]) -> pd.DataFrame:
    zero = np.zeros_like(y)
    rows = []
    for name, f in forecasts.items():
        dm, cw, pt = diebold_mariano(y, f, zero), clark_west(y, f, zero), pesaran_timmermann(y, f)
        rows.append({"model": name, "oos_r2": oos_r2_vs_zero(y, f), "dm_stat": dm["dm_stat"], "dm_p": dm["dm_p"],
                     "cw_stat": cw["cw_stat"], "cw_p": cw["cw_p"], "hit_rate": pt["hit_rate"],
                     "share_up_forecasts": float(np.mean(f > 0)), "pt_p": pt["pt_p"]})
    t = pd.DataFrame(rows)
    t["dm_p_holm"], t["cw_p_holm"], t["pt_p_holm"] = holm(t["dm_p"]), holm(t["cw_p"]), holm(t["pt_p"])
    return t


def by_year(y: pd.Series, forecasts: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for year in sorted(set(y.index.year)):
        m = (y.index.year == year)
        row = {"year": year, "days": int(m.sum())}
        for name, f in forecasts.items():
            row[name] = oos_r2_vs_zero(y.to_numpy()[m], np.asarray(f)[m])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    torch.set_num_threads(1)
    cfg = load_config()
    prices = get_prices(cfg)
    data = build_dataset(prices, cfg)
    cols = feature_columns(data)
    sp = chronological_split(data, cfg.train_frac, cfg.val_frac)
    fit80 = pd.concat([sp.train, sp.val])
    daily = daily_features(prices, cfg.target_ticker, cfg.volume_window)
    y = sp.test["target"]
    out = ROOT / "reports"

    # ---------- 1. Significance of the official Phase 2/3 forecasts ----------
    p2 = pd.read_csv(out / "phase2_test_predictions.csv", index_col=0, parse_dates=True)
    p3 = pd.read_csv(out / "phase3_test_predictions.csv", index_col=0, parse_dates=True)
    assert (p2.index == sp.test.index).all() and (p3.index == sp.test.index).all()
    official = {
        "train mean (80%)": np.full(len(y), fit80["target"].mean()),
        "Ridge": p2["ridge"].to_numpy(),
        "FNN 5-seed ensemble": p2["fnn_ensemble"].to_numpy(),
        "LSTM 5-seed ensemble": p3["lstm_ensemble"].to_numpy(),
        "legacy price LSTM (implied return)": np.log(p3["legacy_price"] / p3["close_t"]).to_numpy(),
    }
    sig = test_table(y.to_numpy(), official)
    sig.to_csv(out / "phase4_significance.csv", index=False, float_format="%.6f")
    legacy_price_dm = diebold_mariano(p3["close_next"], p3["legacy_price"], p3["close_t"])

    # ---------- 2. Seed distribution (20 seeds, hyperparameters fixed) ----------
    n_seeds = int(cfg.phase4["extra_seeds"])
    fcfg, f_ep = chosen_fnn(cfg)
    lcfg, l_ep = chosen_lstm(cfg)
    fnn_preds = [fit_fnn(fit80[cols], fit80["target"], fcfg, s, epochs=f_ep).predict(sp.test[cols]) for s in range(n_seeds)]
    lstm_preds = [fit_lstm(daily, fit80, lcfg, s, epochs=l_ep).predict(daily, sp.test) for s in range(n_seeds)]
    seed_rows = []
    for name, preds in (("FNN", fnn_preds), ("LSTM", lstm_preds)):
        for s, p in enumerate(preds):
            seed_rows.append({"model": name, "seed": s, "oos_r2": oos_r2_vs_zero(y, p),
                              "cw_p": clark_west(y, p, np.zeros(len(y)))["cw_p"]})
    seeds_df = pd.DataFrame(seed_rows)
    seeds_df.to_csv(out / "phase4_seeds.csv", index=False, float_format="%.6f")
    seed_summary = seeds_df.groupby("model").agg(
        mean=("oos_r2", "mean"), sd=("oos_r2", "std"), min=("oos_r2", "min"), max=("oos_r2", "max"),
        share_positive=("oos_r2", lambda s: float((s > 0).mean())),
        share_cw_p_below_5pct=("cw_p", lambda s: float((s < 0.05).mean()))).reset_index()
    ens20 = {"FNN 20-seed ensemble": np.mean(fnn_preds, axis=0), "LSTM 20-seed ensemble": np.mean(lstm_preds, axis=0)}
    sig20 = test_table(y.to_numpy(), ens20)
    t_seeds = time.time() - t0

    # ---------- 3. Walk-forward over the test period ----------
    alpha = chosen_alpha()
    wf_seeds = [int(s) for s in cfg.phase4["walk_forward_seeds"]]
    step = int(cfg.phase4["walk_forward_step"])
    models = {
        "train mean": lambda fit, pred: np.full(len(pred), fit["target"].mean()),
        "Ridge": lambda fit, pred: fit_ridge(fit[cols], fit["target"], alpha).predict(pred[cols]),
        "FNN 5-seed ensemble": lambda fit, pred: np.mean(
            [fit_fnn(fit[cols], fit["target"], fcfg, s, epochs=f_ep).predict(pred[cols]) for s in wf_seeds], axis=0),
        "LSTM 5-seed ensemble": lambda fit, pred: np.mean(
            [fit_lstm(daily, fit, lcfg, s, epochs=l_ep).predict(daily, pred) for s in wf_seeds], axis=0),
    }
    wf = walk_forward(data, sp.test.index, step, models)
    wf.to_csv(out / "phase4_walkforward_predictions.csv", float_format="%.8f")
    wf_fc = {k: wf[k].to_numpy() for k in models}
    wf_sig = test_table(y.to_numpy(), wf_fc)
    wf_sig.to_csv(out / "phase4_walkforward_significance.csv", index=False, float_format="%.6f")
    n_blocks = int(wf["refit_block"].max()) + 1

    years = by_year(y, {**{f"{k} (fixed)": v for k, v in official.items() if "legacy" not in k},
                        **{f"{k} (walk-fwd)": v for k, v in wf_fc.items()}})
    years.to_csv(out / "phase4_by_year.csv", index=False, float_format="%.6f")

    # ---------- Report ----------
    def sigfmt(t: pd.DataFrame) -> str:
        t = t.copy()
        for c in t.columns:
            if c != "model":
                t[c] = t[c].map(lambda x: "n/a" if pd.isna(x) else f"{x:.4f}")
        return t.to_markdown(index=False)

    lines = [
        "# Phase 4 report: is any of this significant?",
        "",
        "Generated by `scripts/run_phase4.py`. All numbers are script output. Test period "
        f"{sp.test.index.min().date()}..{sp.test.index.max().date()} ({len(y)} days). Benchmark for every test: "
        "the zero-return forecast (random walk). All p-values are one-sided, H1 = model beats the benchmark.",
        "",
        "Tests: Diebold-Mariano on squared errors (Newey-West lag 9, HLN correction); Clark-West (adjusted for the "
        "nested benchmark; the primary test); Pesaran-Timmermann for direction (n/a when the forecast never changes "
        "sign). `_holm` columns adjust for testing all rows of the table together.",
        "",
        "## 1. Official forecasts from Phases 2 and 3",
        "",
        sigfmt(sig),
        "",
        f"Legacy price LSTM on price levels vs the random walk (DM, H1 = legacy better): "
        f"stat {legacy_price_dm['dm_stat']:.3f}, p = {legacy_price_dm['dm_p']:.4f}.",
        "",
        f"## 2. Seed distribution ({n_seeds} seeds, hyperparameters and epochs fixed from Phases 2-3)",
        "",
        f"FNN: {fcfg.hidden}, dropout {fcfg.dropout}, weight decay {fcfg.weight_decay}, {f_ep} epochs. "
        f"LSTM: window {lcfg.window}, hidden {lcfg.hidden}, dropout {lcfg.dropout}, {l_ep} epoch(s). "
        "Only the random seed changes, so this is not tuning on the test set.",
        "",
        seed_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "20-seed ensembles:",
        "",
        sigfmt(sig20),
        "",
        f"## 3. Walk-forward over the test period (expanding window, refit every {step} days, {n_blocks} refits)",
        "",
        f"Ridge alpha {alpha:g}; FNN and LSTM as above, each a {len(wf_seeds)}-seed ensemble per refit.",
        "",
        sigfmt(wf_sig),
        "",
        "## 4. OOS R^2 vs zero by calendar year",
        "",
        years.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Runtime: {time.time() - t0:.0f} s (seed study done at {t_seeds:.0f} s).",
        "",
    ]
    (out / "phase4_report.md").write_text("\n".join(lines), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(sig.to_string(index=False))
    print(seed_summary.to_string(index=False))
    print(sig20.to_string(index=False))
    print(wf_sig.to_string(index=False))
    print(years.to_string(index=False))
    print("legacy price DM", legacy_price_dm, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

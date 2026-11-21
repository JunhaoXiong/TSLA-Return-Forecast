"""Phase 2: Ridge regression and FNN, tuned on validation, refit on 80%, tested once.

Usage:  python scripts/run_phase2.py
Writes reports/phase2_report.md and reports/phase2_*.csv.
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

from tslaf.baselines import train_mean_forecast, zero_forecast  # noqa: E402
from tslaf.config import load_config  # noqa: E402
from tslaf.data import get_prices  # noqa: E402
from tslaf.features import build_dataset, feature_columns  # noqa: E402
from tslaf.metrics import evaluate  # noqa: E402
from tslaf.models.fnn import config_grid, fit_fnn, tune_fnn  # noqa: E402
from tslaf.models.ridge import fit_ridge, tune_ridge  # noqa: E402
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
    torch.set_num_threads(1)  # reproducible, and fast enough for this size
    cfg = load_config()
    data = build_dataset(get_prices(cfg), cfg)
    cols = feature_columns(data)
    sp = chronological_split(data, cfg.train_frac, cfg.val_frac)
    fit80 = pd.concat([sp.train, sp.val])  # train + validation = first 80%
    seeds = [int(s) for s in cfg.fnn["seeds"]]
    out = ROOT / "reports"

    # ---------------- Stage 1: tuning on train -> validation ----------------
    alpha, ridge_tab = tune_ridge(sp.train[cols], sp.train["target"], sp.val[cols], sp.val["target"],
                                  cfg.ridge_alphas)
    zero_val_mse = float(np.mean(sp.val["target"] ** 2))
    ridge_tab["val_mse_vs_zero"] = ridge_tab["val_mse"] / zero_val_mse
    ridge_tab.to_csv(out / "phase2_ridge_tuning.csv", index=False, float_format="%.8f")

    grid = config_grid(cfg.fnn)
    best_cfg, epochs, fnn_tab = tune_fnn(sp.train[cols], sp.train["target"], sp.val[cols], sp.val["target"],
                                         grid, seeds)
    fnn_tab["val_mse_vs_zero"] = fnn_tab["val_mse_mean"] / zero_val_mse
    fnn_tab.to_csv(out / "phase2_fnn_tuning.csv", index=False, float_format="%.8f")

    # validation metrics of the chosen settings (models fitted on train only)
    val_rows = [{"model": "zero (random walk)", **evaluate(sp.val, zero_forecast(sp.val))},
                {"model": "train mean", **evaluate(sp.val, train_mean_forecast(sp.train, sp.val))}]
    r70 = fit_ridge(sp.train[cols], sp.train["target"], alpha)
    val_rows.append({"model": f"ridge (alpha={alpha:g})", **evaluate(sp.val, r70.predict(sp.val[cols]))})
    val_preds = [fit_fnn(sp.train[cols], sp.train["target"], best_cfg, s, sp.val[cols], sp.val["target"])
                 .predict(sp.val[cols]) for s in seeds]
    per_seed = pd.DataFrame([evaluate(sp.val, p) for p in val_preds])
    val_rows.append({"model": "FNN, mean of 5 seeds", **per_seed.mean().to_dict()})
    val_rows.append({"model": "FNN, 5-seed ensemble", **evaluate(sp.val, np.mean(val_preds, axis=0))})
    val_res = pd.DataFrame(val_rows)

    # ---------------- Stage 2: refit on 80%, test once ----------------
    test_rows = [{"model": "zero (random walk)", **evaluate(sp.test, zero_forecast(sp.test))},
                 {"model": "train mean (80%)", **evaluate(sp.test, train_mean_forecast(fit80, sp.test))}]
    r80 = fit_ridge(fit80[cols], fit80["target"], alpha)
    ridge_test = r80.predict(sp.test[cols])
    test_rows.append({"model": f"ridge (alpha={alpha:g})", **evaluate(sp.test, ridge_test)})
    test_preds = [fit_fnn(fit80[cols], fit80["target"], best_cfg, s, epochs=epochs).predict(sp.test[cols])
                  for s in seeds]
    seed_test = pd.DataFrame([{"seed": s, **evaluate(sp.test, p)} for s, p in zip(seeds, test_preds)])
    test_rows.append({"model": "FNN, mean of 5 seeds", **seed_test.drop(columns="seed").mean().to_dict()})
    test_rows.append({"model": "FNN, 5-seed ensemble", **evaluate(sp.test, np.mean(test_preds, axis=0))})
    test_res = pd.DataFrame(test_rows)

    val_res.to_csv(out / "phase2_validation.csv", index=False, float_format="%.6f")
    test_res.to_csv(out / "phase2_test.csv", index=False, float_format="%.6f")
    seed_test.to_csv(out / "phase2_fnn_test_by_seed.csv", index=False, float_format="%.6f")
    preds = sp.test[["close_t", "close_next", "target"]].assign(
        ridge=ridge_test, fnn_ensemble=np.mean(test_preds, axis=0))
    preds.to_csv(out / "phase2_test_predictions.csv", float_format="%.8f")

    coef = pd.Series(r80.model.coef_, index=cols)
    top = coef.reindex(coef.abs().sort_values(ascending=False).index[:8]).to_frame("coef (per 1 sd)")
    ens = np.mean(test_preds, axis=0)
    pred_stats = pd.DataFrame({
        "forecast": ["ridge", "FNN ensemble", "actual target"],
        "std": [ridge_test.std(), ens.std(), sp.test["target"].std()],
        "share_positive": [(ridge_test > 0).mean(), (ens > 0).mean(), (sp.test["target"] > 0).mean()],
    })

    sd = seed_test["ret_oos_r2_vs_zero"]
    lines = [
        "# Phase 2 report: Ridge regression and FNN",
        "",
        "Generated by `scripts/run_phase2.py`. All numbers are script output.",
        "",
        "## Protocol",
        "",
        "1. Hyperparameters chosen on validation only (fit on train 2010-07-28..2021-05-11, "
        "score on validation 2021-05-12..2022-11-22).",
        "2. Final models refit on train + validation (first 80%) with those hyperparameters; "
        "scalers refit on the same 80%.",
        f"3. Test set ({sp.test.index.min().date()}..{sp.test.index.max().date()}, {len(sp.test)} days) "
        "evaluated once.",
        "",
        "## Tuning",
        "",
        f"Ridge: chosen alpha = {alpha:g}. `val_mse_vs_zero` < 1 would mean beating the random walk on validation.",
        "",
        ridge_tab.to_markdown(index=False, floatfmt=".6g"),
        "",
        f"FNN: chosen hidden={best_cfg.hidden}, dropout={best_cfg.dropout}, weight_decay={best_cfg.weight_decay}; "
        f"final training epochs = {epochs} (median best epoch over seeds {seeds}).",
        "",
        fnn_tab.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Validation results (models fitted on train only)",
        "",
        fmt_table(val_res),
        "",
        "## Test results (models refit on 80%, evaluated once)",
        "",
        fmt_table(test_res),
        "",
        f"FNN test OOS R^2 across seeds: mean {sd.mean():.4f}, sd {sd.std():.4f}, "
        f"min {sd.min():.4f}, max {sd.max():.4f}.",
        "",
        "Per-seed test results:",
        "",
        fmt_table(seed_test),
        "",
        "## What the forecasts look like (test set)",
        "",
        pred_stats.to_markdown(index=False, floatfmt=".5f"),
        "",
        "Largest Ridge coefficients (80% fit, standardised features):",
        "",
        top.to_markdown(floatfmt=".6f"),
        "",
        "## Notes generated from the numbers above",
        "",
        f"- Ridge alpha {alpha:g} is {'at the upper edge' if alpha == max(cfg.ridge_alphas) else 'inside'} of the grid. "
        "As alpha grows, Ridge shrinks every coefficient toward 0 and its forecast tends to the training mean.",
        f"- FNN early stopping picked a median of {epochs} epoch(s): validation loss stopped improving almost at once.",
        f"- Forecast std on test: Ridge {ridge_test.std():.5f}, FNN ensemble {ens.std():.5f}, "
        f"versus {sp.test['target'].std():.5f} for actual returns. Both models output a nearly constant small drift.",
        "",
        "Significance testing (Diebold-Mariano vs the random walk) is Phase 4.",
        "",
        f"Runtime: {time.time() - t0:.0f} s.",
        "",
    ]
    (out / "phase2_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(val_res.to_string(index=False))
    print(test_res.to_string(index=False))
    print(f"alpha={alpha} fnn={best_cfg} epochs={epochs} runtime={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

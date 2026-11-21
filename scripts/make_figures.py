"""Figures for the final report, built from the saved Phase 2-4 outputs in reports/.

Usage:  python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP, FIG = ROOT / "reports", ROOT / "reports" / "figures"
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.edgecolor": "#b5b4ad",
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.frameon": False,
})
C = {"Ridge": "#2a78d6", "FNN": "#eb6834", "LSTM": "#1baf7a", "train mean": "#8a8984",
     "legacy": "#c43d3d", "actual": "#2b2b2b", "rw": "#8a8984"}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p2 = pd.read_csv(REP / "phase2_test_predictions.csv", index_col=0, parse_dates=True)
    p3 = pd.read_csv(REP / "phase3_test_predictions.csv", index_col=0, parse_dates=True)
    y = p2["target"]

    # 1) Cumulative squared-error advantage over the random walk
    fc = {"Ridge": p2["ridge"], "FNN": p2["fnn_ensemble"], "LSTM": p3["lstm_ensemble"]}
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    nudge = {"Ridge": 0, "FNN": -6, "LSTM": 6}  # points, to keep end labels apart
    for name, f in fc.items():
        adv = (y ** 2 - (y - f) ** 2).cumsum()
        ax.plot(adv.index, adv * 1e4, color=C[name], lw=1.3)
        ax.annotate(name, (adv.index[-1], adv.iloc[-1] * 1e4), xytext=(4, nudge[name]), textcoords="offset points",
                    color=C[name], va="center", fontsize=8)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_ylabel("cumulative (SE random walk − SE model), ×1e-4")
    ax.set_title("Above 0 = model ahead of the random walk so far (test period)", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "cumulative_advantage.png", dpi=180)
    plt.close(fig)

    # 2) Seed distribution
    seeds = pd.read_csv(REP / "phase4_seeds.csv")
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    rng = np.random.default_rng(0)
    for i, name in enumerate(["FNN", "LSTM"]):
        v = seeds.loc[seeds["model"] == name, "oos_r2"].to_numpy() * 100
        ax.scatter(v, i + rng.uniform(-0.12, 0.12, len(v)), color=C[name], s=22, alpha=0.85, zorder=3)
        ax.plot([v.mean()] * 2, [i - 0.25, i + 0.25], color=C[name], lw=2)
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_yticks([0, 1], ["FNN", "LSTM"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("test OOS R² vs zero forecast (%), one dot per seed; bar = mean")
    ax.set_title("20 seeds, same hyperparameters: half land above zero, half below", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "seed_distribution.png", dpi=180)
    plt.close(fig)

    # 3) Legacy price model vs actual and random walk
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(p3.index, p3["close_next"], color=C["actual"], lw=1.0, label="actual next close")
    a1.plot(p3.index, p3["legacy_price"], color=C["legacy"], lw=1.0, label="legacy price LSTM forecast")
    a1.set_ylabel("price ($)")
    a1.legend(loc="upper left")
    a1.set_title("Old design (price-level LSTM) on the clean pipeline, test period", loc="left", fontsize=9)
    e_leg = (p3["legacy_price"] - p3["close_next"]).abs().rolling(21).mean()
    e_rw = (p3["close_t"] - p3["close_next"]).abs().rolling(21).mean()
    a2.plot(e_leg.index, e_leg, color=C["legacy"], lw=1.1, label="legacy LSTM")
    a2.plot(e_rw.index, e_rw, color=C["rw"], lw=1.1, label="random walk (yesterday's close)")
    a2.set_ylabel("21-day mean |error| ($)")
    a2.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "legacy_vs_random_walk.png", dpi=180)
    plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()

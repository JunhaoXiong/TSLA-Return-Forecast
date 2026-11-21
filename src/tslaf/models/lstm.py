"""LSTM on daily feature sequences, trained with the same protocol as the FNN.

* Per-feature StandardScaler fitted on daily rows dated up to the last fit date only.
* Target divided by its fit-set std for training; predictions multiplied back.
* Early stopping on validation MSE; the final 80% model trains for the tuned epoch count.
* Rows whose L-day window would start inside the indicator warm-up are dropped; this only
  affects the first few training rows, never validation or test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from ..sequences import make_windows, window_positions
from .fnn import set_seed


class LSTMNet(nn.Module):
    def __init__(self, n_in: int, hidden: int, dropout: float, layers: int = 1, inter_dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True,
                            dropout=inter_dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1])).squeeze(-1)


@dataclass
class LSTMConfig:
    window: int = 20
    hidden: int = 32
    dropout: float = 0.1
    weight_decay: float = 1e-4
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 20


@dataclass
class LSTMModel:
    net: LSTMNet
    scaler: StandardScaler
    y_scale: float
    window: int
    epochs_trained: int
    best_epoch: int
    history: list[float] = field(default_factory=list)

    def windows(self, daily: pd.DataFrame, part: pd.DataFrame, require_all: bool = True) -> tuple[np.ndarray, np.ndarray]:
        pos = window_positions(daily, part.index, self.window)
        keep = pos >= 0
        if require_all and not keep.all():
            raise ValueError(f"{(~keep).sum()} rows lack a full {self.window}-day window.")
        vals = self.scaler.transform(daily.fillna(0.0).to_numpy())
        return make_windows(vals, pos[keep], self.window), keep

    def predict(self, daily: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
        X, _ = self.windows(daily, part)
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(X, dtype=torch.float32)).numpy().astype(float) * self.y_scale


def train_loop(net, xt, yt, cfg, gen, xv=None, yv=None, y_scale=1.0, epochs=None):
    """Shared mini-batch loop with optional early stopping. Returns (epochs_run, best_epoch, history)."""
    opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    has_val = xv is not None
    n_epochs = cfg.max_epochs if has_val else int(epochs)
    best_mse, best_epoch, best_state, wait, history = np.inf, 0, None, 0, []
    epoch = 0
    for epoch in range(1, n_epochs + 1):
        net.train()
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(xt), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss_fn(net(xt[idx]), yt[idx]).backward()
            opt.step()
        if has_val:
            net.eval()
            with torch.no_grad():
                pv = net(xv).numpy().astype(float) * y_scale
            mse = float(np.mean((yv - pv) ** 2))
            history.append(mse)
            if mse < best_mse - 1e-12:
                best_mse, best_epoch, wait = mse, epoch, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= cfg.patience:
                    break
    if has_val and best_state is not None:
        net.load_state_dict(best_state)
    else:
        best_epoch = epoch
    return epoch, best_epoch, history


def fit_lstm(daily: pd.DataFrame, fit_part: pd.DataFrame, cfg: LSTMConfig, seed: int,
             val_part: pd.DataFrame | None = None, epochs: int | None = None) -> LSTMModel:
    if val_part is None and epochs is None:
        raise ValueError("Without a validation set, the number of epochs must be given.")
    set_seed(seed)
    fit_end = fit_part.index.max()
    scaler = StandardScaler().fit(daily.loc[:fit_end].dropna().to_numpy())
    model = LSTMModel(net=None, scaler=scaler, y_scale=0.0, window=cfg.window, epochs_trained=0, best_epoch=0)
    X, keep = model.windows(daily, fit_part, require_all=False)
    y = fit_part["target"].to_numpy()[keep]
    model.y_scale = float(y.std())
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y / model.y_scale, dtype=torch.float32)
    model.net = LSTMNet(X.shape[2], cfg.hidden, cfg.dropout)
    gen = torch.Generator().manual_seed(seed)
    xv = yv = None
    if val_part is not None:
        Xv, _ = model.windows(daily, val_part)
        xv, yv = torch.tensor(Xv, dtype=torch.float32), val_part["target"].to_numpy()
    model.epochs_trained, model.best_epoch, model.history = train_loop(
        model.net, xt, yt, cfg, gen, xv, yv, model.y_scale, epochs)
    return model


def config_grid(lstm_cfg: dict) -> list[LSTMConfig]:
    return [LSTMConfig(window=int(w), hidden=int(h), dropout=float(d),
                       weight_decay=float(lstm_cfg["weight_decay"]),
                       learning_rate=float(lstm_cfg["learning_rate"]),
                       batch_size=int(lstm_cfg["batch_size"]),
                       max_epochs=int(lstm_cfg["max_epochs"]),
                       patience=int(lstm_cfg["patience"]))
            for w in lstm_cfg["window_grid"] for h in lstm_cfg["hidden_grid"] for d in lstm_cfg["dropout_grid"]]


def tune_lstm(daily, train, val, grid: list[LSTMConfig], seeds) -> tuple[LSTMConfig, int, pd.DataFrame]:
    """Same selection rule as the FNN: lowest mean validation MSE over seeds; epochs = median best epoch."""
    rows = []
    for k, c in enumerate(grid):
        for s in seeds:
            m = fit_lstm(daily, train, c, s, val_part=val)
            rows.append({"config": k, "window": c.window, "hidden": c.hidden, "dropout": c.dropout,
                         "seed": s, "best_epoch": m.best_epoch, "val_mse": min(m.history)})
    table = pd.DataFrame(rows)
    summary = (table.groupby(["config", "window", "hidden", "dropout"], as_index=False)
               .agg(val_mse_mean=("val_mse", "mean"), val_mse_std=("val_mse", "std"),
                    best_epoch_median=("best_epoch", "median")))
    best = summary.loc[summary["val_mse_mean"].idxmin()]
    return grid[int(best["config"])], max(1, int(round(best["best_epoch_median"]))), summary

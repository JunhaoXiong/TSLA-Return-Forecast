"""Feed-forward neural network (multilayer perceptron) in PyTorch.

Design notes
------------
* Inputs are standardised with a scaler fitted on the training rows only.
* The target (daily log return, std about 0.035) is divided by its training std so the
  network trains on unit-scale numbers; predictions are multiplied back.
* Mini-batches are shuffled *within* the training rows. That is not look-ahead: which
  dates are in the training set is fixed chronologically; the order in which the optimiser
  visits them does not reveal anything about the validation or test period.
* Early stopping watches validation MSE and keeps the weights of the best epoch.
* For the final model there is no validation set left (it is part of the 80%), so we train
  for the number of epochs chosen during tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class FNNConfig:
    hidden: tuple[int, ...] = (64, 32)
    dropout: float = 0.1
    weight_decay: float = 1e-4
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 20


@dataclass
class FNNModel:
    net: MLP
    scaler: StandardScaler
    y_scale: float
    epochs_trained: int
    best_epoch: int
    history: list[float] = field(default_factory=list)  # validation MSE per epoch (if any)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(self.scaler.transform(X.to_numpy()), dtype=torch.float32)
            return self.net(x).numpy().astype(float) * self.y_scale


def fit_fnn(X: pd.DataFrame, y: pd.Series, cfg: FNNConfig, seed: int,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None,
            epochs: int | None = None) -> FNNModel:
    """Train an MLP.

    With a validation set: early stopping, returns the best-epoch weights.
    Without one: trains exactly ``epochs`` epochs (must be given).
    """
    if X_val is None and epochs is None:
        raise ValueError("Without a validation set, the number of epochs must be given.")
    set_seed(seed)
    scaler = StandardScaler().fit(X.to_numpy())
    y_scale = float(y.std())
    xt = torch.tensor(scaler.transform(X.to_numpy()), dtype=torch.float32)
    yt = torch.tensor(y.to_numpy() / y_scale, dtype=torch.float32)

    net = MLP(xt.shape[1], tuple(cfg.hidden), cfg.dropout)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)

    has_val = X_val is not None
    if has_val:
        xv = torch.tensor(scaler.transform(X_val.to_numpy()), dtype=torch.float32)
        yv = y_val.to_numpy()

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
    return FNNModel(net=net, scaler=scaler, y_scale=y_scale, epochs_trained=epoch,
                    best_epoch=best_epoch, history=history)


def config_grid(fnn_cfg: dict) -> list[FNNConfig]:
    grid = []
    for h in fnn_cfg["hidden_grid"]:
        for d in fnn_cfg["dropout_grid"]:
            for wd in fnn_cfg["weight_decay_grid"]:
                grid.append(FNNConfig(hidden=tuple(h), dropout=float(d), weight_decay=float(wd),
                                      learning_rate=float(fnn_cfg["learning_rate"]),
                                      batch_size=int(fnn_cfg["batch_size"]),
                                      max_epochs=int(fnn_cfg["max_epochs"]),
                                      patience=int(fnn_cfg["patience"])))
    return grid


def tune_fnn(X_tr, y_tr, X_val, y_val, grid: list[FNNConfig], seeds) -> tuple[FNNConfig, int, pd.DataFrame]:
    """Each config is trained once per seed with early stopping.

    Selection: lowest mean validation MSE across seeds. The final epoch count is the
    median best epoch across seeds for the chosen config.
    """
    rows = []
    for k, c in enumerate(grid):
        for s in seeds:
            m = fit_fnn(X_tr, y_tr, c, s, X_val, y_val)
            rows.append({"config": k, "hidden": str(c.hidden), "dropout": c.dropout,
                         "weight_decay": c.weight_decay, "seed": s,
                         "best_epoch": m.best_epoch, "val_mse": min(m.history)})
    table = pd.DataFrame(rows)
    summary = (table.groupby(["config", "hidden", "dropout", "weight_decay"], as_index=False)
               .agg(val_mse_mean=("val_mse", "mean"), val_mse_std=("val_mse", "std"),
                    best_epoch_median=("best_epoch", "median")))
    best_row = summary.loc[summary["val_mse_mean"].idxmin()]
    best_cfg = grid[int(best_row["config"])]
    epochs = max(1, int(round(best_row["best_epoch_median"])))
    return best_cfg, epochs, summary

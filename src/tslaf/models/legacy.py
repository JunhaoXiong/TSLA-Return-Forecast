"""Replication of the old iRisk notebook design, on the clean pipeline.

Old design: predict the next price level from a 60-day window of prices with four stacked
LSTM(50) layers and Dropout(0.2), MinMax-scaled. Here it is run the right way, so that any
failure is the design's, not a bug's:

* the input and the target are the same series (Close); the old notebook trained on High and
  tested on Open by mistake;
* MinMaxScaler is fitted on closes dated up to the last fit date only;
* same chronological split, validation early stopping and 80% refit as every other model.

Its output is a price; the implied return forecast is ln(P_hat / C_t).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from .fnn import set_seed
from .lstm import train_loop


class StackedLSTM(nn.Module):
    def __init__(self, units: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(1, units, num_layers=layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(units, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1])).squeeze(-1)


@dataclass
class LegacyConfig:
    window: int = 60
    units: int = 50
    layers: int = 4
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    max_epochs: int = 300
    patience: int = 20

    @classmethod
    def from_dict(cls, d: dict) -> "LegacyConfig":
        return cls(**{k: d[k] for k in ("window", "units", "layers", "dropout", "learning_rate",
                                         "batch_size", "max_epochs", "patience")})


@dataclass
class LegacyModel:
    net: StackedLSTM
    scaler: MinMaxScaler
    window: int
    epochs_trained: int
    best_epoch: int

    def _windows(self, close: pd.Series, part: pd.DataFrame):
        pos = close.index.get_indexer(part.index)
        keep = pos - self.window + 1 >= 0
        vals = self.scaler.transform(close.to_numpy()[:, None])[:, 0]
        idx = pos[keep][:, None] + np.arange(-self.window + 1, 1)[None, :]
        return vals[idx][:, :, None], keep

    def predict_price(self, close: pd.Series, part: pd.DataFrame) -> np.ndarray:
        X, keep = self._windows(close, part)
        if not keep.all():
            raise ValueError("Rows without a full window in a prediction set.")
        self.net.eval()
        with torch.no_grad():
            z = self.net(torch.tensor(X, dtype=torch.float32)).numpy().astype(float)
        return self.scaler.inverse_transform(z[:, None])[:, 0]


def fit_legacy(close: pd.Series, fit_part: pd.DataFrame, cfg: LegacyConfig, seed: int,
               val_part: pd.DataFrame | None = None, epochs: int | None = None) -> LegacyModel:
    if val_part is None and epochs is None:
        raise ValueError("Without a validation set, the number of epochs must be given.")
    set_seed(seed)
    scaler = MinMaxScaler().fit(close.loc[:fit_part.index.max()].to_numpy()[:, None])
    model = LegacyModel(net=StackedLSTM(cfg.units, cfg.layers, cfg.dropout), scaler=scaler,
                        window=cfg.window, epochs_trained=0, best_epoch=0)
    X, keep = model._windows(close, fit_part)
    y = scaler.transform(fit_part["close_next"].to_numpy()[keep][:, None])[:, 0]
    xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    xv = yv = None
    if val_part is not None:
        Xv, kv = model._windows(close, val_part)
        assert kv.all()
        xv = torch.tensor(Xv, dtype=torch.float32)
        yv = scaler.transform(val_part["close_next"].to_numpy()[:, None])[:, 0]
    gen = torch.Generator().manual_seed(seed)
    model.epochs_trained, model.best_epoch, _ = train_loop(model.net, xt, yt, cfg, gen, xv, yv, 1.0, epochs)
    return model

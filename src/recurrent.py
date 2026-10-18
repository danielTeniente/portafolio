"""Small CPU recurrent regressors with training-only epoch selection."""
import numpy as np
import torch
from torch import nn

torch.set_num_threads(2)


class ForecastNetwork(nn.Module):
    def __init__(self, kind, hidden, outputs):
        super().__init__()
        self.recurrent = getattr(nn, kind)(1, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        sequence, _ = self.recurrent(x)
        return self.head(sequence[:, -1, :])


class RecurrentRegressor:
    """Select epochs on a training tail, then restart and refit all training data."""
    def __init__(self, kind, hidden, seed=42, max_epochs=100, patience=12):
        self.kind, self.hidden, self.seed = kind, hidden, seed
        self.max_epochs, self.patience = max_epochs, patience

    def _initialize(self, x, y):
        torch.manual_seed(self.seed)
        self.x_mean_, self.x_scale_ = float(x.mean()), max(float(x.std()), 1.)
        self.y_mean_, self.y_scale_ = float(y.mean()), max(float(y.std()), 1.)
        self.model_ = ForecastNetwork(self.kind, self.hidden, y.shape[1])
        self.parameter_count_ = sum(p.numel() for p in self.model_.parameters())
        return torch.optim.Adam(self.model_.parameters(), lr=.01, weight_decay=.001)

    def _x(self, x):
        return torch.tensor((x-self.x_mean_)/self.x_scale_, dtype=torch.float32).unsqueeze(-1)

    def _y(self, y):
        return torch.tensor((y-self.y_mean_)/self.y_scale_, dtype=torch.float32)

    def _step(self, x, y, optimizer):
        self.model_.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(self.model_(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model_.parameters(), 1.)
        optimizer.step()

    def fit(self, x, y, epochs=None):
        if epochs is None:
            split = int(len(x)*.8)
            self.inner_train_days_ = split
            optimizer = self._initialize(x[:split], y[:split])
            tx, ty = self._x(x[:split]), self._y(y[:split])
            best, stale, self.epochs_ = float('inf'), 0, 1
            for epoch in range(1, self.max_epochs+1):
                self._step(tx, ty, optimizer)
                # Same nonnegative postprocessing and MAE objective as selection.
                mae = float(np.abs(np.maximum(0, self.predict(x[split:]))-y[split:]).mean())
                if mae < best - .01:
                    best, stale, self.epochs_ = mae, 0, epoch
                else:
                    stale += 1
                if stale >= self.patience:
                    break
            self.search_epochs_ = epoch
        else:
            self.epochs_, self.search_epochs_ = int(epochs), 0
        optimizer = self._initialize(x, y)
        tx, ty = self._x(x), self._y(y)
        for _ in range(self.epochs_):
            self._step(tx, ty, optimizer)
        return self

    def predict(self, x):
        self.model_.eval()
        with torch.inference_mode():
            return self.model_(self._x(x)).numpy()*self.y_scale_ + self.y_mean_

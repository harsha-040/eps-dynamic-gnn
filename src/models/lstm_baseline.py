"""Baseline: LSTM over the raw flattened multivariate time series (no graph
structure)."""
import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    def __init__(self, n_nodes: int, n_features: int, n_classes: int,
                 hidden_size: int = 96, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        in_dim = n_nodes * n_features
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor):
        B, T, N, Fd = x.shape
        seq = x.reshape(B, T, N * Fd)
        _, (h_n, _) = self.lstm(seq)
        last = h_n[-1]
        return self.classifier(self.dropout(last))

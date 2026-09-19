"""Baseline: 1D CNN over the raw multivariate time series (no graph structure).
Node/feature dims are flattened into channels; temporal convolutions extract
local fault signatures, followed by global pooling and a classifier head.
"""
import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    def __init__(self, n_nodes: int, n_features: int, n_classes: int,
                 channels=(64, 64, 128), kernel_size: int = 7, dropout: float = 0.2):
        super().__init__()
        in_ch = n_nodes * n_features
        layers = []
        prev = in_ch
        for c in channels:
            layers += [
                nn.Conv1d(prev, c, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(c),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ]
            prev = c
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(prev, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor):
        # x: (B, T, N, F) -> (B, N*F, T)
        B, T, N, Fd = x.shape
        x = x.reshape(B, T, N * Fd).permute(0, 2, 1)
        h = self.conv(x)
        h = self.pool(h).squeeze(-1)
        return self.classifier(h)

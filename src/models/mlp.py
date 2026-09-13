from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(input_dim: int, num_classes: int, cfg: dict) -> MLP:
    return MLP(input_dim, num_classes, list(cfg["hidden_dims"]), float(cfg["dropout"]))

"""Learned halting controller and halting-distribution utilities."""
from __future__ import annotations
import torch
from torch import nn


def halting_distribution(lambdas: torch.Tensor) -> torch.Tensor:
    """Convert (batch, layers) hazards to a normalized first-hit distribution."""
    if lambdas.ndim != 2: raise ValueError("lambdas must have shape (batch, layers)")
    if lambdas.shape[1] == 1: return torch.ones_like(lambdas)
    survival = torch.cumprod(torch.cat((torch.ones_like(lambdas[:, :1]), 1-lambdas[:, :-1]), dim=1), dim=1)
    p = survival * lambdas
    return torch.cat((p[:, :-1], (1-p[:, :-1].sum(dim=1, keepdim=True)).clamp_min(0)), dim=1)


class HaltingGate(nn.Module):
    """Shared MLP with learned layer embedding, optionally independent per layer."""
    def __init__(self, repr_size: int, layers: int, hidden: int = 128, use_layer_embedding: bool = True,
                 per_layer: bool = False, use_variance: bool = True):
        super().__init__(); self.layers, self.per_layer = layers, per_layer
        self.use_variance = use_variance
        self.emb = nn.Embedding(layers, 16) if use_layer_embedding and not per_layer else None
        extra = 16 if self.emb else 0
        input_size = repr_size + (2 if use_variance else 0) + 2 + extra
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(input_size, hidden), nn.GELU(), nn.Linear(hidden, 1))
                                  for _ in range(layers if per_layer else 1)])
    def forward(self, representation: torch.Tensor, logvar: torch.Tensor, delta: torch.Tensor, layer: int) -> torch.Tensor:
        parts = [representation]
        if self.use_variance: parts.append(logvar)
        parts.append(delta)
        if self.emb is not None: parts.append(self.emb.weight[layer].expand(representation.shape[0], -1))
        return torch.sigmoid(self.mlps[layer if self.per_layer else 0](torch.cat(parts, dim=-1)))

"""Aspect-conditioned attention readout over a token sequence."""
from __future__ import annotations
import torch
from torch import nn


class AspectReadout(nn.Module):
    """Single-head scaled dot-product attention with aspect-span query pooling."""
    def __init__(self, hidden_size: int):
        super().__init__(); self.query = nn.Linear(hidden_size, hidden_size); self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size); self.scale = hidden_size ** -0.5
    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor, sentence_index: torch.Tensor,
                aspect_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return concatenated aspect/context vectors and pooled aspect query."""
        tokens = hidden.index_select(0, sentence_index)
        mask = attention_mask.index_select(0, sentence_index).bool()
        weights = (aspect_mask.bool() & mask)
        counts = weights.sum(dim=1, keepdim=True)
        cls = torch.zeros_like(weights); cls[:, 0] = True
        weights = torch.where(counts > 0, weights, cls)
        q = (tokens * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
        scores = torch.einsum("bd,btd->bt", self.query(q), self.key(tokens)) * self.scale
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        context = torch.einsum("bt,btd->bd", attn, self.value(tokens))
        return torch.cat((q, context), dim=-1), q


def span_mask(offset_mapping: torch.Tensor, char_starts: torch.Tensor, char_ends: torch.Tensor) -> torch.Tensor:
    """Map character spans to tokenizer offsets; missing spans produce all false."""
    start, end = char_starts[:, None], char_ends[:, None]
    return (offset_mapping[..., 1] > start) & (offset_mapping[..., 0] < end) & (end > start)

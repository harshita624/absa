"""Analytic cost estimates; see README for assumptions and formula."""
from __future__ import annotations


def analytic_flops(sequence_length: int, hidden_size: int, encoder_layers: int, encoder_depth: float,
                   aspects: float, evaluated_layers: float, vocab_size: int = 30522) -> float:
    """Estimate multiply-add FLOPs for encoder layers and aspect heads/readouts."""
    d, s = hidden_size, sequence_length
    # Transformer layer: QKV/output projections, attention products, and 4x FFN.
    per_encoder = 4*s*d*d + 2*s*s*d + 8*s*d*d
    embedding = s*d*0 + vocab_size*d*0  # embeddings are lookup operations, excluded.
    # Per aspect/layer: four dxd projections (readout q/k/v + Gaussian hidden),
    # attention dot products and small output/gate MLP.
    per_readout = 4*d*d + 2*s*d + 12*d*d
    return float(encoder_depth*per_encoder + aspects*evaluated_layers*per_readout + embedding)

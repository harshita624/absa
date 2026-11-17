"""Shared sentence encoder, aspect readouts, Gaussian heads and halting gate."""
from __future__ import annotations
import torch
from torch import nn
from transformers import AutoModel
from .readout import AspectReadout, span_mask
from .gate import HaltingGate


class AdaptiveDepthModel(nn.Module):
    """A shared transformer pass followed by aspect-specific layer predictions."""
    def __init__(self, encoder_name: str = "bert-base-uncased", layers: int | None = None,
                 gate_per_layer: bool = False, gate_detach_inputs: bool = True, gate_use_variance: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name, output_hidden_states=True)
        self.hidden_size = self.encoder.config.hidden_size
        self.layers = layers or self.encoder.config.num_hidden_layers
        if self.layers != self.encoder.config.num_hidden_layers:
            raise ValueError("configured layer count must match the loaded encoder")
        self.readout = nn.ModuleList([AspectReadout(self.hidden_size) for _ in range(self.layers)])
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.GELU(),
                                                  nn.Dropout(.1), nn.Linear(self.hidden_size, 4))
                                    for _ in range(self.layers)])
        self.gate = HaltingGate(self.hidden_size*2, self.layers, per_layer=gate_per_layer, use_variance=gate_use_variance)
        self.gate_detach_inputs = gate_detach_inputs

    def predict_from_hidden(self, hidden_states: tuple[torch.Tensor, ...], batch: dict[str, torch.Tensor],
                            stop_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate aspect outputs and gate hazards at each provided hidden layer."""
        indices = batch["sentence_index"]
        offsets = batch["offset_mapping"].index_select(0, indices)
        amask = span_mask(offsets, batch["char_starts"], batch["char_ends"])
        mus, logvars, gates = [], [], []
        prev = None
        for i, hidden in enumerate(hidden_states[:self.layers]):
            r, _ = self.readout[i](hidden, batch["attention_mask"], indices, amask)
            raw = self.heads[i](r); mu, logvar = raw[:, :2], raw[:, 2:].clamp(-6, 4)
            delta = torch.zeros_like(mu) if prev is None else (mu-prev).abs()
            gin_mu = mu.detach() if self.gate_detach_inputs else mu
            gin_s = logvar.detach() if self.gate_detach_inputs else logvar
            gates.append(self.gate(r, gin_s, delta.detach() if self.gate_detach_inputs else delta, i).squeeze(-1))
            mus.append(mu); logvars.append(logvar); prev = mu
        return torch.stack(mus), torch.stack(logvars), torch.stack(gates), amask

    def forward(self, batch: dict[str, torch.Tensor]):
        """Run the encoder once and return per-layer means, log-variances and hazards."""
        outputs = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                               token_type_ids=batch.get("token_type_ids"), output_hidden_states=True, return_dict=True)
        return self.predict_from_hidden(outputs.hidden_states[1:], batch)

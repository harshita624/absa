"""Gaussian objective and adaptive-depth training loss."""
from __future__ import annotations
import torch


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-example Gaussian NLL summed over output dimensions."""
    logvar = logvar.clamp(-6, 4)
    return ((target-mu).square() * torch.exp(-logvar) / 2 + logvar / 2).sum(dim=-1)


def adaptive_loss(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, lambdas: torch.Tensor,
                  alpha: float = .3, beta: float = .003, stage: str = "joint") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine halting-weighted NLL, deep supervision and expected-depth cost."""
    from .gate import halting_distribution
    layer_losses = torch.stack([gaussian_nll(mu[i], logvar[i], target) for i in range(mu.shape[0])])
    probs = halting_distribution(lambdas.transpose(0, 1)).transpose(0, 1)
    expected = (probs * layer_losses).sum(0).mean()
    deep = layer_losses.mean(0).mean()
    depth = (probs * torch.arange(1, mu.shape[0]+1, device=mu.device, dtype=mu.dtype)[:,None]).sum(0).mean()
    total = deep if stage == "stage1" else expected + alpha * deep + beta * depth
    return total, {"expected_nll": expected.detach(), "deep_nll": deep.detach(), "expected_depth": depth.detach(),
                   "halting": probs.mean(dim=1).detach()}

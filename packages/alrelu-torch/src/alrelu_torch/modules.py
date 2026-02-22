"""ALReLU module API for PyTorch."""

from __future__ import annotations

import torch
from torch import nn

from .functional import alrelu


class ALReLU(nn.Module):
    """ALReLU with fixed alpha (default behavior)."""

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return alrelu(x, alpha=self.alpha)


class TrainableALReLU(nn.Module):
    """ALReLU with trainable alpha."""

    def __init__(self, alpha_init: float = 0.01, non_negative: bool = True):
        super().__init__()
        self.non_negative = bool(non_negative)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def effective_alpha(self) -> torch.Tensor:
        if self.non_negative:
            return torch.clamp(self.alpha, min=0.0)
        return self.alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return alrelu(x, alpha=self.effective_alpha())

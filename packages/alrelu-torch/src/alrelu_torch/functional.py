"""ALReLU functional API for PyTorch."""

from __future__ import annotations

from typing import Union

import torch


AlphaType = Union[float, torch.Tensor]


def _as_alpha_tensor(alpha: AlphaType, x: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(alpha):
        return alpha.to(dtype=x.dtype, device=x.device)
    return torch.as_tensor(alpha, dtype=x.dtype, device=x.device)


def alrelu(x: torch.Tensor, alpha: AlphaType = 0.01) -> torch.Tensor:
    """ALReLU activation.

    output = max(abs(alpha * x), x)
    """

    alpha_tensor = _as_alpha_tensor(alpha, x)
    return torch.maximum(torch.abs(alpha_tensor * x), x)

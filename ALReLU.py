"""Backward-compatible shim.

Prefer importing from `alrelu_keras`.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from alrelu_keras import ALReLU, TrainableALReLU, alrelu, register_alrelu
except ImportError:  # pragma: no cover
    repo_src = Path(__file__).resolve().parent / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    from alrelu_keras import ALReLU, TrainableALReLU, alrelu, register_alrelu

__all__ = ["ALReLU", "TrainableALReLU", "alrelu", "register_alrelu"]

"""ALReLU activation functions."""

from __future__ import annotations

from typing import Any

from ._compat import get_custom_objects, ops, register_keras_serializable


def _cast_alpha(alpha: Any, x: Any) -> Any:
    dtype = getattr(x, "dtype", None)
    if dtype is None:
        return alpha
    return ops.cast(alpha, dtype)


@register_keras_serializable(package="alrelu_keras")
def alrelu(x: Any, alpha: float = 0.01) -> Any:
    """ALReLU activation.

    This preserves the current project behavior:
    output = max(abs(alpha * x), x)
    """

    alpha_tensor = _cast_alpha(alpha, x)
    return ops.maximum(ops.abs(alpha_tensor * x), x)


def register_alrelu() -> None:
    """Register ALReLU callable in Keras custom objects."""

    get_custom_objects().update({"alrelu": alrelu, "ALReLU": alrelu})


register_alrelu()

"""Keras layers for ALReLU variants."""

from __future__ import annotations

from typing import Any

from ._compat import Layer, constraints, initializers, register_keras_serializable
from .activations import alrelu


@register_keras_serializable(package="alrelu_keras")
class ALReLU(Layer):
    """ALReLU with fixed alpha (default behavior)."""

    def __init__(self, alpha: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = float(alpha)

    def call(self, inputs):
        return alrelu(inputs, alpha=self.alpha)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config


@register_keras_serializable(package="alrelu_keras")
class TrainableALReLU(Layer):
    """ALReLU with trainable alpha."""

    def __init__(
        self,
        alpha_initializer: Any = 0.01,
        alpha_constraint: Any = "non_neg",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(alpha_initializer, (int, float)):
            alpha_initializer = initializers.Constant(float(alpha_initializer))

        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        self.alpha = None

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=self.alpha_initializer,
            constraint=self.alpha_constraint,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return alrelu(inputs, alpha=self.alpha)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha_initializer": initializers.serialize(self.alpha_initializer),
                "alpha_constraint": constraints.serialize(self.alpha_constraint),
            }
        )
        return config

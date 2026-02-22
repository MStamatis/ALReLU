"""Compatibility imports for Keras 3 and tf.keras."""

from __future__ import annotations

try:
    import keras
    from keras import constraints, initializers, ops
    from keras.layers import Layer
    from keras.saving import get_custom_objects, register_keras_serializable
except ImportError:  # pragma: no cover
    from tensorflow import keras
    from tensorflow.keras import backend as K
    from tensorflow.keras import constraints, initializers
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.utils import get_custom_objects, register_keras_serializable

    class _Ops:
        """Subset of ops API needed by this package."""

        @staticmethod
        def abs(x):
            return K.abs(x)

        @staticmethod
        def cast(x, dtype):
            return K.cast(x, dtype)

        @staticmethod
        def maximum(x, y):
            return K.maximum(x, y)

        @staticmethod
        def convert_to_numpy(x):
            return K.get_value(x)

    ops = _Ops()

__all__ = [
    "constraints",
    "get_custom_objects",
    "initializers",
    "keras",
    "Layer",
    "ops",
    "register_keras_serializable",
]

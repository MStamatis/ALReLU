import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

keras = pytest.importorskip("keras", exc_type=ImportError)

from alrelu_keras import ALReLU, TrainableALReLU, alrelu


def _to_numpy(x):
    if hasattr(keras.ops, "convert_to_numpy"):
        return keras.ops.convert_to_numpy(x)
    return np.array(x)


def _scalar_value(x):
    return float(np.array(_to_numpy(x)))


def test_default_alrelu_matches_existing_formula():
    x = keras.ops.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype="float32")
    y = alrelu(x)
    expected = np.array([0.02, 0.01, 0.0, 1.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(_to_numpy(y), expected, rtol=1e-6, atol=1e-6)


def test_fixed_layer_uses_configured_alpha():
    layer = ALReLU(alpha=0.2)
    x = keras.ops.array([-2.0, 2.0], dtype="float32")
    y = layer(x)
    np.testing.assert_allclose(
        _to_numpy(y),
        np.array([0.4, 2.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_trainable_alpha_is_updated_by_optimization():
    layer = TrainableALReLU(alpha_initializer=0.01)
    model = keras.Sequential([keras.layers.Input(shape=(1,)), layer])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1), loss="mse")

    x = np.array([[-1.0], [-2.0], [-3.0]], dtype=np.float32)
    y = np.array([[0.5], [1.0], [1.5]], dtype=np.float32)

    model(x[:1])  # Build model and layer weights.
    alpha_before = _scalar_value(layer.alpha)
    model.fit(x, y, epochs=5, verbose=0)
    alpha_after = _scalar_value(layer.alpha)

    assert alpha_after != pytest.approx(alpha_before)


def test_fixed_layer_config_round_trip():
    layer = ALReLU(alpha=0.02, name="my_alrelu")
    restored = ALReLU.from_config(layer.get_config())
    assert restored.alpha == pytest.approx(0.02)
